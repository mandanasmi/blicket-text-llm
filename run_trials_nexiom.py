import datetime
import hashlib
import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

import hydra
import numpy as np
from omegaconf import DictConfig
from openai.types.completion_usage import CompletionUsage

import env.blicket_text as blicket_text
from agent.prompt_variants import resolve_comprehension_rule

eLog = logging.getLogger(__name__)


def strip_np(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, list):
        return [strip_np(y) for y in x]
    if isinstance(x, bool):
        return str(x).lower()
    return x


def deterministic_seed(base_seed: int, episode_id: int) -> int:
    seed_str = f"{base_seed}-{episode_id}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)


def _write_action_log(CWD, trial_idx, log_suffix, log_entry):
    with open(os.path.join(CWD, f"action_log_trial-{trial_idx}{log_suffix}.jsonl"), "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def _write_thinking_log(CWD, trial_idx, log_suffix, log_entry):
    thinking_dir = os.path.join(CWD, "thinking_logs")
    os.makedirs(thinking_dir, exist_ok=True)
    with open(os.path.join(thinking_dir, f"thinking_trial-{trial_idx}{log_suffix}.jsonl"), "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def _serialize_info_dict(info_dict):
    serialized = {}
    for k, v in info_dict.items():
        if "usage" in k and isinstance(v, CompletionUsage):
            serialized[k] = {k1: vv for k1, vv in dict(v).items() if isinstance(vv, (int, float))}
        else:
            serialized[k] = strip_np(v)
    return serialized


def _run_phase(agent, env, phase_name, trial_idx, CWD, CFG, log_suffix, test_limit):
    if hasattr(agent, "set_phase"):
        agent.set_phase(phase_name)

    game_state = env.reset()
    obs = game_state["feedback"]
    agent.init_episode()

    done = False
    steps = 0
    phase_start_time = datetime.datetime.now()
    max_wall_clock_seconds = CFG.get("max_wall_clock_seconds", None)
    test_actions_used = 0
    phase_actions = []
    num_api_errors = 0
    has_object_interaction = False
    has_test = False
    entered_qa = False

    while not done and test_actions_used < test_limit:
        if max_wall_clock_seconds is not None and max_wall_clock_seconds > 0:
            elapsed_seconds = (datetime.datetime.now() - phase_start_time).total_seconds()
            if elapsed_seconds >= max_wall_clock_seconds:
                eLog.warning(
                    f"Trial {trial_idx} phase {phase_name} hit wall-clock timeout "
                    f"({elapsed_seconds:.2f}s >= {max_wall_clock_seconds}s); stopping phase early."
                )
                break

        game_state["phase_tests_remaining"] = test_limit - test_actions_used
        action, act_info = agent.act(obs, game_state)
        if action is None:
            action = "look"
        normalized_action = action.lower().strip().rstrip(".")
        phase_actions.append(action)

        if env.put_regex.fullmatch(normalized_action) or env.take_regex.fullmatch(normalized_action):
            has_object_interaction = True

        if env.test_regex.fullmatch(normalized_action):
            has_test = True

        if CFG.save_trial_log:
            log_entry = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "agent_class": str(agent),
                "trial_idx": trial_idx,
                "phase": phase_name,
                "steps": steps,
                "game_state": {},
                "action": action,
            }
            for gs in game_state:
                log_entry["game_state"][gs] = strip_np(game_state[gs])
            for k, v in act_info.items():
                if "usage" in k and isinstance(v, CompletionUsage):
                    log_entry[k] = {k1: vv for k1, vv in dict(v).items() if isinstance(vv, (int, float))}
                else:
                    log_entry[k] = strip_np(v)
            _write_action_log(CWD, trial_idx, log_suffix, log_entry)

            # Capture verbose model thinking separately, only when a valid command was produced.
            raw_msg = act_info.get("raw_response_message", "")
            parse_error = bool(act_info.get("parse_error", False))
            if (
                isinstance(raw_msg, str)
                and raw_msg.strip()
                and not parse_error
                and re.search(r"(?im)^\s*(thinking process|analysis|reasoning)\s*:?", raw_msg)
            ):
                _write_thinking_log(CWD, trial_idx, log_suffix, {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "trial_idx": trial_idx,
                    "phase": phase_name,
                    "steps": steps,
                    "action": action,
                    "prompt": act_info.get("prompt", ""),
                    "raw_response_message": raw_msg,
                })

        if hasattr(agent, "is_ready_action") and agent.is_ready_action(normalized_action):
            if phase_name == "comprehension" and not (has_object_interaction and has_test):
                obs = (
                    "You are not ready for Q&A yet. You must interact with objects and run at least one test "
                    "before entering Q&A."
                )
                steps += 1
                continue
            entered_qa = True
            done = True
            # Hard stop exploration once ready-for-Q&A is accepted.
            obs = "You entered the question answering phase. Exploration is now closed."
            break

        game_state, _, done = env.step(action)
        obs = game_state["feedback"]
        if env.test_regex.fullmatch(normalized_action):
            test_actions_used += 1
        steps += 1
        num_api_errors += int(act_info.get("api_error", False))

    agent.add_obs_to_queue(obs, game_state)
    return {
        "steps": steps,
        "num_test_actions": test_actions_used,
        "actions": phase_actions,
        "entered_qa": entered_qa,
        "num_api_errors": num_api_errors,
        "final_game_state": game_state,
    }


def run_trial(CFG: DictConfig, CWD: str, trial_idx: int, base_seed: int, run_timestamp: str = ""):
    start_time = datetime.datetime.now()
    log_suffix = f"_{run_timestamp}" if run_timestamp else ""

    seed = deterministic_seed(base_seed, trial_idx)
    random.seed(seed)
    np.random.seed(seed)

    agent = hydra.utils.instantiate(CFG.agent)

    variant = getattr(agent, "prompt_variant", None)
    comp_rule = resolve_comprehension_rule(
        variant, trial_idx, seed, fallback=CFG.comprehension_env_kwargs.rule
    ) if variant is not None else CFG.comprehension_env_kwargs.rule

    comp_env = blicket_text.BlicketTextEnv(
        num_objects=3,
        num_blickets=CFG.comprehension_env_kwargs.num_blickets,
        init_prob=CFG.comprehension_env_kwargs.init_prob,
        transition_noise=CFG.comprehension_env_kwargs.transition_noise,
        rule=comp_rule,
        seed=seed,
        object_names=["A", "B", "C"],
    )

    main_env_kwargs = dict(CFG.env_kwargs)
    if str(main_env_kwargs.get("rule", "")).lower() == "alternate":
        main_env_kwargs["rule"] = "conjunctive" if trial_idx % 2 == 0 else "disjunctive"
    main_env = blicket_text.BlicketTextEnv(
        **main_env_kwargs,
        seed=seed + 1,
        object_names=["object 1", "object 2", "object 3", "object 4"],
    )

    eLog.info(
        f"Trial {trial_idx} | comprehension blickets={comp_env.blicket_indices} | "
        f"main blickets={main_env.blicket_indices} | rule={main_env.rule}"
    )

    comp_phase = _run_phase(
        agent=agent,
        env=comp_env,
        phase_name="comprehension",
        trial_idx=trial_idx,
        CWD=CWD,
        CFG=CFG,
        log_suffix=log_suffix,
        test_limit=CFG.comprehension_test_limit,
    )

    if hasattr(agent, "set_phase"):
        agent.set_phase("comprehension")

    comp_question = (
        "Comprehension check. Select your answer:\n"
        "- Object A is a Nexiom\n"
        "- Object B is a Nexiom\n"
        "- Object C is a Nexiom\n"
        "- I don't know\n\n"
        "Reply with exactly one option."
    )
    comp_answer, comp_ans_info = agent.answer_text(comp_question)
    comp_labels = ["A", "B", "C"]
    comp_true_answers = [comp_labels[i] for i in comp_env.blicket_indices]
    if len(comp_true_answers) == 1:
        comp_feedback = f"Correct answer: Object {comp_true_answers[0]} is a Nexiom."
    else:
        comp_feedback = f"Correct answer: Objects {', '.join(comp_true_answers)} are Nexioms."

    if CFG.save_trial_log:
        _write_action_log(CWD, trial_idx, log_suffix, {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent_class": str(agent),
            "trial_idx": trial_idx,
            "phase": "comprehension_qa",
            "question": comp_question,
            "answer": comp_answer,
            "true_answer": comp_true_answers,
            "feedback_shown": comp_feedback,
            **_serialize_info_dict(comp_ans_info),
        })

    prior_study_q = (
        "Have you done a study or read about a study like this before, where you have to figure out "
        "which objects make a machine, light, or toy turn on?"
    )
    prior_study_ans, prior_study_info = agent.answer_text(prior_study_q)
    if CFG.save_trial_log:
        _write_action_log(CWD, trial_idx, log_suffix, {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent_class": str(agent),
            "trial_idx": trial_idx,
            "phase": "comprehension_qa",
            "question": prior_study_q,
            "answer": prior_study_ans,
            **_serialize_info_dict(prior_study_info),
        })

    if hasattr(agent, "set_comprehension_result"):
        agent.set_comprehension_result(comp_true_answers)

    main_phase = _run_phase(
        agent=agent,
        env=main_env,
        phase_name="main",
        trial_idx=trial_idx,
        CWD=CWD,
        CFG=CFG,
        log_suffix=log_suffix,
        test_limit=agent.horizon,
    )

    if hasattr(agent, "set_phase"):
        agent.set_phase("main")

    n_questions = 0
    n_correct = 0
    n_ans_api_errors = 0
    nexiom_answers = {}
    all_object_names = main_env.object_names
    nexiom_names = [all_object_names[i] for i in main_env.blicket_indices]

    for idx, obj_name in enumerate(all_object_names, start=1):
        question_str = f"Is object {idx} a Nexiom?"
        bool_answer, ans_info = agent.answer_tf(question_str, env=main_env)
        nexiom_answers[obj_name] = bool_answer
        true_answer = obj_name in nexiom_names
        n_correct += int(bool_answer == true_answer)
        n_questions += 1
        n_ans_api_errors += int(ans_info.get("api_error", False))

        if CFG.save_trial_log:
            _write_action_log(CWD, trial_idx, log_suffix, {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "agent_class": str(agent),
                "trial_idx": trial_idx,
                "phase": "main_qa",
                "question": question_str,
                "answer": str(bool_answer).lower(),
                "true_answer": str(true_answer).lower(),
                **_serialize_info_dict(ans_info),
            })

    rule_inference_response, rule_inference_info = agent.answer_rule_inference(env=main_env)
    rule_type_response, rule_type_info = agent.answer_rule_type(
        nexiom_answers, rule_inference_response, env=main_env
    )
    n_ans_api_errors += int(rule_inference_info.get("api_error", False))
    n_ans_api_errors += int(rule_type_info.get("api_error", False))

    if CFG.save_trial_log:
        _write_action_log(CWD, trial_idx, log_suffix, {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent_class": str(agent),
            "trial_idx": trial_idx,
            "phase": "main_qa",
            "question": "rule_inference",
            "answer": str(rule_inference_response),
            **_serialize_info_dict(rule_inference_info),
        })
        _write_action_log(CWD, trial_idx, log_suffix, {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent_class": str(agent),
            "trial_idx": trial_idx,
            "phase": "main_qa",
            "question": "rule_type",
            "answer": str(rule_type_response),
            "true_answer": main_env.rule,
            **_serialize_info_dict(rule_type_info),
        })

    all_nexioms_correct = (n_correct == n_questions)
    rule_type_correct = (str(rule_type_response).strip().lower() == str(main_env.rule).strip().lower())
    eval_reward = 1.0 if (all_nexioms_correct and rule_type_correct) else (
        0.5 if (all_nexioms_correct or rule_type_correct) else 0.0
    )

    actions_fname = f"actions_trial-{trial_idx}{log_suffix}.json" if log_suffix else f"actions_trial-{trial_idx}.json"
    with open(os.path.join(CWD, actions_fname), "w") as f:
        json.dump(
            {
                "trial_idx": trial_idx,
                "comprehension_actions": comp_phase["actions"],
                "main_actions": main_phase["actions"],
            },
            f,
            indent=2,
        )

    trial_duration = (datetime.datetime.now() - start_time).total_seconds()
    trial_data = {
        "trial_idx": trial_idx,
        "comprehension_steps": comp_phase["steps"],
        "comprehension_num_test_actions": comp_phase["num_test_actions"],
        "comprehension_entered_qa": comp_phase["entered_qa"],
        "comprehension_answer": str(comp_answer),
        "comprehension_true_answer": comp_true_answers,
        "prior_study_answer": str(prior_study_ans),
        "main_num_steps": main_phase["steps"],
        "main_num_test_actions": main_phase["num_test_actions"],
        "main_entered_qa": main_phase["entered_qa"],
        "num_questions": n_questions,
        "num_correct": n_correct,
        "all_nexioms_correct": all_nexioms_correct,
        "rule_type_correct": rule_type_correct,
        "eval_reward": eval_reward,
        "rule_inference_response": str(rule_inference_response),
        "rule_type_response": str(rule_type_response),
        "true_rule": main_env.rule,
        "comprehension_rule": comp_env.rule,
        "trial_duration": trial_duration,
        "num_traj_api_errors": comp_phase["num_api_errors"] + main_phase["num_api_errors"],
        "num_ans_api_errors": n_ans_api_errors,
        "cost_estimate": agent.total_cost,
        "prompt_variant": (variant.to_dict() if variant is not None else None),
        "model": getattr(agent, "model", None),
        "temperature": getattr(agent, "temperature", None),
    }
    results_fname = f"results{log_suffix}.jsonl" if log_suffix else "results.jsonl"
    with open(os.path.join(CWD, results_fname), "a") as f:
        f.write(json.dumps(trial_data) + "\n")
    return trial_data


@hydra.main(version_base=None, config_path=".", config_name="run_trials_nexiom")
def main(CFG: DictConfig):
    CWD = os.getcwd()
    eLog.info(f"Current working directory: {CWD}")
    start_time = datetime.datetime.now()
    run_timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    if CFG.use_threadpool:
        executor = ThreadPoolExecutor(max_workers=CFG.tp_max_workers)

        def _run_single(trial_idx):
            return run_trial(CFG, CWD, trial_idx, CFG.seed, run_timestamp)

        results = list(executor.map(_run_single, range(CFG.num_trials)))
        executor.shutdown()
    else:
        results = [run_trial(CFG, CWD, trial_idx, CFG.seed, run_timestamp) for trial_idx in range(CFG.num_trials)]

    total_time = (datetime.datetime.now() - start_time).total_seconds()
    total_cost = sum(r.get("cost_estimate", 0.0) for r in results)
    eLog.info(f"Num trials: {CFG.num_trials}. Total time: {total_time}. Estimated total cost: {total_cost}.")


if __name__ == "__main__":
    main()
