import argparse
import os
import pickle
import logging
import json
import datetime


import hydra
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import pandas as pd

import env.blicket_text as blicket_text

eLog = logging.getLogger(__name__)  # hydra logger


def strip_np(x):
    """Turn np object into json-serializable object."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, list):
        return [strip_np(y) for y in x]
    return x


@hydra.main(version_base=None, config_path=".", config_name="run_explore_blicket")
def main(CFG: DictConfig):
    CWD = os.getcwd()
    eLog.info(f'Current working directory: {CWD}')

    # Set random seed for reproducibility
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)

    # 
    eLog.info(f'Running agent {CFG.agent._target_} for {CFG.num_trials} trials...')
    for trial_idx in range(CFG.num_trials):
        eLog.info(f'Starting trial {trial_idx}...')
        start_time = datetime.datetime.now()

        # ==
        # Initialize agent and environment in a per-trial manner 
        agent = hydra.utils.instantiate(CFG.agent)

        # environment
        env = blicket_text.BlicketTextEnv(**CFG.env_kwargs, seed=None)
        eLog.info(
            f'Environment created: {env}, with {env.num_objects} objects. '
            f'Object names: {env.object_names}. '
            f'Blicket indices: {env.blicket_indices}'
        )
        
        # ==
        # Run episode
        game_state = env.reset()
        obs = game_state['feedback']
        done = False

        agent.init_episode()

        steps = 0
        total_reward = 0.0
        num_api_errors = 0
        while not done and game_state['moves'] < agent.horizon:
            action, act_info = agent.act(obs, game_state)

            if CFG.save_lm_log: 
                log_entry = {
                    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "agent_class": str(agent),
                    "trial_idx": trial_idx,
                    "steps": steps,
                    "game_state": None,
                }

                log_entry["game_state"] = {}
                for gs in game_state:
                    log_entry["game_state"][gs] = strip_np(game_state[gs])

                for k in act_info:
                    if k == "usage":
                        log_entry["usage"] = {k: v for k, v in dict(act_info["usage"]).items() if isinstance(v, (int, float))}
                    else:
                        log_entry[k] = strip_np(act_info[k])

                """
                if "prompt" in act_info:
                    log_entry.update({
                        "model": act_info["model"],
                        "system_message": act_info["system_message"],
                        "prompt": act_info["prompt"],
                        "response": act_info["response_message"],
                        "usage": {k: v for k, v in dict(act_info["usage"]).items() if isinstance(v, (int, float))},
                    })
                """
                if "items_dict" in act_info:
                    log_entry["items_dict"] = act_info["items_dict"]

                with open(os.path.join(CWD, f"action_log_trial-{trial_idx}.jsonl"), "a") as f:
                    f.write(json.dumps(log_entry) + "\n") 

            game_state, reward, done = env.step(action)
            obs = game_state['feedback']

            total_reward += reward
            steps += 1
            num_api_errors += act_info["api_error"]

        # add last obs
        agent.add_obs_to_queue(obs, game_state)  # add the last observation to the queue

        # ==
        # Evaluate agent
        n_questions = 0
        n_correct = 0
        n_ans_api_errors = 0
        blicket_answers = {}

        all_object_names = env.object_names
        blicket_names = [all_object_names[i] for i in env.blicket_indices]

        for obj_name in all_object_names:
            question_str = f"Is {obj_name} a blicket?"
            bool_answer, ans_info = agent.answer_tf(question_str, env=env)
            blicket_answers[obj_name] = bool_answer

            if CFG.save_lm_log:
                log_entry = {
                    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "agent_class": str(agent),
                    "trial_idx": trial_idx,
                    "question": question_str,
                    "answer": bool_answer,
                    "true_answer": obj_name in blicket_names,
                }

                for k in ans_info:
                    if k == "usage":
                        log_entry["usage"] = {k: v for k, v in dict(ans_info["usage"]).items() if isinstance(v, (int, float))}
                    else:
                        log_entry[k] = strip_np(ans_info[k])

                """
                if "prompt" in ans_info: 
                    log_entry.update({
                        "model": ans_info["model"],
                        "system_message": ans_info["system_message"],
                        "prompt": ans_info["prompt"],
                        "response": ans_info["response_message"],
                        "usage": {k: v for k, v in dict(act_info["usage"]).items() if isinstance(v, (int, float))},
                    })
                """
                with open(os.path.join(CWD, f"action_log_trial-{trial_idx}.jsonl"), "a") as f:
                    f.write(json.dumps(log_entry) + "\n") 

            if bool_answer == (obj_name in blicket_names):
                n_correct += 1
            n_questions += 1
            n_ans_api_errors += ans_info["api_error"]

        # Phase 2: Rule inference (independent prompt)
        rule_inference_response = ""
        rule_inference_info = {}
        try:
            rule_inference_response, rule_inference_info = agent.answer_rule_inference(env=env)
            if CFG.save_lm_log:
                log_entry = {
                    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "agent_class": str(agent),
                    "trial_idx": trial_idx,
                    "question": "rule_inference",
                    "answer": str(rule_inference_response),
                }
                for k in rule_inference_info:
                    if k == "usage":
                        log_entry["usage"] = {k1: v for k1, v in dict(rule_inference_info["usage"]).items() if isinstance(v, (int, float))}
                    else:
                        log_entry[k] = strip_np(rule_inference_info.get(k))
                with open(os.path.join(CWD, f"action_log_trial-{trial_idx}.jsonl"), "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            n_ans_api_errors += rule_inference_info.get("api_error", False)
        except NotImplementedError:
            pass

        # Phase 3: Rule type selection (independent prompt)
        rule_type_response = ""
        rule_type_info = {}
        try:
            rule_type_response, rule_type_info = agent.answer_rule_type(blicket_answers, rule_inference_response, env=env)
            if CFG.save_lm_log:
                log_entry = {
                    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "agent_class": str(agent),
                    "trial_idx": trial_idx,
                    "question": "rule_type",
                    "answer": str(rule_type_response),
                    "true_answer": env.rule,
                }
                for k in rule_type_info:
                    if k == "usage":
                        log_entry["usage"] = {k1: v for k1, v in dict(rule_type_info["usage"]).items() if isinstance(v, (int, float))}
                    else:
                        log_entry[k] = strip_np(rule_type_info.get(k))
                with open(os.path.join(CWD, f"action_log_trial-{trial_idx}.jsonl"), "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            n_ans_api_errors += rule_type_info.get("api_error", False)
        except NotImplementedError:
            pass

        eLog.info(f"Trial {trial_idx} completed in {steps} steps. Questions correct: {n_correct}/{n_questions}")

        # Save trial data
        trial_duration = (datetime.datetime.now() - start_time).total_seconds()
        total_duration = (datetime.datetime.now() - start_time).total_seconds()

        trial_data = {
            "trial_idx": trial_idx,
            "unique_state_visited": game_state["unique_state_visited"],
            "turn_machine_on": int(game_state["turn_machine_on"]),  # no bool in json
            "num_steps": steps,
            "total_reward": total_reward,
            "num_questions": n_questions,
            "num_correct": n_correct,
            "rule_inference_response": str(rule_inference_response),
            "rule_type_response": str(rule_type_response),
            "true_rule": env.rule,
            "trial_duration": trial_duration,
            "total_duration": total_duration,
            "num_traj_api_errors": num_api_errors,
            "num_ans_api_errors": n_ans_api_errors,
            "cost_esti_so_far": agent.total_cost,
        }
        with open(os.path.join(CWD, "results.jsonl"), "a") as f:
            f.write(json.dumps(trial_data) + "\n")

        eLog.info(f"Estimated cost so far: {agent.total_cost}")

    # Log estimated total cost        
    eLog.info(f"Estimated total cost: {agent.total_cost}")
    


if __name__ == '__main__':
    main()
