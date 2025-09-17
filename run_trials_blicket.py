import datetime
import hashlib
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor

import env.blicket_text as blicket_text
import hydra
import numpy as np
from openai.types.completion_usage import CompletionUsage
from omegaconf import DictConfig

eLog = logging.getLogger(__name__)  # hydra logger


def strip_np(x):
    """Turn np object into json-serializable object for logging"""
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
    """Generate a unique seed for each episode using a hash function to avoid seed collision"""
    seed_str = f"{base_seed}-{episode_id}"
    hashed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
    return hashed


def run_trial(CFG: DictConfig, CWD: str, trial_idx: int, base_seed: int):
    """
    Run a single agent in a single environment for a single trial (ie episode) 
    """
    # ==
    # Initialization
    start_time = datetime.datetime.now()

    # set seed
    seed = deterministic_seed(base_seed, trial_idx)
    random.seed(seed)
    np.random.seed(seed)
    
    agent = hydra.utils.instantiate(CFG.agent)
    env = blicket_text.BlicketTextEnv(**CFG.env_kwargs, seed=None)
    
    eLog.info(
        f'Trial: {trial_idx}. Environment: {env}, with {env.num_objects} objects. '
        f'Object names: {env.object_names}. '
        f'Blicket indices: {env.blicket_indices}'
    )
    
    # ==
    # Run exploration episode
    game_state = env.reset()
    obs = game_state['feedback']
    agent.init_episode()

    done = False
    steps = 0
    total_reward = 0.0
    num_api_errors = 0
    while not done and game_state['moves'] < agent.horizon:
        action, act_info = agent.act(obs, game_state)

        if action is None:
            action = ""  # hacky, safety check to ensure action is a string

        if CFG.save_trial_log: 
            log_entry = {
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "agent_class": str(agent),
                "trial_idx": trial_idx,
                "steps": steps,
                "game_state": None,
                "action": action,
            }

            log_entry["game_state"] = {}
            for gs in game_state:
                log_entry["game_state"][gs] = strip_np(game_state[gs])

            for k in act_info:
                if "usage" in k and isinstance(act_info[k], CompletionUsage):
                    log_entry[k] = {k1: v for k1, v in dict(act_info[k]).items() if isinstance(v, (int, float))}
                else:
                    log_entry[k] = strip_np(act_info[k])

            with open(os.path.join(CWD, f"action_log_trial-{trial_idx}.jsonl"), "a") as f:
                f.write(json.dumps(log_entry) + "\n") 

        game_state, reward, done = env.step(action)
        obs = game_state['feedback']

        total_reward += reward
        steps += 1
        if "api_error" in act_info:
            num_api_errors += act_info["api_error"]

    # add last obs
    agent.add_obs_to_queue(obs, game_state)  # add the last observation to the queue

    # ==
    # Evaluate agent
    n_questions = 0
    n_correct = 0
    n_ans_api_errors = 0

    all_object_names = env.object_names
    blicket_names = [all_object_names[i] for i in env.blicket_indices]

    for obj_name in all_object_names:
        question_str = f"Is {obj_name} a blicket?"
        bool_answer, ans_info = agent.answer_tf(question_str, env=env)

        if CFG.save_trial_log:
            log_entry = {
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "agent_class": str(agent),
                "trial_idx": trial_idx,
                "question": question_str,
                "answer": str(bool_answer).lower(),
                "true_answer": str(obj_name in blicket_names).lower(),
            }

            for k in ans_info:
                if "usage" in k and isinstance(ans_info[k], CompletionUsage):
                    log_entry[k] = {k: v for k, v in dict(ans_info[k]).items() if isinstance(v, (int, float))}
                else:
                    log_entry[k] = strip_np(ans_info[k])

            with open(os.path.join(CWD, f"action_log_trial-{trial_idx}.jsonl"), "a") as f:
                f.write(json.dumps(log_entry) + "\n") 

        if bool_answer == (obj_name in blicket_names):
            n_correct += 1
        n_questions += 1
        if "api_error" in ans_info:
            n_ans_api_errors += ans_info["api_error"]

    eLog.info(f"Trial {trial_idx} completed in {steps} steps. Questions correct: {n_correct}/{n_questions}")

    # Save trial data
    trial_duration = (datetime.datetime.now() - start_time).total_seconds()

    trial_data = {
        "trial_idx": trial_idx,
        "unique_state_visited": game_state["unique_state_visited"],
        "turn_machine_on": int(game_state["turn_machine_on"]),  # no bool in json
        "num_steps": steps,
        "total_reward": total_reward,
        "num_questions": n_questions,
        "num_correct": n_correct,
        "trial_duration": trial_duration,
        "num_traj_api_errors": num_api_errors,
        "num_ans_api_errors": n_ans_api_errors,
        "cost_estimate": agent.total_cost,
    }
    with open(os.path.join(CWD, "results.jsonl"), "a") as f:
        f.write(json.dumps(trial_data) + "\n")

    return trial_data


@hydra.main(version_base=None, config_path=".", config_name="run_trials_blicket")
def main(CFG: DictConfig):
    CWD = os.getcwd()
    eLog.info(f'Current working directory: {CWD}')
    
    start_time = datetime.datetime.now()
    eLog.info(f'Running agent {CFG.agent._target_} for {CFG.num_trials} trials...')

    if CFG.use_threadpool:
        # parallel runs
        executor = ThreadPoolExecutor(max_workers=CFG.tp_max_workers)

        def _run_trial(trial_idx):
            return run_trial(CFG, CWD, trial_idx, CFG.seed)
        results = list(executor.map(_run_trial, range(CFG.num_trials)))

        executor.shutdown()

    else: 
        results = [run_trial(CFG, CWD, trial_idx, CFG.seed) 
                   for trial_idx in range(CFG.num_trials)]

    # Log estimated total cost        
    total_time = (datetime.datetime.now() - start_time).total_seconds()

    total_cost = 0.0
    for r in results:
        if "cost_estimate" in r:
            total_cost += r["cost_estimate"]

    eLog.info(f"Num trials: {CFG.num_trials}. Total time: {total_time}. Estimated total cost: {total_cost}.")
    

if __name__ == '__main__':
    main()
