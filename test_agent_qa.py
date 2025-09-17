import datetime
import hashlib
import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import backoff
import hydra
import numpy as np
import openai
import yaml
from openai.types.completion_usage import CompletionUsage
from omegaconf import DictConfig, OmegaConf

import lm_api

eLog = logging.getLogger(__name__)  # hydra logger

OAI_CLIENT = openai.OpenAI()


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


def deterministic_seed(base_seed: int, traj_path: str) -> int:
    """Generate a unique seed for each episode using a hash function to avoid seed collision"""
    seed_str = f"{base_seed}-{traj_path}"
    hashed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
    return hashed


def get_prompt_method(method: str):
    if method == "default":
        return "Directly output the answer in the format \'> True/False\'. "\
            "Ensure only one answer is included."
    elif method == "react":
        return "First briefly reason and think about the information collected. "\
            "Then, output the answer in the format \'> True/False\'. "\
            "Ensure only one answer is included."
    elif method == "reflexion":
        prompt = (
            "Please first reflect on the collected information and analyze any potential issues with your reasoning, "
            "then output the final answer in the format '> True/False'. Ensure only one answer is included."
        )
        return prompt
    elif method == "cot":
        prompt = (
            "Please provide a detailed chain-of-thought explaining your reasoning, "
            "and then output the final answer in the format '> True/False'. Ensure only one answer is included."
        )
        return prompt
    else:
        raise ValueError(f"Unknown prompt method: {method}")


def query_llm_api(client, model, system_message, prompt, temperature=0.0):
    if "o3" in model or "o1" in model:
        chat_kwargs = {
            "max_completion_tokens": 1024,
        }
    else:
        chat_kwargs = {
            "temperature": temperature,
            "max_tokens": 1024,
            "top_p": 0.2,
            "stop": None,
        }

    response, cost = lm_api.query_llm(client, model, system_message, prompt, 
                                      chat_kwargs)

    def _extract_action(text):
        # This pattern ensures that the "> " pattern appears at the start of the string or after a newline,
        # and captures everything until a period (if present) or the end of the line.
        action_pattern = r"> (.*?)(?:\.|$)"

        # Search for the pattern in the text
        match = re.search(action_pattern, text)

        # If a match is found, return the matching group which contains the action
        if match:
            action = match.group(1).strip()  # Strip any whitespace around the action
            return action.rstrip('.')  # Remove any trailing period
        else:
            # Return None or an appropriate message if no action is found
            return None

    ans_str = _extract_action(response.choices[0].message.content)

    return response, ans_str, cost


def construct_history_obs_txt(data: List[Dict]):
    """Construct a history obs string based on trajectory data"""
    obs_txt = ""
    for i, dd in enumerate(data):
        obs_txt += dd['game_state']["feedback"] + "\n"
        if i < len(data) - 1:
            # NOTE: omitting the last action and the final observation
            obs_txt += f"> {dd['action']}" + "\n"

    return obs_txt


def run_qa_test(CFG: DictConfig, CWD: str, client: openai.OpenAI, traj_path: Path, base_seed: int):
    """
    Evaluate a single agent, using a single trajectory
    """
    # ==
    # Initialization
    start_time = datetime.datetime.now()

    # set seed
    seed = deterministic_seed(base_seed, str(traj_path))
    random.seed(seed)
    np.random.seed(seed)

    # Read trajectory into data
    data = []
    ques_data = []
    data = []
    with open(traj_path, 'r') as file:
        for line in file:
            cur_json = json.loads(line)
            # data = append_dict(data, cur_json)
            if "question" in cur_json:
                ques_data.append((cur_json))
            else:
                data.append((cur_json))

    obj_name_list = data[-1]['game_state']['object_names']  # List[str]
    blicket_indices = data[-1]['game_state']['blicket_indices']  # List[int]

    if 'history_obs' in data[-1]:
        history_obs_txt = data[-1]['history_obs']  # str
        constructed_history_obs = False
    else:
        history_obs_txt = construct_history_obs_txt(data)  # str
        constructed_history_obs = True

    # Read the associated yaml file from the traj_path file
    config_path = traj_path.parent / ".hydra" / "config.yaml"
    with open(config_path, 'r') as yaml_file:
        traj_config = yaml.safe_load(yaml_file)

    # Get system message
    if CFG.system_msg_path.startswith('/'):
        sys_msg_path = CFG.system_msg_path
    else:
        sys_msg_path = Path(hydra.utils.get_original_cwd()) / CFG.system_msg_path

    with open(sys_msg_path, 'r') as f:
        system_message = f.read()

    # ==
    # 
    n_questions = 0
    n_correct = 0
    n_ans_api_errors = 0
    total_cost_esti = 0.0

    total_response_len = 0
    total_response_tokens = 0

    log_list_dict = []

    for obj_idx, obj_name in enumerate(obj_name_list):

        user_prompt = history_obs_txt + "\n\n"
        user_prompt += f"Based on the information above, is {obj_name} a blicket?\n\n"
        user_prompt += get_prompt_method(CFG.prompt_method)

        response_msg = None
        api_error = False
        try:
            response, ans_str, cost = query_llm_api(client, CFG.model, system_message, user_prompt, temperature=CFG.temperature)
            response_msg = response.choices[0].message.content

            if ans_str == 'True':
                ans_bool = True
            elif ans_str == 'False':
                ans_bool = False
            else:
                ans_bool = np.random.choice([True, False])

        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as e:
            print(f'Error: {e}')
            ans_bool = np.random.choice([True, False])
            api_error = True
        
        # 
        true_answer = obj_idx in blicket_indices
        answer_correct = ans_bool == true_answer
        
        # Log
        log_entry = {
            "metric_type": "per_question",
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "system_message": system_message,
            "user_prompt": user_prompt,
            "question_obj_idx": obj_idx,
            "question_obj_name": obj_name,
            "response_msg": response_msg,
            "answer": str(ans_bool).lower(),
            "true_answer": str(true_answer).lower(),
            "answer_correct": str(answer_correct).lower(),
            "api_error": str(api_error).lower(),
            "estimated_cost": cost,
            "traj_path": str(traj_path),
        }

        if response is not None and hasattr(response.choices[0].message, "reasoning_content"):
            log_entry["reasoning_content"] = response.choices[0].message.reasoning_content

        if response is not None and hasattr(response, "model"):
            log_entry["response_model"] = response.model
        if response is not None and hasattr(response, "usage") and isinstance(response.usage, CompletionUsage):
            log_entry["usage"] = {k: v for k, v in dict(response.usage).items() if isinstance(v, (int, float))}

        log_entry = {k: strip_np(v) for k, v in log_entry.items()}  # make json-serializable
        log_list_dict.append(log_entry)

        # TODO: save this to file
        if CFG.save_trial_log:
            uniq_id = hashlib.md5(str(traj_path).encode()).hexdigest()[:8]
            trial_log_path = CWD / f"qa-log_s{seed}_{uniq_id}.jsonl"
            with trial_log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

        # 
        n_questions += 1
        n_correct += answer_correct
        n_ans_api_errors += api_error
        total_cost_esti += cost

        total_response_len += len(response_msg)
        if response is not None and hasattr(response, "usage") and isinstance(response.usage, CompletionUsage):
            total_response_tokens += response.usage.completion_tokens
    

    # 
    trial_duration = (datetime.datetime.now() - start_time).total_seconds()
    trial_metrics = {
        "metric_type": "trial",
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "num_questions": n_questions,
        "num_correct": n_correct,
        "avg_response_len": total_response_len / n_questions,
        "avg_response_tokens": total_response_tokens / n_questions,
        "num_ans_api_errors": n_ans_api_errors,
        "trial_duration": trial_duration,
        "trial_cost_esti": total_cost_esti,
        "obj_name_list": obj_name_list,
        "blicket_indices": blicket_indices,
        "history_obs": history_obs_txt,
        "constructed_history_obs": str(constructed_history_obs).lower(),
        "obs_traj_len": len(data),
        "expl_cfg": traj_config,
        "qa_cfg": OmegaConf.to_container(CFG, resolve=True),
        "traj_path": str(traj_path),
    }
    trial_metrics = {k: strip_np(v) for k, v in trial_metrics.items()}  # make json-serializable

    results_path = CWD / "results.jsonl"
    with results_path.open("a") as f:
        f.write(json.dumps(trial_metrics) + "\n")

    return trial_metrics
    

@hydra.main(version_base=None, config_path=".", config_name="test_agent_qa")
def main(CFG: DictConfig):
    CWD = Path.cwd()
    eLog.info(f'Current working directory: {CWD}')
    
    start_time = datetime.datetime.now()

    in_traj_paths = []
    for cur_wc in CFG.read_path_wc:
        in_traj_paths += list(Path(CFG.read_path_parent).glob(cur_wc))
    eLog.info(f'Finding files matching patterns {CFG.read_path_wc}')
    eLog.info(f'Num total trajectory files found: {len(in_traj_paths)}')

    # filter out files depending on cfg
    def _filter_passed(traj_path):
        """Helper functions to filter subset of the trajectories by their config"""
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        config_path = traj_path.parent / ".hydra" / "config.yaml"
        with open(config_path, 'r') as yaml_file:
            traj_cfg = yaml.safe_load(yaml_file)  # dict
        traj_cfg = flatten_dict(traj_cfg)
        cur_filt = flatten_dict(OmegaConf.to_container(CFG.expl_filter))

        for k in cur_filt:
            if k in traj_cfg and traj_cfg[k] != cur_filt[k]:
                return False
        
        return True
    
    filtered_traj_paths = [tp for tp in in_traj_paths if _filter_passed(tp)]
    eLog.info(f'Applying filters: {CFG.expl_filter}.')
    eLog.info(f'Num trajectory files after filtering: {len(filtered_traj_paths)}')

    post_filt_dir_paths = sorted(list(set([str(tp.parent) for tp in filtered_traj_paths])))
    _dir_paths_str = "\n\t".join(post_filt_dir_paths)
    eLog.info(f'Post-filtering directories ({len(post_filt_dir_paths)}): \n\t{_dir_paths_str}')

    # start client
    client = lm_api.get_client(CFG.model)
    eLog.info(f'Started LM client: {client}')

    # start running
    if CFG.use_threadpool:
        # parallel runs
        executor = ThreadPoolExecutor(max_workers=CFG.tp_max_workers)

        def _run_trial(traj_path):
            return run_qa_test(CFG, CWD, client, traj_path, CFG.seed)
        results = list(executor.map(_run_trial, filtered_traj_paths))

        executor.shutdown()

    else: 
        results = [run_qa_test(CFG, CWD, client, traj_path, CFG.seed)
                   for traj_path in filtered_traj_paths]

    # Log estimated total time and cost        
    total_time = (datetime.datetime.now() - start_time).total_seconds()

    total_cost = 0.0
    for r in results:
        if "trial_cost_esti" in r:
            total_cost += r["trial_cost_esti"]

    eLog.info(f"Num total trajs: {len(filtered_traj_paths)}. Total time: {total_time}. Estimated total cost: {total_cost}.")
    

if __name__ == '__main__':
    main()
