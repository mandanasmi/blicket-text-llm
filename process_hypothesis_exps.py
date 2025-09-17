#
# Process files and save into duckdb database
# 

import argparse
import json
import yaml
import concurrent.futures
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
import copy
import time

import numpy as np
import duckdb
import uuid

import hypothesis_helper as hp_helper


def setup_logger(output_dir: Path):
    """Set up a logger that writes to the output directory and also prints to stdout."""
    logger = logging.getLogger("experiment_logger")
    logger.setLevel(logging.INFO)
    # Avoid adding multiple handlers if already set.
    if not logger.handlers:
        log_file = output_dir / "processing.log"
        file_handler = logging.FileHandler(log_file, mode="a")
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter("[%(asctime)s][%(levelname)s] - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


def flatten_dict(d, parent_key='', sep='.'):
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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



def process_files(exp_dir: Path, config: dict, results: List[dict], action_logs: List[List[dict]]):
    # Flatten the config dictionary and add to results
    flattened_config = flatten_dict(config)

    def _process_traj_log(traj_log: List[dict]):
        # Separate into action and question logs
        actions_traj = []
        questions_traj = []
        for d in copy.deepcopy(traj_log):
            if "action" in d:
                actions_traj.append(flatten_dict(d)) 
            elif "question" in d:
                questions_traj.append(d)
            else:
                raise ValueError("Unknown action log entry: {}".format(d))

        cur_trial_idx = actions_traj[0]['trial_idx']

        # compute hypothesis
        state_traj = [d['game_state.true_state'] for d in actions_traj]
        state_traj = np.array(state_traj)  # [T, num_object + 1]
        
        for t in range(0, len(state_traj)):
            hypo_left = hp_helper.compute_num_valid_hypothesis(state_traj[:t+1])
            actions_traj[t]['n_hypothesis_left'] = hypo_left

        # Fill up to flattened_config['max_actions_per_trial'] with dir path and config values
        max_traj_len = flattened_config.get('max_actions_per_trial', len(actions_traj))
        for j in range(len(actions_traj), max_traj_len):
            padded_entry = {'steps': j,  'trial_idx': cur_trial_idx, 'n_hypothesis_left': hypo_left}
            actions_traj.append(padded_entry)

        # Fill with key
        for i in range(len(actions_traj)):
            actions_traj[i]['dir_path'] = str(exp_dir)  # absolute dir path as key
            for key, value in flattened_config.items():
                actions_traj[i][f"cfg.{key}"] = value

        return actions_traj, questions_traj
    

    # Process the each trajectory file
    processed_action_traj = []
    processed_question_traj = []
    for i in range(len(action_logs)):
        proc_act_traj, proc_ques_traj = _process_traj_log(action_logs[i])
        processed_action_traj.append(proc_act_traj)
        processed_question_traj.append(proc_ques_traj)

    # Process the results file
    processed_results = []
    for i in range(len(results)):
        result = copy.deepcopy(results[i])
        result["dir_path"] = str(exp_dir)  # absolute dir path as key
        for key, value in flattened_config.items():
            result[f"cfg.{key}"] = value
        processed_results.append(result)
    
    # ==
    # Add things from action and QA trajectories file to results file

    def _get_avg_response_len(traj: List[dict]) -> Tuple[float, float]:
        """Takes in """
        if len(traj) == 0:
            return None 
        
        # avg tokens
        toks_list = []
        resp_lens = []
        for j in range(len(traj)):
            if 'usage' in traj[j] and isinstance(traj[j]['usage'], dict) and 'total_tokens' in traj[j]['usage']:
                toks_list.append(traj[j]['usage']['total_tokens'])
            elif 'usage.total_tokens' in traj[j]:
                toks_list.append(traj[j]['usage.total_tokens'])

            if 'response_message' in traj[j] and isinstance(traj[j]['response_message'], str):
                resp_lens.append(len(traj[j]['response_message']))
            
        avg_toks = np.mean(toks_list) if len(toks_list) > 0 else None
        avg_resp_len = np.mean(resp_lens) if len(resp_lens) > 0 else None
        
        return avg_toks, avg_resp_len
    
    def _find_results_idx(trial_idx: int) -> int:
        for i in range(len(processed_results)):
            if processed_results[i].get('trial_idx') == trial_idx:
                return i
        return None

    # Add from action trajectory to results files
    for i in range(len(processed_action_traj)):
        cur_action_traj = processed_action_traj[i]
        if len(cur_action_traj) == 0:
            continue

        trial_idx = cur_action_traj[0]['trial_idx']
        r_idx = _find_results_idx(trial_idx)
        if r_idx is None:
            continue

        processed_results[r_idx]['n_hypothesis_left'] = cur_action_traj[-1]['n_hypothesis_left']

        avg_act_tokens, avg_act_resp_len = _get_avg_response_len(cur_action_traj)
        processed_results[r_idx]['avg_act_tokens'] = avg_act_tokens
        processed_results[r_idx]['avg_resp_len'] = avg_act_resp_len

    # Add things from question trajectory file to results file
    for i in range(len(processed_question_traj)):
        cur_question_traj = processed_question_traj[i]
        if len(cur_question_traj) == 0:
            continue

        trial_idx = cur_question_traj[0]['trial_idx']
        r_idx = _find_results_idx(trial_idx)
        if r_idx is None:
            continue

        avg_qa_tokens, avg_qa_resp_len = _get_avg_response_len(cur_question_traj)
        processed_results[r_idx]['avg_qa_tokens'] = avg_qa_tokens
        processed_results[r_idx]['avg_qa_resp_len'] = avg_qa_resp_len
    

    # ==
    # un-nest
    processed_action_traj = [item for sublist in processed_action_traj for item in sublist]
    processed_question_traj = [item for sublist in processed_question_traj for item in sublist]

    # strip
    def _strip(list_d):
        return [{k: strip_np(v) for k, v in d.items()} for d in list_d]
    
    processed_results = _strip(processed_results)
    processed_action_traj = _strip(processed_action_traj)
    processed_question_traj = _strip(processed_question_traj)

    return processed_results, processed_action_traj, processed_question_traj


def process_experiment(exp_dir: Path, output_base: Path):
    """
    Process one experiment directory and save processed data into two DuckDB databases:
    one for the results and one for the action logs.
    """
    # Set up a logger in this process.
    logger = setup_logger(output_base)

    # Determine the output directory for this experiment, preserving folder hierarchy.
    # NOTE: assume output is under the 'exp_output' folder
    rel_path = str(exp_dir).split('exp_output/')[1]
    out_exp_dir = output_base / rel_path

    results_db_path = out_exp_dir / "results.duckdb"
    action_log_db_path = out_exp_dir / "trial_action_trajs.duckdb"
    question_log_db_path = out_exp_dir / "trial_question_trajs.duckdb"
    
    if results_db_path.exists() and action_log_db_path.exists() and question_log_db_path.exists():
        msg = f"Skipping already processed experiment: {exp_dir}"
        logger.info(msg)
        return True

    out_exp_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Read experiment files
    # ----------------------------
    results_file = exp_dir / "results.jsonl"
    results = []
    try:
        with results_file.open("r") as f:
            for line in f:
                results.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error reading {results_file}: {e}")
        return
    
    config_file = exp_dir / ".hydra" / "config.yaml"
    config = {}
    try:
        with config_file.open("r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error reading {config_file}: {e}")
        return 

    actions_logs = []
    for file_path in exp_dir.glob("action_log_trial-*.jsonl"):
        cur_action_log = []
        try:
            with file_path.open("r") as f:
                for line in f:
                    cur_action_log.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return
        actions_logs.append(cur_action_log)
    if len(actions_logs) == 0:
        logger.warning(f"No action logs found in {exp_dir}")
        return
    
    # ----------------------------
    # Process the data
    # ----------------------------
    try:
        processed_results, processed_actions_logs, processed_questions_logs \
            = process_files(exp_dir, config, results, actions_logs)
    except ValueError as e:
        # logger.warning(f"Error processing files in {exp_dir}: {e}")
        return
    
    # ------------------------------------------
    # Save processed data to DuckDB databases
    # ------------------------------------------
    # Generate a unique ID for the temp file name
    unique_id = uuid.uuid4()

    # Save results data
    temp_results_file = out_exp_dir / f"temp_results_{unique_id}.jsonl"
    try:
        # Save results data
        con = duckdb.connect(str(results_db_path))
        temp_results_file = out_exp_dir / "temp_results.jsonl"
        with temp_results_file.open("w") as f:
            for row in processed_results:
                f.write(json.dumps(row) + "\n")
        con.execute("CREATE TABLE results AS SELECT * FROM read_json_auto(?)", (str(temp_results_file),))
        con.close()
        temp_results_file.unlink()  # remove temporary file
    except Exception as e:
        logger.error(f"Error processing results in {exp_dir}: {e}")

    try:
        # Save action logs data
        con = duckdb.connect(str(action_log_db_path))
        temp_action_file = out_exp_dir / f"temp_action_logs_{unique_id}.jsonl"
        with temp_action_file.open("w") as f:
            for row in processed_actions_logs:
                f.write(json.dumps(row) + "\n")
        con.execute("CREATE TABLE action_logs AS SELECT * FROM read_json_auto(?)", (str(temp_action_file),))
        con.close()
        temp_action_file.unlink()  # remove temporary file
    except Exception as e:
        logger.error(f"Error processing action logs in {exp_dir}: {e}")

    try:
        # Save questions logs data
        con = duckdb.connect(str(question_log_db_path))
        temp_questions_file = out_exp_dir / f"temp_questions_logs_{unique_id}.jsonl"
        with temp_questions_file.open("w") as f:
            for row in processed_questions_logs:
                f.write(json.dumps(row) + "\n")
        con.execute("CREATE TABLE question_logs AS SELECT * FROM read_json_auto(?)", (str(temp_questions_file),))
        con.close()
        temp_questions_file.unlink()  # remove temporary file
    except Exception as e:
        logger.error(f"Error processing question logs in {exp_dir}: {e}")

    #msg = f"Processed experiment: {exp_dir}"
    #logger.info(msg)
    return True

def main():
    # ===
    # Parse
    parser = argparse.ArgumentParser(
        description="Process experiments and save outputs into DuckDB databases using pathlib, a progress bar, and logging."
    )
    parser.add_argument(
        "results_files",
        nargs="+",
        help="Paths to results.jsonl files (wildcards allowed, e.g., 'experiments/*/results.jsonl')."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Base directory for saving processed experiment outputs."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of worker processes to use. Set to 1 for sequential processing."
    )
    args = parser.parse_args()


    # ==
    # Path set-ups
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"Processing experiments. Output dir: {output_dir}")

    start_time = time.time()

    experiment_dirs = set()
    for result_path in args.results_files:
        exp_dir = Path(result_path).resolve().parent
        experiment_dirs.add(exp_dir)
    experiment_dirs = list(experiment_dirs)

    logger.info(f"Found {len(experiment_dirs)} experiments to process. Parent directories:")
    for p in sorted(list(set([str(p.parent) for p in experiment_dirs]))):
        logger.info(f"\t{p}")

    # If max_workers is 1, process sequentially.
    if args.max_workers == 1:
        process_outs = []
        for exp_dir in tqdm(experiment_dirs, desc="Processing experiments"):
            try:
                process_out = process_experiment(exp_dir, output_dir)
                process_outs.append(process_out)
            except (KeyboardInterrupt, SystemExit, EOFError):
                raise
            #except Exception as exc:
            #    msg = f"Experiment {exp_dir} generated an exception: {exc}"
            #    print(msg)
            #    logger.error(msg)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_experiment, exp_dir, output_dir): exp_dir for exp_dir in experiment_dirs}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing experiments"):
                exp_dir = futures[future]
                try:
                    future.result()
                except (KeyboardInterrupt, SystemExit, EOFError):
                    raise
                #except Exception as exc:
                #    msg = f"Experiment {exp_dir} generated an exception: {exc}"
                #    print(msg)
                #    logger.error(msg)
        
        # Collect and log the output from all the runs
        process_outs = [future.result() for future in futures if future.done() and not future.exception()]
    
    num_processed = sum([n for n in process_outs if n is not None])
    time_elapsed = time.time() - start_time
    logger.info(f"Processed {num_processed} experiments (out of {len(experiment_dirs)}) in {time_elapsed:.2f} seconds.")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
