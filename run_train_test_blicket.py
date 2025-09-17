#!/usr/bin/env python3
import os
import json
import random
import re
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

client = OpenAI()

SYSTEM_MESSAGE = (
    "You are an intelligent, curious agent. You are playing a game where you are in a room with different objects, and a machine. "
    "Some of these objects are blickets. You can't tell which object is a blicket just by looking at it, but they have blicketness inside of them. "
    "Blicketness makes the machine turn on, following some hidden rule.\n\n"
    "You will be given a history of another agent's interactions within the same game, where the agent's action is indicated with '>'. "
    "The objects and the machine that you're given are the same as the ones given to that agent. You will then be asked a True/False question. "
    "You should answer the question to the best of your ability based on the given history of interaction.\n\n"
    "Your goals are:\n"
    "- Infer from the interactions which objects are blickets.\n"
    "- Infer the underlying rule for how the machine turns on.\n"
    "- Do the above in order to answer the question correctly.\n\n"
    "Tips:\n"
    "- All objects can be either on the machine or on the floor."
)

QUESTION_INSTRUCTION = (
    "Based on the information above, is object A a blicket?\n\n"
    "Directly output the answer in the format '> True/False'. Ensure only one answer is included."
)

def extract_answer(text: str):
    match = re.search(r">\s*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower() == "true"
    return None

def query_llm(system_message: str, user_message: str, model: str = "gpt-4", temperature: float = 0.0):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=150
    )
    return response.choices[0].message.content

def load_prompt_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)

def load_training_prompt_by_condition(two_training_trials: bool, condition: str):
    if two_training_trials:
        path = os.path.join(os.path.dirname(__file__), "agent", "prompts", "train_prompts_2trials.json")
    else: 
        path = os.path.join(os.path.dirname(__file__), "agent", "prompts", "train_prompts_1trial.json")
    data = load_prompt_json(path)
    if condition not in data:
        raise ValueError(f"Condition '{condition}' not found in training prompts.")
    return "".join(data[condition])

def load_test_prompt_by_key(key: str = "test"):
    path = os.path.join(os.path.dirname(__file__), "agent", "prompts", "test_prompt.json")
    data = load_prompt_json(path)
    if key not in data:
        raise ValueError(f"Test key '{key}' not found in test prompts.")
    return "".join(data[key])

import datetime
import hashlib
from pathlib import Path

CWD = Path(os.getcwd())  # or override as needed

def run_trial(training_prompt: str, test_prompt: str, model: str, temperature: float,
              trial_idx: int, seed_idx: int, condition: str, save_trial_log: bool = True):
    full_prompt = training_prompt + "\n\n" + test_prompt + "\n\n" + QUESTION_INSTRUCTION
    response_msg = None
    api_error = False
    cost = 0.0
    ans_bool = None
    usage = {}

    try:
        response_text = query_llm(SYSTEM_MESSAGE, full_prompt, model=model, temperature=temperature)
        response_msg = response_text
        ans_bool = extract_answer(response_text)

        if ans_bool is None:
            ans_bool = random.choice([True, False])
            api_error = True

    except (KeyboardInterrupt, EOFError):
        raise
    except Exception as e:
        print(f"Error during API call: {e}")
        ans_bool = random.choice([True, False])
        api_error = True

    # Simulate response usage if needed
    response_usage = {
        "completion_tokens": len(response_msg.split()) if response_msg else 0,
        "total_tokens": len(full_prompt.split()) + (len(response_msg.split()) if response_msg else 0)
    }

    trial_data = {
        "metric_type": "per_question",
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "system_message": SYSTEM_MESSAGE,
        "user_prompt": full_prompt,
        "response_msg": response_msg,
        "answer": str(ans_bool).lower(),
        "api_error": str(api_error).lower(),
        "response_model": model,
        "usage": response_usage,
        "estimated_cost": cost,
        "condition": condition,
        "trial_idx": trial_idx,
        "seed": seed_idx
    }

    uniq_id = hashlib.md5(full_prompt.encode()).hexdigest()[:8]
    if save_trial_log:
        with open("trial_results.jsonl", "w") as f:
            f.write(json.dumps(trial_data) + "\n")

    return ans_bool, response_msg


def save_results(results, cfg, mean_prop_true, std_prop_true, mean_prop_false, std_prop_false):
    output = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "results": results,
        "aggregated": {
            "mean_prop_true": mean_prop_true,
            "std_prop_true": std_prop_true,
            "mean_prop_false": mean_prop_false,
            "std_prop_false": std_prop_false
        }
    }
    with open("results.json", "w") as f:
        json.dump(output, f, indent=4)
    print("Results saved to results.json")

@hydra.main(config_path=".", config_name="run_train_test_blicket")
def main(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("----------------------------------------\n")

    
    training_prompt = load_training_prompt_by_condition(cfg.two_training_trials, cfg.condition)
    test_prompt = load_test_prompt_by_key()
    print("Training prompt:")
    print(training_prompt)
    results = []
    seed_proportions_true, seed_proportions_false = [], []

    for seed_idx in range(1, cfg.num_seeds + 1):
        random.seed(seed_idx)
        np.random.seed(seed_idx)
        true_count, false_count = 0, 0

        print(f"Seed {seed_idx}:")

        for trial_idx in range(1, cfg.num_trials + 1):
            answer, raw_response = run_trial(training_prompt, test_prompt, cfg.model, cfg.temperature, trial_idx, seed_idx, cfg.condition, cfg.save_trial_log)
            if answer is None:
                answer = random.choice([True, False])
                print(f"  Trial {trial_idx}: No answer parsed; randomly set to {answer}")
            else:
                print(f"  Trial {trial_idx}: Answer = {answer}")
            true_count += int(answer)
            false_count += int(not answer)

        proportion_true = true_count / cfg.num_trials
        proportion_false = false_count / cfg.num_trials
        seed_proportions_true.append(proportion_true)
        seed_proportions_false.append(proportion_false)

        print(f"  Proportion True: {proportion_true:.3f}")
        print(f"  Proportion False: {proportion_false:.3f}\n")

        results.append({
            "seed": seed_idx,
            "trial_details": cfg.num_trials,
            "proportion_true": proportion_true,
            "proportion_false": proportion_false
        })

    mean_prop_true = np.mean(seed_proportions_true)
    std_prop_true = np.std(seed_proportions_true)
    mean_prop_false = np.mean(seed_proportions_false)
    std_prop_false = np.std(seed_proportions_false)

    print("========================================")
    print(f"Condition: {cfg.condition}")
    print(f"Mean proportion True: {mean_prop_true:.3f} ± {std_prop_true:.3f}")
    print(f"Mean proportion False: {mean_prop_false:.3f} ± {std_prop_false:.3f}")
    print("========================================")

    with open("results.jsonl", "a") as f:
        json.dump({
            "config": OmegaConf.to_container(cfg, resolve=True),
            "results": results,
            "aggregated": {
                "mean_prop_true": mean_prop_true,
                "std_prop_true": std_prop_true,
                "mean_prop_false": mean_prop_false,
                "std_prop_false": std_prop_false
            }
        }, f)
        f.write("\n")

    print("Results saved to results.jsonl")

if __name__ == "__main__":
    main()
