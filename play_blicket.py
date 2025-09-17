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


def parse_args():
    parser = argparse.ArgumentParser(description="Play Blicket Text Environment")
    parser.add_argument("--num_objects", type=int, default=4, help="Number of objects in the environment")
    parser.add_argument("--num_blickets", type=int, default=2, help="Number of blickets in the environment")
    parser.add_argument("--init_prob", type=float, default=0.1, help="Initial probability for object on machine")
    parser.add_argument("--noise", type=float, default=0.0, help="Probability of machine light to randomly flip")
    parser.add_argument("--rule", type=str, default="conjunctive", help="Rule type for determining blickets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    CWD = os.getcwd()
    print(f'Current working directory: {CWD}')

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # environment
    env = blicket_text.BlicketTextEnv(
        num_objects=args.num_objects,
        num_blickets=args.num_blickets,
        init_prob=args.init_prob, 
        rule=args.rule,
        transition_noise=args.noise,
        seed=args.seed,
    )

    exp_data = {
        "trial_idx": [],
        "unique_state_visited": [],
        "turn_machine_on": [],
        "num_steps": [],
        "total_reward": [],
        "num_questions": [],
        "num_correct": [],
    }

    # ==
    # Run Game
    game_state = env.reset()
    obs = game_state['feedback']
    done = False

    steps = 0
    total_reward = 0.0
    while not done:
        print(obs)
        action = input("> ")

        game_state, reward, done = env.step(action)
        obs = game_state['feedback']

        total_reward += reward
        steps += 1

    exp_data["num_steps"].append(steps)
    exp_data["total_reward"].append(total_reward)
    exp_data["unique_state_visited"].append(game_state["unique_state_visited"])
    exp_data["turn_machine_on"].append(game_state["turn_machine_on"])

    # ==
    # Eval Q & A
    n_questions = 0
    n_correct = 0

    all_object_names = env.object_names
    blicket_names = [all_object_names[i] for i in env.blicket_indices]
    
    print("\n\n === Question & Answer ===")
    for obj_name in all_object_names:
        ans_good = False
        while not ans_good:
            str_answer = input(f"Is {obj_name} a blicket? (y/n): ").strip().lower()
            if str_answer in ["y", "n"]:
                ans_good = True
                bool_answer = str_answer == "y"
            else:
                print("Invalid answer. Please enter 'y' or 'n'.")

        if bool_answer == (obj_name in blicket_names):
            n_correct += 1
        n_questions += 1

    exp_data["num_questions"].append(n_questions)
    exp_data["num_correct"].append(n_correct)

    # Print experiment data
    print("\n\n=== Completed Run Data: ===")
    for key, value in exp_data.items():
        print(f"{key}: {value}")
    print(f"Blicket Indices: {env.blicket_indices}")


if __name__ == '__main__':
    main()
