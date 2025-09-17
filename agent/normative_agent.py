import ast
import copy
import os
import re
import itertools
from typing import List, Dict, Tuple, Optional

from collections import OrderedDict

import numpy as np
import openai
import backoff
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif


from agent.agents import Agent

# ==
# 


def estimate_independence(data: np.ndarray):
    df = pd.DataFrame(data).fillna("missing")
    df = df.apply(lambda x: pd.factorize(x)[0])  # encode to categorical

    analysis = {}
    for col1, col2 in itertools.combinations(df.columns, 2):
        # Chi-Square Test
        chi2_s, chi2_p, _, _ = chi2_contingency(pd.crosstab(df[col1], df[col2]))
        mi_score = mutual_info_classif(df[[col1]], df[col2], discrete_features=True)

        analysis[(col1, col2)] = {
            "chi2_statistic": chi2_s,
            "chi2_p_value": chi2_p,
            "mi_score": mi_score.item(),  # assume it is scalar
        }

    return analysis


def compute_chi2_p_value(data: np.ndarray):
    df = pd.DataFrame(data).fillna("missing")
    df = df.apply(lambda x: pd.factorize(x)[0])  # encode to categorical

    analysis = []
    for col1, col2 in itertools.combinations(df.columns, 2):
        chi2_s, chi2_p, _, _ = chi2_contingency(pd.crosstab(df[col1], df[col2]))
        analysis.append((col1, col2, chi2_s, chi2_p))

    analysis_df = pd.DataFrame(analysis, columns=['x1', 'x2', 'statistic', 'p_value'])
    return analysis_df


def compute_mutual_info(data: np.ndarray):
    df = pd.DataFrame(data).fillna("missing")
    df = df.apply(lambda x: pd.factorize(x)[0])  # encode to categorical

    analysis = []
    for col1, col2 in itertools.combinations(df.columns, 2):
        mi_score = mutual_info_classif(df[[col1]], df[col2], discrete_features=True)
        analysis.append((col1, col2, mi_score.item()))
    
    analysis_df = pd.DataFrame(analysis, columns=['x1', 'x2', 'mi_score'])
    return analysis_df


# ==
#

class NormativeAgent(Agent):
    """
    Agent using the ground-truth state
    """
    def __init__(self, horizon, explore_strategy: str, indep_test: str):
        super().__init__(horizon, filter_actions=False)
        self.explore_strategy = explore_strategy
        self.indep_test = indep_test

        self.state_queue = []
        self.object_names = []

        self.action_counts = None

        self._rng = np.random.RandomState()
    
    def create_history_obs(self, current_obs: str = None):
        """
        Create a history of actions and observations in a formatted string
        """
        formatted_lines = []
        for i, (action, items_dict, observation) in enumerate(zip(self.acts_queue, self.items_queue, self.obs_queue), 1):
            formatted_lines.append(observation)
            formatted_lines.append(f"Item-Attribute Set: {items_dict}")
            formatted_lines.append(f"> {action}\n")

        if len(self.obs_queue) > len(self.acts_queue):
            formatted_lines.append(self.obs_queue[-1])
        
        if current_obs:
            formatted_lines.append(current_obs)

        result_string = "\n".join(formatted_lines)
        return result_string

    def act(self, obs, game_state):
        """Overwrite the parent act method directly"""
        true_state = game_state['true_state']
        obj_names = game_state['object_names']

        if self.action_counts is None:
            self.action_counts = np.ones((len(obj_names), 2))  # start at 1 to smooth
        
        # pick item based on inverse frequency
        if self.explore_strategy == "random":
            obj_idx = self._rng.choice(len(obj_names))  # random object 
            obj_state_idx = self._rng.choice(2)  # random proposed state
        elif self.explore_strategy == "object_count":
            count_vec = self.action_counts.mean(axis=1)
            intr_value = 1 / np.sqrt(count_vec)

            intr_prob = np.exp(intr_value / 0.01)  # low temp, mainly for the same counts
            intr_prob /= intr_prob.sum()
            obj_idx = self._rng.choice(len(obj_names), p=intr_prob)

            # try to flip the state
            obj_state_idx = 1 - true_state[obj_idx]  # flip te state
        else:
            raise ValueError(f"Unknown exploration strategy: {self.explore_strategy}")
        
        chosen_obj = obj_names[obj_idx]
        action_str = f'put {chosen_obj} on the machine' if obj_state_idx == 1 \
            else f'take {chosen_obj} off the machine'
        
        # 
        self.obs_queue.append(obs)
        self.acts_queue.append(action_str)
        self.state_queue.append(true_state)
        self.object_names = copy.deepcopy(obj_names)

        self.action_counts[obj_idx, obj_state_idx] += 1  # increment count

        act_info = {
            #"model": "None",
            #"system_message": "None",
            #"prompt": "None",
            #"response_message": "None",
            #"usage": "None",
            "api_error": False,
        } 

        return action_str, act_info
    
    def add_obs_to_queue(self, obs, game_state):
        true_state = game_state['true_state']
        self.obs_queue.append(obs)
        self.state_queue.append(true_state)
    
    def answer_tf(self, question: str, env: Optional[object] = None):
        # parse
        true_index = [i for i, n in enumerate(self.object_names) if n in question]
        questioned_obj_idx = true_index[0]  # assume unique
        machine_idx = len(self.object_names)  # assume last one is always machine 

        # stat test
        state_traj = np.stack(self.state_queue, axis=0)

        if self.indep_test == "mi":
            mi_df = compute_mutual_info(state_traj)
            df = mi_df[mi_df['x2']==machine_idx]  # only relationship to machine
            df = df[df['mi_score'] > 0.01]  # threshold  TODO: make custom
            dep_item_idxs = df['x1'].tolist()

        elif self.indep_test == "chi2":
            chi2_df = compute_chi2_p_value(state_traj)
            df = chi2_df[chi2_df['x2']==machine_idx]  # only relationship to machine
            df = df[df['p_value'] > 0.05]  # not significant -> possibly dependent
            dep_item_idxs = df['x1'].tolist()
        else:
            raise ValueError(f"Unknown independence test: {self.indep_test}")       

        #
        ans = questioned_obj_idx in dep_item_idxs

        # 
        ans_info = {
            "api_error": False,
        }
    
        return ans, ans_info
    
