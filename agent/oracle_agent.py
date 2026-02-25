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
# TODO delete these


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


def hypothesis_fn(x: np.array, blicket_id_mask: np.array, rule: str):
    """
    Computes f(x), where f is defined by blicket_id_mask and the rule
    """
    blicket_id_mask = np.array(blicket_id_mask, dtype=np.bool)
    if rule == "conjunctive":
        return np.all(x[blicket_id_mask])
    elif rule == "disjunctive":
        return np.any(x[blicket_id_mask])
    else:
        raise ValueError(f"Unknown rule: {rule}")


def compute_hypothesis_probs(hypothesis_space: List[Tuple[np.array, str]], obs_mat: np.array):
    """
    Compute the posterior probability of each hypothesis given a set of observations

    
    """
    # Compute p(y=1 | x) for each hypothesis
    corrects = []
    for obs in obs_mat:
        obj_x = obs[:-1]
        machine_y = obs[-1]

        preds = np.array([hypothesis_fn(obj_x, *h_tup) 
                          for h_tup in hypothesis_space])  # [num_hypotheses,]
        preds_correct = preds == machine_y  # [num_hypotheses,]
        corrects.append(preds_correct)

    corrects = np.stack(corrects, axis=0)  # [num_observations, num_hypotheses]
    all_correct = np.all(corrects, axis=0)  # [num_hypotheses,]

    # only consider this subset of all correct hypothesis
    posterior_prob = all_correct / all_correct.sum()  # [num_hypotheses,]

    return posterior_prob

# ==
#

class OracleAgent(Agent):
    """
    Agent using the ground-truth state
    """
    def __init__(self, horizon):
        super().__init__(horizon, filter_actions=False)

        self.state_queue = []
        self.object_names = []

        self.action_counts = None

        self._rng = np.random.RandomState()

        # 
        self.hypothesis_space = None
        self.hypothesis_probs = None

    def _initialize_hypothesis_space(self, num_objects):
        """
        Initialize the hypothesis space based on the initial state

        There are 2^N - 1 possible blicket identities (ignoring case of none of 
        objects are blickets), and 2 rules, for a space of size 2 * (2^N-1).
        """
        rules = ["conjunctive", "disjunctive"]

        blic_identities = list(itertools.product([0, 1], repeat=num_objects))
        blic_identities = [identity for identity in blic_identities if any(identity)]  # remove all zeros

        hyp_space = list(itertools.product(blic_identities, rules))

        self.hypothesis_space = hyp_space
        self.hypothesis_probs = np.ones(len(hyp_space)) / len(hyp_space)
    
    def init_episode(self):
        """Call at start of episode to initialize observation"""
        self.state_queue = []
        self.object_names = []

        self.action_counts = None

        self._rng = np.random.RandomState()

        self.hypothesis_space = None
        self.hypothesis_probs = None

        return
    
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

        self.obs_queue.append(obs)
        self.state_queue.append(true_state)
        self.object_names = copy.deepcopy(obj_names)

        # If hypothesis space is not initialized, initialize it
        if self.hypothesis_space is None:
            self._initialize_hypothesis_space(len(obj_names))

        # Construct prior over hypothesis space over observations seen so far
        obs_mat = np.stack(self.state_queue, axis=0)  # [num_observations, num_objects+1]
        hypothesis_prior = compute_hypothesis_probs(self.hypothesis_space, obs_mat)


        # Generate the set of candidate next states X to reach, for now only
        # consider X's which are *one action* away from the current state
        # TODO: I don't know if this is a good idea or not, we shoulod consider multi-steps?
        candidate_xs = []
        for i in range(len(obj_names)):
            flipped_state = true_state[:-1].copy()
            flipped_state[i] = 1 - flipped_state[i]  # flip the bit
            candidate_xs.append(flipped_state)
        candidate_xs = np.array(candidate_xs)  # [num_objects, num_objects]

        # Compute f(x) for each candidate x, shape [num_candidates, num_hypotheses]
        # N.B. p(y|x, f) = 1 if y == f(x), 0 otherwise
        f_xs = []
        for x in candidate_xs:
            f_xs.append([hypothesis_fn(x, *h_tup) for h_tup in self.hypothesis_space])
        f_xs = np.array(f_xs)  # [num_candidates, num_hypotheses] boolean

        # Compute marginal p(y=1|x) for each candidate x, shape [num_candidates]
        y_pos_marg = np.sum((f_xs == 1) * hypothesis_prior[None,:], axis=1)

        # Compute p(f|y, x) for each candidate x, shape [num_candidates, num_hypotheses]
        # p(f | y=1, x) = p(y=1 | x, f) * p(f) / p(y=1 | x)
        p_f_pos_y_x = (f_xs == 1) * hypothesis_prior[None,:] / (y_pos_marg[:,None] + 1e-8)
        # p(f | y=0, x) = p(y=0 | x, f) * p(f) / p(y=0 | x)
        p_f_neg_y_x = (f_xs == 0) * hypothesis_prior[None,:] / ((1 - y_pos_marg)[:,None] + 1e-8)


        # Compute expected entropy H(f | y, x) for each candidate
        # First, H(f|y=1,x) = - sum_f p(f|y=1,x) log p(f|y=1,x)
        ent_pos_y = - np.sum(p_f_pos_y_x * np.log(p_f_pos_y_x + 1e-8), axis=1)  # [num_candidates,]
        ent_neg_y = - np.sum(p_f_neg_y_x * np.log(p_f_neg_y_x + 1e-8), axis=1)  # [num_candidates,]

        # expected entropy
        exp_ent = y_pos_marg * ent_pos_y + (1 - y_pos_marg) * ent_neg_y

        # expected information gain
        prior_ent = - np.sum(hypothesis_prior * np.log(hypothesis_prior + 1e-8))
        info_gain = prior_ent - exp_ent  # [num_candidates,]

        # pick the candidate with the highest information gain, sampling using
        # softmax to break ties
        sm_probs = np.exp(info_gain / 0.01) / np.exp(info_gain / 0.01).sum()
        chosen_x_idx = self._rng.choice(len(candidate_xs), p=sm_probs)
        chosen_x = candidate_xs[chosen_x_idx]


        # pick the action that corresponds to the chosen x
        # assume there is single difference in object configurations, pick that object
        obj_idx = np.argmax(true_state[:-1] != chosen_x) 
        chosen_obj = obj_names[obj_idx]
        action_str = f'put {chosen_obj} on the machine' if chosen_x[obj_idx] \
            else f'take {chosen_obj} off the machine'
        
        if np.sum(hypothesis_prior > 0) == 1:
            action_str = "exit"
        
        # 
        self.acts_queue.append(action_str)

        act_info = {
            #"model": "None",
            #"system_message": "None",
            #"prompt": "None",
            #"response_message": "None",
            #"usage": "None",
            "n_nonzero_hypothesis": np.sum(hypothesis_prior > 0).item(),
            "prior_entropy": prior_ent.item(),
            "max_info_gain": info_gain.max().item(),
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
        state_traj = np.stack(self.state_queue, axis=0)  # [T, num_object + 1]

        post_prob = compute_hypothesis_probs(self.hypothesis_space, state_traj)  # [num_hypotheses,]
        best_guess_idx = np.argmax(post_prob)
        guess_blick_id, guess_rule = self.hypothesis_space[best_guess_idx]

        ans = guess_blick_id[questioned_obj_idx] == 1
        """
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
        

        ans = questioned_obj_idx in dep_item_idxs
        """

        # 
        ans_info = {
            "best_guess_blicket_ids": list(guess_blick_id),
            "best_guess_rule": guess_rule,
            "api_error": False,
        }
    
        return ans, ans_info

    def answer_rule_inference(self, env: Optional[object] = None):
        if len(self.state_queue) == 0:
            return "Unknown", {"api_error": False}
        state_traj = np.stack(self.state_queue, axis=0)
        post_prob = compute_hypothesis_probs(self.hypothesis_space, state_traj)
        best_guess_idx = np.argmax(post_prob)
        _, guess_rule = self.hypothesis_space[best_guess_idx]
        return f"The rule appears to be {guess_rule}.", {"best_guess_rule": guess_rule, "api_error": False}

    def answer_rule_type(self, blicket_answers: dict, rule_inference_response: str, env: Optional[object] = None):
        if len(self.state_queue) == 0:
            return "unknown", {"api_error": False}
        state_traj = np.stack(self.state_queue, axis=0)
        post_prob = compute_hypothesis_probs(self.hypothesis_space, state_traj)
        best_guess_idx = np.argmax(post_prob)
        _, guess_rule = self.hypothesis_space[best_guess_idx]
        return guess_rule, {"best_guess_rule": guess_rule, "api_error": False}
    
