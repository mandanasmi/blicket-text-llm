import ast
import copy
import os
import re
import itertools
from typing import List, Dict, Tuple, Optional

import numpy as np
import openai
import backoff
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

from agent.agents import Agent

PRINT_LLM_DEBUG = False

OAI_CLIENT = openai.OpenAI()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_base = "https://api.openai.com/v1"


ENV_DESCRIPTION = '''You are an agent playing the Blicket game, a text-based adventure where you explore objects and a machine to determine which objects are blickets.
Your goal is to determine which objects are blickets and the rule for turning on the machine.
Available commands include: look, put ... on ..., take ... off ..., and exit.
You have #HORIZON# steps to complete the task.
'''

RESPONSE_INSTRUCTION = '''Reply concisely and exactly with the requested format.'''

ITEM_ATTR_SYSTEM_MESSAGE = ENV_DESCRIPTION + '\n\n' + '''For each observation, output an item-attribute set in JSON format (e.g., {"item1": "attrA", "item2": "attrB", ...}).
Include any object names you think may be relevant to identifying the blickets.
''' + '\n' + RESPONSE_INSTRUCTION

def query_llm_api(model, system_message, msg, temperature=0.3):
    # (This function wraps the call to OpenAI's API)
    response = OAI_CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=1024,
        top_p=0.2,
    )
    # Cost calculation omitted for brevity.
    return response, 0.0

def extract_action(text):
    action_pattern = r"> (.*?)(?:\.|$)"
    match = re.search(action_pattern, text)
    if match:
        return match.group(1).strip().rstrip('.')
    return None

def estimate_independence(items_queue):
    # If items_queue is not a list, wrap it in a list
    if not isinstance(items_queue, list):
        items_queue = [items_queue]
  
    # Convert items_queue to DataFrame
    df = pd.DataFrame(items_queue)
    
    # If the resulting df is a Series, convert it to a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame().T

    # If df is empty, return an empty dictionary
    if df.empty:
        return {}

    # Fill missing values
    df = df.fillna("missing")
    
    # Convert columns to categorical by factorizing each column
    df = df.apply(lambda x: pd.factorize(x)[0])
    
    analysis = {}
    # Iterate over all combinations of columns
    for col1, col2 in itertools.combinations(df.columns, 2):
        chi2_stat, chi2_p, _, _ = chi2_contingency(pd.crosstab(df[col1], df[col2]))
        mi_score = mutual_info_classif(df[[col1]], df[col2], discrete_features=True)
        analysis[(col1, col2)] = {
            "chi2_statistic": chi2_stat,
            "chi2_p_value": chi2_p,
            "mi_score": mi_score.item(),
        }
    
    return analysis


class LMActionItemAttrAgentChi(Agent):
    """
    An LM-driven agent that, in addition to inferring the item-attribute set,
    conducts chi-square independence tests on the collected sets.
    It filters the tests to include only statistically significant pairs, summarizing
    the top results for the LM during Q&A.
    """
    def __init__(self, horizon, filter_actions, model, react, temperature, do_indep_analysis):
        super().__init__(horizon, filter_actions)
        self.model = model
        self.react = react
        self.temperature = temperature
        self.do_indep_analysis = do_indep_analysis
        self.system_message = ITEM_ATTR_SYSTEM_MESSAGE.replace('#HORIZON#', str(horizon))
        self.items_queue = []
        self.acts_queue = []
        self.obs_queue = []
        self.total_cost = 0.0
        self._rng = np.random.RandomState()

    def create_history_obs(self, current_obs: str = None):
        formatted_lines = []
        for action, items_dict, observation in zip(self.acts_queue, self.items_queue, self.obs_queue):
            formatted_lines.append(observation)
            formatted_lines.append(f"> {items_dict}")
            formatted_lines.append(f"> {action}\n")
        if len(self.obs_queue) > len(self.acts_queue):
            formatted_lines.append(self.obs_queue[-1])
        if current_obs:
            formatted_lines.append(current_obs)
        return "\n".join(formatted_lines)

    def act(self, obs, game_state):
        prompt = "Interact with the environment to find the blickets and discover the rule.\n"
        prompt += self.create_history_obs(obs)
        
        if self.filter_actions:
            raise NotImplementedError("Action filtering is not implemented in this agent.")
        
        if self.react:
            prompt += "\nFirst briefly reason about the relevant items and attributes, then output the item-attribute set in the format '> { ... }'."
        else:
            prompt += "\nDirectly output the item-attribute set in the format '> { ... }'."
        
        api_error = False
        response_msg = ""
        response_usage = {}
        
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt, temperature=self.temperature)
            self.total_cost += cost
            response_msg = response.choices[0].message.content
            response_usage = response.usage  # May be None sometimes; we handle that later.
            extracted_txt = extract_action(response_msg)
            items_dict = ast.literal_eval(extracted_txt) if extracted_txt else {}
        except Exception as e:
            print(f"Error during LLM call: {e}")
            api_error = True
            items_dict = {}
            response_msg = "Error in LLM response"
            response_usage = {}
        
        # Choose a random action from game_state
        obj_names = game_state.get('object_names', [])
        if obj_names:
            chosen_obj = self._rng.choice(obj_names)
            action = f'put {chosen_obj} on the machine' if self._rng.rand() < 0.5 else f'take {chosen_obj} off the machine'
        else:
            action = "look"  # Fallback action
        
        self.obs_queue.append(obs)
        self.acts_queue.append(action)
        self.items_queue.append(items_dict)
        
        act_info = {
            "model": self.model,
            "system_message": self.system_message,
            "prompt": prompt,
            "response_message": response_msg,
            "usage": response_usage if response_usage is not None else {},
            "api_error": api_error,
        }
        
        return action, act_info


    def filter_chi2_stats(self, analysis: Dict[Tuple[str, str], Dict[str, float]], significance=0.05, max_pairs=3):
        """Filter the chi-square results to only include pairs with a p-value below 'significance'
        and return at most 'max_pairs' sorted by descending chi-square statistic."""
        significant = []
        for pair, stats in analysis.items():
            if stats['chi2_p_value'] < significance:
                significant.append((pair, stats))
        significant.sort(key=lambda x: x[1]['chi2_statistic'], reverse=True)
        return significant[:max_pairs]

    def answer_tf(self, question: str, env: Optional[object] = None):
        prompt = self.create_history_obs()
        prompt += f"\n\nBased on the information collected, answer the following question: {question}\n"
        
        if self.do_indep_analysis:
            indep_analysis = estimate_independence(self.items_queue)
            prompt += "\n\nChi-square Analysis Results:\n"
            for pair, stats in indep_analysis.items():
                prompt += (f"Pair {pair}: chi2={stats['chi2_statistic']:.2f}, "
                        f"p={stats['chi2_p_value']:.3f}, MI={stats['mi_score']:.3f}\n")
        
        if self.react:
            prompt += "\nFirst briefly reason about the collected information, then output the answer in the format '> True' or '> False'."
        else:
            prompt += "\nDirectly output the answer in the format '> True' or '> False'."
        
        api_error = False
        response_msg = ""
        response_usage = {}
        bool_answer = None

        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt, temperature=self.temperature)
            self.total_cost += cost
            response_msg = response.choices[0].message.content
            response_usage = response.usage  # May be None
            answer_str = extract_action(response_msg)
            if answer_str == 'True':
                bool_answer = True
            elif answer_str == 'False':
                bool_answer = False
            else:
                bool_answer = False  # or choose a default value
        except Exception as e:
            print(f"Error during LLM call in answer_tf: {e}")
            api_error = True
            response_msg = "Error in answer response"
            response_usage = {}

        # Build ans_info with all required keys
        ans_info = {
            "model": self.model,
            "system_message": self.system_message,
            "prompt": prompt,
            "response_message": response_msg,            # Ensures the key is always present
            "usage": response_usage if response_usage is not None else {},
            "api_error": api_error,
        }
        
        return bool_answer, ans_info

