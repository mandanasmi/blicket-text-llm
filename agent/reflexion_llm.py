import ast
import copy
import os
import re
import itertools
from typing import List, Dict, Optional

from collections import OrderedDict

import hydra
import numpy as np
import openai
import backoff
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

from agent.agents import Agent, RULE_INFERENCE_QUESTION, RULE_TYPE_QUESTION

# ==
# Global settings and environment description

PRINT_LLM_DEBUG = False
OAI_CLIENT = openai.OpenAI()

ENV_DESCRIPTION = '''You are an agent playing the Blicket game, a text-based adventure game where you are in a room with different objects and a machine. A subset of the objects are 'blickets', which turn on the light on the machine following some rule.
Your goal is to explore the relationship between the objects and the machine to determine which objects are blickets, and the rule for turning on the machine.

Here are the available commands:
  look:                describe the current room
  put ... on ...:      put an object on the machine or the floor
  take ... off ...:    take an object off the machine
  test:                test the machine to see if the light turns on
  exit:                exit the game

Tips:
- The machine's light state is only revealed when you use the 'test' command. Put and take do not show whether the light is on or off.
- All objects can be either on the machine or on the floor.
- You should think about how to efficiently explore the relationship between the objects and the machine.

You have #HORIZON# steps to complete the task. You can also exit the task early if you think you understand the relationship between the objects and the machine.
After the task is done, you will be asked which objects are blickets, and the rule for turning on the machine.
'''

RESPONSE_INSTRUCTION = '''Reply concisely and exactly with the requested format.'''

# ==

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_llm_api(model, system_message, msg, temperature=0.3):
    if "o3" in model or "o1" in model:
        chat_kwargs = {"max_completion_tokens": 1024}
    else:
        chat_kwargs = {
            "temperature": temperature,
            "max_tokens": 1024,
            "top_p": 0.2,
            "stop": None,
        }
    
    response = OAI_CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        **chat_kwargs,
    )

    def _calculate_openai_cost(prompt_tokens, completion_tokens, model="gpt-4-turbo"):
        pricing = {
            "gpt-4o-2024-05-13": {"input": 5, "output": 15},
            "gpt-4o-2024-08-06": {"input": 2.5, "output": 10.0},
            "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.6},
            "o3-mini-2025-01-31": {"input": 1.10, "output": 4.40},
            "o1-mini-2024-09-12": {"input": 1.10, "output": 4.40},
        }
        if model not in pricing:
            raise ValueError("Unknown model. Please check OpenAI pricing.")
        input_cost = (prompt_tokens / 1_000_000) * pricing[model]["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing[model]["output"]
        return input_cost + output_cost

    cost = _calculate_openai_cost(
        prompt_tokens=response.usage.prompt_tokens, 
        completion_tokens=response.usage.completion_tokens, 
        model=model
    )
    return response, cost

def extract_action(text):
    action_pattern = r"> (.*?)(?:\.|$)"
    match = re.search(action_pattern, text)
    if match:
        action = match.group(1).strip()
        return action.rstrip('.')
    return None

def estimate_independence(items_queue: List[Dict[str, str]]):
    df = pd.DataFrame(items_queue).fillna("missing")
    df = df.apply(lambda x: pd.factorize(x)[0])  # encode to categorical
    analysis = {}
    for col1, col2 in itertools.combinations(df.columns, 2):
        chi2_s, chi2_p, _, _ = chi2_contingency(pd.crosstab(df[col1], df[col2]))
        mi_score = mutual_info_classif(df[[col1]], df[col2], discrete_features=True)
        analysis[(col1, col2)] = {
            "chi2_statistic": chi2_s,
            "chi2_p_value": chi2_p,
            "mi_score": mi_score.item(),
        }
    return analysis

# ==
class ReflexionPromptsAgent(Agent):
    """
    Agent that optionally uses Reflexion prompting.
    If reflexion=True, the agent reflects on its reasoning and self-critiques before providing the final output.
    If reflexion=False, the agent directly outputs the action or answer.
    """
    def __init__(self, horizon, filter_actions, model, reflexion, temperature, system_msg_path):
        super().__init__(horizon, filter_actions)
        self.model = model
        self.temperature = temperature
        self.reflexion = reflexion  

        if system_msg_path.startswith('/'):
            sys_msg_path = system_msg_path
        else:
            sys_msg_path = os.path.join(hydra.utils.get_original_cwd(), system_msg_path)
        with open(sys_msg_path, 'r') as f:
            sys_message = f.read()
        self.system_message = sys_message.replace('#HORIZON#', str(horizon))
        self.items_queue = []
        self._rng = np.random.RandomState()

    def create_history_obs(self, current_obs: str = None):
        formatted_lines = []
        for action, observation in zip(self.acts_queue, self.obs_queue):
            formatted_lines.append(observation)
            formatted_lines.append(f"> {action}")
        if len(self.obs_queue) > len(self.acts_queue):
            formatted_lines.append(self.obs_queue[-1])
        if current_obs:
            formatted_lines.append(current_obs)
        return "\n".join(formatted_lines)

    def act(self, obs, game_state):
        prompt = "Interact with the environment to find the blickets and discover the rule.\n"
        history_obs = self.create_history_obs(obs)
        prompt += history_obs

        if self.filter_actions:
            raise NotImplementedError  # Implement later if necessary

        if self.reflexion:
            prompt += (
                "\n\nPlease first reflect on your plan to solve the task. "
                "Explain your reasoning and self-evaluate any potential issues before outputting the final command "
                "in the format '> command'. Ensure only one command is included."
            )
        else:
            prompt += (
                "\n\nDirectly output the command in the format '> command'. "
                "Ensure only one command is included."
            )

        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt, temperature=self.temperature)
            self.total_cost += cost
            response_msg = response.choices[0].message.content
            action = extract_action(response_msg)
            response_usage = response.usage
            if PRINT_LLM_DEBUG:
                print('-'*5, 'prompt', '-'*5)
                print(prompt)
                print('-'*5, 'response', '-'*5)
                print(response_msg)
                print('='*20, "\n")
        except Exception as e:
            print(f'Error: {e}')
            action = 'look'
            api_error = True
        self.obs_queue.append(obs)
        self.acts_queue.append(action)
        act_info = {
            "model": self.model,
            "system_message": self.system_message,
            "prompt": prompt,
            "response_message": response_msg,
            "history_obs": history_obs,
            "usage": response_usage,
            "api_error": api_error,
        }
        return action, act_info

    def answer_tf(self, question: str, env: Optional[object] = None):
        history_obs = self.create_history_obs()
        prompt = history_obs
        prompt += f"\n\nBased on the information you have gathered, answer the following question: {question}\n"

        if self.reflexion:
            prompt += (
                "\nPlease first reflect on the collected information and analyze any potential issues with your reasoning, "
                "then output the final answer in the format '> True/False'. Ensure only one answer is included."
            )
        else:
            prompt += (
                "\nDirectly output the answer in the format '> True/False'. Ensure only one answer is included."
            )

        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt, temperature=self.temperature)
            self.total_cost += cost
            response_msg = response.choices[0].message.content
            response_usage = response.usage
            answer_str = extract_action(response_msg)
            if answer_str == 'True':
                ans = True
            elif answer_str == 'False':
                ans = False
            else:
                ans = np.random.choice([True, False])
            if PRINT_LLM_DEBUG:
                print('-'*5, 'prompt', '-'*5)
                print(prompt)
                print('-'*5, 'response', '-'*5)
                print(response_msg)
                print('='*20, "\n")
        except Exception as e:
            print(f'Error: {e}')
            ans = True
            api_error = True
        ans_info = {
            "model": self.model,
            "system_message": self.system_message,
            "prompt": prompt,
            "response_message": response_msg,
            "history_obs": history_obs,
            "usage": response_usage,
            "api_error": api_error,
        } 
        return ans, ans_info

    def answer_rule_inference(self, env: Optional[object] = None):
        history_obs = self.create_history_obs()
        prompt = history_obs
        prompt += f"\n\n{RULE_INFERENCE_QUESTION}\n"
        prompt += "\nProvide your description."

        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt, temperature=self.temperature)
            self.total_cost += cost
            response_msg = response.choices[0].message.content
            response_usage = response.usage
        except Exception as e:
            print(f'Error: {e}')
            response_msg = ""
            api_error = True

        ans_info = {
            "model": self.model,
            "system_message": self.system_message,
            "prompt": prompt,
            "response_message": response_msg,
            "history_obs": history_obs,
            "usage": response_usage,
            "api_error": api_error,
        }
        return response_msg, ans_info

    def answer_rule_type(self, blicket_answers: dict, rule_inference_response: str, env: Optional[object] = None):
        history_obs = self.create_history_obs()
        prompt = history_obs
        prompt += "\n\nYour answers about which objects are blickets:\n"
        for obj_name, ans in blicket_answers.items():
            prompt += f"- {obj_name}: {'Yes' if ans else 'No'}\n"
        prompt += "\n\nYour rule inference:\n"
        prompt += rule_inference_response
        prompt += f"\n\n{RULE_TYPE_QUESTION}\n"

        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt, temperature=self.temperature)
            self.total_cost += cost
            response_msg = response.choices[0].message.content
            response_usage = response.usage
            answer_str = extract_action(response_msg)
            if answer_str and "conjunctive" in answer_str.lower():
                rule_type = "conjunctive"
            elif answer_str and "disjunctive" in answer_str.lower():
                rule_type = "disjunctive"
            else:
                rule_type = "unknown"
        except Exception as e:
            print(f'Error: {e}')
            rule_type = "unknown"
            api_error = True

        ans_info = {
            "model": self.model,
            "system_message": self.system_message,
            "prompt": prompt,
            "response_message": response_msg,
            "history_obs": history_obs,
            "usage": response_usage,
            "api_error": api_error,
        }
        return rule_type, ans_info