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


from agent.agents import Agent, RULE_INFERENCE_QUESTION, RULE_TYPE_QUESTION

# ==
# 

PRINT_LLM_DEBUG = False

OAI_CLIENT = openai.OpenAI()


ENV_DESCRIPTION = '''You are an agent playing the Blicket game, a text-based adventure game where you are in a room with different objects and a machine. A subset of the objects are 'blickets', which turn on the light on the machine following some rule. 
Your goal is to explore the relationship between the objects and the machines to determine which objects are blickets, and the rule for turning on the machine.

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
#DEFAULT_SYSTEM_MESSAGE = ENV_DESCRIPTION + '\n\n' + '''You will be prompted at each turn to choose actions.''' + \
#                         '\n\n' + RESPONSE_INSTRUCTION

ITEM_ATTR_SYSTEM_MESSAGE = ENV_DESCRIPTION + '\n\n' + '''For each observation, come up with an item-attribute set which you think is relevant.
- Format it as json, e.g. {"item 1": "attribute a", "item 2": "attribute b", "item 3", "attribute a", ...}.
- The items should be unique, but the attributes can be re-used.
- Come up with a set of item names by looking at the full history of past observations
- Include any and all item names that you think may be potentially relevant and related to the goal, including the goal itself.
- You can also include items that are currently un-observed but relevant for achieving the goal.
- Come up with a set of attribute names by looking at the full history of past observations and actions, including if past observations indicate that items' attributes have changed.
- Try to keep each attribute short. Do not use multiple spellings to describe the same attribute.
- Be accurate and concise in your description.
''' + '\n' + RESPONSE_INSTRUCTION


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_llm_api(model, system_message, msg, temperature=0.3):
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
    
    # 
    response = OAI_CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        **chat_kwargs,
    )

    # 
    def _calculate_openai_cost(prompt_tokens, completion_tokens, model="gpt-4-turbo"):
        pricing = {
            "gpt-4o-2024-05-13": {"input": 5, "output": 15},
            # "gpt-4-turbo-2024-04-09": 
            # "gpt-3.5-turbo-0125": 
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

def estimate_independence(items_queue: List[Dict[str, str]]):
    df = pd.DataFrame(items_queue).fillna("missing")
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

# ==
#

class RandActionItemAttrAgent(Agent):
    """
    This agent takes random actions. It infers the item-attribute set at each
    time-step and stores them in a list. It can do independence testing on the
    stored item-attribute sets to provide additional context for the prompt
    during the Q&A phase.
    """
    def __init__(self, horizon, filter_actions, model, react, temperature, do_indep_analysis):
        super().__init__(horizon, filter_actions)
        self.model = model
        self.react = react
        self.temperature = temperature
        self.do_indep_analysis = do_indep_analysis

        self.system_message = ITEM_ATTR_SYSTEM_MESSAGE.replace('#HORIZON#', str(horizon))

        self.items_queue = []

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
        prompt = "Interact with the environment to find the blickets and discover the rule.\n"
        prompt += self.create_history_obs(obs)

        if self.filter_actions:
            raise NotImplementedError  # TODO: implement this later based on IGE but without priviledged information
        
        if self.react:
            prompt += "\nFirst briefly reason and think about what are the relevant items and attributes for this task. "\
                "Then, output the item-attribute set as: '> {item-attribute-set}\'."
        else:
            prompt += "\nDirectly output the item-attribute set in the format \'> {...}\'. "

        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt, 
                                           temperature=self.temperature)
            self.total_cost += cost

            response_msg = response.choices[0].message.content
            response_usage = response.usage
            extracted_txt_output = extract_action(response_msg)

            if PRINT_LLM_DEBUG:
                print('-' * 5, 'prompt', '-' * 5)
                print(prompt)
                print('-' * 5, 'response', '-' * 5)
                print(response_msg)
                print('=' * 20)
                print()

            items_dict = ast.literal_eval(extracted_txt_output)
            if not isinstance(items_dict, dict):
                print(f"Failed to parse into items dict: {extracted_txt_output}")
                items_dict = {}

        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as e:
            print(f'Error: {e}')
            items_dict = {}
            api_error = True

        # NOTE: just do random actions
        obj_names = game_state['object_names']
        chosen_obj = self._rng.choice(obj_names)

        if self._rng.rand() < 0.5:
            action = f'put {chosen_obj} on the machine'
        else:
            action = f'take {chosen_obj} off the machine'

        # 
        self.obs_queue.append(obs)
        self.acts_queue.append(action)
        self.items_queue.append(items_dict)

        act_info = {
            "model": self.model,
            "system_message": self.system_message,
            "prompt": prompt,
            "response_message": response_msg,
            "usage": response_usage,
            "api_error": api_error,
            "items_dict": copy.deepcopy(items_dict),
        } 

        return action, act_info
    
    def answer_tf(self, question: str, env: Optional[object] = None):
        prompt = self.create_history_obs()

        prompt += f"\n\nBased on the information you have gathered, answer the following question: {question}\n"

        # 
        if self.do_indep_analysis:
            indep_analysis = estimate_independence(self.items_queue)  # dictionary
            prompt += f"\n\nTo help with the answer, I have also conducted an independence test between the items based on the {len(self.items_queue)} item-attribute sets you provided. "\
                "Below I will, for each pair of items, provide the chi2 statistic, chi2 p-value, and a mutual information score from this analysis.\n\n"
            for pair, analysis in indep_analysis.items():
                prompt += f"Pair: {pair}.\tAnalysis: {analysis}.\n"

        if self.react:
            prompt += "\nFirst briefly reason and think about the information collected. "\
                "Then, output the answer in the format \'> True/False\'. "\
                "Ensure only one answer is included."
        else:
            prompt += "\nDirectly output the answer in the format \'> True/False\'. "\
                "Ensure only one answer is included."
        
        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt)
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
                print('-' * 5, 'prompt', '-' * 5)
                print(prompt)
                print('-' * 5, 'response', '-' * 5)
                print(response_msg)
                print('=' * 20)
                print()

        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as e:
            print(f'Error: {e}')
            api_error = True
            ans = True

        #         
        ans_info = {
            "model": self.model,
            "system_message": self.system_message,
            "prompt": prompt,
            "response_message": response_msg,
            "usage": response_usage,
            "api_error": api_error,
        }
    
        return ans, ans_info

    def answer_rule_inference(self, env: Optional[object] = None):
        prompt = self.create_history_obs()
        prompt += f"\n\n{RULE_INFERENCE_QUESTION}\n"
        prompt += "\nProvide your description."

        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt)
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
            "usage": response_usage,
            "api_error": api_error,
        }
        return response_msg, ans_info

    def answer_rule_type(self, blicket_answers: dict, rule_inference_response: str, env: Optional[object] = None):
        prompt = self.create_history_obs()
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
            response, cost = query_llm_api(self.model, self.system_message, prompt)
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
            "usage": response_usage,
            "api_error": api_error,
        }
        return rule_type, ans_info
    


ITEM_ATTR_ACTION_SYSTEM_MESSAGE = ENV_DESCRIPTION + '\n\n' + '''For each observation, come up with an item-attribute set which you think is relevant.
- Format it as json, e.g. {"item 1": "attribute a", "item 2": "attribute b", "item 3", "attribute a", ...}.
- The items should be unique, but the attributes can be re-used.
- Come up with a set of item names by looking at the full history of past observations
- Include any and all item names that you think may be potentially relevant and related to the goal, including the goal itself.
- You can also include items that are currently un-observed but relevant for achieving the goal.
- Come up with a set of attribute names by looking at the full history of past observations and actions, including if past observations indicate that items' attributes have changed.
- Try to keep each attribute short. Do not use multiple spellings to describe the same attribute.
- Be accurate and concise in your description.

In addition to the item-attribute set, you will also be prompted at each turn to choose actions.
''' + '\n' + RESPONSE_INSTRUCTION


class LMActionItemAttrAgent(RandActionItemAttrAgent):
    """
    This agent takes actions according to the LM. It also infers the item-attribute set at each
    time-step and stores them in a list. It can do independence testing on the
    stored item-attribute sets to provide additional context for the prompt
    during the Q&A phase.
    """
    def __init__(self, horizon, filter_actions, model, react, temperature, do_indep_analysis):
        super().__init__(horizon, filter_actions, model, react, temperature, do_indep_analysis)
        self.system_message = ITEM_ATTR_ACTION_SYSTEM_MESSAGE.replace('#HORIZON#', str(horizon))

    def create_history_obs(self, current_obs: str = None):
        """
        Create a history of actions and observations in a formatted string
        """
        formatted_lines = []
        for i, (action, items_dict, observation) in enumerate(zip(self.acts_queue, self.items_queue, self.obs_queue), 1):
            formatted_lines.append(observation)
            formatted_lines.append(f"> {items_dict}")
            formatted_lines.append(f"> {action}\n")

        if len(self.obs_queue) > len(self.acts_queue):
            formatted_lines.append(self.obs_queue[-1])
        
        if current_obs:
            formatted_lines.append(current_obs)

        result_string = "\n".join(formatted_lines)
        return result_string
    
    def act(self, obs, game_state):
        """Overwrite the parent act method directly"""
        prompt = "Interact with the environment to find the blickets and discover the rule.\n"
        prompt += self.create_history_obs(obs)

        if self.filter_actions:
            raise NotImplementedError  # TODO: implement this later based on IGE but without priviledged information
        
        if self.react:
            prompt += "\nFirst briefly reason and think about what are the relevant items and attributes for this task, and your plan to solve the task. "\
                "Then, output the item-attribute set followed by the action command, exactly as: "\
                "\'> {item 1: attribute 1, ...} \n> command\'. Ensure only one command is included"
        else:
            prompt += "\nDirectly output the item-attribute set in exactly the format \'> {item 1: attribute 1, ...} \n> command\'. "\
                "Ensure only one command is included"

        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt, 
                                           temperature=self.temperature)
            self.total_cost += cost

            response_msg = response.choices[0].message.content
            response_usage = response.usage

            if PRINT_LLM_DEBUG:
                print('-' * 5, 'prompt', '-' * 5)
                print(prompt)
                print('-' * 5, 'response', '-' * 5)
                print(response_msg)
                print('=' * 20)
                print()
            
            # parse
            re_match = re.search(r'>\s*({.*?})\s*>?\s*(.+)', response_msg, re.DOTALL)
            dictionary_str, command = re_match.groups()

            items_dict = ast.literal_eval(dictionary_str)
            action = command
   
            if not isinstance(items_dict, dict):
                print(f"Failed to parse into items dict: {dictionary_str}")
                items_dict = {}
        
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as e:
            print(f'Error: {e}')
            items_dict = {}
            api_error = True
            action = "look"

        # 
        self.obs_queue.append(obs)
        self.acts_queue.append(action)
        self.items_queue.append(items_dict)

        act_info = {
            "model": self.model,
            "system_message": self.system_message,
            "prompt": prompt,
            "response_message": response_msg,
            "usage": response_usage,
            "api_error": api_error,
        } 

        return action, act_info
    
