import ast
import copy
import os
import re
import itertools
from typing import List, Dict, Tuple, Optional

from collections import OrderedDict

import hydra
import numpy as np
import openai
import backoff
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif


from agent.agents import Agent

# ==
# 

PRINT_LLM_DEBUG = False

OAI_CLIENT = openai.OpenAI()


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


def get_prompt_method(method: str):
    if method == "default":
        return "Directly output the answer, ensure only one answer is included."
    elif method == "react":
        return (
            "First briefly reason and think about the information collected. "
            "Then, output the answer. Ensure only one answer is included."
        )
    elif method == "reflexion":
        return (
            "Please first reflect on the collected information and analyze any potential issues with your reasoning, "
            "then output the final answer. Ensure only one answer is included."
        )
    elif method == "cot":
        return (
            "Please provide a detailed chain-of-thought explaining your reasoning, "
            "and then output the final answer. Ensure only one answer is included."
        )
    else:
        raise ValueError(f"Unknown prompt method: {method}")
    
# ==
#

class HypothesisElimAgent(Agent):
    """
    This agent takes random actions. It infers the item-attribute set at each
    time-step and stores them in a list. It can do independence testing on the
    stored item-attribute sets to provide additional context for the prompt
    during the Q&A phase.
    """
    def __init__(self, horizon, filter_actions, model, prompt_type, temperature, system_msg_path, 
                 add_elim_hypothesis, hyp_space_update):
        super().__init__(horizon, filter_actions)
        self.model = model
        self.prompt_type = prompt_type
        self.temperature = temperature
        # self.do_indep_analysis = do_indep_analysis

        self.add_elim_hypothesis = add_elim_hypothesis
        self.hyp_space_update = hyp_space_update

        # Get system message
        if system_msg_path.startswith('/'):
            sys_msg_path = system_msg_path
        else:
            sys_msg_path = os.path.join(hydra.utils.get_original_cwd(), system_msg_path)

        with open(sys_msg_path, 'r') as f:
            sys_message = f.read()
        self.system_message = sys_message.replace('#HORIZON#', str(horizon))

        # Hypothesis management
        self.hypothesis_space = []
        self.elim_hypothesis = set([])

        self._rng = np.random.RandomState()
    
    def create_history_obs(self, current_obs: str = None):
        """
        Create a history of actions and observations in a formatted string
        """
        formatted_lines = []
        for i, (action, observation) in enumerate(zip(self.acts_queue, self.obs_queue), 1):
            formatted_lines.append(observation)
            formatted_lines.append(f"> {action}")

        if len(self.obs_queue) > len(self.acts_queue):
            formatted_lines.append(self.obs_queue[-1])
        
        if current_obs:
            formatted_lines.append(current_obs)

        result_string = "\n".join(formatted_lines)
        return result_string

    def get_hypothesis_query(self, prompt: str) -> Tuple[openai.ChatCompletion, List[str], bool, bool]:
        # query LM
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt, 
                                           temperature=self.temperature)
            self.total_cost += cost

            response_msg = response.choices[0].message.content
            api_error = False
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as e:
            print(f'Error: {e}')
            response_msg = ""
            api_error = True

        # parse list
        match = re.search(r'>\s*(\[[^\]]*\])\s*$', response_msg)
        if match:
            list_str = match.group(1)  # Extract the matched list as a string
            try:
                hypothesis_list = ast.literal_eval(list_str)  # Safely convert to a Python list
                parse_error = False
            except (SyntaxError, ValueError):
                hypothesis_list = []
                parse_error = True
        else:
            hypothesis_list = []
            parse_error = True
        
        return response, hypothesis_list, api_error, parse_error
    
    def get_action_query(self, prompt: str) -> Tuple[openai.ChatCompletion, List[str], bool, bool]:
        # query LM
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt, 
                                           temperature=self.temperature)
            self.total_cost += cost

            response_msg = response.choices[0].message.content
            api_error = False
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as e:
            print(f'Error: {e}')
            response_msg = ""
            api_error = True

        # parse list
        match = re.search(r"> (.*?)(?:\.|$)", response_msg)
        if match:
            action_str = match.group(1)  # Extract the matched list as a string
            parse_error = False
        else:
            action_str = ""
            parse_error = True
        
        return response, action_str, api_error, parse_error

    
    def act(self, obs, game_state):
        history_obs = self.create_history_obs(obs)

        # Step 1: generate or eliminate hypothesis
        if len(self.hypothesis_space) == 0:
            h_prompt = "You have seen the following observations so far:\n\n"
            h_prompt += history_obs
            h_prompt += "\n\n---\n\n"

            h_prompt += (
                "Come up with as many hypothesis as possible about how the environment works. "
                "The hypothesis can contradict each other, the point is to then act to figure out "
                "which hypothesis is correct. "
            )

            if self.add_elim_hypothesis:
                h_prompt += "\n\nYou have so far eliminated the following hypothesis (do not use them): \n"
                h_prompt += "\n".join([f"- {h}" for h in self.elim_hypothesis])
                h_prompt += "\n\n---\n\n"

            h_prompt += "\n\n" + get_prompt_method(self.prompt_type)
            h_prompt += (
                "\nReturn a list of hypothesis where each hypothesis is a string sentence. "
                "The list needs to have exactly the format \'> [\"hypothesis 1\", ...]\'. "
            )
        
        else:
            h_prompt = "You are currently entertaining the following list of hypothesis: \n"
            h_prompt += "\n".join([f"- {h}" for h in self.hypothesis_space])
            h_prompt += "\n\n---\n\n"

            if self.add_elim_hypothesis:
                h_prompt += "You have so far eliminated the following hypothesis: \n"
                h_prompt += "\n".join([f"- {h}" for h in self.elim_hypothesis])
                h_prompt += "\n\n---\n\n"
            
            h_prompt += "You have seen the following observations so far:\n\n"
            h_prompt += history_obs 
            h_prompt += "\n\n---\n\n"

            if self.hyp_space_update == "elim_only":
                h_prompt += (
                    "Based on the observations so far, eliminate all hypothesis that are not consistent "
                    "with the observations. Return an updated list containing only hypothesis that are "
                    "consistent with the observations. Each hypothesis should be a string sentence. "
                    "output the list of hypothesis exactly in the format \'> [\"hypothesis 1\", ...]\'."
                )
            elif self.hyp_space_update == "elim_add":
                h_prompt += (
                    "Based on the observations, first eliminate any currently entertained hypothesis that "
                    "are inconsistent with the observations. Optionally, you may add new hypothesis which can "
                    "contradict existing ones but must be consistent with the observations so far. "
                )
            elif self.hyp_space_update == "update_modify":
                h_prompt += (
                    "Based on the observations, update the currently entertained list of hypothesis so "
                    "they are consistent with the observations. You can add, remove or modify the hypothesis. "
                )
            else:
                raise ValueError(f"Unknown hypothesis update method: {self.hyp_space_update}")

            h_prompt += "\n\n" + get_prompt_method(self.prompt_type)
            h_prompt += (
                "\nReturn a new list of hypothesis where each hypothesis is a string sentence. "
                "The list needs to have exactly the format \'> [\"hypothesis 1\", ...]\'. "
            )

        h_response, hypothesis_list, h_api_error, h_parse_error = \
            self.get_hypothesis_query(h_prompt)
        
        elim_hyps = set(self.hypothesis_space) - set(hypothesis_list)
        self.elim_hypothesis.update(elim_hyps)
        self.hypothesis_space = copy.deepcopy(hypothesis_list)        

        # Step 2: generate action based on current hypothesis
        a_prompt = "You have seen the following observations so far:\n\n"
        a_prompt += history_obs
        a_prompt += "\n\n---\n\n"

        a_prompt += "You are currently entertaining the following list of hypothesis: \n"
        a_prompt += "\n".join([f"- {h}" for h in hypothesis_list])  # TODO maybe fix
        a_prompt += "\n\n---\n\n"

        a_prompt += (
            "Given the observations so far, and the list of hypothesis (hypothesis space), take an action "
            "which you expect will lead to an outcome that maximially reduces the hypothesis space. "
            # TODO mention info gain?
        )
        a_prompt += "\n\n" + get_prompt_method(self.prompt_type)
        a_prompt += (
            "\nOutput the action in the format \'> action\'. Ensure only one action is included."
        )

        a_response, action, a_api_error, a_parse_error = \
            self.get_action_query(a_prompt)

        # Update internal states and log 
        self.obs_queue.append(obs)
        self.acts_queue.append(action)

        if False:  # TODO delete
            print(h_prompt)
            print('\n --- \n')
            print(h_response.choices[0].message.content)
            print('\n --- \n')
            print('hypothesis:', hypothesis_list)
            print()
            print('eliminated:', self.elim_hypothesis)
            print('\n --- \n')

        act_info = {
            "model": self.model,
            "system_message": self.system_message,
            "history_obs": history_obs,
            "hyp_prompt": h_prompt,
            "hyp_response": h_response.choices[0].message.content,
            "hyp_usage": h_response.usage,
            "hyp_space": copy.deepcopy(hypothesis_list), 
            "hyp_elim": list(self.elim_hypothesis),
            "hyp_api_error": h_api_error,
            "hyp_parse_error": h_parse_error,
            "act_prompt": a_prompt,
            "act_response": a_response.choices[0].message.content,
            "act_usage": a_response.usage,
            "action": action,
            "act_api_error": a_api_error,
            "act_parse_error": a_parse_error,
        }

        return action, act_info
    
    def answer_tf(self, question: str, env: Optional[object] = None):
        history_obs = self.create_history_obs()

        prompt = "You have seen the following observations so far:\n\n"
        prompt += history_obs
        prompt += "\n\n---\n\n"
        
        prompt += "Importantly, you are currently entertaining the following list of hypothesis: \n"
        prompt += "\n".join([f"- {h}" for h in self.hypothesis_space])
        prompt += "\n\n---\n\n"

        prompt += f"\n\nBased on the information above, answer the following question: {question}\n"

        prompt += "Output the answer in the format \'> True/False\'. Ensure only one answer is included."
        prompt += "\n\n" + get_prompt_method(self.prompt_type)
        
        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt)
            self.total_cost += cost

            response_msg = response.choices[0].message.content
            response_usage = response.usage

            match = re.search(r"> (.*?)(?:\.|$)", response_msg)  # TODO move this outside of the api try block?
            if match:
                answer_str = match.group(1).strip()  # Strip any whitespace around the action
                answer_str = answer_str.rstrip('.')  # Remove any trailing period
                parse_error = False
            else:
                answer_str = ""
                parse_error = True

            if answer_str == 'True':
                ans = True
            elif answer_str == 'False':
                ans = False
            else:
                ans = np.random.choice([True, False])

        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as e:
            print(f'Error: {e}')
            ans = "True"
            api_error = True


        # TODO delete below
        #print(prompt)
        #print(response_msg)
            

        ans_info = {
            "model": self.model,
            "system_message": self.system_message,
            "prompt": prompt,
            "response_message": response_msg,
            "history_obs": history_obs,
            "usage": response_usage,
            "api_error": api_error,
            "parse_error": parse_error,
        } 
    
        return ans, ans_info
    
