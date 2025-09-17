import itertools
import os
import re
from typing import Dict, List, Optional

import hydra
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

import lm_api
from agent.agents import Agent


PRINT_LLM_DEBUG = False


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

class PromptsAgent(Agent):
    """
    This agent takes random actions. It infers the item-attribute set at each
    time-step and stores them in a list. It can do independence testing on the
    stored item-attribute sets to provide additional context for the prompt
    during the Q&A phase.
    """
    def __init__(self, horizon, filter_actions, model, react, temperature, system_msg_path):
        super().__init__(horizon, filter_actions)
        self.model = model
        self.react = react
        self.temperature = temperature
        # self.do_indep_analysis = do_indep_analysis

        self.chat_kwargs = {
            "temperature": temperature,
            "max_tokens": 1024,
            "top_p": 0.2,
            "stop": None,
        }

        # Get system message
        if system_msg_path.startswith('/'):
            sys_msg_path = system_msg_path
        else:
            sys_msg_path = os.path.join(hydra.utils.get_original_cwd(), system_msg_path)

        with open(sys_msg_path, 'r') as f:
            sys_message = f.read()
        self.system_message = sys_message.replace('#HORIZON#', str(horizon))

        self.items_queue = []

        self._rng = np.random.RandomState()
        self._client = lm_api.get_client(model)
    
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

    def act(self, obs, game_state):
        prompt = "Interact with the environment to find the blickets and discover the rule.\n"
        history_obs = self.create_history_obs(obs)
        prompt += history_obs

        if self.filter_actions:
            raise NotImplementedError  # TODO: implement this later based on IGE but without priviledged information
        
        if self.react:
            prompt += "\n\nFirst briefly reason and think about your plan to solve the task. "\
                "Then, output the command in the format \'> command\'. "\
                "Ensure only one command is included."
        else:
            prompt += "\n\nDirectly output the command in the format \'> command\'. "\
                "Ensure only one command is included."

        response = None
        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = lm_api.query_llm(self._client, self.model, 
                                              self.system_message, prompt, 
                                              self.chat_kwargs)
            self.total_cost += cost

            response_msg = response.choices[0].message.content
            action = extract_action(response_msg)

            response_usage = response.usage

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
        if response is not None and hasattr(response.choices[0].message, "reasoning_content"):
            act_info["reasoning_content"] = response.choices[0].message.reasoning_content
        
        return action, act_info
    
    def answer_tf(self, question: str, env: Optional[object] = None):
        history_obs = self.create_history_obs()
        prompt = history_obs

        prompt += f"\n\nBased on the information you have gathered, answer the following question: {question}\n"

        if self.react:
            prompt += "\nFirst briefly reason and think about the information collected. "\
                "Then, output the answer in the format \'> True/False\'. "\
                "Ensure only one answer is included."
        else:
            prompt += "\nDirectly output the answer in the format \'> True/False\'. "\
                "Ensure only one answer is included."
        
        response = None
        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = lm_api.query_llm(self._client, self.model, 
                                              self.system_message, prompt, 
                                              self.chat_kwargs)
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
            ans = "True"
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
        if response is not None and hasattr(response.choices[0].message, "reasoning_content"):
            ans_info["reasoning_content"] = response.choices[0].message.reasoning_content
    
        return ans, ans_info
    

class OReasonPromptsAgent(PromptsAgent):
    """
    This agent is identical to PromptsAgent but disables reasoning in prompts.
    """
    def __init__(self, horizon, filter_actions, model, react, reasoning_effort, system_msg_path):
        super().__init__(horizon, filter_actions, model, react=react, 
                         temperature=0, system_msg_path=system_msg_path)  # temperature is dummy
        self.chat_kwargs = {
            "reasoning_effort": reasoning_effort,
        }
        