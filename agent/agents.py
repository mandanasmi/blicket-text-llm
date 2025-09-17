import copy
from collections import OrderedDict
from typing import Optional

import numpy as np
import openai

import os
import openai
import backoff
import re


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
  exit:                exit the game

Tips:
- All objects can be either on the machine or on the floor.
- You should think about how to efficiently explore the relationship between the objects and the machine.

You have #HORIZON# steps to complete the task. You can also exit the task early if you think you understand the relationship between the objects and the machine.
After the task is done, you will be asked which objects are blickets, and the rule for turning on the machine.
'''

RESPONSE_INSTRUCTION = '''Reply concisely and exactly with the requested format.'''
DEFAULT_SYSTEM_MESSAGE = ENV_DESCRIPTION + '\n\n' + '''You will be prompted at each turn to choose actions.''' + \
                         '\n\n' + RESPONSE_INSTRUCTION



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


class Agent:
    def __init__(self, horizon, filter_actions):
        self.horizon = horizon
        self.filter_actions = filter_actions

        # global attributes
        self.max_score = 0
        self.total_cost = 0

        # per episode attributes
        self.obs_queue = []
        self.acts_queue = []
        self.tried_actions = {}
        self.archive = OrderedDict()        

    def init_episode(self):
        """Call at start of episode to initialize observation"""
        self.obs_queue = []
        self.acts_queue = []
        self.tried_actions = {}
        self.archive = OrderedDict()

        #self.obs_queue.append(obs)
        # self.add_or_update(obs, infos)
        # self.recipe = infos['extra.recipe']
        #self.init_obs_queue = [copy.deepcopy(obs)]  # TODO what is this from IGE?
        return

    def select_action(self, obs, game_state):
        raise NotImplementedError

    def act(self, obs, game_state):
        action, info = self.select_action(obs, game_state)

        self.obs_queue.append(obs)
        self.acts_queue.append(action)

        # 
        #state_id = (location, inventory, points)
        #if self.filter_actions:
        #    if state_id in self.tried_actions and action not in self.tried_actions[state_id]:
        #        self.tried_actions[state_id].append(action)
        #    else:
        #        self.tried_actions[state_id] = [action]
        #return action

        return action, info
    
    def answer_tf(self, question: str, env: Optional[object] = None):
        raise NotImplementedError
 
    def add_obs_to_queue(self, obs, game_state):
        self.obs_queue.append(obs)
    
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
    
    def choose_new_state(self, input_archive=None):
        return next(iter(self.archive.values()))

    #def reset_state(self):
    #    # by default select the first state
    #    self.obs_queue = []
    #    self.acts_queue = []
    #
    #    chosen_state, state_id = self.choose_new_state()
    #    self.acts_queue, self.obs_queue = copy.deepcopy(chosen_state[2])
    #    return chosen_state


class RandomAgent(Agent):
    def __init__(self, horizon, filter_actions):
        super().__init__(horizon, filter_actions)
        self._rng = np.random.RandomState()

    def select_action(self, obs, game_state):
        obj_names = game_state['object_names']
        chosen_obj = self._rng.choice(obj_names)

        if self._rng.rand() < 0.5:
            action = f'put {chosen_obj} on the machine'
        else:
            action = f'take {chosen_obj} off the machine'

        return action, {}
        
    def answer_tf(self, question, env: Optional[object] = None):
        ans = self._rng.choice([True, False])
        return ans, {}
    

# ==
#

class NaiveLLM(Agent):
    def __init__(self, horizon, filter_actions, model, react, temperature):
        super().__init__(horizon, filter_actions)
        self.model = model
        self.react = react
        self.temperature = temperature
        self.system_message = DEFAULT_SYSTEM_MESSAGE.replace('#HORIZON#', str(horizon))

    def select_action(self, obs, game_state):
        prompt = "Interact with the environment to find the blickets and discover the rule.\n"
        prompt += self.create_history_obs(obs)

        if self.filter_actions:
            raise NotImplementedError  # TODO: implement this later based on IGE but without priviledged information
        
        if self.react:
            prompt += "\nFirst briefly reason and think about your plan to solve the task. "\
                "Then, output the command in the format \'> command\'. "\
                "Ensure only one command is included."
        else:
            prompt += "\nDirectly output the command in the format \'> command\'. "\
                "Ensure only one command is included."

        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = query_llm_api(self.model, self.system_message, prompt, 
                                           temperature=self.temperature)
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

        act_info = {
            "model": self.model,
            "system_message": self.system_message,
            "prompt": prompt,
            "response_message": response_msg,
            "usage": response_usage,
            "api_error": api_error,
        }

        return action, act_info
    
    def answer_tf(self, question: str, env: Optional[object] = None):
        prompt = self.create_history_obs()

        prompt += f"\n\nBased on the information you have gathered, answer the following question: {question}\n"

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
            ans = "True"
            api_error = True

        ans_info = {
            "model": self.model,
            "system_message": self.system_message,
            "prompt": prompt,
            "response_message": response_msg,
            "usage": response_usage,
            "api_error": api_error,
        } 
    
        return ans, ans_info
    

