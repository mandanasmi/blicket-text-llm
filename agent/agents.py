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

import lm_api
OAI_CLIENT = lm_api.get_client_from_env()


ENV_DESCRIPTION_SHARED = '''You are an agent playing the Nexiom game, a text-based adventure game where you are in a room with different objects and a machine. A subset of the objects are 'nexioms', which turn on the light on the machine following some rule.
Your goal is to explore the relationship between the objects and the machines to determine which objects are nexioms, and the rule for turning on the machine.

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
- The game has two rounds: a comprehension round (to help you learn the interface) and a main experiment round with a new machine and new objects. Details about the main round will be shown after you finish the comprehension round.
- Each round has an exploration phase and a Q&A phase. Once you enter Q&A, you can no longer explore the objects or test the machine.
- You won't be able to start the Q&A phase until you explore and interact with the objects and the machine at least once.

You can also exit the task early if you think you understand the relationship between the objects and the machine.
After each round is done, you will be asked which objects are nexioms, and the rule for turning on the machine.
'''

COMPREHENSION_INSTRUCTIONS = '''Comprehension Phase: This phase helps you learn the interface.

Here are the instructions for the comprehension phase:

- You will see 3 objects that may or may not be Nexioms. To test if they are Nexioms, place them on the Nexiom machine.
- You can select one or more objects, then test the machine.
- If the Nexiom machine switches on, it means that at least one of the objects you put on the machine is a Nexiom. It could be just one of them, some of them, or all of them.
- A record of your tests and outcomes will be kept for you.
- You can use up to #COMP_HORIZON# test actions (actions that reveal machine feedback) in the comprehension phase. You may take as many non-test actions as needed.
'''

MAIN_INSTRUCTIONS = '''Main Phase:
- You will see 4 new objects and a different Nexiom machine.
- Please note that the rules may be completely different from the comprehension round: which object(s) count as Nexioms can change entirely, and the machine may behave differently as well!
- You must complete this one game to finish the experiment.
- You can use up to #MAIN_HORIZON# test actions (actions that reveal machine feedback) in the main phase. You may take as many non-test actions as needed.
'''

# Legacy alias: some older agents still import ENV_DESCRIPTION expecting a
# single-string environment description.
ENV_DESCRIPTION = ENV_DESCRIPTION_SHARED

RESPONSE_INSTRUCTION = '''Reply concisely and exactly with the requested format.'''

RULE_INFERENCE_QUESTION = "Describe how you think the objects turn on the machine."

RULE_TYPE_QUESTION = (
    "Based on the action history and your answers, what type of rule do you think governs this? "
    "Conjunctive rule: the machine switches on when ALL of the nexioms are present on the machine. "
    "Disjunctive rule: the machine switches on when ANY of the nexioms are present on the machine. "
)

DEFAULT_SYSTEM_MESSAGE = ENV_DESCRIPTION + '\n\n' + '''You will be prompted at each turn to choose actions.''' + \
                         '\n\n' + RESPONSE_INSTRUCTION


MAIN_TRANSITION_TEMPLATE_SINGLE = (
    "The true Nexiom in the practice round was Object {names}. "
    "This is because only Object {names} can turn on the Nexiom machine.\n\n"
)

MAIN_TRANSITION_TEMPLATE_MULTI = (
    "The true Nexioms in the practice round were Objects {names}. "
    "This is because only those objects can turn on the Nexiom machine.\n\n"
)


def format_main_transition_message(nexiom_names) -> str:
    names = list(nexiom_names)
    if len(names) == 1:
        return MAIN_TRANSITION_TEMPLATE_SINGLE.format(names=names[0])
    if len(names) == 2:
        joined = f"{names[0]} and {names[1]}"
    else:
        joined = ", ".join(names[:-1]) + f", and {names[-1]}"
    return MAIN_TRANSITION_TEMPLATE_MULTI.format(names=joined)



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
    """Extract a valid nexiom command from model output."""
    if not text or not isinstance(text, str):
        return None

    # 1) Preferred format: "> command"
    match = re.search(r"(?:^|\n)\s*>\s*(.+?)(?:\n|$)", text)
    if match:
        action = match.group(1).strip().rstrip(".")
        if action:
            return action

    # 2) Fallback: first line that looks like a valid command.
    cmd_pattern = re.compile(
        r"^(look|exit|test(?:\s+the\s+machine)?|"
        r"put\s+.+\s+on\s+(?:the\s+)?(?:machine|floor)|"
        r"take\s+.+\s+off(?:\s+of)?(?:\s+the)?\s+machine)$",
        re.IGNORECASE,
    )
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("`").strip().rstrip(".")
        if line.startswith(">"):
            line = line[1:].strip()
        if cmd_pattern.match(line):
            return line

    return None


def extract_rule_type(text):
    """Extract conjunctive/disjunctive from free-form model text."""
    if not text or not isinstance(text, str):
        return "unknown"

    # Prefer explicit single-token/line matches first.
    line_match = re.search(r"^\s*>?\s*(conjunctive|disjunctive)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if line_match:
        return line_match.group(1).lower()

    # Fallback: first mention in free-form response.
    token_match = re.search(r"\b(conjunctive|disjunctive)\b", text, re.IGNORECASE)
    if token_match:
        return token_match.group(1).lower()

    return "unknown"


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

    def answer_rule_inference(self, env: Optional[object] = None):
        """Describe how objects turn on the machine. Returns (response_text, ans_info)."""
        raise NotImplementedError

    def answer_rule_type(self, blicket_answers: dict, rule_inference_response: str, env: Optional[object] = None):
        """Select conjunctive or disjunctive. Returns (rule_type_str, ans_info)."""
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
        structured_history = self.create_action_outcome_history(current_obs=current_obs)
        if structured_history:
            result_string += "\n\n" + structured_history
        return result_string

    def _is_test_action(self, action: str) -> bool:
        if not action:
            return False
        return re.fullmatch(r"test(\s+(?:the\s+)?machine)?", action.strip().lower()) is not None

    def _extract_machine_status(self, feedback: str):
        if not feedback:
            return None
        match = re.search(r"The light on the machine is [^.]+\.", feedback)
        if match:
            return match.group(0)
        return None

    def create_action_outcome_history(self, current_obs: str = None):
        """
        Build explicit action->feedback history, including machine status after test actions.
        """
        if not self.acts_queue:
            return ""

        observations = list(self.obs_queue)
        if current_obs:
            observations.append(current_obs)

        lines = ["Structured action history:"]
        for idx, action in enumerate(self.acts_queue, start=1):
            lines.append(f"{idx}. > {action}")

            # Observation after action i is at observations[i] (if available).
            feedback_after_action = observations[idx] if idx < len(observations) else None
            if feedback_after_action:
                lines.append(f"   feedback: {feedback_after_action}")
                if self._is_test_action(action):
                    machine_status = self._extract_machine_status(feedback_after_action)
                    if machine_status:
                        lines.append(f"   machine_after_test: {machine_status}")

        return "\n".join(lines)
    
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

        r = self._rng.rand()
        if r < 1 / 3:
            action = f'put {chosen_obj} on the machine'
        elif r < 2 / 3:
            action = f'take {chosen_obj} off the machine'
        else:
            action = 'test'

        return action, {}
        
    def answer_tf(self, question, env: Optional[object] = None):
        ans = self._rng.choice([True, False])
        return ans, {}

    def answer_rule_inference(self, env: Optional[object] = None):
        return "Random guess", {}

    def answer_rule_type(self, blicket_answers: dict, rule_inference_response: str, env: Optional[object] = None):
        ans = self._rng.choice(["conjunctive", "disjunctive"])
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
        self._client = lm_api.get_client(model)
        self.chat_kwargs = {
            "temperature": temperature,
            "max_tokens": 1024,
            "top_p": 0.2,
            "stop": None,
        }

    def select_action(self, obs, game_state):
        prompt = "Interact with the environment to find the nexioms and discover the rule.\n"
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
        reasoning_msg = None
        response_usage = None
        api_error = False
        parse_error = False
        try:
            # Retry once when content is empty (observed with some local models).
            max_empty_content_retries = 1
            for attempt in range(max_empty_content_retries + 1):
                response, cost = lm_api.query_llm(self._client, self.model, self.system_message,
                                                 prompt, self.chat_kwargs)
                self.total_cost += cost
                response_usage = response.usage

                msg = response.choices[0].message
                response_msg = msg.content or ""
                reasoning_msg = getattr(msg, "reasoning_content", None)
                if response_msg or attempt == max_empty_content_retries:
                    break

            action = extract_action(response_msg)
            if action is None and reasoning_msg:
                action = extract_action(reasoning_msg)
            if action is None:
                action = "look"
                parse_error = True

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
            "reasoning_message": reasoning_msg,
            "usage": response_usage,
            "api_error": api_error,
            "parse_error": parse_error,
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
            response, cost = lm_api.query_llm(self._client, self.model, self.system_message,
                                             prompt, self.chat_kwargs)
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

    def answer_rule_inference(self, env: Optional[object] = None):
        history_obs = self.create_history_obs()
        prompt = history_obs
        prompt += f"\n\n{RULE_INFERENCE_QUESTION}\n"
        prompt += "\nProvide your description."

        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = lm_api.query_llm(self._client, self.model, self.system_message,
                                             prompt, self.chat_kwargs)
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
        history_obs = self.create_history_obs()
        prompt = history_obs
        prompt += "\n\nYour answers about which objects are nexioms:\n"
        for obj_name, ans in blicket_answers.items():
            prompt += f"- {obj_name}: {'Yes' if ans else 'No'}\n"
        prompt += "\n\nYour rule inference:\n"
        prompt += rule_inference_response
        prompt += f"\n\n{RULE_TYPE_QUESTION}\n"
        prompt += "Reply with exactly one word: conjunctive or disjunctive."

        response_msg = None
        response_usage = None
        api_error = False
        try:
            response, cost = lm_api.query_llm(self._client, self.model, self.system_message,
                                             prompt, self.chat_kwargs)
            self.total_cost += cost
            response_msg = response.choices[0].message.content
            response_usage = response.usage
            rule_type = extract_rule_type(response_msg)
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


