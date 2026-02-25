import ast
import copy
import os
import re
import itertools
from typing import List, Dict, Tuple, Optional
import textwrap

from collections import OrderedDict

import hydra
import numpy as np
import openai
import backoff
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

import lm_api
from agent.agents import Agent, RULE_INFERENCE_QUESTION, RULE_TYPE_QUESTION

# ==
# 

PRINT_LLM_DEBUG = False

"""

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
"""

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


def extract_functions_resilient(text):
    """Parses text to extract Python functions, handling imports."""
    functions = {}

    # Extract all import statements
    import_lines = "\n".join(
        line for line in text.splitlines() if line.strip().startswith(("import ", "from "))
    )
    shared_globals = {}
    exec(import_lines, shared_globals)

    # Split text by `def`
    blocks = re.split(r'(?=^def\s)', text, flags=re.MULTILINE)

    for block in blocks:
        if not block.strip().startswith("def"):
            continue  # Skip non-function code

        try:
            tree = ast.parse(block)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    lines = block.splitlines()
                    start_line = node.lineno - 1
                    end_line = getattr(node, 'end_lineno', None)
                    source = "\n".join(lines[start_line:end_line]) if end_line else block
                    func_str = textwrap.dedent(source)

                    local_namespace = {}
                    exec(source, shared_globals, local_namespace)
                    functions[func_str] = local_namespace[func_name]

        except SyntaxError:
            continue

    return functions


def evaluate_hypothesis(hypothesis_fn, obs_states):
    """Assume each obs_state has the form [obs1, obs2, ..., obsN, target_state]"""
    # Evaluate the hypothesis function
    valid = True
    n_correct = 0
    fn_exception = False
    for obs_state in obs_states:
        try:
            pred = hypothesis_fn(obs_state[:-1])
            valid = np.all(pred == obs_state[-1])
            n_correct += 1
        except Exception as e:
            valid = False
            fn_exception = True

    # extra check to make sure valid is a boolean
    try:
        if valid:
            pass 
    except ValueError:
        valid = False
        fn_exception = True
    
    return valid, n_correct, fn_exception


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

class HypothesisCodeAgent(Agent):
    """
    This agent takes random actions. It infers the item-attribute set at each
    time-step and stores them in a list. It can do independence testing on the
    stored item-attribute sets to provide additional context for the prompt
    during the Q&A phase.
    """
    def __init__(self, horizon, filter_actions, model, prompt_type, 
                 temperature, max_tokens,
                 system_msg_path, hyp_gen_msg_path,
                 infer_state, gen_n_hypothesis, add_elim_hypothesis):
        super().__init__(horizon, filter_actions)
        self.model = model
        self.prompt_type = prompt_type
        self.temperature = temperature

        self.chat_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.2,
            "stop": None,
        }

        self.infer_state = infer_state
        self.gen_n_hypothesis = gen_n_hypothesis
        self.add_elim_hypothesis = add_elim_hypothesis

        # Get system message
        def _parse_path(path):
            if path.startswith('/'):
                return path
            else:
                return os.path.join(hydra.utils.get_original_cwd(), path)
            
        sys_msg_path = _parse_path(system_msg_path)
        with open(sys_msg_path, 'r') as f:
            sys_message = f.read()
        self.system_message = sys_message.replace('#HORIZON#', str(horizon))

        # Get hypothesis message
        hyp_gen_msg_path =  _parse_path(hyp_gen_msg_path)
        with open(hyp_gen_msg_path, 'r') as f:
            self.hypothesis_gen_msg = f.read()

        # Hypothesis management
        self.inferred_state_queue = []
        self.hypothesis_space = {}
        self.elim_hypothesis = {}

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

    def get_state_query(self, prompt: str):
        raise NotImplementedError

    def get_hypothesis_query(self, prompt: str) -> Tuple[openai.ChatCompletion, List[str], bool, bool]:
        # query LM
        try:
            chat_kwargs = copy.deepcopy(self.chat_kwargs)
            chat_kwargs["max_tokens"] = 1024
            response, cost = lm_api.query_llm(self._client, self.model, 
                                              self.system_message, prompt, 
                                              self.chat_kwargs)
            self.total_cost += cost

            response_msg = response.choices[0].message.content
            api_error = False
        except (KeyboardInterrupt, EOFError):
            raise
        except ValueError as e:
            print(f'Error: {e}')
            response, response_msg = None, ""
            api_error = True

        funcs = extract_functions_resilient(response_msg)
        parse_error = len(funcs) == 0

        return response, funcs, api_error, parse_error
    
    def get_action_query(self, prompt: str) -> Tuple[openai.ChatCompletion, List[str], bool, bool]:
        # query LM
        try:
            response, cost = lm_api.query_llm(self._client, self.model, 
                                              self.system_message, prompt, 
                                              self.chat_kwargs)
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
        true_state = game_state['true_state']

        # (Optional) get LM to infer the environmental state
        if self.infer_state:
            raise NotImplementedError 
        else:
            inferred_state = true_state
        self.inferred_state_queue.append(inferred_state)
        num_objects = len(inferred_state) - 1

        # Step 1: generate or eliminate hypothesis
        h_prompt = None
        if len(self.hypothesis_space) == 0:
            while len(self.hypothesis_space) < self.gen_n_hypothesis:
                # Sample until quota is met
                h_prompt = "You have seen the following observations so far:\n\n"
                h_prompt += history_obs
                h_prompt += "\n\n---\n\n"

                n_hyp_left = int((self.gen_n_hypothesis - len(self.hypothesis_space)) * 1.5 + 1)
                hyp_gen_msg = self.hypothesis_gen_msg
                hyp_gen_msg = hyp_gen_msg.replace('#NUM_OBJECTS#', str(num_objects))
                hyp_gen_msg = hyp_gen_msg.replace('#NUM_HYPOTHESES#', f'{str(n_hyp_left)}') 

                h_prompt += hyp_gen_msg

                if self.add_elim_hypothesis and len(self.elim_hypothesis) > 0:
                    h_prompt += "\n\nDo not generate the hypothesis you have already eliminated: \n"
                    h_prompt += "\n".join([f"{k}" for k in self.elim_hypothesis])
                    h_prompt += "\n\n---\n\n"
                
                if len(self.hypothesis_space) > 0:
                    h_prompt += "\n\nDo not generate hypothesis you have already generated: \n"
                    h_prompt += "\n".join([f"{k}" for k in self.hypothesis_space])
                    h_prompt += "\n\n---\n\n"

                h_prompt += "\n\n" + get_prompt_method(self.prompt_type)
                h_prompt += (
                    "\nReturn a set of hypothesis as python functions."
                )
            
                h_response, hypothesis_funcs, h_api_error, h_parse_error = \
                    self.get_hypothesis_query(h_prompt)

                self.hypothesis_space.update(hypothesis_funcs)
        
        # Eliminate hypothesis based on observation history
        for k in list(self.hypothesis_space.keys()):
            h_valid, n_correct, had_exception = \
                evaluate_hypothesis(self.hypothesis_space[k], self.inferred_state_queue)
            if not h_valid:
                self.elim_hypothesis[k] = self.hypothesis_space[k]
                del self.hypothesis_space[k]

        # Step 2: generate action based on current hypothesis
        a_prompt = "You are currently entertaining the following list of hypothesis: \n"
        a_prompt += "\n".join([f"{k}" for k in self.hypothesis_space])  # TODO maybe fix
        a_prompt += "\n\n---\n\n"

        a_prompt += "You have seen the following observations so far:\n\n"
        a_prompt += history_obs
        a_prompt += "\n\n---\n\n"

        a_prompt += (
            "Given the observations so far, and the list of hypothesis (hypothesis space), take an action "
            "which will disprove the existing hypothesis."
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
            print(self.inferred_state_queue)
            print('\n --- \n')
            print("\n".join([f"{k}" for k in self.hypothesis_space]))
            print('\n --- \n')
            print('action:', action)
            print('\n --- \n')

        act_info = {
            "model": self.model,
            "system_message": self.system_message,
            "history_obs": history_obs,
            "hyp_space": copy.deepcopy(list(self.hypothesis_space.keys())), 
            "hyp_elim": copy.deepcopy(list(self.elim_hypothesis.keys())),
            "act_prompt": a_prompt,
            "act_response": a_response.choices[0].message.content,
            "act_usage": a_response.usage,
            "action": action,
            "act_api_error": a_api_error,
            "act_parse_error": a_parse_error,
        }

        if h_prompt is not None:
            act_info.update({
                "hyp_prompt": h_prompt,
                "hyp_response": h_response.choices[0].message.content,
                "hyp_usage": h_response.usage,
                "hyp_api_error": h_api_error,
                "hyp_parse_error": h_parse_error,
            })

        return action, act_info
    
    def answer_tf(self, question: str, env: Optional[object] = None):
        history_obs = self.create_history_obs()

        prompt = "You have seen the following observations so far:\n\n"
        prompt += history_obs
        prompt += "\n\n---\n\n"

        if self.add_elim_hypothesis and len(self.elim_hypothesis) > 0:
            prompt += "You have disproven the following hypothesis: \n"
            prompt += "\n".join([f"{k}" for k in self.elim_hypothesis])
            prompt += "\n\n---\n\n"
        
        prompt += "You have not yet disproven the following hypothesis: \n"
        prompt += "\n".join([f"{k}" for k in self.hypothesis_space])
        prompt += "\n\n---\n\n"

        prompt += f"\n\nBased on the information above, answer the following question: {question}\n"

        prompt += "Output the answer in the format \'> True/False\'. Ensure only one answer is included."
        prompt += "\n\n" + get_prompt_method(self.prompt_type)
        
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

    def answer_rule_inference(self, env: Optional[object] = None):
        history_obs = self.create_history_obs()
        prompt = "You have seen the following observations so far:\n\n"
        prompt += history_obs
        prompt += f"\n\n{RULE_INFERENCE_QUESTION}\n"
        prompt += "\nProvide your description."

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
        prompt = "You have seen the following observations so far:\n\n"
        prompt += history_obs
        prompt += "\n\n---\n\n"
        prompt += "Your answers about which objects are blickets:\n"
        for obj_name, ans in blicket_answers.items():
            prompt += f"- {obj_name}: {'Yes' if ans else 'No'}\n"
        prompt += "\n\nYour rule inference:\n"
        prompt += rule_inference_response
        prompt += f"\n\n{RULE_TYPE_QUESTION}\n"

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
            match = re.search(r"> (.*?)(?:\.|$)", response_msg or "")
            answer_str = match.group(1).strip() if match else ""
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
    
