import re
from typing import Optional

import numpy as np

import lm_api
from agent.agents import (
    Agent,
    COMPREHENSION_INSTRUCTIONS,
    ENV_DESCRIPTION_SHARED,
    MAIN_INSTRUCTIONS,
    PRINT_LLM_DEBUG,
    RESPONSE_INSTRUCTION,
    RULE_INFERENCE_QUESTION,
    RULE_TYPE_QUESTION,
    extract_rule_type,
    format_main_transition_message,
)
from agent.prompt_variants import (
    PromptVariant,
    build_comprehension_instructions,
    build_rule_type_question,
    coerce_variant,
)


TURN_PROMPT_LINE = "You will be prompted at each turn to choose actions."


def extract_nexiom_action(text: str):
    if not text or not isinstance(text, str):
        return None

    preferred = re.search(r"(?:^|\n)\s*>\s*(.+?)(?:\n|$)", text)
    if preferred:
        candidate = preferred.group(1).strip().rstrip(".")
        if candidate:
            return candidate

    cmd_pattern = re.compile(
        r"^(look|"
        r"test(?:\s+the\s+machine)?|"
        r"put\s+.+\s+on\s+(?:the\s+)?(?:machine|floor)|"
        r"take\s+.+\s+off(?:\s+of)?(?:\s+the)?\s+machine|"
        r"ready(?:\s+for\s+q(?:&|and)?a)?)$",
        re.IGNORECASE,
    )
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("`").strip().rstrip(".")
        if line.startswith(">"):
            line = line[1:].strip()
        line = line.strip("'\"")
        if cmd_pattern.fullmatch(line):
            return line

    return None


def extract_yes_no(text: str):
    if not text or not isinstance(text, str):
        return None
    match = re.search(r"\b(yes|no|true|false)\b", text, re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).lower()
    if token in {"yes", "true"}:
        return True
    return False


class NexiomLLM(Agent):
    # Keep comprehension scaffolding in this file for future use, but disable it for now.
    USE_COMPREHENSION_PHASE = False
    USE_BOOTSTRAP_POLICY = False

    def __init__(
        self,
        horizon,
        filter_actions,
        model,
        react,
        temperature,
        comprehension_horizon=4,
        prompt_variant=None,
    ):
        super().__init__(horizon, filter_actions)
        self.model = model
        self.react = react
        self.temperature = temperature
        self.comprehension_horizon = comprehension_horizon
        self.prompt_variant: PromptVariant = coerce_variant(prompt_variant)
        self.phase = "comprehension"

        self._client = lm_api.get_client(model)
        self.chat_kwargs = {
            "temperature": temperature,
            "max_tokens": 1024,
            "top_p": 0.2,
            "stop": None,
        }
        # Force short actionable outputs for local models.
        self.action_chat_kwargs = {
            "temperature": float(temperature),
            "max_tokens": 128,
            "top_p": 0.2,
            "stop": None,
            "timeout": 12,
        }
        self.qa_chat_kwargs = {
            "temperature": float(temperature),
            "max_tokens": 256,
            "top_p": 0.2,
            "stop": None,
        }

        self._system_message_emitted = {}
        self._comprehension_result: Optional[tuple[str, ...]] = None
        self.set_phase("comprehension")

    def _render_comprehension_system_message(self) -> str:
        comp_instr = build_comprehension_instructions(
            self.prompt_variant, self.comprehension_horizon
        )
        return (
            ENV_DESCRIPTION_SHARED
            + "\n"
            + comp_instr
            + "\n"
            + TURN_PROMPT_LINE
            + "\n\n"
            + RESPONSE_INSTRUCTION
        )

    def _render_main_system_message(self) -> str:
        main_instr = MAIN_INSTRUCTIONS.replace("#MAIN_HORIZON#", str(self.horizon))
        transition = ""
        if self._comprehension_result:
            transition = format_main_transition_message(self._comprehension_result)
        return (
            ENV_DESCRIPTION_SHARED
            + "\n"
            + transition
            + main_instr
            + "\n"
            + TURN_PROMPT_LINE
            + "\n\n"
            + RESPONSE_INSTRUCTION
        )

    def set_comprehension_result(self, nexiom_names):
        """Record the correct answer from the practice round so the main-phase system
        message can include the transition preamble revealing it."""
        self._comprehension_result = tuple(nexiom_names) if nexiom_names else None

    def init_episode(self):
        super().init_episode()
        self._system_message_emitted = {}

    def set_phase(self, phase: str):
        self.phase = phase
        if phase == "main":
            self.system_message = self._render_main_system_message()
        else:
            self.system_message = self._render_comprehension_system_message()

    def _system_message_for_logging(self):
        # Log the full system instructions once per phase (instead of each step).
        if not self._system_message_emitted.get(self.phase, False):
            self._system_message_emitted[self.phase] = True
            return self.system_message
        return None

    def _build_action_prompt(self, obs, game_state):
        # Match agents.py style: system message carries detailed instructions,
        # action prompt stays compact and only includes history + output format.
        prompt = "Interact with the environment to find the nexioms and discover the rule.\n"
        tests_remaining = game_state.get("phase_tests_remaining", None)
        if tests_remaining is not None:
            prompt += f"Remaining test actions in this phase: {tests_remaining}\n"

        prompt += self.create_history_obs(obs)
        prompt += (
            "\nDirectly output the command in the format '> command'. "
            "Ensure only one command is included."
        )
        return prompt

    def is_ready_action(self, action: str):
        if not action:
            return False
        return re.fullmatch(
            r"ready(?:\s+for\s+q(?:&|and)?a)?",
            action.strip().lower().rstrip("."),
        ) is not None

    def _with_no_think(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            return ""
        return f"\\no_think\n{prompt}"

    def _strip_extended_thinking(self, text: str, expects_action: bool = False) -> str:
        if not text or not isinstance(text, str):
            return ""
        cleaned = text.strip()

        # If the model includes verbose reasoning, keep only a concise final answer.
        has_reasoning_header = re.search(
            r"(?im)^\s*(thinking process|analysis|reasoning)\s*:?",
            cleaned,
        )
        if has_reasoning_header:
            # Prefer explicit final answer markers when present.
            final_match = re.search(r"(?is)(?:final answer|answer)\s*:\s*(.+)$", cleaned)
            if final_match:
                cleaned = final_match.group(1).strip()
            elif expects_action:
                cmd = extract_nexiom_action(cleaned)
                if cmd:
                    cleaned = f"> {cmd}"
                else:
                    # For action selection, reject verbose reasoning-only output.
                    return ""
            else:
                # Fallback: keep the last meaningful non-empty line.
                lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
                for ln in reversed(lines):
                    if re.match(r"^(thinking process|analysis|reasoning)\s*:?", ln, re.IGNORECASE):
                        continue
                    if re.match(r"^\d+[\.\)]", ln):
                        continue
                    cleaned = ln
                    break

        return cleaned.strip()

    def _query_with_empty_retry(self, prompt: str, chat_kwargs: dict, expects_action: bool = False):
        response_msg = ""
        reasoning_msg = None
        response_usage = None
        api_error = False
        empty_response = False
        parse_hint_prompt = self._with_no_think(prompt)
        raw_response_msg = ""

        for attempt in range(2):
            try:
                response, cost = lm_api.query_llm(
                    self._client, self.model, self.system_message, parse_hint_prompt, chat_kwargs
                )
                self.total_cost += cost
                response_usage = response.usage
                msg = response.choices[0].message
                raw_response_msg = msg.content or ""
                response_msg = raw_response_msg
                reasoning_msg = getattr(msg, "reasoning_content", None)
                response_msg = self._strip_extended_thinking(response_msg, expects_action=expects_action)
                if response_msg and response_msg.strip():
                    return (
                        response_msg.strip(),
                        reasoning_msg,
                        response_usage,
                        api_error,
                        empty_response,
                        parse_hint_prompt,
                        raw_response_msg,
                    )
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception as e:
                print(f"Error: {e}")
                api_error = True
                return "", None, None, api_error, False, parse_hint_prompt, raw_response_msg

            # Retry once with a stricter instruction when content is empty/whitespace.
            if attempt == 0:
                empty_response = True
                if expects_action:
                    parse_hint_prompt = self._with_no_think(
                        prompt
                        + "\n\nIMPORTANT: Output exactly one command and nothing else."
                        + "\nDo not output any thinking process, reasoning, analysis, or explanation."
                        + "\nAllowed: look | test | put <object> on the machine | take <object> off the machine | ready for Q&A"
                    )
                else:
                    parse_hint_prompt = self._with_no_think(
                        prompt
                        + "\n\nIMPORTANT: Reply with one short line only. Do not leave the answer blank."
                        + "\nDo not output any thinking process, reasoning, analysis, or explanation."
                    )

        return "", reasoning_msg, response_usage, api_error, True, parse_hint_prompt, raw_response_msg

    def _fallback_action(self, game_state):
        # Do not inject deterministic exploration policy when parsing fails.
        # Keep fallback minimally invasive so action choice remains model-driven.
        return "look"

    def _canonicalize_action(self, action: str, game_state):
        if not action:
            return None
        action = action.strip().lower().rstrip(".")
        object_names = [str(x).strip().lower() for x in game_state.get("object_names", [])]

        def _resolve_single_object(obj_raw: str):
            obj = obj_raw.strip().lower()
            # Reject multi-object phrases (e.g. "object 1 and object 2").
            if re.search(r"\b(and|or)\b|,|&|/|\\+", obj):
                return None
            if obj in object_names:
                return obj
            if re.fullmatch(r"\d+", obj):
                candidate = f"object {obj}"
                if candidate in object_names:
                    return candidate
            return None

        if re.fullmatch(r"look(\s+around)?", action):
            return "look"
        if re.fullmatch(r"test(\s+(?:the\s+)?machine)?", action):
            return "test"
        if self.is_ready_action(action):
            return "ready for q&a"

        put_match = re.fullmatch(r"put\s+(.+?)\s+on\s+(?:the\s+)?(machine|floor)", action)
        if put_match:
            obj_raw, dst = put_match.group(1).strip(), put_match.group(2)
            resolved = _resolve_single_object(obj_raw)
            if resolved is None:
                return None
            return f"put {resolved} on the {dst}"

        take_match = re.fullmatch(r"take\s+(.+?)\s+off(?:\s+of)?(?:\s+the)?\s+machine", action)
        if take_match:
            obj_raw = take_match.group(1).strip()
            resolved = _resolve_single_object(obj_raw)
            if resolved is None:
                return None
            return f"take {resolved} off the machine"

        return action

    def _has_object_interaction(self) -> bool:
        for a in self.acts_queue:
            s = str(a).strip().lower()
            if s.startswith("put ") or s.startswith("take "):
                return True
        return False

    def _has_test_action(self) -> bool:
        for a in self.acts_queue:
            if re.fullmatch(r"test(\s+(?:the\s+)?machine)?", str(a).strip().lower()):
                return True
        return False

    def _bootstrap_action_if_needed(self, game_state):
        if not self.USE_BOOTSTRAP_POLICY:
            return None

        """Deterministic warm-start:
        1) put first object on machine if no object interaction yet
        2) test once after first placement if no test yet
        """
        object_names = [str(x).strip().lower() for x in game_state.get("object_names", [])]
        if not object_names:
            return None

        state_arr = np.asarray(game_state.get("true_state", [])).reshape(-1)
        on_machine = []
        if state_arr.size >= len(object_names):
            on_machine = [bool(state_arr[i]) for i in range(len(object_names))]

        if not self._has_object_interaction():
            if not on_machine or not any(on_machine):
                return f"put {object_names[0]} on the machine"

        if self._has_object_interaction() and not self._has_test_action():
            if on_machine and any(on_machine):
                return "test"

        return None

    def select_action(self, obs, game_state):
        bootstrap_action = self._bootstrap_action_if_needed(game_state)
        if bootstrap_action is not None:
            act_info = {
                "model": self.model,
                "phase": self.phase,
                "system_message": self._system_message_for_logging(),
                "prompt": "bootstrap-policy",
                "response_message": "BOOTSTRAP_POLICY",
                "raw_response_message": "BOOTSTRAP_POLICY",
                "reasoning_message": None,
                "usage": None,
                "api_error": False,
                "empty_response": False,
                "parse_error": False,
            }
            return bootstrap_action, act_info

        prompt = self._build_action_prompt(obs, game_state)

        response_msg = ""
        reasoning_msg = None
        response_usage = None
        api_error = False
        empty_response = False
        parse_error = False
        response_msg, reasoning_msg, response_usage, api_error, empty_response, prompt, raw_response_msg = (
            self._query_with_empty_retry(prompt, self.action_chat_kwargs, expects_action=True)
        )
        action = extract_nexiom_action(response_msg)
        if action is None and reasoning_msg:
            action = extract_nexiom_action(reasoning_msg)
        action = self._canonicalize_action(action, game_state)

        # One strict retry: force exact command format if parsing failed.
        if action is None:
            retry_prompt = self._with_no_think(
                prompt
                + "\n\nIMPORTANT: Your previous reply did not contain a valid command."
                + "\nReply with EXACTLY one command in this format: > <command>"
                + "\nAllowed commands only: look | test | put <object> on the machine | "
                + "take <object> off the machine | put <object> on the floor | ready for Q&A"
                + "\nDo not include any other text."
            )
            retry_msg, retry_reasoning, retry_usage, retry_api_error, retry_empty, prompt, retry_raw_response_msg = (
                self._query_with_empty_retry(retry_prompt, self.action_chat_kwargs, expects_action=True)
            )
            # Preserve the latest model metadata for logging.
            response_msg = retry_msg
            reasoning_msg = retry_reasoning
            response_usage = retry_usage
            raw_response_msg = retry_raw_response_msg
            api_error = api_error or retry_api_error
            empty_response = empty_response or retry_empty
            action = extract_nexiom_action(response_msg)
            if action is None and reasoning_msg:
                action = extract_nexiom_action(reasoning_msg)
            action = self._canonicalize_action(action, game_state)

        # Second strict retry in the same turn before fallback.
        if action is None:
            retry_prompt_2 = self._with_no_think(
                self.create_history_obs(obs)
                + "\n\nOutput EXACTLY one line in this format:"
                + "\n> <command>"
                + "\nAllowed commands:"
                + "\n- > look"
                + "\n- > test"
                + "\n- > put <object> on the machine"
                + "\n- > put <object> on the floor"
                + "\n- > take <object> off the machine"
                + "\n- > ready for q&a"
                + "\nNo analysis. No explanation. No extra text."
            )
            retry_msg_2, retry_reasoning_2, retry_usage_2, retry_api_error_2, retry_empty_2, prompt, retry_raw_response_msg_2 = (
                self._query_with_empty_retry(retry_prompt_2, self.action_chat_kwargs, expects_action=True)
            )
            response_msg = retry_msg_2
            reasoning_msg = retry_reasoning_2
            response_usage = retry_usage_2
            raw_response_msg = retry_raw_response_msg_2
            api_error = api_error or retry_api_error_2
            empty_response = empty_response or retry_empty_2
            action = extract_nexiom_action(response_msg)
            if action is None and reasoning_msg:
                action = extract_nexiom_action(reasoning_msg)
            action = self._canonicalize_action(action, game_state)

        if action is None:
            action = self._fallback_action(game_state)
            parse_error = True

        if PRINT_LLM_DEBUG:
            print("----- prompt -----")
            print(prompt)
            print("----- response -----")
            print(response_msg)
            print("====================")

        act_info = {
            "model": self.model,
            "phase": self.phase,
            "system_message": self._system_message_for_logging(),
            "prompt": prompt,
            "response_message": response_msg if response_msg else "NO_RESPONSE",
            "raw_response_message": raw_response_msg if raw_response_msg else "NO_RESPONSE",
            "reasoning_message": reasoning_msg,
            "usage": response_usage,
            "api_error": api_error,
            "empty_response": empty_response,
            "parse_error": parse_error,
        }
        return action, act_info

    def answer_tf(self, question: str, env: Optional[object] = None):
        history_obs = self.create_history_obs()
        prompt = history_obs
        prompt += f"\n\n{question}\n"
        prompt += "Reply with exactly one token: Yes or No."

        response_msg = ""
        response_usage = None
        api_error = False
        empty_response = False
        response_msg, _, response_usage, api_error, empty_response, prompt, _ = (
            self._query_with_empty_retry(prompt, self.qa_chat_kwargs, expects_action=False)
        )
        ans = extract_yes_no(response_msg)
        if ans is None:
            ans = np.random.choice([True, False])

        ans_info = {
            "model": self.model,
            "phase": self.phase,
            "system_message": self._system_message_for_logging(),
            "prompt": prompt,
            "response_message": response_msg if response_msg else "NO_RESPONSE",
            "usage": response_usage,
            "api_error": api_error,
            "empty_response": empty_response,
        }
        return bool(ans), ans_info

    def answer_text(self, question: str):
        history_obs = self.create_history_obs()
        prompt = history_obs + f"\n\n{question}\nProvide a concise answer."

        response_msg = ""
        response_usage = None
        api_error = False
        empty_response = False
        response_msg, _, response_usage, api_error, empty_response, prompt, _ = (
            self._query_with_empty_retry(prompt, self.qa_chat_kwargs, expects_action=False)
        )
        if not response_msg:
            response_msg = "NO_RESPONSE"

        ans_info = {
            "model": self.model,
            "phase": self.phase,
            "system_message": self._system_message_for_logging(),
            "prompt": prompt,
            "response_message": response_msg,
            "usage": response_usage,
            "api_error": api_error,
            "empty_response": empty_response,
        }
        return response_msg, ans_info

    def answer_rule_inference(self, env: Optional[object] = None):
        question = RULE_INFERENCE_QUESTION
        return self.answer_text(question)

    def answer_rule_type(self, nexiom_answers: dict, rule_inference_response: str, env: Optional[object] = None):
        history_obs = self.create_history_obs()
        prompt = history_obs
        prompt += "\n\nYour answers about which objects are Nexioms:\n"
        for obj_name, ans in nexiom_answers.items():
            prompt += f"- {obj_name}: {'Yes' if ans else 'No'}\n"
        prompt += "\nYour rule inference:\n"
        prompt += str(rule_inference_response) + "\n\n"
        prompt += build_rule_type_question(self.prompt_variant)

        response_msg = ""
        response_usage = None
        api_error = False
        empty_response = False
        response_msg, _, response_usage, api_error, empty_response, prompt, _ = (
            self._query_with_empty_retry(prompt, self.qa_chat_kwargs, expects_action=False)
        )
        rule_type = extract_rule_type(response_msg)
        if not response_msg:
            response_msg = "NO_RESPONSE"

        ans_info = {
            "model": self.model,
            "phase": self.phase,
            "system_message": self._system_message_for_logging(),
            "prompt": prompt,
            "response_message": response_msg,
            "usage": response_usage,
            "api_error": api_error,
            "empty_response": empty_response,
        }
        return rule_type, ans_info
