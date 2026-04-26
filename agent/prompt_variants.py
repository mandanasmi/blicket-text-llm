"""Prompt variants for the Nexiom task — lets us A/B each of the four
rule-type biases identified in the Nexiom prompt.

Axes (all default to current behavior):
    option_order            "conj_first" (default) | "disj_first"
    comprehension_rule      "default" (default — use YAML's comprehension_env_kwargs.rule)
                            | "disjunctive" | "conjunctive" | "alternate" | "random"
    comprehension_phrasing  "at_least_one" (default) | "neutral"
    reasoning_cue           False (default) | True

Notes
-----
- If you switch `comprehension_rule` to "conjunctive" (or to alternate/random and
  an alternation lands on conjunctive), you will likely want ≥2 comprehension
  blickets for the example to be non-trivial. Override at the YAML level:
      comprehension_env_kwargs.num_blickets=2
- Defaults are chosen so a run without any overrides behaves identically to the
  pre-existing prompts in agent/agents.py.
"""
from dataclasses import dataclass
import random as _random
from typing import Any


@dataclass
class PromptVariant:
    option_order: str = "conj_first"
    comprehension_rule: str = "default"
    comprehension_phrasing: str = "at_least_one"
    reasoning_cue: bool = False

    def tag(self) -> str:
        return (
            f"ord-{self.option_order}"
            f"_comp-{self.comprehension_rule}"
            f"_phr-{self.comprehension_phrasing}"
            f"_cue-{int(bool(self.reasoning_cue))}"
        )

    def to_dict(self) -> dict:
        return {
            "option_order": self.option_order,
            "comprehension_rule": self.comprehension_rule,
            "comprehension_phrasing": self.comprehension_phrasing,
            "reasoning_cue": bool(self.reasoning_cue),
        }


def coerce_variant(pv: Any) -> PromptVariant:
    """Accept None / dict / DictConfig / PromptVariant and return a PromptVariant."""
    if pv is None:
        return PromptVariant()
    if isinstance(pv, PromptVariant):
        return pv
    # DictConfig, OmegaConf, or plain dict all support dict(...)
    d = dict(pv)
    return PromptVariant(
        option_order=str(d.get("option_order", "conj_first")),
        comprehension_rule=str(d.get("comprehension_rule", "default")),
        comprehension_phrasing=str(d.get("comprehension_phrasing", "at_least_one")),
        reasoning_cue=bool(d.get("reasoning_cue", False)),
    )


_COMP_BULLETS_HEAD = (
    "- You will see 3 objects that may or may not be Nexioms. "
    "To test if they are Nexioms, place them on the Nexiom machine.\n"
    "- You can select one or more objects, then test the machine.\n"
)
_COMP_BULLET_AT_LEAST_ONE = (
    "- If the Nexiom machine switches on, it means that at least one of the objects you "
    "put on the machine is a Nexiom. It could be just one of them, some of them, or all of them.\n"
)
_COMP_BULLET_NEUTRAL = (
    "- The Nexiom machine turns on or off depending on which subset of objects is placed on it, "
    "according to a hidden activation rule. The rule could depend on any single nexiom being "
    "present, on all nexioms being present together, or on some other combination — the light "
    "outcome reflects whether the current subset satisfies the rule.\n"
)
_COMP_BULLETS_TAIL = (
    "- A record of your tests and outcomes will be kept for you.\n"
    "- You can use up to #COMP_HORIZON# test actions (actions that reveal machine feedback) "
    "in the comprehension phase. You may take as many non-test actions as needed.\n"
)


def build_comprehension_instructions(variant: PromptVariant, comp_horizon: int) -> str:
    activation = (
        _COMP_BULLET_NEUTRAL
        if variant.comprehension_phrasing == "neutral"
        else _COMP_BULLET_AT_LEAST_ONE
    )
    body = _COMP_BULLETS_HEAD + activation + _COMP_BULLETS_TAIL
    return (
        "Comprehension Phase: This phase helps you learn the interface.\n\n"
        "Here are the instructions for the comprehension phase:\n\n"
        + body
    ).replace("#COMP_HORIZON#", str(comp_horizon))


_CONJ_DEF = (
    "Conjunctive rule: the machine switches on when ALL of the nexioms are present on the machine."
)
_DISJ_DEF = (
    "Disjunctive rule: the machine switches on when ANY of the nexioms are present on the machine."
)
_REASONING_CUE = (
    "\nHint: review your action history. "
    "If there is a test where only ONE nexiom was on the machine and the light turned ON, the rule is disjunctive. "
    "If the light only turned ON when MULTIPLE nexioms were simultaneously on the machine "
    "(and removing any one of them turned it OFF), the rule is conjunctive."
)


def build_rule_type_question(variant: PromptVariant) -> str:
    if variant.option_order == "disj_first":
        defs = f"{_DISJ_DEF} {_CONJ_DEF}"
        reply = "Reply with exactly one word: disjunctive or conjunctive."
    else:
        defs = f"{_CONJ_DEF} {_DISJ_DEF}"
        reply = "Reply with exactly one word: conjunctive or disjunctive."
    prompt = (
        "Based on the action history and your answers, what type of rule do you think governs this? "
        + defs
    )
    if variant.reasoning_cue:
        prompt += _REASONING_CUE
    prompt += "\n" + reply
    return prompt


def resolve_comprehension_rule(
    variant: PromptVariant, trial_idx: int, seed: int, fallback: str
) -> str:
    """Pick the comprehension-env rule for this trial based on the variant.

    `fallback` is the rule set in the top-level YAML; returned when the variant
    is the sentinel "default" (i.e. defer to YAML)."""
    mode = (variant.comprehension_rule or "").strip().lower()
    if mode in ("", "default"):
        return fallback
    if mode in ("conjunctive", "disjunctive"):
        return mode
    if mode == "alternate":
        return "conjunctive" if trial_idx % 2 == 0 else "disjunctive"
    if mode == "random":
        return _random.Random(seed).choice(["conjunctive", "disjunctive"])
    return fallback
