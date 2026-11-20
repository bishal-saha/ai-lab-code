"""The six judge designs, Claude to run them, and two fakes for the dry run.

Every design judges the same criterion against the same ticket, order facts and policy:

    CRITERION: the reply states the outcome the returns policy requires for this order,
               and promises nothing the policy doesn't allow.

    pf          pass/fail on the one criterion (the article's design)
    pf-crit     pass/fail, with a short critique written first
    pf-neutral  pf, plus length-neutral wording: "a short response can pass"
    score       a 1-10 score on the same criterion; the gate is >= 7
    multi       one pass/fail over three criteria at once (policy, tone, completeness)
    pair        two replies to the same ticket: which better meets the criterion, A or B?

The policy and the design's instructions sit in a cached system block; the case is the user turn.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import anthropic

from judge_the_judge.cases import POLICY, Case
from judge_the_judge.pricing import resolve

CRITERION = "The reply states the outcome the returns policy requires for this order, and promises nothing the policy doesn't allow."
DESIGNS = ["pf", "pf-crit", "pf-neutral", "score", "multi"]
VERDICT = {"type": "string", "enum": ["PASS", "FAIL"]}


def _schema(props: dict) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


INSTRUCTIONS: dict[str, tuple[str, dict]] = {
    "pf": (
        f"Criterion: {CRITERION}\nAnswer PASS if the reply meets the criterion, FAIL if it doesn't.",
        _schema({"verdict": VERDICT}),
    ),
    "pf-crit": (
        f"Criterion: {CRITERION}\nFirst write a short critique (two sentences at most) of the reply against the criterion, then answer PASS or FAIL.",
        _schema({"critique": {"type": "string"}, "verdict": VERDICT}),
    ),
    "pf-neutral": (
        f"Criterion: {CRITERION}\nJudge only this criterion. Length and warmth don't count either way: a short response can pass, and a long, friendly one can fail.\nAnswer PASS or FAIL.",
        _schema({"verdict": VERDICT}),
    ),
    "score": (
        f"Criterion: {CRITERION}\nScore how well the reply meets the criterion from 1 (not at all) to 10 (fully).",
        _schema({"score": {"type": "integer", "description": "An integer from 1 to 10"}}),
    ),
    "multi": (
        "Judge the reply on three criteria together: (1) it states the outcome the returns policy requires and promises nothing the policy doesn't allow; "
        "(2) its tone is polite and professional; (3) it fully answers the customer's request.\nAnswer PASS only if it meets all three, otherwise FAIL.",
        _schema({"verdict": VERDICT}),
    ),
    "pair": (
        f"Criterion: {CRITERION}\nYou will see two replies, A and B, to the same ticket. Answer which one better meets the criterion.",
        _schema({"winner": {"type": "string", "enum": ["A", "B"]}}),
    ),
}


def system_prompt(design: str) -> str:
    return f"You grade customer-support replies for Kestrel Outfitters.\n\n{POLICY}\n\n{INSTRUCTIONS[design][0]}"


def case_prompt(c: Case) -> str:
    return f"Ticket: {c.ticket}\n\nOrder facts: {c.facts()}\n\nReply to grade:\n{c.reply}"


def pair_prompt(first: Case, second: Case) -> str:
    return f"Ticket: {first.ticket}\n\nOrder facts: {first.facts()}\n\nReply A:\n{first.reply}\n\nReply B:\n{second.reply}"


@dataclass
class Judgement:
    value: dict | None  # the parsed structured output
    stop_reason: str
    usage: dict[str, int]


class ClaudeModel:
    def __init__(
        self, model: str, *, effort: str = "low", client: anthropic.AsyncAnthropic | None = None
    ):
        self.name = resolve(model)
        self.effort = effort
        self.client = client or anthropic.AsyncAnthropic()

    async def judge(
        self, design: str, prompt: str, meta: dict[str, Any] | None = None
    ) -> Judgement:
        r = await self.client.messages.create(
            model=self.name,
            max_tokens=4000,
            system=[
                {
                    "type": "text",
                    "text": system_prompt(design),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": INSTRUCTIONS[design][1]},
            },
        )
        text = "".join(getattr(b, "text", "") for b in r.content if b.type == "text")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        u = r.usage
        return Judgement(
            value,
            r.stop_reason or "",
            {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_read": u.cache_read_input_tokens or 0,
                "cache_write": u.cache_creation_input_tokens or 0,
            },
        )


def _coin(*parts: str) -> float:
    """A deterministic pseudo-random number in [0, 1) from the arguments."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16) / 0x1_0000_0000


class FakeModel:
    """Offline stand-ins.

    perfect  always right; pairwise picks the correct reply, and between two correct replies picks by a fair coin
    biased   passes most long FAIL replies (verbosity bias), lets the multi design fail short PASS replies
             as "incomplete", and in pairs prefers position A whenever both replies are correct (position bias)
    """

    def __init__(self, behaviour: str = "perfect"):
        self.behaviour = behaviour
        self.name = "fake"

    async def judge(
        self, design: str, prompt: str, meta: dict[str, Any] | None = None
    ) -> Judgement:
        meta = meta or {}
        usage = {"input_tokens": len(prompt) // 4 + 200, "output_tokens": 30}
        if design == "pair":
            first, second, repeat = meta["first"], meta["second"], str(meta.get("repeat", 0))
            if first.label != second.label:
                winner = "A" if first.label == "PASS" else "B"
            elif self.behaviour == "biased":
                winner = "A"
            else:
                winner = "A" if _coin(first.id, second.id, repeat) < 0.5 else "B"
            return Judgement({"winner": winner}, "end_turn", usage)
        c: Case = meta["case"]
        passed = c.label == "PASS"
        if self.behaviour == "biased":
            if (
                c.label == "FAIL"
                and c.length == "long"
                and design != "pf-neutral"
                and _coin(c.id, design) < 0.75
            ):
                passed = True
            if (
                design == "multi"
                and c.label == "PASS"
                and c.length == "short"
                and _coin(c.id) < 0.3
            ):
                passed = False
        if design == "score":
            return Judgement({"score": (9 if passed else 2 + c.severity)}, "end_turn", usage)
        value = {"verdict": "PASS" if passed else "FAIL"}
        if design == "pf-crit":
            value = {"critique": "Checked against the policy.", **value}
        return Judgement(value, "end_turn", usage)
