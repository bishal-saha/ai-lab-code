"""The two model clients: real Claude, and an offline scripted fake.

The fake exists to test the plumbing (servers, loop, grading, reporting) without
a key or a cent. It knows each task's answer and calls a plausible tool first, so
its "results" say nothing about tool design; --dry-run output is never written
to results/.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import anthropic

from tool_shape.agent import Turn
from tool_shape.pricing import resolve
from tool_shape.tasks import Task


class ClaudeModel:
    """Claude through the Anthropic SDK. Thinking stays at the model's default (adaptive);
    effort is pinned so both models run the same setting."""

    def __init__(
        self, model: str, *, effort: str = "medium", client: anthropic.AsyncAnthropic | None = None
    ):
        self.name = resolve(model)
        self.effort = effort
        self.client = client or anthropic.AsyncAnthropic()

    async def create(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn:
        response = await self.client.messages.create(
            model=self.name,
            max_tokens=16000,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # the growing transcript is re-read every turn
        )
        blocks = [
            {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
            if b.type == "tool_use"
            else {"type": b.type}
            for b in response.content
        ]
        usage = response.usage
        return Turn(
            content=response.content,  # thinking blocks must go back unchanged
            blocks=blocks,
            stop_reason=response.stop_reason or "",
            usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
            },
        )


# The first tool call the fake makes for each task family, per design.
FIRST_CALLS: dict[str, dict[str, Any]] = {
    "endpoints": {
        "delivery": lambda t, ref: [
            ("get_order", {"order_id": ref}),
            ("get_order_shipment", {"order_id": ref}),
        ],
        "returns": lambda t, ref: [
            ("get_order", {"order_id": ref}),
            ("get_policy", {"name": "returns-policy"}),
        ],
        "ticket": lambda t, ref: [("get_ticket", {"ticket_id": ref})],
        "stock": lambda t, ref: [("get_stock", {"sku": ref})],
        "policy": lambda t, ref: [("list_policies", {})],
    },
    "tasks": {
        "delivery": lambda t, ref: [("order_status", {"order_id": ref})],
        "returns": lambda t, ref: [
            ("check_return_eligibility", {"order_id": ref, "as_of": t.prompt[9:19]})
        ],
        "ticket": lambda t, ref: [("ticket_context", {"ticket_id": ref})],
        "stock": lambda t, ref: [("stock_check", {"sku": ref})],
        "policy": lambda t, ref: [("read_policy", {"topic": "shipping"})],
    },
}


def _reference(task: Task) -> str:
    """The order, ticket or SKU id named in the prompt."""
    for word in task.prompt.replace(",", " ").replace("?", " ").split():
        if word.startswith(("O-", "T-", "SKU-")):
            return word.rstrip(".")
    return ""


class FakeModel:
    """Scripted: turn 1 calls the family's usual tool(s), turn 2 submits.

    behaviour: "correct" submits the ground truth, "wrong" flips the first field,
    "silent" never calls a tool, "broken" repeats a failing call (trips the breaker).
    Usage is a rough characters/4 estimate so --dry-run exercises the cost path.
    """

    name = "fake"

    def __init__(self, task: Task, server: str, behaviour: str = "correct") -> None:
        self.task, self.server, self.behaviour = task, server, behaviour
        self._ids = (f"toolu_{n:04d}" for n in itertools.count())
        self._turn = 0

    def _use(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"type": "tool_use", "id": next(self._ids), "name": name, "input": args}

    def _answer(self) -> dict[str, Any]:
        answer = dict(self.task.expected)
        if self.behaviour == "wrong":
            first = next(iter(answer))
            value = answer[first]
            answer[first] = (not value) if isinstance(value, bool) else f"not-{value}"
        return answer

    async def create(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn:
        self._turn += 1
        if self.behaviour == "silent":
            blocks = [{"type": "text", "text": "I think it's fine."}]
        elif self.behaviour == "broken":
            blocks = [self._use(tools[0]["name"], {"order_id": "O-DOES-NOT-EXIST"})]
        elif self._turn == 1:
            calls = FIRST_CALLS[self.server][self.task.family](self.task, _reference(self.task))
            blocks = [self._use(name, args) for name, args in calls]
        else:
            blocks = [self._use("submit_answer", self._answer())]
        prompt_chars = len(system) + len(json.dumps(tools)) + len(json.dumps(messages, default=str))
        return Turn(
            content=blocks,
            blocks=blocks,
            stop_reason="tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn",
            usage={"input_tokens": prompt_chars // 4, "output_tokens": 60},
        )
