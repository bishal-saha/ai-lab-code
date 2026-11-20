"""What every architecture shares: the episode record, the model protocol, and a
per-episode context that counts every LLM call and tool call the same way.

The four designs in `architectures/` differ only in who decides the next call.
They all go through `Context.llm`, `Context.json` and `Context.call`, so tokens,
cost, calls and failures are measured identically, and the same policy text
(`POLICY`) gives every design the same domain knowledge.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from mcp import Client

from four_shapes.ledger import Ledger
from four_shapes.scenarios import REFUND_LIMIT_USD, Scenario

MAX_LLM_CALLS = 30  # per episode, across every agent and node: a hard bound on cost
MAX_LOOP_TURNS = 12  # per tool-calling loop
BREAKER_CONSECUTIVE_ERRORS = 3

POLICY = f"""Kestrel Outfitters is an outdoor-gear retailer. Support resolves tickets with these rules:
- A ticket without an order id: find the order with customer_orders, using the customer's email from the ticket.
- A delivery question ("where is my order?"): look the order up with order_status and reply with where it is. No money moves.
- A return request: check it with check_return_eligibility (as of today's date) before any money action, then take the one outcome the policy calls for:
  - inside the window and total USD {REFUND_LIMIT_USD} or less: issue_refund for the full order total;
  - inside the window and total above USD {REFUND_LIMIT_USD}: escalate_to_finance;
  - outside the window: reply_to_customer to explain that the return can't be accepted.
- A website or app bug the customer reports: route_to_engineering with a short summary.
- Anything these tools can't resolve: escalate_to_human.
A tool that reports a temporary failure can be retried."""


@dataclass
class Turn:
    """One model response, normalised. `content` goes back into a transcript unchanged."""

    content: Any
    blocks: list[dict[str, Any]]
    stop_reason: str
    usage: dict[str, int]
    text: str = ""


class Model(Protocol):
    name: str

    async def create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        role: str = "",
        meta: dict[str, Any] | None = None,
    ) -> Turn: ...


class Stop(Exception):
    """Ends an episode early; `reason` becomes Episode.stop (refusal, budget, breaker)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason


@dataclass
class Episode:
    scenario: str
    kind: str
    outcome: str
    arch: str
    condition: str  # normal | faults
    model: str
    repeat: int
    success: bool = False
    outcome_ok: bool = False
    violation: bool = False
    missed_intent: bool = False
    extra_engineering: bool = False
    to_human: bool = False
    duplicate_replies: int = 0
    sequence: list[str] = field(default_factory=list)
    invalid_plans: int = 0  # planner: plans rejected by validation
    replans: int = 0  # planner: plans requested after the first
    handoff_invalid: int = 0  # multi-agent: handoffs rejected by the code checks
    notes: list[str] = field(
        default_factory=list
    )  # design-specific facts (the graph's intent, a handoff)
    stop: str = "done"  # done | refusal | budget | breaker | error
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def anthropic_tool(tool: Any) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.input_schema,
    }


def result_text(result: Any) -> str:
    return "\n".join(getattr(part, "text", "") for part in result.content) or "(empty result)"


class Context:
    """One episode's handles: the scenario, the model, the MCP client and the ledger."""

    def __init__(
        self,
        s: Scenario,
        model: Model,
        client: Client,
        ledger: Ledger,
        ep: Episode,
        tools: list[dict[str, Any]],
    ):
        self.s, self.model, self.client, self.ledger, self.ep = s, model, client, ledger, ep
        self.all_tools = tools

    def tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        return (
            self.all_tools if names is None else [t for t in self.all_tools if t["name"] in names]
        )

    async def llm(
        self,
        role: str,
        system: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Turn:
        if self.ep.llm_calls >= MAX_LLM_CALLS:
            raise Stop("budget", f"more than {MAX_LLM_CALLS} LLM calls")
        turn = await self.model.create(
            system=system, messages=messages, tools=tools, schema=schema, role=role, meta=meta
        )
        ep = self.ep
        ep.llm_calls += 1
        ep.input_tokens += turn.usage.get("input_tokens", 0)
        ep.output_tokens += turn.usage.get("output_tokens", 0)
        ep.cache_read_tokens += turn.usage.get("cache_read_input_tokens", 0)
        ep.cache_write_tokens += turn.usage.get("cache_creation_input_tokens", 0)
        if turn.stop_reason == "refusal":
            raise Stop("refusal", f"model refused at {role}")
        return turn

    async def json(
        self,
        role: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """A structured-output call: the response is JSON matching `schema`."""
        turn = await self.llm(
            role, system, [{"role": "user", "content": user}], schema=schema, meta=meta
        )
        return json.loads(turn.text)

    async def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Run one tool. Returns (text, is_error); a failure is data, never an exception."""
        self.ep.tool_calls += 1
        result = await self.client.call_tool(name, args)
        is_error = bool(result.is_error)
        if is_error:
            self.ep.tool_errors += 1
        return result_text(result), is_error

    async def tool_loop(
        self,
        role: str,
        system: str,
        user: str,
        tool_names: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """A model-driven loop: call the tools it asks for until it stops. Returns its final text.

        Circuit breaker: the same failing call twice, or three errors in a row, ends the loop
        (raised as Stop("breaker")). Shared by the single loop and each multi-agent specialist.
        """
        tools = self.tools(tool_names)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        consecutive, failed = 0, set()
        for _ in range(MAX_LOOP_TURNS):
            turn = await self.llm(role, system, messages, tools=tools, meta=meta)
            messages.append({"role": "assistant", "content": turn.content})
            uses = [b for b in turn.blocks if b["type"] == "tool_use"]
            if not uses:
                return turn.text
            results, tripped = [], False
            for use in uses:
                text, is_error = await self.call(use["name"], use["input"])
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": use["id"],
                        "content": text,
                        "is_error": is_error,
                    }
                )
                if is_error:
                    consecutive += 1
                    key = f"{use['name']}:{json.dumps(use['input'], sort_keys=True)}"
                    # Retrying a call the tool itself called temporary is allowed; repeating any other failure trips.
                    repeated = key in failed and "temporarily unavailable" not in text
                    tripped = tripped or repeated or consecutive >= BREAKER_CONSECUTIVE_ERRORS
                    failed.add(key)
                else:
                    consecutive = 0
            messages.append({"role": "user", "content": results})
            if tripped:
                raise Stop("breaker", f"{role}: repeated tool failures")
        return ""
