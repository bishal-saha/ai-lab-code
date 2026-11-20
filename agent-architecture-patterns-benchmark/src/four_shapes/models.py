"""The two model clients: real Claude, and an offline scripted fake.

`ClaudeModel` makes tool-calling turns and structured-output calls (a JSON schema in
`output_config.format`). `FakeModel` plays every role of every architecture from
the scenario's ground truth, so the dry run tests the plumbing without a key:

    oracle  solves each ticket the way its design allows, retrying a temporary failure
    sloppy  makes the failures the grader must catch: money before the eligibility
            check (loop, billing agent), a dropped bug report (loop, triage), and a
            first plan that refunds before checking and calls a tool that doesn't exist

Its numbers say nothing about the designs; --dry-run output is never written to results/.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import anthropic

from four_shapes.episode import Turn
from four_shapes.pricing import resolve
from four_shapes.scenarios import Scenario


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
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        role: str = "",
        meta: dict[str, Any] | None = None,
    ) -> Turn:
        output_config: dict[str, Any] = {"effort": self.effort}
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kwargs: dict[str, Any] = {"tools": tools} if tools else {}
        response = await self.client.messages.create(
            model=self.name,
            max_tokens=16000,
            system=system,
            messages=messages,
            output_config=output_config,
            cache_control={
                "type": "ephemeral"
            },  # a loop's growing transcript is re-read every turn
            **kwargs,
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
            text="".join(getattr(b, "text", "") for b in response.content if b.type == "text"),
        )


# ─── Ground-truth calls, shared by the fake's roles ────────────────────────────


def outcome_call(s: Scenario) -> tuple[str, dict[str, Any]]:
    if s.outcome == "refund":
        return "issue_refund", {
            "order_id": s.order_id,
            "amount_usd": s.total,
            "reason": "Return inside the window",
        }
    if s.outcome == "escalate":
        return "escalate_to_finance", {"order_id": s.order_id, "reason": "Total above USD 500"}
    if s.outcome == "status":
        return "reply_to_customer", {
            "ticket_id": s.ticket_id,
            "message": "Here is where your order is.",
        }
    return "reply_to_customer", {
        "ticket_id": s.ticket_id,
        "message": "Sorry, this return is outside the window.",
    }


def args_for(s: Scenario, tool: str) -> dict[str, Any]:
    """What an executor that knows the answer would pass to `tool`."""
    if tool == "customer_orders":
        return {"email": s.customer_email}
    if tool == "order_status":
        return {"order_id": s.order_id}
    if tool == "check_return_eligibility":
        return {"order_id": s.order_id, "as_of": s.today}
    if tool == "route_to_engineering":
        return {"ticket_id": s.ticket_id, "summary": s.bug or ""}
    if tool in ("issue_refund", "escalate_to_finance", "reply_to_customer"):
        name, args = outcome_call(s)
        return (
            args
            if name == tool
            else {"ticket_id": s.ticket_id, "message": "Update on your ticket."}
        )
    return {"ticket_id": s.ticket_id}


def solution(
    s: Scenario, *, sloppy: bool = False, billing_only: bool = False, technical_only: bool = False
) -> list[tuple[str, dict[str, Any]]]:
    """The tool calls that resolve a ticket, in order (the single loop's script)."""
    if technical_only:
        return [("route_to_engineering", args_for(s, "route_to_engineering"))]
    calls = [] if billing_only else [("get_ticket", {"ticket_id": s.ticket_id})]
    if not s.order_in_ticket:
        calls.append(("customer_orders", args_for(s, "customer_orders")))
    if s.outcome == "status":
        calls += [("order_status", args_for(s, "order_status")), outcome_call(s)]
    elif sloppy:
        calls += [
            outcome_call(s),
            ("check_return_eligibility", args_for(s, "check_return_eligibility")),
        ]
    else:
        calls += [
            ("check_return_eligibility", args_for(s, "check_return_eligibility")),
            outcome_call(s),
        ]
    if s.bug and not sloppy and not billing_only:
        calls.append(("route_to_engineering", args_for(s, "route_to_engineering")))
    return calls


def plan_for(s: Scenario) -> list[str]:
    tools = []
    if not s.order_in_ticket:
        tools.append("customer_orders")
    tools += (
        ["order_status", "reply_to_customer"]
        if s.outcome == "status"
        else ["check_return_eligibility", outcome_call(s)[0]]
    )
    if s.bug:
        tools.append("route_to_engineering")
    return tools


class FakeModel:
    """Scripted stand-in for Claude in every role. Usage is a rough characters/4 estimate."""

    name = "fake"

    def __init__(self, s: Scenario, behaviour: str = "oracle") -> None:
        self.s, self.sloppy = s, behaviour == "sloppy"
        self._ids = (f"toolu_{n:04d}" for n in itertools.count())
        self._scripts: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._last: dict[str, tuple[str, dict[str, Any]]] = {}

    def _turn(
        self,
        blocks: list[dict[str, Any]],
        text: str,
        system: str,
        messages: list,
        tools: list | None,
    ) -> Turn:
        chars = len(system) + len(json.dumps(tools or [])) + len(json.dumps(messages, default=str))
        return Turn(
            content=blocks,
            blocks=blocks,
            stop_reason="tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn",
            usage={"input_tokens": chars // 4, "output_tokens": 80},
            text=text,
        )

    def _loop_turn(
        self, role: str, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str]:
        s = self.s
        if role not in self._scripts:
            self._scripts[role] = {
                "loop": lambda: solution(s, sloppy=self.sloppy),
                "billing": lambda: solution(s, sloppy=self.sloppy, billing_only=True),
                "technical": lambda: solution(s, technical_only=True),
            }[role]()
        last = messages[-1]["content"]
        retry = (
            isinstance(last, list)
            and any(
                r.get("is_error") and "temporarily unavailable" in str(r.get("content"))
                for r in last
            )
            and role in self._last
        )
        if retry:
            call = self._last[role]
        elif self._scripts[role]:
            call = self._scripts[role].pop(0)
        else:
            return [{"type": "text", "text": "Done."}], "Done."
        self._last[role] = call
        name, args = call
        return [{"type": "tool_use", "id": next(self._ids), "name": name, "input": args}], ""

    def _json(self, role: str, meta: dict[str, Any]) -> dict[str, Any]:
        s = self.s
        if role == "classify":
            intent = "order_status" if s.outcome == "status" else "refund_request"
            return {"intent": intent, "order_id": s.order_id if s.order_in_ticket else ""}
        if role == "draft":
            return {"message": "Thanks for getting in touch. Here is what we did."}
        if role == "plan":
            if self.sloppy and meta.get("attempt") == 0:
                return {
                    "steps": [
                        {"tool": "lookup_contract", "purpose": "find the contract"},
                        {"tool": "issue_refund", "purpose": "refund"},
                    ]
                }
            # A replan covers what remains: drop the steps that already succeeded, once each.
            done = list(meta.get("done", []))
            remaining = []
            for t in plan_for(s):
                if t in done:
                    done.remove(t)
                else:
                    remaining.append(t)
            return {"steps": [{"tool": t, "purpose": f"step: {t}"} for t in remaining]}
        if role == "execute":
            return {
                "action": "call",
                "args_json": json.dumps(args_for(s, meta["tool"])),
                "note": "per the plan",
            }
        if role == "triage":
            return {
                "billing": "status" if s.outcome == "status" else "refund",
                "order_id": s.order_id if s.order_in_ticket else "",
                "technical": bool(s.bug) and not self.sloppy,
                "bug_summary": (s.bug or "") if not self.sloppy else "",
                "human": False,
                "reason": "Customer request",
            }
        raise ValueError(f"no scripted answer for role {role!r}")

    async def create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        role: str = "",
        meta: dict[str, Any] | None = None,
    ) -> Turn:
        if schema:
            text = json.dumps(self._json(role, meta or {}))
            return self._turn([{"type": "text", "text": text}], text, system, messages, tools)
        blocks, text = self._loop_turn(role, messages)
        return self._turn(blocks, text, system, messages, tools)
