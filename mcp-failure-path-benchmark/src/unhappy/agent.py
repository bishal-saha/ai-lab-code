"""One episode: a fresh window, a real MCP server, one task.

The loop lists the server's tools through the in-process MCP client, converts them for the Claude API
(experiment 1's mcp_tool_to_anthropic), and calls tools until the agent calls submit_answer, stops
calling tools twice (one nudge), or runs out of turns. Every call and its result text are logged
here, because schema violations are rejected by the SDK before the server's own code sees them.
"""

from __future__ import annotations

import json
import time
from typing import Any

from mcp import Client

from unhappy.grading import grade
from unhappy.pricing import cost_usd
from unhappy.servers import Ledger, build_server
from unhappy.tasks import Task
from unhappy.ticketing import TicketingAPI

MAX_TURNS = 15
SYSTEM = """You are a support agent for Kestrel Outfitters, an outdoor-gear retailer, working in the support ticketing system through its tools.
Do what the user asks with the tools. Look things up; never guess a ticket's details or a count.
When you have finished, or if you can't finish, call submit_answer exactly once."""
NUDGE = "Please call submit_answer now."


def mcp_tool_to_anthropic(tool: Any) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.input_schema,
    }


def _text(result: Any) -> str:
    return "\n".join(getattr(p, "text", "") for p in result.content) or "(empty result)"


async def episode(
    model, model_name: str, variant: str, task: Task, data, *, budget=None
) -> dict[str, Any]:
    api = TicketingAPI(data, task.make_faults())
    ledger = Ledger()
    calls: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    stop, nudged, t0 = "max_turns", False, time.monotonic()
    async with Client(build_server(variant, api, ledger)) as client:
        tools = [mcp_tool_to_anthropic(t) for t in (await client.list_tools()).tools]
        messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
        for _ in range(MAX_TURNS):
            if budget:
                budget.check()
            turn = await model.create(
                model=model_name, system=SYSTEM, messages=messages, tools=tools, task=task
            )
            u = turn.usage
            cost = cost_usd(
                model_name, u["input_tokens"], u["output_tokens"], u["cache_read"], u["cache_write"]
            )
            if budget:
                budget.add(cost)
            usage_rows.append({**u, "cost_usd": cost})
            if turn.stop_reason == "refusal":
                stop = "refusal"
                break
            messages.append({"role": "assistant", "content": turn.content})
            uses = [b for b in turn.blocks if b["type"] == "tool_use"]
            if not uses:
                if nudged:
                    stop = "no_answer"
                    break
                nudged = True
                messages.append({"role": "user", "content": NUDGE})
                continue
            results = []
            for b in uses:
                r = await client.call_tool(b["name"], b["input"])
                text, is_error = _text(r), bool(r.is_error)
                if b["name"] != "submit_answer":
                    calls.append(
                        {
                            "tool": b["name"],
                            "args": b["input"],
                            "is_error": is_error,
                            "text": text[:300],
                        }
                    )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b["id"],
                        "content": text,
                        **({"is_error": True} if is_error else {}),
                    }
                )
            messages.append({"role": "user", "content": results})
            if ledger.answer is not None:
                stop = "answered"
                break
    g = grade(task, api.comments, calls, ledger.answer)
    return {
        "variant": variant,
        "task": task.id,
        "mode": task.mode,
        "model": model_name,
        "stop": stop,
        **g,
        "answer": ledger.answer,
        "calls": calls,
        "server_errors": [c.__dict__ for c in ledger.calls if not c.ok],
        "comments": [c.__dict__ for c in api.comments],
        "usage": usage_rows,
        "cost_usd": round(sum(x["cost_usd"] for x in usage_rows), 6),
        "turns": len(usage_rows),
        "seconds": round(time.monotonic() - t0, 2),
    }


def transcript_text(messages: list[dict]) -> str:  # for debugging: what the model saw
    return json.dumps(messages, default=str, indent=1)
