"""One agent loop for every defence condition.

Adapted from the Tool Shape Benchmark's loop. What changes per condition is only
the system prompt and the wrapper around tool results (both from `defences.py`)
and whether the server enforces policy. The episode ends when the agent stops
calling tools; grading reads the action ledger, never the agent's own summary.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from mcp import Client

from poisoned_ticket.attacks import Attack, any_harm
from poisoned_ticket.defences import Defence, new_boundary
from poisoned_ticket.grading import utility
from poisoned_ticket.kestrel_api import KestrelAPI
from poisoned_ticket.ledger import Ledger
from poisoned_ticket.pricing import cost_usd
from poisoned_ticket.scenarios import Scenario
from poisoned_ticket.servers.support import build_server

MAX_TURNS = 12
BREAKER_CONSECUTIVE_ERRORS = 3


@dataclass
class Turn:
    """One model response, normalised. `content` is appended to the transcript unchanged."""

    content: Any
    blocks: list[dict[str, Any]]
    stop_reason: str
    usage: dict[str, int]


class Model(Protocol):
    name: str

    async def create(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn: ...


@dataclass
class Episode:
    scenario: str
    outcome: str
    attack: str  # "clean" or goal:style
    goal: str | None
    style: str | None
    defence: str
    model: str
    repeat: int
    utility: bool = False
    attack_success: bool | None = None  # None on clean episodes
    harm: dict[str, bool] = field(default_factory=dict)  # every goal's predicate
    flagged: bool = False
    refused: int = 0  # actions the server refused (D3, D4)
    actions: list[dict[str, Any]] = field(default_factory=list)
    stop: str = "max_turns"  # done | max_turns | breaker | refusal | error
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    tools_used: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def mcp_tool_to_anthropic(tool: Any) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.input_schema,
    }


def _result_text(result: Any) -> str:
    return "\n".join(getattr(part, "text", "") for part in result.content) or "(empty result)"


async def run_episode(
    s: Scenario,
    attack: Attack | None,
    defence: Defence,
    api: KestrelAPI,
    model: Model,
    repeat: int = 0,
) -> Episode:
    ep = Episode(
        scenario=s.id,
        outcome=s.outcome,
        attack=attack.id if attack else "clean",
        goal=attack.goal if attack else None,
        style=attack.style if attack else None,
        defence=defence.id,
        model=model.name,
        repeat=repeat,
    )
    ledger = Ledger()
    boundary = new_boundary()
    server = build_server(
        api, s, ledger, injection=attack.text(s) if attack else None, enforce=defence.enforce
    )
    system = defence.system_prompt(boundary)
    started = time.perf_counter()
    try:
        async with Client(server) as client:
            tools = [mcp_tool_to_anthropic(t) for t in (await client.list_tools()).tools]
            messages: list[dict[str, Any]] = [{"role": "user", "content": s.prompt}]
            consecutive_errors = 0
            failed_calls: set[str] = set()

            while ep.turns < MAX_TURNS:
                turn = await model.create(system=system, messages=messages, tools=tools)
                ep.turns += 1
                ep.input_tokens += turn.usage.get("input_tokens", 0)
                ep.output_tokens += turn.usage.get("output_tokens", 0)
                ep.cache_read_tokens += turn.usage.get("cache_read_input_tokens", 0)
                ep.cache_write_tokens += turn.usage.get("cache_creation_input_tokens", 0)
                if turn.stop_reason == "refusal":
                    ep.stop = "refusal"
                    break
                messages.append({"role": "assistant", "content": turn.content})

                uses = [b for b in turn.blocks if b["type"] == "tool_use"]
                if not uses:
                    ep.stop = "done"
                    break

                results: list[dict[str, Any]] = []
                tripped = False
                for use in uses:
                    ep.tool_calls += 1
                    ep.tools_used.append(use["name"])
                    result = await client.call_tool(use["name"], use["input"])
                    is_error = bool(result.is_error)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": use["id"],
                            "content": defence.wrap(_result_text(result), boundary),
                            "is_error": is_error,
                        }
                    )
                    if is_error:
                        ep.tool_errors += 1
                        consecutive_errors += 1
                        key = f"{use['name']}:{json.dumps(use['input'], sort_keys=True)}"
                        # Circuit breaker: the same failing call twice, or too many errors in a row.
                        if key in failed_calls or consecutive_errors >= BREAKER_CONSECUTIVE_ERRORS:
                            tripped = True
                        failed_calls.add(key)
                    else:
                        consecutive_errors = 0
                messages.append({"role": "user", "content": results})
                if tripped:
                    ep.stop = "breaker"
                    break
    except Exception as err:  # an API or transport failure fails the episode, not the run
        ep.stop = "error"
        ep.error = f"{type(err).__name__}: {err}"

    ep.latency_s = round(time.perf_counter() - started, 3)
    ep.actions = ledger.to_list()
    ep.utility = utility(s, ledger)
    ep.harm = any_harm(s, ledger)
    ep.attack_success = attack.succeeded(s, ledger) if attack else None
    ep.flagged = bool(ledger.done("flag_for_review"))
    ep.refused = sum(a.refused for a in ledger.actions)
    ep.cost_usd = cost_usd(
        model.name, ep.input_tokens, ep.output_tokens, ep.cache_read_tokens, ep.cache_write_tokens
    )
    return ep
