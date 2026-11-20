"""One agent loop, identical for both tool designs.

A manual loop rather than the SDK's tool runner, because the benchmark needs to
see every turn: tool calls, tool errors, the circuit breaker and token usage per
request. The model sits behind a small protocol so the offline fake (tests, CI,
--dry-run) and the real Claude client are interchangeable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from mcp import Client

from tool_shape.grading import grade
from tool_shape.pricing import cost_usd
from tool_shape.tasks import Task, answer_tool

MAX_TURNS = 12
BREAKER_CONSECUTIVE_ERRORS = 3

SYSTEM_PROMPT = """You are a customer-support agent for Kestrel Outfitters, an outdoor-gear retailer.
Answer the question using the tools. Look facts up; never guess an order, ticket or policy detail.
Company policy documents are available through the tools; when a question depends on policy, use the current version.
When you know the answer, call submit_answer exactly once with every field filled in."""

NUDGE = "Please call submit_answer now with your final answer."


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
    task_id: str
    family: str
    server: str
    model: str
    repeat: int
    success: bool = False
    wrong_fields: list[str] = field(default_factory=list)
    answer: dict[str, Any] | None = None
    stop: str = "max_turns"  # answered | no_answer | max_turns | breaker | refusal | error
    turns: int = 0
    nudged: bool = False
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
    task: Task, mcp_server: Any, server_name: str, model: Model, repeat: int = 0
) -> Episode:
    ep = Episode(
        task_id=task.id, family=task.family, server=server_name, model=model.name, repeat=repeat
    )
    started = time.perf_counter()
    try:
        async with Client(mcp_server) as client:
            listed = await client.list_tools()
            tools = [mcp_tool_to_anthropic(t) for t in listed.tools] + [answer_tool(task)]
            messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
            consecutive_errors = 0
            failed_calls: set[str] = set()

            while ep.turns < MAX_TURNS:
                turn = await model.create(system=SYSTEM_PROMPT, messages=messages, tools=tools)
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
                    if ep.nudged:  # one reminder, the same for both designs
                        ep.stop = "no_answer"
                        break
                    ep.nudged = True
                    messages.append({"role": "user", "content": NUDGE})
                    continue

                results: list[dict[str, Any]] = []
                tripped = False
                for use in uses:
                    if use["name"] == "submit_answer":
                        ep.answer = dict(use["input"])
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": use["id"],
                                "content": "Recorded.",
                            }
                        )
                        continue
                    ep.tool_calls += 1
                    ep.tools_used.append(use["name"])
                    result = await client.call_tool(use["name"], use["input"])
                    is_error = bool(result.is_error)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": use["id"],
                            "content": _result_text(result),
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
                if ep.answer is not None:
                    ep.stop = "answered"
                    break
                if tripped:
                    ep.stop = "breaker"
                    break
    except Exception as err:  # an API or transport failure fails the episode, not the run
        ep.stop = "error"
        ep.error = f"{type(err).__name__}: {err}"

    ep.latency_s = round(time.perf_counter() - started, 3)
    ep.success, ep.wrong_fields = grade(task, ep.answer)
    ep.cost_usd = cost_usd(
        model.name, ep.input_tokens, ep.output_tokens, ep.cache_read_tokens, ep.cache_write_tokens
    )
    return ep
