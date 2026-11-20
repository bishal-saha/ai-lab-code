"""One agent, one queue, one context policy: the long tool loop the experiment measures.

For each ticket in queue order the harness announces it, then loops: ask the policy what the
window holds, call the model, run its tool calls, until the model ends its turn without a tool
call (the ticket is done) or the ticket's turn budget runs out. A closed ticket is frozen
(thinking dropped) and, under the summary policies, folded into the running summary.

Every model call is recorded with its ticket position and token counts, so the report can show
the window growing and cost per ticket; every tool call lands in the ledger for grading.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from kestrel.services import KestrelData

from ticket_thirty.grading import TicketGrade, grade_ticket, summary_keeps
from ticket_thirty.ledger import Ledger
from ticket_thirty.policies import (
    SUMMARISED,
    SUMMARISER_SYSTEM,
    TicketLog,
    build_messages,
    close,
    summariser_prompt,
)
from ticket_thirty.pricing import cost_usd
from ticket_thirty.queue import SYSTEM
from ticket_thirty.scenarios import Scenario
from ticket_thirty.tools import Tools

MAX_TURNS_PER_TICKET = 12


class BudgetExceeded(Exception):
    """The run's spending cap was reached; the run is abandoned and not recorded (resume reruns it)."""


class Budget:
    """A spending cap shared by every concurrent run, checked before each model call, so a
    long queue can't overshoot it by more than the calls already in flight."""

    def __init__(self, max_usd: float, spent: float = 0.0):
        self.max_usd, self.spent = max_usd, spent

    def check(self) -> None:
        if self.spent >= self.max_usd:
            raise BudgetExceeded(f"cap USD {self.max_usd:.2f} reached")

    def add(self, cost: float) -> None:
        self.spent += cost


@dataclass
class RunResult:
    grades: list[TicketGrade]
    calls: list[dict[str, Any]] = field(default_factory=list)  # one row per model call
    summaries: list[dict[str, Any]] = field(default_factory=list)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)  # every closed ticket's turns
    cost_usd: float = 0.0
    error: str | None = None


async def run_queue(
    model,
    policy: str,
    queue: list[Scenario],
    data: KestrelData,
    *,
    summariser=None,
    distractors: bool = False,
    max_turns: int = MAX_TURNS_PER_TICKET,
    budget: Budget | None = None,
) -> RunResult:
    ledger = Ledger()
    tools = Tools(
        data, queue, ledger, mode="trim" if policy == "trim" else "verbose", distractors=distractors
    )
    schemas = tools.schemas()
    logs: list[TicketLog] = []
    summary: str | None = None
    result = RunResult(grades=[])

    for pos, s in enumerate(queue, 1):
        ledger.current = s.ticket_id
        log = TicketLog(s.ticket_id)
        logs.append(log)
        for turn in range(max_turns):
            messages = build_messages(policy, logs, summary)
            if budget:
                budget.check()
            t0 = time.monotonic()
            t = await model.create(system=SYSTEM, messages=messages, tools=schemas)
            u = t.usage
            cost = cost_usd(
                model.name, u["input_tokens"], u["output_tokens"], u["cache_read"], u["cache_write"]
            )
            result.cost_usd += cost
            if budget:
                budget.add(cost)
            result.calls.append(
                {
                    "position": pos,
                    "ticket": s.ticket_id,
                    "turn": turn,
                    "stop_reason": t.stop_reason,
                    "window_tokens": u["input_tokens"] + u["cache_read"] + u["cache_write"],
                    **u,
                    "cost_usd": cost,
                    "seconds": round(time.monotonic() - t0, 2),
                }
            )
            log.turns.append({"role": "assistant", "content": t.content})
            if not t.blocks:
                break
            if t.stop_reason in ("max_tokens", "refusal"):  # every tool_use still needs a result
                log.turns.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": b["id"],
                                "content": "Not run: the response was cut off.",
                                "is_error": True,
                            }
                            for b in t.blocks
                        ],
                    }
                )
                break
            results = []
            for b in t.blocks:
                text, is_error = tools.call(
                    b["name"], b["input"] if isinstance(b["input"], dict) else {}
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b["id"],
                        "content": text,
                        **({"is_error": True} if is_error else {}),
                    }
                )
            log.turns.append({"role": "user", "content": results})
        close(log)
        result.grades.append(grade_ticket(pos, s, ledger))
        if policy in SUMMARISED and pos < len(queue):
            summary, su = await summariser.summarise(
                system=SUMMARISER_SYSTEM, prompt=summariser_prompt(summary, log)
            )
            scost = cost_usd(
                summariser.name,
                su["input_tokens"],
                su["output_tokens"],
                su["cache_read"],
                su["cache_write"],
            )
            result.cost_usd += scost
            if budget:
                budget.add(scost)
            result.summaries.append(
                {
                    "after_position": pos,
                    "summary": summary,
                    "keeps": summary_keeps(summary),
                    "cost_usd": scost,
                    **su,
                }
            )
    result.ledger = ledger.to_list()
    result.transcript = [{"ticket": log.ticket_id, "turns": log.turns} for log in logs]
    return result
