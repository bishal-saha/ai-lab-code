"""The model clients: real Claude, and a literal-minded fake for the dry run.

`LiteralFake` does exactly what the server's messages tell it, and the naive thing when they tell it
nothing:
- an error that says to use search_tickets: it searches by the customer's name; an opaque error: it gives up
- a guessed status or priority: when the server lists the allowed values it picks the matching one;
  when the server just returns "No tickets match." it reports zero
- a count: when the server gives a total it uses it; otherwise it counts the lines it was shown
- a dropped write: it resends once (the naive retry), and the server's keys decide if that duplicates
- a temporary failure: it retries if told that's safe, gives up on an opaque error
- a locked ticket: it stops if told retrying won't help; otherwise it tries three times

Against the production server it should finish every task; each ablation should break exactly the
failure mode it removes. That is what --dry-run checks. Its numbers say nothing about real models.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

MAX_TOKENS = 4000


@dataclass
class Turn:
    content: list[Any]
    blocks: list[dict]
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)


class ClaudeModel:
    """Claude through the Anthropic SDK; thinking at the model's default (adaptive), effort pinned."""

    def __init__(self, *, effort: str = "medium", client: anthropic.AsyncAnthropic | None = None):
        self.effort = effort
        self.client = client or anthropic.AsyncAnthropic(max_retries=6)

    async def create(
        self, *, model: str, system: str, messages: list[dict], tools: list[dict], task=None
    ) -> Turn:
        r = await self.client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            tools=tools,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        )
        blocks = [
            {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
            for b in r.content
            if b.type == "tool_use"
        ]
        u = r.usage
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read": u.cache_read_input_tokens or 0,
            "cache_write": u.cache_creation_input_tokens or 0,
        }
        return Turn(list(r.content), blocks, r.stop_reason or "", usage)


# --- the literal-minded fake -----------------------------------------------------------------

_ids = itertools.count(1)
NAIVE_FILTERS = {  # the plain-words guess an agent makes before it knows the API's values
    "e1": {"status": "waiting_on_customer", "priority": "high"},
    "e2": {"status": "waiting_on_customer", "channel": "chat"},
    "e3": {"status": "waiting_on_customer", "priority": "low"},
    "e4": {"status": "closed", "priority": "critical"},
    "l1": {"status": "open"},
    "l2": {"status": "pending"},
    "l3": {"status": "open", "priority": "normal"},
    "l4": {"status": "solved", "priority": "high"},
    "tr-s1": {"status": "open", "priority": "urgent"},
    "tr-s2": {"status": "pending", "channel": "email"},
}
SYNONYMS = {"waiting_on_customer": "pending", "closed": "solved", "critical": "urgent"}


def _history(messages: list[dict]) -> list[tuple[str, dict, str, bool]]:
    """(tool, input, result text, is_error) for every call so far."""
    uses, out = {}, []
    for m in messages:
        if m["role"] == "assistant":
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    uses[b["id"]] = b
        elif isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    u = uses[b["tool_use_id"]]
                    out.append((u["name"], u["input"], b["content"], bool(b.get("is_error"))))
    return out


class LiteralFake:
    async def create(
        self, *, model: str, system: str, messages: list[dict], tools: list[dict], task=None
    ) -> Turn:
        call = self._next(task, _history(messages))
        name, args = call
        block = {"type": "tool_use", "id": f"toolu_fake{next(_ids)}", "name": name, "input": args}
        return Turn(
            [block],
            [block],
            "tool_use",
            {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_write": 0},
        )

    def _next(self, task, h) -> tuple[str, dict]:
        def submit(outcome: str, answer: str = "", **kw) -> tuple[str, dict]:
            return "submit_answer", {"outcome": outcome, "answer": answer, **kw}

        last = h[-1] if h else None
        mode = task.mode
        if mode == "not_found":
            bad = re.search(r"T-\d{6}", task.prompt).group(0)
            name = task.prompt.split(" says")[0]
            note = re.search(r'note to their ticket: "(.*)"', task.prompt).group(1)
            if not h:
                return "get_ticket", {"ticket_id": bad}
            if last[0] == "get_ticket" and last[3]:
                return (
                    ("search_tickets", {"customer": name})
                    if "search_tickets" in last[2]
                    else submit("could_not_complete", "Ticket not found.")
                )
            if last[0] == "search_tickets":
                ids = re.findall(r"^(T-\d{6})", last[2], re.M)
                return (
                    ("add_ticket_comment", {"ticket_id": ids[0], "body": note})
                    if ids
                    else submit("could_not_complete", "No ticket.")
                )
            return submit("done", "Note added.")
        if mode in ("enum", "large") or (mode == "transient" and task.count is not None):
            filters = dict(NAIVE_FILTERS[task.id])
            if not h:
                return "search_tickets", filters
            if (
                last[3] and "Input should be" in last[2]
            ):  # the strict schema named the field: use its values
                return "search_tickets", {k: SYNONYMS.get(v, v) for k, v in filters.items()}
            if last[3]:  # a temporary failure
                return (
                    ("search_tickets", last[1])
                    if "retry" in last[2].lower()
                    else submit("could_not_complete", "Search failed.")
                )
            if last[2].startswith("No tickets match"):
                return submit("done", "None.", count=0)
            total = re.match(r"(\d+) tickets match", last[2])
            return submit(
                "done",
                "Counted.",
                count=int(total.group(1)) if total else len(last[2].splitlines()),
            )
        if mode == "transient":
            tid = re.search(r"T-\d{6}", task.prompt).group(0)
            if not h:
                return "get_ticket", {"ticket_id": tid}
            if last[3]:
                return (
                    ("get_ticket", {"ticket_id": tid})
                    if "retry" in last[2].lower()
                    else submit("could_not_complete", "Lookup failed.")
                )
            _, status, priority = (p.strip() for p in last[2].split("|")[:3])
            return submit("done", f"Status {status}, priority {priority}.")
        tid = re.search(r"T-\d{6}", task.prompt).group(0)
        note = re.search(r': "(.*)"', task.prompt).group(1)
        posts = [x for x in h if x[0] == "add_ticket_comment"]
        if mode == "dropped_write":
            if not posts or (last[3] and len(posts) == 1):  # the naive retry after a failed write
                return "add_ticket_comment", {"ticket_id": tid, "body": note}
            return submit("done", "Note added.")
        # permission
        if not posts:
            return "add_ticket_comment", {"ticket_id": tid, "body": note}
        if "won't help" in last[2] or len(posts) >= 3:
            return submit("could_not_complete", "The ticket is locked.")
        return "add_ticket_comment", {"ticket_id": tid, "body": note}
