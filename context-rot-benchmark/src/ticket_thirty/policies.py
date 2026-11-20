"""The six context policies: each turns the agent's full history into the next request's messages.

The agent loop keeps the whole history (a TicketLog per ticket) and asks a policy, before
every request, what the window should hold:

    raw          everything, in order of arrival
    trim         the same as raw; the difference is the tools' output mode (trimmed in code)
    clear        raw, but once a ticket is closed its tool results become a short stub with a
                 reference (client-side context editing). Cleared in one batch per ticket, so
                 the cached prefix breaks once per ticket, not once per call
    summary      after each ticket, everything so far is folded into a running summary; the
                 window is that summary plus the current ticket. The standing instructions
                 live only in what the summariser chose to keep
    summary+pin  the same summary, with the standing instructions kept verbatim in a pinned block
    fresh        a new window per ticket, seeded with the opening message (sub-agent isolation)

Under every policy, thinking blocks from closed tickets are dropped (the API needs them only
within the current tool loop), so all six differ only in what the policy keeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ticket_thirty.queue import CONSTRAINTS, OPENING, announce, opening_for

POLICIES = ["raw", "trim", "clear", "summary", "summary+pin", "fresh"]
SUMMARISED = ("summary", "summary+pin")


@dataclass
class TicketLog:
    ticket_id: str
    turns: list[dict[str, Any]] = field(
        default_factory=list
    )  # assistant / user(tool_result) messages
    closed: bool = False


def _plain(block: Any) -> dict | None:
    """A closed ticket's assistant block as a plain dict; thinking is dropped."""
    kind = block["type"] if isinstance(block, dict) else block.type
    if kind == "text":
        text = block["text"] if isinstance(block, dict) else block.text
        return {"type": "text", "text": text} if text else None
    if kind == "tool_use":
        if isinstance(block, dict):
            return {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": block["input"],
            }
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return None  # thinking, redacted_thinking


def close(log: TicketLog) -> None:
    """Freeze a finished ticket: plain dicts, no thinking, never an empty assistant message."""
    for m in log.turns:
        if m["role"] == "assistant":
            blocks = [b for b in map(_plain, m["content"]) if b]
            m["content"] = blocks or [{"type": "text", "text": "(done)"}]
    log.closed = True


def _add_user(msgs: list[dict], blocks: list[dict]) -> None:
    """Append user content, merging into the previous message when it is also a user turn
    (a ticket that ran out of turns ends on tool results)."""
    if msgs and msgs[-1]["role"] == "user":
        prev = msgs[-1]["content"]
        msgs[-1] = {
            "role": "user",
            "content": (prev if isinstance(prev, list) else [{"type": "text", "text": prev}])
            + blocks,
        }
    else:
        msgs.append({"role": "user", "content": blocks})


def _text(t: str) -> list[dict]:
    return [{"type": "text", "text": t}]


def _stub(log: TicketLog) -> list[dict]:
    """A closed ticket with every tool result replaced by a stub that keeps a reference."""
    names = {
        b["id"]: (b["name"], b["input"])
        for m in log.turns
        if m["role"] == "assistant"
        for b in m["content"]
        if b["type"] == "tool_use"
    }
    out = []
    for m in log.turns:
        if m["role"] == "user":
            content = []
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    name, args = names.get(b["tool_use_id"], ("tool", {}))
                    ref = ", ".join(
                        f"{k}={v}"
                        for k, v in args.items()
                        if k in ("order_id", "ticket_id", "email")
                    )
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": b["tool_use_id"],
                            "content": f"[cleared after {log.ticket_id} closed: {name}({ref}). Call it again if needed.]",
                        }
                    )
                else:
                    content.append(b)
            out.append({"role": "user", "content": content})
        else:
            out.append(m)
    return out


def build_messages(policy: str, logs: list[TicketLog], summary: str | None) -> list[dict]:
    """The window for the next request. `logs` ends with the current (open) ticket."""
    current = logs[-1]
    msgs: list[dict] = []
    if policy in ("raw", "trim", "clear"):
        for i, log in enumerate(logs):
            _add_user(
                msgs, _text(opening_for(log.ticket_id) if i == 0 else announce(log.ticket_id))
            )
            msgs += _stub(log) if (policy == "clear" and log.closed) else log.turns
        return msgs
    if policy == "fresh" or summary is None:
        _add_user(msgs, _text(opening_for(current.ticket_id)))
        return msgs + current.turns
    if policy in SUMMARISED:
        head = f"Summary of the work so far (it replaces the earlier conversation):\n{summary}"
        if policy == "summary+pin":
            head = f"Standing instructions from the start of the queue, kept word for word:\n{CONSTRAINTS}\n\n{head}"
        _add_user(msgs, _text(f"{head}\n\n{announce(current.ticket_id)}"))
        return msgs + current.turns
    raise ValueError(f"unknown policy {policy!r}")


SUMMARISER_SYSTEM = """You compact an AI support agent's conversation so it can keep working in a fresh context window.
Write a concise summary that will replace everything below: what the agent is doing, what it has done so far, and anything it needs to continue the work.
Fold the previous summary (if any) into the new one. Write only the summary."""


def render(messages: list[dict]) -> str:
    """A transcript as plain text, for the summariser."""
    lines = []
    for m in messages:
        content = m["content"] if isinstance(m["content"], list) else _text(m["content"])
        for b in content:
            kind = b.get("type")
            if kind == "text":
                lines.append(f"{m['role'].upper()}: {b['text']}")
            elif kind == "tool_use":
                lines.append(f"AGENT CALLED {b['name']}({b['input']})")
            elif kind == "tool_result":
                body = (
                    b["content"]
                    if isinstance(b["content"], str)
                    else " ".join(x.get("text", "") for x in b["content"])
                )
                lines.append(f"TOOL RESULT: {body}")
    return "\n".join(lines)


def summariser_prompt(previous: str | None, log: TicketLog) -> str:
    """What the summariser sees: the previous summary (or, the first time, the opening message)
    and the ticket that just closed, as text."""
    start = f"Previous summary:\n{previous}" if previous else f"USER: {OPENING}"
    return (
        f"{start}\n\nConversation since then:\nUSER: {announce(log.ticket_id)}\n{render(log.turns)}"
    )
