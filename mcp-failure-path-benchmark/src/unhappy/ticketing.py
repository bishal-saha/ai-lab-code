"""The ticketing API behind every server: Kestrel Outfitters' 300 support tickets, in memory.

It is the article's running example ("an MCP server in front of an internal ticketing API: search
tickets, get a ticket, add a comment"), with the behaviour a real API has:
- search filters by status, priority, channel, customer and text, and returns at most PAGE_MAX
  tickets per call (every server variant has the same cap);
- add_comment honours an idempotency key: a repeat with a key it has seen is not posted again;
- a `Faults` plan, set per task, injects one failure deterministically: a read that times out
  once, a rate limit once, a write that lands and then times out (the dropped response), or a
  ticket locked against comments.

The servers decide only how these failures are presented to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kestrel.data import generate
from kestrel.services import KestrelData

SEED, SCALE = 2026, 0.1  # 300 tickets
PAGE_MAX = 20
STATUSES = ("open", "pending", "solved")
PRIORITIES = ("low", "normal", "high", "urgent")
CHANNELS = ("chat", "email", "phone", "web form")


class NotFound(Exception):
    pass


class Timeout(Exception):
    pass


class RateLimited(Exception):
    pass


class Forbidden(Exception):
    pass


@dataclass
class Faults:
    """At most one failure per task, injected the first time its trigger is met."""

    timeout_once: str | None = None  # tool name whose first call times out
    rate_limit_once: str | None = None  # tool name whose first call is rate-limited
    drop_write_once: str | None = None  # ticket id: the first comment lands, then times out
    locked: tuple[str, ...] = ()  # ticket ids that refuse comments
    fired: set[str] = field(default_factory=set)

    def trip(self, kind: str) -> bool:
        if kind in self.fired:
            return False
        self.fired.add(kind)
        return True


@dataclass(frozen=True)
class Comment:
    ticket_id: str
    body: str
    internal: bool
    key: str | None


def dataset() -> KestrelData:
    return KestrelData(generate(SEED, SCALE))


class TicketingAPI:
    def __init__(self, data: KestrelData, faults: Faults | None = None):
        self.d = data
        self.faults = faults or Faults()
        self.tickets = {t.id: t for t in data.dataset.tickets}
        self.comments: list[Comment] = []
        self._keys: set[str] = set()

    # -- helpers ------------------------------------------------------------------------
    def customer(self, t) -> tuple[str, str]:
        c = self.d.customers[t.customer_id]
        return c.name, c.email

    def updated(self, t) -> str:
        return max(m.sent_at for m in t.messages).date().isoformat()

    def _maybe_fail(self, tool: str) -> None:
        if self.faults.timeout_once == tool and self.faults.trip(f"timeout:{tool}"):
            raise Timeout(tool)
        if self.faults.rate_limit_once == tool and self.faults.trip(f"rate:{tool}"):
            raise RateLimited(tool)

    # -- the API --------------------------------------------------------------------------
    def search(
        self,
        *,
        status=None,
        priority=None,
        channel=None,
        customer=None,
        text=None,
        offset: int = 0,
        limit: int = PAGE_MAX,
    ):
        """Matching tickets, newest activity first, and the total number that match."""
        self._maybe_fail("search_tickets")
        rows = []
        for t in self.tickets.values():
            if status and t.status.value != status:
                continue
            if priority and t.priority.value != priority:
                continue
            if channel and t.channel != channel:
                continue
            name, email = self.customer(t)
            if customer and customer.lower() not in f"{name} {email}".lower():
                continue
            if (
                text
                and text.lower()
                not in (t.subject + " " + " ".join(m.body for m in t.messages)).lower()
            ):
                continue
            rows.append(t)
        rows.sort(key=lambda t: (self.updated(t), t.id), reverse=True)
        limit = max(1, min(int(limit), PAGE_MAX))
        return rows[offset : offset + limit], len(rows)

    def get(self, ticket_id: str):
        self._maybe_fail("get_ticket")
        t = self.tickets.get(ticket_id.strip().upper())
        if t is None:
            raise NotFound(ticket_id)
        return t

    def add_comment(
        self, ticket_id: str, body: str, *, internal: bool = True, key: str | None = None
    ) -> Comment:
        t = self.tickets.get(ticket_id.strip().upper())
        if t is None:
            raise NotFound(ticket_id)
        if t.id in self.faults.locked:
            raise Forbidden(t.id)
        if key and key in self._keys:  # already posted: an idempotent repeat
            return next(c for c in self.comments if c.key == key)
        c = Comment(t.id, body, internal, key)
        self.comments.append(c)
        if key:
            self._keys.add(key)
        if self.faults.drop_write_once == t.id and self.faults.trip(f"drop:{t.id}"):
            raise Timeout("add_ticket_comment")  # the write landed; the response never arrived
        return c
