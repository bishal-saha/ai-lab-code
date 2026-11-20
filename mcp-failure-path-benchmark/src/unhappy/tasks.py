"""The 24 tasks: six failure modes from the article's table, four tasks each, built from the data.

    not_found     the user gives a ticket id that doesn't exist; find the real one with search_tickets
                  (by customer) and add the note there
    enum          a question in plain words ("waiting on the customer", "closed", "critical") whose filter
                  values are the API's own (pending, solved, urgent); the answers are small (4-12)
    large         a count of 21-57 tickets: more than one page
    dropped_write add a note; the first write lands and then times out
    transient     a lookup or a count whose first call times out or is rate-limited
    permission    add a note to a ticket locked against comments: the right outcome is to say so

Each task has one right outcome, graded in code from the API's comment store, the agent's call log
and its submit_answer call (see grading.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unhappy.ticketing import Faults, TicketingAPI, dataset

MODES = ["not_found", "enum", "large", "dropped_write", "transient", "permission"]
NOTE = "Customer chased by phone; please prioritise."


@dataclass(frozen=True)
class Task:
    id: str
    mode: str
    prompt: str
    faults: dict[str, Any] = field(default_factory=dict)
    count: int | None = None  # the right answer to a "how many" question
    comment_on: str | None = None  # the ticket that should get exactly one comment
    must_mention: tuple[str, ...] = ()  # words the answer must contain
    must_refuse: bool = False  # the right outcome is could_not_complete

    def make_faults(self) -> Faults:
        return Faults(**self.faults)


def build_tasks() -> list[Task]:
    d = dataset()
    api = TicketingAPI(d)
    tickets = sorted(api.tickets.values(), key=lambda t: t.id)
    per_customer: dict[str, list] = {}
    for t in tickets:
        per_customer.setdefault(t.customer_id, []).append(t)
    count = lambda **f: api.search(**f, limit=1)[1]  # noqa: E731
    out: list[Task] = []

    # not_found: customers with a single, still-open ticket; the id the user quotes doesn't exist
    # (and a search by their name finds exactly that one ticket, so the right ticket is unambiguous)
    singles = [
        ts[0]
        for ts in per_customer.values()
        if len(ts) == 1
        and ts[0].status.value in ("open", "pending")
        and count(customer=api.customer(ts[0])[0]) == 1
    ]
    used = {t.id for t in singles[:4]}
    for t in singles[:4]:
        name, _ = api.customer(t)
        bad = f"T-{1000 + int(t.id[2:]):06d}"
        out.append(
            Task(
                f"nf-{t.id}",
                "not_found",
                f'{name} says their ticket {bad} about "{t.subject}" hasn\'t been answered. Add an internal note to their ticket: "{NOTE}"',
                comment_on=t.id,
            )
        )

    # enum: plain-words statuses and priorities; the API's values are pending / solved / urgent
    for key, prompt, filters in (
        (
            "e1",
            "How many high-priority tickets are waiting on the customer to reply?",
            dict(status="pending", priority="high"),
        ),
        (
            "e2",
            "How many tickets that came in by chat are waiting on the customer to reply?",
            dict(status="pending", channel="chat"),
        ),
        (
            "e3",
            "How many low-priority tickets are waiting on the customer to reply?",
            dict(status="pending", priority="low"),
        ),
        (
            "e4",
            "How many closed tickets were marked critical?",
            dict(status="solved", priority="urgent"),
        ),
    ):
        out.append(Task(key, "enum", prompt, count=count(**filters)))

    # large: more matches than one page holds; the API's own words, so only paging is tested
    for key, prompt, filters in (
        ("l1", "How many tickets have the status open?", dict(status="open")),
        ("l2", "How many tickets have the status pending?", dict(status="pending")),
        (
            "l3",
            "How many open tickets have normal priority?",
            dict(status="open", priority="normal"),
        ),
        (
            "l4",
            "How many solved tickets have high priority?",
            dict(status="solved", priority="high"),
        ),
    ):
        out.append(Task(key, "large", prompt, count=count(**filters)))

    opens = [
        t for t in tickets if t.status.value == "open" and t.id not in used
    ]  # every task has its own ticket
    # dropped_write: the first comment lands, then the response times out
    for t in opens[:4]:
        out.append(
            Task(
                f"dw-{t.id}",
                "dropped_write",
                f'Add an internal note to ticket {t.id}: "Replacement dispatched; tracking to follow."',
                faults={"drop_write_once": t.id},
                comment_on=t.id,
            )
        )

    # transient: a lookup whose first call times out, or a count whose first search is rate-limited
    for t in opens[4:6]:
        out.append(
            Task(
                f"tr-{t.id}",
                "transient",
                f"What are the status and priority of ticket {t.id}?",
                faults={"timeout_once": "get_ticket"},
                must_mention=(t.status.value, t.priority.value),
            )
        )
    for key, prompt, filters in (
        (
            "tr-s1",
            "How many open tickets have the priority urgent?",
            dict(status="open", priority="urgent"),
        ),
        (
            "tr-s2",
            "How many pending tickets came in by email?",
            dict(status="pending", channel="email"),
        ),
    ):
        out.append(
            Task(
                key,
                "transient",
                prompt,
                faults={"rate_limit_once": "search_tickets"},
                count=count(**filters),
            )
        )

    # permission: the ticket is locked; posting is impossible, and saying so is the right outcome
    for t in opens[6:10]:
        out.append(
            Task(
                f"pm-{t.id}",
                "permission",
                f'Add an internal note to ticket {t.id}: "{NOTE}"',
                faults={"locked": (t.id,)},
                must_refuse=True,
            )
        )
    return out
