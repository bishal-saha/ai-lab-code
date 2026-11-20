"""Six real MCP servers over one ticketing API: the same three task-shaped tools, different failure paths.

Every variant has search_tickets, get_ticket and add_ticket_comment (plus the harness's
submit_answer), with the same descriptions. Four practices from the article are switched on or off:

    strict      enums and bounds in the schema; a bad value comes back as an isError result that
                names the field (Python SDK v2 does this before the handler runs). Off: free strings,
                so a guessed value is valid input that matches nothing
    actionable  errors say what happened and what to do next. Off: "Request failed."
    pages       a search says how many tickets match and gives a cursor for the next page. Off: the
                same page with no total and no cursor
    keys        the server derives an idempotency key from ticket and body, and a dropped write says
                a resend is safe. Off: a resend posts again

    quickstart      none of the four
    production      all four
    prod-no-errors, prod-no-schema, prod-no-pages, prod-no-keys: production minus one

The descriptions never list the allowed values: in the strict variants the enum carries them, and
in the free-string variants that is the point. The API's page size is capped at 20 everywhere.
"""

import base64
import hashlib
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from unhappy.ticketing import PAGE_MAX, Forbidden, NotFound, RateLimited, TicketingAPI, Timeout

VARIANTS = {
    "quickstart": dict(strict=False, actionable=False, pages=False, keys=False),
    "production": dict(strict=True, actionable=True, pages=True, keys=True),
    "prod-no-errors": dict(strict=True, actionable=False, pages=True, keys=True),
    "prod-no-schema": dict(strict=False, actionable=True, pages=True, keys=True),
    "prod-no-pages": dict(strict=True, actionable=True, pages=False, keys=True),
    "prod-no-keys": dict(strict=True, actionable=True, pages=True, keys=False),
}

OPAQUE = "Request failed."

SEARCH_DOC = (
    "Search support tickets by status, priority, channel, customer (name or email) or text in the subject and messages. "
    "Use it to find tickets or count them; use get_ticket for one ticket's full message thread. "
    "Returns one line per ticket: id, status, priority, channel, customer, subject and last activity date."
)
GET_DOC = (
    "Read one ticket: status, priority, channel, customer, subject and every message in its thread."
)
COMMENT_DOC = "Add a comment to a ticket. Comments are internal notes (not sent to the customer) unless internal is false."
ANSWER_DOC = (
    "Finish the task: call exactly once. outcome is done if you completed what was asked, or could_not_complete if you "
    "couldn't (say why in answer). Give count when asked how many, and ticket_ids for any tickets you acted on or listed."
)


@dataclass
class Call:
    tool: str
    args: dict[str, Any]
    ok: bool
    error: str = ""  # not_found | timeout | rate_limited | forbidden | write_timeout | invalid


@dataclass
class Ledger:
    calls: list[Call] = field(default_factory=list)
    answer: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {"calls": [c.__dict__ for c in self.calls], "answer": self.answer}


def _cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(f"o:{offset}".encode()).decode().rstrip("=")


def _offset(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        return int(raw.split(":", 1)[1])
    except (ValueError, IndexError):
        raise ToolError(
            "cursor is not valid: pass the next_cursor value from the previous search_tickets result unchanged."
        ) from None


def build_server(variant: str, api: TicketingAPI, ledger: Ledger) -> MCPServer:
    cfg = VARIANTS[variant]
    server = MCPServer(
        "kestrel-tickets", instructions="Kestrel Outfitters' support ticketing system."
    )

    def fail(tool: str, args: dict, kind: str, actionable_text: str) -> None:
        ledger.calls.append(Call(tool, args, False, kind))
        raise ToolError(actionable_text if cfg["actionable"] else OPAQUE)

    def line(t) -> str:
        name, email = api.customer(t)
        return f"{t.id} | {t.status.value} | {t.priority.value} | {t.channel} | {name} <{email}> | {t.subject} | last activity {api.updated(t)}"

    def search(args: dict, *, status, priority, channel, customer, text, limit, cursor) -> str:
        offset = _offset(cursor) if cfg["pages"] else 0
        try:
            rows, total = api.search(
                status=status,
                priority=priority,
                channel=channel,
                customer=customer,
                text=text,
                offset=offset,
                limit=limit,
            )
        except Timeout:
            fail(
                "search_tickets",
                args,
                "timeout",
                "The ticketing API timed out. This is temporary: retrying the same call is safe.",
            )
        except RateLimited:
            fail(
                "search_tickets",
                args,
                "rate_limited",
                "Rate limited by the ticketing API. This is temporary: retry the same call.",
            )
        ledger.calls.append(Call("search_tickets", args, True))
        if not rows:
            return "No tickets match."
        body = "\n".join(line(t) for t in rows)
        if not cfg["pages"]:
            return body
        shown_to = offset + len(rows)
        head = f"{total} tickets match; showing {offset + 1}-{shown_to}."
        if shown_to < total:
            head += (
                f" next_cursor: {_cursor(shown_to)} (pass it as cursor for the next page, "
                "or add filters such as status, priority, channel or customer to narrow the search)."
            )
        return f"{head}\n{body}"

    if cfg["strict"]:
        Status = Literal["open", "pending", "solved"]
        Priority = Literal["low", "normal", "high", "urgent"]
        Channel = Literal["chat", "email", "phone", "web form"]

        if cfg["pages"]:

            @server.tool(description=SEARCH_DOC)
            def search_tickets(
                status: Annotated[Status | None, Field(description="Ticket status.")] = None,
                priority: Annotated[Priority | None, Field(description="Ticket priority.")] = None,
                channel: Annotated[
                    Channel | None, Field(description="How the customer contacted us.")
                ] = None,
                customer: Annotated[
                    str | None, Field(description="Part of the customer's name or email.")
                ] = None,
                text: Annotated[
                    str | None, Field(description="Words to find in the subject or messages.")
                ] = None,
                limit: Annotated[
                    int, Field(ge=1, le=PAGE_MAX, description="Tickets per page.")
                ] = PAGE_MAX,
                cursor: Annotated[
                    str | None, Field(description="next_cursor from the previous page.")
                ] = None,
            ) -> str:
                args = {
                    k: v
                    for k, v in dict(
                        status=status,
                        priority=priority,
                        channel=channel,
                        customer=customer,
                        text=text,
                        limit=limit,
                        cursor=cursor,
                    ).items()
                    if v is not None
                }
                return search(
                    args,
                    status=status,
                    priority=priority,
                    channel=channel,
                    customer=customer,
                    text=text,
                    limit=limit,
                    cursor=cursor,
                )
        else:

            @server.tool(description=SEARCH_DOC)
            def search_tickets(
                status: Annotated[Status | None, Field(description="Ticket status.")] = None,
                priority: Annotated[Priority | None, Field(description="Ticket priority.")] = None,
                channel: Annotated[
                    Channel | None, Field(description="How the customer contacted us.")
                ] = None,
                customer: Annotated[
                    str | None, Field(description="Part of the customer's name or email.")
                ] = None,
                text: Annotated[
                    str | None, Field(description="Words to find in the subject or messages.")
                ] = None,
                limit: Annotated[
                    int, Field(ge=1, le=PAGE_MAX, description="Tickets per page.")
                ] = PAGE_MAX,
            ) -> str:
                args = {
                    k: v
                    for k, v in dict(
                        status=status,
                        priority=priority,
                        channel=channel,
                        customer=customer,
                        text=text,
                        limit=limit,
                    ).items()
                    if v is not None
                }
                return search(
                    args,
                    status=status,
                    priority=priority,
                    channel=channel,
                    customer=customer,
                    text=text,
                    limit=limit,
                    cursor=None,
                )
    else:
        if cfg["pages"]:

            @server.tool(description=SEARCH_DOC)
            def search_tickets(
                status: Annotated[str | None, Field(description="Ticket status.")] = None,
                priority: Annotated[str | None, Field(description="Ticket priority.")] = None,
                channel: Annotated[
                    str | None, Field(description="How the customer contacted us.")
                ] = None,
                customer: Annotated[
                    str | None, Field(description="Part of the customer's name or email.")
                ] = None,
                text: Annotated[
                    str | None, Field(description="Words to find in the subject or messages.")
                ] = None,
                limit: Annotated[int, Field(description="Tickets per page.")] = PAGE_MAX,
                cursor: Annotated[
                    str | None, Field(description="next_cursor from the previous page.")
                ] = None,
            ) -> str:
                args = {
                    k: v
                    for k, v in dict(
                        status=status,
                        priority=priority,
                        channel=channel,
                        customer=customer,
                        text=text,
                        limit=limit,
                        cursor=cursor,
                    ).items()
                    if v is not None
                }
                return search(
                    args,
                    status=status,
                    priority=priority,
                    channel=channel,
                    customer=customer,
                    text=text,
                    limit=limit,
                    cursor=cursor,
                )
        else:

            @server.tool(description=SEARCH_DOC)
            def search_tickets(
                status: Annotated[str | None, Field(description="Ticket status.")] = None,
                priority: Annotated[str | None, Field(description="Ticket priority.")] = None,
                channel: Annotated[
                    str | None, Field(description="How the customer contacted us.")
                ] = None,
                customer: Annotated[
                    str | None, Field(description="Part of the customer's name or email.")
                ] = None,
                text: Annotated[
                    str | None, Field(description="Words to find in the subject or messages.")
                ] = None,
                limit: Annotated[int, Field(description="Tickets per page.")] = PAGE_MAX,
            ) -> str:
                args = {
                    k: v
                    for k, v in dict(
                        status=status,
                        priority=priority,
                        channel=channel,
                        customer=customer,
                        text=text,
                        limit=limit,
                    ).items()
                    if v is not None
                }
                return search(
                    args,
                    status=status,
                    priority=priority,
                    channel=channel,
                    customer=customer,
                    text=text,
                    limit=limit,
                    cursor=None,
                )

    TicketId = Annotated[
        str, Field(description="Ticket id, like T-000123." if cfg["strict"] else "Ticket id.")
    ]

    @server.tool(description=GET_DOC)
    def get_ticket(ticket_id: TicketId) -> str:
        args = {"ticket_id": ticket_id}
        try:
            t = api.get(ticket_id)
        except NotFound:
            fail(
                "get_ticket",
                args,
                "not_found",
                f"Ticket {ticket_id!r} not found. Ticket ids look like T-000123. Use search_tickets to find the ticket by customer name, email or subject.",
            )
        except Timeout:
            fail(
                "get_ticket",
                args,
                "timeout",
                "The ticketing API timed out. This is temporary: retrying the same call is safe.",
            )
        except RateLimited:
            fail(
                "get_ticket",
                args,
                "rate_limited",
                "Rate limited by the ticketing API. This is temporary: retry the same call.",
            )
        ledger.calls.append(Call("get_ticket", args, True))
        msgs = "\n".join(f"- {m.author} on {m.sent_at.date()}: {m.body}" for m in t.messages)
        return line(t) + "\n\nMessages:\n" + msgs

    @server.tool(description=COMMENT_DOC)
    def add_ticket_comment(
        ticket_id: TicketId,
        body: Annotated[str, Field(description="The comment text.")],
        internal: bool = True,
    ) -> str:
        args = {"ticket_id": ticket_id, "body": body, "internal": internal}
        key = (
            hashlib.sha256(f"{ticket_id.strip().upper()}\n{body}".encode()).hexdigest()[:16]
            if cfg["keys"]
            else None
        )
        try:
            c = api.add_comment(ticket_id, body, internal=internal, key=key)
        except NotFound:
            fail(
                "add_ticket_comment",
                args,
                "not_found",
                f"Ticket {ticket_id!r} not found. Ticket ids look like T-000123. Use search_tickets to find the ticket by customer name, email or subject.",
            )
        except Forbidden:
            fail(
                "add_ticket_comment",
                args,
                "forbidden",
                f"Not allowed to comment on {ticket_id}: the ticket is locked for a legal hold. Retrying won't help; tell the user.",
            )
        except Timeout:
            fail(
                "add_ticket_comment",
                args,
                "write_timeout",
                "Comment unconfirmed: the ticketing API timed out after receiving it. Resending the same ticket_id and body is safe; it will not be posted twice."
                if cfg["keys"]
                else "Comment unconfirmed: the ticketing API timed out. The comment may or may not have been posted.",
            )
        ledger.calls.append(Call("add_ticket_comment", args, True))
        return f"Comment added to {c.ticket_id} ({'internal note' if c.internal else 'visible to the customer'})."

    @server.tool(description=ANSWER_DOC)
    def submit_answer(
        outcome: Literal["done", "could_not_complete"],
        answer: Annotated[
            str, Field(description="Your answer or a one-line summary of what you did.")
        ],
        count: int | None = None,
        ticket_ids: list[str] | None = None,
    ) -> str:
        ledger.answer = {
            "outcome": outcome,
            "answer": answer,
            "count": count,
            "ticket_ids": ticket_ids or [],
        }
        return "Recorded."

    return server
