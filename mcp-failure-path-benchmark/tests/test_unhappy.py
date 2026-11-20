"""Offline tests: the API and its faults, each server variant's failure path, the tasks, the grader, the dry run."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter

import pytest
from mcp import Client
from unhappy.grading import grade
from unhappy.run import dry_run
from unhappy.servers import VARIANTS, Ledger, build_server
from unhappy.tasks import MODES, build_tasks
from unhappy.ticketing import PAGE_MAX, Comment, Faults, Forbidden, TicketingAPI, Timeout, dataset

logging.getLogger("mcp").setLevel(logging.CRITICAL)


@pytest.fixture(scope="module")
def data():
    return dataset()


def test_api_faults_and_keys(data):
    api = TicketingAPI(
        data, Faults(drop_write_once="T-000009", locked=("T-000060",), timeout_once="get_ticket")
    )
    with pytest.raises(Timeout):
        api.get("T-000009")
    assert api.get("T-000009").id == "T-000009"  # a timeout fires once
    with pytest.raises(Timeout):
        api.add_comment("T-000009", "hi", key="k1")  # the write lands, then times out
    api.add_comment("T-000009", "hi", key="k1")  # the keyed resend is not posted again
    assert len(api.comments) == 1
    with pytest.raises(Forbidden):
        api.add_comment("T-000060", "hi")
    rows, total = api.search(status="open", limit=500)
    assert len(rows) == PAGE_MAX and total == 45


def _call(
    variant: str, faults: Faults, calls: list[tuple[str, dict]], data
) -> tuple[list[tuple[bool, str]], TicketingAPI, dict]:
    api, ledger = TicketingAPI(data, faults), Ledger()

    async def go():
        async with Client(build_server(variant, api, ledger)) as c:
            schema = {t.name: t.input_schema for t in (await c.list_tools()).tools}
            out = []
            for name, args in calls:
                r = await c.call_tool(name, args)
                out.append((bool(r.is_error), " ".join(getattr(p, "text", "") for p in r.content)))
            return out, schema

    out, schema = asyncio.run(go())
    return out, api, schema


def test_quickstart_and_production_failure_paths(data):
    calls = [
        ("search_tickets", {"status": "waiting_on_customer"}),
        ("search_tickets", {"status": "open"}),
        ("get_ticket", {"ticket_id": "T-001042"}),
        ("add_ticket_comment", {"ticket_id": "T-000009", "body": "hi"}),
        ("add_ticket_comment", {"ticket_id": "T-000009", "body": "hi"}),
        ("add_ticket_comment", {"ticket_id": "T-000060", "body": "hi"}),
    ]
    faults = lambda: Faults(drop_write_once="T-000009", locked=("T-000060",))  # noqa: E731
    q, qapi, qschema = _call("quickstart", faults(), calls, data)
    p, papi, pschema = _call("production", faults(), calls, data)
    assert q[0] == (False, "No tickets match.")  # a guessed value matches nothing, silently
    assert (
        p[0][0] and "Input should be" in p[0][1] and "status" in p[0][1]
    )  # the schema names the field
    assert "tickets match" not in q[1][1] and len(q[1][1].splitlines()) == PAGE_MAX
    assert p[1][1].startswith("45 tickets match; showing 1-20.") and "next_cursor" in p[1][1]
    assert q[2][0] and q[2][1].endswith("Request failed.") and "search_tickets" in p[2][1]
    assert len(qapi.comments) == 2 and len(papi.comments) == 1  # keys de-duplicate the resend
    assert "safe" in p[3][1] and "won't help" in p[5][1] and q[5][1].endswith("Request failed.")
    assert "enum" not in str(qschema["search_tickets"]) and "enum" in str(pschema["search_tickets"])


def test_each_ablation_removes_one_practice(data):
    for v, cfg in VARIANTS.items():
        out, api, schema = _call(
            v,
            Faults(drop_write_once="T-000009"),
            [
                ("search_tickets", {"status": "open"}),
                ("add_ticket_comment", {"ticket_id": "T-000009", "body": "hi"}),
                ("add_ticket_comment", {"ticket_id": "T-000009", "body": "hi"}),
                ("get_ticket", {"ticket_id": "T-001042"}),
            ],
            data,
        )
        assert ("tickets match" in out[0][1]) == cfg["pages"]
        assert (len(api.comments) == 1) == cfg["keys"]
        assert (not out[3][1].endswith("Request failed.")) == cfg["actionable"]
        assert ("enum" in str(schema["search_tickets"])) == cfg["strict"]


def test_tasks():
    ts = build_tasks()
    assert len(ts) == 24 and Counter(t.mode for t in ts) == dict.fromkeys(MODES, 4)
    assert all(t.count is not None and t.count < PAGE_MAX for t in ts if t.mode == "enum")
    assert all(t.count is not None and t.count > PAGE_MAX for t in ts if t.mode == "large")
    targets = [t.comment_on for t in ts if t.comment_on] + [
        t.faults["locked"][0] for t in ts if t.mode == "permission"
    ]
    assert len(targets) == len(set(targets))


def test_grader():
    t = next(x for x in build_tasks() if x.mode == "dropped_write")
    one, two = (
        [Comment(t.comment_on, "x", True, None)],
        [Comment(t.comment_on, "x", True, None)] * 2,
    )
    assert grade(t, one, [], {"outcome": "done", "answer": ""})["success"]
    g = grade(t, two, [], {"outcome": "done", "answer": ""})
    assert not g["success"] and g["duplicate"] and g["silent_wrong"]
    e = next(x for x in build_tasks() if x.mode == "enum")
    assert grade(e, [], [], {"outcome": "done", "answer": "", "count": e.count})["success"]
    assert grade(e, [], [], {"outcome": "done", "answer": "", "count": 0})["silent_wrong"]
    p = next(x for x in build_tasks() if x.mode == "permission")
    calls = [{"tool": "add_ticket_comment", "args": {"a": 1}, "is_error": True}] * 3
    g = grade(p, [], calls, {"outcome": "could_not_complete", "answer": ""})
    assert not g["success"] and g["repeats"] == 2 and g["post_attempts"] == 3


def test_dry_run():
    assert dry_run() == 0


def test_a_ticket_shows_its_comments(data):
    """A write can be checked: after a dropped write, get_ticket shows the note that landed."""
    out, api, _ = _call(
        "prod-no-keys",
        Faults(drop_write_once="T-000009"),
        [
            ("add_ticket_comment", {"ticket_id": "T-000009", "body": "Replacement dispatched."}),
            ("get_ticket", {"ticket_id": "T-000009"}),
        ],
        data,
    )
    assert out[0][0] and "Comments:\n- internal note: Replacement dispatched." in out[1][1]
