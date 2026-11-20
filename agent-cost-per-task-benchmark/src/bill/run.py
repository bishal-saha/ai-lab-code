"""Run the experiment: 24 tickets x 7 conditions x repeats, each ticket worked alone.

    uv run bill-for-one-ticket --dry-run                       # a fake that bills like the API; $0
    uv run bill-for-one-ticket --estimate                      # projected cost
    uv run --env-file .env bill-for-one-ticket --cache-min-check   # a few cents
    uv run --env-file .env bill-for-one-ticket --pilot --max-usd 2
    uv run --env-file .env bill-for-one-ticket --max-usd 12     # resumes from results/tickets.jsonl

Then `uv run python -m bill.report` builds the tables and charts.

One job is one ticket under one condition: a first attempt, and a second when the first failed (the
article's max_attempts = 2). A handoff to a person is final. A ticket that fails twice is abandoned.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from bill.agent import Budget, BudgetExceeded, attempt
from bill.conditions import CONDITIONS, HAIKU, SONNET, condition
from bill.models import ClaudeModel, FakeModel
from bill.pricing import cost_usd
from bill.scenarios import build_scenarios, dataset

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
REPEATS = 2  # planned 3; the pilot measured about 2x the estimate, and 3 would not fit a $12 cap
MAX_ATTEMPTS = 2
PILOT_TICKETS = 8


def job_key(row: dict) -> tuple:
    return (row["condition"], row["ticket"], row["repeat"])


async def execute(
    jobs: list[dict], client, max_usd: float, concurrency: int, out: Path | None, quiet=False
) -> tuple[list[dict], float]:
    data = dataset()
    scenarios = build_scenarios(data)
    by_ticket = {s.ticket_id: s for s in scenarios}
    done = (
        {job_key(json.loads(x)) for x in out.read_text().splitlines()}
        if out and out.exists()
        else set()
    )
    budget, gate, rows = Budget(max_usd), asyncio.Semaphore(concurrency), []

    async def one(job: dict) -> None:
        if job_key(job) in done:
            return
        async with gate:
            if budget.spent >= max_usd:
                return
            s, cond = by_ticket[job["ticket"]], condition(job["condition"])
            attempts, tag = [], uuid.uuid4().hex[:8]
            try:
                for _ in range(MAX_ATTEMPTS):
                    a = await attempt(client, cond, s, data, scenarios, budget=budget, job_tag=tag)
                    attempts.append(a)
                    if a.outcome != "failed":
                        break
            except BudgetExceeded:
                return
            except Exception as err:  # reported, not recorded: resume reruns the job
                print(
                    f"  ! {job_key(job)}: {type(err).__name__}: {err}", file=sys.stderr, flush=True
                )
                return
            final = attempts[-1].outcome if attempts[-1].outcome != "failed" else "abandoned"
            row = {
                **job,
                "job_tag": tag,
                "kind": s.kind,
                "expected": s.expected,
                "final": final,
                "n_attempts": len(attempts),
                "cost_usd": round(sum(a.cost_usd for a in attempts), 6),
                "attempts": [
                    {
                        "outcome": a.outcome,
                        "violation": a.violation,
                        "turn_budget_hit": a.turn_budget_hit,
                        "cost_usd": round(a.cost_usd, 6),
                        "calls": a.calls,
                        "summaries": a.summaries,
                        "ledger": a.ledger,
                    }
                    for a in attempts
                ],
            }
            rows.append(row)
            if out:
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            if not quiet:
                print(
                    f"  {job['condition']:12} {job['ticket']} r{job['repeat']}  {final:9} attempts {len(attempts)}  "
                    f"${row['cost_usd']:.4f}  (total ${budget.spent:.2f})",
                    flush=True,
                )

    await asyncio.gather(*(one(j) for j in jobs))
    return rows, budget.spent


def jobs(tickets: int, repeats: int, conditions: list[str] = CONDITIONS) -> list[dict]:
    scenarios = build_scenarios()[:tickets]
    return [
        {"condition": c, "ticket": s.ticket_id, "repeat": r}
        for r in range(repeats)
        for c in conditions
        for s in scenarios
    ]


def recompute(row: dict) -> float:
    """A row's cost from the reported usage alone, to check the accounting."""
    total = 0.0
    for a in row["attempts"]:
        for c in a["calls"]:
            total += cost_usd(
                c["model"], c["input_tokens"], c["output_tokens"], c["cache_read"], c["cache_write"]
            )
        for s in a["summaries"]:
            total += cost_usd(
                HAIKU, s["input_tokens"], s["output_tokens"], s["cache_read"], s["cache_write"]
            )
    return total


def dry_run() -> int:
    scenarios = build_scenarios()
    problems, mean_cost = [], {}
    for c in CONDITIONS:
        # a fresh fake per condition, so one condition's cache doesn't warm another's
        rows, _ = asyncio.run(
            execute(jobs(24, 1, [c]), FakeModel(scenarios), 1e9, 1, None, quiet=True)
        )
        if any(r["final"] != "resolved" for r in rows):
            problems.append(f"{c}: the fake didn't resolve every ticket")
        if any(abs(recompute(r) - r["cost_usd"]) > 1e-6 for r in rows):
            problems.append(f"{c}: recorded cost differs from the usage")
        mean_cost[c] = sum(r["cost_usd"] for r in rows) / len(rows)
        calls = [x for r in rows for a in r["attempts"] for x in a["calls"]]
        reads = sum(x["cache_read"] for x in calls) / max(1, sum(x["window_tokens"] for x in calls))
        models = {x["model"] for x in calls}
        print(
            f"  {c:12} ${mean_cost[c]:.5f} per ticket   cache hit {reads:5.1%}   models {sorted(m[7:12] for m in models)}"
        )
    expect = [
        ("no-cache", ">", "cache"),
        ("cache-break", ">", "cache"),
        ("haiku", "<", "cache"),
        ("phase", "<", "cache"),
    ]
    for a, op, b in expect:
        if not (mean_cost[a] > mean_cost[b] if op == ">" else mean_cost[a] < mean_cost[b]):
            problems.append(f"expected {a} {op} {b}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(
        "\nDry run OK: every cost recomputes from the usage, the fake resolves every ticket under every condition,\n"
        "and the cache simulator bills no-cache and cache-break above the cached baseline, Haiku and phase routing below."
    )
    return 0


def estimate() -> None:
    """Window sizes from the fake on the real tools, x1.3 for the model's own text; the measured
    cache behaviour of the fake's simulator; 150 output tokens per call (experiment 7 measured about 130)."""
    scenarios = build_scenarios()
    total = 0.0
    for c in CONDITIONS:
        rows, _ = asyncio.run(
            execute(jobs(24, 1, [c]), FakeModel(scenarios), 1e9, 1, None, quiet=True)
        )
        cost = 0.0
        for r in rows:
            for a in r["attempts"]:
                for x in a["calls"]:
                    cost += cost_usd(
                        x["model"],
                        int(x["input_tokens"] * 1.3),
                        150,
                        int(x["cache_read"] * 1.3),
                        int(x["cache_write"] * 1.3),
                    )
                for s in a["summaries"]:
                    cost += cost_usd(HAIKU, s["input_tokens"], 400)
        total += cost * REPEATS
        print(f"  {c:12} ${cost * REPEATS:6.2f}   (${cost / len(rows):.4f} per ticket)")
    print(
        f"  {'total':12} ${total:6.2f}  (24 tickets x 7 conditions x {REPEATS} repeats, first attempts only)"
    )


async def cache_min_check() -> dict:
    """Send a short and a long system prompt twice each: does the short one silently skip caching?"""
    client = ClaudeModel()
    short = "You are a helpful support agent for an outdoor-gear shop. " * 20  # about 250 tokens
    long_ = (
        "You are a helpful support agent for an outdoor-gear shop. Follow the returns policy exactly. "
        * 250
    )  # about 5k
    out = {}
    for model in (HAIKU, SONNET):
        for name, text in (("short", short), ("long", long_)):
            first = await client.count_cache(model=model, text=text)
            second = await client.count_cache(model=model, text=text)
            out[f"{model}:{name}"] = {"first": first, "second": second}
            print(
                f"  {model:18} {name:5}  1st: write {first['cache_write']:5} read {first['cache_read']:5}   "
                f"2nd: write {second['cache_write']:5} read {second['cache_read']:5}   uncached input {second['input_tokens']}"
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--estimate", action="store_true")
    mode.add_argument("--cache-min-check", action="store_true")
    mode.add_argument(
        "--pilot", action="store_true", help=f"{PILOT_TICKETS} tickets x 7 conditions, 1 repeat"
    )
    parser.add_argument("--max-usd", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)

    if args.dry_run:
        return dry_run()
    if args.estimate:
        estimate()
        return 0
    RESULTS.mkdir(exist_ok=True)
    if args.cache_min_check:
        res = asyncio.run(cache_min_check())
        (RESULTS / "cache_min_check.json").write_text(json.dumps(res, indent=2) + "\n")
        return 0
    job_list = jobs(PILOT_TICKETS, 1) if args.pilot else jobs(24, REPEATS)
    rows, spent = asyncio.run(
        execute(job_list, ClaudeModel(), args.max_usd, args.concurrency, RESULTS / "tickets.jsonl")
    )
    print(f"{len(rows)} tickets this session, ${spent:.3f}")
    (RESULTS / "run.json").write_text(
        json.dumps(
            {
                "last_run": datetime.now(UTC).isoformat(timespec="seconds"),
                "models": [SONNET, HAIKU],
                "effort": "medium",
                "conditions": CONDITIONS,
                "repeats": REPEATS,
                "max_attempts": MAX_ATTEMPTS,
                "packages": {p: version(p) for p in ("anthropic", "kestrel-outfitters")},
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
