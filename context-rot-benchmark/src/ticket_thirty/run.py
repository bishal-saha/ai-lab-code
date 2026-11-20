"""Run the experiment: the long-queue study (six context policies) and the tool-count study.

    uv run ticket-thirty --dry-run                       # fakes prove the grader sees forgetting; $0
    uv run ticket-thirty --estimate                      # projected cost of the full matrix
    uv run --env-file .env ticket-thirty --pilot --max-usd 2
    uv run --env-file .env ticket-thirty --max-usd 15    # resumes from results/runs.jsonl

Then `uv run python -m ticket_thirty.report` builds the tables and charts.

    queue study   6 policies x 2 models x 3 queue orders, 24 tickets per run, one window per run
    tools study   24 tickets x 2 tool sets (10 real tools, or 40 with look-alikes) x 2 models x 2 repeats,
                  a fresh window per ticket
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from ticket_thirty.agent import Budget, BudgetExceeded, RunResult, run_queue
from ticket_thirty.models import ClaudeModel, FakeModel, LossySummariser
from ticket_thirty.policies import POLICIES, SUMMARISED
from ticket_thirty.pricing import cost_usd, resolve
from ticket_thirty.queue import ORDERS, queue_order
from ticket_thirty.scenarios import build_scenarios, dataset

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TOOL_SETS = ["core", "distractors"]
TOOL_REPEATS = 2
SUMMARISER_MODEL = "haiku"


def job_key(row: dict) -> tuple:
    return (
        row["study"],
        row["policy"],
        row["model"],
        row["order"],
        row.get("ticket", ""),
        row.get("repeat", 0),
    )


def jobs(models: list[str], orders: int, tool_tickets: int, tool_repeats: int) -> list[dict]:
    out = []
    for m in models:
        for o in range(orders):
            out += [
                {"study": "queue", "policy": p, "model": resolve(m), "order": o} for p in POLICIES
            ]
    scenarios = build_scenarios()
    for m in models:
        for r in range(tool_repeats):
            for ts in TOOL_SETS:
                out += [
                    {
                        "study": "tools",
                        "policy": ts,
                        "model": resolve(m),
                        "order": 0,
                        "ticket": s.ticket_id,
                        "repeat": r,
                    }
                    for s in scenarios[:tool_tickets]
                ]
    return out


def to_row(job: dict, r: RunResult) -> dict:
    return {
        **job,
        "cost_usd": round(r.cost_usd, 6),
        "grades": [g.to_dict() for g in r.grades],
        "calls": r.calls,
        "summaries": r.summaries,
        "ledger": r.ledger,
    }


async def execute(
    job_list: list[dict],
    make_model,
    summariser,
    max_usd: float,
    concurrency: int,
    out: Path | None,
    quiet=False,
) -> tuple[list[dict], float]:
    data = dataset()
    scenarios = build_scenarios(data)
    by_ticket = {s.ticket_id: s for s in scenarios}
    done = (
        {job_key(json.loads(line)) for line in out.read_text().splitlines()}
        if out and out.exists()
        else set()
    )
    budget = Budget(max_usd)
    gate = asyncio.Semaphore(concurrency)
    rows: list[dict] = []

    async def one(job: dict) -> None:
        if job_key(job) in done:
            return
        async with gate:
            if budget.spent >= max_usd:
                return
            if job["study"] == "queue":
                kwargs = {"policy": job["policy"], "queue": queue_order(scenarios, job["order"])}
                kwargs["summariser"] = summariser if job["policy"] in SUMMARISED else None
            else:
                kwargs = {
                    "policy": "fresh",
                    "queue": [by_ticket[job["ticket"]]],
                    "distractors": job["policy"] == "distractors",
                }
            try:
                r = await run_queue(make_model(job["model"]), data=data, budget=budget, **kwargs)
            except BudgetExceeded:
                return
            except Exception as err:  # a failed run is reported, not recorded: resume reruns it
                print(
                    f"  ! {job_key(job)}: {type(err).__name__}: {err}", file=sys.stderr, flush=True
                )
                return
            row = to_row(job, r)
            rows.append(row)
            if out:
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if job["study"] == "queue":
                    tdir = RESULTS / "transcripts"
                    tdir.mkdir(exist_ok=True)
                    name = f"{job['policy'].replace('+', '-')}_{job['model']}_order{job['order']}.json.gz"
                    with gzip.open(tdir / name, "wt") as f:
                        json.dump(
                            r.transcript,
                            f,
                            default=lambda o: (
                                o.model_dump() if hasattr(o, "model_dump") else str(o)
                            ),
                        )
            if not quiet:
                ok = sum(g.success for g in r.grades)
                print(
                    f"  {job['study']:5} {job['policy']:12} {job['model']:18} order {job['order']} {job.get('ticket', ''):6} "
                    f"success {ok}/{len(r.grades)}  ${r.cost_usd:.3f}  (total ${budget.spent:.2f})",
                    flush=True,
                )

    await asyncio.gather(*(one(j) for j in job_list))
    return rows, budget.spent


def dry_run() -> int:
    scenarios = build_scenarios()
    problems = []
    for behaviour in ("oracle", "forgetful"):
        rows, _ = asyncio.run(
            execute(
                jobs(["fake"], 1, 4, 1),
                lambda m, b=behaviour: FakeModel(b, scenarios),
                LossySummariser(),
                1e9,
                16,
                None,
                quiet=True,
            )
        )
        for row in rows:
            if row["study"] != "queue":
                continue
            g = row["grades"]
            ok, frz, aud = (
                sum(x["success"] for x in g),
                sum(x["freeze_violation"] for x in g),
                sum(x["audit_flag"] for x in g),
            )
            print(
                f"  {behaviour:9} {row['policy']:12} success {ok:2}/24  freeze violations {frz}  audit flags {aud:2}/24  "
                f"final window {row['calls'][-1]['window_tokens']:6} tokens"
            )
            forgets = behaviour == "forgetful" and row["policy"] == "summary"
            if forgets and (frz == 0 or aud > 1):
                problems.append(
                    f"forgetful summary: forgetting not detected (freeze violations {frz}, audit {aud})"
                )
            if not forgets and (ok != 24 or frz or aud != 24):
                problems.append(f"{behaviour} {row['policy']}: expected a clean queue")
        tools = [r for r in rows if r["study"] == "tools"]
        if any(g["wrong_tool"] for r in tools for g in r["grades"]):
            problems.append("a fake called a distractor tool")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(
        "\nDry run OK: the oracle keeps both instructions under every policy; the forgetful fake loses them only\n"
        "under the unpinned summary, and the grader catches every escalation and every missing audit flag."
    )
    return 0


def estimate(models: list[str]) -> None:
    """Window sizes from an oracle fake on the real tools, scaled for the model's own text (x1.3)
    plus the system prompt and tool schemas; caching as the loop gets it (each call writes what's
    new and reads the rest); 150 output tokens per call (experiment 3's single loop measured 115-130
    on the same tools at the same effort)."""
    scenarios = build_scenarios()
    rows, _ = asyncio.run(
        execute(
            jobs(["fake"], 1, 24, 1),
            lambda m: FakeModel("oracle", scenarios),
            LossySummariser(),
            1e9,
            16,
            None,
            quiet=True,
        )
    )
    overhead = {"core": 2300, "distractors": 4800}
    total = 0.0
    for m in models:
        name, sub = resolve(m), 0.0
        for row in rows:
            prev, cost = 0, 0.0
            fixed = overhead["distractors" if row["policy"] == "distractors" else "core"]
            for c in row["calls"]:
                w = int(c["window_tokens"] * 1.3)
                new = w - prev if w >= prev else w
                cost += cost_usd(name, 0, 150, cache_read=fixed + (w - new), cache_write=new)
                prev = w
            if row["policy"] in SUMMARISED:
                cost += 23 * cost_usd(resolve(SUMMARISER_MODEL), 3000, 400)
            sub += cost * (ORDERS if row["study"] == "queue" else TOOL_REPEATS)
        total += sub
        print(f"  {name:18} ${sub:6.2f}")
    print(
        f"  {'total':18} ${total:6.2f}  (queue: 6 policies x {ORDERS} orders; tools: 24 tickets x 2 sets x {TOOL_REPEATS} repeats; per model)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--estimate", action="store_true")
    mode.add_argument(
        "--pilot",
        action="store_true",
        help="queue order 0 on every policy, Haiku only, plus 4 tool-study tickets",
    )
    parser.add_argument("--models", default="haiku,sonnet")
    parser.add_argument("--max-usd", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)

    if args.dry_run:
        return dry_run()
    models = [m for m in args.models.split(",") if m]
    if args.estimate:
        estimate(models)
        return 0
    job_list = jobs(["haiku"], 1, 4, 1) if args.pilot else jobs(models, ORDERS, 24, TOOL_REPEATS)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "runs.jsonl"
    summariser = ClaudeModel(SUMMARISER_MODEL, effort="low")
    rows, spent = asyncio.run(
        execute(job_list, lambda m: ClaudeModel(m), summariser, args.max_usd, args.concurrency, out)
    )
    print(f"{len(rows)} runs this session, ${spent:.3f}")
    (RESULTS / "run.json").write_text(
        json.dumps(
            {
                "last_run": datetime.now(UTC).isoformat(timespec="seconds"),
                "models": [resolve(m) for m in models],
                "summariser": resolve(SUMMARISER_MODEL),
                "effort": "medium",
                "policies": POLICIES,
                "packages": {p: version(p) for p in ("anthropic", "kestrel-outfitters")},
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
