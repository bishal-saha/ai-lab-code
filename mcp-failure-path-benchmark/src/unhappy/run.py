"""Run the experiment: 24 tasks x 6 server variants x 2 models x repeats.

    uv run unhappy-path --dry-run                       # the literal fake against every variant; $0
    uv run unhappy-path --estimate                      # projected cost
    uv run --env-file .env unhappy-path --pilot --max-usd 1
    uv run --env-file .env unhappy-path --max-usd 8     # resumes from results/episodes.jsonl

Then `uv run python -m unhappy.report` builds the tables and charts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from unhappy.agent import episode
from unhappy.models import ClaudeModel, LiteralFake
from unhappy.pricing import cost_usd, resolve
from unhappy.servers import VARIANTS
from unhappy.tasks import MODES, build_tasks
from unhappy.ticketing import dataset

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
REPEATS = 3

# What the literal fake must score per variant (mode -> tasks solved of 4): each ablation breaks only its own mode.
EXPECTED_FAKE = {
    "quickstart": dict.fromkeys(MODES, 0),
    "production": dict.fromkeys(MODES, 4),
    "prod-no-errors": {**dict.fromkeys(MODES, 4), "not_found": 0, "transient": 0, "permission": 0},
    "prod-no-schema": {**dict.fromkeys(MODES, 4), "enum": 0},
    "prod-no-pages": {**dict.fromkeys(MODES, 4), "large": 0},
    "prod-no-keys": {**dict.fromkeys(MODES, 4), "dropped_write": 0},
}


class BudgetExceeded(Exception):
    pass


class Budget:
    """A cap shared by every concurrent episode, checked before each model call."""

    def __init__(self, max_usd: float):
        self.max_usd, self.spent = max_usd, 0.0

    def check(self) -> None:
        if self.spent >= self.max_usd:
            raise BudgetExceeded(f"cap USD {self.max_usd:.2f} reached")

    def add(self, cost: float) -> None:
        self.spent += cost


def key(r: dict) -> tuple:
    return (r["variant"], r["task"], r["model"], r["repeat"])


def jobs(models: list[str], repeats: int, task_ids: list[str] | None = None) -> list[dict]:
    tasks = [t for t in build_tasks() if task_ids is None or t.id in task_ids]
    return [
        {"variant": v, "task": t.id, "model": m, "repeat": r}
        for r in range(repeats)
        for m in models
        for v in VARIANTS
        for t in tasks
    ]


async def execute(
    job_list: list[dict], model, max_usd: float, concurrency: int, out: Path | None, quiet=False
) -> tuple[list[dict], float]:
    data, tasks = dataset(), {t.id: t for t in build_tasks()}
    done = (
        {key(json.loads(x)) for x in out.read_text().splitlines()}
        if out and out.exists()
        else set()
    )
    budget, gate, rows = Budget(max_usd), asyncio.Semaphore(concurrency), []

    async def one(job: dict) -> None:
        if key(job) in done:
            return
        async with gate:
            if budget.spent >= max_usd:
                return
            try:
                r = await episode(
                    model, job["model"], job["variant"], tasks[job["task"]], data, budget=budget
                )
            except BudgetExceeded:
                return
            except Exception as err:  # reported, not recorded: resume reruns it
                print(f"  ! {key(job)}: {type(err).__name__}: {err}", file=sys.stderr, flush=True)
                return
            r["repeat"] = job["repeat"]
            rows.append(r)
            if out:
                with out.open("a") as f:
                    f.write(json.dumps(r, default=str) + "\n")
            if not quiet:
                print(
                    f"  {job['variant']:15} {job['task']:12} {job['model'][7:13]} r{job['repeat']}  "
                    f"{'ok ' if r['success'] else 'FAIL'}  ${r['cost_usd']:.4f}  (total ${budget.spent:.2f})",
                    flush=True,
                )

    await asyncio.gather(*(one(j) for j in job_list))
    return rows, budget.spent


def dry_run() -> int:
    rows, _ = asyncio.run(execute(jobs(["fake"], 1), LiteralFake(), 1e9, 8, None, quiet=True))
    got: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        got[r["variant"]][r["mode"]] += r["success"]
    problems = []
    for v in VARIANTS:
        print(f"  {v:15} " + "  ".join(f"{m}:{got[v][m]}/4" for m in MODES))
        if dict(got[v]) | {m: got[v][m] for m in MODES} != EXPECTED_FAKE[v]:
            problems.append(f"{v}: expected {EXPECTED_FAKE[v]}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(
        "\nDry run OK: the literal fake finishes every task on the production server, none on the quickstart,\n"
        "and each ablation breaks exactly the failure mode it removes."
    )
    return 0


def estimate(models: list[str], repeats: int) -> None:
    """From the literal fake's call counts: 3.5k tokens of tools and system per call (cached after the first),
    300 new tokens and 150 output tokens per call."""
    rows, _ = asyncio.run(execute(jobs(["fake"], 1), LiteralFake(), 1e9, 8, None, quiet=True))
    total = 0.0
    for m in models:
        cost = 0.0
        for r in rows:
            for i in range(r["turns"]):
                cost += cost_usd(
                    m, 0, 150, cache_read=3500 + 300 * i if i else 0, cache_write=300 if i else 3500
                )
        cost *= repeats
        total += cost
        print(f"  {m:18} ${cost:6.2f}")
    print(
        f"  {'total':18} ${total:6.2f}  (24 tasks x {len(VARIANTS)} variants x {repeats} repeats per model)"
    )


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("mcp").setLevel(
        logging.CRITICAL
    )  # the SDK logs every tool error; the ledger has them
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--estimate", action="store_true")
    mode.add_argument(
        "--pilot",
        action="store_true",
        help="Haiku, 1 repeat, 2 tasks per failure mode, every variant",
    )
    parser.add_argument("--max-usd", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    if args.dry_run:
        return dry_run()
    if args.estimate:
        estimate(MODELS, REPEATS)
        return 0
    RESULTS.mkdir(exist_ok=True)
    if args.pilot:
        by_mode = defaultdict(list)
        for t in build_tasks():
            by_mode[t.mode].append(t.id)
        job_list = jobs(["claude-haiku-5-5"], 1, [i for ids in by_mode.values() for i in ids[:2]])
    else:
        job_list = jobs(MODELS, REPEATS)
    rows, spent = asyncio.run(
        execute(job_list, ClaudeModel(), args.max_usd, args.concurrency, RESULTS / "episodes.jsonl")
    )
    print(f"{len(rows)} episodes this session, ${spent:.3f}")
    (RESULTS / "run.json").write_text(
        json.dumps(
            {
                "last_run": datetime.now(UTC).isoformat(timespec="seconds"),
                "models": [resolve(m) for m in MODELS],
                "effort": "medium",
                "variants": list(VARIANTS),
                "repeats": REPEATS,
                "packages": {p: version(p) for p in ("anthropic", "mcp", "kestrel-outfitters")},
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
