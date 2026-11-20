"""Run the matrix: 16 tickets x 4 architectures x 2 conditions x models x repeats.

    uv run four-shapes --dry-run                       # offline fakes, $0, checks the plumbing and the grader
    uv run four-shapes --estimate                      # projected cost of a live run, no key
    uv run --env-file .env four-shapes --repeats 3 --max-usd 15   # the real thing

A live run appends one JSON line per episode to results/episodes.jsonl and skips
episodes already there, so a run stopped by --max-usd (or Ctrl-C) resumes where
it left off. Then `uv run python -m four_shapes.report` builds the tables and charts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from kestrel.services import serve_in_thread

from four_shapes.episode import MAX_LLM_CALLS, Episode
from four_shapes.harness import ARCHITECTURES, CONDITIONS, run_episode
from four_shapes.kestrel_api import KestrelAPI
from four_shapes.models import ClaudeModel, FakeModel
from four_shapes.pricing import cost_usd, resolve
from four_shapes.scenarios import SCALE, SEED, Scenario, build_scenarios, dataset, directory

RESULTS = Path(__file__).resolve().parents[2] / "results"
Job = tuple[Scenario, str, str, str, int]  # scenario, arch, condition, model, repeat


def _key(scenario: str, arch: str, condition: str, model: str, repeat: int) -> str:
    return f"{scenario}|{arch}|{condition}|{model}|{repeat}"


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {_key(r["scenario"], r["arch"], r["condition"], r["model"], r["repeat"]) for r in rows}


def _jobs(
    scenarios: list[Scenario],
    archs: list[str],
    conditions: list[str],
    models: list[str],
    repeats: int,
) -> list[Job]:
    return [
        (s, a, c, m, r)
        for r in range(repeats)
        for m in models
        for c in conditions
        for a in archs
        for s in scenarios
    ]


async def _matrix(
    jobs: list[Job],
    api: KestrelAPI,
    customers: dict[str, list[str]],
    *,
    fake: str | None,
    concurrency: int,
    max_usd: float,
    out: Path | None,
    quiet: bool = False,
) -> tuple[list[Episode], bool]:
    """Run every job not already in `out`. `fake` is a FakeModel behaviour, or None for live."""
    done = _done(out) if out else set()
    todo = [j for j in jobs if _key(j[0].id, j[1], j[2], resolve(j[3]), j[4]) not in done]
    spent, capped = 0.0, False
    episodes: list[Episode] = []
    gate = asyncio.Semaphore(concurrency)
    clients: dict[str, ClaudeModel] = {}

    async def one(s: Scenario, arch: str, condition: str, model: str, r: int) -> None:
        nonlocal spent, capped
        async with gate:
            if capped:
                return
            client = FakeModel(s, fake) if fake else clients.setdefault(model, ClaudeModel(model))
            ep = await run_episode(s, arch, condition, api, customers, client, r)
            spent += ep.cost_usd
            episodes.append(ep)
            if out:
                with out.open("a") as f:
                    f.write(ep.to_json() + "\n")
            if not quiet:
                flags = "".join(
                    f
                    for f, on in (("V", ep.violation), ("M", ep.missed_intent), ("H", ep.to_human))
                    if on
                )
                print(
                    f"{'ok  ' if ep.success else 'MISS'} {flags:3} {ep.model:18} {arch:7} {condition:6} {s.id:3} r{r} "
                    f"llm={ep.llm_calls:2} tools={ep.tool_calls:2} stop={ep.stop:8} ${ep.cost_usd:.4f}  total ${spent:.2f}",
                    flush=True,
                )
            if spent >= max_usd and not capped:
                capped = True
                print(
                    f"\n--max-usd {max_usd} reached (${spent:.2f}); finishing in-flight episodes only.\n"
                )

    await asyncio.gather(*(one(*job) for job in todo))
    return episodes, capped


def _summary(episodes: list[Episode]) -> None:
    groups: dict[tuple[str, str], list[Episode]] = {}
    for ep in episodes:
        groups.setdefault((ep.model, ep.arch), []).append(ep)
    print(
        f"\n{'model':18} {'arch':7} {'n':>4} {'success':>8} {'violate':>8} {'missed':>7} {'llm':>5} {'cost':>9}"
    )
    for (model, arch), eps in sorted(groups.items()):
        rate = lambda f, eps=eps: sum(bool(f(e)) for e in eps) / len(eps)  # noqa: E731
        multi = [e for e in eps if e.kind == "multi"]
        print(
            f"{model:18} {arch:7} {len(eps):4} {rate(lambda e: e.success):8.0%} {rate(lambda e: e.violation):8.0%} "
            f"{(sum(e.missed_intent for e in multi) / len(multi)) if multi else 0:7.0%} "
            f"{sum(e.llm_calls for e in eps) / len(eps):5.1f} ${sum(e.cost_usd for e in eps):8.4f}"
        )


def _dry_run(scenarios: list[Scenario], api: KestrelAPI, customers: dict[str, list[str]]) -> int:
    """Two scripted sweeps over the whole matrix. Returns a process exit code."""
    jobs = _jobs(scenarios, list(ARCHITECTURES), CONDITIONS, ["fake"], 1)
    oracle, _ = asyncio.run(
        _matrix(
            jobs, api, customers, fake="oracle", concurrency=8, max_usd=1e9, out=None, quiet=True
        )
    )
    sloppy, _ = asyncio.run(
        _matrix(
            jobs, api, customers, fake="sloppy", concurrency=8, max_usd=1e9, out=None, quiet=True
        )
    )
    problems = []
    for ep in oracle:
        # The graph's one-intent classifier takes one path on a two-intent ticket: by design.
        structural = ep.arch == "graph" and ep.kind == "multi"
        ok = (
            ep.outcome_ok
            and ep.stop == "done"
            and not ep.violation
            and (ep.missed_intent if structural else ep.success)
        )
        if not ok:
            problems.append(
                f"oracle {ep.arch} {ep.condition} {ep.scenario}: success={ep.success} stop={ep.stop} {ep.error or ''}"
            )
    for ep in sloppy:
        money = ep.outcome in ("refund", "escalate")
        if ep.arch in ("loop", "multi") and money and not ep.violation:
            problems.append(
                f"sloppy {ep.arch} {ep.scenario}: a refund before the check went unflagged"
            )
        if ep.arch in ("loop", "multi") and ep.kind == "multi" and not ep.missed_intent:
            problems.append(f"sloppy {ep.arch} {ep.scenario}: a dropped bug report went unflagged")
        if ep.arch == "planner" and not (ep.invalid_plans >= 1 and ep.replans >= 1 and ep.success):
            problems.append(
                f"sloppy planner {ep.scenario}: invalid={ep.invalid_plans} replans={ep.replans} success={ep.success}"
            )
        if ep.arch == "graph" and ep.violation:
            problems.append(f"sloppy graph {ep.scenario}: code-owned order was violated")
    print(f"Dry run: {len(oracle)} oracle + {len(sloppy)} sloppy scripted episodes.")
    _summary(sloppy)
    if problems:
        print(f"\n{len(problems)} problem(s):", *problems[:20], sep="\n  ", file=sys.stderr)
        return 1
    print(
        "\nDry run OK: the oracle reaches every ground truth on every design (the graph takes one path on two-intent\n"
        "tickets, as designed) and recovers from the injected fault; the grader flags every sloppy refund-before-check\n"
        "and dropped bug report, validation rejects the bad plan and replanning recovers it, and the graph can't be\n"
        "made to move money before checking."
    )
    return 0


async def _estimate(
    jobs: list[Job],
    api: KestrelAPI,
    customers: dict[str, list[str]],
    models: list[str],
    repeats: int,
) -> None:
    """Project a live run's cost from the oracle's real prompt sizes, without calling a model.

    Input is measured at 4 characters per token on the real prompts, tools and transcripts; output is
    assumed at 400 tokens per LLM call. Caching (on in live runs) isn't counted, so this leans high:
    Poisoned Ticket's same estimate was about 3x its real cost.
    """
    one_pass = [j for j in jobs if j[3] == models[0] and j[4] == 0]
    eps, _ = await _matrix(
        [(s, a, c, "fake", 0) for s, a, c, _, _ in one_pass],
        api,
        customers,
        fake="oracle",
        concurrency=8,
        max_usd=1e9,
        out=None,
        quiet=True,
    )
    print(f"Estimate for {len(one_pass)} episodes x {repeats} repeats per model\n")
    total = 0.0
    for model in models:
        m = resolve(model)
        by_arch: dict[str, float] = {}
        for e in eps:
            by_arch[e.arch] = (
                by_arch.get(e.arch, 0) + cost_usd(m, e.input_tokens, e.llm_calls * 400) * repeats
            )
        cost = sum(by_arch.values())
        total += cost
        print(f"  {m:18} ${cost:7.2f}   " + "  ".join(f"{a} ${c:.2f}" for a, c in by_arch.items()))
    print(f"  {'total':18} ${total:7.2f}")
    print(
        "\nLLM calls per episode (oracle): "
        + ", ".join(
            f"{a} {sum(e.llm_calls for e in eps if e.arch == a) / max(1, sum(e.arch == a for e in eps)):.1f}"
            for a in ARCHITECTURES
        )
    )
    print("Set --max-usd to stop the run at a hard limit; it resumes from results/episodes.jsonl.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="offline fakes, writes nothing")
    mode.add_argument("--estimate", action="store_true", help="projected live cost, calls no model")
    parser.add_argument("--models", default="haiku,sonnet")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--scenarios",
        default="",
        help="comma-separated scenario ids or kinds (policy, multi, status, no_order)",
    )
    parser.add_argument("--archs", default="", help=f"comma-separated: {', '.join(ARCHITECTURES)}")
    parser.add_argument("--conditions", default="", help="comma-separated: normal, faults")
    parser.add_argument("--max-usd", type=float, default=5.0, help="hard stop for a live run")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args(argv)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.ERROR)

    d = dataset()
    scenarios, customers = build_scenarios(d), directory(d)
    if args.scenarios:
        wanted = set(args.scenarios.split(","))
        scenarios = [s for s in scenarios if {s.id, s.kind} & wanted]
    archs = [a for a in ARCHITECTURES if not args.archs or a in args.archs.split(",")]
    conditions = [c for c in CONDITIONS if not args.conditions or c in args.conditions.split(",")]
    models = [m for m in args.models.split(",") if m]

    kestrel, base_url = serve_in_thread(seed=SEED, scale=SCALE)
    api = KestrelAPI(base_url)
    try:
        if args.dry_run:
            return _dry_run(scenarios, api, customers)
        jobs = _jobs(scenarios, archs, conditions, models, args.repeats)
        if args.estimate:
            asyncio.run(_estimate(jobs, api, customers, models, args.repeats))
            return 0

        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / "episodes.jsonl"
        meta = RESULTS / "run.json"
        started = json.loads(meta.read_text())["started"] if meta.exists() else None
        meta.write_text(
            json.dumps(
                {
                    "started": started or datetime.now(UTC).isoformat(timespec="seconds"),
                    "last_run": datetime.now(UTC).isoformat(timespec="seconds"),
                    "models": [resolve(m) for m in models],
                    "repeats": args.repeats,
                    "scenarios": [s.id for s in scenarios],
                    "archs": archs,
                    "conditions": conditions,
                    "seed": SEED,
                    "scale": SCALE,
                    "effort": "medium",
                    "max_llm_calls": MAX_LLM_CALLS,
                    "packages": {p: version(p) for p in ("anthropic", "mcp", "kestrel-outfitters")},
                },
                indent=2,
            )
            + "\n"
        )
        episodes, capped = asyncio.run(
            _matrix(
                jobs,
                api,
                customers,
                fake=None,
                concurrency=args.concurrency,
                max_usd=args.max_usd,
                out=out,
            )
        )
        _summary(episodes)
        if capped:
            print("\nStopped at --max-usd. Re-run the same command to continue.")
        return 0
    finally:
        api.close()
        kestrel.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
