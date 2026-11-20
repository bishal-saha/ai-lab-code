"""Run the benchmark matrix: tasks x tool designs x models x repeats.

    uv run tool-shape --dry-run                         # offline fake model, $0, checks the plumbing
    uv run tool-shape --estimate                        # projected cost of a live run, no key needed
    uv run tool-shape --models haiku,sonnet --repeats 3 --max-usd 20   # the real thing

A live run appends one JSON line per episode to results/episodes.jsonl and skips
episodes already there, so a run stopped by --max-usd (or Ctrl-C) resumes where
it left off. Then `uv run python -m tool_shape.report` builds the summary and charts.
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
from typing import Any

from kestrel.services import serve_in_thread

from tool_shape.agent import MAX_TURNS, SYSTEM_PROMPT, Episode, mcp_tool_to_anthropic, run_episode
from tool_shape.kestrel_api import KestrelAPI
from tool_shape.models import ClaudeModel, FakeModel
from tool_shape.pricing import cost_usd, resolve
from tool_shape.servers import endpoints, tasks
from tool_shape.tasks import SCALE, SEED, Task, answer_tool, build_tasks

SERVERS = {"endpoints": endpoints.build_server, "tasks": tasks.build_server}
RESULTS = Path(__file__).resolve().parents[2] / "results"


def _key(task_id: str, server: str, model: str, repeat: int) -> str:
    return f"{task_id}|{server}|{model}|{repeat}"


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        keys.add(_key(row["task_id"], row["server"], row["model"], row["repeat"]))
    return keys


async def _matrix(
    task_list: list[Task],
    servers: dict[str, Any],
    models: list[str],
    repeats: int,
    *,
    fake: bool,
    concurrency: int,
    max_usd: float,
    out: Path | None,
) -> tuple[list[Episode], bool]:
    """Run every episode not already in `out`. Returns (episodes, stopped_by_cap)."""
    done = _done(out) if out else set()
    jobs = [
        (task, server, model, r)
        for r in range(repeats)
        for model in models
        for server in servers
        for task in task_list
        if _key(task.id, server, resolve(model) if not fake else "fake", r) not in done
    ]
    spent = 0.0
    capped = False
    episodes: list[Episode] = []
    gate = asyncio.Semaphore(concurrency)
    clients: dict[str, ClaudeModel] = {}

    async def one(task: Task, server: str, model: str, r: int) -> None:
        nonlocal spent, capped
        async with gate:
            if capped:
                return
            client = (
                FakeModel(task, server) if fake else clients.setdefault(model, ClaudeModel(model))
            )
            ep = await run_episode(task, servers[server], server, client, r)
            spent += ep.cost_usd
            episodes.append(ep)
            if out:
                with out.open("a") as f:
                    f.write(ep.to_json() + "\n")
            mark = "ok " if ep.success else "BAD"
            print(
                f"{mark} {ep.model:18} {server:9} {task.id:11} r{r} turns={ep.turns:2} calls={ep.tool_calls:2} "
                f"stop={ep.stop:9} ${ep.cost_usd:.4f}  total ${spent:.2f}",
                flush=True,
            )
            if spent >= max_usd and not capped:
                capped = True
                print(
                    f"\n--max-usd {max_usd} reached (${spent:.2f}); finishing in-flight episodes only.\n"
                )

    await asyncio.gather(*(one(*job) for job in jobs))
    return episodes, capped


def _summary(episodes: list[Episode]) -> None:
    groups: dict[tuple[str, str], list[Episode]] = {}
    for ep in episodes:
        groups.setdefault((ep.model, ep.server), []).append(ep)
    print(
        f"\n{'model':18} {'server':9} {'n':>4} {'success':>8} {'calls':>6} {'turns':>6} {'cost':>9}"
    )
    for (model, server), eps in sorted(groups.items()):
        n = len(eps)
        print(
            f"{model:18} {server:9} {n:4} {sum(e.success for e in eps) / n:8.0%} "
            f"{sum(e.tool_calls for e in eps) / n:6.1f} {sum(e.turns for e in eps) / n:6.1f} "
            f"${sum(e.cost_usd for e in eps):8.4f}"
        )


async def _tool_specs(servers: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    from mcp import Client

    specs = {}
    for name, server in servers.items():
        async with Client(server) as client:
            specs[name] = [mcp_tool_to_anthropic(t) for t in (await client.list_tools()).tools]
    return specs


async def _estimate(
    task_list: list[Task], servers: dict[str, Any], models: list[str], repeats: int
) -> None:
    """Project a live run's cost from the real prompt sizes, without calling a model.

    Tokens are estimated at 4 characters each. The context per request is the
    system prompt + tool definitions + task, plus the tool results the fake run
    actually received (a real model reads the same results, often more of them).
    "Typical" assumes 4 requests per episode; "ceiling" assumes every episode runs
    all 12 turns with the transcript growing each turn. Prompt caching (on in live
    runs) would cut both; it isn't counted, so the figures lean high.
    """
    specs = await _tool_specs(servers)
    fake_eps = {}
    for server in servers:
        for task in task_list:
            fake_eps[(server, task.id)] = await run_episode(
                task, servers[server], server, FakeModel(task, server)
            )
    print(
        f"Estimate for {len(task_list)} tasks x {len(servers)} designs x {repeats} repeats per model\n"
    )
    grand_typ = grand_max = 0.0
    for model in models:
        m = resolve(model)
        typ = ceil = 0.0
        for server in servers:
            for task in task_list:
                base = (
                    len(SYSTEM_PROMPT)
                    + len(json.dumps(specs[server] + [answer_tool(task)]))
                    + len(task.prompt)
                ) // 4
                fake = fake_eps[(server, task.id)]
                per_turn_growth = max(fake.input_tokens // max(fake.turns, 1) - base, 200)
                typical_in = sum(base + per_turn_growth * k for k in range(4))
                ceiling_in = sum(base + per_turn_growth * k for k in range(MAX_TURNS))
                typ += cost_usd(m, typical_in, 4 * 400)
                ceil += cost_usd(m, ceiling_in, MAX_TURNS * 600)
        typ *= repeats
        ceil *= repeats
        grand_typ += typ
        grand_max += ceil
        print(f"  {m:18} typical ${typ:7.2f}   ceiling ${ceil:7.2f}")
    print(f"  {'total':18} typical ${grand_typ:7.2f}   ceiling ${grand_max:7.2f}")
    print(
        "\nSet --max-usd to stop the run at a hard limit; it resumes from results/episodes.jsonl."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="offline fake model, writes nothing")
    mode.add_argument("--estimate", action="store_true", help="projected live cost, calls no model")
    parser.add_argument("--models", default="haiku,sonnet")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--tasks", default="", help="comma-separated task ids or families (default: all)"
    )
    parser.add_argument("--max-usd", type=float, default=5.0, help="hard stop for a live run")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args(argv)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.ERROR)

    task_list = build_tasks()
    if args.tasks:
        wanted = set(args.tasks.split(","))
        task_list = [t for t in task_list if t.id in wanted or t.family in wanted]
    models = [m for m in args.models.split(",") if m]

    kestrel, base_url = serve_in_thread(seed=SEED, scale=SCALE)
    api = KestrelAPI(base_url)
    servers = {name: build(api) for name, build in SERVERS.items()}
    try:
        if args.estimate:
            asyncio.run(_estimate(task_list, servers, models, args.repeats))
            return 0
        if args.dry_run:
            episodes, _ = asyncio.run(
                _matrix(
                    task_list, servers, ["fake"], 1, fake=True, concurrency=8, max_usd=1e9, out=None
                )
            )
            _summary(episodes)
            failed = [e for e in episodes if not e.success]
            if failed:
                print(
                    f"\n{len(failed)} dry-run episode(s) failed: the plumbing is broken.",
                    file=sys.stderr,
                )
                return 1
            print("\nDry run OK: every scripted episode reached its tools and graded correctly.")
            return 0

        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / "episodes.jsonl"
        meta = RESULTS / "run.json"
        # Keep the first start time across resumes; the rest describes this invocation,
        # so a smoke run on a subset doesn't leave the full matrix mislabelled.
        started = json.loads(meta.read_text())["started"] if meta.exists() else None
        meta.write_text(
            json.dumps(
                {
                    "started": started or datetime.now(UTC).isoformat(timespec="seconds"),
                    "last_run": datetime.now(UTC).isoformat(timespec="seconds"),
                    "models": [resolve(m) for m in models],
                    "repeats": args.repeats,
                    "tasks": len(task_list),
                    "seed": SEED,
                    "scale": SCALE,
                    "effort": "medium",
                    "max_turns": MAX_TURNS,
                    "packages": {p: version(p) for p in ("anthropic", "mcp", "kestrel-outfitters")},
                },
                indent=2,
            )
            + "\n"
        )
        episodes, capped = asyncio.run(
            _matrix(
                task_list,
                servers,
                models,
                args.repeats,
                fake=False,
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
