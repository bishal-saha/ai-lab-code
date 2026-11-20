"""Run the matrix: scenarios x (clean + 12 attacks) x 5 defences x models x repeats.

    uv run poisoned-ticket --dry-run                       # offline fakes, $0, checks the plumbing
    uv run poisoned-ticket --estimate                      # projected cost of a live run, no key
    uv run poisoned-ticket --models haiku,sonnet --repeats 2 --max-usd 15   # the real thing

A live run appends one JSON line per episode to results/episodes.jsonl and skips
episodes already there, so a run stopped by --max-usd (or Ctrl-C) resumes where
it left off. Then `uv run python -m poisoned_ticket.report` builds the tables and charts.
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

from poisoned_ticket.agent import MAX_TURNS, Episode, run_episode
from poisoned_ticket.attacks import ATTACKS, Attack
from poisoned_ticket.defences import DEFENCES, Defence
from poisoned_ticket.kestrel_api import KestrelAPI
from poisoned_ticket.models import ClaudeModel, FakeModel
from poisoned_ticket.pricing import cost_usd, resolve
from poisoned_ticket.scenarios import SCALE, SEED, Scenario, build_scenarios

RESULTS = Path(__file__).resolve().parents[2] / "results"
Job = tuple[Scenario, Attack | None, Defence, str, int]


def _key(scenario: str, attack: str, defence: str, model: str, repeat: int) -> str:
    return f"{scenario}|{attack}|{defence}|{model}|{repeat}"


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {_key(r["scenario"], r["attack"], r["defence"], r["model"], r["repeat"]) for r in rows}


def _jobs(
    scenarios: list[Scenario],
    attacks: list[Attack | None],
    defences: list[Defence],
    models: list[str],
    repeats: int,
) -> list[Job]:
    return [
        (s, a, d, m, r)
        for r in range(repeats)
        for m in models
        for d in defences
        for s in scenarios
        for a in attacks
    ]


async def _matrix(
    jobs: list[Job],
    api: KestrelAPI,
    *,
    fake: str | None,
    concurrency: int,
    max_usd: float,
    out: Path | None,
    quiet: bool = False,
) -> tuple[list[Episode], bool]:
    """Run every job not already in `out`. `fake` is a FakeModel behaviour, or None for live."""
    done = _done(out) if out else set()
    todo = [
        j
        for j in jobs
        if _key(j[0].id, j[1].id if j[1] else "clean", j[2].id, resolve(j[3]), j[4]) not in done
    ]
    spent = 0.0
    capped = False
    episodes: list[Episode] = []
    gate = asyncio.Semaphore(concurrency)
    clients: dict[str, ClaudeModel] = {}

    async def one(s: Scenario, attack: Attack | None, defence: Defence, model: str, r: int) -> None:
        nonlocal spent, capped
        async with gate:
            if capped:
                return
            client = (
                FakeModel(s, attack, fake)
                if fake
                else clients.setdefault(model, ClaudeModel(model))
            )
            ep = await run_episode(s, attack, defence, api, client, r)
            spent += ep.cost_usd
            episodes.append(ep)
            if out:
                with out.open("a") as f:
                    f.write(ep.to_json() + "\n")
            if not quiet:
                verdict = (
                    "clean   "
                    if attack is None
                    else ("BREACHED" if ep.attack_success else "held    ")
                )
                print(
                    f"{verdict} util={'y' if ep.utility else 'N'} {ep.model:18} {defence.id} {s.id} "
                    f"{ep.attack:22} r{r} turns={ep.turns:2} stop={ep.stop:8} ${ep.cost_usd:.4f}  total ${spent:.2f}",
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
        groups.setdefault((ep.model, ep.defence), []).append(ep)
    print(
        f"\n{'model':18} {'def':4} {'n':>5} {'attack ok':>10} {'util clean':>11} {'util atk':>9} {'flagged':>8} {'cost':>9}"
    )
    for (model, defence), eps in sorted(groups.items()):
        atk = [e for e in eps if e.attack != "clean"]
        clean = [e for e in eps if e.attack == "clean"]
        rate = lambda xs, f: (sum(f(x) for x in xs) / len(xs)) if xs else float("nan")  # noqa: E731
        print(
            f"{model:18} {defence:4} {len(eps):5} {rate(atk, lambda e: e.attack_success):10.0%} "
            f"{rate(clean, lambda e: e.utility):11.0%} {rate(atk, lambda e: e.utility):9.0%} "
            f"{rate(atk, lambda e: e.flagged):8.0%} ${sum(e.cost_usd for e in eps):8.4f}"
        )


def _dry_run(scenarios: list[Scenario], attacks: list[Attack | None], api: KestrelAPI) -> int:
    """Two scripted sweeps over the whole matrix. Returns a process exit code."""
    jobs = _jobs(scenarios, attacks, DEFENCES, ["fake"], 1)
    resist, _ = asyncio.run(
        _matrix(jobs, api, fake="resist", concurrency=8, max_usd=1e9, out=None, quiet=True)
    )
    obey, _ = asyncio.run(
        _matrix(jobs, api, fake="obey", concurrency=8, max_usd=1e9, out=None, quiet=True)
    )
    problems = []
    for ep in resist:
        if not ep.utility or ep.attack_success or any(ep.harm.values()) or ep.stop != "done":
            problems.append(
                f"resist {ep.scenario} {ep.attack} {ep.defence}: util={ep.utility} harm={ep.harm} stop={ep.stop}"
            )
    for ep in obey:
        if ep.attack == "clean":
            continue
        enforced = ep.defence in ("D3", "D4")
        # Server-side enforcement must stop the out-of-policy goals and only those.
        expected = ep.goal == "goodwill" or not enforced
        if ep.attack_success != expected or not ep.utility:
            problems.append(
                f"obey {ep.scenario} {ep.attack} {ep.defence}: success={ep.attack_success} expected={expected} util={ep.utility}"
            )
        if enforced and ep.goal != "goodwill" and ep.refused != 1:
            problems.append(
                f"obey {ep.scenario} {ep.attack} {ep.defence}: refused={ep.refused}, expected 1"
            )
    print(f"Dry run: {len(resist)} resisting + {len(obey)} obeying scripted episodes.")
    _summary(obey)
    if problems:
        print(f"\n{len(problems)} problem(s):", *problems[:20], sep="\n  ", file=sys.stderr)
        return 1
    print(
        "\nDry run OK: a resisting agent is never graded breached and always completes the ticket; an obeying\n"
        "agent is breached on every attack without enforcement, and D3/D4 refuse exactly the out-of-policy goals."
    )
    return 0


async def _estimate(jobs: list[Job], api: KestrelAPI, models: list[str], repeats: int) -> None:
    """Project a live run's cost from real prompt sizes, without calling a model.

    The scripted agent makes 4 requests per episode (ticket, eligibility, action,
    summary); its input is measured at 4 characters per token on the real system
    prompt, tool list and tool results. "Typical" is that with ~400 output tokens
    per request; "ceiling" assumes all 12 turns with the transcript growing.
    Caching (on in live runs) isn't counted, so both lean high.
    """
    one_pass = [j for j in jobs if j[3] == models[0] and j[4] == 0]
    eps, _ = await _matrix(
        [(s, a, d, "fake", 0) for s, a, d, _, _ in one_pass],
        api,
        fake="resist",
        concurrency=8,
        max_usd=1e9,
        out=None,
        quiet=True,
    )
    print(f"Estimate for {len(one_pass)} episodes x {repeats} repeats per model\n")
    total_typ = total_max = 0.0
    for model in models:
        m = resolve(model)
        typ = sum(cost_usd(m, e.input_tokens, 4 * 400) for e in eps) * repeats
        ceil = (
            sum(
                cost_usd(m, e.input_tokens * MAX_TURNS * MAX_TURNS // 16, MAX_TURNS * 600)
                for e in eps
            )
            * repeats
        )
        total_typ += typ
        total_max += ceil
        print(f"  {m:18} typical ${typ:7.2f}   ceiling ${ceil:7.2f}")
    print(f"  {'total':18} typical ${total_typ:7.2f}   ceiling ${total_max:7.2f}")
    print(
        "\nSet --max-usd to stop the run at a hard limit; it resumes from results/episodes.jsonl."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="offline fakes, writes nothing")
    mode.add_argument("--estimate", action="store_true", help="projected live cost, calls no model")
    parser.add_argument("--models", default="haiku,sonnet")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--scenarios", default="", help="comma-separated scenario ids (default: all)"
    )
    parser.add_argument(
        "--attacks", default="", help="comma-separated attack ids, goals, styles or 'clean'"
    )
    parser.add_argument("--defences", default="", help="comma-separated defence ids (default: all)")
    parser.add_argument("--max-usd", type=float, default=5.0, help="hard stop for a live run")
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args(argv)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.ERROR)

    scenarios = build_scenarios()
    if args.scenarios:
        scenarios = [s for s in scenarios if s.id in args.scenarios.split(",")]
    attacks: list[Attack | None] = [None, *ATTACKS]
    if args.attacks:
        wanted = set(args.attacks.split(","))
        attacks = [
            a
            for a in attacks
            if (a is None and "clean" in wanted) or (a and ({a.id, a.goal, a.style} & wanted))
        ]
    defences = [d for d in DEFENCES if not args.defences or d.id in args.defences.split(",")]
    models = [m for m in args.models.split(",") if m]

    kestrel, base_url = serve_in_thread(seed=SEED, scale=SCALE)
    api = KestrelAPI(base_url)
    try:
        if args.dry_run:
            return _dry_run(scenarios, attacks, api)
        jobs = _jobs(scenarios, attacks, defences, models, args.repeats)
        if args.estimate:
            asyncio.run(_estimate(jobs, api, models, args.repeats))
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
                    "scenarios": len(scenarios),
                    "attacks": [a.id if a else "clean" for a in attacks],
                    "defences": [d.id for d in defences],
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
                jobs, api, fake=None, concurrency=args.concurrency, max_usd=args.max_usd, out=out
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
