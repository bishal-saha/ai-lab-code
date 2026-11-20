"""Run the judges: every case by every design, every pair in both orders, x models x repeats.

    uv run judge-the-judge --dry-run                     # a perfect and a biased fake; $0
    uv run judge-the-judge --estimate                    # projected cost
    uv run --env-file .env judge-the-judge --pilot --max-usd 1
    uv run --env-file .env judge-the-judge --max-usd 10  # resumes from results/judgements.jsonl

Then `uv run python -m judge_the_judge.report` builds the tables and charts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from judge_the_judge.cases import Case, Pair, build
from judge_the_judge.judges import (
    DESIGNS,
    ClaudeModel,
    FakeModel,
    case_prompt,
    pair_prompt,
    system_prompt,
)
from judge_the_judge.metrics import confusion, majority
from judge_the_judge.pricing import cost_usd, resolve

ROOT = Path(__file__).resolve().parents[2]
REPEATS = 3


def verdict(design: str, value: dict | None) -> str | None:
    """The gate verdict a judgement implies (score >= 7 passes)."""
    if not value:
        return None
    if design == "score":
        return "PASS" if int(value.get("score", 0)) >= 7 else "FAIL"
    return value.get("verdict")


def _key(r: dict) -> tuple:
    return (r["kind"], r["design"], r["item"], r.get("order", ""), r["model"], r["repeat"])


async def run(
    cases: list[Case],
    pairs: list[Pair],
    models,
    repeats: int,
    out: Path | None,
    max_usd: float,
    concurrency: int,
    quiet=False,
) -> tuple[list[dict], float]:
    done = (
        {_key(json.loads(line)) for line in out.read_text().splitlines()}
        if out and out.exists()
        else set()
    )
    spent, rows, gate = 0.0, [], asyncio.Semaphore(concurrency)

    async def one(
        kind: str, design: str, item: str, order: str, model, repeat: int, prompt: str, meta: dict
    ) -> None:
        nonlocal spent
        row = {
            "kind": kind,
            "design": design,
            "item": item,
            "order": order,
            "model": model.name,
            "repeat": repeat,
        }
        if _key(row) in done:
            return
        async with gate:
            if spent >= max_usd:
                return
            try:
                j = await model.judge(design, prompt, {**meta, "repeat": repeat})
                cost = cost_usd(
                    model.name,
                    j.usage["input_tokens"],
                    j.usage["output_tokens"],
                    j.usage.get("cache_read", 0),
                    j.usage.get("cache_write", 0),
                )
                row.update(value=j.value, stop_reason=j.stop_reason, cost_usd=cost, error=None)
            except Exception as err:  # a failed call fails the row, not the run
                row.update(
                    value=None,
                    stop_reason="error",
                    cost_usd=0.0,
                    error=f"{type(err).__name__}: {err}",
                )
            spent += row["cost_usd"]
            rows.append(row)
            if out:
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            if not quiet and len(rows) % 100 == 0:
                print(f"  {len(rows)} judgements, ${spent:.2f}", flush=True)

    jobs = []
    for model in models:
        for r in range(repeats):
            for design in DESIGNS:
                jobs += [
                    one("single", design, c.id, "", model, r, case_prompt(c), {"case": c})
                    for c in cases
                ]
            for p in pairs:
                jobs.append(
                    one(
                        "pair",
                        "pair",
                        p.id,
                        "AB",
                        model,
                        r,
                        pair_prompt(p.a, p.b),
                        {"first": p.a, "second": p.b},
                    )
                )
                jobs.append(
                    one(
                        "pair",
                        "pair",
                        p.id,
                        "BA",
                        model,
                        r,
                        pair_prompt(p.b, p.a),
                        {"first": p.b, "second": p.a},
                    )
                )
    await asyncio.gather(*jobs)
    return rows, spent


def dry_run(cases: list[Case], pairs: list[Pair]) -> int:
    problems = []
    labels = {c.id: c.label for c in cases}
    for behaviour in ("perfect", "biased"):
        rows, _ = asyncio.run(
            run(cases, pairs, [FakeModel(behaviour)], REPEATS, None, 1e9, 64, quiet=True)
        )
        for design in DESIGNS:
            per_case: dict[str, list[str]] = {}
            for r in rows:
                if r["kind"] == "single" and r["design"] == design:
                    per_case.setdefault(r["item"], []).append(verdict(design, r["value"]))
            ids = sorted(per_case)
            cm = confusion([labels[i] for i in ids], [majority(per_case[i]) for i in ids])
            if behaviour == "perfect" and (cm.tpr != 1 or cm.tnr != 1):
                problems.append(f"perfect {design}: TPR {cm.tpr:.2f} TNR {cm.tnr:.2f}")
            if behaviour == "biased" and design == "pf" and cm.tpr > 0.75:
                problems.append(f"biased pf: verbosity bias didn't lower TPR ({cm.tpr:.2f})")
            print(
                f"  {behaviour:7} {design:10} TPR {cm.tpr:.2f}  TNR {cm.tnr:.2f}  agreement {cm.agreement:.2f}  kappa {cm.kappa:.2f}"
            )
        firsts = [
            r
            for r in rows
            if r["kind"] == "pair"
            and r["item"] in {p.id for p in pairs if p.kind == "short_vs_long"}
        ]
        a_rate = sum(r["value"]["winner"] == "A" for r in firsts) / len(firsts)
        print(
            f"  {behaviour:7} pairwise: first position wins {a_rate:.0%} between two correct replies"
        )
        if behaviour == "biased" and a_rate < 0.99:
            problems.append("biased pairwise: position bias not visible")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(
        "\nDry run OK: a perfect judge scores TPR = TNR = 1 on every design; the biased judge's verbosity and position\n"
        "biases show up in the metrics, as they should."
    )
    return 0


def estimate(cases: list[Case], pairs: list[Pair], models, repeats: int) -> None:
    """Characters/4 on the real prompts; 60 output tokens a judgement (120 with a critique). No caching counted."""
    total = 0.0
    for m in models:
        cost = 0.0
        for design in DESIGNS:
            out_tokens = 120 if design == "pf-crit" else 60
            cost += sum(
                cost_usd(
                    m.name, (len(system_prompt(design)) + len(case_prompt(c))) // 4, out_tokens
                )
                for c in cases
            )
        cost += (
            sum(
                cost_usd(m.name, (len(system_prompt("pair")) + len(pair_prompt(p.a, p.b))) // 4, 40)
                for p in pairs
            )
            * 2
        )
        cost *= repeats
        total += cost
        print(f"  {m.name:18} ${cost:6.2f}")
    print(
        f"  {'total':18} ${total:6.2f}  ({len(cases)} cases x {len(DESIGNS)} designs + {len(pairs)} pairs x 2 orders, x {repeats} repeats)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--estimate", action="store_true")
    mode.add_argument("--pilot", action="store_true", help="20 cases and 10 pairs, 1 repeat")
    parser.add_argument("--models", default="haiku,sonnet")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max-usd", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args(argv)

    cases, pairs = build()
    if args.dry_run:
        return dry_run(cases, pairs)
    models = [ClaudeModel(m) for m in args.models.split(",") if m]
    if args.estimate:
        estimate(cases, pairs, models, args.repeats)
        return 0
    repeats = args.repeats
    if args.pilot:
        cases, pairs, repeats = cases[:20], pairs[:5] + pairs[50:55], 1
    out = ROOT / "results" / "judgements.jsonl"
    out.parent.mkdir(exist_ok=True)
    rows, spent = asyncio.run(
        run(cases, pairs, models, repeats, out, args.max_usd, args.concurrency)
    )
    print(
        f"{len(rows)} judgements this run, ${spent:.3f}; errors: {sum(bool(r['error']) for r in rows)}"
    )
    meta = ROOT / "results" / "run.json"
    meta.write_text(
        json.dumps(
            {
                "last_run": datetime.now(UTC).isoformat(timespec="seconds"),
                "models": [resolve(m) for m in args.models.split(",")],
                "designs": DESIGNS + ["pair"],
                "repeats": args.repeats,
                "effort": "low",
                "packages": {p: version(p) for p in ("anthropic", "kestrel-outfitters")},
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
