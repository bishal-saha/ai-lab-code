"""Run the matrix: 120 invoices x 7 conditions x models.

    uv run right-shape --dry-run                      # fakes, $0: an oracle and a mistake-maker through every condition
    uv run right-shape --estimate                     # projected cost of a live run
    uv run --env-file .env right-shape --pilot --max-usd 1   # one invoice per trap and layout mix, every condition
    uv run --env-file .env right-shape --max-usd 10   # the full matrix (resumes from results/extractions.jsonl)

Then `uv run python -m right_shape.report` builds the tables and charts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from right_shape import rules
from right_shape.grading import fields
from right_shape.invoices import TRAPS, Invoice, build, dataset
from right_shape.models import ClaudeModel, FakeModel
from right_shape.pricing import cost_usd, resolve
from right_shape.schemas import BASE, DESIGNED, DESIGNED_NO_NOTES, schema_errors

ROOT = Path(__file__).resolve().parents[2]
# id: (mechanism, schema, validate-and-retry)
CONDITIONS = {
    "prompt": ("prompt", BASE, False),
    "tool": ("tool", BASE, False),
    "strict": ("strict", BASE, False),
    "so": ("so", BASE, False),
    "so+d": ("so", DESIGNED, False),
    "so+d-n": ("so", DESIGNED_NO_NOTES, False),
    "so+d+v": ("so", DESIGNED, True),
}


def _cost(model: str, u: dict[str, int]) -> float:
    return cost_usd(
        model,
        u["input_tokens"],
        u["output_tokens"],
        u.get("cache_read", 0),
        u.get("cache_write", 0),
    )


async def run_one(inv: Invoice, condition: str, model) -> dict:
    mechanism, schema, validate = CONDITIONS[condition]
    started = time.perf_counter()
    row = {
        "invoice": inv.id,
        "trap": inv.trap,
        "layout": inv.layout,
        "condition": condition,
        "model": model.name,
        "calls": 0,
        "cost_usd": 0.0,
    }
    try:
        reply = await model.extract(inv, mechanism, schema)
        row["calls"] += 1
        row["cost_usd"] += _cost(model.name, reply.usage)
        record, row["stop_reason"] = reply.record, reply.stop_reason
        first = record
        rule_errors = rules.check(record) if record and not schema_errors(record, schema) else []
        row["rule_errors"] = rule_errors
        row["retried"] = False
        if validate and rule_errors:
            reply = await model.extract(inv, mechanism, schema, feedback=(record, rule_errors))
            row["calls"] += 1
            row["cost_usd"] += _cost(model.name, reply.usage)
            record, row["retried"] = reply.record or record, True
            row["rule_errors_after"] = (
                rules.check(record) if record and not schema_errors(record, schema) else []
            )
        row["error"] = None
    except Exception as err:  # an API failure fails the row, not the run
        first = record = None
        row.update(
            stop_reason="error", error=f"{type(err).__name__}: {err}", rule_errors=[], retried=False
        )
    errors = schema_errors(record, schema) if record else ["nothing parseable"]
    f, f_first = fields(inv, record), fields(inv, first)
    row.update(
        parsed=record is not None,
        schema_valid=record is not None and not errors,
        schema_errors=errors[:5],
        fields=f,
        all_correct=all(f.values()),
        first_all_correct=all(f_first.values()),
        valid_but_wrong=record is not None and not errors and not all(f.values()),
        review=bool(record) and rules.needs_review(record),
        record=record,
        latency_s=round(time.perf_counter() - started, 3),
        cost_usd=round(row["cost_usd"], 6),
    )
    return row


async def matrix(
    invoices, conditions, models, out: Path | None, max_usd: float, concurrency: int, quiet=False
) -> tuple[list[dict], float]:
    done = set()
    if out and out.exists():
        done = {
            (r["invoice"], r["condition"], r["model"])
            for r in map(json.loads, out.read_text().splitlines())
        }
    spent, rows, gate = 0.0, [], asyncio.Semaphore(concurrency)

    async def one(inv, cond, model):
        nonlocal spent
        async with gate:
            if spent >= max_usd:
                return
            row = await run_one(inv, cond, model)
            spent += row["cost_usd"]
            rows.append(row)
            if out:
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            if not quiet:
                flag = (
                    "ok   "
                    if row["all_correct"]
                    else ("VALID-BUT-WRONG" if row["valid_but_wrong"] else "INVALID")
                )
                print(
                    f"{flag:15} {row['model']:18} {cond:7} {inv.id} {inv.trap:11} ${row['cost_usd']:.4f} total ${spent:.2f}",
                    flush=True,
                )

    await asyncio.gather(
        *(
            one(inv, c, m)
            for m in models
            for c in conditions
            for inv in invoices
            if (inv.id, c, m.name) not in done
        )
    )
    return rows, spent


def summary(rows: list[dict]) -> None:
    print(
        f"\n{'model':18} {'cond':7} {'n':>4} {'valid':>6} {'correct':>8} {'valid-wrong':>12} {'cost':>8}"
    )
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["model"], r["condition"]), []).append(r)
    for (m, c), g in sorted(
        groups.items(), key=lambda kv: (kv[0][0], list(CONDITIONS).index(kv[0][1]))
    ):
        rate = lambda k, g=g: sum(bool(x[k]) for x in g) / len(g)  # noqa: E731
        print(
            f"{m:18} {c:7} {len(g):4} {rate('schema_valid'):6.0%} {rate('all_correct'):8.0%} {rate('valid_but_wrong'):12.0%} ${sum(x['cost_usd'] for x in g):7.3f}"
        )


def dry_run(invoices) -> int:
    oracle, _ = asyncio.run(
        matrix(invoices, list(CONDITIONS), [FakeModel("oracle")], None, 1e9, 32, quiet=True)
    )
    mistake, _ = asyncio.run(
        matrix(invoices, list(CONDITIONS), [FakeModel("mistake")], None, 1e9, 32, quiet=True)
    )
    problems = []
    for r in oracle:
        base = CONDITIONS[r["condition"]][1] is BASE
        expect = not (base and r["trap"] == "currency")  # a closed enum can't name JPY or CAD
        if not r["schema_valid"] or r["all_correct"] != expect or r["rule_errors"]:
            problems.append(
                f"oracle {r['condition']} {r['invoice']} {r['trap']}: valid={r['schema_valid']} correct={r['all_correct']} rules={r['rule_errors']} {r['schema_errors']}"
            )
    for r in mistake:
        if r["trap"] in ("subtotal", "discount", "words") and not r["rule_errors"]:
            problems.append(
                f"mistake {r['condition']} {r['invoice']} {r['trap']}: the rules missed an arithmetic error"
            )
        if (
            r["condition"] == "so+d+v"
            and r["trap"] in ("subtotal", "discount", "words")
            and not (r["retried"] and r["all_correct"])
        ):
            problems.append(f"mistake so+d+v {r['invoice']}: retry didn't repair")
        if r["trap"] != "none" and r["all_correct"] and not r["retried"]:
            problems.append(
                f"mistake {r['condition']} {r['invoice']} {r['trap']}: a planted mistake graded correct"
            )
    print(f"Dry run: {len(oracle)} oracle + {len(mistake)} mistake rows.")
    summary(mistake)
    if problems:
        print(f"\n{len(problems)} problem(s):", *problems[:15], sep="\n  ", file=sys.stderr)
        return 1
    print(
        "\nDry run OK: oracle records are schema-valid and graded right (the closed enum can't name JPY/CAD, as designed);\n"
        "every planted mistake is graded wrong, the rules catch every arithmetic one, and the retry repairs them."
    )
    return 0


def estimate(invoices, models) -> None:
    """Characters/4 on the invoice and the schema or example; 350 output tokens a call; 10% of
    so+d+v rows retried. No caching counted."""
    total = 0.0
    for m in models:
        cost = 0.0
        for _, schema, validate in CONDITIONS.values():
            per = sum(
                cost_usd(m.name, (len(inv.text) + len(json.dumps(schema)) + 600) // 4, 350)
                for inv in invoices
            )
            cost += per * (1.1 if validate else 1.0)
        total += cost
        print(
            f"  {m.name:18} ${cost:6.2f}  ({len(invoices)} invoices x {len(CONDITIONS)} conditions)"
        )
    print(f"  {'total':18} ${total:6.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--estimate", action="store_true")
    mode.add_argument(
        "--pilot", action="store_true", help="2 invoices per trap (16), every condition"
    )
    parser.add_argument("--models", default="haiku,sonnet")
    parser.add_argument(
        "--conditions", default="", help=f"comma-separated: {', '.join(CONDITIONS)}"
    )
    parser.add_argument("--max-usd", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)

    invoices = build(dataset())
    if args.dry_run:
        return dry_run(invoices)
    models = [ClaudeModel(m) for m in args.models.split(",") if m]
    if args.estimate:
        estimate(invoices, models)
        return 0
    if args.pilot:
        invoices = [inv for trap in TRAPS for inv in [i for i in invoices if i.trap == trap][:2]]
    conditions = [c for c in CONDITIONS if not args.conditions or c in args.conditions.split(",")]
    out = ROOT / "results" / "extractions.jsonl"
    out.parent.mkdir(exist_ok=True)
    rows, spent = asyncio.run(
        matrix(invoices, conditions, models, out, args.max_usd, args.concurrency)
    )
    summary(rows)
    print(f"\nspent ${spent:.3f} this run")
    meta = ROOT / "results" / "run.json"
    previous = json.loads(meta.read_text()) if meta.exists() else {}
    meta.write_text(
        json.dumps(
            {
                **previous,
                "last_run": datetime.now(UTC).isoformat(timespec="seconds"),
                "models": [resolve(m) for m in args.models.split(",")],
                "conditions": list(CONDITIONS),
                "invoices": len(build(dataset())),
                "effort": "medium",
                "packages": {
                    p: version(p) for p in ("anthropic", "jsonschema", "kestrel-outfitters")
                },
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
