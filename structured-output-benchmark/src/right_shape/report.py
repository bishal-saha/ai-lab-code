"""Turn results/extractions.jsonl into tables and charts.

    uv run python -m right_shape.report

Every number on the AI Lab page comes from this script.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from right_shape.grading import FIELDS  # noqa: E402
from right_shape.invoices import TRAPS  # noqa: E402
from right_shape.run import CONDITIONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CONDS = list(CONDITIONS)
SHORT = {"claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5"}
TRAP_LABELS = {
    "subtotal": "Subtotal vs total",
    "discount": "Discount",
    "currency": "Unsupported currency",
    "no_due_date": "No due date",
    "credit_note": "Credit note",
    "date": "Ambiguous date",
    "words": "Total in words",
    "none": "Control",
}


def load() -> list[dict]:
    return [
        json.loads(line)
        for line in (RESULTS / "extractions.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _rate(rows, pred) -> float | None:
    return round(sum(bool(pred(r)) for r in rows) / len(rows), 4) if rows else None


def summarise(rows: list[dict], by=("model", "condition")) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in by)].append(r)
    out = []
    for key, g in sorted(
        groups.items(), key=lambda kv: tuple(CONDS.index(v) if v in CONDS else v for v in kv[0])
    ):
        valid = [r for r in g if r["schema_valid"]]
        first_wrong = [r for r in valid if not r["first_all_correct"]]
        first_right = [r for r in valid if r["first_all_correct"]]
        row = dict(zip(by, key, strict=True))
        row.update(
            {
                "rows": len(g),
                "parsed": _rate(g, lambda r: r["parsed"]),
                "schema_valid": _rate(g, lambda r: r["schema_valid"]),
                "all_correct": _rate(g, lambda r: r["all_correct"]),
                "valid_but_wrong": _rate(g, lambda r: r["valid_but_wrong"]),
                **{f"field_{f}": _rate(g, lambda r, f=f: r["fields"][f]) for f in FIELDS},
                **{
                    f"trap_{t}": _rate([r for r in g if r["trap"] == t], lambda r: r["all_correct"])
                    for t in TRAPS
                },
                "currency_other_or_real": _rate(
                    [r for r in g if r["trap"] == "currency"], lambda r: r["fields"]["currency"]
                ),
                "due_date_fabricated": _rate(
                    [r for r in g if r["trap"] in ("no_due_date", "credit_note")],
                    lambda r: not r["fields"]["due_date"],
                ),
                "rules_catch": _rate(first_wrong, lambda r: r["rule_errors"]),
                "rules_false_alarm": _rate(first_right, lambda r: r["rule_errors"]),
                "retried": _rate(g, lambda r: r["retried"]),
                "repaired": sum(
                    r["retried"] and not r["first_all_correct"] and r["all_correct"] for r in g
                ),
                "broken_by_retry": sum(
                    r["retried"] and r["first_all_correct"] and not r["all_correct"] for r in g
                ),
                "refusals": sum(r.get("stop_reason") == "refusal" for r in g),
                "max_tokens": sum(r.get("stop_reason") == "max_tokens" for r in g),
                "errors": sum(r.get("stop_reason") == "error" for r in g),
                "calls": sum(r["calls"] for r in g),
                "cost_usd": round(sum(r["cost_usd"] for r in g), 4),
                "cost_per_correct_usd": round(
                    sum(r["cost_usd"] for r in g) / max(1, sum(r["all_correct"] for r in g)), 5
                ),
                "latency_s_median": round(statistics.median(r["latency_s"] for r in g), 2),
            }
        )
        out.append(row)
    return out


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def charts(rows: list[dict], out: Path) -> None:
    models = sorted({r["model"] for r in rows})
    table = summarise(rows)

    fig, axes = plt.subplots(
        1, len(models), figsize=(6.2 * len(models), 4.4), dpi=150, sharey=True, squeeze=False
    )
    for ax, m in zip(axes[0], models, strict=True):
        conds = [c for c in CONDS if any(t["model"] == m and t["condition"] == c for t in table)]
        valid = [
            next(t["schema_valid"] for t in table if t["model"] == m and t["condition"] == c)
            for c in conds
        ]
        right = [
            next(t["all_correct"] for t in table if t["model"] == m and t["condition"] == c)
            for c in conds
        ]
        xs = range(len(conds))
        ax.bar([x - 0.2 for x in xs], valid, 0.4, label="schema-valid", color="#c7d2fe")
        bars = ax.bar([x + 0.2 for x in xs], right, 0.4, label="every field right", color="#4f46e5")
        for bar, v in zip(bars, right, strict=True):
            ax.annotate(
                f"{v:.0%}",
                (bar.get_x() + bar.get_width() / 2, v),
                ha="center",
                va="bottom",
                fontsize=7,
                xytext=(0, 2),
                textcoords="offset points",
            )
        ax.set_xticks(list(xs), conds, fontsize=8, rotation=20)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(SHORT.get(m, m), fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2
    )
    fig.suptitle("Right shape is not right answer", fontsize=12, x=0.01, ha="left")
    _save(fig, out / "valid-vs-correct")

    pooled = summarise(rows, ("condition",))
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)
    grid = [
        [next(p[f"trap_{t}"] for p in pooled if p["condition"] == c) or 0 for t in TRAPS]
        for c in CONDS
        if any(p["condition"] == c for p in pooled)
    ]
    ax.imshow(grid, cmap="Purples", vmin=0, vmax=1, aspect="auto")
    for i, line in enumerate(grid):
        for j, v in enumerate(line):
            ax.text(
                j,
                i,
                f"{v:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.6 else "#1e1b4b",
            )
    ax.set_xticks(
        range(len(TRAPS)), [TRAP_LABELS[t] for t in TRAPS], fontsize=8, rotation=25, ha="right"
    )
    ax.set_yticks(
        range(len(grid)), [c for c in CONDS if any(p["condition"] == c for p in pooled)], fontsize=8
    )
    ax.set_title("Every field right, by trap (both models)", fontsize=12, loc="left", pad=10)
    _save(fig, out / "accuracy-by-trap")

    rows_v = [r for r in rows if r["condition"] == "so+d+v"]
    if rows_v:
        v = summarise(rows_v, ("model",))
        fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=150)
        metrics = [
            ("rules_catch", "wrong records the rules flagged"),
            ("rules_false_alarm", "right records flagged (false alarm)"),
        ]
        xs = range(len(v))
        for i, (k, label) in enumerate(metrics):
            vals = [x[k] or 0 for x in v]
            bars = ax.bar(
                [x + (i - 0.5) * 0.38 for x in xs],
                vals,
                0.38,
                label=label,
                color=["#4f46e5", "#c7d2fe"][i],
            )
            for bar, val in zip(bars, vals, strict=True):
                ax.annotate(
                    f"{val:.0%}",
                    (bar.get_x() + bar.get_width() / 2, val),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    xytext=(0, 2),
                    textcoords="offset points",
                )
        ax.set_xticks(
            list(xs),
            [
                f"{SHORT.get(x['model'], x['model'])}\n{x['repaired']} repaired by one retry"
                for x in v
            ],
            fontsize=9,
        )
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(
            "What business rules caught after the schema passed", fontsize=12, loc="left", pad=10
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
        _save(fig, out / "rules")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    rows = load()
    (RESULTS / "charts").mkdir(parents=True, exist_ok=True)
    table = summarise(rows)
    _write_csv(RESULTS / "summary.csv", table)
    _write_csv(RESULTS / "by_trap.csv", summarise(rows, ("model", "condition", "trap")))
    charts(rows, RESULTS / "charts")
    print(f"{len(rows)} rows -> results/summary.csv, by_trap.csv, charts/")
    for t in table:
        print(
            f"  {SHORT.get(t['model'], t['model']):10} {t['condition']:7} valid {t['schema_valid']:.0%}  correct {t['all_correct']:.0%}  valid-but-wrong {t['valid_but_wrong']:.0%}  cost ${t['cost_usd']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
