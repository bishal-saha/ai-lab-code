"""Turn results/episodes.jsonl into results/summary.csv, per-family tables and charts.

    uv run python -m tool_shape.report                 # reads results/, writes results/ + charts/
    uv run python -m tool_shape.report --input other.jsonl --out /tmp/report

Every number on the AI Lab page comes from this script, so anyone can re-derive
them from the committed episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DESIGNS = ["endpoints", "tasks"]
LABELS = {"endpoints": "Endpoint wrappers", "tasks": "Task-shaped tools"}
COLOURS = {"endpoints": "#94a3b8", "tasks": "#4f46e5"}  # slate, brand indigo
SHORT = {"claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "fake": "fake"}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["server"])].append(r)
    out = []
    for (model, server), eps in sorted(groups.items()):
        n = len(eps)
        wins = [e for e in eps if e["success"]]
        cost = sum(e["cost_usd"] for e in eps)
        out.append(
            {
                "model": model,
                "design": server,
                "episodes": n,
                "success_rate": round(len(wins) / n, 4),
                "tool_calls_mean": round(statistics.mean(e["tool_calls"] for e in eps), 2),
                "turns_mean": round(statistics.mean(e["turns"] for e in eps), 2),
                "tool_errors_mean": round(statistics.mean(e["tool_errors"] for e in eps), 2),
                "input_tokens_mean": round(
                    statistics.mean(
                        e["input_tokens"] + e["cache_read_tokens"] + e["cache_write_tokens"]
                        for e in eps
                    )
                ),
                "output_tokens_mean": round(statistics.mean(e["output_tokens"] for e in eps)),
                "cost_usd_total": round(cost, 4),
                "cost_per_success_usd": round(cost / len(wins), 5) if wins else None,
                "latency_s_median": round(statistics.median(e["latency_s"] for e in eps), 2),
                "breaker_trips": sum(e["stop"] == "breaker" for e in eps),
                "no_answer": sum(e["stop"] in ("no_answer", "max_turns") for e in eps),
                "errors": sum(e["stop"] == "error" for e in eps),
            }
        )
    return out


def by_family(rows: list[dict]) -> dict[str, dict[tuple[str, str], float]]:
    acc: dict[str, dict[tuple[str, str], list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["family"]][(r["model"], r["server"])].append(r["success"])
    return {
        fam: {k: sum(v) / len(v) for k, v in cells.items()} for fam, cells in sorted(acc.items())
    }


def _bars(summary: list[dict], metric: str, title: str, ylabel: str, path: Path, pct=False):
    models = sorted({s["model"] for s in summary})
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    width = 0.36
    for i, design in enumerate(DESIGNS):
        values = [
            next(
                (s[metric] or 0 for s in summary if s["model"] == m and s["design"] == design), 0
            )
            for m in models
        ]
        xs = [j + (i - 0.5) * width for j in range(len(models))]
        bars = ax.bar(xs, values, width, label=LABELS[design], color=COLOURS[design])
        for bar, v in zip(bars, values, strict=True):
            text = f"{v:.0%}" if pct else (f"${v:.4f}" if metric.startswith("cost") else f"{v:.1f}")
            ax.annotate(
                text,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )
    ax.set_xticks(range(len(models)), [SHORT.get(m, m) for m in models])
    ax.set_title(title, fontsize=12, loc="left", pad=12)
    ax.set_ylabel(ylabel)
    if pct:
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "episodes.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    rows = load(args.input)
    args.out.mkdir(parents=True, exist_ok=True)
    charts = args.out / "charts"
    charts.mkdir(exist_ok=True)

    summary = summarise(rows)
    with (args.out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    families = by_family(rows)
    with (args.out / "by_family.csv").open("w", newline="") as f:
        cols = sorted({k for cells in families.values() for k in cells})
        writer = csv.writer(f)
        writer.writerow(["family", *[f"{SHORT.get(m, m)} / {d}" for m, d in cols]])
        for fam, cells in families.items():
            writer.writerow([fam, *[round(cells.get(c, 0), 4) for c in cols]])

    _bars(summary, "success_rate", "Task success rate", "tasks answered correctly", charts / "success-rate", pct=True)  # fmt: skip
    _bars(summary, "tool_calls_mean", "Tool calls per task", "mean MCP tool calls", charts / "tool-calls")  # fmt: skip
    _bars(summary, "cost_per_success_usd", "Cost per correct answer", "USD", charts / "cost-per-success")  # fmt: skip

    print(f"{len(rows)} episodes -> {args.out}/summary.csv, by_family.csv, charts/")
    for s in summary:
        print(
            f"  {SHORT.get(s['model'], s['model']):10} {s['design']:9} success {s['success_rate']:.0%}  "
            f"calls {s['tool_calls_mean']:.1f}  tokens in {s['input_tokens_mean']}  "
            f"cost/success ${s['cost_per_success_usd'] or 0:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
