"""Turn results/episodes.jsonl into summary tables and charts.

    uv run python -m four_shapes.report                  # reads results/, writes results/ + charts/
    uv run python -m four_shapes.report --input other.jsonl --out /tmp/report

Every number on the AI Lab page comes from this script, so anyone can re-derive
them from the committed episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARCHS = ["loop", "graph", "planner", "multi"]
ARCH_LABELS = {"loop": "Single loop", "graph": "Graph workflow", "planner": "Planner-executor", "multi": "Multi-agent"}
KINDS = ["policy", "multi", "status", "no_order"]
KIND_LABELS = {"policy": "Refund policy", "multi": "Two intents", "status": "Where is it?", "no_order": "No order id"}
SHORT = {"claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "fake": "fake"}
COLOURS = ["#a5b4fc", "#4f46e5", "#1e1b4b"]  # light indigo, brand indigo, deep indigo


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rate(rows: list[dict], key) -> float | None:
    return round(sum(bool(key(r)) for r in rows) / len(rows), 4) if rows else None


def predictability(rows: list[dict]) -> float | None:
    """Mean share of repeats whose action sequence matches the most common one, per ticket."""
    cells: dict[tuple, list[tuple]] = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["arch"], r["condition"], r["scenario"])].append(tuple(r["sequence"]))
    shares = [Counter(seqs).most_common(1)[0][1] / len(seqs) for seqs in cells.values() if len(seqs) > 1]
    return round(statistics.mean(shares), 4) if shares else None


def summarise(rows: list[dict], by: tuple[str, ...] = ("model", "arch")) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in by)].append(r)
    out = []
    for key, eps in sorted(groups.items()):
        multi = [e for e in eps if e["kind"] == "multi"]
        faults = [e for e in eps if e["condition"] == "faults"]
        successes = sum(e["success"] for e in eps)
        row = dict(zip(by, key, strict=True))
        row.update(
            {
                "episodes": len(eps),
                "success": _rate(eps, lambda e: e["success"]),
                **{f"success_{k}": _rate([e for e in eps if e["kind"] == k], lambda e: e["success"]) for k in KINDS},
                "success_faults": _rate(faults, lambda e: e["success"]),
                "violation": _rate(eps, lambda e: e["violation"]),
                "missed_intent": _rate(multi, lambda e: e["missed_intent"]),
                "to_human": _rate(eps, lambda e: e["to_human"]),
                "duplicate_replies": _rate(eps, lambda e: e["duplicate_replies"] > 0),
                "invalid_plan": _rate(eps, lambda e: e["invalid_plans"] > 0),
                "handoff_invalid": _rate(eps, lambda e: e["handoff_invalid"] > 0),
                "predictability": predictability(eps),
                "llm_calls_mean": round(statistics.mean(e["llm_calls"] for e in eps), 2),
                "tool_calls_mean": round(statistics.mean(e["tool_calls"] for e in eps), 2),
                "input_tokens_mean": round(statistics.mean(e["input_tokens"] + e["cache_read_tokens"] + e["cache_write_tokens"] for e in eps)),
                "cost_usd_total": round(sum(e["cost_usd"] for e in eps), 4),
                "cost_per_episode_usd": round(statistics.mean(e["cost_usd"] for e in eps), 5),
                "cost_per_success_usd": round(sum(e["cost_usd"] for e in eps) / successes, 5) if successes else None,
                "latency_s_median": round(statistics.median(e["latency_s"] for e in eps), 2),
                "stops": dict(Counter(e["stop"] for e in eps if e["stop"] != "done")),
                "errors": sum(e["stop"] == "error" for e in eps),
            }
        )
        out.append(row)
    return out


def _bars(ax, summary: list[dict], metric: str, models: list[str], fmt: str = "{:.0%}", scale: float = 1.0) -> None:
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        values = [(next((s[metric] for s in summary if s["model"] == model and s["arch"] == a), None) or 0) * scale for a in ARCHS]
        xs = [j + (i - (len(models) - 1) / 2) * width for j in range(len(ARCHS))]
        bars = ax.bar(xs, values, width, label=SHORT.get(model, model), color=COLOURS[i % len(COLOURS)])
        for bar, v in zip(bars, values, strict=True):
            ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(ARCHS)), [ARCH_LABELS[a] for a in ARCHS], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def charts(summary: list[dict], rows: list[dict], out: Path) -> None:
    models = sorted({s["model"] for s in summary})

    # Success by ticket kind: one heatmap per model.
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 3.6), dpi=150, squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        grid, labels = [], []
        for a in ARCHS:
            line, text = [], []
            for k in KINDS:
                cell = [r for r in rows if r["model"] == model and r["arch"] == a and r["kind"] == k]
                v = sum(r["success"] for r in cell) / len(cell) if cell else float("nan")
                line.append(v)
                text.append(f"{v:.0%}" if cell else "–")
            grid.append(line)
            labels.append(text)
        ax.imshow(grid, cmap="Purples", vmin=0, vmax=1, aspect="auto")
        for i, line in enumerate(labels):
            for j, v in enumerate(line):
                ax.text(j, i, v, ha="center", va="center", fontsize=9, color="white" if grid[i][j] > 0.6 else "#1e1b4b")
        ax.set_xticks(range(len(KINDS)), [KIND_LABELS[k] for k in KINDS], fontsize=8)
        ax.set_yticks(range(len(ARCHS)), [ARCH_LABELS[a] for a in ARCHS], fontsize=8)
        ax.set_title(SHORT.get(model, model), fontsize=11, loc="left")
    fig.suptitle("Tickets resolved correctly, by ticket type", fontsize=12, x=0.01, ha="left")
    _save(fig, out / "success-by-kind")

    for metric, title, ylabel, name, fmt, scale in [
        ("cost_per_success_usd", "Cost per correctly resolved ticket", "USD per 1,000 correct tickets", "cost-per-success", "{:.1f}", 1000),
        ("predictability", "Same actions on every repeat", "share of repeats matching the most common action sequence", "predictability", "{:.0%}", 1),
        ("llm_calls_mean", "LLM calls per ticket", "mean LLM calls per episode", "llm-calls", "{:.1f}", 1),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
        _bars(ax, summary, metric, models, fmt, scale)
        if metric == "predictability":
            ax.set_ylim(0, 1.15)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, fontsize=12, loc="left", pad=12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=len(models))
        _save(fig, out / name)

    # Failure modes: what goes wrong, per design (both models pooled).
    modes = [("violation", "Money before the check"), ("missed_intent", "Missed second intent"), ("to_human", "Handed to a human"),
             ("duplicate_replies", "Two replies"), ("invalid_plan", "Invalid plan proposed")]
    pooled = {s["arch"]: s for s in summarise(rows, ("arch",))}
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    width = 0.8 / len(modes)
    for i, (metric, label) in enumerate(modes):
        values = [pooled.get(a, {}).get(metric) or 0 for a in ARCHS]
        xs = [j + (i - (len(modes) - 1) / 2) * width for j in range(len(ARCHS))]
        ax.bar(xs, values, width, label=label, color=["#c7d2fe", "#818cf8", "#4f46e5", "#3730a3", "#1e1b4b"][i])
    ax.set_xticks(range(len(ARCHS)), [ARCH_LABELS[a] for a in ARCHS], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("share of episodes (missed intent: of two-intent tickets)", fontsize=8)
    ax.set_title("How each design fails", fontsize=12, loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3)
    _save(fig, out / "failure-modes")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "episodes.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    rows = load(args.input)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "charts").mkdir(exist_ok=True)

    summary = summarise(rows)
    _write_csv(args.out / "summary.csv", summary)
    _write_csv(args.out / "by_condition.csv", summarise(rows, ("model", "arch", "condition")))
    _write_csv(args.out / "by_kind.csv", summarise(rows, ("model", "arch", "kind")))
    charts(summary, rows, args.out / "charts")

    print(f"{len(rows)} episodes -> {args.out}/summary.csv, by_condition.csv, by_kind.csv, charts/")
    for s in summary:
        print(
            f"  {SHORT.get(s['model'], s['model']):10} {s['arch']:7} success {s['success'] or 0:.0%}  violation {s['violation'] or 0:.0%}  "
            f"missed {s['missed_intent'] or 0:.0%}  predictable {s['predictability'] or 0:.0%}  llm {s['llm_calls_mean']}  "
            f"cost/success ${s['cost_per_success_usd'] or 0:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
