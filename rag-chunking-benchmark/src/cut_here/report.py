"""Turn results/retrieval.jsonl (and answers.jsonl, when present) into tables and charts.

    uv run python -m cut_here.report

Every number on the AI Lab page comes from this script, so anyone can re-derive
them from the committed results.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ORDER = ["F256", "F256o", "F512o", "R512", "SEM", "S", "S+H", "S+H+C"]
RETRIEVERS = ["dense", "bm25", "hybrid"]
QTYPES = ["table", "heading", "prose"]
QTYPE_LABELS = {"table": "Table rows", "heading": "Heading-dependent", "prose": "Prose"}
COLOURS = {"dense": "#4f46e5", "bm25": "#a5b4fc", "hybrid": "#1e1b4b"}
SHORT = {"claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5"}


def load(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def _at(rows: list[dict], k: int, key: str = "rank") -> float:
    return (
        round(sum(r[key] is not None and r[key] <= k for r in rows) / len(rows), 4) if rows else 0.0
    )


def retrieval_table(rows: list[dict], stats: dict) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["strategy"], r["retriever"])].append(r)
    out = []
    for s in [s for s in ORDER if any(k[0] == s for k in groups)]:
        for ret in RETRIEVERS:
            g = groups[(s, ret)]
            out.append(
                {
                    "strategy": s,
                    "retriever": ret,
                    "questions": len(g),
                    "recall@1": _at(g, 1),
                    "recall@5": _at(g, 5),
                    "recall@10": _at(g, 10),
                    "span_recall@5": _at(g, 5, "span_rank"),
                    "mrr": round(sum(1 / r["rank"] for r in g if r["rank"]) / len(g), 4),
                    **{f"recall@5_{t}": _at([r for r in g if r["qtype"] == t], 5) for t in QTYPES},
                    "miss5_no_context": sum(r["miss5"] == "no context" for r in g),
                    "miss5_no_span": sum(r["miss5"] == "no span" for r in g),
                    "chunks": stats.get(s, {}).get("chunks"),
                    "mean_tokens": stats.get(s, {}).get("mean_tokens"),
                    "embedded_tokens": stats.get(s, {}).get("embedded_tokens"),
                }
            )
    return out


def answer_table(answers: list[dict], table: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for a in answers:
        groups[(a["strategy"], a["model"])].append(a)
    recall = {(r["strategy"], r["retriever"]): r["recall@5"] for r in table}
    out = []
    for (s, m), g in sorted(groups.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
        out.append(
            {
                "strategy": s,
                "model": m,
                "questions": len(g),
                "accuracy": round(sum(a["correct"] for a in g) / len(g), 4),
                **{
                    f"accuracy_{t}": round(
                        sum(a["correct"] for a in g if a["qtype"] == t)
                        / max(1, sum(a["qtype"] == t for a in g)),
                        4,
                    )
                    for t in QTYPES
                },
                "hybrid_recall@5": recall.get((s, "hybrid")),
                "correct_without_hit": sum(a["correct"] and not a["hit5"] for a in g),
                "wrong_with_hit": sum(a["hit5"] and not a["correct"] for a in g),
                "cost_usd": round(sum(a["cost_usd"] for a in g), 4),
            }
        )
    return out


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def charts(table: list[dict], answers: list[dict], out: Path) -> None:
    strategies = list(dict.fromkeys(r["strategy"] for r in table))

    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
    width = 0.8 / len(RETRIEVERS)
    for i, ret in enumerate(RETRIEVERS):
        values = [
            next(r["recall@5"] for r in table if r["strategy"] == s and r["retriever"] == ret)
            for s in strategies
        ]
        xs = [j + (i - 1) * width for j in range(len(strategies))]
        bars = ax.bar(xs, values, width, label=ret, color=COLOURS[ret])
        for bar, v in zip(bars, values, strict=True):
            ax.annotate(
                f"{v:.0%}",
                (bar.get_x() + bar.get_width() / 2, v),
                ha="center",
                va="bottom",
                fontsize=7,
                xytext=(0, 2),
                textcoords="offset points",
            )
    ax.set_xticks(range(len(strategies)), strategies)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("questions with a usable chunk in the top 5", fontsize=9)
    ax.set_title("Recall@5 by chunking strategy and retriever", fontsize=12, loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08))
    _save(fig, out / "recall-at-5")

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    grid = [
        [
            next(
                r[f"recall@5_{t}"]
                for r in table
                if r["strategy"] == s and r["retriever"] == "hybrid"
            )
            for t in QTYPES
        ]
        for s in strategies
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
                fontsize=9,
                color="white" if v > 0.6 else "#1e1b4b",
            )
    ax.set_xticks(range(len(QTYPES)), [QTYPE_LABELS[t] for t in QTYPES], fontsize=9)
    ax.set_yticks(range(len(strategies)), strategies, fontsize=9)
    ax.set_title("Recall@5 by question type (hybrid retrieval)", fontsize=12, loc="left", pad=10)
    _save(fig, out / "recall-by-type")

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    for r in (r for r in table if r["retriever"] == "hybrid"):
        ax.scatter(r["chunks"], r["recall@5"], s=70, color="#4f46e5")
        ax.annotate(
            r["strategy"],
            (r["chunks"], r["recall@5"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("chunks in the index", fontsize=9)
    ax.set_ylabel("recall@5 (hybrid)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("More chunks didn't buy more recall", fontsize=12, loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out / "chunks-vs-recall")

    if answers:
        acc = answer_table(answers, table)
        models = sorted({a["model"] for a in acc})
        fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
        names = list(dict.fromkeys(a["strategy"] for a in acc))
        width = 0.8 / (len(models) + 1)
        xs0 = range(len(names))
        recall = [next(a["hybrid_recall@5"] for a in acc if a["strategy"] == s) for s in names]
        ax.bar([x - width for x in xs0], recall, width, label="recall@5 (hybrid)", color="#c7d2fe")
        for i, m in enumerate(models):
            vals = [
                next(a["accuracy"] for a in acc if a["strategy"] == s and a["model"] == m)
                for s in names
            ]
            bars = ax.bar(
                [x + i * width for x in xs0],
                vals,
                width,
                label=f"answer accuracy, {SHORT.get(m, m)}",
                color=["#4f46e5", "#1e1b4b"][i % 2],
            )
            for bar, v in zip(bars, vals, strict=True):
                ax.annotate(
                    f"{v:.0%}",
                    (bar.get_x() + bar.get_width() / 2, v),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    xytext=(0, 2),
                    textcoords="offset points",
                )
        ax.set_xticks(list(xs0), names)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title("Answers follow retrieval", fontsize=12, loc="left", pad=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08))
        _save(fig, out / "answers-vs-recall")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    rows, answers = load(RESULTS / "retrieval.jsonl"), load(RESULTS / "answers.jsonl")
    stats = json.loads((RESULTS / "chunks.json").read_text())
    (RESULTS / "charts").mkdir(parents=True, exist_ok=True)
    table = retrieval_table(rows, stats)
    _write_csv(RESULTS / "retrieval_summary.csv", table)
    if answers:
        _write_csv(RESULTS / "answers_summary.csv", answer_table(answers, table))
    charts(table, answers, RESULTS / "charts")
    print(
        f"{len(rows)} retrieval rows, {len(answers)} answers -> results/*_summary.csv, results/charts/"
    )
    for r in (r for r in table if r["retriever"] == "hybrid"):
        print(
            f"  {r['strategy']:6} chunks {r['chunks']:4}  recall@5 {r['recall@5']:.0%}  (span only {r['span_recall@5']:.0%})  table {r['recall@5_table']:.0%}  heading {r['recall@5_heading']:.0%}  prose {r['recall@5_prose']:.0%}"
        )
    if answers:
        for a in answer_table(answers, table):
            print(
                f"  answers {a['strategy']:6} {SHORT.get(a['model'], a['model']):10} accuracy {a['accuracy']:.0%}  cost ${a['cost_usd']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
