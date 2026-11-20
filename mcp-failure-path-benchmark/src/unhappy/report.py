"""Turn results/episodes.jsonl into tables and charts.

    uv run python -m unhappy.report

Writes results/variants.csv (one row per variant x model), results/modes.csv (variant x mode x
model) and results/charts/*.png, and prints the headline numbers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from unhappy.servers import VARIANTS  # noqa: E402
from unhappy.tasks import MODES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
SHORT = {"claude-haiku-5-5": "Haiku", "claude-sonnet-5-5": "Sonnet"}


def load() -> list[dict]:
    return [
        json.loads(x) for x in (RESULTS / "episodes.jsonl").read_text().splitlines() if x.strip()
    ]


def rate(xs: list) -> float:
    return round(sum(map(bool, xs)) / len(xs), 3) if xs else float("nan")


def variant_table(rows: list[dict]) -> list[dict]:
    out = []
    for m in MODELS:
        for v in VARIANTS:
            rs = [r for r in rows if r["model"] == m and r["variant"] == v]
            if not rs:
                continue
            out.append(
                {
                    "model": m,
                    "variant": v,
                    "episodes": len(rs),
                    "success": rate([r["success"] for r in rs]),
                    "duplicates": sum(r["duplicate"] for r in rs),
                    "silent_wrong": sum(r["silent_wrong"] for r in rs),
                    "gave_up": sum(r["gave_up"] for r in rs),
                    "repeats_of_failed_calls": sum(r["repeats"] for r in rs),
                    "extra_locked_posts": sum(
                        r["post_attempts"] - 1
                        for r in rs
                        if r["mode"] == "permission" and r["post_attempts"]
                    ),
                    "no_answer": sum(not r["answered"] for r in rs),
                    "mean_calls": round(mean(len(r["calls"]) for r in rs), 2),
                    "cost_per_episode": round(mean(r["cost_usd"] for r in rs), 5),
                }
            )
    return out


def mode_table(rows: list[dict]) -> list[dict]:
    out = []
    for m in MODELS:
        for v in VARIANTS:
            for mode in MODES:
                rs = [
                    r for r in rows if r["model"] == m and r["variant"] == v and r["mode"] == mode
                ]
                if not rs:
                    continue
                out.append(
                    {
                        "model": m,
                        "variant": v,
                        "mode": mode,
                        "episodes": len(rs),
                        "success": rate([r["success"] for r in rs]),
                        "duplicates": sum(r["duplicate"] for r in rs),
                        "silent_wrong": sum(r["silent_wrong"] for r in rs),
                        "gave_up": sum(r["gave_up"] for r in rs),
                        "repeats": sum(r["repeats"] for r in rs),
                        "mean_post_attempts": round(mean(r["post_attempts"] for r in rs), 2),
                    }
                )
    return out


def write_csv(name: str, table: list[dict]) -> None:
    with (RESULTS / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def chart_heatmap(modes: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    variants = list(VARIANTS)
    for ax, m in zip(axes, MODELS, strict=True):
        grid = [
            [
                next(
                    (
                        x["success"]
                        for x in modes
                        if x["model"] == m and x["variant"] == v and x["mode"] == mode
                    ),
                    float("nan"),
                )
                * 100
                for mode in MODES
            ]
            for v in variants
        ]
        ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=9, color="black")
        ax.set_xticks(range(len(MODES)), [x.replace("_", " ") for x in MODES], rotation=20)
        ax.set_yticks(range(len(variants)), variants)
        ax.set_title(SHORT[m])
    fig.suptitle("Tasks completed (%), by server variant and failure mode")
    fig.tight_layout()
    fig.savefig(CHARTS / "success-heatmap.png", dpi=150)
    plt.close(fig)


def chart_harms(variants: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    names = list(VARIANTS)
    for ax, metric, title in zip(
        axes,
        ("duplicates", "silent_wrong", "extra_locked_posts"),
        (
            "duplicate notes posted",
            "wrong answers reported as done",
            "posts to a locked ticket after a refusal",
        ),
        strict=True,
    ):
        for i, m in enumerate(MODELS):
            pts = {x["variant"]: x[metric] for x in variants if x["model"] == m}
            ax.barh(
                [k + (i - 0.5) * 0.38 for k in range(len(names))],
                [pts.get(v, 0) for v in names],
                0.38,
                label=SHORT[m],
                color=["#93c5fd", "#1d4ed8"][i],
            )
        ax.set_title(title)
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
    axes[0].legend()
    fig.suptitle("What went wrong, by server variant (count over all episodes)")
    fig.tight_layout()
    fig.savefig(CHARTS / "harms.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rows = load()
    CHARTS.mkdir(parents=True, exist_ok=True)
    variants, modes = variant_table(rows), mode_table(rows)
    write_csv("variants.csv", variants)
    write_csv("modes.csv", modes)
    chart_heatmap(modes)
    chart_harms(variants)
    for x in variants:
        print(x)
    print(f"\n{len(rows)} episodes, USD {sum(r['cost_usd'] for r in rows):.2f}")


if __name__ == "__main__":
    main()
