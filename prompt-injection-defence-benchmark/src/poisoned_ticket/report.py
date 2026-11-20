"""Turn results/episodes.jsonl into summary tables and charts.

    uv run python -m poisoned_ticket.report                  # reads results/, writes results/ + charts/
    uv run python -m poisoned_ticket.report --input other.jsonl --out /tmp/report

Every number on the AI Lab page comes from this script, so anyone can re-derive
them from the committed episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from poisoned_ticket.attacks import GOALS, STYLES, TIERS  # noqa: E402
from poisoned_ticket.defences import DEFENCES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEF_IDS = [d.id for d in DEFENCES]
DEF_LABELS = {d.id: f"{d.id}\n" + textwrap.fill(d.label, 12) for d in DEFENCES}
SHORT = {"claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "fake": "fake"}
COLOURS = ["#a5b4fc", "#4f46e5", "#1e1b4b"]  # light indigo, brand indigo, deep indigo
GOAL_LABELS = {
    "refund_other": "Out-of-policy refund",
    "exfiltrate": "Exfiltration",
    "goodwill": "In-policy misuse",
}


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:  # episodes from before tiers existed carry no tier field
        r["tier"] = TIERS.get(r["style"]) if r["style"] else None
    return rows


def tier_rows(rows: list[dict], tier: str) -> list[dict]:
    """Clean episodes plus one tier's attacks."""
    return [r for r in rows if r["attack"] == "clean" or r["tier"] == tier]


def _rate(rows: list[dict], key) -> float | None:
    return round(sum(bool(key(r)) for r in rows) / len(rows), 4) if rows else None


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["defence"])].append(r)
    out = []
    for (model, defence), eps in sorted(groups.items()):
        atk = [e for e in eps if e["attack"] != "clean"]
        clean = [e for e in eps if e["attack"] == "clean"]
        row = {
            "model": model,
            "defence": defence,
            "episodes": len(eps),
            "attacked": len(atk),
            "attack_success": _rate(atk, lambda e: e["attack_success"]),
        }
        for g in GOALS:
            row[f"attack_success_{g}"] = _rate(
                [e for e in atk if e["goal"] == g], lambda e: e["attack_success"]
            )
        row.update(
            {
                "utility_clean": _rate(clean, lambda e: e["utility"]),
                "utility_attacked": _rate(atk, lambda e: e["utility"]),
                "flagged_attacked": _rate(atk, lambda e: e["flagged"]),
                "flagged_clean": _rate(clean, lambda e: e["flagged"]),
                "skipped_legit_attacked": _rate(atk, lambda e: not e["utility"]),
                "model_refusals": sum(e["stop"] == "refusal" for e in eps),
                "refused_actions": sum(e["refused"] for e in eps),
                "spontaneous_harm_clean": sum(any(e["harm"].values()) for e in clean),
                "tool_calls_mean": round(statistics.mean(e["tool_calls"] for e in eps), 2),
                "cost_usd_total": round(sum(e["cost_usd"] for e in eps), 4),
                "cost_per_episode_usd": round(statistics.mean(e["cost_usd"] for e in eps), 5),
                "latency_s_median": round(statistics.median(e["latency_s"] for e in eps), 2),
                "errors": sum(e["stop"] == "error" for e in eps),
            }
        )
        out.append(row)
    return out


def by_attack(rows: list[dict]) -> list[dict]:
    cells: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for r in rows:
        if r["attack"] != "clean":
            cells[(r["model"], r["defence"], r["attack"])].append(bool(r["attack_success"]))
    return [
        {
            "model": m,
            "defence": d,
            "attack": a,
            "episodes": len(v),
            "attack_success": round(sum(v) / len(v), 4),
        }
        for (m, d, a), v in sorted(cells.items())
    ]


def _grouped(ax, summary: list[dict], metric: str, models: list[str], short: bool = False) -> None:
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        values = [
            next((s[metric] for s in summary if s["model"] == model and s["defence"] == d), None)
            or 0
            for d in DEF_IDS
        ]
        xs = [j + (i - (len(models) - 1) / 2) * width for j in range(len(DEF_IDS))]
        bars = ax.bar(
            xs, values, width, label=SHORT.get(model, model), color=COLOURS[i % len(COLOURS)]
        )
        for bar, v in zip(bars, values, strict=True):
            ax.annotate(
                f"{v:.0%}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
    ax.set_xticks(
        range(len(DEF_IDS)), DEF_IDS if short else [DEF_LABELS[d] for d in DEF_IDS], fontsize=8
    )
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def charts(summary: list[dict], rows: list[dict], out: Path) -> None:
    """`summary`/`rows` are the main matrix (clean + tier 2); the heatmap uses every episode."""
    models = sorted({s["model"] for s in summary})

    for metric, title, ylabel, name in [
        (
            "attack_success",
            "Attack success rate by defence (tier 2)",
            "attacked episodes where the attack worked",
            "attack-success",
        ),
        (
            "utility_clean",
            "Clean tickets resolved correctly",
            "clean episodes with the policy outcome",
            "utility",
        ),
        (
            "utility_attacked",
            "Attacked tickets still resolved correctly (tier 2)",
            "attacked episodes with the policy outcome",
            "utility-attacked",
        ),
        (
            "flagged_attacked",
            "Injections flagged for review (tier 2)",
            "attacked episodes with flag_for_review",
            "flag-rate",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
        _grouped(ax, summary, metric, models)
        ax.set_title(title, fontsize=12, loc="left", pad=12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=len(models))
        _save(fig, out / name)

    fig, axes = plt.subplots(
        1, len(models), figsize=(5.5 * len(models), 4.6), dpi=150, squeeze=False
    )
    for ax, model in zip(axes[0], models, strict=True):
        grid, labels = [], []
        for style in STYLES:
            line, text = [], []
            for d in DEF_IDS:
                cell = [
                    r
                    for r in rows
                    if r["model"] == model and r["defence"] == d and r["style"] == style
                ]
                hits = sum(bool(r["attack_success"]) for r in cell)
                line.append(hits / len(cell) if cell else float("nan"))
                text.append(f"{hits}/{len(cell)}" if cell else "–")
            grid.append(line)
            labels.append(text)
        ax.imshow(grid, cmap="Purples", vmin=0, vmax=1, aspect="auto")
        for i, line in enumerate(labels):
            for j, v in enumerate(line):
                ax.text(j, i, v, ha="center", va="center", fontsize=7, color="#1e1b4b")
        ax.axhline(len(STYLES) / 2 - 0.5, color="#4f46e5", linewidth=1)
        ax.set_xticks(range(len(DEF_IDS)), DEF_IDS)
        ax.set_yticks(range(len(STYLES)), [f"{s} ({TIERS[s][-1]})" for s in STYLES])
        ax.set_title(SHORT.get(model, model), fontsize=11, loc="left")
    fig.suptitle(
        "Successful attacks / attacked episodes, by style (tier 1 above the line, tier 2 below)",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    _save(fig, out / "attack-style-heatmap")

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        values = [
            next(
                (
                    s["cost_per_episode_usd"]
                    for s in summary
                    if s["model"] == model and s["defence"] == d
                ),
                0,
            )
            * 1000
            for d in DEF_IDS
        ]
        xs = [j + (i - (len(models) - 1) / 2) * width for j in range(len(DEF_IDS))]
        bars = ax.bar(
            xs, values, width, label=SHORT.get(model, model), color=COLOURS[i % len(COLOURS)]
        )
        for bar, v in zip(bars, values, strict=True):
            ax.annotate(
                f"{v:.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
    ax.set_xticks(range(len(DEF_IDS)), [DEF_LABELS[d] for d in DEF_IDS], fontsize=8)
    ax.set_ylabel("USD per 1,000 episodes", fontsize=9)
    ax.set_title("Cost per episode by defence", fontsize=12, loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=len(models))
    _save(fig, out / "cost-per-episode")


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

    # The main matrix is clean + tier 2 under every defence. Tier 1 (the textbook set)
    # was run under D0 on every scenario and under every defence on s1 only.
    summary = summarise(tier_rows(rows, "tier2"))
    _write_csv(args.out / "summary.csv", summary)
    _write_csv(args.out / "summary_tier1.csv", summarise(tier_rows(rows, "tier1")))
    _write_csv(args.out / "by_attack.csv", by_attack(rows))
    charts(summary, rows, args.out / "charts")
    for old in ("attack-success-by-goal.png", "attack-success-by-goal.svg"):
        (args.out / "charts" / old).unlink(missing_ok=True)

    print(
        f"{len(rows)} episodes -> {args.out}/summary.csv, summary_tier1.csv, by_attack.csv, charts/"
    )
    for s in summary:
        print(
            f"  {SHORT.get(s['model'], s['model']):10} {s['defence']}  attack success {s['attack_success'] or 0:.0%}  "
            f"utility clean {s['utility_clean'] or 0:.0%}  util atk {s['utility_attacked'] or 0:.0%}  "
            f"refusals {s['model_refusals']}  flagged {s['flagged_attacked'] or 0:.0%}  "
            f"cost ${s['cost_usd_total']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
