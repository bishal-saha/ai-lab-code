"""Turn results/runs.jsonl into tables and charts.

    uv run python -m ticket_thirty.report

Writes results/queue.csv (one row per policy x model), results/by_third.csv (queue positions
1-8, 9-16, 17-24), results/tools.csv (the tool-count study), results/summaries.csv (whether each
running summary still carries each instruction) and results/charts/*.png.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ticket_thirty.policies import POLICIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
SHORT = {"claude-haiku-5-5": "Haiku", "claude-sonnet-5-5": "Sonnet"}
THIRDS = [(1, 8), (9, 16), (17, 24)]
COLOURS = {
    "raw": "#c2410c",
    "trim": "#2563eb",
    "clear": "#0891b2",
    "summary": "#a21caf",
    "summary+pin": "#7c3aed",
    "fresh": "#15803d",
}


def load() -> list[dict]:
    return [
        json.loads(line)
        for line in (RESULTS / "runs.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 3) if values else float("nan")


def queue_table(rows: list[dict]) -> list[dict]:
    out = []
    for m in MODELS:
        for p in POLICIES:
            runs = [
                r for r in rows if r["study"] == "queue" and r["model"] == m and r["policy"] == p
            ]
            if not runs:
                continue
            g = [x for r in runs for x in r["grades"]]
            delayed = [x for x in g if x["expected"] == "delayed"]
            calls = [c for r in runs for c in r["calls"]]
            ok = sum(x["success"] for x in g)
            cost = sum(r["cost_usd"] for r in runs)
            out.append(
                {
                    "model": m,
                    "policy": p,
                    "runs": len(runs),
                    "tickets": len(g),
                    "success": _rate([x["success"] for x in g]),
                    "outcome_ok": _rate([x["outcome_ok"] for x in g]),
                    "freeze_violations": sum(x["freeze_violation"] for x in g),
                    "freeze_tickets": len(delayed),
                    "audit_flag": _rate([x["audit_flag"] for x in g]),
                    "money_before_check": sum(x["violation"] for x in g),
                    "no_action_tickets": sum(
                        1
                        for r in runs
                        for x in r["grades"]
                        if not any(c["ticket"] == x["ticket"] for c in r["ledger"])
                    ),
                    "mean_window_tokens": round(mean(c["window_tokens"] for c in calls)),
                    "final_window_tokens": round(
                        mean(r["calls"][-1]["window_tokens"] for r in runs)
                    ),
                    "model_calls_per_ticket": round(len(calls) / len(g), 2),
                    "cost_per_run": round(cost / len(runs), 4),
                    "cost_per_success": round(cost / ok, 5) if ok else float("nan"),
                }
            )
    return out


def third_table(rows: list[dict]) -> list[dict]:
    out = []
    for m in MODELS:
        for p in POLICIES:
            g = [
                x
                for r in rows
                if r["study"] == "queue" and r["model"] == m and r["policy"] == p
                for x in r["grades"]
            ]
            if not g:
                continue
            for lo, hi in THIRDS:
                part = [x for x in g if lo <= x["position"] <= hi]
                out.append(
                    {
                        "model": m,
                        "policy": p,
                        "positions": f"{lo}-{hi}",
                        "tickets": len(part),
                        "success": _rate([x["success"] for x in part]),
                        "audit_flag": _rate([x["audit_flag"] for x in part]),
                        "freeze_violations": sum(x["freeze_violation"] for x in part),
                    }
                )
    return out


def tools_table(rows: list[dict]) -> list[dict]:
    out = []
    for m in MODELS:
        for ts in ("core", "distractors"):
            runs = [
                r for r in rows if r["study"] == "tools" and r["model"] == m and r["policy"] == ts
            ]
            if not runs:
                continue
            g = [x for r in runs for x in r["grades"]]
            out.append(
                {
                    "model": m,
                    "tools": 10 if ts == "core" else 40,
                    "episodes": len(runs),
                    "success": _rate([x["success"] for x in g]),
                    "wrong_tool_calls": sum(x["wrong_tool"] for x in g),
                    "episodes_with_wrong_tool": sum(1 for x in g if x["wrong_tool"]),
                    "bad_argument_calls": sum(x["bad_args"] for x in g),
                    "audit_flag": _rate([x["audit_flag"] for x in g]),
                    "mean_window_tokens": round(
                        mean(c["window_tokens"] for r in runs for c in r["calls"])
                    ),
                    "cost_per_episode": round(sum(r["cost_usd"] for r in runs) / len(runs), 5),
                }
            )
    return out


def summary_table(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r["study"] != "queue" or not r["summaries"]:
            continue
        s = r["summaries"]
        out.append(
            {
                "model": r["model"],
                "policy": r["policy"],
                "order": r["order"],
                "summaries": len(s),
                "keeps_freeze": _rate([x["keeps"]["freeze"] for x in s]),
                "keeps_audit": _rate([x["keeps"]["audit"] for x in s]),
                "mean_summary_tokens": round(mean(x["output_tokens"] for x in s)),
            }
        )
    return out


def write_csv(name: str, table: list[dict]) -> None:
    if not table:
        return
    with (RESULTS / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def chart_by_third(thirds: list[dict]) -> None:
    """Tickets solved per third of the queue. Each policy is nudged sideways a little so lines that
    coincide (most of them sit at 100%) stay visible; the audit flag was ~100% everywhere (queue.csv)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, m in zip(axes, MODELS, strict=True):
        for k, p in enumerate(POLICIES):
            pts = [x for x in thirds if x["model"] == m and x["policy"] == p]
            if pts:
                xs = [n + (k - 2.5) * 0.04 for n in (1, 2, 3)]
                ax.plot(
                    xs, [x["success"] * 100 for x in pts], marker="o", label=p, color=COLOURS[p]
                )
        ax.set_xticks([1, 2, 3], ["tickets 1-8", "9-16", "17-24"])
        ax.set_title(SHORT[m])
        ax.set_ylim(75, 101.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("tickets solved (%, axis from 75)")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Tickets solved by queue position: every dip is one of the three already-returned orders"
    )
    fig.tight_layout()
    fig.savefig(CHARTS / "by-position.png", dpi=150)
    plt.close(fig)


def chart_window(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, m in zip(axes, MODELS, strict=True):
        for p in POLICIES:
            run = next(
                (
                    r
                    for r in rows
                    if r["study"] == "queue"
                    and r["model"] == m
                    and r["policy"] == p
                    and r["order"] == 0
                ),
                None,
            )
            if run:
                ax.plot(
                    range(1, len(run["calls"]) + 1),
                    [c["window_tokens"] / 1000 for c in run["calls"]],
                    label=p,
                    color=COLOURS[p],
                    lw=1.4,
                )
        ax.set_title(f"{SHORT[m]}, queue order 0")
        ax.set_xlabel("model call")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("input tokens per request (thousands)")
    axes[1].legend(fontsize=8)
    fig.suptitle("What each request sends: the window under each policy")
    fig.tight_layout()
    fig.savefig(CHARTS / "window.png", dpi=150)
    plt.close(fig)


def chart_cost(queue: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.38
    for i, m in enumerate(MODELS):
        pts = {x["policy"]: x for x in queue if x["model"] == m}
        xs = [j + (i - 0.5) * width for j in range(len(POLICIES))]
        ax.bar(
            xs,
            [pts[p]["cost_per_run"] if p in pts else 0 for p in POLICIES],
            width,
            label=SHORT[m],
            color=["#93c5fd", "#1d4ed8"][i],
        )
    ax.set_xticks(range(len(POLICIES)), POLICIES)
    ax.set_ylabel("USD per 24-ticket queue")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Cost of working the whole queue, per policy (log scale)")
    fig.tight_layout()
    fig.savefig(CHARTS / "cost.png", dpi=150)
    plt.close(fig)


def chart_tools(tools: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, label in (
        (axes[0], "success", "tickets solved (%)"),
        (axes[1], "cost_per_episode", "USD per ticket"),
    ):
        for i, m in enumerate(MODELS):
            pts = {x["tools"]: x for x in tools if x["model"] == m}
            vals = [
                (pts[n][metric] * (100 if metric == "success" else 1)) if n in pts else 0
                for n in (10, 40)
            ]
            ax.bar(
                [j + (i - 0.5) * 0.38 for j in range(2)],
                vals,
                0.38,
                label=SHORT[m],
                color=["#93c5fd", "#1d4ed8"][i],
            )
        ax.set_xticks([0, 1], ["10 tools", "40 tools"])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(loc="lower right")
    wrong = sum(x["wrong_tool_calls"] for x in tools)
    n = sum(x["episodes"] for x in tools)
    fig.suptitle(
        f"10 tools or 40: {wrong} calls to a look-alike tool in {n} tickets; the extra tools cost tokens, not accuracy"
    )
    fig.tight_layout()
    fig.savefig(CHARTS / "tools.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rows = load()
    CHARTS.mkdir(parents=True, exist_ok=True)
    queue, thirds, tools, summaries = (
        queue_table(rows),
        third_table(rows),
        tools_table(rows),
        summary_table(rows),
    )
    for name, table in (
        ("queue.csv", queue),
        ("by_third.csv", thirds),
        ("tools.csv", tools),
        ("summaries.csv", summaries),
    ):
        write_csv(name, table)
    chart_by_third(thirds)
    chart_window(rows)
    chart_cost(queue)
    chart_tools(tools)
    for table in (queue, tools, summaries):
        for x in table:
            print(x)
        print()
    print(f"{len(rows)} runs, USD {sum(r['cost_usd'] for r in rows):.2f}")


if __name__ == "__main__":
    main()
