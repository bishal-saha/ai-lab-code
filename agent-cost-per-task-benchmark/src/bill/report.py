"""Turn results/tickets.jsonl into tables and charts.

    uv run python -m bill.report

Writes results/conditions.csv (one row per condition), results/model.csv (the article's planning
model, filled from the pilot subset, against what was measured), results/charts/*.png, and prints
the headline numbers.

All-in cost per ticket follows the article: every attempt's tokens plus human time, divided by
tickets. Human time uses the article's illustrative placeholders, HANDOFF_USD and ABANDON_USD.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
from dataclasses import replace  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from bill.conditions import CONDITIONS  # noqa: E402
from bill.planning_model import fit, predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
HANDOFF_USD, ABANDON_USD = 6.00, 9.00  # the article's illustrative human costs
PILOT_TICKETS = [f"T-B{n:03d}" for n in range(1, 9)]
COLOUR = {
    "cache": "#2563eb",
    "no-cache": "#c2410c",
    "cache-break": "#ea580c",
    "cap": "#a21caf",
    "haiku": "#15803d",
    "phase": "#0891b2",
    "switch": "#7c3aed",
}


def load() -> list[dict]:
    return [
        json.loads(x) for x in (RESULTS / "tickets.jsonl").read_text().splitlines() if x.strip()
    ]


def p90(xs: list[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(0.9 * (len(s) - 1)))]


def condition_table(rows: list[dict]) -> list[dict]:
    out = []
    for c in CONDITIONS:
        rs = [r for r in rows if r["condition"] == c]
        if not rs:
            continue
        firsts = [r["attempts"][0] for r in rs]
        calls = [x for r in rs for a in r["attempts"] for x in a["calls"]]
        cost = [r["cost_usd"] for r in rs]
        handoffs = sum(r["final"] == "handoff" for r in rs)
        abandoned = sum(r["final"] == "abandoned" for r in rs)
        resolved = sum(r["final"] == "resolved" for r in rs)
        out.append(
            {
                "condition": c,
                "tickets": len(rs),
                "resolved": round(resolved / len(rs), 3),
                "handoff": round(handoffs / len(rs), 3),
                "abandoned": round(abandoned / len(rs), 3),
                "first_attempt_resolved": round(
                    sum(a["outcome"] == "resolved" for a in firsts) / len(rs), 3
                ),
                "attempts_per_ticket": round(mean(r["n_attempts"] for r in rs), 3),
                "steps_first_attempt": round(mean(len(a["calls"]) for a in firsts), 2),
                "cache_hit": round(
                    sum(x["cache_read"] for x in calls) / sum(x["window_tokens"] for x in calls), 3
                ),
                "first_attempt_mean": round(mean(a["cost_usd"] for a in firsts), 5),
                "ticket_mean": round(mean(cost), 5),
                "ticket_median": round(median(cost), 5),
                "ticket_p90": round(p90(cost), 5),
                "ticket_max": round(max(cost), 5),
                "tokens_per_resolved": round(sum(cost) / resolved, 5) if resolved else float("nan"),
                "all_in_per_ticket": round(
                    (sum(cost) + HANDOFF_USD * handoffs + ABANDON_USD * abandoned) / len(rs), 4
                ),
            }
        )
    return out


def model_table(rows: list[dict], conds: list[dict]) -> tuple[list[dict], dict]:
    """The article's model filled from the pilot subset (tickets 1-8, repeat 0, cached baseline), as
    specified: one whole step count (the pilot's median). Alongside, the same model averaged over
    the baseline's measured step counts: the model takes one integer, and real tickets vary."""
    pilot = [
        r["attempts"][0]
        for r in rows
        if r["condition"] == "cache" and r["repeat"] == 0 and r["ticket"] in PILOT_TICKETS
    ]
    base = fit(pilot)
    mix = [len(r["attempts"][0]["calls"]) for r in rows if r["condition"] == "cache"]
    measured = {x["condition"]: x["first_attempt_mean"] for x in conds}
    out = []
    for c in CONDITIONS:
        if c not in measured:
            continue
        p = predict(c, base)
        pm = mean(predict(c, replace(base, steps=n)) for n in mix)
        out.append(
            {
                "condition": c,
                "measured": measured[c],
                "predicted": round(p, 5),
                "error_pct": round(100 * (p - measured[c]) / measured[c], 1),
                "predicted_step_mix": round(pm, 5),
                "error_step_mix_pct": round(100 * (pm - measured[c]) / measured[c], 1),
            }
        )
    fitted = {
        k: getattr(base, k)
        for k in ("steps", "prefix_tokens", "tokens_added_per_step", "output_tokens_per_step")
    }
    fitted["baseline_mean_steps"] = round(mean(mix), 2)
    return out, fitted


def spreadsheet_error(rows: list[dict]) -> dict:
    """The article's warning: one step measured and multiplied by the step count."""
    firsts = [r["attempts"][0] for r in rows if r["condition"] == "no-cache"]
    ratio_in = [
        sum(c["window_tokens"] for c in a["calls"])
        / (a["calls"][0]["window_tokens"] * len(a["calls"]))
        for a in firsts
    ]
    ratio_cost = [a["cost_usd"] / (a["calls"][0]["cost_usd"] * len(a["calls"])) for a in firsts]
    return {
        "attempts": len(firsts),
        "input_actual_over_naive": round(mean(ratio_in), 3),
        "cost_actual_over_naive": round(mean(ratio_cost), 3),
    }


def write_csv(name: str, table: list[dict]) -> None:
    with (RESULTS / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def chart_costs(rows: list[dict], conds: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, c in enumerate(x["condition"] for x in conds):
        ys = [r["cost_usd"] * 100 for r in rows if r["condition"] == c]
        xs = [i + ((k % 9) - 4) * 0.035 for k in range(len(ys))]
        ax.scatter(xs, ys, s=9, alpha=0.45, color=COLOUR[c])
        row = next(x for x in conds if x["condition"] == c)
        ax.hlines(row["ticket_median"] * 100, i - 0.3, i + 0.3, color="black", lw=2)
        ax.hlines(
            row["ticket_p90"] * 100, i - 0.3, i + 0.3, color="black", lw=1, linestyles="dotted"
        )
    ax.set_xticks(range(len(conds)), [x["condition"] for x in conds])
    ax.set_ylabel("US cents per ticket (log)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(
        "Token cost of one ticket, per condition (dots: tickets; solid: median; dotted: p90)"
    )
    fig.tight_layout()
    fig.savefig(CHARTS / "cost-per-ticket.png", dpi=150)
    plt.close(fig)


def chart_model(model: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    xs = list(range(len(model)))
    w = 0.27
    ax.bar(
        [x - w for x in xs],
        [m["measured"] * 100 for m in model],
        w,
        label="measured",
        color="#1d4ed8",
    )
    ax.bar(
        xs,
        [m["predicted"] * 100 for m in model],
        w,
        label="model, pilot's median steps (as specified)",
        color="#93c5fd",
    )
    ax.bar(
        [x + w for x in xs],
        [m["predicted_step_mix"] * 100 for m in model],
        w,
        label="model, over the measured step mix",
        color="#a5b4fc",
    )
    for x, m in zip(xs, model, strict=True):
        ax.annotate(
            f"{m['error_pct']:+.0f}%",
            (x, m["predicted"] * 100),
            ha="center",
            va="bottom",
            fontsize=7,
        )
        ax.annotate(
            f"{m['error_step_mix_pct']:+.0f}%",
            (x + w, m["predicted_step_mix"] * 100),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xticks(xs, [m["condition"] for m in model])
    ax.set_ylabel("US cents per attempt")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(
        "The article's planning model, filled from an 8-ticket pilot, against the real agent"
    )
    fig.tight_layout()
    fig.savefig(CHARTS / "model-vs-measured.png", dpi=150)
    plt.close(fig)


def chart_history(rows: list[dict]) -> None:
    firsts = [r["attempts"][0] for r in rows if r["condition"] == "no-cache"]
    n = round(median(len(a["calls"]) for a in firsts))
    same = [a for a in firsts if len(a["calls"]) == n]
    per_step = [mean(a["calls"][i]["window_tokens"] for a in same) for i in range(n)]
    cum = [sum(per_step[: i + 1]) / 1000 for i in range(n)]
    naive = [per_step[0] * (i + 1) / 1000 for i in range(n)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        range(1, n + 1), cum, marker="o", label="actual: history re-sent each step", color="#c2410c"
    )
    ax.plot(
        range(1, n + 1),
        naive,
        marker="o",
        ls="--",
        label="spreadsheet: first step x steps",
        color="#64748b",
    )
    ax.set_xlabel("model call")
    ax.set_ylabel("cumulative input tokens (thousands)")
    ax.set_xticks(range(1, n + 1))
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(f"Input tokens across a {n}-call ticket (no caching, {len(same)} tickets)")
    fig.tight_layout()
    fig.savefig(CHARTS / "history.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rows = load()
    CHARTS.mkdir(parents=True, exist_ok=True)
    conds = condition_table(rows)
    model, fitted = model_table(rows, conds)
    write_csv("conditions.csv", conds)
    write_csv("model.csv", model)
    sheet = spreadsheet_error(rows)
    (RESULTS / "summary.json").write_text(
        json.dumps(
            {
                "fitted_from_pilot": fitted,
                "spreadsheet": sheet,
                "tickets": len(rows),
                "usd": round(sum(r["cost_usd"] for r in rows), 3),
            },
            indent=2,
        )
        + "\n"
    )
    chart_costs(rows, conds)
    chart_model(model)
    chart_history(rows)
    for x in conds:
        print(x)
    print()
    for x in model:
        print(x)
    print("\nfitted from the pilot:", fitted, "\nspreadsheet:", sheet)
    print(f"{len(rows)} tickets, USD {sum(r['cost_usd'] for r in rows):.2f}")


if __name__ == "__main__":
    main()
