"""Turn results/judgements.jsonl into calibration tables and charts.

    uv run python -m judge_the_judge.report

Single judgements are reduced to one verdict per case and model by majority over
the repeats (the article's advice for a judge that varies between calls).
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

from judge_the_judge.cases import FAIL_TYPES, build  # noqa: E402
from judge_the_judge.judges import DESIGNS  # noqa: E402
from judge_the_judge.metrics import confusion, gate_simulation, majority, spearman, wilson  # noqa: E402
from judge_the_judge.run import verdict  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SHORT = {"claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5"}
COLOURS = {"claude-haiku-5-5": "#a5b4fc", "claude-sonnet-5-5": "#4f46e5"}
SCENARIOS = [(0.10, 0.15), (0.20, 0.10)]


def load() -> list[dict]:
    return [
        json.loads(line)
        for line in (RESULTS / "judgements.jsonl").read_text().splitlines()
        if line.strip()
    ]


def single_table(rows: list[dict], cases) -> list[dict]:
    by_id = {c.id: c for c in cases}
    out = []
    for model in sorted({r["model"] for r in rows}):
        for design in DESIGNS:
            per: dict[str, list] = defaultdict(list)
            for r in rows:
                if (
                    r["kind"] == "single"
                    and r["model"] == model
                    and r["design"] == design
                    and verdict(design, r["value"])
                ):
                    per[r["item"]].append(r)
            if not per:
                continue
            ids = sorted(per)
            labels = [by_id[i].label for i in ids]
            final = [majority([verdict(design, r["value"]) for r in per[i]]) for i in ids]
            cm = confusion(labels, final)
            consistent = sum(
                len({verdict(design, r["value"]) for r in per[i]}) == 1 for i in ids
            ) / len(ids)

            def pass_rate(label: str, length: str, ids=ids, final=final) -> float | None:
                return _pass_rate(by_id, ids, final, label, length)

            row = {
                "model": model,
                "design": design,
                "cases": len(ids),
                "tpr": round(cm.tpr, 4),
                "tpr_lo": round(wilson(cm.tp, cm.tp + cm.fn)[0], 4),
                "tpr_hi": round(wilson(cm.tp, cm.tp + cm.fn)[1], 4),
                "tnr": round(cm.tnr, 4),
                "tnr_lo": round(wilson(cm.tn, cm.tn + cm.fp)[0], 4),
                "tnr_hi": round(wilson(cm.tn, cm.tn + cm.fp)[1], 4),
                "agreement": round(cm.agreement, 4),
                "kappa": round(cm.kappa, 4),
                "youden": round(cm.youden, 4),
                "tp": cm.tp,
                "fn": cm.fn,
                "fp": cm.fp,
                "tn": cm.tn,
                "self_consistency": round(consistent, 4),
                "pass_rate_fail_short": pass_rate("FAIL", "short"),
                "pass_rate_fail_long": pass_rate("FAIL", "long"),
                "pass_rate_pass_short": pass_rate("PASS", "short"),
                "pass_rate_pass_long": pass_rate("PASS", "long"),
                **{
                    f"caught_{ft}": _share(
                        [
                            v == "FAIL"
                            for i, v in zip(ids, final, strict=True)
                            if by_id[i].fail_type == ft
                        ]
                    )
                    for ft in FAIL_TYPES
                },
                "spearman": None,
                "cost_usd": round(sum(r["cost_usd"] for i in ids for r in per[i]), 4),
            }
            if design == "score":
                mean_score = [
                    statistics.mean(int(r["value"]["score"]) for r in per[i]) for i in ids
                ]
                row["spearman"] = round(spearman(mean_score, [by_id[i].severity for i in ids]), 4)
            for p0, p1 in SCENARIOS:
                sim = gate_simulation(labels, final, p0, p1, seed=7)
                row[f"gate_{int(p0 * 100)}_{int(p1 * 100)}_measured"] = sim["measured_delta"]
                row[f"gate_{int(p0 * 100)}_{int(p1 * 100)}_predicted"] = sim["predicted_delta"]
            out.append(row)
    return out


def _pass_rate(
    by_id: dict, ids: list[str], final: list[str], label: str, length: str
) -> float | None:
    """Share of cases with this true label and length that the judge passed."""
    return _share(
        [
            v == "PASS"
            for i, v in zip(ids, final, strict=True)
            if by_id[i].label == label and by_id[i].length == length
        ]
    )


def _share(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def pair_table(rows: list[dict], pairs) -> list[dict]:
    by_id = {p.id: p for p in pairs}
    out = []
    for model in sorted({r["model"] for r in rows}):
        got: dict[tuple, dict] = {}
        for r in rows:
            if r["kind"] == "pair" and r["model"] == model and r["value"]:
                got[(r["item"], r["repeat"], r["order"])] = r["value"]["winner"]
        for kind in ("correct_vs_wrong", "short_vs_long"):
            chosen, consistent, first_wins, total = [], 0, 0, 0
            for (item, repeat, order), w in got.items():
                p = by_id[item]
                if p.kind != kind or order != "AB" or (item, repeat, "BA") not in got:
                    continue
                pick_ab = p.a if w == "A" else p.b  # in AB, A is p.a
                pick_ba = p.b if got[(item, repeat, "BA")] == "A" else p.a  # in BA, A is p.b
                consistent += pick_ab.id == pick_ba.id
                first_wins += (w == "A") + (got[(item, repeat, "BA")] == "A")
                total += 1
                chosen += [pick_ab, pick_ba]
            if not total:
                continue
            out.append(
                {
                    "model": model,
                    "kind": kind,
                    "comparisons": total,
                    "swap_consistent": round(consistent / total, 4),
                    "first_position_wins": round(first_wins / (2 * total), 4),
                    "picked_correct": _share([c.label == "PASS" for c in chosen])
                    if kind == "correct_vs_wrong"
                    else None,
                    "picked_long": _share([c.length == "long" for c in chosen])
                    if kind == "short_vs_long"
                    else None,
                }
            )
    return out


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def charts(table: list[dict], pairs: list[dict], out: Path) -> None:
    models = sorted({t["model"] for t in table})

    # 1. TPR and TNR with 95% intervals.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, sharey=True)
    for ax, (metric, title) in zip(
        axes, [("tpr", "Failures caught (TPR)"), ("tnr", "Good replies passed (TNR)")], strict=True
    ):
        for k, m in enumerate(models):
            rows = [t for t in table if t["model"] == m]
            xs = [DESIGNS.index(t["design"]) + (k - 0.5) * 0.3 for t in rows]
            ys = [t[metric] for t in rows]
            err = [
                [y - t[f"{metric}_lo"] for y, t in zip(ys, rows, strict=True)],
                [t[f"{metric}_hi"] - y for y, t in zip(ys, rows, strict=True)],
            ]
            ax.errorbar(
                xs,
                ys,
                yerr=err,
                fmt="o",
                color=COLOURS.get(m, "#4f46e5"),
                label=SHORT.get(m, m),
                capsize=3,
            )
        ax.set_xticks(range(len(DESIGNS)), DESIGNS, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        "Judge calibration on 200 replies with known labels (95% intervals)",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    _save(fig, out / "tpr-tnr")

    # 2. Raw agreement vs failures caught.
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    labels = [f"{SHORT.get(t['model'], t['model']).split()[0]}\n{t['design']}" for t in table]
    xs = range(len(table))
    ax.bar(
        [x - 0.2 for x in xs],
        [t["agreement"] for t in table],
        0.4,
        label="raw agreement",
        color="#c7d2fe",
    )
    bars = ax.bar(
        [x + 0.2 for x in xs],
        [t["tpr"] for t in table],
        0.4,
        label="failures caught (TPR)",
        color="#4f46e5",
    )
    for bar, t in zip(bars, table, strict=True):
        ax.annotate(
            f"{t['tpr']:.0%}",
            (bar.get_x() + bar.get_width() / 2, t["tpr"]),
            ha="center",
            va="bottom",
            fontsize=7,
            xytext=(0, 2),
            textcoords="offset points",
        )
    ax.set_xticks(list(xs), labels, fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Raw agreement against failures caught", fontsize=12, loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    _save(fig, out / "agreement-vs-tpr")

    # 3. Verbosity: how often wrong replies pass, short vs long.
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    for i, (key, label, colour) in enumerate(
        [
            ("pass_rate_fail_short", "wrong reply, short", "#c7d2fe"),
            ("pass_rate_fail_long", "wrong reply, long", "#1e1b4b"),
        ]
    ):
        vals = [t[key] or 0 for t in table]
        bars = ax.bar([x + (i - 0.5) * 0.4 for x in xs], vals, 0.4, label=label, color=colour)
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
    ax.set_xticks(list(xs), labels, fontsize=7)
    ax.set_ylim(0, max(0.3, max((t["pass_rate_fail_long"] or 0) for t in table) * 1.3))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Wrong replies the judge passed, by length", fontsize=12, loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    _save(fig, out / "verbosity")

    # 4. Pairwise: swap consistency and first-position wins.
    if pairs:
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
        names = [
            f"{SHORT.get(p['model'], p['model']).split()[0]}\n{'right vs wrong' if p['kind'] == 'correct_vs_wrong' else 'both right'}"
            for p in pairs
        ]
        px = range(len(pairs))
        for i, (key, label, colour) in enumerate(
            [
                ("swap_consistent", "same verdict after swapping A and B", "#4f46e5"),
                ("first_position_wins", "first position wins", "#c7d2fe"),
            ]
        ):
            vals = [p[key] for p in pairs]
            bars = ax.bar([x + (i - 0.5) * 0.4 for x in px], vals, 0.4, label=label, color=colour)
            for bar, v in zip(bars, vals, strict=True):
                ax.annotate(
                    f"{v:.0%}",
                    (bar.get_x() + bar.get_width() / 2, v),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    xytext=(0, 2),
                    textcoords="offset points",
                )
        ax.axhline(0.5, color="#94a3b8", linewidth=0.8, linestyle="--")
        ax.set_xticks(list(px), names, fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title("Pairwise judging: position bias", fontsize=12, loc="left", pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
        _save(fig, out / "pairwise")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    rows = load()
    cases, pairs = build()
    (RESULTS / "charts").mkdir(parents=True, exist_ok=True)
    table, ptable = single_table(rows, cases), pair_table(rows, pairs)
    _write_csv(RESULTS / "calibration.csv", table)
    if ptable:
        _write_csv(RESULTS / "pairwise.csv", ptable)
    charts(table, ptable, RESULTS / "charts")
    print(f"{len(rows)} judgements -> results/calibration.csv, pairwise.csv, charts/")
    for t in table:
        print(
            f"  {SHORT.get(t['model'], t['model']):10} {t['design']:10} TPR {t['tpr']:.0%} [{t['tpr_lo']:.0%}-{t['tpr_hi']:.0%}]  TNR {t['tnr']:.0%}  agree {t['agreement']:.0%}  kappa {t['kappa']:.2f}  long-wrong pass {t['pass_rate_fail_long'] or 0:.0%}  consistent {t['self_consistency']:.0%}  ${t['cost_usd']:.3f}"
        )
    for p in ptable:
        print(
            f"  pairs {SHORT.get(p['model'], p['model']):10} {p['kind']:16} swap-consistent {p['swap_consistent']:.0%}  first wins {p['first_position_wins']:.0%}  correct {p['picked_correct']}  long {p['picked_long']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
