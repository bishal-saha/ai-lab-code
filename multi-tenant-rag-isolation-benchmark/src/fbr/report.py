"""Turn the three studies' results into tables and charts.

    uv run python -m fbr.report

Writes results/study_a.csv (pass x mode x k x tenant-size band), results/study_b.csv (the matrix),
results/study_c.csv (model x case), results/threshold.csv (the offline sweep) and results/charts/*.png.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fbr.study_a import MODES  # noqa: E402
from fbr.study_b import BUGS, DEFENCES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
BANDS = [
    ("large (over 10%)", 0.10, 1.0),
    ("medium (1-10%)", 0.01, 0.10),
    ("small (under 1%)", 0.0, 0.01),
]
COLOUR = {
    "post-filter": "#c2410c",
    "post-filter-4x": "#fb923c",
    "in-query": "#2563eb",
    "in-query-iter": "#15803d",
    "rls-only": "#7c3aed",
}
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]


def write_csv(name: str, table: list[dict]) -> None:
    with (RESULTS / name).open("w", newline="") as f:
        fields = list(
            dict.fromkeys(k for row in table for k in row)
        )  # rows can carry different fields
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(table)


def study_a(rows: list[dict]) -> list[dict]:
    out = []
    for p in ("auto", "hnsw"):
        for k in (5, 10):
            for band, lo, hi in [("all", 0, 1.0), *BANDS]:
                rs = [
                    r
                    for r in rows
                    if r["pass"] == p
                    and r["k"] == k
                    and lo <= r["share"] < hi + (1e-9 if hi == 1.0 else 0)
                ]
                if not rs:
                    continue
                row = {"pass": p, "k": k, "tenants": band, "queries": len(rs)}
                for m in MODES:
                    row[f"{m}_returned"] = round(mean(r[m]["returned"] for r in rs), 2)
                    row[f"{m}_recall"] = round(mean(r[m]["recall"] for r in rs), 3)
                    row[f"{m}_ms"] = round(mean(r[m]["ms"] for r in rs), 2)
                out.append(row)
    return out


def chart_a(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, p, title in (
        (axes[0], "auto", "planner chooses (tenant index available)"),
        (axes[1], "hnsw", "HNSW forced"),
    ):
        for m in MODES:
            pts = {}
            for r in rows:
                if r["pass"] == p and r["k"] == 10:
                    pts.setdefault(r["share"], []).append(r[m]["recall"])
            xs = sorted(pts)
            ax.plot(
                [x * 100 for x in xs],
                [mean(pts[x]) * 100 for x in xs],
                marker="o",
                ms=3,
                lw=1.3,
                label=m,
                color=COLOUR[m],
            )
        ax.set_xscale("log")
        ax.set_xlabel("tenant's share of the index (%, log)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("recall@10 against an exact scan (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Where the tenant filter goes: recall@10 by tenant size")
    fig.tight_layout()
    fig.savefig(CHARTS / "recall-by-tenant-size.png", dpi=150)
    plt.close(fig)


def chart_b(cells: list[dict]) -> None:
    colours = {"safe": "#bbf7d0", "blind": "#fde68a", "error": "#fdba74", "leaked": "#fca5a5"}
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for i, d in enumerate(DEFENCES):
        for j, b in enumerate(BUGS):
            res = next(c["result"] for c in cells if c["defence"] == d and c["bug"] == b)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colours[res], ec="white", lw=2))
            ax.text(j + 0.5, i + 0.5, res, ha="center", va="center", fontsize=9)
    ax.set_xlim(0, len(BUGS))
    ax.set_ylim(0, len(DEFENCES))
    ax.set_xticks(
        [j + 0.5 for j in range(len(BUGS))],
        [
            "no bug",
            "query forgets\ntenant",
            "query forgets\nrole",
            "app is table\nowner, no FORCE",
            "session SET on a\npooled connection",
            "cast without\nNULLIF",
        ],
        fontsize=8,
    )
    ax.set_yticks(
        [i + 0.5 for i in range(len(DEFENCES))], ["query filter only", "RLS only", "both"]
    )
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("The cross-tenant probe under each bug and defence")
    fig.tight_layout()
    fig.savefig(CHARTS / "isolation-matrix.png", dpi=150)
    plt.close(fig)


def study_c(rows: list[dict]) -> list[dict]:
    out = []
    for m in MODELS:
        for case in ("answerable", "other-tenant", "nowhere"):
            rs = [r for r in rows if r["model"] == m and r["case"] == case]
            if not rs:
                continue
            out.append(
                {
                    "model": m,
                    "case": case,
                    "answers": len(rs),
                    "refused": sum(r["refused"] for r in rs),
                    "correct": sum(r["correct"] for r in rs),
                    "supported": sum(r["supported"] for r in rs),
                    "confabulated": sum(r["confabulated"] for r in rs),
                    "invalid_citation": sum(r["invalid_citation"] for r in rs),
                    "cost_usd": round(sum(r["cost_usd"] for r in rs), 4),
                }
            )
    return out


def sweep(rows: list[dict]) -> list[dict]:
    """With a threshold T, a question whose closest chunk is farther than T is refused before the model;
    otherwise the model's own answer stands. The same runs give every T."""
    out = []
    for m in MODELS:
        rs = [r for r in rows if r["model"] == m]
        if not rs:
            continue
        for t in [x / 100 for x in range(30, 91, 2)]:
            ans = [r for r in rs if r["case"] == "answerable"]
            una = [r for r in rs if r["case"] != "answerable"]
            gated = lambda r, t=t: r["distances"][0] > t  # noqa: E731
            out.append(
                {
                    "model": m,
                    "threshold": t,
                    "answerable_correct": round(
                        sum((not gated(r)) and r["correct"] for r in ans) / len(ans), 3
                    ),
                    "answerable_refused_by_threshold": round(
                        sum(gated(r) for r in ans) / len(ans), 3
                    ),
                    "unanswerable_answered": round(
                        sum((not gated(r)) and r["confabulated"] for r in una) / len(una), 3
                    ),
                }
            )
    return out


def chart_c(rows: list[dict], swept: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for case, colour in (
        ("answerable", "#15803d"),
        ("other-tenant", "#c2410c"),
        ("nowhere", "#7c3aed"),
    ):
        ds = [
            r["distances"][0]
            for r in rows
            if r["case"] == case and r["model"] == MODELS[0] and r["repeat"] == 0
        ]
        axes[0].hist(ds, bins=20, range=(0.2, 0.9), alpha=0.55, label=case, color=colour)
    axes[0].set_xlabel("cosine distance to the closest retrieved chunk")
    axes[0].set_ylabel("questions")
    axes[0].set_title("How far the best chunk is, by question type")
    axes[0].legend(fontsize=8)
    for m, ls in zip(MODELS, ("-", "--"), strict=True):
        pts = [s for s in swept if s["model"] == m]
        if not pts:
            continue
        xs = [p["threshold"] for p in pts]
        axes[1].plot(
            xs,
            [p["answerable_correct"] * 100 for p in pts],
            ls,
            color="#15803d",
            label=f"{m[7:13]}: answerable answered correctly",
        )
        axes[1].plot(
            xs,
            [p["unanswerable_answered"] * 100 for p in pts],
            ls,
            color="#c2410c",
            label=f"{m[7:13]}: unanswerable answered anyway",
        )
    axes[1].set_xlabel("similarity threshold (max cosine distance)")
    axes[1].set_ylabel("%")
    axes[1].set_title("The threshold sweep (computed offline)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "refusals.png", dpi=150)
    plt.close(fig)


def main() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    a = [json.loads(x) for x in (RESULTS / "study_a.jsonl").read_text().splitlines()]
    b = json.loads((RESULTS / "study_b.json").read_text())
    c = [json.loads(x) for x in (RESULTS / "study_c.jsonl").read_text().splitlines()]
    ta, tc, sw = study_a(a), study_c(c), sweep(c)
    write_csv("study_a.csv", ta)
    write_csv(
        "study_b.csv",
        [
            {
                k: v
                for k, v in x.items()
                if k
                in (
                    "defence",
                    "bug",
                    "result",
                    "leaked_tenant",
                    "leaked_role",
                    "positive_control",
                    "detail",
                )
            }
            for x in b
        ],
    )
    write_csv("study_c.csv", tc)
    write_csv("threshold.csv", sw)
    chart_a(a)
    chart_b(b)
    chart_c(c, sw)
    for x in ta:
        if x["tenants"] in ("all", "small (under 1%)"):
            print({k: v for k, v in x.items() if not k.endswith("_ms")})
    for x in tc:
        print(x)
    print(f"study C cost USD {sum(r['cost_usd'] for r in c):.3f}")


if __name__ == "__main__":
    main()
