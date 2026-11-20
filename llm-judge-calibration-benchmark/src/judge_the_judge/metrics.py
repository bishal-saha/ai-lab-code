"""The calibration arithmetic, from the article, as plain functions.

The positive class is FAIL: a judge's job at a gate is to catch failures.
    TPR  failures the judge marks FAIL       (catch rate)
    TNR  passes the judge marks PASS
Raw agreement and Cohen's kappa are reported alongside, so the gap the article
warns about (agreement that looks strong while the TPR is weak) is visible.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Confusion:
    tp: int  # human FAIL, judge FAIL
    fn: int  # human FAIL, judge PASS
    fp: int  # human PASS, judge FAIL
    tn: int  # human PASS, judge PASS

    @property
    def n(self) -> int:
        return self.tp + self.fn + self.fp + self.tn

    @property
    def tpr(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def tnr(self) -> float:
        return self.tn / (self.tn + self.fp) if self.tn + self.fp else float("nan")

    @property
    def agreement(self) -> float:
        return (self.tp + self.tn) / self.n

    @property
    def kappa(self) -> float:
        po = self.agreement
        p_fail = ((self.tp + self.fn) / self.n) * ((self.tp + self.fp) / self.n)
        p_pass = ((self.fp + self.tn) / self.n) * ((self.fn + self.tn) / self.n)
        pe = p_fail + p_pass
        return (po - pe) / (1 - pe) if pe < 1 else float("nan")

    @property
    def youden(self) -> float:
        """TPR + TNR - 1: the factor a measured change is scaled by."""
        return self.tpr + self.tnr - 1


def confusion(labels: list[str], verdicts: list[str]) -> Confusion:
    c = Counter(zip(labels, verdicts, strict=True))
    return Confusion(
        c[("FAIL", "FAIL")], c[("FAIL", "PASS")], c[("PASS", "FAIL")], c[("PASS", "PASS")]
    )


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return cov / math.sqrt(vx * vy) if vx and vy else float("nan")


def majority(verdicts: list[str]) -> str:
    """The verdict most repeats gave; a tie goes to FAIL (the cautious side)."""
    c = Counter(verdicts)
    return "FAIL" if c["FAIL"] >= c["PASS"] else "PASS"


def gate_simulation(
    labels: list[str],
    verdicts: list[str],
    p0: float,
    p1: float,
    n: int = 1000,
    reps: int = 300,
    seed: int = 0,
) -> dict:
    """Draw baseline and candidate sets with true failure rates p0 and p1 from the judged cases,
    and compare the judge's measured change in failure rate with the true change."""
    rng = random.Random(seed)
    fails = [v for lbl, v in zip(labels, verdicts, strict=True) if lbl == "FAIL"]
    passes = [v for lbl, v in zip(labels, verdicts, strict=True) if lbl == "PASS"]

    def measured(p: float) -> float:
        k = round(n * p)
        sample = rng.choices(fails, k=k) + rng.choices(passes, k=n - k)
        return sum(v == "FAIL" for v in sample) / n

    deltas = [measured(p1) - measured(p0) for _ in range(reps)]
    c = confusion(labels, verdicts)
    return {
        "true_delta": round(p1 - p0, 4),
        "measured_delta": round(sum(deltas) / reps, 4),
        "predicted_delta": round((p1 - p0) * c.youden, 4),
        "youden": round(c.youden, 4),
    }
