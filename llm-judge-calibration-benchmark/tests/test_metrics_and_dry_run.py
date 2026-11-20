"""The calibration arithmetic, checked on the article's own illustrative matrix, and the dry run."""

import math

from judge_the_judge.cases import build
from judge_the_judge.metrics import (
    Confusion,
    confusion,
    gate_simulation,
    majority,
    spearman,
    wilson,
)
from judge_the_judge.run import dry_run, verdict


def test_the_articles_illustrative_matrix():
    # Human FAIL (16): judge FAIL 12, PASS 4. Human PASS (64): judge FAIL 4, PASS 60.
    c = Confusion(tp=12, fn=4, fp=4, tn=60)
    assert c.agreement == 0.9  # "raw agreement looks strong"
    assert c.tpr == 0.75  # "misses a quarter of refund failures"
    assert abs(c.tnr - 60 / 64) < 1e-9
    assert abs(c.youden - (0.75 + 60 / 64 - 1)) < 1e-9
    lo, hi = wilson(12, 16)
    assert (
        round(lo, 2) == 0.51 and round(hi, 2) == 0.90
    )  # 51% to 90%: "barely pinned down" with 16 failures


def test_kappa_and_confusion():
    assert confusion(["FAIL", "PASS", "PASS"], ["FAIL", "PASS", "FAIL"]) == Confusion(1, 0, 1, 1)
    assert Confusion(10, 0, 0, 10).kappa == 1.0
    assert abs(Confusion(5, 5, 5, 5).kappa) < 1e-9


def test_spearman_with_ties_and_majority():
    assert abs(spearman([1, 2, 3], [10, 20, 30]) - 1) < 1e-9
    assert abs(spearman([1, 2, 3], [30, 20, 10]) + 1) < 1e-9
    assert not math.isnan(spearman([1, 1, 2, 2], [0, 1, 0, 1]))
    assert majority(["PASS", "FAIL", "PASS"]) == "PASS" and majority(["PASS", "FAIL"]) == "FAIL"


def test_gate_simulation_scales_by_youden():
    labels = ["FAIL"] * 40 + ["PASS"] * 160
    perfect = gate_simulation(labels, labels, 0.10, 0.15)
    assert abs(perfect["measured_delta"] - 0.05) < 0.01 and perfect["youden"] == 1
    half = (
        ["FAIL"] * 20 + ["PASS"] * 20 + ["PASS"] * 160
    )  # catches half the failures, no false alarms
    sim = gate_simulation(labels, half, 0.10, 0.15)
    assert abs(sim["measured_delta"] - 0.025) < 0.01 and abs(sim["predicted_delta"] - 0.025) < 1e-9


def test_score_gate():
    assert verdict("score", {"score": 7}) == "PASS" and verdict("score", {"score": 6}) == "FAIL"
    assert verdict("pf", {"verdict": "FAIL"}) == "FAIL" and verdict("pf", None) is None


def test_dry_run_passes():
    cases, pairs = build()
    assert dry_run(cases, pairs) == 0
