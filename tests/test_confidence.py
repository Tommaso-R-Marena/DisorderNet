"""Tests for the per-residue confidence module (calibration + conformal)."""

from __future__ import annotations

import numpy as np
import pytest

from confidence import (
    apply_calibrator,
    conformal_quantile,
    conformal_report,
    conformal_sets,
    expected_calibration_error,
    fit_calibrator,
)


def _miscalibrated_data(n=20000, seed=0):
    """Labels with a known rate, plus deliberately over-confident probabilities."""
    rng = np.random.RandomState(seed)
    true_p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < true_p).astype(int)
    # squash toward extremes -> miscalibrated scores that still rank well
    prob = np.clip(true_p ** 0.5, 0, 1)
    prob = np.where(true_p > 0.5, np.minimum(1.0, prob * 1.15), prob * 0.85)
    return y, prob.astype(np.float32)


def test_ece_zero_for_perfect_calibration():
    rng = np.random.RandomState(1)
    p = rng.uniform(0, 1, 50000)
    y = (rng.uniform(0, 1, 50000) < p).astype(int)
    assert expected_calibration_error(y, p, n_bins=15) < 0.02


def test_isotonic_calibration_reduces_ece():
    y, prob = _miscalibrated_data()
    # split into calibration / test halves (no leakage)
    half = len(y) // 2
    iso = fit_calibrator(prob[:half], y[:half])
    cal = apply_calibrator(iso, prob[half:])
    ece_before = expected_calibration_error(y[half:], prob[half:])
    ece_after = expected_calibration_error(y[half:], cal)
    assert ece_after < ece_before
    assert ece_after < 0.03


def test_calibration_is_monotonic_preserves_ranking():
    from sklearn.metrics import roc_auc_score

    y, prob = _miscalibrated_data()
    half = len(y) // 2
    iso = fit_calibrator(prob[:half], y[:half])
    cal = apply_calibrator(iso, prob[half:])
    auc_before = roc_auc_score(y[half:], prob[half:])
    auc_after = roc_auc_score(y[half:], cal)
    # Isotonic is monotonic non-decreasing, so it never improves ranking and only
    # marginally lowers AUC via ties (flat segments scored as 0.5).
    assert auc_after <= auc_before + 1e-9
    assert (auc_before - auc_after) < 5e-3


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_conformal_coverage_guarantee(alpha):
    y, prob = _miscalibrated_data(n=40000, seed=2)
    half = len(y) // 2
    q = conformal_quantile(prob[:half], y[:half], alpha=alpha, class_conditional=True)
    rep = conformal_report(prob[half:], y[half:], q)
    # class-conditional split-conformal: coverage within each class >= 1 - alpha
    # (allow a small finite-sample slack)
    assert rep["coverage_disorder"] >= (1 - alpha) - 0.03
    assert rep["coverage_order"] >= (1 - alpha) - 0.03


def test_conformal_tighter_alpha_gives_more_abstention():
    y, prob = _miscalibrated_data(n=40000, seed=3)
    half = len(y) // 2
    q_loose = conformal_quantile(prob[:half], y[:half], alpha=0.30)
    q_tight = conformal_quantile(prob[:half], y[:half], alpha=0.02)
    rep_loose = conformal_report(prob[half:], y[half:], q_loose)
    rep_tight = conformal_report(prob[half:], y[half:], q_tight)
    # demanding higher coverage (smaller alpha) => abstain more often
    assert rep_tight["abstain_rate"] >= rep_loose["abstain_rate"]
    assert rep_tight["coverage"] >= rep_loose["coverage"]


def test_conformal_sets_decisions_consistent():
    prob = np.array([0.99, 0.5, 0.01], dtype=np.float32)
    sets = conformal_sets(prob, {0: 0.3, 1: 0.3})
    # high prob -> disorder in set; low prob -> order in set
    assert sets["has_disorder"][0] and not sets["has_order"][0]
    assert sets["has_order"][2] and not sets["has_disorder"][2]
    assert set(np.unique(sets["decision"])).issubset({-1, 0, 1, 2})
