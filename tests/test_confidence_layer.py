"""Tests for the GPU-pipeline confidence layer (colab/confidence_layer.py)."""
from __future__ import annotations

import numpy as np
import pytest

from colab.confidence_layer import (
    add_confidence_to_fold_results,
    apply_confidence,
    fit_confidence,
    DECISION_LABEL,
)


def _make_fold_results(n_folds=5, per_fold=4000, seed=0):
    """Synthetic OOF folds: well-ranked but miscalibrated (over/under-confident)."""
    rng = np.random.RandomState(seed)
    folds = []
    for f in range(n_folds):
        true_p = rng.uniform(0, 1, per_fold)
        y = (rng.uniform(0, 1, per_fold) < true_p).astype(np.float32)
        p = np.clip(true_p ** 0.5, 0, 1)  # miscalibrated monotone transform
        folds.append({"fold": f + 1, "val_probs": p.astype(np.float32),
                      "val_labels": y})
    return folds


def test_add_confidence_augments_and_reduces_ece():
    folds = _make_fold_results()
    rep = add_confidence_to_fold_results(folds, alpha=0.1)
    assert rep["ece_calibrated"] < rep["ece_raw"]
    for fr in folds:
        assert "val_probs_calibrated" in fr
        assert "val_conformal_decision" in fr
        assert len(fr["val_probs_calibrated"]) == len(fr["val_probs"])
        assert set(np.unique(fr["val_conformal_decision"])).issubset({-1, 0, 1, 2})


def test_add_confidence_preserves_auc():
    from sklearn.metrics import roc_auc_score
    folds = _make_fold_results(seed=1)
    y = np.concatenate([f["val_labels"] for f in folds])
    p_raw = np.concatenate([f["val_probs"] for f in folds])
    rep = add_confidence_to_fold_results(folds, alpha=0.1)
    p_cal = np.concatenate([f["val_probs_calibrated"] for f in folds])
    # monotonic calibration keeps ranking within tie-tolerance
    assert abs(roc_auc_score(y, p_raw) - roc_auc_score(y, p_cal)) < 5e-3
    assert abs(rep["auc"] - roc_auc_score(y, p_raw)) < 1e-9


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_conformal_coverage_holds(alpha):
    folds = _make_fold_results(per_fold=6000, seed=2)
    rep = add_confidence_to_fold_results(folds, alpha=alpha)
    assert rep["empirical_coverage"] >= (1 - alpha) - 0.03


def test_fit_and_apply_confidence_roundtrip():
    folds = _make_fold_results(seed=3)
    conf = fit_confidence(folds, alpha=0.1)
    out = apply_confidence(conf, folds[0]["val_probs"])
    L = len(folds[0]["val_probs"])
    assert len(out["p_calibrated"]) == L
    assert len(out["decision_labels"]) == L
    assert out["p_calibrated"].min() >= 0 and out["p_calibrated"].max() <= 1
    assert all(lab in DECISION_LABEL.values() for lab in set(out["decision_labels"]))


def test_requires_two_folds():
    with pytest.raises(ValueError):
        add_confidence_to_fold_results(_make_fold_results(n_folds=1))
