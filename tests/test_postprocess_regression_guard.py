"""A post-processing step that degrades the pooled metric must not be adopted."""
from __future__ import annotations

import numpy as np
import pytest

from colab.sota_postprocess import SOUP_REGRESSION_TOLERANCE, run_sota_postprocess


def _folds(seed=0, n=400):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(3):
        lab = rng.integers(0, 2, n).astype(np.float32)
        # Informative probabilities: AUC well above chance.
        pr = np.clip(0.5 + 0.30 * (lab - 0.5) + rng.normal(0, 0.10, n), 0, 1).astype(np.float32)
        out.append({"val_probs": pr, "val_labels": lab, "val_ids": [f"P{i}" for i in range(n)]})
    return out


class _Cfg:
    n_folds = 3
    checkpoint_dir = "/nonexistent"


def test_degrading_soup_is_discarded_and_flagged(monkeypatch):
    import colab.sota_postprocess as sp

    folds = _folds()

    def wrecking_soup(**kw):
        rng = np.random.default_rng(1)
        wrecked = [
            {**f, "val_probs": rng.random(len(f["val_probs"])).astype(np.float32)}
            for f in kw["fold_results"]
        ]
        return {"mode": "held_out"}, wrecked

    monkeypatch.setattr(sp, "run_fold_model_soup", wrecking_soup)
    monkeypatch.setattr(sp, "print_fold_soup_report", lambda r: None)

    report, current = run_sota_postprocess(
        proteins=[], esm_backbone=None, batch_converter=None, cfg=_Cfg(),
        fold_results=folds, apply_soup=True, calibrate=False,
    )
    soup = report["fold_soup"]
    assert soup["regression_detected"] is True
    assert soup["applied"] is False
    # The good predictions survived.
    assert np.allclose(current[0]["val_probs"], folds[0]["val_probs"])
    assert report["final_pooled"]["auc"] > 0.7


def test_improving_soup_is_adopted(monkeypatch):
    import colab.sota_postprocess as sp

    folds = _folds()

    def helpful_soup(**kw):
        better = [
            {**f, "val_probs": np.clip(
                f["val_labels"] * 0.9 + 0.05, 0, 1).astype(np.float32)}
            for f in kw["fold_results"]
        ]
        return {"mode": "held_out"}, better

    monkeypatch.setattr(sp, "run_fold_model_soup", helpful_soup)
    monkeypatch.setattr(sp, "print_fold_soup_report", lambda r: None)

    report, current = run_sota_postprocess(
        proteins=[], esm_backbone=None, batch_converter=None, cfg=_Cfg(),
        fold_results=folds, apply_soup=True, calibrate=False,
    )
    assert report["fold_soup"]["regression_detected"] is False
    assert report["fold_soup"]["applied"] is True
    assert report["final_pooled"]["auc"] > 0.95


def test_tolerance_allows_rounding_noise(monkeypatch):
    import colab.sota_postprocess as sp

    folds = _folds()

    def nearly_identical(**kw):
        same = [
            {**f, "val_probs": (f["val_probs"] + 1e-6).astype(np.float32)}
            for f in kw["fold_results"]
        ]
        return {"mode": "held_out"}, same

    monkeypatch.setattr(sp, "run_fold_model_soup", nearly_identical)
    monkeypatch.setattr(sp, "print_fold_soup_report", lambda r: None)
    report, _ = run_sota_postprocess(
        proteins=[], esm_backbone=None, batch_converter=None, cfg=_Cfg(),
        fold_results=folds, apply_soup=True, calibrate=False,
    )
    assert report["fold_soup"]["applied"] is True
    assert abs(report["fold_soup"]["delta_auc_pooled"]) < SOUP_REGRESSION_TOLERANCE


def test_degrading_calibration_is_discarded(monkeypatch):
    import colab.sota_postprocess as sp

    folds = _folds()

    def wrecking_cal(fold_results, method="temperature"):
        rng = np.random.default_rng(2)
        return (
            [{**f, "val_probs": rng.random(len(f["val_probs"])).astype(np.float32)}
             for f in fold_results],
            {"method": method},
        )

    monkeypatch.setattr(sp, "calibrate_fold_results", wrecking_cal)
    report, current = run_sota_postprocess(
        proteins=[], esm_backbone=None, batch_converter=None, cfg=_Cfg(),
        fold_results=folds, apply_soup=False, calibrate=True,
    )
    assert report["calibration"]["regression_detected"] is True
    assert report["calibration"]["applied"] is False
    assert report["final_pooled"]["auc"] > 0.7
