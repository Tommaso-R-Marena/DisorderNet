"""Calibrated + conformal per-residue confidence for the GPU/LoRA pipeline.

The CPU model gained trustworthy uncertainty via confidence.py (isotonic
calibration + split-conformal prediction sets). This module brings the same
capability to the GPU cross-validation results produced by
``colab.disordernet_gpu.run_cross_validation``.

Given ``fold_results`` (each a dict with ``val_probs`` / ``val_labels`` from an
out-of-fold GPU model), it does **cross-fitted** calibration and conformal
thresholding (fit on the other folds, apply to the held-out fold — no leakage),
augments each fold result with calibrated probabilities and conformal decisions,
and returns a pooled report.

For deployment on new sequences, ``fit_confidence`` returns a single
(calibrator, conformal-threshold) pair fit on all OOF predictions, and
``apply_confidence`` turns raw model probabilities into calibrated probabilities
plus conformal decisions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from confidence import (
    apply_calibrator,
    conformal_quantile,
    conformal_report,
    conformal_sets,
    expected_calibration_error,
    fit_calibrator,
)
from sklearn.metrics import roc_auc_score

# decision codes (match confidence.conformal_sets): 1=disorder, 0=order, -1=abstain, 2=empty
DECISION_LABEL = {1: "disorder", 0: "order", -1: "abstain", 2: "uncertain"}


def _pool(fold_results, key):
    return np.concatenate([np.asarray(fr[key], dtype=np.float64) for fr in fold_results])


def add_confidence_to_fold_results(fold_results: list, alpha: float = 0.10,
                                   class_conditional: bool = True) -> dict:
    """Cross-fit isotonic calibration + conformal on GPU OOF predictions.

    Augments each fold result in place with ``val_probs_calibrated`` and
    ``val_conformal_decision``; returns a pooled report. Calibration is monotonic,
    so pooled AUC is unchanged (reported for reference).
    """
    n = len(fold_results)
    if n < 2:
        raise ValueError("need >=2 folds to cross-fit calibration without leakage")

    for k in range(n):
        cal_p = np.concatenate([np.asarray(fold_results[j]["val_probs"], dtype=np.float64)
                                for j in range(n) if j != k])
        cal_y = np.concatenate([np.asarray(fold_results[j]["val_labels"], dtype=np.float64)
                                for j in range(n) if j != k])
        iso = fit_calibrator(cal_p, cal_y)
        q = conformal_quantile(apply_calibrator(iso, cal_p), cal_y,
                               alpha=alpha, class_conditional=class_conditional)
        fr = fold_results[k]
        cp = apply_calibrator(iso, np.asarray(fr["val_probs"], dtype=np.float64))
        fr["val_probs_calibrated"] = cp
        fr["val_conformal_decision"] = conformal_sets(cp, q)["decision"]

    y = _pool(fold_results, "val_labels").astype(np.int64)
    p_raw = _pool(fold_results, "val_probs")
    p_cal = _pool(fold_results, "val_probs_calibrated")
    decision = np.concatenate([np.asarray(fr["val_conformal_decision"]) for fr in fold_results])

    rep = conformal_report(p_cal, y, conformal_quantile(p_cal, y, alpha=alpha,
                                                        class_conditional=class_conditional))
    # conformal_report recomputes q on pooled data for its coverage summary; the
    # per-fold decisions above are the leakage-free ones used downstream.
    return {
        "alpha": alpha,
        "auc": float(roc_auc_score(y, p_raw)) if len(np.unique(y)) > 1 else float("nan"),
        "ece_raw": expected_calibration_error(y, p_raw),
        "ece_calibrated": expected_calibration_error(y, p_cal),
        "empirical_coverage": float(np.mean(np.where(y == 1,
            np.isin(decision, (1,)) | (decision == -1),
            np.isin(decision, (0,)) | (decision == -1)))),
        "confident_rate": float(np.isin(decision, (0, 1)).mean()),
        "abstain_rate": float((decision == -1).mean()),
        "selective_accuracy": (float((decision[np.isin(decision, (0, 1))] ==
                                      y[np.isin(decision, (0, 1))]).mean())
                               if np.isin(decision, (0, 1)).any() else float("nan")),
        "pooled_report": rep,
    }


def fit_confidence(fold_results: list, alpha: float = 0.10, class_conditional: bool = True):
    """Fit a deployable (calibrator, conformal-threshold) pair on all OOF predictions."""
    p = _pool(fold_results, "val_probs")
    y = _pool(fold_results, "val_labels")
    iso = fit_calibrator(p, y)
    q = conformal_quantile(apply_calibrator(iso, p), y, alpha=alpha,
                           class_conditional=class_conditional)
    return {"calibrator": iso, "conformal_q": q, "alpha": alpha}


def apply_confidence(confidence: dict, probs) -> dict:
    """Apply a fitted confidence bundle to raw model probabilities for a new protein."""
    cp = apply_calibrator(confidence["calibrator"], np.asarray(probs, dtype=np.float64))
    dec = conformal_sets(cp, confidence["conformal_q"])["decision"]
    return {
        "p_calibrated": cp.astype(np.float32),
        "decision": dec,
        "decision_labels": [DECISION_LABEL[int(d)] for d in dec],
    }
