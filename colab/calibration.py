"""
Post-hoc calibration on OOF predictions (temperature scaling).
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_temperature_scaling(
    labels: np.ndarray,
    probs: np.ndarray,
    temperatures: Optional[np.ndarray] = None,
) -> dict:
    """
    Grid-search temperature T on logits: sigmoid(logit(p) / T).

    T > 1 softens overconfident predictions; T < 1 sharpens.
    """
    labels = np.asarray(labels, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    if len(np.unique(labels)) < 2:
        return {"temperature": 1.0, "insufficient_data": True}

    if temperatures is None:
        temperatures = np.arange(0.5, 3.05, 0.05)

    logits = _logit(probs)
    best_t = 1.0
    best_nll = float("inf")
    for t in temperatures:
        calibrated = _sigmoid(logits / t)
        nll = -np.mean(
            labels * np.log(calibrated + 1e-8)
            + (1.0 - labels) * np.log(1.0 - calibrated + 1e-8)
        )
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    cal_probs = apply_temperature(probs, best_t)
    return {
        "temperature": best_t,
        "nll_before": float(-np.mean(
            labels * np.log(probs + 1e-8) + (1.0 - labels) * np.log(1.0 - probs + 1e-8)
        )),
        "nll_after": float(best_nll),
        "auc_before": float(roc_auc_score(labels, probs)),
        "auc_after": float(roc_auc_score(labels, cal_probs)),
        "ap_after": float(average_precision_score(labels, cal_probs)),
        "insufficient_data": False,
    }


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return np.asarray(probs, dtype=np.float32)
    logits = _logit(probs)
    return _sigmoid(logits / temperature).astype(np.float32)


def fit_isotonic_calibration(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Isotonic regression calibrator (can help AP; use with care on small OOF)."""
    labels = np.asarray(labels, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    if len(np.unique(labels)) < 2 or len(labels) < 100:
        return {"insufficient_data": True}
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, labels)
    cal = iso.predict(probs).astype(np.float32)
    return {
        "insufficient_data": False,
        "auc_before": float(roc_auc_score(labels, probs)),
        "auc_after": float(roc_auc_score(labels, cal)),
        "ap_before": float(average_precision_score(labels, probs)),
        "ap_after": float(average_precision_score(labels, cal)),
        "calibrator": "isotonic",
    }


def apply_isotonic(probs: np.ndarray, iso: IsotonicRegression) -> np.ndarray:
    return iso.predict(probs).astype(np.float32)


def calibrate_fold_results(
    fold_results: list,
    method: str = "temperature",
) -> tuple[list, dict]:
    """Return fold_results with calibrated val_probs and calibration report."""
    all_probs = np.concatenate([r["val_probs"] for r in fold_results])
    all_labels = np.concatenate([r["val_labels"] for r in fold_results])

    if method == "temperature":
        report = fit_temperature_scaling(all_labels, all_probs)
        if report.get("insufficient_data"):
            return fold_results, report
        t = report["temperature"]
        updated = []
        for r in fold_results:
            fr = dict(r)
            fr["val_probs"] = apply_temperature(np.asarray(r["val_probs"]), t)
            fr["calibrated"] = True
            updated.append(fr)
        report["method"] = "temperature"
        return updated, report

    if method == "isotonic":
        iso_report = fit_isotonic_calibration(all_labels, all_probs)
        if iso_report.get("insufficient_data"):
            return fold_results, iso_report
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(all_probs, all_labels)
        updated = []
        for r in fold_results:
            fr = dict(r)
            fr["val_probs"] = apply_isotonic(np.asarray(r["val_probs"]), iso)
            fr["calibrated"] = True
            updated.append(fr)
        iso_report["method"] = "isotonic"
        return updated, iso_report

    return fold_results, {"method": "none", "skipped": True}


def save_calibration_report(report: dict, path: str = "calibration_report.json") -> str:
    out = {k: v for k, v in report.items() if k != "calibrator"}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path
