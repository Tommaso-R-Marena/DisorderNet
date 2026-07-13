"""
Learned meta-ensemble for OOF predictions (replaces coarse grid search).

Fits a regularized logistic stacker on pooled OOF residues.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from colab.biological_utility import align_fold_predictions
from colab.inference_fusion import compute_pooled_metrics, write_fused_probs_to_fold_results


def _build_stacker_matrix(
    aligned: list[dict],
    streams: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack aligned prediction streams into (N, n_streams) feature matrix."""
    names = list(streams.keys())
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []

    for item in aligned:
        pid = item["id"]
        cols = []
        ok = True
        for name in names:
            arr = streams[name].get(pid)
            if arr is None:
                ok = False
                break
            gpu_p = np.asarray(item["probs"], dtype=np.float32)
            if len(arr) != len(gpu_p):
                ok = False
                break
            cols.append(arr)
        if not ok:
            continue
        mat = np.stack(cols, axis=1)
        chunks_x.append(mat)
        chunks_y.append(np.asarray(item["labels"], dtype=np.float32))

    if not chunks_x:
        raise ValueError("No aligned residues for meta-ensemble stacking")
    return np.vstack(chunks_x), np.concatenate(chunks_y)


def fit_meta_stacker(
    labels: np.ndarray,
    features: np.ndarray,
    C: float = 0.5,
) -> tuple[LogisticRegression, StandardScaler]:
    """Fit L2-regularized logistic meta-learner."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    model = LogisticRegression(
        C=C,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X, labels.astype(np.int8))
    return model, scaler


def apply_meta_stacker(
    proteins: list,
    fold_results: list,
    streams: dict[str, dict[str, np.ndarray]],
    n_folds: int = 5,
    C: float = 0.5,
) -> tuple[dict, list]:
    """
    Learned blend of named prediction streams on OOF residues.

    streams: {"gpu": probs_by_id, "v6": ..., "physics": ...}
    """
    before = compute_pooled_metrics(fold_results)
    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)

    try:
        X, y = _build_stacker_matrix(aligned, streams)
    except ValueError as exc:
        return {"skipped": True, "reason": str(exc)}, fold_results

    if len(np.unique(y)) < 2:
        return {"skipped": True, "reason": "insufficient label diversity"}, fold_results

    model, scaler = fit_meta_stacker(y, X, C=C)
    X_scaled = scaler.transform(X)
    stacked_probs = model.predict_proba(X_scaled)[:, 1].astype(np.float32)

    stream_names = list(streams.keys())
    coefs = dict(zip(stream_names, model.coef_[0].tolist()))
    report_fit = {
        "stream_names": stream_names,
        "coefficients": coefs,
        "intercept": float(model.intercept_[0]),
        "train_auc": float(roc_auc_score(y, stacked_probs)),
        "train_ap": float(average_precision_score(y, stacked_probs)),
    }

    offset = 0
    aligned_stacked = []
    for item in aligned:
        pid = item["id"]
        n = len(item["probs"])
        if all(pid in streams[k] for k in stream_names):
            new_item = dict(item)
            new_item["probs"] = stacked_probs[offset:offset + n].copy()
            new_item["meta_ensemble"] = True
            aligned_stacked.append(new_item)
            offset += n
        else:
            aligned_stacked.append(item)

    fold_results_stacked = write_fused_probs_to_fold_results(
        proteins, fold_results, aligned_stacked, n_folds=n_folds,
    )
    for fr in fold_results_stacked:
        fr["meta_ensemble"] = True
        fr["meta_coefficients"] = coefs

    after = compute_pooled_metrics(fold_results_stacked)
    report = {
        "fit": report_fit,
        "before": {"pooled": {k: before[k] for k in ("auc", "ap", "n_residues")}},
        "after": {"pooled": {k: after[k] for k in ("auc", "ap", "n_residues")}},
        "delta_auc_pooled": after["auc"] - before["auc"],
        "delta_ap_pooled": after["ap"] - before["ap"],
        "gap_to_esmdispred": 0.895 - after["auc"],
        "method": "logistic_meta_stacker",
    }
    return report, fold_results_stacked


def print_meta_ensemble_report(report: dict) -> None:
    print(f"\n{'═' * 64}")
    print(" LEARNED META-ENSEMBLE (logistic stacker)")
    print(f"{'═' * 64}")
    if report.get("skipped"):
        print(f"  Skipped: {report.get('reason')}")
        return
    fit = report["fit"]
    print(f"  Streams: {', '.join(fit['stream_names'])}")
    for name, coef in fit["coefficients"].items():
        print(f"    {name:10s}: {coef:+.4f}")
    b, a = report["before"]["pooled"], report["after"]["pooled"]
    print(f"  Before : AUC={b['auc']:.4f}  AP={b['ap']:.4f}")
    print(f"  After  : AUC={a['auc']:.4f}  AP={a['ap']:.4f}")
    print(f"  Δ AUC  : {report['delta_auc_pooled']:+.4f}")
    print(f"  Gap→ESMDisPred: {report.get('gap_to_esmdispred', 0):+.4f}")
    print(f"{'═' * 64}")


def save_meta_ensemble_report(report: dict, path: str = "meta_ensemble_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
