"""
CAID-style evaluation reporting (Tier 1 rigor).

Residue-level metrics with stratification by disorder content, sequence length,
and organism (when present in DisProt entries).
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from colab.biological_utility import align_fold_predictions

# Stratification bins (CAID-style reporting)
DISORDER_FRAC_BINS = [
    ("low_idr", 0.0, 0.15),
    ("medium_idr", 0.15, 0.35),
    ("high_idr", 0.35, 1.01),
]

LENGTH_BINS = [
    ("short", 20, 100),
    ("medium", 100, 300),
    ("long", 300, 10_000),
]


def compute_f1_max(labels: np.ndarray, probs: np.ndarray, n_thresholds: int = 101) -> dict:
    """CAID-style F1_max: maximum F1 over uniform thresholds in [0, 1]."""
    labels = np.asarray(labels, dtype=np.int8)
    probs = np.asarray(probs, dtype=np.float32)
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return {"f1_max": None, "threshold_at_f1_max": None}

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        preds = (probs >= t).astype(np.int8)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return {"f1_max": best_f1, "threshold_at_f1_max": best_t}


def compute_caid_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Standard CAID-style residue metrics."""
    labels = np.asarray(labels, dtype=np.int8)
    probs = np.asarray(probs, dtype=np.float32)
    n = len(labels)
    if n < 5 or len(np.unique(labels)) < 2:
        return {
            "n_residues": n,
            "insufficient_data": True,
            "auc": None,
            "ap": None,
            "f1_max": None,
            "mcc": None,
            "f1_at_threshold": None,
        }

    preds = (probs >= threshold).astype(np.int8)
    f1m = compute_f1_max(labels, probs)
    preds_f1max = (probs >= f1m["threshold_at_f1_max"]).astype(np.int8) if f1m["f1_max"] is not None else preds

    return {
        "n_residues": n,
        "insufficient_data": False,
        "auc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
        "f1_max": f1m["f1_max"],
        "threshold_at_f1_max": f1m["threshold_at_f1_max"],
        "mcc_at_f1_max": float(matthews_corrcoef(labels, preds_f1max)),
        "f1_at_threshold": float(f1_score(labels, preds, zero_division=0)),
        "mcc_at_threshold": float(matthews_corrcoef(labels, preds)),
        "threshold": float(threshold),
        "disorder_fraction": float(labels.mean()),
    }


def _bin_label(value: float, bins: list[tuple[str, float, float]]) -> Optional[str]:
    for name, lo, hi in bins:
        if lo <= value < hi:
            return name
    return None


def _aggregate_stratum(
    labels_list: list[np.ndarray],
    probs_list: list[np.ndarray],
    threshold: float,
) -> dict:
    if not labels_list:
        return {"insufficient_data": True, "n_residues": 0, "n_proteins": 0}
    labels = np.concatenate(labels_list)
    probs = np.concatenate(probs_list)
    m = compute_caid_metrics(labels, probs, threshold=threshold)
    m["n_proteins"] = len(labels_list)
    return m


def run_stratified_caid_report(
    proteins: list,
    fold_results: list,
    threshold: float = 0.5,
    n_folds: int = 5,
) -> dict:
    """
    Stratified CAID metrics by disorder fraction, length, and organism.
    Uses out-of-fold predictions pooled within each stratum.
    """
    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)

    by_disorder: dict[str, tuple[list, list]] = {b[0]: ([], []) for b in DISORDER_FRAC_BINS}
    by_length: dict[str, tuple[list, list]] = {b[0]: ([], []) for b in LENGTH_BINS}
    by_organism: dict[str, tuple[list, list]] = {}

    for item in aligned:
        p = item["protein"]
        labels = item["labels"]
        probs = item["probs"]

        d_bin = _bin_label(p.get("frac_dis", 0.0), DISORDER_FRAC_BINS)
        if d_bin:
            by_disorder[d_bin][0].append(labels)
            by_disorder[d_bin][1].append(probs)

        l_bin = _bin_label(p.get("length", 0), LENGTH_BINS)
        if l_bin:
            by_length[l_bin][0].append(labels)
            by_length[l_bin][1].append(probs)

        org = (p.get("organism") or "").strip() or "unknown"
        if org not in by_organism:
            by_organism[org] = ([], [])
        by_organism[org][0].append(labels)
        by_organism[org][1].append(probs)

    # Top organisms by protein count (cap at 8 + other)
    org_counts = sorted(by_organism.items(), key=lambda x: len(x[1][0]), reverse=True)
    top_orgs = org_counts[:8]
    other_labels, other_probs = [], []
    for org, (labs, prbs) in org_counts[8:]:
        other_labels.extend(labs)
        other_probs.extend(prbs)

    organism_strata = {}
    for org, (labs, prbs) in top_orgs:
        organism_strata[org] = _aggregate_stratum(labs, prbs, threshold)
    if other_labels:
        organism_strata["other"] = _aggregate_stratum(other_labels, other_probs, threshold)

    pooled_labels = np.concatenate([item["labels"] for item in aligned])
    pooled_probs = np.concatenate([item["probs"] for item in aligned])
    pooled = compute_caid_metrics(pooled_labels, pooled_probs, threshold=threshold)

    return {
        "pooled": pooled,
        "by_disorder_fraction": {
            name: _aggregate_stratum(*by_disorder[name], threshold)
            for name in by_disorder
        },
        "by_length": {
            name: _aggregate_stratum(*by_length[name], threshold)
            for name in by_length
        },
        "by_organism": organism_strata,
        "n_proteins": len(aligned),
        "threshold": float(threshold),
        "bin_definitions": {
            "disorder_fraction": DISORDER_FRAC_BINS,
            "length": LENGTH_BINS,
        },
    }


def run_per_fold_caid_report(
    fold_results: list,
    threshold: float = 0.5,
) -> dict:
    """Per-fold CAID metrics for stability reporting."""
    folds = []
    for i, r in enumerate(fold_results):
        labels = np.asarray(r["val_labels"], dtype=np.int8)
        probs = np.asarray(r["val_probs"], dtype=np.float32)
        m = compute_caid_metrics(labels, probs, threshold=threshold)
        m["fold"] = r.get("fold", i + 1)
        folds.append(m)

    aucs = [f["auc"] for f in folds if f.get("auc") is not None]
    aps = [f["ap"] for f in folds if f.get("ap") is not None]
    f1s = [f["f1_max"] for f in folds if f.get("f1_max") is not None]

    return {
        "per_fold": folds,
        "summary": {
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "std_auc": float(np.std(aucs)) if aucs else None,
            "mean_ap": float(np.mean(aps)) if aps else None,
            "mean_f1_max": float(np.mean(f1s)) if f1s else None,
            "n_folds": len(folds),
        },
    }


def run_per_fold_threshold_report(
    fold_results: list,
    fixed_threshold: float = 0.5,
) -> dict:
    """
    Per-fold threshold metrics: optimize threshold on each fold's val set only.

    Pooled F1@0.5 is unbiased; mean per-fold F1@fold-opt summarizes fold-local
  performance without pooling threshold optimization across OOF residues.
    """
    per_fold = []
    for i, r in enumerate(fold_results):
        labels = np.asarray(r["val_labels"], dtype=np.int8)
        probs = np.asarray(r["val_probs"], dtype=np.float32)
        f1m = compute_f1_max(labels, probs)
        t_opt = f1m["threshold_at_f1_max"] if f1m["f1_max"] is not None else fixed_threshold
        m_fixed = compute_caid_metrics(labels, probs, threshold=fixed_threshold)
        m_opt = compute_caid_metrics(labels, probs, threshold=t_opt)
        per_fold.append({
            "fold": r.get("fold", i + 1),
            "f1_at_fixed": m_fixed.get("f1_at_threshold"),
            "mcc_at_fixed": m_fixed.get("mcc_at_threshold"),
            "f1_at_fold_opt": m_opt.get("f1_at_threshold"),
            "mcc_at_fold_opt": m_opt.get("mcc_at_threshold"),
            "threshold_fold_opt": t_opt,
            "auc": m_fixed.get("auc"),
            "f1_max": f1m.get("f1_max"),
        })

    def _mean(key: str) -> Optional[float]:
        vals = [f[key] for f in per_fold if f.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "fixed_threshold": float(fixed_threshold),
        "per_fold": per_fold,
        "summary": {
            "mean_f1_at_fixed": _mean("f1_at_fixed"),
            "mean_mcc_at_fixed": _mean("mcc_at_fixed"),
            "mean_f1_at_fold_opt": _mean("f1_at_fold_opt"),
            "mean_mcc_at_fold_opt": _mean("mcc_at_fold_opt"),
            "n_folds": len(per_fold),
        },
    }


def run_full_caid_report(
    proteins: list,
    fold_results: list,
    threshold: float = 0.5,
    n_folds: int = 5,
    include_segment_f1: bool = True,
    apply_postprocess: bool = True,
) -> dict:
    """Complete CAID-style report: pooled + stratified + per-fold + segment F1."""
    report = {
        "stratified": run_stratified_caid_report(proteins, fold_results, threshold, n_folds),
        "per_fold": run_per_fold_caid_report(fold_results, threshold),
        "per_fold_threshold": run_per_fold_threshold_report(fold_results, fixed_threshold=threshold),
    }
    if include_segment_f1:
        from colab.segment_postprocess import pooled_segment_f1

        aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)
        if aligned:
            proteins_ordered = [item["protein"] for item in aligned]
            all_probs = np.concatenate([item["probs"] for item in aligned])
            all_labels = np.concatenate([item["labels"] for item in aligned])
            seg_f1 = pooled_segment_f1(
                proteins_ordered, all_probs, all_labels,
                threshold=threshold,
                apply_postprocess=apply_postprocess,
            )
        else:
            seg_f1 = 0.0
        report["segment_f1_postprocessed"] = float(seg_f1)
    return report


def print_caid_report(report: dict) -> None:
    """Pretty-print CAID stratified report."""
    strat = report.get("stratified", report)
    pooled = strat.get("pooled", {})
    print(f"\n{'═' * 64}")
    print(" CAID-STYLE EVALUATION REPORT")
    print(f"{'═' * 64}")
    if pooled.get("insufficient_data"):
        print("  Insufficient data.")
        print(f"{'═' * 64}")
        return

    print(f"  Pooled ({pooled['n_residues']:,} residues)")
    print(f"    AUC     : {pooled['auc']:.4f}")
    print(f"    AP      : {pooled['ap']:.4f}")
    print(f"    F1_max  : {pooled['f1_max']:.4f}  (t={pooled['threshold_at_f1_max']:.3f})")
    print(f"    MCC@F1* : {pooled['mcc_at_f1_max']:.4f}")
    print(f"    F1@0.5  : {pooled['f1_at_threshold']:.4f}  MCC@0.5={pooled['mcc_at_threshold']:.4f}")
    seg = report.get("segment_f1_postprocessed")
    if seg is not None:
        print(f"    Seg F1* : {seg:.4f}  (post-processed regions)")

    pft = report.get("per_fold_threshold", {}).get("summary", {})
    if pft.get("mean_f1_at_fixed") is not None:
        print(
            f"\n  Per-fold thresholds (unbiased): "
            f"F1@0.5={pft['mean_f1_at_fixed']:.4f}  "
            f"mean fold-opt F1={pft.get('mean_f1_at_fold_opt', 0):.4f}"
        )

    pf = report.get("per_fold", {}).get("summary", {})
    if pf.get("mean_auc") is not None:
        print(f"\n  Per-fold AUC: {pf['mean_auc']:.4f} ± {pf['std_auc']:.4f}  ({pf['n_folds']} folds)")

    print(f"\n── By disorder fraction ──")
    for name, m in strat.get("by_disorder_fraction", {}).items():
        if m.get("insufficient_data"):
            print(f"  {name:<12} n={m.get('n_residues', 0):>6,}  (insufficient)")
        else:
            print(f"  {name:<12} n={m['n_residues']:>6,}  AUC={m['auc']:.4f}  F1_max={m['f1_max']:.4f}")

    print(f"\n── By sequence length ──")
    for name, m in strat.get("by_length", {}).items():
        if m.get("insufficient_data"):
            print(f"  {name:<12} n={m.get('n_residues', 0):>6,}  (insufficient)")
        else:
            print(f"  {name:<12} n={m['n_residues']:>6,}  AUC={m['auc']:.4f}  F1_max={m['f1_max']:.4f}")

    orgs = strat.get("by_organism", {})
    if orgs:
        print(f"\n── By organism (top strata) ──")
        for name, m in sorted(orgs.items(), key=lambda x: -x[1].get("n_proteins", 0))[:6]:
            if m.get("insufficient_data"):
                continue
            label = name[:30]
            print(f"  {label:<30} proteins={m['n_proteins']:>4}  AUC={m['auc']:.4f}")

    print(f"{'═' * 64}")


def save_caid_report(report: dict, path: str = "caid_evaluation_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
