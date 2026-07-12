"""
Refresh downstream evaluation reports after fold_results are updated.

Used after inference fusion (Cells 10b, 11c) to avoid duplicating CAID/bio/benchmark code.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

from colab.benchmark_tables import print_matched_benchmark_report
from colab.biological_utility import (
    print_biological_utility_report,
    run_biological_utility_report,
    save_biological_utility_report,
)
from colab.caid_reporting import print_caid_report, run_full_caid_report, save_caid_report
from colab.colab_figures import optimal_threshold


def refresh_downstream_metrics(
    proteins: list,
    fold_results: list,
    n_folds: int = 5,
    threshold: Optional[float] = None,
    apply_postprocess: bool = True,
    print_reports: bool = True,
) -> dict:
    """
    Recompute CAID, biological utility, and benchmark tables from fold_results.

    Returns dict with pooled metrics and report objects.
    """
    all_probs = np.concatenate([r["val_probs"] for r in fold_results])
    all_labels = np.concatenate([r["val_labels"] for r in fold_results])

    if threshold is None:
        threshold, _ = optimal_threshold(all_labels, all_probs)

    preds_opt = (all_probs >= threshold).astype(int)
    our_auc = float(roc_auc_score(all_labels, all_probs))
    our_ap = float(average_precision_score(all_labels, all_probs))
    our_f1 = float(f1_score(all_labels.astype(int), preds_opt))
    our_mcc = float(matthews_corrcoef(all_labels.astype(int), preds_opt))

    caid_report = run_full_caid_report(
        proteins=proteins,
        fold_results=fold_results,
        threshold=threshold,
        n_folds=n_folds,
    )
    bio_report = run_biological_utility_report(
        proteins=proteins,
        fold_results=fold_results,
        threshold=threshold,
        n_folds=n_folds,
        apply_postprocess=apply_postprocess,
    )
    f1_max = caid_report["stratified"]["pooled"].get("f1_max")
    segment_f1 = bio_report.get("segment_metrics", {}).get("segment_f1")

    if print_reports:
        print_caid_report(caid_report)
        save_caid_report(caid_report, "caid_evaluation_report.json")
        print_biological_utility_report(bio_report)
        save_biological_utility_report(bio_report, "biological_utility_report.json")

    from colab.benchmark_tables import print_matched_benchmark_report
    benchmark_report = print_matched_benchmark_report(
        gpu_auc=our_auc,
        gpu_ap=our_ap,
        gpu_f1_max=f1_max,
        gpu_mcc=our_mcc,
    )

    return {
        "our_auc": our_auc,
        "our_ap": our_ap,
        "our_f1": our_f1,
        "our_mcc": our_mcc,
        "opt_threshold": float(threshold),
        "f1_max": f1_max,
        "segment_f1": segment_f1,
        "caid_report": caid_report,
        "bio_report": bio_report,
        "benchmark_report": benchmark_report,
        "all_probs": all_probs,
        "all_labels": all_labels,
    }
