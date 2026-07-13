"""
Blend OOF predictions from multiple training seeds (same CV splits).

Run full CV with seeds e.g. 42, 43, 44 then average aligned OOF probabilities.
Splits are identical when protein list and n_folds match (seed-independent).
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from colab.biological_utility import align_fold_predictions
from colab.inference_fusion import compute_pooled_metrics, write_fused_probs_to_fold_results


def average_fold_results_multi_seed(
    proteins: list,
    fold_results_per_seed: dict[int, list],
    n_folds: int = 5,
) -> tuple[list, dict]:
    """
    Average OOF val_probs across seeds (equal weight).

    fold_results_per_seed: {seed: fold_results} from separate CV runs.
    """
    seeds = sorted(fold_results_per_seed.keys())
    if len(seeds) < 2:
        raise ValueError("Need at least 2 seeds to blend")

    before = compute_pooled_metrics(fold_results_per_seed[seeds[0]])
    aligned_per_seed = {
        s: align_fold_predictions(proteins, fold_results_per_seed[s], n_folds=n_folds)
        for s in seeds
    }

    n_items = len(aligned_per_seed[seeds[0]])
    averaged_aligned = []
    for idx in range(n_items):
        ref = aligned_per_seed[seeds[0]][idx]
        pid = ref["id"]
        prob_stack = [aligned_per_seed[s][idx]["probs"] for s in seeds]
        labels = ref["labels"]
        avg_probs = np.mean(np.stack(prob_stack, axis=0), axis=0).astype(np.float32)
        averaged_aligned.append({
            "id": pid,
            "probs": avg_probs,
            "labels": labels,
            "multi_seed": True,
            "seeds": seeds,
        })

    base_fr = fold_results_per_seed[seeds[0]]
    blended = write_fused_probs_to_fold_results(
        proteins, base_fr, averaged_aligned, n_folds=n_folds,
    )
    for fr in blended:
        fr["multi_seed_blend"] = True
        fr["blend_seeds"] = seeds

    after = compute_pooled_metrics(blended)
    report = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "before": {"pooled": {k: before[k] for k in ("auc", "ap", "n_residues")}},
        "after": {"pooled": {k: after[k] for k in ("auc", "ap", "n_residues")}},
        "delta_auc_pooled": after["auc"] - before["auc"],
        "method": "equal_weight_oof_average",
    }
    return blended, report


def print_multi_seed_report(report: dict) -> None:
    print(f"\n{'═' * 64}")
    print(" MULTI-SEED OOF BLEND")
    print(f"{'═' * 64}")
    print(f"  Seeds averaged : {report['seeds']}")
    b, a = report["before"]["pooled"], report["after"]["pooled"]
    print(f"  Before : AUC={b['auc']:.4f}  AP={b['ap']:.4f}")
    print(f"  After  : AUC={a['auc']:.4f}  AP={a['ap']:.4f}")
    print(f"  Δ AUC  : {report['delta_auc_pooled']:+.4f}")
    print(f"{'═' * 64}")


def save_multi_seed_report(report: dict, path: str = "multi_seed_blend_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
