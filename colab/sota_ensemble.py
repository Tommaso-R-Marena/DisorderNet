"""
SOTA stacking ensemble: GPU + v6 + physics-based disorder prior.

Three-way OOF blend optimized on pooled validation residues (same protocol as
GPU+v6 ensemble). Physics prior uses fast disorder-propensity features as a
complementary signal to PLM embeddings.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from colab.biological_utility import align_fold_predictions
from colab.ensemble_v6 import (
    aligned_probs_from_oof,
    apply_gpu_v6_ensemble,
    load_v6_probs_cache,
    run_v6_lite_oof,
    save_v6_probs_cache,
)
from colab.inference_fusion import compute_pooled_metrics, write_fused_probs_to_fold_results
from features_fast import compute_features_fast


def build_physics_disorder_prior(proteins: list) -> dict[str, np.ndarray]:
    """
    Per-residue physics prior in [0, 1] from fast disorder-propensity features.

    Uses column 0 of features_fast (local disorder propensity scale) min-max
    normalized per protein for stability.
    """
    priors: dict[str, np.ndarray] = {}
    for p in proteins:
        feats = compute_features_fast(p["sequence"])
        length = min(p["length"], feats.shape[0])
        col = feats[:length, 0].astype(np.float32)
        lo, hi = float(col.min()), float(col.max())
        if hi - lo < 1e-6:
            prior = np.full(length, 0.5, dtype=np.float32)
        else:
            prior = ((col - lo) / (hi - lo)).astype(np.float32)
        priors[p["id"]] = prior
    return priors


def _search_three_way_blend(
    labels: np.ndarray,
    gpu: np.ndarray,
    v6: np.ndarray,
    physics: np.ndarray,
    step: float = 0.1,
) -> dict:
    """Grid search w_gpu + w_v6 + w_phys (sum ≤ 1; remainder stays on GPU)."""
    labels = np.asarray(labels, dtype=np.float32)
    best_auc = -1.0
    best_w = (0.7, 0.2, 0.1)
    results = []
    steps = np.arange(0.0, 1.0 + step / 2, step)
    for w_v6 in steps:
        for w_phys in steps:
            if w_v6 + w_phys > 1.0:
                continue
            w_gpu = 1.0 - w_v6 - w_phys
            blended = w_gpu * gpu + w_v6 * v6 + w_phys * physics
            if len(np.unique(labels)) < 2:
                continue
            auc = float(roc_auc_score(labels, blended))
            ap = float(average_precision_score(labels, blended))
            results.append({
                "w_gpu": w_gpu, "w_v6": w_v6, "w_physics": w_phys,
                "auc": auc, "ap": ap,
            })
            if auc > best_auc:
                best_auc = auc
                best_w = (w_gpu, w_v6, w_phys)

    return {
        "best_weights": {"gpu": best_w[0], "v6": best_w[1], "physics": best_w[2]},
        "best_auc": best_auc,
        "n_grid_points": len(results),
        "top_grid": sorted(results, key=lambda x: -x["auc"])[:5],
    }


def blend_three_way_aligned(
    aligned: list[dict],
    v6_probs_by_id: dict[str, np.ndarray],
    physics_by_id: dict[str, np.ndarray],
    weights: dict[str, float],
) -> list[dict]:
    w_gpu = weights["gpu"]
    w_v6 = weights["v6"]
    w_phys = weights["physics"]
    out = []
    for item in aligned:
        gpu_p = np.asarray(item["probs"], dtype=np.float32)
        pid = item["id"]
        v6_p = np.asarray(v6_probs_by_id.get(pid, gpu_p), dtype=np.float32)
        phys_p = np.asarray(physics_by_id.get(pid, np.full_like(gpu_p, 0.5)), dtype=np.float32)
        n = len(gpu_p)
        if len(v6_p) != n:
            v6_p = gpu_p
        if len(phys_p) != n:
            phys_p = np.full(n, 0.5, dtype=np.float32)
        blended = w_gpu * gpu_p + w_v6 * v6_p + w_phys * phys_p
        new_item = dict(item)
        new_item["probs"] = blended.astype(np.float32)
        new_item["sota_stack"] = True
        out.append(new_item)
    return out


def apply_sota_stack(
    proteins: list,
    fold_results: list,
    n_folds: int = 5,
    weights: Optional[dict[str, float]] = None,
    v6_probs_by_id: Optional[dict[str, np.ndarray]] = None,
    v6_cache_path: str = "v6_oof_probs_cache.json",
    run_v6_if_missing: bool = True,
    seed: int = 42,
    use_v6_pro: bool = False,
    use_meta_ensemble: bool = False,
) -> tuple[dict, list, dict[str, np.ndarray]]:
    """
    GPU → optional v6 blend → three-way stack with physics prior.

    If fold_results already ensembled (Cell 7b), treats them as the GPU stream.
    """
    before = compute_pooled_metrics(fold_results)
    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)
    physics_by_id = build_physics_disorder_prior(proteins)

    if v6_probs_by_id is None:
        v6_probs_by_id = load_v6_probs_cache(v6_cache_path)
    if v6_probs_by_id is None and run_v6_if_missing:
        if use_v6_pro:
            from colab.v6_pro_ensemble import get_v6_pro_oof_probs
            pro_cache = v6_cache_path.replace(".json", "_pro.json")
            v6_probs_by_id = get_v6_pro_oof_probs(
                proteins, n_folds=n_folds, seed=seed, cache_path=pro_cache,
            )
        else:
            oof_probs, oof_labels, _ = run_v6_lite_oof(proteins, n_folds=n_folds, seed=seed)
            v6_probs_by_id = aligned_probs_from_oof(proteins, oof_probs)
            save_v6_probs_cache(v6_probs_by_id, v6_cache_path)

    if use_meta_ensemble and v6_probs_by_id:
        from colab.meta_ensemble import apply_meta_stacker
        gpu_by_id = {item["id"]: np.asarray(item["probs"], dtype=np.float32)
                     for item in aligned}
        streams = {"gpu": gpu_by_id, "v6": v6_probs_by_id, "physics": physics_by_id}
        meta_report, fold_results_stacked = apply_meta_stacker(
            proteins, fold_results, streams, n_folds=n_folds,
        )
        if not meta_report.get("skipped"):
            after = compute_pooled_metrics(fold_results_stacked)
            report = {
                "weights": meta_report["fit"]["coefficients"],
                "weight_search": meta_report["fit"],
                "before": meta_report["before"],
                "after": meta_report["after"],
                "delta_auc_pooled": meta_report["delta_auc_pooled"],
                "delta_ap_pooled": meta_report["delta_ap_pooled"],
                "target_sota_auc": 0.895,
                "gap_to_esmdispred": 0.895 - after["auc"],
                "method": "meta_ensemble",
            }
            return report, fold_results_stacked, v6_probs_by_id

    gpu_chunks, v6_chunks, phys_chunks, label_chunks = [], [], [], []
    for item in aligned:
        pid = item["id"]
        gpu_p = np.asarray(item["probs"], dtype=np.float32)
        if pid not in v6_probs_by_id:
            continue
        v6_p = np.asarray(v6_probs_by_id[pid], dtype=np.float32)
        phys_p = np.asarray(physics_by_id[pid], dtype=np.float32)
        if len(v6_p) != len(gpu_p) or len(phys_p) != len(gpu_p):
            continue
        gpu_chunks.append(gpu_p)
        v6_chunks.append(v6_p)
        phys_chunks.append(phys_p)
        label_chunks.append(item["labels"])

    if gpu_chunks:
        gpu_all = np.concatenate(gpu_chunks)
        v6_all = np.concatenate(v6_chunks)
        phys_all = np.concatenate(phys_chunks)
        labels_all = np.concatenate(label_chunks)
    else:
        gpu_all = before["all_probs"]
        v6_all = gpu_all.copy()
        phys_all = np.full_like(gpu_all, 0.5)
        labels_all = before["all_labels"]

    if weights is None:
        search = _search_three_way_blend(labels_all, gpu_all, v6_all, phys_all, step=0.05)
        weights = search["best_weights"]
    else:
        search = {"best_weights": weights, "provided": True}

    aligned_stacked = blend_three_way_aligned(aligned, v6_probs_by_id, physics_by_id, weights)
    fold_results_stacked = write_fused_probs_to_fold_results(
        proteins, fold_results, aligned_stacked, n_folds=n_folds,
    )
    for fr in fold_results_stacked:
        fr["sota_stack"] = True
        fr["sota_weights"] = weights

    after = compute_pooled_metrics(fold_results_stacked)
    report = {
        "weights": weights,
        "weight_search": search,
        "before": {"pooled": {k: before[k] for k in ("auc", "ap", "n_residues")}},
        "after": {"pooled": {k: after[k] for k in ("auc", "ap", "n_residues")}},
        "delta_auc_pooled": after["auc"] - before["auc"],
        "delta_ap_pooled": after["ap"] - before["ap"],
        "target_sota_auc": 0.895,
        "gap_to_esmdispred": 0.895 - after["auc"],
    }
    return report, fold_results_stacked, v6_probs_by_id


def print_sota_stack_report(report: dict) -> None:
    print(f"\n{'═' * 64}")
    print(" SOTA THREE-WAY STACK (GPU + v6 + physics prior)")
    print(f"{'═' * 64}")
    w = report["weights"]
    print(f"  Weights  : GPU={w['gpu']:.2f}  v6={w['v6']:.2f}  physics={w['physics']:.2f}")
    b, a = report["before"], report["after"]
    print(f"  Before   : AUC={b['pooled']['auc']:.4f}  AP={b['pooled']['ap']:.4f}")
    print(f"  After    : AUC={a['pooled']['auc']:.4f}  AP={a['pooled']['ap']:.4f}")
    print(f"  Δ AUC    : {report['delta_auc_pooled']:+.4f}")
    gap = report.get("gap_to_esmdispred", 0)
    print(f"  Gap→ESMDisPred (0.895): {gap:+.4f}")
    print(f"{'═' * 64}")


def save_sota_stack_report(report: dict, path: str = "sota_stack_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
