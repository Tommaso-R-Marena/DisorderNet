"""
Inference-time AF pLDDT fusion for DisorderNet CV predictions.

After Phase 2 pLDDT fetch, blends out-of-fold DisorderNet probabilities with
inverse-pLDDT on AF-covered residues. Your prior run showed fusion α=0.50
reached ~0.831 AUC on AF-covered residues vs 0.816 DisorderNet-only.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from colab.cv_splits import get_cv_splits

from colab.biological_utility import align_fold_predictions
from colab.phase3_synthesis import find_optimal_fusion_alpha, fuse_disorder_score


def build_combined_plddt_map(
    plddt_af2: dict[str, np.ndarray],
    plddt_af3: Optional[dict[str, np.ndarray]] = None,
    prefer: str = "af3",
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Merge AF2 and AF3 pLDDT maps.

    prefer='af3': AF3 where available, else AF2.
    prefer='af2': AF2 where available, else AF3.
    """
    af2 = plddt_af2 or {}
    af3 = plddt_af3 or {}
    combined: dict[str, np.ndarray] = {}
    stats = {"from_af2_only": 0, "from_af3_only": 0, "from_af3_preferred": 0, "from_af2_preferred": 0}

    all_ids = set(af2) | set(af3)
    for pid in all_ids:
        has2 = pid in af2
        has3 = pid in af3
        if prefer == "af3":
            if has3:
                combined[pid] = np.asarray(af3[pid], dtype=np.float32)
                stats["from_af3_preferred" if has2 else "from_af3_only"] += 1
            elif has2:
                combined[pid] = np.asarray(af2[pid], dtype=np.float32)
                stats["from_af2_only"] += 1
        else:
            if has2:
                combined[pid] = np.asarray(af2[pid], dtype=np.float32)
                stats["from_af2_preferred" if has3 else "from_af2_only"] += 1
            elif has3:
                combined[pid] = np.asarray(af3[pid], dtype=np.float32)
                stats["from_af3_only"] += 1

    stats["n_proteins"] = len(combined)
    stats["prefer"] = prefer
    return combined, stats


def apply_combined_plddt_fusion_to_cv(
    proteins: list,
    fold_results: list,
    plddt_af2: dict[str, np.ndarray],
    plddt_af3: Optional[dict[str, np.ndarray]] = None,
    prefer: str = "af3",
    n_folds: int = 5,
    alpha: Optional[float] = None,
) -> tuple[dict, list]:
    """
    Fuse CV predictions using combined AF2/AF3 pLDDT (AF3 preferred by default).
    """
    combined, coverage_stats = build_combined_plddt_map(plddt_af2, plddt_af3, prefer=prefer)
    report, fused_folds = apply_plddt_fusion_to_cv(
        proteins=proteins,
        fold_results=fold_results,
        plddt_by_protein=combined,
        n_folds=n_folds,
        alpha=alpha,
    )
    report["combined_plddt"] = coverage_stats
    report["prefer"] = prefer
    return report, fused_folds


def fuse_aligned_predictions(
    aligned: list[dict],
    plddt_by_protein: dict[str, np.ndarray],
    alpha: float,
) -> tuple[list[dict], dict]:
    """
    Fuse DisorderNet probs with inverse-pLDDT where AF coverage exists.

    Returns (updated aligned list, coverage stats).
    """
    n_fused_residues = 0
    n_total_residues = 0
    n_proteins_fused = 0

    for item in aligned:
        pid = item["id"]
        probs = np.asarray(item["probs"], dtype=np.float32).copy()
        n_total_residues += len(probs)

        if pid not in plddt_by_protein:
            item["probs"] = probs
            continue

        plddt = np.asarray(plddt_by_protein[pid], dtype=np.float32)
        if len(plddt) != len(probs):
            item["probs"] = probs
            continue

        valid = ~np.isnan(plddt)
        if valid.sum() == 0:
            item["probs"] = probs
            continue

        fused = fuse_disorder_score(probs, plddt, alpha=alpha)
        # Keep DisorderNet where pLDDT missing (shouldn't happen if same length)
        probs[valid] = fused[valid]
        item["probs"] = probs
        n_fused_residues += int(valid.sum())
        n_proteins_fused += 1

    return aligned, {
        "n_proteins_fused": n_proteins_fused,
        "n_residues_fused": n_fused_residues,
        "n_residues_total": n_total_residues,
        "coverage_fraction": n_fused_residues / max(n_total_residues, 1),
    }


def write_fused_probs_to_fold_results(
    proteins: list,
    fold_results: list,
    aligned: list[dict],
    n_folds: int = 5,
) -> list:
    """Write per-protein fused probs back into fold_results val_probs arrays."""
    by_id = {item["id"]: item["probs"] for item in aligned}
    updated = []

    splits = get_cv_splits(proteins, n_folds)

    for fold_idx, (_, val_idx) in enumerate(splits):
        if fold_idx >= len(fold_results):
            break
        fold_copy = dict(fold_results[fold_idx])
        val_proteins = [proteins[i] for i in val_idx]
        chunks = []
        for p in val_proteins:
            if p["id"] in by_id:
                chunks.append(np.asarray(by_id[p["id"]], dtype=np.float32))
            else:
                # fallback: slice original val_probs
                raise KeyError(f"Missing aligned predictions for {p['id']}")
        fold_copy["val_probs"] = np.concatenate(chunks).astype(np.float32)
        fold_copy["fused"] = True
        updated.append(fold_copy)

    return updated


def compute_pooled_metrics(fold_results: list) -> dict:
    """Pooled AUC/AP from fold val_probs."""
    all_probs = np.concatenate([r["val_probs"] for r in fold_results])
    all_labels = np.concatenate([r["val_labels"] for r in fold_results])
    return {
        "auc": float(roc_auc_score(all_labels, all_probs)),
        "ap": float(average_precision_score(all_labels, all_probs)),
        "n_residues": int(len(all_labels)),
        "all_probs": all_probs,
        "all_labels": all_labels,
    }


def compute_af_subset_metrics(
    proteins: list,
    fold_results: list,
    plddt_by_protein: dict[str, np.ndarray],
    n_folds: int = 5,
) -> dict:
    """AUC/AP on residues with valid AF pLDDT only."""
    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)
    probs_list, labels_list = [], []

    for item in aligned:
        pid = item["id"]
        if pid not in plddt_by_protein:
            continue
        plddt = np.asarray(plddt_by_protein[pid], dtype=np.float32)
        if len(plddt) != len(item["probs"]):
            continue
        valid = ~np.isnan(plddt)
        if valid.sum() < 5:
            continue
        probs_list.append(item["probs"][valid])
        labels_list.append(item["labels"][valid])

    if not probs_list:
        return {"auc": None, "ap": None, "n_residues": 0}

    probs = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
        "n_residues": int(len(labels)),
    }


def apply_plddt_fusion_to_cv(
    proteins: list,
    fold_results: list,
    plddt_by_protein: dict[str, np.ndarray],
    n_folds: int = 5,
    alpha: Optional[float] = None,
) -> tuple[dict, list]:
    """
    Optimize fusion α (if needed), fuse CV predictions, return report + updated folds.
    """
    before_all = compute_pooled_metrics(fold_results)
    before_af = compute_af_subset_metrics(proteins, fold_results, plddt_by_protein, n_folds)

    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)

    # Optimize α on AF-covered OOF residues
    if alpha is None:
        af_probs, af_labels, af_plddt = [], [], []
        for item in aligned:
            pid = item["id"]
            if pid not in plddt_by_protein:
                continue
            plddt = np.asarray(plddt_by_protein[pid], dtype=np.float32)
            if len(plddt) != len(item["probs"]):
                continue
            valid = ~np.isnan(plddt)
            if valid.sum() < 10:
                continue
            af_probs.append(item["probs"][valid])
            af_labels.append(item["labels"][valid])
            af_plddt.append(plddt[valid])

        if af_probs:
            search = find_optimal_fusion_alpha(
                np.concatenate(af_labels),
                np.concatenate(af_probs),
                np.concatenate(af_plddt),
            )
            alpha = search["best_alpha"]
            alpha_search = search
        else:
            alpha = 0.5
            alpha_search = {"best_alpha": 0.5, "insufficient_data": True}
    else:
        alpha_search = {"best_alpha": float(alpha), "provided": True}

    aligned_fused, coverage = fuse_aligned_predictions(aligned, plddt_by_protein, alpha)
    fold_results_fused = write_fused_probs_to_fold_results(
        proteins, fold_results, aligned_fused, n_folds=n_folds,
    )

    after_all = compute_pooled_metrics(fold_results_fused)
    after_af = compute_af_subset_metrics(proteins, fold_results_fused, plddt_by_protein, n_folds)

    report = {
        "fusion_alpha": float(alpha),
        "alpha_search": alpha_search,
        "coverage": coverage,
        "before": {
            "pooled": {k: before_all[k] for k in ("auc", "ap", "n_residues")},
            "af_subset": before_af,
        },
        "after": {
            "pooled": {k: after_all[k] for k in ("auc", "ap", "n_residues")},
            "af_subset": after_af,
        },
        "delta_auc_pooled": after_all["auc"] - before_all["auc"],
        "delta_auc_af_subset": (
            (after_af["auc"] - before_af["auc"])
            if after_af.get("auc") is not None and before_af.get("auc") is not None
            else None
        ),
        "delta_ap_pooled": after_all["ap"] - before_all["ap"],
    }
    return report, fold_results_fused


def print_fusion_report(report: dict) -> None:
    print(f"\n{'═' * 64}")
    print(" AF–DISORDERNET INFERENCE FUSION")
    print(f"{'═' * 64}")
    print(f"  Fusion α         : {report['fusion_alpha']:.2f}")
    cov = report["coverage"]
    print(f"  AF residues fused: {cov['n_residues_fused']:,} / {cov['n_residues_total']:,} "
          f"({100 * cov['coverage_fraction']:.1f}%)")
    print(f"  Proteins fused   : {cov['n_proteins_fused']:,}")

    b, a = report["before"], report["after"]
    print(f"\n── Pooled CV (all residues) ──")
    print(f"  Before : AUC={b['pooled']['auc']:.4f}  AP={b['pooled']['ap']:.4f}")
    print(f"  After  : AUC={a['pooled']['auc']:.4f}  AP={a['pooled']['ap']:.4f}")
    print(f"  Δ AUC  : {report['delta_auc_pooled']:+.4f}  "
          f"Δ AP: {report['delta_ap_pooled']:+.4f}")

    if a["af_subset"].get("auc") is not None:
        print(f"\n── AF-covered subset ──")
        print(f"  Before : AUC={b['af_subset']['auc']:.4f}")
        print(f"  After  : AUC={a['af_subset']['auc']:.4f}")
        if report["delta_auc_af_subset"] is not None:
            print(f"  Δ AUC  : {report['delta_auc_af_subset']:+.4f}")
    print(f"{'═' * 64}")


def save_fusion_report(report: dict, path: str = "inference_fusion_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
