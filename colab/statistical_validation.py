"""
Paired statistical validation (Tier 1 rigor).

Per-fold paired comparisons of DisorderNet vs AF pLDDT baseline on the same
validation residues, with sign tests and bootstrap CIs on fold-level AUC deltas.
"""

from __future__ import annotations

import json
from math import comb
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score
from colab.cv_splits import get_cv_splits

from colab.af_plddt import plddt_to_disorder_score


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float32)
    if len(labels) < 5 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Bootstrap CI for the mean of fold-level statistics (small-n safe)."""
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 2:
        return {
            "point": float(values.mean()) if len(values) else None,
            "ci_low": None,
            "ci_high": None,
            "n_boot": 0,
            "insufficient_data": True,
        }
    point = float(np.mean(values))
    rng = np.random.default_rng(seed)
    boots = [
        float(np.mean(rng.choice(values, size=len(values), replace=True)))
        for _ in range(n_boot)
    ]
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "insufficient_data": False,
        "ci_level": ci,
    }


def sign_test_two_sided(n_positive: int, n_negative: int) -> dict:
    """
    Exact two-sided sign test on paired differences.
    n_positive: folds where method A > method B
    n_negative: folds where method A < method B
    """
    n = n_positive + n_negative
    if n == 0:
        return {"p_value": 1.0, "n_discordant": 0, "insufficient_data": True}

    k = min(n_positive, n_negative)
    p_one_sided = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p_two_sided = min(1.0, 2.0 * p_one_sided)
    return {
        "p_value": float(p_two_sided),
        "n_discordant": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "insufficient_data": False,
        "favors": "a" if n_positive > n_negative else ("b" if n_negative > n_positive else "tie"),
    }


def wilcoxon_signed_rank_test(deltas: list[float]) -> dict:
    """
    Wilcoxon signed-rank test on paired per-fold AUC deltas (DisorderNet − baseline).

    More powerful than the sign test when fold deltas vary in magnitude.
    """
    deltas = [d for d in deltas if d != 0]
    n = len(deltas)
    if n < 2:
        return {"p_value": None, "n_nonzero": n, "insufficient_data": True}
    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(deltas, alternative="greater", zero_method="wilcox")
        return {
            "p_value": float(p),
            "statistic": float(stat),
            "n_nonzero": n,
            "insufficient_data": False,
            "favors": "disordernet" if sum(deltas) > 0 else "baseline",
        }
    except ImportError:
        return {
            "p_value": None,
            "n_nonzero": n,
            "insufficient_data": True,
            "note": "scipy not available",
        }


def run_per_fold_paired_comparison(
    proteins: list,
    fold_results: list,
    plddt_by_protein: dict[str, np.ndarray],
    n_folds: int = 5,
    *,
    split_method: str = "protein",
    homology_min_identity: float = 0.4,
) -> dict:
    """
    Per-fold AUC: DisorderNet vs inverse-pLDDT on AF-covered validation residues.
    """
    splits = get_cv_splits(
        proteins,
        n_folds,
        split_method=split_method,
        homology_min_identity=homology_min_identity,
    )
    fold_rows = []

    for fold_idx, (_, val_idx) in enumerate(splits):
        if fold_idx >= len(fold_results):
            break

        val_proteins = [proteins[i] for i in val_idx]
        fold_result = fold_results[fold_idx]

        offset = 0
        labels_chunks: list[np.ndarray] = []
        dn_chunks: list[np.ndarray] = []
        base_chunks: list[np.ndarray] = []
        n_with_af = 0

        for p in val_proteins:
            pid = p["id"]
            length = p["length"]
            if pid not in plddt_by_protein:
                offset += length
                continue
            plddt = plddt_by_protein[pid]
            if len(plddt) != length:
                offset += length
                continue

            chunk_labels = np.asarray(fold_result["val_labels"][offset:offset + length], dtype=np.int8)
            chunk_probs = np.asarray(fold_result["val_probs"][offset:offset + length], dtype=np.float32)
            valid = ~np.isnan(plddt)
            if valid.sum() < 5:
                offset += length
                continue

            labels_chunks.append(chunk_labels[valid])
            dn_chunks.append(chunk_probs[valid])
            base_chunks.append(plddt_to_disorder_score(plddt[valid]))
            n_with_af += 1
            offset += length

        if not labels_chunks:
            fold_rows.append({
                "fold": fold_idx + 1,
                "insufficient_data": True,
                "n_proteins_with_af": 0,
            })
            continue

        labels_cat = np.concatenate(labels_chunks)
        dn_cat = np.concatenate(dn_chunks)
        base_cat = np.concatenate(base_chunks)

        auc_dn = _safe_auc(labels_cat, dn_cat)
        auc_base = _safe_auc(labels_cat, base_cat)
        delta = (auc_dn - auc_base) if auc_dn is not None and auc_base is not None else None

        fold_rows.append({
            "fold": fold_idx + 1,
            "insufficient_data": False,
            "n_proteins_with_af": n_with_af,
            "n_residues": int(len(labels_cat)),
            "auc_disordernet": auc_dn,
            "auc_plddt_baseline": auc_base,
            "delta_auc": delta,
        })

    valid_rows = [r for r in fold_rows if not r.get("insufficient_data") and r.get("delta_auc") is not None]
    deltas = [r["delta_auc"] for r in valid_rows]
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    n_ties = sum(1 for d in deltas if d == 0)
    sign = sign_test_two_sided(n_pos, n_neg)
    wilcoxon = wilcoxon_signed_rank_test(deltas)

    bootstrap_fold = None
    if len(deltas) >= 2:
        bootstrap_fold = _bootstrap_mean_ci(np.asarray(deltas, dtype=np.float32), n_boot=2000, seed=42)

    return {
        "per_fold": fold_rows,
        "summary": {
            "n_folds_valid": len(valid_rows),
            "mean_delta_auc": float(np.mean(deltas)) if deltas else None,
            "std_delta_auc": float(np.std(deltas)) if deltas else None,
            "median_delta_auc": float(np.median(deltas)) if deltas else None,
            "all_folds_positive": n_pos == len(deltas) if deltas else False,
            "n_ties": n_ties,
        },
        "sign_test_disordernet_vs_plddt": sign,
        "wilcoxon_disordernet_vs_plddt": wilcoxon,
        "bootstrap_mean_delta_auc": bootstrap_fold,
        "split_method": split_method,
        "homology_min_identity": homology_min_identity if split_method == "homology" else None,
        "comparison": "DisorderNet vs inverse-pLDDT (AF-covered val residues per fold)",
    }


def run_cv_fold_stability_report(fold_results: list) -> dict:
    """Sign test / summary on per-fold best AUC vs cross-fold mean (stability)."""
    aucs = [r["best_auc"] for r in fold_results]
    if not aucs:
        return {"insufficient_data": True}

    mean_auc = float(np.mean(aucs))
    above = sum(1 for a in aucs if a > mean_auc)
    below = sum(1 for a in aucs if a < mean_auc)

    return {
        "fold_aucs": [float(a) for a in aucs],
        "mean_auc": mean_auc,
        "std_auc": float(np.std(aucs)),
        "min_auc": float(np.min(aucs)),
        "max_auc": float(np.max(aucs)),
        "bootstrap_fold_auc": _bootstrap_mean_ci(np.asarray(aucs, dtype=np.float32), n_boot=2000, seed=42),
        "sign_test_vs_mean": sign_test_two_sided(above, below),
    }


def run_full_statistical_validation(
    proteins: list,
    fold_results: list,
    plddt_by_protein: Optional[dict[str, np.ndarray]] = None,
    n_folds: int = 5,
    *,
    split_method: str = "protein",
    homology_min_identity: float = 0.4,
) -> dict:
    """Combined statistical validation report."""
    report = {
        "cv_fold_stability": run_cv_fold_stability_report(fold_results),
        "split_method": split_method,
    }
    if plddt_by_protein:
        report["paired_af_baseline"] = run_per_fold_paired_comparison(
            proteins, fold_results, plddt_by_protein, n_folds=n_folds,
            split_method=split_method,
            homology_min_identity=homology_min_identity,
        )
    else:
        report["paired_af_baseline"] = {
            "insufficient_data": True,
            "note": "No pLDDT dict provided; run Phase 2 first.",
        }
    return report


def print_statistical_validation(report: dict) -> None:
    print(f"\n{'═' * 64}")
    print(" STATISTICAL VALIDATION (per-fold paired)")
    print(f"{'═' * 64}")

    stab = report.get("cv_fold_stability", {})
    if not stab.get("insufficient_data"):
        print(f"\n── CV fold stability ──")
        print(f"  Fold AUCs : {[f'{a:.4f}' for a in stab['fold_aucs']]}")
        print(f"  Mean ± std: {stab['mean_auc']:.4f} ± {stab['std_auc']:.4f}")
        boot = stab.get("bootstrap_fold_auc", {})
        if boot.get("ci_low") is not None:
            print(f"  95% CI    : [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")

    paired = report.get("paired_af_baseline", {})
    if paired.get("insufficient_data"):
        print(f"\n  AF baseline pairing: {paired.get('note', 'N/A')}")
    else:
        s = paired.get("summary", {})
        print(f"\n── DisorderNet vs inverse-pLDDT (per fold, AF-covered) ──")
        print(f"  Mean Δ AUC: {s.get('mean_delta_auc', 0):+.4f} ± {s.get('std_delta_auc', 0):.4f}")
        sign = paired.get("sign_test_disordernet_vs_plddt", {})
        if not sign.get("insufficient_data"):
            print(f"  Sign test : p={sign['p_value']:.4f}  "
                  f"({sign['n_positive']} folds DN wins, {sign['n_negative']} baseline wins, "
                  f"{s.get('n_ties', 0)} ties)")
        wilcox = paired.get("wilcoxon_disordernet_vs_plddt", {})
        if wilcox and not wilcox.get("insufficient_data") and wilcox.get("p_value") is not None:
            print(f"  Wilcoxon  : p={wilcox['p_value']:.4f}  (n={wilcox['n_nonzero']} non-zero deltas)")
        boot = paired.get("bootstrap_mean_delta_auc", {})
        if boot and boot.get("ci_low") is not None:
            print(f"  Δ AUC 95% CI: [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")

    print(f"{'═' * 64}")


def save_statistical_validation(report: dict, path: str = "statistical_validation_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
