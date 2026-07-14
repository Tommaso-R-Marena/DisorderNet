"""
Phase 2: AlphaFold pLDDT hallucination rescue analysis.

Quantifies where structure predictors assign high confidence to genuinely
disordered regions, and whether DisorderNet correctly flags those cases.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from colab.af_plddt import (
    DEFAULT_CACHE_DIR,
    fetch_plddt_batch,
    plddt_to_disorder_score,
)
from colab.biological_utility import align_fold_predictions


def compute_hallucination_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    plddt: np.ndarray,
    threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
) -> dict:
    """
    Compute hallucination and rescue rates on residues with valid pLDDT.

    Hallucination: DisProt-disordered residue with pLDDT >= high_plddt_threshold.
    Rescue: hallucinated residue where DisorderNet predicts disorder.
    """
    labels = np.asarray(labels, dtype=np.int8)
    probs = np.asarray(probs, dtype=np.float32)
    plddt = np.asarray(plddt, dtype=np.float32)

    valid = ~np.isnan(plddt)
    if valid.sum() == 0:
        return {"n_residues": 0, "insufficient_data": True}

    lab = labels[valid]
    prb = probs[valid]
    pld = plddt[valid]
    preds = (prb >= threshold).astype(np.int8)

    disordered = lab == 1
    ordered = lab == 0
    n_dis = int(disordered.sum())
    n_ord = int(ordered.sum())

    hallucinated = disordered & (pld >= high_plddt_threshold)
    n_halluc = int(hallucinated.sum())
    rescued = hallucinated & (preds == 1)
    n_rescued = int(rescued.sum())

    # Ordered regions wrongly given high pLDDT (overconfidence in order)
    false_confidence_ordered = ordered & (pld >= high_plddt_threshold)

    return {
        "n_residues": int(valid.sum()),
        "n_disordered": n_dis,
        "n_ordered": n_ord,
        "insufficient_data": False,
        "high_plddt_threshold": float(high_plddt_threshold),
        "disorder_threshold": float(threshold),
        "hallucination_rate": float(n_halluc / n_dis) if n_dis > 0 else 0.0,
        "n_hallucinated": n_halluc,
        "rescue_rate": float(n_rescued / n_halluc) if n_halluc > 0 else 0.0,
        "n_rescued": n_rescued,
        "rescue_of_disordered": float(n_rescued / n_dis) if n_dis > 0 else 0.0,
        "mean_plddt_disordered": float(pld[disordered].mean()) if n_dis > 0 else 0.0,
        "mean_plddt_ordered": float(pld[ordered].mean()) if n_ord > 0 else 0.0,
        "false_confidence_ordered_rate": float(false_confidence_ordered.sum() / n_ord) if n_ord > 0 else 0.0,
    }


def compute_plddt_baseline_auc(
    labels: np.ndarray,
    plddt: np.ndarray,
) -> dict:
    """AUC/AP for inverse-pLDDT disorder score vs DisProt labels."""
    labels = np.asarray(labels, dtype=np.int8)
    plddt = np.asarray(plddt, dtype=np.float32)
    valid = ~np.isnan(plddt)
    if valid.sum() < 5 or len(np.unique(labels[valid])) < 2:
        return {"auc": None, "ap": None, "n_residues": int(valid.sum())}

    scores = plddt_to_disorder_score(plddt[valid])
    lab = labels[valid]
    return {
        "auc": float(roc_auc_score(lab, scores)),
        "ap": float(average_precision_score(lab, scores)),
        "n_residues": int(valid.sum()),
    }


def run_af_rescue_report(
    proteins: list,
    fold_results: list,
    plddt_by_protein: dict[str, np.ndarray],
    threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    n_folds: int = 5,
    source: str = "AlphaFold DB (AF2 models)",
) -> dict:
    """
    Pooled Phase 2 report: hallucination rescue + pLDDT baseline comparison.
    """
    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)

    all_labels: list = []
    all_probs: list = []
    all_plddt: list = []
    per_protein: list = []
    proteins_with_plddt = 0
    proteins_missing = 0

    for item in aligned:
        pid = item["id"]
        if pid not in plddt_by_protein:
            proteins_missing += 1
            continue
        plddt = plddt_by_protein[pid]
        if len(plddt) != item["protein"]["length"]:
            proteins_missing += 1
            continue

        labels = item["labels"]
        probs = item["probs"]
        metrics = compute_hallucination_metrics(
            labels, probs, plddt, threshold, high_plddt_threshold,
        )
        if metrics.get("insufficient_data"):
            proteins_missing += 1
            continue

        proteins_with_plddt += 1
        per_protein.append({"id": pid, **metrics})
        valid = ~np.isnan(plddt)
        all_labels.append(labels[valid])
        all_probs.append(probs[valid])
        all_plddt.append(plddt[valid])

    if not all_labels:
        return {
            "insufficient_data": True,
            "proteins_with_plddt": 0,
            "proteins_missing_plddt": proteins_missing,
        }

    labels_cat = np.concatenate(all_labels)
    probs_cat = np.concatenate(all_probs)
    plddt_cat = np.concatenate(all_plddt)

    pooled_halluc = compute_hallucination_metrics(
        labels_cat, probs_cat, plddt_cat, threshold, high_plddt_threshold,
    )
    plddt_baseline = compute_plddt_baseline_auc(labels_cat, plddt_cat)

    # DisorderNet AUC on same residue subset
    if len(np.unique(labels_cat)) >= 2:
        disordernet_auc = float(roc_auc_score(labels_cat, probs_cat))
        disordernet_ap = float(average_precision_score(labels_cat, probs_cat))
    else:
        disordernet_auc = disordernet_ap = None

    delta_auc = None
    if plddt_baseline["auc"] is not None and disordernet_auc is not None:
        delta_auc = disordernet_auc - plddt_baseline["auc"]

    return {
        "insufficient_data": False,
        "proteins_with_plddt": proteins_with_plddt,
        "proteins_missing_plddt": proteins_missing,
        "coverage_fraction": proteins_with_plddt / max(len(aligned), 1),
        "threshold": float(threshold),
        "high_plddt_threshold": float(high_plddt_threshold),
        "source": source,
        "pooled": pooled_halluc,
        "plddt_baseline": plddt_baseline,
        "disordernet_on_af_subset": {
            "auc": disordernet_auc,
            "ap": disordernet_ap,
            "n_residues": int(len(labels_cat)),
        },
        "delta_auc_vs_plddt_baseline": delta_auc,
        "per_protein_summary": {
            "mean_hallucination_rate": float(np.mean([p["hallucination_rate"] for p in per_protein])),
            "mean_rescue_rate": float(np.mean([p["rescue_rate"] for p in per_protein if p["n_hallucinated"] > 0] or [0.0])),
            "n_proteins": len(per_protein),
        },
    }


def run_af2_af3_comparison_report(
    af2_report: dict,
    af3_report: dict,
) -> dict:
    """Summarize AF2 vs AF3 hallucination rescue on overlapping protein coverage."""
    if af2_report.get("insufficient_data") or af3_report.get("insufficient_data"):
        return {
            "insufficient_data": True,
            "af2_proteins": af2_report.get("proteins_with_plddt", 0),
            "af3_proteins": af3_report.get("proteins_with_plddt", 0),
        }

    a2 = af2_report["pooled"]
    a3 = af3_report["pooled"]
    b2 = af2_report["plddt_baseline"].get("auc")
    b3 = af3_report["plddt_baseline"].get("auc")
    d2 = af2_report["disordernet_on_af_subset"].get("auc")
    d3 = af3_report["disordernet_on_af_subset"].get("auc")

    return {
        "insufficient_data": False,
        "af2": {
            "source": af2_report.get("source"),
            "proteins_with_plddt": af2_report["proteins_with_plddt"],
            "hallucination_rate": a2["hallucination_rate"],
            "rescue_rate": a2["rescue_rate"],
            "plddt_baseline_auc": b2,
            "disordernet_auc": d2,
            "delta_auc": af2_report.get("delta_auc_vs_plddt_baseline"),
        },
        "af3": {
            "source": af3_report.get("source"),
            "proteins_with_plddt": af3_report["proteins_with_plddt"],
            "hallucination_rate": a3["hallucination_rate"],
            "rescue_rate": a3["rescue_rate"],
            "plddt_baseline_auc": b3,
            "disordernet_auc": d3,
            "delta_auc": af3_report.get("delta_auc_vs_plddt_baseline"),
        },
        "delta_hallucination_af3_minus_af2": a3["hallucination_rate"] - a2["hallucination_rate"],
        "delta_rescue_af3_minus_af2": a3["rescue_rate"] - a2["rescue_rate"],
    }


def fetch_and_run_boltz_rescue_report(
    proteins: list,
    fold_results: list,
    boltz_output_root: str,
    threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    n_folds: int = 5,
    cache_dir: str = "boltz_plddt_cache",
) -> tuple[dict, dict[str, np.ndarray]]:
    """Load Boltz-2 pLDDT then run hallucination rescue report."""
    from colab.boltz_plddt import DEFAULT_BOLTZ_CACHE_DIR, load_boltz_plddt_batch

    cache_dir = cache_dir or DEFAULT_BOLTZ_CACHE_DIR
    plddt_by_protein = load_boltz_plddt_batch(
        proteins, output_root=boltz_output_root, cache_dir=cache_dir, verbose=True,
    )
    report = run_af_rescue_report(
        proteins=proteins,
        fold_results=fold_results,
        plddt_by_protein=plddt_by_protein,
        threshold=threshold,
        high_plddt_threshold=high_plddt_threshold,
        n_folds=n_folds,
        source="Boltz-2 (pinned auto-download)",
    )
    report["n_plddt_loaded"] = len(plddt_by_protein)
    report["boltz_output_root"] = boltz_output_root
    report["cache_dir"] = cache_dir
    return report, plddt_by_protein


def fetch_and_run_af3_rescue_report(
    proteins: list,
    fold_results: list,
    af3_output_root: str,
    threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    n_folds: int = 5,
    cache_dir: str = "af3_plddt_cache",
) -> tuple[dict, dict[str, np.ndarray]]:
    """Load AF3 pLDDT from Drive outputs then run Phase 2b report."""
    from colab.af3_plddt import DEFAULT_AF3_CACHE_DIR, load_af3_plddt_batch

    cache_dir = cache_dir or DEFAULT_AF3_CACHE_DIR
    plddt_by_protein = load_af3_plddt_batch(
        proteins, output_root=af3_output_root, cache_dir=cache_dir, verbose=True,
    )
    report = run_af_rescue_report(
        proteins=proteins,
        fold_results=fold_results,
        plddt_by_protein=plddt_by_protein,
        threshold=threshold,
        high_plddt_threshold=high_plddt_threshold,
        n_folds=n_folds,
        source="AlphaFold 3 (local/Drive outputs)",
    )
    report["n_plddt_loaded"] = len(plddt_by_protein)
    report["af3_output_root"] = af3_output_root
    report["cache_dir"] = cache_dir
    return report, plddt_by_protein


def fetch_and_run_af_rescue_report(
    proteins: list,
    fold_results: list,
    threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    n_folds: int = 5,
    cache_dir: str = DEFAULT_CACHE_DIR,
    sleep_s: float = 0.1,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Fetch AlphaFold pLDDT then run Phase 2 report."""
    plddt_by_protein = fetch_plddt_batch(
        proteins, cache_dir=cache_dir, sleep_s=sleep_s, verbose=True,
    )
    report = run_af_rescue_report(
        proteins=proteins,
        fold_results=fold_results,
        plddt_by_protein=plddt_by_protein,
        threshold=threshold,
        high_plddt_threshold=high_plddt_threshold,
        n_folds=n_folds,
        source="AlphaFold DB (AF2 models)",
    )
    report["n_plddt_fetched"] = len(plddt_by_protein)
    report["cache_dir"] = cache_dir
    return report, plddt_by_protein


def print_af_rescue_report(report: dict) -> None:
    """Pretty-print Phase 2 report."""
    print(f"\n{'═' * 64}")
    print(" ALPHA FOLD pLDDT HALLUCINATION RESCUE (Phase 2)")
    print(f"{'═' * 64}")

    if report.get("insufficient_data"):
        print("  Insufficient AlphaFold pLDDT coverage for analysis.")
        print(f"  Proteins with pLDDT: {report.get('proteins_with_plddt', 0)}")
        print(f"  Missing: {report.get('proteins_missing_plddt', 0)}")
        print(f"{'═' * 64}")
        return

    print(f"  Source           : {report.get('source', 'AlphaFold DB')}")
    print(f"  Proteins w/ pLDDT: {report['proteins_with_plddt']} "
          f"({100 * report['coverage_fraction']:.1f}% of CV set)")
    print(f"  pLDDT fetched    : {report.get('n_plddt_fetched', 'N/A')}")
    print(f"  High pLDDT cutoff: {report['high_plddt_threshold']:.0f}")

    p = report["pooled"]
    print(f"\n── Pooled hallucination metrics ({p['n_residues']:,} residues) ──")
    print(f"  Disordered residues     : {p['n_disordered']:,}")
    print(f"  Hallucination rate      : {p['hallucination_rate']:.3f}  "
          f"(high pLDDT | truly disordered)")
    print(f"  Hallucinated residues   : {p['n_hallucinated']:,}")
    print(f"  Rescue rate             : {p['rescue_rate']:.3f}  "
          f"(DisorderNet flags hallucinations)")
    print(f"  Rescued residues        : {p['n_rescued']:,}")
    print(f"  Rescue / all disordered : {p['rescue_of_disordered']:.3f}")
    print(f"  Mean pLDDT (disordered) : {p['mean_plddt_disordered']:.1f}")
    print(f"  Mean pLDDT (ordered)    : {p['mean_plddt_ordered']:.1f}")

    base = report["plddt_baseline"]
    dn = report["disordernet_on_af_subset"]
    print(f"\n── AUC on AlphaFold-covered residues ──")
    if base["auc"] is not None:
        print(f"  AF2 pLDDT baseline AUC  : {base['auc']:.4f}")
    if dn["auc"] is not None:
        print(f"  DisorderNet AUC         : {dn['auc']:.4f}")
    if report.get("delta_auc_vs_plddt_baseline") is not None:
        print(f"  Δ AUC (DisorderNet)     : {report['delta_auc_vs_plddt_baseline']:+.4f}")

    ps = report.get("per_protein_summary", {})
    if ps:
        print(f"\n── Per-protein means ({ps['n_proteins']} proteins) ──")
        print(f"  Mean hallucination rate : {ps['mean_hallucination_rate']:.3f}")
        print(f"  Mean rescue rate        : {ps['mean_rescue_rate']:.3f}")

    print(f"{'═' * 64}")


def save_af_rescue_report(report: dict, path: str = "af_rescue_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def print_af2_af3_comparison(comparison: dict) -> None:
    """Pretty-print AF2 vs AF3 side-by-side."""
    print(f"\n{'═' * 64}")
    print(" AF2 vs AF3 HALLUCINATION RESCUE (Phase 2b)")
    print(f"{'═' * 64}")
    if comparison.get("insufficient_data"):
        print("  Insufficient data for AF2/AF3 comparison.")
        print(f"  AF2 proteins: {comparison.get('af2_proteins', 0)}")
        print(f"  AF3 proteins: {comparison.get('af3_proteins', 0)}")
        print(f"{'═' * 64}")
        return

    for key, label in [("af2", "AF2 (AlphaFold DB)"), ("af3", "AF3 (Drive outputs)")]:
        r = comparison[key]
        print(f"\n── {label} ({r['proteins_with_plddt']} proteins) ──")
        print(f"  Hallucination rate : {r['hallucination_rate']:.3f}")
        print(f"  Rescue rate        : {r['rescue_rate']:.3f}")
        if r["plddt_baseline_auc"] is not None:
            print(f"  pLDDT baseline AUC : {r['plddt_baseline_auc']:.4f}")
        if r["disordernet_auc"] is not None:
            print(f"  DisorderNet AUC    : {r['disordernet_auc']:.4f}")
        if r["delta_auc"] is not None:
            print(f"  Δ AUC              : {r['delta_auc']:+.4f}")

    print(f"\n── AF3 − AF2 ──")
    print(f"  Δ hallucination rate: {comparison['delta_hallucination_af3_minus_af2']:+.3f}")
    print(f"  Δ rescue rate       : {comparison['delta_rescue_af3_minus_af2']:+.3f}")
    print(f"{'═' * 64}")


def save_af2_af3_comparison(comparison: dict, path: str = "af2_af3_comparison.json") -> str:
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)
    return path


def _collect_af_subset_arrays(
    proteins: list,
    fold_results: list,
    plddt_by_protein: dict[str, np.ndarray],
    n_folds: int = 5,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    """Concatenate labels, DisorderNet probs, pLDDT on AF-covered residues."""
    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)
    labels_list: list = []
    probs_list: list = []
    plddt_list: list = []
    n_proteins = 0

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
        labels_list.append(item["labels"][valid])
        probs_list.append(item["probs"][valid])
        plddt_list.append(plddt[valid])
        n_proteins += 1

    if not labels_list:
        return None, None, None, 0

    return (
        np.concatenate(labels_list),
        np.concatenate(probs_list),
        np.concatenate(plddt_list),
        n_proteins,
    )


def _collect_overlap_arrays(
    proteins: list,
    fold_results: list,
    plddt_af2: dict[str, np.ndarray],
    plddt_af3: dict[str, np.ndarray],
    n_folds: int = 5,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    """Residues where both AF2 and AF3 pLDDT are available (same proteins)."""
    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)
    labels_list: list = []
    probs_list: list = []
    af2_list: list = []
    af3_list: list = []
    n_proteins = 0

    for item in aligned:
        pid = item["id"]
        if pid not in plddt_af2 or pid not in plddt_af3:
            continue
        p2 = np.asarray(plddt_af2[pid], dtype=np.float32)
        p3 = np.asarray(plddt_af3[pid], dtype=np.float32)
        if len(p2) != len(item["probs"]) or len(p3) != len(p2):
            continue
        valid = (~np.isnan(p2)) & (~np.isnan(p3))
        if valid.sum() < 5:
            continue
        labels_list.append(item["labels"][valid])
        probs_list.append(item["probs"][valid])
        af2_list.append(p2[valid])
        af3_list.append(p3[valid])
        n_proteins += 1

    if not labels_list:
        return None, None, None, None, 0

    return (
        np.concatenate(labels_list),
        np.concatenate(probs_list),
        np.concatenate(af2_list),
        np.concatenate(af3_list),
        n_proteins,
    )


def compute_fusion_lift_vs_plddt(
    proteins: list,
    fold_results_gpu: list,
    plddt_by_protein: dict[str, np.ndarray],
    fusion_alpha: float = 0.5,
    n_folds: int = 5,
) -> dict:
    """
    Δ AUC of fused predictions vs inverse-pLDDT baseline on AF-covered residues.
    """
    from colab.phase3_synthesis import fuse_disorder_score

    labels, gpu_probs, plddt, n_proteins = _collect_af_subset_arrays(
        proteins, fold_results_gpu, plddt_by_protein, n_folds=n_folds,
    )
    if labels is None or gpu_probs is None or plddt is None:
        return {"insufficient_data": True, "n_proteins": 0, "n_residues": 0}

    if len(np.unique(labels)) < 2:
        return {"insufficient_data": True, "n_proteins": n_proteins, "n_residues": int(len(labels))}

    baseline_scores = plddt_to_disorder_score(plddt)
    fused_scores = fuse_disorder_score(gpu_probs, plddt, alpha=fusion_alpha)

    baseline_auc = float(roc_auc_score(labels, baseline_scores))
    gpu_auc = float(roc_auc_score(labels, gpu_probs))
    fused_auc = float(roc_auc_score(labels, fused_scores))

    return {
        "insufficient_data": False,
        "n_proteins": n_proteins,
        "n_residues": int(len(labels)),
        "fusion_alpha": float(fusion_alpha),
        "plddt_baseline_auc": baseline_auc,
        "gpu_auc": gpu_auc,
        "fused_auc": fused_auc,
        "delta_fused_vs_plddt": fused_auc - baseline_auc,
        "delta_gpu_vs_plddt": gpu_auc - baseline_auc,
        "delta_fused_vs_gpu": fused_auc - gpu_auc,
    }


def run_af2_af3_breakthrough_summary(
    proteins: list,
    fold_results_gpu: list,
    af2_report: dict,
    af3_report: dict,
    af2_af3_comparison: dict,
    plddt_af2_by_protein: dict[str, np.ndarray],
    plddt_af3_by_protein: dict[str, np.ndarray],
    fusion_alpha: Optional[float] = None,
    fusion_report: Optional[dict] = None,
    threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    n_folds: int = 5,
) -> dict:
    """
    Three headline comparisons for breakthrough assessment after AF3 run.

    1. AF3 vs AF2 hallucination rate (pooled + same-protein overlap)
    2. DisorderNet rescue vs AF3 vs AF2 (rescue rate + Δ AUC)
    3. Fusion lift vs pLDDT baseline: larger on AF3 than AF2?
    """
    if fusion_alpha is None and fusion_report:
        fusion_alpha = fusion_report.get("fusion_alpha", 0.5)
    if fusion_alpha is None:
        fusion_alpha = 0.5

    summary: dict = {
        "insufficient_data": True,
        "fusion_alpha": float(fusion_alpha),
    }

    # ── Comparison 1: hallucination rate ──────────────────────────────────
    comp1: dict = {"insufficient_data": True}
    if not af2_af3_comparison.get("insufficient_data"):
        a2 = af2_af3_comparison["af2"]
        a3 = af2_af3_comparison["af3"]
        delta_h = af2_af3_comparison["delta_hallucination_af3_minus_af2"]
        comp1 = {
            "insufficient_data": False,
            "af2_hallucination_rate": a2["hallucination_rate"],
            "af3_hallucination_rate": a3["hallucination_rate"],
            "delta_af3_minus_af2": delta_h,
            "af3_hallucinates_more": delta_h > 0,
            "n_af2_proteins": a2["proteins_with_plddt"],
            "n_af3_proteins": a3["proteins_with_plddt"],
        }

    labels_ov, probs_ov, plddt2_ov, plddt3_ov, n_ov_proteins = _collect_overlap_arrays(
        proteins, fold_results_gpu, plddt_af2_by_protein, plddt_af3_by_protein, n_folds=n_folds,
    )
    if labels_ov is not None and probs_ov is not None and plddt2_ov is not None and plddt3_ov is not None:
        m2 = compute_hallucination_metrics(
            labels_ov, probs_ov, plddt2_ov, threshold, high_plddt_threshold,
        )
        m3 = compute_hallucination_metrics(
            labels_ov, probs_ov, plddt3_ov, threshold, high_plddt_threshold,
        )
        comp1["overlap"] = {
            "n_proteins": n_ov_proteins,
            "n_residues": m2["n_residues"],
            "af2_hallucination_rate": m2["hallucination_rate"],
            "af3_hallucination_rate": m3["hallucination_rate"],
            "delta_af3_minus_af2": m3["hallucination_rate"] - m2["hallucination_rate"],
            "af3_hallucinates_more": m3["hallucination_rate"] > m2["hallucination_rate"],
        }

    summary["comparison_1_hallucination"] = comp1

    # ── Comparison 2: DisorderNet rescue ──────────────────────────────────
    comp2: dict = {"insufficient_data": True}
    if not af2_af3_comparison.get("insufficient_data"):
        a2 = af2_af3_comparison["af2"]
        a3 = af2_af3_comparison["af3"]
        d_rescue = af2_af3_comparison["delta_rescue_af3_minus_af2"]
        d_auc2 = a2.get("delta_auc")
        d_auc3 = a3.get("delta_auc")
        comp2 = {
            "insufficient_data": False,
            "af2_rescue_rate": a2["rescue_rate"],
            "af3_rescue_rate": a3["rescue_rate"],
            "delta_rescue_af3_minus_af2": d_rescue,
            "af2_disordernet_delta_auc": d_auc2,
            "af3_disordernet_delta_auc": d_auc3,
            "disordernet_rescue_better_on_af3": (
                d_rescue > 0 or (d_auc3 is not None and d_auc2 is not None and d_auc3 > d_auc2)
            ),
        }
        if labels_ov is not None and probs_ov is not None and plddt2_ov is not None and plddt3_ov is not None:
            m2 = compute_hallucination_metrics(
                labels_ov, probs_ov, plddt2_ov, threshold, high_plddt_threshold,
            )
            m3 = compute_hallucination_metrics(
                labels_ov, probs_ov, plddt3_ov, threshold, high_plddt_threshold,
            )
            comp2["overlap"] = {
                "af2_rescue_rate": m2["rescue_rate"],
                "af3_rescue_rate": m3["rescue_rate"],
                "delta_rescue_af3_minus_af2": m3["rescue_rate"] - m2["rescue_rate"],
            }

    summary["comparison_2_rescue"] = comp2

    # ── Comparison 3: fusion lift vs pLDDT baseline ───────────────────────
    lift_af2 = compute_fusion_lift_vs_plddt(
        proteins, fold_results_gpu, plddt_af2_by_protein, fusion_alpha, n_folds,
    )
    lift_af3 = compute_fusion_lift_vs_plddt(
        proteins, fold_results_gpu, plddt_af3_by_protein, fusion_alpha, n_folds,
    )

    comp3: dict = {"insufficient_data": True}
    if not lift_af2.get("insufficient_data") and not lift_af3.get("insufficient_data"):
        d2 = lift_af2["delta_fused_vs_plddt"]
        d3 = lift_af3["delta_fused_vs_plddt"]
        comp3 = {
            "insufficient_data": False,
            "fusion_alpha": float(fusion_alpha),
            "af2": lift_af2,
            "af3": lift_af3,
            "delta_lift_af3_minus_af2": d3 - d2,
            "fusion_beats_plddt_more_on_af3": d3 > d2,
        }
    elif not lift_af2.get("insufficient_data"):
        comp3 = {"insufficient_data": True, "af2_only": lift_af2, "af3": lift_af3}
    elif not lift_af3.get("insufficient_data"):
        comp3 = {"insufficient_data": True, "af2": lift_af2, "af3_only": lift_af3}

    summary["comparison_3_fusion_lift"] = comp3
    summary["insufficient_data"] = (
        comp1.get("insufficient_data", True)
        and comp2.get("insufficient_data", True)
        and comp3.get("insufficient_data", True)
    )
    return summary


def print_af2_af3_breakthrough_summary(summary: dict) -> None:
    """Print the three headline AF2 vs AF3 breakthrough comparisons."""
    print(f"\n{'═' * 64}")
    print(" AF2 vs AF3 BREAKTHROUGH SUMMARY (Cell 11b)")
    print(f"{'═' * 64}")

    if summary.get("insufficient_data"):
        print("  Insufficient data — run Cells 10, 10b, and 11 with AF3 outputs first.")
        print(f"{'═' * 64}")
        return

    print(f"  Fusion α (for comp. 3): {summary.get('fusion_alpha', 0.5):.2f}")

    # 1. Hallucination
    c1 = summary.get("comparison_1_hallucination", {})
    print(f"\n── 1. AF3 vs AF2 hallucination rate (disordered + pLDDT ≥ 70) ──")
    if c1.get("insufficient_data"):
        print("  Insufficient data.")
    else:
        print(f"  AF2 pooled rate     : {c1['af2_hallucination_rate']:.3f}  "
              f"({c1['n_af2_proteins']} proteins)")
        print(f"  AF3 pooled rate     : {c1['af3_hallucination_rate']:.3f}  "
              f"({c1['n_af3_proteins']} proteins)")
        print(f"  Δ AF3 − AF2         : {c1['delta_af3_minus_af2']:+.3f}  "
              f"({'AF3 worse' if c1['af3_hallucinates_more'] else 'AF3 better or equal'})")
        ov = c1.get("overlap")
        if ov:
            print(f"  Same-protein overlap ({ov['n_proteins']} proteins, {ov['n_residues']:,} res):")
            print(f"    AF2 rate          : {ov['af2_hallucination_rate']:.3f}")
            print(f"    AF3 rate          : {ov['af3_hallucination_rate']:.3f}")
            print(f"    Δ AF3 − AF2       : {ov['delta_af3_minus_af2']:+.3f}")

    # 2. Rescue
    c2 = summary.get("comparison_2_rescue", {})
    print(f"\n── 2. DisorderNet rescue: AF3 vs AF2 ──")
    if c2.get("insufficient_data"):
        print("  Insufficient data.")
    else:
        print(f"  AF2 rescue rate     : {c2['af2_rescue_rate']:.3f}")
        print(f"  AF3 rescue rate     : {c2['af3_rescue_rate']:.3f}")
        print(f"  Δ rescue AF3 − AF2  : {c2['delta_rescue_af3_minus_af2']:+.3f}")
        if c2.get("af2_disordernet_delta_auc") is not None:
            print(f"  DisorderNet Δ AUC   : AF2 {c2['af2_disordernet_delta_auc']:+.4f}  "
                  f"AF3 {c2['af3_disordernet_delta_auc']:+.4f}")
        verdict = "YES" if c2.get("disordernet_rescue_better_on_af3") else "NO"
        print(f"  Rescue stronger on AF3? {verdict}")
        ov = c2.get("overlap")
        if ov:
            print(f"  Overlap Δ rescue    : {ov['delta_rescue_af3_minus_af2']:+.3f}")

    # 3. Fusion lift
    c3 = summary.get("comparison_3_fusion_lift", {})
    print(f"\n── 3. Fusion lift vs pLDDT baseline (AF3 vs AF2) ──")
    if c3.get("insufficient_data"):
        print("  Insufficient data (need GPU fold_results + both pLDDT sources).")
    else:
        a2, a3 = c3["af2"], c3["af3"]
        print(f"  AF2-covered residues: {a2['n_residues']:,} ({a2['n_proteins']} proteins)")
        print(f"    pLDDT baseline AUC  : {a2['plddt_baseline_auc']:.4f}")
        print(f"    Fused AUC           : {a2['fused_auc']:.4f}")
        print(f"    Δ fused − pLDDT     : {a2['delta_fused_vs_plddt']:+.4f}")
        print(f"  AF3-covered residues: {a3['n_residues']:,} ({a3['n_proteins']} proteins)")
        print(f"    pLDDT baseline AUC  : {a3['plddt_baseline_auc']:.4f}")
        print(f"    Fused AUC           : {a3['fused_auc']:.4f}")
        print(f"    Δ fused − pLDDT     : {a3['delta_fused_vs_plddt']:+.4f}")
        print(f"  Δ lift (AF3 − AF2)    : {c3['delta_lift_af3_minus_af2']:+.4f}  "
              f"({'fusion helps more on AF3' if c3['fusion_beats_plddt_more_on_af3'] else 'fusion helps more on AF2'})")

    print(f"{'═' * 64}")


def save_af2_af3_breakthrough_summary(
    summary: dict,
    path: str = "af2_af3_breakthrough_summary.json",
) -> str:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path
