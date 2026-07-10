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
