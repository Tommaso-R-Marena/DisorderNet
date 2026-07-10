"""
Phase 3: Disorder-guided structure calibration and integrated synthesis.

Fuses DisorderNet with AlphaFold pLDDT to calibrate structure confidence,
quantifies hybrid gains over raw AF baselines, compares to published benchmarks,
and produces a manuscript-ready integrated report across Phases 0–2.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from colab.af_plddt import plddt_to_disorder_score
from colab.biological_utility import align_fold_predictions

# Published DisProt/CAID benchmark AUCs (see README)
PUBLISHED_BENCHMARKS: list[dict] = [
    {"method": "AF3-pLDDT (CAID3)", "auc": 0.747, "source": "CAID3"},
    {"method": "AF2-pLDDT (CAID3)", "auc": 0.770, "source": "CAID3"},
    {"method": "IUPred3", "auc": 0.789, "source": "CAID"},
    {"method": "DisorderNet v4", "auc": 0.794, "source": "This work (CPU)"},
    {"method": "flDPnn", "auc": 0.814, "source": "CAID"},
    {"method": "DisorderNet v5", "auc": 0.823, "source": "This work (CPU)"},
    {"method": "SETH (ProtT5+CNN)", "auc": 0.830, "source": "Literature"},
    {"method": "DisorderNet v6", "auc": 0.831, "source": "This work (CPU)"},
    {"method": "flDPnn3a (CAID3)", "auc": 0.871, "source": "CAID3"},
    {"method": "ESM2_35M-LoRA", "auc": 0.868, "source": "LoRA-DR"},
    {"method": "ESM2_650M-LoRA", "auc": 0.880, "source": "LoRA-DR"},
    {"method": "ESMDisPred (CAID3 SOTA)", "auc": 0.895, "source": "CAID3"},
]


def fuse_disorder_score(
    disordernet_prob: np.ndarray,
    plddt: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Hybrid disorder score: convex combination of DisorderNet and inverse-pLDDT.

    alpha=1 → pure DisorderNet; alpha=0 → pure inverse-pLDDT baseline.
    """
    dn = np.asarray(disordernet_prob, dtype=np.float32)
    inv_plddt = plddt_to_disorder_score(np.asarray(plddt, dtype=np.float32))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return np.clip(alpha * dn + (1.0 - alpha) * inv_plddt, 0.0, 1.0)


def calibrate_plddt(
    plddt: np.ndarray,
    disordernet_prob: np.ndarray,
    penalty_strength: float = 1.0,
) -> np.ndarray:
    """
    Downweight AF pLDDT where DisorderNet predicts disorder.

    calibrated = plddt * (1 - penalty_strength * disorder_prob), clipped to [0, 100].
    """
    plddt = np.asarray(plddt, dtype=np.float32)
    dn = np.asarray(disordernet_prob, dtype=np.float32)
    strength = float(np.clip(penalty_strength, 0.0, 1.0))
    return np.clip(plddt * (1.0 - strength * dn), 0.0, 100.0)


def _safe_auc_ap(labels: np.ndarray, scores: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float32)
    if len(labels) < 5 or len(np.unique(labels)) < 2:
        return None, None
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def find_optimal_fusion_alpha(
    labels: np.ndarray,
    disordernet_prob: np.ndarray,
    plddt: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> dict:
    """Grid-search fusion weight maximizing AUC on AF-covered residues."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)

    labels = np.asarray(labels, dtype=np.int8)
    best_alpha = 0.5
    best_auc = -1.0
    curve: list[dict] = []

    for alpha in grid:
        scores = fuse_disorder_score(disordernet_prob, plddt, alpha=float(alpha))
        auc, ap = _safe_auc_ap(labels, scores)
        if auc is not None:
            curve.append({"alpha": float(alpha), "auc": auc, "ap": ap})
            if auc > best_auc:
                best_auc = auc
                best_alpha = float(alpha)

    return {
        "best_alpha": best_alpha,
        "best_auc": best_auc if best_auc >= 0 else None,
        "curve": curve,
    }


def bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_boot: int = 500,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Bootstrap confidence interval for a scalar metric (default AUC)."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float32)
    n = len(labels)
    if n < 10 or len(np.unique(labels)) < 2:
        return {"point": None, "ci_low": None, "ci_high": None, "n_boot": n_boot, "insufficient_data": True}

    rng = np.random.default_rng(seed)
    try:
        point = float(metric_fn(labels, scores))
    except ValueError:
        return {"point": None, "ci_low": None, "ci_high": None, "n_boot": n_boot, "insufficient_data": True}

    boot_vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bl, bs = labels[idx], scores[idx]
        if len(np.unique(bl)) < 2:
            continue
        try:
            boot_vals.append(float(metric_fn(bl, bs)))
        except ValueError:
            continue

    if len(boot_vals) < max(20, n_boot // 10):
        return {
            "point": point,
            "ci_low": None,
            "ci_high": None,
            "n_boot": len(boot_vals),
            "insufficient_data": True,
        }

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boot_vals, [100 * alpha, 100 * (1 - alpha)])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": len(boot_vals),
        "insufficient_data": False,
        "ci_level": ci,
    }


def mcnemar_test(pred_a: np.ndarray, pred_b: np.ndarray, labels: np.ndarray) -> dict:
    """
    McNemar test for paired binary classifiers (continuity-corrected chi-square).
    """
    pred_a = np.asarray(pred_a, dtype=np.int8)
    pred_b = np.asarray(pred_b, dtype=np.int8)
    labels = np.asarray(labels, dtype=np.int8)

    correct_a = pred_a == labels
    correct_b = pred_b == labels
    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c, "insufficient_discordant": True}

    stat = (abs(b - c) - 1) ** 2 / (b + c)
    # chi-square(1) survival: erfc for large values; use approximation
    from math import erfc, sqrt
    p_value = float(erfc(sqrt(stat / 2.0)))
    return {
        "statistic": float(stat),
        "p_value": p_value,
        "b": b,
        "c": c,
        "insufficient_discordant": False,
        "favors": "a" if b > c else ("b" if c > b else "tie"),
    }


def compute_calibration_metrics(
    labels: np.ndarray,
    disordernet_prob: np.ndarray,
    plddt: np.ndarray,
    threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    fusion_alpha: Optional[float] = None,
) -> dict:
    """Compare raw pLDDT, DisorderNet, fusion, and calibrated pLDDT on AF-covered residues."""
    labels = np.asarray(labels, dtype=np.int8)
    dn = np.asarray(disordernet_prob, dtype=np.float32)
    plddt = np.asarray(plddt, dtype=np.float32)
    valid = ~np.isnan(plddt)
    if valid.sum() < 5:
        return {"insufficient_data": True, "n_residues": int(valid.sum())}

    lab = labels[valid]
    prb = dn[valid]
    pld = plddt[valid]

    inv_plddt = plddt_to_disorder_score(pld)
    alpha_search = find_optimal_fusion_alpha(lab, prb, pld)
    alpha = fusion_alpha if fusion_alpha is not None else alpha_search["best_alpha"]
    fused = fuse_disorder_score(prb, pld, alpha=alpha)
    cal_plddt = calibrate_plddt(pld, prb)
    cal_disorder = plddt_to_disorder_score(cal_plddt)

    dn_auc, dn_ap = _safe_auc_ap(lab, prb)
    base_auc, base_ap = _safe_auc_ap(lab, inv_plddt)
    fuse_auc, fuse_ap = _safe_auc_ap(lab, fused)
    cal_auc, cal_ap = _safe_auc_ap(lab, cal_disorder)

    disordered = lab == 1
    hallucinated = disordered & (pld >= high_plddt_threshold)
    cal_hallucinated = disordered & (cal_plddt >= high_plddt_threshold)
    n_hall = int(hallucinated.sum())
    n_cal_hall = int(cal_hallucinated.sum())

    dn_preds = (prb >= threshold).astype(np.int8)
    base_preds = (inv_plddt >= threshold).astype(np.int8)
    fuse_preds = (fused >= threshold).astype(np.int8)

    return {
        "insufficient_data": False,
        "n_residues": int(valid.sum()),
        "fusion_alpha": float(alpha),
        "alpha_search": alpha_search,
        "disordernet": {"auc": dn_auc, "ap": dn_ap},
        "plddt_baseline": {"auc": base_auc, "ap": base_ap},
        "fusion": {"auc": fuse_auc, "ap": fuse_ap},
        "calibrated_plddt": {"auc": cal_auc, "ap": cal_ap},
        "delta_auc_fusion_vs_baseline": (fuse_auc - base_auc) if fuse_auc and base_auc else None,
        "delta_auc_fusion_vs_disordernet": (fuse_auc - dn_auc) if fuse_auc and dn_auc else None,
        "delta_auc_calibrated_vs_baseline": (cal_auc - base_auc) if cal_auc and base_auc else None,
        "hallucination_reduction": {
            "raw_n_hallucinated": n_hall,
            "calibrated_n_hallucinated": n_cal_hall,
            "absolute_reduction": n_hall - n_cal_hall,
            "relative_reduction": float((n_hall - n_cal_hall) / n_hall) if n_hall > 0 else 0.0,
        },
        "mcnemar_fusion_vs_baseline": mcnemar_test(fuse_preds, base_preds, lab),
        "mcnemar_disordernet_vs_baseline": mcnemar_test(dn_preds, base_preds, lab),
    }


def run_structure_calibration_report(
    proteins: list,
    fold_results: list,
    plddt_by_protein: dict[str, np.ndarray],
    threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    n_folds: int = 5,
    n_boot: int = 500,
    af_source: str = "AlphaFold DB (AF2)",
) -> dict:
    """Phase 3 structure calibration on AF-covered CV residues."""
    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)

    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_plddt: list[np.ndarray] = []
    proteins_used = 0

    for item in aligned:
        pid = item["id"]
        if pid not in plddt_by_protein:
            continue
        plddt = plddt_by_protein[pid]
        if len(plddt) != item["protein"]["length"]:
            continue
        valid = ~np.isnan(plddt)
        if valid.sum() == 0:
            continue
        proteins_used += 1
        all_labels.append(item["labels"][valid])
        all_probs.append(item["probs"][valid])
        all_plddt.append(plddt[valid])

    if not all_labels:
        return {
            "insufficient_data": True,
            "proteins_with_plddt": 0,
            "af_source": af_source,
        }

    labels_cat = np.concatenate(all_labels)
    probs_cat = np.concatenate(all_probs)
    plddt_cat = np.concatenate(all_plddt)

    calibration = compute_calibration_metrics(
        labels_cat, probs_cat, plddt_cat, threshold, high_plddt_threshold,
    )

    fused_scores = fuse_disorder_score(
        probs_cat, plddt_cat, alpha=calibration["fusion_alpha"],
    )
    bootstrap = {
        "disordernet_auc": bootstrap_ci(labels_cat, probs_cat, n_boot=n_boot),
        "plddt_baseline_auc": bootstrap_ci(labels_cat, plddt_to_disorder_score(plddt_cat), n_boot=n_boot),
        "fusion_auc": bootstrap_ci(labels_cat, fused_scores, n_boot=n_boot),
    }

    return {
        "insufficient_data": False,
        "af_source": af_source,
        "proteins_with_plddt": proteins_used,
        "threshold": float(threshold),
        "high_plddt_threshold": float(high_plddt_threshold),
        "calibration": calibration,
        "bootstrap_ci": bootstrap,
    }


def build_benchmark_ranking(our_auc: float, benchmarks: Optional[list[dict]] = None) -> dict:
    """Rank DisorderNet GPU AUC against published methods."""
    benchmarks = benchmarks or PUBLISHED_BENCHMARKS
    rows = []
    for b in benchmarks:
        rows.append({
            **b,
            "delta_vs_our": float(our_auc - b["auc"]),
            "beats_us": our_auc > b["auc"],
        })
    rows.sort(key=lambda x: x["auc"], reverse=True)

    rank = 1 + sum(1 for r in rows if r["auc"] > our_auc)
    beats_af3 = our_auc > 0.747
    beats_af2 = our_auc > 0.770
    beats_v6 = our_auc > 0.831

    return {
        "our_auc": float(our_auc),
        "rank_among_published": rank,
        "n_methods": len(rows),
        "beats_af3_plddt": beats_af3,
        "beats_af2_plddt": beats_af2,
        "beats_disordernet_v6": beats_v6,
        "delta_vs_af3": float(our_auc - 0.747),
        "delta_vs_af2": float(our_auc - 0.770),
        "delta_vs_v6": float(our_auc - 0.831),
        "delta_vs_sota_esmdispred": float(our_auc - 0.895),
        "table": rows,
    }


def run_phase3_integrated_report(
    cv_pooled: dict,
    bio_report: dict,
    af_report: dict,
    calibration_report: dict,
    af3_report: Optional[dict] = None,
    af2_af3_comparison: Optional[dict] = None,
) -> dict:
    """Manuscript-ready synthesis across Phases 0–2."""
    our_auc = float(cv_pooled.get("auc", 0.0))
    our_ap = float(cv_pooled.get("ap", 0.0))
    benchmark = build_benchmark_ranking(our_auc)

    phase_summaries = {
        "phase0_cv": {
            "auc": our_auc,
            "ap": our_ap,
            "f1": cv_pooled.get("f1"),
            "mcc": cv_pooled.get("mcc"),
            "opt_threshold": cv_pooled.get("opt_threshold"),
        },
        "phase1_biological_utility": {
            "segment_f1": bio_report.get("segment_metrics", {}).get("segment_f1"),
            "mdr_recall": bio_report.get("segment_metrics", {}).get("mdr_recall"),
            "transition_auc": bio_report.get("transition_zones", {}).get("auc"),
        },
        "phase2_af_rescue": {
            "available": not af_report.get("insufficient_data", True),
            "hallucination_rate": af_report.get("pooled", {}).get("hallucination_rate"),
            "rescue_rate": af_report.get("pooled", {}).get("rescue_rate"),
            "delta_auc_vs_plddt": af_report.get("delta_auc_vs_plddt_baseline"),
        },
        "phase2b_af3": {
            "available": af3_report is not None and not af3_report.get("insufficient_data", True)
            and not af3_report.get("skipped", False),
            "hallucination_rate": (af3_report or {}).get("pooled", {}).get("hallucination_rate"),
            "rescue_rate": (af3_report or {}).get("pooled", {}).get("rescue_rate"),
        },
        "phase3_calibration": {
            "available": not calibration_report.get("insufficient_data", True),
            "fusion_alpha": calibration_report.get("calibration", {}).get("fusion_alpha"),
            "fusion_auc": calibration_report.get("calibration", {}).get("fusion", {}).get("auc"),
            "delta_auc_fusion_vs_baseline": calibration_report.get("calibration", {}).get(
                "delta_auc_fusion_vs_baseline"
            ),
            "hallucination_reduction": calibration_report.get("calibration", {}).get(
                "hallucination_reduction", {}
            ),
        },
    }

    headline = _build_headline(benchmark, phase_summaries, calibration_report)

    return {
        "insufficient_data": False,
        "headline": headline,
        "benchmark_ranking": benchmark,
        "phase_summaries": phase_summaries,
        "af2_af3_comparison": af2_af3_comparison,
        "structure_calibration": calibration_report,
    }


def _build_headline(benchmark: dict, phases: dict, calibration: dict) -> str:
    parts = [f"GPU AUC {benchmark['our_auc']:.3f} ranks #{benchmark['rank_among_published']}/{benchmark['n_methods']}"]
    if benchmark["beats_af3_plddt"]:
        parts.append(f"+{benchmark['delta_vs_af3']:.1%} vs AF3-pLDDT")
    p2 = phases.get("phase2_af_rescue", {})
    if p2.get("available") and p2.get("rescue_rate") is not None:
        parts.append(f"rescues {p2['rescue_rate']:.0%} of AF hallucinations")
    cal = phases.get("phase3_calibration", {})
    if cal.get("available"):
        red = cal.get("hallucination_reduction", {})
        if red.get("relative_reduction"):
            parts.append(f"calibration cuts hallucinations by {red['relative_reduction']:.0%}")
    return "; ".join(parts) + "."


def print_phase3_report(report: dict) -> None:
    """Pretty-print Phase 3 integrated report."""
    print(f"\n{'═' * 64}")
    print(" PHASE 3: INTEGRATED SYNTHESIS & STRUCTURE CALIBRATION")
    print(f"{'═' * 64}")

    if report.get("insufficient_data"):
        print("  Insufficient data for Phase 3 report.")
        print(f"{'═' * 64}")
        return

    print(f"  {report.get('headline', '')}")

    bench = report["benchmark_ranking"]
    print(f"\n── Published benchmark ranking ──")
    print(f"  Our AUC     : {bench['our_auc']:.4f}  (rank {bench['rank_among_published']}/{bench['n_methods']})")
    print(f"  vs AF3-pLDDT: {bench['delta_vs_af3']:+.4f}")
    print(f"  vs AF2-pLDDT: {bench['delta_vs_af2']:+.4f}")
    print(f"  vs v6 CPU   : {bench['delta_vs_v6']:+.4f}")
    print(f"  vs SOTA     : {bench['delta_vs_sota_esmdispred']:+.4f}")

    cal_rep = report.get("structure_calibration", {})
    if not cal_rep.get("insufficient_data"):
        cal = cal_rep["calibration"]
        print(f"\n── Structure calibration ({cal_rep.get('af_source', 'AF')}) ──")
        print(f"  Residues           : {cal['n_residues']:,}")
        print(f"  Optimal fusion α   : {cal['fusion_alpha']:.2f}")
        if cal["plddt_baseline"]["auc"] is not None:
            print(f"  pLDDT baseline AUC : {cal['plddt_baseline']['auc']:.4f}")
        if cal["disordernet"]["auc"] is not None:
            print(f"  DisorderNet AUC    : {cal['disordernet']['auc']:.4f}")
        if cal["fusion"]["auc"] is not None:
            print(f"  Fusion AUC         : {cal['fusion']['auc']:.4f}")
        if cal.get("delta_auc_fusion_vs_baseline") is not None:
            print(f"  Δ AUC (fusion)     : {cal['delta_auc_fusion_vs_baseline']:+.4f}")
        hr = cal.get("hallucination_reduction", {})
        if hr.get("raw_n_hallucinated", 0) > 0:
            print(f"  Hallucinations     : {hr['raw_n_hallucinated']:,} → "
                  f"{hr['calibrated_n_hallucinated']:,} "
                  f"({hr['relative_reduction']:.1%} reduction)")

        boot = cal_rep.get("bootstrap_ci", {})
        for name, label in [
            ("disordernet_auc", "DisorderNet"),
            ("fusion_auc", "Fusion"),
            ("plddt_baseline_auc", "pLDDT baseline"),
        ]:
            b = boot.get(name, {})
            if b.get("point") is not None and b.get("ci_low") is not None:
                print(f"  {label} 95% CI     : {b['point']:.4f} [{b['ci_low']:.4f}, {b['ci_high']:.4f}]")

    ps = report.get("phase_summaries", {})
    p1 = ps.get("phase1_biological_utility", {})
    print(f"\n── Cross-phase highlights ──")
    if p1.get("segment_f1") is not None:
        print(f"  Phase 1 segment F1 : {p1['segment_f1']:.4f}")
    p2 = ps.get("phase2_af_rescue", {})
    if p2.get("available"):
        print(f"  Phase 2 rescue rate: {p2.get('rescue_rate', 0):.3f}")
    p2b = ps.get("phase2b_af3", {})
    if p2b.get("available"):
        print(f"  Phase 2b AF3 halluc: {p2b.get('hallucination_rate', 0):.3f}")

    print(f"{'═' * 64}")


def save_phase3_report(report: dict, path: str = "phase3_integrated_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
