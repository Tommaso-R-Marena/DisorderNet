"""
Labeled structure-distrust / hallucination benchmark (paper architecture 1).

Protocol rule: rescue rates are ONLY valid when disorder labels are independent
of DisorderNet predictions (e.g. DisProt). Proxy screening (DN≥θ ∩ pLDDT≥70)
must never be reported as scientific rescue.

This module is the evaluation spine for the claim:
  "DisorderNet is the default post-structure distrust layer after AF/Boltz."
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from colab.af_hallucination import (
    compute_hallucination_metrics,
    compute_plddt_baseline_auc,
    run_af_rescue_report,
)
from colab.af_plddt import plddt_to_disorder_score
from colab.biological_utility import align_fold_predictions
from sklearn.metrics import average_precision_score, roc_auc_score


PROTOCOL_VERSION = "1.0.0"


def _safe_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(s, dtype=np.float32)
    if len(y) < 5 or len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def _safe_ap(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(s, dtype=np.float32)
    if len(y) < 5 or int(y.sum()) == 0:
        return None
    try:
        return float(average_precision_score(y, s))
    except ValueError:
        return None


def compare_distrust_baselines(
    labels: np.ndarray,
    disorder_probs: np.ndarray,
    plddt: np.ndarray,
    *,
    disorder_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
) -> dict:
    """
    Head-to-head on matched residues with valid pLDDT:

    - DisorderNet disorder probabilities
    - Inverse-pLDDT baseline (structure-only)
    - Distrust-priority score: upweight high-pLDDT disagreement cue

    All metrics use *independent* ``labels`` (never DN hard calls).
    """
    labels = np.asarray(labels, dtype=np.int8).ravel()
    probs = np.asarray(disorder_probs, dtype=np.float32).ravel()
    plddt = np.asarray(plddt, dtype=np.float32).ravel()
    n = min(len(labels), len(probs), len(plddt))
    labels, probs, plddt = labels[:n], probs[:n], plddt[:n]
    valid = ~np.isnan(plddt)
    if int(valid.sum()) < 5:
        return {"enabled": False, "insufficient_data": True, "n_residues": int(valid.sum())}

    y = labels[valid]
    dn = probs[valid]
    pld = plddt[valid]
    inv = plddt_to_disorder_score(pld)
    # Soft distrust score: DN disorder × structure overconfidence (high pLDDT)
    overconf = np.clip(pld / 100.0, 0.0, 1.0)
    distrust_score = dn * overconf

    hall = compute_hallucination_metrics(
        y, dn, pld,
        threshold=disorder_threshold,
        high_plddt_threshold=high_plddt_threshold,
    )
    plddt_base = compute_plddt_baseline_auc(y, pld)

    return {
        "enabled": True,
        "insufficient_data": False,
        "protocol_version": PROTOCOL_VERSION,
        "definition": "labeled_independent",
        "n_residues": int(valid.sum()),
        "disordernet": {
            "auc": _safe_auc(y, dn),
            "ap": _safe_ap(y, dn),
        },
        "plddt_inverse_baseline": plddt_base,
        "distrust_priority_score": {
            "auc": _safe_auc(y, distrust_score),
            "ap": _safe_ap(y, distrust_score),
            "note": "DN_prob × (pLDDT/100) — ranks structure-overconfident disordered sites",
        },
        "hallucination_rescue": hall,
        "delta_auc_dn_minus_plddt": (
            None
            if _safe_auc(y, dn) is None or plddt_base.get("auc") is None
            else round(float(_safe_auc(y, dn) - plddt_base["auc"]), 4)
        ),
        "thresholds": {
            "disorder_threshold": disorder_threshold,
            "high_plddt_threshold": high_plddt_threshold,
        },
    }


def attach_caid3_credibility_floor(
    bench: dict,
    caid3_report: Optional[dict] = None,
) -> dict:
    """Attach optional CAID3 disorder evaluation as credibility floor."""
    bench = dict(bench)
    if not caid3_report:
        bench["caid3_credibility_floor"] = {
            "available": False,
            "note": "No caid3_eval_report.json attached",
        }
        return bench
    pooled = caid3_report.get("pooled") or caid3_report.get("metrics") or {}
    bench["caid3_credibility_floor"] = {
        "available": True,
        "auc": pooled.get("auc") or pooled.get("AUC"),
        "ap": pooled.get("ap") or pooled.get("AP"),
        "n_scored": (
            caid3_report.get("n_scored")
            or pooled.get("n_residues")
            or caid3_report.get("n_proteins")
        ),
        "delta_vs_esmdispred": caid3_report.get("delta_vs_esmdispred"),
        "source_keys": sorted(caid3_report.keys())[:20],
        "note": "Disorder competitiveness floor — not a hallucination metric",
    }
    return bench


def finalize_distrust_benchmark_with_caid3(
    checkpoint_dir: str,
    cfg=None,
    *,
    regenerate_figure: bool = True,
) -> Optional[dict]:
    """
    Patch structure_distrust_benchmark.json in-place after CAID3 lands.

    Eval usually runs before CAID3 in the Rockfish pipeline, so the first
    benchmark write has no credibility floor. Call this after CAID3 to attach
    the floor without re-running the full labeled eval.
    """
    import os

    bench_path = os.path.join(checkpoint_dir, "structure_distrust_benchmark.json")
    caid3_path = os.path.join(checkpoint_dir, "caid3_eval_report.json")
    if not os.path.isfile(bench_path):
        return None
    if not os.path.isfile(caid3_path):
        return None

    with open(bench_path) as f:
        bench = json.load(f)
    with open(caid3_path) as f:
        caid3_report = json.load(f)

    bench = attach_caid3_credibility_floor(bench, caid3_report)
    if cfg is not None:
        try:
            from colab.training_contamination_audit import attach_contamination_flags
            bench = attach_contamination_flags(bench, cfg)
        except Exception:
            pass

    save_distrust_benchmark(bench, bench_path)

    if regenerate_figure:
        try:
            from colab.colab_figures import generate_distrust_benchmark_figure
            fig_dir = os.path.join(checkpoint_dir, "distrust_figures")
            generate_distrust_benchmark_figure(bench, out_dir=fig_dir)
        except Exception:
            pass

    return bench


def run_labeled_distrust_benchmark(
    proteins: list,
    fold_results: list,
    plddt_by_id: dict[str, np.ndarray],
    *,
    n_folds: int = 5,
    disorder_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    structure_source: str = "af2",
    cfg=None,
    caid3_report: Optional[dict] = None,
) -> dict:
    """
    Full labeled benchmark: Phase-2 style rescue report + matched baselines.

    Requires DisProt (or equivalent) labels via fold OOF alignments.
    """
    from colab.training_contamination_audit import attach_contamination_flags

    rescue = run_af_rescue_report(
        proteins,
        fold_results,
        plddt_by_id,
        threshold=disorder_threshold,
        high_plddt_threshold=high_plddt_threshold,
        n_folds=n_folds,
        source=structure_source,
    )

    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    all_plddt: list[np.ndarray] = []
    for item in aligned:
        pid = item["id"]
        if pid not in plddt_by_id:
            continue
        L = len(item["probs"])
        pld = np.asarray(plddt_by_id[pid], dtype=np.float32).ravel()
        if len(pld) < L:
            continue
        all_y.append(np.asarray(item["labels"], dtype=np.int8).ravel()[:L])
        all_p.append(np.asarray(item["probs"], dtype=np.float32).ravel()[:L])
        all_plddt.append(pld[:L])

    if not all_y:
        baselines = {"enabled": False, "insufficient_data": True}
    else:
        baselines = compare_distrust_baselines(
            np.concatenate(all_y),
            np.concatenate(all_p),
            np.concatenate(all_plddt),
            disorder_threshold=disorder_threshold,
            high_plddt_threshold=high_plddt_threshold,
        )

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "claim": (
            "Post-structure distrust layer: independent labels required for "
            "hallucination rescue rates; pLDDT-only is the null baseline"
        ),
        "structure_source": structure_source,
        "labeled_rescue_report": rescue,
        "matched_baselines": baselines,
        "non_claims": [
            "proxy_DN_threshold_intersection_is_not_rescue",
            "not_a_conformational_ensemble_predictor",
            "not_an_alphafold_replacement",
        ],
    }
    report = attach_contamination_flags(report, cfg)
    report = attach_caid3_credibility_floor(report, caid3_report)
    return report


def save_distrust_benchmark(report: dict, path: str) -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def print_distrust_benchmark(report: dict) -> None:
    print(f"\n{'═' * 60}")
    print(" Structure distrust benchmark (labeled protocol)")
    print(f"{'═' * 60}")
    print(f"  protocol={report.get('protocol_version')}  source={report.get('structure_source')}")
    rescue = report.get("labeled_rescue_report") or {}
    pooled = rescue.get("pooled") or {}
    if pooled:
        print(
            f"  halluc_rate={pooled.get('hallucination_rate')}  "
            f"rescue_rate={pooled.get('rescue_rate')}  "
            f"n_halluc={pooled.get('n_hallucinated')}"
        )
    base = report.get("matched_baselines") or {}
    if base.get("enabled"):
        dn = base.get("disordernet") or {}
        pl = base.get("plddt_inverse_baseline") or {}
        print(
            f"  matched AUC  DN={dn.get('auc')}  inv-pLDDT={pl.get('auc')}  "
            f"Δ={base.get('delta_auc_dn_minus_plddt')}"
        )
    print("  non-claims:", ", ".join(report.get("non_claims") or []))
