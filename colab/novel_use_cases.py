"""
Novel DisorderNet use cases — differentiation vs sequence-only SOTA.

1. AF hallucination screening (structure overconfidence in IDRs)
2. Proteome disorder landscape summaries
3. AF pipeline rescue manifest (which regions to trust / re-predict)
4. Disorder → function: annotate IDR functional roles (binding, PTM, condensate)
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from colab.af_hallucination import compute_hallucination_metrics, compute_plddt_baseline_auc
from colab.af_plddt import plddt_to_disorder_score
from colab.biological_utility import intervals_from_binary


def screen_af_hallucinations(
    sequence: str,
    disorder_probs: np.ndarray,
    plddt: np.ndarray,
    protein_id: str = "query",
    disorder_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
) -> dict:
    """
    Flag residues where AlphaFold is overconfident in structure but DisorderNet
    predicts disorder — the core novel use case vs ESMDisPred (sequence-only).
    """
    n = min(len(sequence), len(disorder_probs), len(plddt))
    labels_proxy = (disorder_probs[:n] >= disorder_threshold).astype(np.int8)
    metrics = compute_hallucination_metrics(
        labels_proxy, disorder_probs[:n], plddt[:n],
        threshold=disorder_threshold,
        high_plddt_threshold=high_plddt_threshold,
    )

    halluc_mask = (
        (labels_proxy == 1)
        & (plddt[:n] >= high_plddt_threshold)
        & ~np.isnan(plddt[:n])
    )
    rescued_mask = halluc_mask & (disorder_probs[:n] >= disorder_threshold)

    flagged_regions = [(s, e) for s, e in intervals_from_binary(halluc_mask.astype(np.int8), min_len=3)]
    rescued_regions = [(s, e) for s, e in intervals_from_binary(rescued_mask.astype(np.int8), min_len=3)]

    return {
        "protein_id": protein_id,
        "length": n,
        "metrics": metrics,
        "n_hallucination_regions": len(flagged_regions),
        "n_rescued_regions": len(rescued_regions),
        "hallucination_regions": flagged_regions[:20],
        "rescued_regions": rescued_regions[:20],
        "recommendation": (
            "Use DisorderNet scores for these regions — AF structure is unreliable."
            if metrics.get("n_hallucinated", 0) > 0
            else "Low AF hallucination burden on disordered calls."
        ),
    }


def proteome_disorder_summary(
    proteins: list[dict],
    preds_by_id: dict[str, np.ndarray],
    disorder_threshold: float = 0.5,
) -> dict:
    """Per-protein disorder fraction and segment stats for proteome-scale runs."""
    rows: list[dict] = []
    for p in proteins:
        pid = p["id"]
        if pid not in preds_by_id:
            continue
        probs = np.asarray(preds_by_id[pid], dtype=np.float32)
        n = min(len(probs), p["length"])
        preds = (probs[:n] >= disorder_threshold).astype(int)
        segs = intervals_from_binary(preds.astype(np.int8), min_len=5)
        rows.append({
            "id": pid,
            "length": n,
            "disorder_fraction": float(preds.mean()),
            "n_idr_segments": len(segs),
            "longest_idr": max((e - s + 1 for s, e in segs), default=0),
            "mean_disorder_prob": float(probs[:n].mean()),
        })

    if not rows:
        return {"insufficient_data": True, "n_proteins": 0}

    fracs = [r["disorder_fraction"] for r in rows]
    return {
        "insufficient_data": False,
        "n_proteins": len(rows),
        "mean_disorder_fraction": float(np.mean(fracs)),
        "high_disorder_proteins": sorted(rows, key=lambda x: -x["disorder_fraction"])[:10],
        "low_disorder_proteins": sorted(rows, key=lambda x: x["disorder_fraction"])[:10],
        "per_protein": rows,
    }


def build_af_rescue_manifest(
    proteins: list[dict],
    preds_by_id: dict[str, np.ndarray],
    plddt_by_id: dict[str, np.ndarray],
    disorder_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
) -> dict:
    """
    Export manifest for AlphaFold pipeline integration:
    which proteins/regions need disorder-aware interpretation.
    """
    entries: list[dict] = []
    total_halluc = 0
    total_rescued = 0

    for p in proteins:
        pid = p["id"]
        if pid not in preds_by_id or pid not in plddt_by_id:
            continue
        screen = screen_af_hallucinations(
            p["sequence"], preds_by_id[pid], plddt_by_id[pid],
            protein_id=pid,
            disorder_threshold=disorder_threshold,
            high_plddt_threshold=high_plddt_threshold,
        )
        m = screen["metrics"]
        if m.get("n_hallucinated", 0) > 0:
            entries.append({
                "protein_id": pid,
                "uniprot_acc": p.get("uniprot_acc"),
                "n_hallucinated": m["n_hallucinated"],
                "n_rescued": m["n_rescued"],
                "rescue_rate": m["rescue_rate"],
                "hallucination_regions": screen["hallucination_regions"],
                "action": "prefer_disordernet_over_plddt",
            })
            total_halluc += m["n_hallucinated"]
            total_rescued += m["n_rescued"]

    return {
        "n_proteins_screened": len(entries),
        "total_hallucinated_residues": total_halluc,
        "total_rescued_residues": total_rescued,
        "overall_rescue_rate": float(total_rescued / total_halluc) if total_halluc else 0.0,
        "entries": entries,
        "use_case": "AlphaFold hallucination rescue — unique vs ESMDisPred",
    }




def annotate_idr_functions(
    proteins: list[dict],
    disorder_probs_by_id: dict[str, np.ndarray],
    function_probs_by_id: dict[str, np.ndarray],
    disorder_threshold: float = 0.5,
    function_threshold: float = 0.5,
) -> dict:
    """
    Proteome-scale Disorder → function annotation (novel vs Metapredict / ESMDisPred).

    For each protein, report predicted functional roles of IDR segments.
    """
    from colab.function_predict import predict_protein_functions

    rows: list[dict] = []
    n_roles = 0
    for p in proteins:
        pid = p["id"]
        if pid not in disorder_probs_by_id or pid not in function_probs_by_id:
            continue
        ann = predict_protein_functions(
            disorder_probs_by_id[pid],
            function_probs_by_id[pid],
            p["sequence"],
            protein_id=pid,
            disorder_threshold=disorder_threshold,
            function_threshold=function_threshold,
        )
        n_roles += sum(len(r["predicted_roles"]) for r in ann["idr_function_regions"])
        rows.append(ann)

    return {
        "use_case": "Disorder → function annotation of predicted IDRs",
        "n_proteins": len(rows),
        "n_role_assignments": n_roles,
        "proteins": rows[:50],  # cap for JSON size; full export via per-protein files
        "note": "Requires checkpoints trained with use_function_head / ultra_fun",
    }

def save_novel_use_case_report(report: dict, path: str) -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
