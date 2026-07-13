"""
Novel DisorderNet use cases — differentiation vs sequence-only SOTA.

1. AF / Boltz structure distrust screening (overconfidence on IDRs)
2. Proteome disorder landscape summaries
3. Structure-pipeline distrust manifest (which regions to re-interpret)
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
    labels: Optional[np.ndarray] = None,
) -> dict:
    """
    Flag residues where structure confidence is high but disorder evidence says IDR.

    Two definitions (do not conflate):

    - **labeled** (``labels`` provided): independent DisProt-style labels → true
      hallucination / rescue rates.
    - **proxy** (default): DisorderNet itself as the disorder call → deployment
      *distrust flags* only. ``rescue_rate`` is undefined / tautological and is
      reported as such — never treat proxy rescue as a scientific claim.
    """
    n = min(len(sequence), len(disorder_probs), len(plddt))
    dis = np.asarray(disorder_probs, dtype=np.float32).ravel()[:n]
    pld = np.asarray(plddt, dtype=np.float32).ravel()[:n]
    valid = ~np.isnan(pld)

    if labels is not None:
        lab = np.asarray(labels, dtype=np.int8).ravel()
        if len(lab) < n:
            pad = np.zeros(n, dtype=np.int8)
            pad[: len(lab)] = lab
            lab = pad
        else:
            lab = lab[:n]
        definition = "labeled_independent"
        metrics = compute_hallucination_metrics(
            lab, dis, pld,
            threshold=disorder_threshold,
            high_plddt_threshold=high_plddt_threshold,
        )
        metrics["definition"] = definition
        metrics["rescue_rate_valid"] = True
        halluc_mask = valid & (lab == 1) & (pld >= high_plddt_threshold)
        rescued_mask = halluc_mask & (dis >= disorder_threshold)
    else:
        definition = "proxy_distrust"
        halluc_mask = valid & (dis >= disorder_threshold) & (pld >= high_plddt_threshold)
        rescued_mask = halluc_mask  # tautological by construction
        n_halluc = int(halluc_mask.sum())
        metrics = {
            "n_residues": int(valid.sum()),
            "insufficient_data": int(valid.sum()) == 0,
            "high_plddt_threshold": float(high_plddt_threshold),
            "disorder_threshold": float(disorder_threshold),
            "definition": definition,
            "n_hallucinated": n_halluc,  # alias: proxy distrust residues
            "n_distrust_residues": n_halluc,
            "n_rescued": n_halluc,  # backward compat; NOT scientific rescue
            "rescue_rate": None,
            "rescue_rate_valid": False,
            "hallucination_rate": None,
            "note": (
                "Proxy distrust (DN∩high-pLDDT). rescue_rate is undefined — "
                "use labeled protocol for scientific rescue claims."
            ),
        }

    flagged_regions = [
        (s, e) for s, e in intervals_from_binary(halluc_mask.astype(np.int8), min_len=3)
    ]
    rescued_regions = [
        (s, e) for s, e in intervals_from_binary(rescued_mask.astype(np.int8), min_len=3)
    ]

    n_flag = int(metrics.get("n_hallucinated", 0) or 0)
    return {
        "protein_id": protein_id,
        "length": n,
        "definition": definition,
        "metrics": metrics,
        "n_hallucination_regions": len(flagged_regions),
        "n_rescued_regions": len(rescued_regions) if metrics.get("rescue_rate_valid") else 0,
        "hallucination_regions": flagged_regions[:20],
        "rescued_regions": rescued_regions[:20] if metrics.get("rescue_rate_valid") else [],
        "distrust_regions": flagged_regions[:20],
        "recommendation": (
            "Use DisorderNet scores for these regions — structure confidence unreliable."
            if n_flag > 0
            else "Low structure-distrust burden on this chain."
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
    labels_by_id: Optional[dict[str, np.ndarray]] = None,
) -> dict:
    """
    Export manifest for structure-pipeline integration:
    which proteins/regions need disorder-aware interpretation.

    Without labels this is a **proxy distrust** manifest (not scientific rescue).
    """
    labels_by_id = labels_by_id or {}
    entries: list[dict] = []
    total_halluc = 0
    total_rescued = 0
    any_labeled = False

    for p in proteins:
        pid = p["id"]
        if pid not in preds_by_id or pid not in plddt_by_id:
            continue
        screen = screen_af_hallucinations(
            p["sequence"], preds_by_id[pid], plddt_by_id[pid],
            protein_id=pid,
            disorder_threshold=disorder_threshold,
            high_plddt_threshold=high_plddt_threshold,
            labels=labels_by_id.get(pid),
        )
        m = screen["metrics"]
        if m.get("n_hallucinated", 0) > 0:
            entry = {
                "protein_id": pid,
                "uniprot_acc": p.get("uniprot_acc"),
                "definition": screen.get("definition"),
                "n_hallucinated": m["n_hallucinated"],
                "n_rescued": m.get("n_rescued"),
                "rescue_rate": m.get("rescue_rate"),
                "rescue_rate_valid": m.get("rescue_rate_valid", False),
                "hallucination_regions": screen["hallucination_regions"],
                "action": "prefer_disordernet_over_plddt",
            }
            entries.append(entry)
            total_halluc += int(m["n_hallucinated"] or 0)
            if m.get("rescue_rate_valid"):
                any_labeled = True
                total_rescued += int(m.get("n_rescued") or 0)

    return {
        "n_proteins_screened": len(entries),
        "total_hallucinated_residues": total_halluc,
        "total_rescued_residues": total_rescued if any_labeled else None,
        "overall_rescue_rate": (
            float(total_rescued / total_halluc) if any_labeled and total_halluc else None
        ),
        "definition": "labeled_independent" if any_labeled else "proxy_distrust",
        "entries": entries,
        "use_case": (
            "Structure distrust / hallucination rescue — post-AF default layer"
        ),
        "note": (
            "Rescue rates only set when independent labels were provided; "
            "proxy manifests flag DN∩high-pLDDT regions for triage."
        ),
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
        "n_proteins": len(rows),
        "total_role_assignments": n_roles,
        "proteins": rows,
        "use_case": "Disorder → function (IDR role annotation)",
    }


def save_novel_use_case_report(report: dict, path: str) -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
