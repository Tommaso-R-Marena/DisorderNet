"""
IDR Biology Layer — the post-structure default DisorderNet product.

Thesis: after Boltz/AF give a fold, DisorderNet answers what structure models
cannot by design:
  1. Where is the chain intrinsically disordered?
  2. Which of those regions look like binding / condensate / PTM / lipid roles?
  3. Where is the structure model overconfident (hallucination)?
  4. Optional cheap "ensemble proxy": Boltz multi-sample pLDDT variance

This module composes existing predictors into one proteome-exportable layer.
Full MD ensembles are explicitly out of scope (see docs/IDR_BIOLOGY_LAYER.md).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from colab.biological_utility import intervals_from_binary
from colab.novel_use_cases import screen_af_hallucinations

# Stable role taxonomy (aligned with FUNCTIONAL_TERM_GROUPS when available)
IDR_ROLE_GROUPS: tuple[str, ...] = (
    "protein binding",
    "nucleic acid binding",
    "post-translational regulation",
    "condensate / assembly",
    "lipid / small molecule binding",
)

LAYER_VERSION = "1.0.0"


def _as_f32(x: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32).ravel()
    if len(arr) < n:
        out = np.full(n, np.nan, dtype=np.float32)
        out[: len(arr)] = arr
        return out
    return arr[:n]


def build_idr_segment_records(
    disorder_probs: np.ndarray,
    *,
    function_probs: Optional[np.ndarray] = None,
    disorder_threshold: float = 0.5,
    function_threshold: float = 0.5,
    min_region_len: int = 5,
    role_names: tuple[str, ...] = IDR_ROLE_GROUPS,
) -> list[dict]:
    """IDR segments with optional multi-label role calls."""
    dis = np.asarray(disorder_probs, dtype=np.float32).ravel()
    mask = dis >= disorder_threshold
    segs = intervals_from_binary(mask.astype(np.int8), min_len=min_region_len)
    records: list[dict] = []
    fn = None
    if function_probs is not None:
        fn = np.asarray(function_probs, dtype=np.float32)
        if fn.ndim == 1:
            fn = fn.reshape(-1, 1)

    for start, end in segs:
        roles: list[dict] = []
        if fn is not None and fn.shape[0] >= end:
            slice_p = fn[start:end]
            mean_p = slice_p.mean(axis=0)
            max_p = slice_p.max(axis=0)
            for gi, name in enumerate(role_names):
                if gi >= mean_p.shape[0]:
                    break
                if mean_p[gi] >= function_threshold or max_p[gi] >= function_threshold + 0.15:
                    roles.append({
                        "group": name,
                        "mean_prob": float(mean_p[gi]),
                        "max_prob": float(max_p[gi]),
                    })
            roles.sort(key=lambda r: -r["mean_prob"])
        records.append({
            "start": start + 1,  # 1-based inclusive
            "end": end,
            "length": end - start,
            "mean_disorder_prob": float(dis[start:end].mean()),
            "predicted_roles": roles,
        })
    return records


def build_protein_idr_layer(
    *,
    protein_id: str,
    sequence: str,
    disorder_probs: np.ndarray,
    plddt: Optional[np.ndarray] = None,
    function_probs: Optional[np.ndarray] = None,
    boltz_plddt_std: Optional[np.ndarray] = None,
    uniprot_acc: str = "",
    disorder_threshold: float = 0.5,
    function_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    structure_source: str = "boltz2",
) -> dict:
    """
    Single-protein IDR biology layer record (JSON-serializable).

    This is the unit of proteome export — designed to sit *after* Boltz/AF.
    """
    n = len(sequence)
    dis = _as_f32(disorder_probs, n)
    assert dis is not None
    plddt_a = _as_f32(plddt, n)
    fn = None
    if function_probs is not None:
        raw = np.asarray(function_probs, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        if raw.shape[0] >= n:
            fn = raw[:n]
        else:
            pad = np.zeros((n, raw.shape[1]), dtype=np.float32)
            pad[: raw.shape[0]] = raw
            fn = pad

    segments = build_idr_segment_records(
        dis,
        function_probs=fn,
        disorder_threshold=disorder_threshold,
        function_threshold=function_threshold,
    )

    halluc = None
    if plddt_a is not None:
        halluc = screen_af_hallucinations(
            sequence, dis, plddt_a,
            protein_id=protein_id,
            disorder_threshold=disorder_threshold,
            high_plddt_threshold=high_plddt_threshold,
        )

    var = _as_f32(boltz_plddt_std, n)
    flexible_proxy_regions: list[tuple[int, int]] = []
    if var is not None:
        # High cross-sample disagreement in disordered regions → cheap ensemble cue
        finite = var[np.isfinite(var)]
        if finite.size:
            thr = float(np.percentile(finite, 75))
            high_var = (var >= thr) & np.isfinite(var)
            flex = (dis >= disorder_threshold) & high_var
            flexible_proxy_regions = [
                (s, e) for s, e in intervals_from_binary(flex.astype(np.int8), min_len=3)
            ][:20]

    n_role_calls = sum(len(s["predicted_roles"]) for s in segments)
    return {
        "layer_version": LAYER_VERSION,
        "protein_id": protein_id,
        "uniprot_acc": uniprot_acc or None,
        "length": n,
        "structure_source": structure_source if plddt_a is not None else None,
        "disorder_fraction": float((dis >= disorder_threshold).mean()),
        "mean_disorder_prob": float(dis.mean()),
        "n_idr_segments": len(segments),
        "idr_segments": segments,
        "n_role_assignments": n_role_calls,
        "hallucination": {
            "n_hallucinated": (halluc or {}).get("metrics", {}).get("n_hallucinated", 0),
            "n_rescued": (halluc or {}).get("metrics", {}).get("n_rescued", 0),
            "rescue_rate": (halluc or {}).get("metrics", {}).get("rescue_rate", 0.0),
            "regions": (halluc or {}).get("hallucination_regions", []),
            "recommendation": (halluc or {}).get("recommendation"),
        } if halluc else None,
        "ensemble_proxy": {
            "method": "boltz_multisample_plddt_std",
            "note": "Cheap flexibility cue — not a physical conformational ensemble",
            "n_high_variance_idr_regions": len(flexible_proxy_regions),
            "regions": flexible_proxy_regions,
            "mean_std": float(np.nanmean(var)) if var is not None else None,
        } if var is not None else None,
        "actions": _recommended_actions(segments, halluc, flexible_proxy_regions),
    }


def _recommended_actions(
    segments: list[dict],
    halluc: Optional[dict],
    flex_regions: list,
) -> list[str]:
    actions: list[str] = []
    if segments:
        actions.append("annotate_idrs_in_downstream_analysis")
    if any(s["predicted_roles"] for s in segments):
        roles = {r["group"] for s in segments for r in s["predicted_roles"]}
        if "condensate / assembly" in roles:
            actions.append("prioritize_phase_separation_assays")
        if "protein binding" in roles or "nucleic acid binding" in roles:
            actions.append("prioritize_binding_partner_screens")
    if halluc and halluc.get("metrics", {}).get("n_hallucinated", 0) > 0:
        actions.append("distrust_structure_model_in_hallucination_regions")
    if flex_regions:
        actions.append("treat_high_variance_idr_as_ensemble_proxy")
    return actions


def build_proteome_idr_layer(
    proteins: list[dict],
    disorder_probs_by_id: dict[str, np.ndarray],
    *,
    plddt_by_id: Optional[dict[str, np.ndarray]] = None,
    function_probs_by_id: Optional[dict[str, np.ndarray]] = None,
    boltz_std_by_id: Optional[dict[str, np.ndarray]] = None,
    disorder_threshold: float = 0.5,
    function_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    structure_source: str = "boltz2",
    max_proteins_in_summary: int = 100,
) -> dict:
    """Proteome-scale IDR biology layer + summary stats."""
    plddt_by_id = plddt_by_id or {}
    function_probs_by_id = function_probs_by_id or {}
    boltz_std_by_id = boltz_std_by_id or {}

    records: list[dict] = []
    for p in proteins:
        pid = p["id"]
        if pid not in disorder_probs_by_id:
            continue
        records.append(build_protein_idr_layer(
            protein_id=pid,
            sequence=p["sequence"],
            disorder_probs=disorder_probs_by_id[pid],
            plddt=plddt_by_id.get(pid),
            function_probs=function_probs_by_id.get(pid),
            boltz_plddt_std=boltz_std_by_id.get(pid),
            uniprot_acc=p.get("uniprot_acc", ""),
            disorder_threshold=disorder_threshold,
            function_threshold=function_threshold,
            high_plddt_threshold=high_plddt_threshold,
            structure_source=structure_source,
        ))

    n_halluc = sum((r.get("hallucination") or {}).get("n_hallucinated", 0) for r in records)
    n_roles = sum(r.get("n_role_assignments", 0) for r in records)
    n_condensate = sum(
        1 for r in records
        for s in r["idr_segments"]
        for role in s["predicted_roles"]
        if role["group"] == "condensate / assembly"
    )
    return {
        "layer_version": LAYER_VERSION,
        "thesis": (
            "Post-structure IDR biology layer: disorder map + functional roles + "
            "structure-hallucination flags (+ optional Boltz variance proxy)."
        ),
        "non_goals": [
            "full_md_conformational_ensembles",
            "alphafold_replacement",
        ],
        "n_proteins": len(records),
        "total_hallucinated_residues": n_halluc,
        "total_role_assignments": n_roles,
        "n_proteins_with_condensate_call": n_condensate,
        "mean_disorder_fraction": float(np.mean([r["disorder_fraction"] for r in records])) if records else 0.0,
        "proteins": records[:max_proteins_in_summary],
        "note": (
            f"Summary includes first {max_proteins_in_summary} proteins; "
            "use JSONL export for full proteome."
        ),
    }


def export_idr_layer_jsonl(records: list[dict], path: str) -> str:
    """Write one protein layer record per line (proteome-scale)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def save_idr_layer_report(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def print_idr_layer_report(report: dict) -> None:
    print(f"\n{'═' * 60}")
    print(" IDR Biology Layer (post-structure default)")
    print(f"{'═' * 60}")
    print(f"  version={report.get('layer_version')}  proteins={report.get('n_proteins', 0)}")
    print(f"  mean disorder fraction={report.get('mean_disorder_fraction', 0):.3f}")
    print(f"  role assignments={report.get('total_role_assignments', 0)}")
    print(f"  hallucinated residues={report.get('total_hallucinated_residues', 0)}")
    print(f"  condensate-role proteins={report.get('n_proteins_with_condensate_call', 0)}")
    print(f"  thesis: {report.get('thesis', '')[:90]}…")
