"""
IDR Biology Layer — the post-structure default DisorderNet product.

Thesis: after Boltz/AF give a fold, DisorderNet answers what structure models
cannot by design:
  1. Where is the chain intrinsically disordered?
  2. What might those IDRs do? (roles + sequence cues)
  3. Where is the structure model overconfident (hallucination)?
  4. Optional cheap ensemble proxy: Boltz multi-sample pLDDT variance
  5. Folding-upon-binding / boundary transition cues (still not MD)

This module composes existing predictors into one proteome-exportable layer.
Full MD ensembles are explicitly out of scope (see docs/IDR_BIOLOGY_LAYER.md).
"""

from __future__ import annotations

import json
import os
import re
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

LAYER_VERSION = "1.1.0"

# Cheap sequence cues that often co-occur with IDR biology (AF-blind)
_MOTIF_PATTERNS: tuple[tuple[str, str], ...] = (
    ("RGG", r"RG{1,3}"),
    ("FG_repeat", r"(?:[FG]G){2,}"),
    ("polyQ", r"Q{5,}"),
    ("polyP", r"P{5,}"),
    ("polyS", r"S{5,}"),
    ("polyE", r"E{5,}"),
    ("polyK", r"K{5,}"),
    ("LxxLL", r"L..LL"),
    ("PxxP", r"P..P"),
    ("NxS_T", r"N.[ST]"),  # N-glycosylation motif (PTM cue)
)

_AA_SETS = {
    "charged": set("DEKRHdekrh"),
    "acidic": set("DEde"),
    "basic": set("KRHkrh"),
    "aromatic": set("FWYfwy"),
    "polar": set("STNQstnq"),
    "hydrophobic": set("AILMFVWilmfvw"),
    "disorder_prone": set("AGSTPQagstpq"),
}


def _as_f32(x: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32).ravel()
    if len(arr) < n:
        out = np.full(n, np.nan, dtype=np.float32)
        out[: len(arr)] = arr
        return out
    return arr[:n]


def compute_idr_sequence_cues(sequence: str, start: int, end: int) -> dict:
    """
    Composition + short linear motif cues for one IDR segment (0-based half-open).

    These are interpretation aids / triage features — they do not change model
    logits. Matching a motif with a high role score strengthens the call.
    """
    seg = (sequence[start:end] or "").upper()
    L = max(len(seg), 1)
    counts = {k: 0 for k in _AA_SETS}
    for aa in seg:
        for name, pool in _AA_SETS.items():
            if aa in pool:
                counts[name] += 1
    frac = {f"frac_{k}": round(v / L, 4) for k, v in counts.items()}
    # Net charge approximation (K/R/H=+1, D/E=-1)
    net_charge = (counts["basic"] - counts["acidic"]) / L

    motifs: list[dict] = []
    for name, pat in _MOTIF_PATTERNS:
        for m in re.finditer(pat, seg, flags=re.IGNORECASE):
            motifs.append({
                "motif": name,
                "start": start + m.start() + 1,  # 1-based absolute
                "end": start + m.end(),
                "match": m.group(0),
            })
            if len(motifs) >= 12:
                break
        if len(motifs) >= 12:
            break

    # Role-biased cue tags (heuristic, reviewable)
    cue_tags: list[str] = []
    if frac["frac_aromatic"] >= 0.08 and frac["frac_charged"] >= 0.2:
        cue_tags.append("condensate_prone_composition")
    if any(m["motif"] in ("RGG", "FG_repeat", "polyQ") for m in motifs):
        cue_tags.append("condensate_motif")
    if any(m["motif"] in ("LxxLL", "PxxP") for m in motifs) or frac["frac_hydrophobic"] >= 0.35:
        cue_tags.append("binding_motif_or_hydrophobic_patch")
    if any(m["motif"] == "NxS_T" for m in motifs) or frac["frac_polar"] >= 0.35:
        cue_tags.append("ptm_prone_composition")
    if frac["frac_hydrophobic"] >= 0.4 and frac["frac_charged"] < 0.15:
        cue_tags.append("lipid_prone_hydrophobic")

    return {
        "length": end - start,
        "composition": {**frac, "net_charge_density": round(float(net_charge), 4)},
        "motifs": motifs,
        "cue_tags": cue_tags,
    }


def detect_boundary_transition_regions(
    disorder_probs: np.ndarray,
    *,
    disorder_threshold: float = 0.5,
    boundary_width: int = 5,
    min_len: int = 3,
    transition_mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Folding-upon-binding style zones (Phase C preview — not MD).

    Uses predicted disorder edges and optional DisProt ``transition_mask``.
    """
    dis = np.asarray(disorder_probs, dtype=np.float32).ravel()
    n = len(dis)
    hard = (dis >= disorder_threshold).astype(np.int8)
    # Mark residues within boundary_width of an order↔disorder edge
    edge = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        if hard[i] != hard[i + 1]:
            lo = max(0, i - boundary_width + 1)
            hi = min(n, i + 1 + boundary_width)
            edge[lo:hi] = True
    # Interface band around order↔disorder edges (folding-upon-binding cue)
    interface = edge
    # Softer: also emphasize mid-probability residues on the edge band
    borderline = interface & (dis >= disorder_threshold - 0.25) & (dis <= disorder_threshold + 0.3)
    use_mask = borderline if int(borderline.sum()) >= min_len else interface
    pred_regions = [
        (s, e) for s, e in intervals_from_binary(use_mask.astype(np.int8), min_len=min_len)
    ][:30]

    annotated_regions: list[tuple[int, int]] = []
    n_ann = 0
    if transition_mask is not None:
        tm = np.asarray(transition_mask, dtype=bool).ravel()
        if len(tm) >= n:
            tm = tm[:n]
        else:
            pad = np.zeros(n, dtype=bool)
            pad[: len(tm)] = tm
            tm = pad
        n_ann = int(tm.sum())
        annotated_regions = [
            (s, e) for s, e in intervals_from_binary(tm.astype(np.int8), min_len=min_len)
        ][:30]

    return {
        "method": "disorder_boundary_plus_optional_disprot_transition",
        "note": (
            "Conditional-disorder / folding-upon-binding *cue* — "
            "not a simulated bound-state ensemble"
        ),
        "predicted_boundary_regions": pred_regions,
        "annotated_transition_regions": annotated_regions,
        "n_predicted_boundary_residues": int(use_mask.sum()),
        "n_annotated_transition_residues": n_ann,
    }


def score_protein_triage(record: dict) -> dict:
    """
    Proteome ranking score — higher = investigate sooner.

    Pure function of already-computed layer fields (no model recompute).
    """
    roles = int(record.get("n_role_assignments", 0))
    halluc = (record.get("hallucination") or {}).get("n_hallucinated", 0) or 0
    flex = (record.get("ensemble_proxy") or {}).get("n_high_variance_idr_regions", 0) or 0
    boundary = (record.get("conditional_disorder") or {}).get(
        "n_predicted_boundary_residues", 0,
    ) or 0
    cues = sum(
        len(s.get("sequence_cues", {}).get("cue_tags", []))
        for s in record.get("idr_segments", [])
    )
    intersections = len(record.get("role_hallucination_intersections", []) or [])

    score = (
        3.0 * intersections
        + 1.5 * roles
        + 0.05 * float(halluc)
        + 1.0 * flex
        + 0.02 * float(boundary)
        + 0.5 * cues
        + 2.0 * float(record.get("disorder_fraction", 0.0))
    )
    reasons: list[str] = []
    if intersections:
        reasons.append("structure_overconfident_on_functional_idr")
    if roles:
        reasons.append("has_idr_role_calls")
    if halluc:
        reasons.append("has_hallucination_flags")
    if flex:
        reasons.append("has_ensemble_proxy_regions")
    if cues:
        reasons.append("has_sequence_cues")
    return {
        "score": round(float(score), 4),
        "reasons": reasons,
    }


def _role_hallucination_intersections(
    segments: list[dict],
    halluc_regions: list,
) -> list[dict]:
    """IDR segments that both carry role calls and overlap hallucination intervals."""
    if not halluc_regions or not segments:
        return []
    out: list[dict] = []
    for seg in segments:
        if not seg.get("predicted_roles"):
            continue
        s0, e0 = seg["start"] - 1, seg["end"]
        for hs, he in halluc_regions:
            overlap = max(0, min(e0, he) - max(s0, hs))
            if overlap >= 3:
                out.append({
                    "idr_start": seg["start"],
                    "idr_end": seg["end"],
                    "hallucination_region": [hs, he],
                    "overlap_residues": int(overlap),
                    "roles": [r["group"] for r in seg["predicted_roles"][:3]],
                    "action": "distrust_structure_prefer_disordernet_roles",
                })
                break
    return out[:20]


def build_idr_segment_records(
    disorder_probs: np.ndarray,
    *,
    sequence: str = "",
    function_probs: Optional[np.ndarray] = None,
    disorder_threshold: float = 0.5,
    function_threshold: float = 0.5,
    min_region_len: int = 5,
    role_names: tuple[str, ...] = IDR_ROLE_GROUPS,
    include_sequence_cues: bool = True,
) -> list[dict]:
    """IDR segments with optional multi-label role calls + sequence cues."""
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
        rec: dict = {
            "start": start + 1,  # 1-based inclusive
            "end": end,
            "length": end - start,
            "mean_disorder_prob": float(dis[start:end].mean()),
            "predicted_roles": roles,
        }
        if include_sequence_cues and sequence:
            cues = compute_idr_sequence_cues(sequence, start, end)
            rec["sequence_cues"] = cues
            tags = set(cues.get("cue_tags", []))
            for role in roles:
                support: list[str] = []
                g = role["group"]
                if g == "condensate / assembly" and (
                    "condensate_prone_composition" in tags or "condensate_motif" in tags
                ):
                    support.append("sequence_cue_agrees")
                if g in ("protein binding", "nucleic acid binding") and (
                    "binding_motif_or_hydrophobic_patch" in tags
                ):
                    support.append("sequence_cue_agrees")
                if g == "post-translational regulation" and "ptm_prone_composition" in tags:
                    support.append("sequence_cue_agrees")
                if g == "lipid / small molecule binding" and "lipid_prone_hydrophobic" in tags:
                    support.append("sequence_cue_agrees")
                if support:
                    role["evidence"] = support
        records.append(rec)
    return records


def build_protein_idr_layer(
    *,
    protein_id: str,
    sequence: str,
    disorder_probs: np.ndarray,
    plddt: Optional[np.ndarray] = None,
    function_probs: Optional[np.ndarray] = None,
    boltz_plddt_std: Optional[np.ndarray] = None,
    transition_mask: Optional[np.ndarray] = None,
    uniprot_acc: str = "",
    disorder_threshold: float = 0.5,
    function_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    structure_source: str = "boltz2",
    include_sequence_cues: bool = True,
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
        sequence=sequence,
        function_probs=fn,
        disorder_threshold=disorder_threshold,
        function_threshold=function_threshold,
        include_sequence_cues=include_sequence_cues,
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
        finite = var[np.isfinite(var)]
        if finite.size:
            thr = float(np.percentile(finite, 75))
            high_var = (var >= thr) & np.isfinite(var)
            flex = (dis >= disorder_threshold) & high_var
            flexible_proxy_regions = [
                (s, e) for s, e in intervals_from_binary(flex.astype(np.int8), min_len=3)
            ][:20]

    conditional = detect_boundary_transition_regions(
        dis,
        disorder_threshold=disorder_threshold,
        transition_mask=transition_mask,
    )

    halluc_regions = (halluc or {}).get("hallucination_regions", []) if halluc else []
    intersections = _role_hallucination_intersections(segments, halluc_regions)

    n_role_calls = sum(len(s["predicted_roles"]) for s in segments)
    record = {
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
            "regions": halluc_regions,
            "recommendation": (halluc or {}).get("recommendation"),
        } if halluc else None,
        "ensemble_proxy": {
            "method": "boltz_multisample_plddt_std",
            "note": "Cheap flexibility cue — not a physical conformational ensemble",
            "n_high_variance_idr_regions": len(flexible_proxy_regions),
            "regions": flexible_proxy_regions,
            "mean_std": float(np.nanmean(var)) if var is not None else None,
        } if var is not None else None,
        "conditional_disorder": conditional,
        "role_hallucination_intersections": intersections,
        "actions": _recommended_actions(
            segments, halluc, flexible_proxy_regions, conditional, intersections,
        ),
    }
    record["triage"] = score_protein_triage(record)
    return record


def _recommended_actions(
    segments: list[dict],
    halluc: Optional[dict],
    flex_regions: list,
    conditional: Optional[dict] = None,
    intersections: Optional[list] = None,
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
        if "post-translational regulation" in roles:
            actions.append("prioritize_ptm_mapping")
    if any(
        "sequence_cue_agrees" in (r.get("evidence") or [])
        for s in segments for r in s.get("predicted_roles", [])
    ):
        actions.append("sequence_cues_corroborate_role_calls")
    if intersections:
        actions.append("critical_review_structure_on_functional_idrs")
    if halluc and halluc.get("metrics", {}).get("n_hallucinated", 0) > 0:
        actions.append("distrust_structure_model_in_hallucination_regions")
    if flex_regions:
        actions.append("treat_high_variance_idr_as_ensemble_proxy")
    if conditional and (
        conditional.get("n_predicted_boundary_residues", 0) > 0
        or conditional.get("annotated_transition_regions")
    ):
        actions.append("consider_folding_upon_binding_at_boundaries")
    return actions


def build_idr_layer_package(
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
    include_sequence_cues: bool = True,
) -> dict:
    """
    Single-pass package: full records + summary report + triage ranking.

    Avoids building each protein twice (summary + JSONL).
    """
    plddt_by_id = plddt_by_id or {}
    function_probs_by_id = function_probs_by_id or {}
    boltz_std_by_id = boltz_std_by_id or {}

    records: list[dict] = []
    for p in proteins:
        pid = p["id"]
        if pid not in disorder_probs_by_id:
            continue
        tmask = p.get("transition_mask")
        records.append(build_protein_idr_layer(
            protein_id=pid,
            sequence=p["sequence"],
            disorder_probs=disorder_probs_by_id[pid],
            plddt=plddt_by_id.get(pid),
            function_probs=function_probs_by_id.get(pid),
            boltz_plddt_std=boltz_std_by_id.get(pid),
            transition_mask=np.asarray(tmask, dtype=np.float32) if tmask is not None else None,
            uniprot_acc=p.get("uniprot_acc", ""),
            disorder_threshold=disorder_threshold,
            function_threshold=function_threshold,
            high_plddt_threshold=high_plddt_threshold,
            structure_source=structure_source,
            include_sequence_cues=include_sequence_cues,
        ))

    records.sort(key=lambda r: -float((r.get("triage") or {}).get("score", 0.0)))

    n_halluc = sum((r.get("hallucination") or {}).get("n_hallucinated", 0) for r in records)
    n_roles = sum(r.get("n_role_assignments", 0) for r in records)
    n_condensate = sum(
        1 for r in records
        for s in r["idr_segments"]
        for role in s["predicted_roles"]
        if role["group"] == "condensate / assembly"
    )
    n_intersections = sum(len(r.get("role_hallucination_intersections") or []) for r in records)
    n_cue_agree = sum(
        1 for r in records
        for s in r["idr_segments"]
        for role in s.get("predicted_roles", [])
        if "sequence_cue_agrees" in (role.get("evidence") or [])
    )

    report = {
        "layer_version": LAYER_VERSION,
        "thesis": (
            "Post-structure IDR biology layer: disorder map + functional roles + "
            "sequence cues + structure-hallucination flags + conditional-disorder "
            "boundary cues (+ optional Boltz variance proxy)."
        ),
        "non_goals": [
            "full_md_conformational_ensembles",
            "alphafold_replacement",
        ],
        "n_proteins": len(records),
        "total_hallucinated_residues": n_halluc,
        "total_role_assignments": n_roles,
        "n_proteins_with_condensate_call": n_condensate,
        "n_role_hallucination_intersections": n_intersections,
        "n_roles_with_sequence_cue_support": n_cue_agree,
        "mean_disorder_fraction": (
            float(np.mean([r["disorder_fraction"] for r in records])) if records else 0.0
        ),
        "top_priority_proteins": [
            {
                "protein_id": r["protein_id"],
                "uniprot_acc": r.get("uniprot_acc"),
                "triage_score": (r.get("triage") or {}).get("score"),
                "reasons": (r.get("triage") or {}).get("reasons"),
                "n_role_assignments": r.get("n_role_assignments"),
                "n_hallucinated": (r.get("hallucination") or {}).get("n_hallucinated", 0),
            }
            for r in records[: min(20, len(records))]
        ],
        "proteins": records[:max_proteins_in_summary],
        "note": (
            f"Summary includes first {max_proteins_in_summary} proteins ranked by triage; "
            "use JSONL export for full proteome."
        ),
    }
    return {"report": report, "records": records}


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
    include_sequence_cues: bool = True,
) -> dict:
    """Proteome-scale IDR biology layer + summary stats (builds records once)."""
    package = build_idr_layer_package(
        proteins,
        disorder_probs_by_id,
        plddt_by_id=plddt_by_id,
        function_probs_by_id=function_probs_by_id,
        boltz_std_by_id=boltz_std_by_id,
        disorder_threshold=disorder_threshold,
        function_threshold=function_threshold,
        high_plddt_threshold=high_plddt_threshold,
        structure_source=structure_source,
        max_proteins_in_summary=max_proteins_in_summary,
        include_sequence_cues=include_sequence_cues,
    )
    return package["report"]


def export_idr_layer_jsonl(records: list[dict], path: str) -> str:
    """Write one protein layer record per line (proteome-scale)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def export_idr_layer_bed(
    records: list[dict],
    path: str,
    *,
    track_name: str = "DisorderNet_IDR",
) -> str:
    """
    BED6 of IDR segments (0-based half-open) for IGV / genome browsers.

    Uses protein_id as chrom name (proteome/browser convenience, not genomic).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f'track name="{track_name}" description="DisorderNet IDR segments"\n')
        for rec in records:
            chrom = rec.get("uniprot_acc") or rec["protein_id"]
            for seg in rec.get("idr_segments", []):
                start0 = int(seg["start"]) - 1
                end = int(seg["end"])
                roles = ",".join(
                    r["group"].replace(" ", "_") for r in seg.get("predicted_roles", [])[:2]
                )
                name = roles or "IDR"
                score = min(1000, int(round(1000 * float(seg.get("mean_disorder_prob", 0.5)))))
                f.write(f"{chrom}\t{start0}\t{end}\t{name}\t{score}\t.\n")
    return path


def export_idr_triage_tsv(records: list[dict], path: str) -> str:
    """Compact proteome triage table (sorted by priority)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "protein_id\tuniprot_acc\ttriage_score\tdisorder_fraction\t"
            "n_idr_segments\tn_role_assignments\tn_hallucinated\t"
            "n_intersections\treasons\n"
        )
        for rec in records:
            tri = rec.get("triage") or {}
            f.write(
                f"{rec['protein_id']}\t{rec.get('uniprot_acc') or ''}\t"
                f"{tri.get('score', 0)}\t{rec.get('disorder_fraction', 0):.4f}\t"
                f"{rec.get('n_idr_segments', 0)}\t{rec.get('n_role_assignments', 0)}\t"
                f"{(rec.get('hallucination') or {}).get('n_hallucinated', 0)}\t"
                f"{len(rec.get('role_hallucination_intersections') or [])}\t"
                f"{','.join(tri.get('reasons') or [])}\n"
            )
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
    print(f"  role∩hallucination={report.get('n_role_hallucination_intersections', 0)}")
    print(f"  roles with sequence-cue support={report.get('n_roles_with_sequence_cue_support', 0)}")
    top = report.get("top_priority_proteins") or []
    if top:
        print("  top priority:")
        for row in top[:5]:
            print(
                f"    {row.get('protein_id')}  score={row.get('triage_score')}  "
                f"roles={row.get('n_role_assignments')}  "
                f"halluc={row.get('n_hallucinated')}"
            )
    print(f"  thesis: {report.get('thesis', '')[:90]}…")
