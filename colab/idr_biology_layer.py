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

LAYER_VERSION = "1.6.0"

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
    ("NLS_monopartite", r"K(?:K|R)(?:K|R).{0,2}(?:K|R)"),
    ("NES_leucine", r"L.{2,3}L.{2,3}L.{1,2}L"),
    ("KEN_box", r"KEN"),
    ("D_box", r"R..L"),
    ("SH3_classI", r"R.LP.P"),
    ("WW_PPxY", r"PP.Y"),
    ("TRAF", r"P.Q.T"),
    ("PDZ_Cterm", r"[ST].[VIL]$"),
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
    Composition + short linear motif + biophysics cues for one IDR segment
    (0-based half-open).

    These are interpretation aids / triage features — they do not change model
    logits. Matching a motif with a high role score strengthens the call.
    """
    from colab.idr_layer_biophysics import compute_idr_biophysics_cues

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
            if len(motifs) >= 16:
                break
        if len(motifs) >= 16:
            break

    # Role-biased cue tags (heuristic, reviewable)
    cue_tags: list[str] = []
    if frac["frac_aromatic"] >= 0.08 and frac["frac_charged"] >= 0.2:
        cue_tags.append("condensate_prone_composition")
    if any(m["motif"] in ("RGG", "FG_repeat", "polyQ") for m in motifs):
        cue_tags.append("condensate_motif")
    if any(m["motif"] in ("LxxLL", "PxxP", "SH3_classI", "WW_PPxY", "TRAF") for m in motifs) or (
        frac["frac_hydrophobic"] >= 0.35
    ):
        cue_tags.append("binding_motif_or_hydrophobic_patch")
    if any(m["motif"] in ("NxS_T", "KEN_box", "D_box") for m in motifs) or frac["frac_polar"] >= 0.35:
        cue_tags.append("ptm_prone_composition")
    if frac["frac_hydrophobic"] >= 0.4 and frac["frac_charged"] < 0.15:
        cue_tags.append("lipid_prone_hydrophobic")
    if any(m["motif"] in ("NLS_monopartite", "NES_leucine") for m in motifs):
        cue_tags.append("nuclear_transport_motif")
    if any(m["motif"] == "PDZ_Cterm" for m in motifs):
        cue_tags.append("pdz_ligand_cterm")

    biophysics = compute_idr_biophysics_cues(sequence, start, end)
    for tag in biophysics.get("cue_tags") or []:
        if tag not in cue_tags:
            cue_tags.append(tag)

    return {
        "length": end - start,
        "composition": {**frac, "net_charge_density": round(float(net_charge), 4)},
        "motifs": motifs,
        "biophysics": biophysics,
        "cue_tags": cue_tags,
    }


def annotate_role_call_confidence(roles: list[dict]) -> list[dict]:
    """
    Attach per-role confidence / uncertainty without changing decision thresholds.

    confidence blends mean_prob with evidence corroboration; uncertainty is the
    max−mean gap (peaky segment calls).
    """
    if not roles:
        return roles
    n_roles = len(roles)
    for role in roles:
        mean_p = float(role.get("mean_prob", 0.0))
        max_p = float(role.get("max_prob", mean_p))
        n_ev = len(role.get("evidence") or [])
        cond = role.get("conditioned_prob")
        base = float(cond) if cond is not None else mean_p
        confidence = min(1.0, 0.65 * base + 0.2 * max_p + 0.1 * min(n_ev, 3) / 3 + 0.05)
        role["confidence"] = round(confidence, 4)
        role["uncertainty"] = round(max(0.0, max_p - mean_p), 4)
        role["n_evidence"] = n_ev
    if n_roles >= 3:
        top_means = sorted((float(r.get("mean_prob", 0.0)) for r in roles), reverse=True)
        if top_means[2] >= (top_means[0] - 0.12):
            for role in roles:
                ev = list(role.get("evidence") or [])
                if "multi_role_conflict" not in ev:
                    ev.append("multi_role_conflict")
                role["evidence"] = ev
                role["multi_role_conflict"] = True
    return roles


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
    # Vectorized edge find: residue indices where hard label flips
    edge = np.zeros(n, dtype=bool)
    if n > 1:
        flips = np.flatnonzero(hard[1:] != hard[:-1])
        for i in flips:
            lo = max(0, int(i) - boundary_width + 1)
            hi = min(n, int(i) + 1 + boundary_width)
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


def compute_quality_flags(record: dict) -> dict:
    """
    Deterministic QA / quarantine flags for a protein layer record.

    Does not change predictions — surfaces when the export needs human review
    or when a run should be quarantined from blind proteome stats.
    """
    flags: list[str] = []
    n = int(record.get("length") or 0)
    dis_frac = float(record.get("disorder_fraction") or 0.0)
    halluc = record.get("hallucination")
    n_halluc = int((halluc or {}).get("n_hallucinated") or 0)
    has_structure = halluc is not None
    if not has_structure:
        flags.append("missing_structure_plddt")
    if n > 0 and n < 30:
        flags.append("short_chain")
    if dis_frac >= 0.95 and n >= 30:
        flags.append("extreme_disorder")
    if dis_frac <= 0.01 and int(record.get("n_idr_segments") or 0) == 0:
        flags.append("fully_ordered")
    if n > 0 and n_halluc / n >= 0.25:
        flags.append("high_hallucination_fraction")
    if int(record.get("n_role_assignments") or 0) == 0 and dis_frac >= 0.3:
        flags.append("idr_rich_no_role_calls")
    if len(record.get("role_hallucination_intersections") or []) >= 2:
        flags.append("multiple_role_structure_conflicts")
    if any(s.get("multi_role_conflict") for s in record.get("idr_segments") or []):
        flags.append("multi_role_conflict")

    quarantine = any(
        f in flags
        for f in (
            "high_hallucination_fraction",
            "multiple_role_structure_conflicts",
            "extreme_disorder",
        )
    )
    severity = "quarantine" if quarantine else ("review" if flags else "ok")
    return {
        "flags": flags,
        "severity": severity,
        "quarantine": quarantine,
        "n_flags": len(flags),
        "has_structure_plddt": has_structure,
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
    quality = record.get("quality") or {}
    quarantine_boost = 2.0 if quality.get("quarantine") else 0.0
    mean_conf = 0.0
    n_conf = 0
    for s in record.get("idr_segments") or []:
        for role in s.get("predicted_roles") or []:
            if "confidence" in role:
                mean_conf += float(role["confidence"])
                n_conf += 1
    if n_conf:
        mean_conf /= n_conf

    score = (
        3.0 * intersections
        + 1.5 * roles
        + 0.05 * float(halluc)
        + 1.0 * flex
        + 0.02 * float(boundary)
        + 0.5 * cues
        + 2.0 * float(record.get("disorder_fraction", 0.0))
        + quarantine_boost
        + 0.5 * mean_conf
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
    if quality.get("quarantine"):
        reasons.append("quality_quarantine")
    if any(s.get("multi_role_conflict") for s in record.get("idr_segments") or []):
        reasons.append("multi_role_conflict")
    return {
        "score": round(float(score), 4),
        "reasons": reasons,
        "mean_role_confidence": round(mean_conf, 4) if n_conf else None,
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
    partner_sequences: Optional[list[str]] = None,
    ligands: Optional[list] = None,
) -> list[dict]:
    """IDR segments with optional multi-label role calls + sequence/partner/ligand cues."""
    from colab.idr_layer_io import ligand_binding_support, partner_binding_support

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
                    "condensate_prone_composition" in tags
                    or "condensate_motif" in tags
                    or "aromatic_charged_sticker_spacer" in tags
                    or "blocky_charge_patterning" in tags
                    or "low_complexity_idr" in tags
                ):
                    support.append("sequence_cue_agrees")
                if g in ("protein binding", "nucleic acid binding") and (
                    "binding_motif_or_hydrophobic_patch" in tags
                    or "nuclear_transport_motif" in tags
                    or "pdz_ligand_cterm" in tags
                    or "mixed_charge_compact_electrostatics" in tags
                ):
                    support.append("sequence_cue_agrees")
                if g == "post-translational regulation" and "ptm_prone_composition" in tags:
                    support.append("sequence_cue_agrees")
                if g == "lipid / small molecule binding" and "lipid_prone_hydrophobic" in tags:
                    support.append("sequence_cue_agrees")
                bio_tags = {
                    "blocky_charge_patterning",
                    "mixed_charge_compact_electrostatics",
                    "segregated_charge_stretching",
                    "aromatic_charged_sticker_spacer",
                    "low_complexity_idr",
                    "strongly_signed_polyelectrolyte",
                }
                if support and (tags & bio_tags):
                    support.append("biophysics_cue_agrees")
                if support:
                    role["evidence"] = support

        if partner_sequences and sequence:
            pb = partner_binding_support(sequence[start:end], partner_sequences)
            if pb["support"] > 0:
                rec["partner_context"] = pb
                for role in roles:
                    if role["group"] in ("protein binding", "nucleic acid binding"):
                        ev = list(role.get("evidence") or [])
                        if pb["support"] >= 0.4:
                            ev.append("partner_context_supports_binding")
                        role["conditioned_prob"] = round(
                            min(1.0, float(role.get("conditioned_prob", role["mean_prob"]))
                                + 0.15 * float(pb["support"])),
                            4,
                        )
                        role["evidence"] = ev

        if ligands and sequence:
            lb = ligand_binding_support(sequence[start:end], ligands)
            if lb["support"] > 0:
                rec["ligand_context"] = lb
                target_set = set(lb.get("target_roles") or [])
                for role in roles:
                    if role["group"] in target_set or (
                        role["group"] == "lipid / small molecule binding"
                        and lb["support"] >= 0.3
                    ) or (
                        role["group"] == "nucleic acid binding"
                        and "nucleic acid binding" in target_set
                    ):
                        ev = list(role.get("evidence") or [])
                        if lb["support"] >= 0.4:
                            ev.append("ligand_context_supports_role")
                        role["conditioned_prob"] = round(
                            min(1.0, float(role.get("conditioned_prob", role["mean_prob"]))
                                + 0.15 * float(lb["support"])),
                            4,
                        )
                        role["evidence"] = ev

        rec["predicted_roles"] = annotate_role_call_confidence(roles)
        if any(r.get("multi_role_conflict") for r in roles):
            rec["multi_role_conflict"] = True
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
    partner_sequences: Optional[list[str]] = None,
    ligands: Optional[list] = None,
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
        partner_sequences=partner_sequences,
        ligands=ligands,
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
            "definition": (halluc or {}).get("definition", "proxy_distrust"),
            "n_hallucinated": (halluc or {}).get("metrics", {}).get("n_hallucinated", 0),
            "n_rescued": (halluc or {}).get("metrics", {}).get("n_rescued"),
            "rescue_rate": (halluc or {}).get("metrics", {}).get("rescue_rate"),
            "rescue_rate_valid": (halluc or {}).get("metrics", {}).get("rescue_rate_valid", False),
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
    record["quality"] = compute_quality_flags(record)
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
    if any(
        "partner_context_supports_binding" in (r.get("evidence") or [])
        for s in segments for r in s.get("predicted_roles", [])
    ):
        actions.append("partner_context_supports_binding_roles")
    if any(
        "ligand_context_supports_role" in (r.get("evidence") or [])
        for s in segments for r in s.get("predicted_roles", [])
    ):
        actions.append("ligand_context_supports_role_calls")
    if any(
        "biophysics_cue_agrees" in (r.get("evidence") or [])
        for s in segments for r in s.get("predicted_roles", [])
    ):
        actions.append("biophysics_patterning_corroborates_roles")
    if any(s.get("multi_role_conflict") for s in segments):
        actions.append("review_multi_role_idr_segments")
    return actions


def _map_term_to_role_group(term_norm: str, role_names: tuple[str, ...] = IDR_ROLE_GROUPS) -> Optional[str]:
    """Map a DisProt term_norm string onto an IDR role group name."""
    try:
        from colab.disordernet_gpu import FUNCTIONAL_TERM_GROUPS
        groups = FUNCTIONAL_TERM_GROUPS
    except Exception:
        groups = {name: frozenset() for name in role_names}
    t = (term_norm or "").strip().lower()
    for name in role_names:
        terms = groups.get(name, frozenset())
        if t == name or t in terms:
            return name
    return None


def evaluate_role_calls_against_annotations(
    records: list[dict],
    proteins: list[dict],
    *,
    min_overlap: int = 3,
) -> dict:
    """
    Segment-level precision/recall of predicted IDR roles vs DisProt functional_regions.

    Accuracy check for the layer — does not change predictions. Requires proteins
    that still carry ``functional_regions`` (CV / DisProt path).
    """
    by_id = {p["id"]: p for p in proteins}
    tp = fp = fn = 0
    per_group = {g: {"tp": 0, "fp": 0, "fn": 0} for g in IDR_ROLE_GROUPS}
    n_with_truth = 0

    for rec in records:
        p = by_id.get(rec["protein_id"])
        if not p:
            continue
        truth_regs = p.get("functional_regions") or []
        if not truth_regs:
            continue
        n_with_truth += 1
        # Truth intervals per group (0-based half-open)
        truth: dict[str, list[tuple[int, int]]] = {g: [] for g in IDR_ROLE_GROUPS}
        for reg in truth_regs:
            g = _map_term_to_role_group(reg.get("term_norm") or reg.get("term_name") or "")
            if not g:
                continue
            s0 = max(0, int(reg["start"]) - 1)
            e0 = int(reg["end"])
            if e0 > s0:
                truth[g].append((s0, e0))

        pred_flags: dict[str, list[tuple[int, int]]] = {g: [] for g in IDR_ROLE_GROUPS}
        for seg in rec.get("idr_segments", []):
            s0, e0 = int(seg["start"]) - 1, int(seg["end"])
            for role in seg.get("predicted_roles", []):
                g = role["group"]
                if g in pred_flags:
                    pred_flags[g].append((s0, e0))

        for g in IDR_ROLE_GROUPS:
            # Precision: each predicted segment that overlaps any truth of g
            for ps, pe in pred_flags[g]:
                hit = any(max(0, min(pe, te) - max(ps, ts)) >= min_overlap for ts, te in truth[g])
                if hit:
                    tp += 1
                    per_group[g]["tp"] += 1
                else:
                    fp += 1
                    per_group[g]["fp"] += 1
            # Recall: each truth region overlapped by a pred of g
            for ts, te in truth[g]:
                hit = any(max(0, min(pe, te) - max(ps, ts)) >= min_overlap for ps, pe in pred_flags[g])
                if hit:
                    per_group[g]["fn"] += 0  # counted via tp path; track miss below
                else:
                    fn += 1
                    per_group[g]["fn"] += 1

    def _prf(t, f_p, f_n):
        prec = t / (t + f_p) if (t + f_p) else None
        rec_ = t / (t + f_n) if (t + f_n) else None
        if prec is None or rec_ is None or (prec + rec_) == 0:
            f1 = None
        else:
            f1 = 2 * prec * rec_ / (prec + rec_)
        return {
            "precision": None if prec is None else round(prec, 4),
            "recall": None if rec_ is None else round(rec_, 4),
            "f1": None if f1 is None else round(f1, 4),
            "tp": t, "fp": f_p, "fn": f_n,
        }

    return {
        "enabled": n_with_truth > 0,
        "n_proteins_with_annotations": n_with_truth,
        "micro": _prf(tp, fp, fn),
        "per_group": {g: _prf(v["tp"], v["fp"], v["fn"]) for g, v in per_group.items()},
        "note": (
            "Segment overlap vs DisProt functional_regions — evaluates role calls, "
            "not disorder AUC"
        ),
    }


def summarize_structure_distrust(records: list[dict]) -> dict:
    """
    Aggregate structure-distrust stats across the proteome.

    Distinguishes proxy distrust flags (default deployment) from labeled rescue
    when protein records carry ``hallucination.definition == labeled_independent``.
    """
    n_halluc = 0
    n_rescued = 0
    n_proteins_flagged = 0
    n_intersections = 0
    n_labeled = 0
    any_proxy = False
    for r in records:
        h = r.get("hallucination") or {}
        nh = int(h.get("n_hallucinated", 0) or 0)
        nr = h.get("n_rescued")
        definition = h.get("definition") or "proxy_distrust"
        if definition == "labeled_independent":
            n_labeled += 1
            if nr is not None:
                n_rescued += int(nr or 0)
        else:
            any_proxy = True
        n_halluc += nh
        if nh > 0:
            n_proteins_flagged += 1
        n_intersections += len(r.get("role_hallucination_intersections") or [])
    rescue_valid = n_labeled > 0 and not any_proxy
    return {
        "n_proteins_with_hallucination": n_proteins_flagged,
        "total_hallucinated_residues": n_halluc,
        "total_rescued_residues": n_rescued if rescue_valid else None,
        "overall_rescue_rate": (
            round(float(n_rescued / n_halluc), 4)
            if rescue_valid and n_halluc
            else None
        ),
        "rescue_rate_valid": rescue_valid,
        "definition": (
            "labeled_independent" if rescue_valid
            else ("mixed" if n_labeled else "proxy_distrust")
        ),
        "n_role_hallucination_intersections": n_intersections,
        "note": (
            "Prefer DisorderNet over structure confidence in flagged regions. "
            "Rescue rates only valid under labeled_independent protocol."
        ),
    }


def build_idr_layer_package(
    proteins: list[dict],
    disorder_probs_by_id: dict[str, np.ndarray],
    *,
    plddt_by_id: Optional[dict[str, np.ndarray]] = None,
    function_probs_by_id: Optional[dict[str, np.ndarray]] = None,
    boltz_std_by_id: Optional[dict[str, np.ndarray]] = None,
    partners_by_id: Optional[dict[str, list[str]]] = None,
    ligands_by_id: Optional[dict[str, list]] = None,
    disorder_threshold: float = 0.5,
    function_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    structure_source: str = "boltz2",
    max_proteins_in_summary: int = 100,
    include_sequence_cues: bool = True,
    validate_roles: bool = True,
    max_workers: int = 1,
    cache_dir: Optional[str] = None,
    cache_tag: str = "",
    skip_protein_ids: Optional[set] = None,
    prior_records: Optional[list[dict]] = None,
) -> dict:
    """
    Single-pass package: full records + summary report + triage ranking.

    ``max_workers>1`` threads protein builds (I/O-light; accuracy-identical).
    Optional ``cache_dir`` stores per-protein records keyed by content hash.
    ``skip_protein_ids`` / ``prior_records`` support resume of large proteomes.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from colab.idr_layer_ops import (
        layer_record_cache_key,
        load_cached_layer_record,
        proteome_landscape_summary,
        quality_summary,
        save_cached_layer_record,
    )

    plddt_by_id = plddt_by_id or {}
    function_probs_by_id = function_probs_by_id or {}
    boltz_std_by_id = boltz_std_by_id or {}
    partners_by_id = partners_by_id or {}
    ligands_by_id = ligands_by_id or {}
    skip = set(skip_protein_ids or set())

    work = [
        p for p in proteins
        if p["id"] in disorder_probs_by_id and p["id"] not in skip
    ]
    cache_hits = 0

    def _one(p: dict) -> tuple[dict, bool]:
        pid = p["id"]
        partners = partners_by_id.get(pid) or partners_by_id.get(p.get("uniprot_acc") or "")
        ligands = ligands_by_id.get(pid) or ligands_by_id.get(p.get("uniprot_acc") or "")
        key = None
        if cache_dir:
            key = layer_record_cache_key(
                protein_id=pid,
                sequence=p["sequence"],
                layer_version=LAYER_VERSION,
                disorder_threshold=disorder_threshold,
                function_threshold=function_threshold,
                high_plddt_threshold=high_plddt_threshold,
                has_function=pid in function_probs_by_id,
                has_plddt=pid in plddt_by_id,
                has_variance=pid in boltz_std_by_id,
                has_partners=bool(partners),
                has_ligands=bool(ligands),
                cache_tag=cache_tag,
            )
            cached = load_cached_layer_record(cache_dir, key)
            if cached is not None:
                return cached, True

        tmask = p.get("transition_mask")
        rec = build_protein_idr_layer(
            protein_id=pid,
            sequence=p["sequence"],
            disorder_probs=disorder_probs_by_id[pid],
            plddt=plddt_by_id.get(pid),
            function_probs=function_probs_by_id.get(pid),
            boltz_plddt_std=boltz_std_by_id.get(pid),
            transition_mask=np.asarray(tmask, dtype=np.float32) if tmask is not None else None,
            partner_sequences=partners,
            ligands=ligands,
            uniprot_acc=p.get("uniprot_acc", ""),
            disorder_threshold=disorder_threshold,
            function_threshold=function_threshold,
            high_plddt_threshold=high_plddt_threshold,
            structure_source=structure_source,
            include_sequence_cues=include_sequence_cues,
        )
        if cache_dir and key:
            save_cached_layer_record(cache_dir, key, rec)
        return rec, False

    records: list[dict] = []
    workers = max(1, min(int(max_workers), len(work) or 1))
    if not work:
        records = []
    elif workers == 1:
        for p in work:
            rec, hit = _one(p)
            records.append(rec)
            if hit:
                cache_hits += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_one, p): p["id"] for p in work}
            by_id = {}
            for fut in as_completed(futs):
                rec, hit = fut.result()
                by_id[rec["protein_id"]] = rec
                if hit:
                    cache_hits += 1
            records = [by_id[p["id"]] for p in work if p["id"] in by_id]

    # Merge resumed prior records (unchanged proteins) with newly built ones
    if prior_records:
        built_ids = {r["protein_id"] for r in records}
        merged = [r for r in prior_records if r.get("protein_id") not in built_ids]
        merged.extend(records)
        records = merged

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
    n_partner = sum(
        1 for r in records
        for s in r["idr_segments"]
        if s.get("partner_context")
    )
    n_ligand = sum(
        1 for r in records
        for s in r["idr_segments"]
        if s.get("ligand_context")
    )

    role_validation = (
        evaluate_role_calls_against_annotations(records, proteins)
        if validate_roles else {"enabled": False}
    )
    structure_distrust = summarize_structure_distrust(records)
    landscape = proteome_landscape_summary(records)
    quality = quality_summary(records)

    report = {
        "layer_version": LAYER_VERSION,
        "thesis": (
            "Post-structure IDR biology layer: disorder map + functional roles + "
            "sequence/partner/ligand cues + structure-hallucination flags + "
            "conditional-disorder boundary cues (+ optional Boltz variance proxy)."
        ),
        "non_goals": [
            "full_md_conformational_ensembles",
            "alphafold_replacement",
        ],
        "n_proteins": len(records),
        "n_built_this_run": len(work),
        "n_resumed": len(skip),
        "total_hallucinated_residues": n_halluc,
        "total_role_assignments": n_roles,
        "n_proteins_with_condensate_call": n_condensate,
        "n_role_hallucination_intersections": n_intersections,
        "n_roles_with_sequence_cue_support": n_cue_agree,
        "n_segments_with_partner_context": n_partner,
        "n_segments_with_ligand_context": n_ligand,
        "mean_disorder_fraction": (
            float(np.mean([r["disorder_fraction"] for r in records])) if records else 0.0
        ),
        "role_validation": role_validation,
        "structure_distrust": structure_distrust,
        "landscape": landscape,
        "quality": quality,
        "cache": {
            "enabled": bool(cache_dir),
            "dir": cache_dir,
            "hits": cache_hits,
            "n_proteins": len(work),
            "tag": cache_tag or None,
        },
        "thresholds": {
            "disorder_threshold": disorder_threshold,
            "function_threshold": function_threshold,
            "high_plddt_threshold": high_plddt_threshold,
        },
        "top_priority_proteins": [
            {
                "protein_id": r["protein_id"],
                "uniprot_acc": r.get("uniprot_acc"),
                "triage_score": (r.get("triage") or {}).get("score"),
                "reasons": (r.get("triage") or {}).get("reasons"),
                "n_role_assignments": r.get("n_role_assignments"),
                "n_hallucinated": (r.get("hallucination") or {}).get("n_hallucinated", 0),
                "quality_severity": (r.get("quality") or {}).get("severity"),
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
    partners_by_id: Optional[dict[str, list[str]]] = None,
    ligands_by_id: Optional[dict[str, list]] = None,
    disorder_threshold: float = 0.5,
    function_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    structure_source: str = "boltz2",
    max_proteins_in_summary: int = 100,
    include_sequence_cues: bool = True,
    max_workers: int = 1,
    cache_dir: Optional[str] = None,
    cache_tag: str = "",
    skip_protein_ids: Optional[set] = None,
    prior_records: Optional[list[dict]] = None,
) -> dict:
    """Proteome-scale IDR biology layer + summary stats (builds records once)."""
    package = build_idr_layer_package(
        proteins,
        disorder_probs_by_id,
        plddt_by_id=plddt_by_id,
        function_probs_by_id=function_probs_by_id,
        boltz_std_by_id=boltz_std_by_id,
        partners_by_id=partners_by_id,
        ligands_by_id=ligands_by_id,
        disorder_threshold=disorder_threshold,
        function_threshold=function_threshold,
        high_plddt_threshold=high_plddt_threshold,
        structure_source=structure_source,
        max_proteins_in_summary=max_proteins_in_summary,
        include_sequence_cues=include_sequence_cues,
        max_workers=max_workers,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        skip_protein_ids=skip_protein_ids,
        prior_records=prior_records,
    )
    return package["report"]


def export_idr_layer_jsonl(
    records: list[dict],
    path: str,
    *,
    gzip: bool = False,
    append: bool = False,
) -> str:
    """Write one protein layer record per line (proteome-scale)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if gzip and not path.endswith(".gz"):
        path = path + ".gz"
    mode = "at" if append else "wt"
    if gzip or path.endswith(".gz"):
        import gzip as _gzip
        # gzip append is byte-append; use "ab" + text wrapper via mode
        gmode = "ab" if append else "wb"
        with _gzip.open(path, gmode) as raw:
            import io
            with io.TextIOWrapper(raw, encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
    else:
        with open(path, mode, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    return path


def export_idr_role_tracks_tsv(records: list[dict], path: str) -> str:
    """
    Compact residue-level top-role track for IDR segments.

    Columns: protein_id, uniprot_acc, start, end, top_role, mean_prob, conditioned_prob
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "protein_id\tuniprot_acc\tstart\tend\ttop_role\t"
            "mean_prob\tconditioned_prob\tevidence\n"
        )
        for rec in records:
            acc = rec.get("uniprot_acc") or ""
            for seg in rec.get("idr_segments", []):
                roles = seg.get("predicted_roles") or []
                if not roles:
                    f.write(
                        f"{rec['protein_id']}\t{acc}\t{seg['start']}\t{seg['end']}\t"
                        f"\t\t\t\n"
                    )
                    continue
                top = roles[0]
                ev = ";".join(top.get("evidence") or [])
                f.write(
                    f"{rec['protein_id']}\t{acc}\t{seg['start']}\t{seg['end']}\t"
                    f"{top['group']}\t{top['mean_prob']:.4f}\t"
                    f"{top.get('conditioned_prob', '')}\t{ev}\n"
                )
    return path


def export_idr_disorder_bedgraph(
    proteins: list[dict],
    disorder_probs_by_id: dict[str, np.ndarray],
    path: str,
    *,
    track_name: str = "DisorderNet_disorder",
) -> str:
    """
    BedGraph of continuous disorder probabilities (protein_id as chrom).

    Efficient proteome viewer track alongside the discrete IDR BED.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(
            f'track type=bedGraph name="{track_name}" '
            f'description="DisorderNet disorder probability"\n'
        )
        for p in proteins:
            pid = p["id"]
            if pid not in disorder_probs_by_id:
                continue
            chrom = p.get("uniprot_acc") or pid
            probs = np.asarray(disorder_probs_by_id[pid], dtype=np.float32).ravel()
            n = min(len(p["sequence"]), len(probs))
            # RLE compress flat runs for smaller files (accuracy-identical)
            i = 0
            while i < n:
                j = i + 1
                v = float(probs[i])
                while j < n and abs(float(probs[j]) - v) < 1e-4:
                    j += 1
                f.write(f"{chrom}\t{i}\t{j}\t{v:.4f}\n")
                i = j
    return path


def export_idr_role_bedgraphs(
    proteins: list[dict],
    function_probs_by_id: dict[str, np.ndarray],
    out_dir: str,
    *,
    role_names: tuple[str, ...] = IDR_ROLE_GROUPS,
) -> dict[str, str]:
    """
    One bedGraph per IDR role group (continuous mean role probability).

    Useful for IGV / genome-browser style proteome review of role landscapes.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}
    if not function_probs_by_id:
        return paths

    for gi, name in enumerate(role_names):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
        path = os.path.join(out_dir, f"idr_role_{safe}.bedgraph")
        with open(path, "w") as f:
            f.write(
                f'track type=bedGraph name="DisorderNet_{safe}" '
                f'description="DisorderNet role: {name}"\n'
            )
            for p in proteins:
                pid = p["id"]
                if pid not in function_probs_by_id:
                    continue
                fn = np.asarray(function_probs_by_id[pid], dtype=np.float32)
                if fn.ndim == 1:
                    fn = fn.reshape(-1, 1)
                if gi >= fn.shape[1]:
                    continue
                chrom = p.get("uniprot_acc") or pid
                col = fn[:, gi]
                n = min(len(p["sequence"]), len(col))
                i = 0
                while i < n:
                    j = i + 1
                    v = float(col[i])
                    while j < n and abs(float(col[j]) - v) < 1e-4:
                        j += 1
                    f.write(f"{chrom}\t{i}\t{j}\t{v:.4f}\n")
                    i = j
        paths[name] = path
    return paths


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
            "n_intersections\tquality\treasons\n"
        )
        for rec in records:
            tri = rec.get("triage") or {}
            qual = (rec.get("quality") or {}).get("severity", "")
            f.write(
                f"{rec['protein_id']}\t{rec.get('uniprot_acc') or ''}\t"
                f"{tri.get('score', 0)}\t{rec.get('disorder_fraction', 0):.4f}\t"
                f"{rec.get('n_idr_segments', 0)}\t{rec.get('n_role_assignments', 0)}\t"
                f"{(rec.get('hallucination') or {}).get('n_hallucinated', 0)}\t"
                f"{len(rec.get('role_hallucination_intersections') or [])}\t"
                f"{qual}\t"
                f"{','.join(tri.get('reasons') or [])}\n"
            )
    return path


def save_idr_layer_report(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def export_idr_layer_bundle(
    *,
    out_dir: str,
    report: dict,
    records: list[dict],
    proteins: list[dict],
    disorder_probs_by_id: dict[str, np.ndarray],
    function_probs_by_id: Optional[dict[str, np.ndarray]] = None,
    gzip_jsonl: bool = False,
    export_caid: bool = True,
    export_html: bool = True,
    export_role_bedgraphs: bool = True,
    export_gff: bool = True,
    export_cards: bool = True,
    cards_top_n: int = 20,
    validate_schema: bool = True,
    min_triage_score: Optional[float] = None,
    quarantine_only: bool = False,
    run_args: Optional[object] = None,
) -> dict[str, str]:
    """Write the standard IDR layer artifact set; returns path map."""
    from colab.idr_layer_ops import (
        export_disorder_caid_bundle,
        export_idr_layer_gff3,
        filter_layer_records,
        validate_idr_layer_records,
        write_idr_layer_html,
        write_idr_layer_markdown,
        write_idr_run_manifest,
        write_triage_protein_cards,
    )

    os.makedirs(out_dir, exist_ok=True)
    if validate_schema:
        schema = validate_idr_layer_records(records)
        report["schema_validation"] = schema

    filtered = filter_layer_records(
        records,
        min_triage_score=min_triage_score,
        quarantine_only=quarantine_only,
    )
    report["export_filter"] = {
        "min_triage_score": min_triage_score,
        "quarantine_only": quarantine_only,
        "n_filtered_for_cards": len(filtered),
        "n_total": len(records),
    }

    paths = {
        "report": save_idr_layer_report(
            report, os.path.join(out_dir, "idr_biology_layer_report.json"),
        ),
        "markdown": write_idr_layer_markdown(
            report, os.path.join(out_dir, "idr_biology_layer_report.md"),
        ),
        "jsonl": export_idr_layer_jsonl(
            records, os.path.join(out_dir, "idr_biology_layer.jsonl"), gzip=gzip_jsonl,
        ),
        "triage": export_idr_triage_tsv(
            records, os.path.join(out_dir, "idr_biology_layer_triage.tsv"),
        ),
        "bed": export_idr_layer_bed(
            records, os.path.join(out_dir, "idr_biology_layer.bed"),
        ),
        "bedgraph": export_idr_disorder_bedgraph(
            proteins, disorder_probs_by_id,
            os.path.join(out_dir, "idr_biology_layer_disorder.bedgraph"),
        ),
        "roles": export_idr_role_tracks_tsv(
            records, os.path.join(out_dir, "idr_biology_layer_roles.tsv"),
        ),
        "manifest": write_idr_run_manifest(
            os.path.join(out_dir, "idr_biology_layer_manifest.json"),
            layer_version=str(report.get("layer_version") or LAYER_VERSION),
            args_ns=run_args,
            extra={"n_proteins": len(records), "thresholds": report.get("thresholds")},
        ),
    }
    if export_gff:
        paths["gff3"] = export_idr_layer_gff3(
            records, os.path.join(out_dir, "idr_biology_layer.gff3"),
        )
    if export_cards:
        cards_dir = os.path.join(out_dir, "idr_biology_layer_cards")
        card_paths = write_triage_protein_cards(
            filtered or records, cards_dir, top_n=cards_top_n,
        )
        paths["cards_dir"] = cards_dir
        paths["cards_n"] = str(len(card_paths))
    if export_html:
        paths["html"] = write_idr_layer_html(
            report, os.path.join(out_dir, "idr_biology_layer_report.html"),
        )
    if export_role_bedgraphs and function_probs_by_id:
        role_dir = os.path.join(out_dir, "idr_biology_layer_role_bedgraphs")
        role_paths = export_idr_role_bedgraphs(
            proteins, function_probs_by_id, role_dir,
        )
        paths["role_bedgraphs"] = role_dir
        paths["role_bedgraph_n"] = str(len(role_paths))
    if export_caid:
        caid_dir = os.path.join(out_dir, "idr_biology_layer_caid")
        caid_paths = export_disorder_caid_bundle(
            proteins, disorder_probs_by_id, caid_dir,
            threshold=float((report.get("thresholds") or {}).get("disorder_threshold", 0.5)),
        )
        paths["caid_dir"] = caid_dir
        paths["caid_n_files"] = str(len(caid_paths))
    return paths


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
    print(f"  partner-context segments={report.get('n_segments_with_partner_context', 0)}")
    print(f"  ligand-context segments={report.get('n_segments_with_ligand_context', 0)}")
    land = report.get("landscape") or {}
    if land and not land.get("insufficient_data"):
        print(
            f"  landscape: idr_rich={land.get('disorder_fraction_bins', {}).get('idr_rich_0.4_0.7', 0)}  "
            f"mostly_disordered={land.get('disorder_fraction_bins', {}).get('mostly_disordered_>=0.7', 0)}"
        )
    sd = report.get("structure_distrust") or {}
    if sd:
        print(
            f"  structure distrust: halluc_prot={sd.get('n_proteins_with_hallucination', 0)}  "
            f"definition={sd.get('definition')}  "
            f"rescue_rate={sd.get('overall_rescue_rate')}"
        )
    qual = report.get("quality") or {}
    if qual:
        print(
            f"  quality: quarantine={qual.get('n_quarantine', 0)}  "
            f"review={qual.get('n_review', 0)}  ok={qual.get('n_ok', 0)}"
        )
    cal = report.get("function_calibration") or {}
    if cal.get("enabled"):
        print(
            f"  function calibration: mean_T={cal.get('mean_temperature')}  "
            f"groups={cal.get('n_groups')}"
        )
    tune = report.get("function_threshold_tuning") or {}
    if tune.get("enabled"):
        print(
            f"  OOF-tuned function threshold={tune.get('threshold')}  "
            f"score={tune.get('best_score')}"
        )
    cache = report.get("cache") or {}
    if cache.get("enabled"):
        print(f"  cache hits={cache.get('hits')} / {cache.get('n_proteins')}")
    if report.get("n_resumed"):
        print(f"  resumed prior proteins={report.get('n_resumed')}  built={report.get('n_built_this_run')}")
    rv = report.get("role_validation") or {}
    if rv.get("enabled"):
        micro = rv.get("micro") or {}
        print(
            f"  role validation (vs DisProt): "
            f"P={micro.get('precision')}  R={micro.get('recall')}  "
            f"F1={micro.get('f1')}  n_prot={rv.get('n_proteins_with_annotations')}"
        )
    schema = report.get("schema_validation") or {}
    if schema:
        print(
            f"  schema: valid={schema.get('n_valid')}  "
            f"invalid={schema.get('n_invalid')}"
        )
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
