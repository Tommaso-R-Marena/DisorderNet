"""
I/O helpers for the IDR biology layer — load predictions, export tracks,
and optional partner-context maps (Phase C).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np


def load_disorder_preds_from_dir(
    pred_dir: str,
    *,
    proteins: Optional[list[dict]] = None,
) -> dict[str, np.ndarray]:
    """
    Load per-residue disorder scores from a predict-stage directory.

    Supports ``*.tsv`` (``position\\tresidue\\tscore``) and ``*.caid``
    (CAID submission: header ``>id``, then ``pos\\taa\\tscore[\\tstate]``).
    When ``proteins`` is given, keys are rematched to protein ids / uniprot.
    """
    pred_dir = str(pred_dir)
    if not os.path.isdir(pred_dir):
        return {}

    by_stem: dict[str, np.ndarray] = {}
    for name in os.listdir(pred_dir):
        path = os.path.join(pred_dir, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        if lower.endswith(".tsv"):
            scores = _read_score_tsv(path)
        elif lower.endswith(".caid"):
            scores = _read_score_caid(path)
        else:
            continue
        if scores is None or len(scores) == 0:
            continue
        stem = Path(name).stem
        by_stem[stem] = scores.astype(np.float32)

    if not proteins:
        return by_stem

    out: dict[str, np.ndarray] = {}
    for p in proteins:
        pid = p["id"]
        candidates = {
            pid,
            re.sub(r"[^\w\-.]", "_", pid),
            (p.get("uniprot_acc") or ""),
            re.sub(r"[^\w\-.]", "_", p.get("uniprot_acc") or ""),
        }
        for key in candidates:
            if key and key in by_stem:
                out[pid] = by_stem[key]
                break
            # Case-insensitive stem match
            for stem, arr in by_stem.items():
                if key and stem.lower() == key.lower():
                    out[pid] = arr
                    break
            if pid in out:
                break
    return out


def _read_score_tsv(path: str) -> Optional[np.ndarray]:
    scores: list[float] = []
    with open(path) as f:
        header = f.readline()
        if "score" not in header.lower() and "\t" in header:
            # Might be data without header
            parts = header.strip().split("\t")
            try:
                if len(parts) >= 3:
                    scores.append(float(parts[2]))
            except ValueError:
                pass
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                scores.append(float(parts[2]))
            except ValueError:
                continue
    return np.asarray(scores, dtype=np.float32) if scores else None


def _read_score_caid(path: str) -> Optional[np.ndarray]:
    scores: list[float] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                scores.append(float(parts[2]))
            except ValueError:
                continue
    return np.asarray(scores, dtype=np.float32) if scores else None


def load_function_preds_from_dir(
    pred_dir: str,
    *,
    proteins: Optional[list[dict]] = None,
    n_groups: int = 5,
) -> dict[str, np.ndarray]:
    """
    Load per-residue multi-label function probs from a directory.

    Supports:
      - ``*.npy`` — shape ``(L, G)`` or ``(L,)``
      - ``*.tsv`` — ``position\\tresidue\\tg0\\tg1\\t...`` (header optional)
      - JSON map ``{protein_id: [[...], ...]}`` if ``function_preds.json`` present
    """
    pred_dir = str(pred_dir)
    if not os.path.isdir(pred_dir):
        return {}

    by_stem: dict[str, np.ndarray] = {}
    json_path = os.path.join(pred_dir, "function_preds.json")
    if os.path.isfile(json_path):
        with open(json_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            by_stem[k] = arr

    for name in os.listdir(pred_dir):
        path = os.path.join(pred_dir, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        stem = Path(name).stem
        if lower.endswith(".npy"):
            arr = np.load(path).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            by_stem[stem] = arr
        elif lower.endswith(".tsv") and stem != "function_preds":
            arr = _read_function_tsv(path, n_groups=n_groups)
            if arr is not None:
                by_stem[stem] = arr

    if not proteins:
        return by_stem

    out: dict[str, np.ndarray] = {}
    for p in proteins:
        pid = p["id"]
        candidates = {
            pid,
            re.sub(r"[^\w\-.]", "_", pid),
            (p.get("uniprot_acc") or ""),
            re.sub(r"[^\w\-.]", "_", p.get("uniprot_acc") or ""),
        }
        for key in candidates:
            if key and key in by_stem:
                out[pid] = by_stem[key]
                break
            for stem, arr in by_stem.items():
                if key and stem.lower() == key.lower():
                    out[pid] = arr
                    break
            if pid in out:
                break
    return out


def _read_function_tsv(path: str, *, n_groups: int = 5) -> Optional[np.ndarray]:
    rows: list[list[float]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            # Skip header
            if parts[0].lower() in ("position", "pos", "residue"):
                continue
            try:
                # position, residue, then group probs
                if len(parts) >= 2 + n_groups:
                    vals = [float(x) for x in parts[2:2 + n_groups]]
                elif len(parts) >= n_groups:
                    vals = [float(x) for x in parts[:n_groups]]
                else:
                    continue
                rows.append(vals)
            except ValueError:
                continue
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float32)


def load_partner_map(path: Optional[str]) -> dict[str, list[str]]:
    """
    Load optional partner sequences for conditioned role cues.

    Formats:
      - JSON ``{protein_id: [\"PARTNERSEQ\", ...]}``
      - TSV ``protein_id\\tPARTNERSEQ`` (multiple rows per id OK)
    """
    if not path or not os.path.isfile(path):
        return {}
    if path.lower().endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
        out: dict[str, list[str]] = {}
        for k, v in raw.items():
            if isinstance(v, str):
                out[k] = [v]
            elif isinstance(v, list):
                out[k] = [str(x) for x in v if x]
        return out

    out: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            out.setdefault(parts[0], []).append(parts[1].strip())
    return out


def partner_binding_support(
    idr_sequence: str,
    partner_sequences: list[str],
) -> dict:
    """
    Cheap complementary binding cue between an IDR segment and partner seqs.

    Does not change model logits — returns evidence tags + a [0,1] support score.
    """
    if not partner_sequences or not idr_sequence:
        return {"support": 0.0, "tags": [], "n_partners": 0}

    idr = idr_sequence.upper()
    idr_charge = sum(1 for a in idr if a in "KRH") - sum(1 for a in idr if a in "DE")
    idr_hydro = sum(1 for a in idr if a in "AILMFVW") / max(len(idr), 1)
    idr_arom = sum(1 for a in idr if a in "FWY") / max(len(idr), 1)

    best = 0.0
    tags: list[str] = []
    for partner in partner_sequences:
        p = (partner or "").upper()
        if len(p) < 3:
            continue
        p_charge = sum(1 for a in p if a in "KRH") - sum(1 for a in p if a in "DE")
        p_hydro = sum(1 for a in p if a in "AILMFVW") / max(len(p), 1)
        p_arom = sum(1 for a in p if a in "FWY") / max(len(p), 1)
        score = 0.0
        local: list[str] = []
        # Opposite net charge → electrostatic attraction cue
        if idr_charge * p_charge < 0 and abs(idr_charge) >= 2 and abs(p_charge) >= 2:
            score += 0.45
            local.append("complementary_charge")
        # Both hydrophobic-rich → sticky interface cue
        if idr_hydro >= 0.3 and p_hydro >= 0.3:
            score += 0.25
            local.append("mutual_hydrophobic")
        # Aromatic contacts
        if idr_arom >= 0.05 and p_arom >= 0.05:
            score += 0.2
            local.append("aromatic_contacts")
        best = max(best, min(1.0, score))
        for t in local:
            if t not in tags:
                tags.append(t)

    return {
        "support": round(float(best), 4),
        "tags": tags,
        "n_partners": len(partner_sequences),
    }


_LIGAND_TYPE_ALIASES: dict[str, str] = {
    "lipid": "lipid",
    "fatty": "lipid",
    "membrane": "lipid",
    "phospholipid": "lipid",
    "nucleotide": "nucleic",
    "nucleic": "nucleic",
    "atp": "nucleic",
    "gtp": "nucleic",
    "dna": "nucleic",
    "rna": "nucleic",
    "metal": "metal",
    "ion": "metal",
    "zn": "metal",
    "zinc": "metal",
    "ca": "metal",
    "calcium": "metal",
    "mg": "metal",
    "small_molecule": "small",
    "small": "small",
    "drug": "small",
    "ligand": "small",
    "metabolite": "small",
}

_LIGAND_TO_ROLE: dict[str, str] = {
    "lipid": "lipid / small molecule binding",
    "small": "lipid / small molecule binding",
    "metal": "lipid / small molecule binding",
    "nucleic": "nucleic acid binding",
}


def normalize_ligand_entry(entry) -> dict:
    """Normalize a ligand JSON/TSV entry to ``{type, name, smiles?}``."""
    if isinstance(entry, str):
        key = entry.strip().lower()
        return {"type": _LIGAND_TYPE_ALIASES.get(key, "small"), "name": entry.strip()}
    if not isinstance(entry, dict):
        return {"type": "small", "name": str(entry)}
    raw_type = (
        entry.get("type") or entry.get("class") or entry.get("category") or "small"
    )
    key = str(raw_type).strip().lower()
    return {
        "type": _LIGAND_TYPE_ALIASES.get(key, "small"),
        "name": entry.get("name") or entry.get("id") or str(raw_type),
        "smiles": entry.get("smiles"),
    }


def load_ligand_map(path: Optional[str]) -> dict[str, list[dict]]:
    """
    Load optional ligand context map for conditioned lipid/NA/metal cues.

    Formats:
      - JSON ``{protein_id: [\"lipid\", {\"type\": \"nucleotide\"}, ...]}``
      - TSV ``protein_id\\ttype[\\tname]``
    """
    if not path or not os.path.isfile(path):
        return {}
    if path.lower().endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
        out: dict[str, list[dict]] = {}
        for k, v in raw.items():
            if isinstance(v, list):
                out[k] = [normalize_ligand_entry(x) for x in v]
            else:
                out[k] = [normalize_ligand_entry(v)]
        return out

    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            entry = {"type": parts[1]}
            if len(parts) >= 3:
                entry["name"] = parts[2]
            out.setdefault(parts[0], []).append(normalize_ligand_entry(entry))
    return out


def ligand_binding_support(
    idr_sequence: str,
    ligands: list,
) -> dict:
    """
    Cheap ligand-conditioned cue for lipid / NA / metal-ish IDR chemistry.

    Does not change model logits — returns support ∈ [0,1], target roles, tags.
    """
    if not ligands or not idr_sequence:
        return {"support": 0.0, "tags": [], "target_roles": [], "n_ligands": 0}

    idr = idr_sequence.upper()
    L = max(len(idr), 1)
    frac_hydro = sum(1 for a in idr if a in "AILMFVW") / L
    frac_arom = sum(1 for a in idr if a in "FWY") / L
    frac_basic = sum(1 for a in idr if a in "KRH") / L
    frac_polar = sum(1 for a in idr if a in "STNQ") / L
    has_cys = "C" in idr

    best = 0.0
    tags: list[str] = []
    targets: list[str] = []
    for raw in ligands:
        lig = normalize_ligand_entry(raw)
        lt = lig["type"]
        score = 0.0
        local: list[str] = []
        role = _LIGAND_TO_ROLE.get(lt)
        if lt == "lipid":
            if frac_hydro >= 0.3:
                score += 0.5
                local.append("hydrophobic_lipid_compatible")
            if frac_arom >= 0.05:
                score += 0.15
                local.append("aromatic_membrane_cue")
        elif lt == "nucleic":
            if frac_basic >= 0.2:
                score += 0.55
                local.append("basic_na_compatible")
            if frac_polar >= 0.2:
                score += 0.15
                local.append("polar_na_cue")
        elif lt == "metal":
            if has_cys or frac_basic >= 0.15:
                score += 0.5
                local.append("metal_chelating_residues")
        else:  # small
            if frac_hydro >= 0.25 or frac_arom >= 0.08:
                score += 0.4
                local.append("small_molecule_pocket_like")
            if frac_polar >= 0.25:
                score += 0.15
                local.append("polar_ligand_cue")

        best = max(best, min(1.0, score))
        for t in local:
            if t not in tags:
                tags.append(t)
        if role and role not in targets and score >= 0.3:
            targets.append(role)

    return {
        "support": round(float(best), 4),
        "tags": tags,
        "target_roles": targets,
        "n_ligands": len(ligands),
        "ligands": [normalize_ligand_entry(x) for x in ligands][:8],
    }

