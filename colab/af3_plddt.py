"""
AlphaFold 3 pLDDT ingestion utilities (Phase 2b).

Loads per-residue pLDDT from AF3 mmCIF (B-factors on CA atoms) or
confidences JSON + mmCIF atom order. Designed for Google Drive–mounted
outputs — never commit AF3 weights or model outputs to GitHub.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Optional

import numpy as np

from colab.af_plddt import align_plddt_to_sequence

DEFAULT_AF3_CACHE_DIR = "af3_plddt_cache"

AA3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X", "MSE": "M",
}


def _parse_mmcif_atom_site(cif_text: str) -> list[dict]:
    """Parse _atom_site loop from mmCIF into row dicts."""
    lines = cif_text.splitlines()
    columns: list[str] = []
    rows: list[dict] = []
    in_loop = False
    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            columns = []
            in_loop = False
            continue
        if stripped.startswith("_atom_site."):
            columns.append(stripped.split(".", 1)[1])
            in_loop = True
            continue
        if in_loop and columns:
            if stripped.startswith("_") or stripped.startswith("#"):
                in_loop = False
                columns = []
                continue
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < len(columns):
                continue
            rows.append(dict(zip(columns, parts)))
    return rows


def parse_plddt_from_mmcif(cif_text: str, ca_only: bool = True) -> tuple[np.ndarray, str]:
    """
    Extract per-residue pLDDT from AF3 mmCIF B_iso_or_equiv on protein CA atoms.

    Returns (plddt array, one-letter sequence).
    """
    atoms = _parse_mmcif_atom_site(cif_text)
    plddt: list[float] = []
    seq_chars: list[str] = []
    seen_residue: set[tuple[str, str]] = set()

    for atom in atoms:
        if atom.get("group_PDB", "").upper() != "ATOM":
            continue
        atom_id = atom.get("label_atom_id") or atom.get("auth_atom_id", "")
        if ca_only and atom_id != "CA":
            continue
        comp_id = atom.get("label_comp_id") or atom.get("auth_comp_id", "UNK")
        chain = atom.get("label_asym_id") or atom.get("auth_asym_id", "A")
        seq_id = atom.get("label_seq_id") or atom.get("auth_seq_id", "")
        if not seq_id or seq_id == ".":
            continue
        residue_key = (chain, seq_id)
        if residue_key in seen_residue:
            continue
        seen_residue.add(residue_key)
        bfac = atom.get("B_iso_or_equiv") or atom.get("occupancy", "0")
        try:
            plddt.append(float(bfac))
        except ValueError:
            continue
        seq_chars.append(AA3TO1.get(comp_id.upper(), "X"))

    return np.asarray(plddt, dtype=np.float32), "".join(seq_chars)


def parse_plddt_from_confidences_json(
    confidences: dict,
    cif_text: str,
    ca_only: bool = True,
) -> tuple[np.ndarray, str]:
    """
    Map atom_plddts from confidences JSON onto protein CA residues via mmCIF order.
    """
    atom_plddts = confidences.get("atom_plddts")
    if not atom_plddts:
        return parse_plddt_from_mmcif(cif_text, ca_only=ca_only)

    atoms = _parse_mmcif_atom_site(cif_text)
    if len(atoms) != len(atom_plddts):
        return parse_plddt_from_mmcif(cif_text, ca_only=ca_only)

    plddt: list[float] = []
    seq_chars: list[str] = []
    seen_residue: set[tuple[str, str]] = set()

    for atom, plddt_val in zip(atoms, atom_plddts):
        if atom.get("group_PDB", "").upper() != "ATOM":
            continue
        atom_id = atom.get("label_atom_id") or atom.get("auth_atom_id", "")
        if ca_only and atom_id != "CA":
            continue
        comp_id = atom.get("label_comp_id") or atom.get("auth_comp_id", "UNK")
        chain = atom.get("label_asym_id") or atom.get("auth_asym_id", "A")
        seq_id = atom.get("label_seq_id") or atom.get("auth_seq_id", "")
        if not seq_id or seq_id == ".":
            continue
        residue_key = (chain, seq_id)
        if residue_key in seen_residue:
            continue
        seen_residue.add(residue_key)
        plddt.append(float(plddt_val))
        seq_chars.append(AA3TO1.get(comp_id.upper(), "X"))

    if not plddt:
        return parse_plddt_from_mmcif(cif_text, ca_only=ca_only)
    return np.asarray(plddt, dtype=np.float32), "".join(seq_chars)


def load_af3_plddt_from_files(
    confidences_path: Optional[str],
    model_cif_path: str,
) -> tuple[np.ndarray, str]:
    """Load pLDDT + sequence from AF3 output files."""
    with open(model_cif_path) as f:
        cif_text = f.read()

    if confidences_path and os.path.exists(confidences_path):
        with open(confidences_path) as f:
            confidences = json.load(f)
        return parse_plddt_from_confidences_json(confidences, cif_text)
    return parse_plddt_from_mmcif(cif_text)


def _sanitize_name(name: str) -> str:
    """AF3 sanitizes job names for directory/file prefixes."""
    return re.sub(r"[^\w\-]", "_", name.strip())


def find_af3_output_pair(
    output_root: str,
    protein_id: str,
    uniprot_acc: str = "",
) -> Optional[tuple[str, str]]:
    """
    Locate (confidences.json, model.cif) for a protein under an AF3 output tree.

    Searches job folders named by protein id, UniProt accession, or sanitized variants.
    """
    if not output_root or not os.path.isdir(output_root):
        return None

    candidates = []
    for key in {protein_id, uniprot_acc.upper(), _sanitize_name(protein_id), _sanitize_name(uniprot_acc)}:
        if not key:
            continue
        candidates.extend([
            os.path.join(output_root, key),
            os.path.join(output_root, key.lower()),
            os.path.join(output_root, key.upper()),
        ])

    for folder in candidates:
        if not os.path.isdir(folder):
            continue
        pair = _pair_from_job_dir(folder)
        if pair:
            return pair

    # Flat layout: files directly under output_root
    for key in {protein_id, uniprot_acc.upper(), _sanitize_name(protein_id)}:
        if not key:
            continue
        for conf in glob.glob(os.path.join(output_root, f"*{key}*_confidences.json")):
            base = conf.replace("_confidences.json", "")
            cif = f"{base}_model.cif"
            if os.path.exists(cif):
                return conf, cif
            alt = conf.replace("_confidences.json", "_model.cif")
            if os.path.exists(alt):
                return conf, alt

    # Deep search (one level of subdirs) — expensive but bounded
    for entry in os.listdir(output_root):
        sub = os.path.join(output_root, entry)
        if not os.path.isdir(sub):
            continue
        for key in {protein_id, uniprot_acc.upper()}:
            if key and key.lower() in entry.lower():
                pair = _pair_from_job_dir(sub)
                if pair:
                    return pair
    return None


def _pair_from_job_dir(job_dir: str) -> Optional[tuple[str, str]]:
    """Find top-ranking or first available confidences + model.cif in a job directory."""
    patterns = [
        ("*_confidences.json", "*_model.cif"),
        ("*confidences.json", "*model.cif"),
    ]
    for conf_pat, cif_pat in patterns:
        conf_files = sorted(glob.glob(os.path.join(job_dir, conf_pat)))
        for conf in conf_files:
            if "summary_confidences" in conf:
                continue
            stem = conf.replace("_confidences.json", "")
            cif = f"{stem}_model.cif"
            if os.path.exists(cif):
                return conf, cif
        cif_files = sorted(glob.glob(os.path.join(job_dir, cif_pat)))
        for cif in cif_files:
            conf = cif.replace("_model.cif", "_confidences.json")
            if os.path.exists(conf):
                return conf, cif

    # seed/sample subdirectories
    for sub in sorted(glob.glob(os.path.join(job_dir, "seed-*"))):
        pair = _pair_from_job_dir(sub)
        if pair:
            return pair
    return None


def load_af3_plddt_for_protein(
    protein_id: str,
    target_sequence: str,
    output_root: str,
    uniprot_acc: str = "",
    cache_dir: str = DEFAULT_AF3_CACHE_DIR,
) -> Optional[np.ndarray]:
    """Load AF3 pLDDT aligned to DisProt sequence, with disk cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{protein_id}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("target_sequence") == target_sequence and "plddt" in cached:
            return np.asarray(cached["plddt"], dtype=np.float32)

    pair = find_af3_output_pair(output_root, protein_id, uniprot_acc)
    if not pair:
        return None

    conf_path, cif_path = pair
    plddt, af_seq = load_af3_plddt_from_files(conf_path, cif_path)
    aligned = align_plddt_to_sequence(plddt, af_seq, target_sequence)
    if aligned is None:
        return None

    with open(cache_path, "w") as f:
        json.dump({
            "protein_id": protein_id,
            "uniprot_acc": uniprot_acc.upper() if uniprot_acc else "",
            "target_sequence": target_sequence,
            "plddt": aligned.tolist(),
            "af_sequence_len": len(af_seq),
            "source_files": {"confidences": conf_path, "model_cif": cif_path},
        }, f)
    return aligned


def load_af3_plddt_batch(
    proteins: list,
    output_root: str,
    cache_dir: str = DEFAULT_AF3_CACHE_DIR,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Load AF3 pLDDT for all proteins with outputs under output_root."""
    from tqdm.auto import tqdm

    results: dict[str, np.ndarray] = {}
    iterator = proteins
    if verbose:
        iterator = tqdm(proteins, desc="Loading AF3 pLDDT")

    for p in iterator:
        plddt = load_af3_plddt_for_protein(
            protein_id=p["id"],
            target_sequence=p["sequence"],
            output_root=output_root,
            uniprot_acc=p.get("uniprot_acc", ""),
            cache_dir=cache_dir,
        )
        if plddt is not None:
            results[p["id"]] = plddt
    return results
