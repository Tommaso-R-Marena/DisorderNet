"""
AlphaFold Database pLDDT fetch and alignment utilities (Phase 2).

Fetches per-residue pLDDT from AlphaFold DB (AF2 models) via the public API
and PDB files. Used to quantify structure-predictor hallucinations in IDRs.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import requests
from tqdm.auto import tqdm

AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction"
DEFAULT_CACHE_DIR = "af_plddt_cache"
REQUEST_TIMEOUT = 60


def parse_plddt_from_pdb(pdb_text: str) -> tuple[np.ndarray, str]:
    """
    Extract CA-atom pLDDT (B-factor column) and one-letter sequence from PDB.

    Returns (plddt array, sequence string).
    """
    plddt: list[float] = []
    seq_chars: list[str] = []
    aa3to1 = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        resname = line[17:20].strip()
        seq_chars.append(aa3to1.get(resname, "X"))
        plddt.append(float(line[60:66]))
    return np.asarray(plddt, dtype=np.float32), "".join(seq_chars)


def align_plddt_to_sequence(
    plddt: np.ndarray,
    af_sequence: str,
    target_sequence: str,
) -> Optional[np.ndarray]:
    """
    Map AlphaFold pLDDT onto a DisProt sequence.

    Returns per-residue pLDDT aligned to target_sequence, or None if alignment fails.
    """
    if len(plddt) == 0:
        return None
    if af_sequence == target_sequence and len(plddt) >= len(target_sequence):
        return plddt[: len(target_sequence)].copy()
    if target_sequence in af_sequence:
        start = af_sequence.index(target_sequence)
        end = start + len(target_sequence)
        if end <= len(plddt):
            return plddt[start:end].copy()
    if af_sequence in target_sequence:
        start = target_sequence.index(af_sequence)
        out = np.full(len(target_sequence), np.nan, dtype=np.float32)
        out[start:start + len(plddt)] = plddt
        return out
    # Prefix/suffix overlap (common for trimmed DisProt entries)
    min_len = min(len(plddt), len(target_sequence))
    if af_sequence[:min_len] == target_sequence[:min_len]:
        out = np.full(len(target_sequence), np.nan, dtype=np.float32)
        out[:min_len] = plddt[:min_len]
        return out
    return None


def plddt_to_disorder_score(plddt: np.ndarray) -> np.ndarray:
    """CAID-style continuous disorder score: low pLDDT → high disorder."""
    return np.clip(1.0 - plddt / 100.0, 0.0, 1.0)


def fetch_afdb_metadata(uniprot_acc: str) -> Optional[dict]:
    """Query AlphaFold DB prediction metadata for a UniProt accession."""
    if not uniprot_acc:
        return None
    url = f"{AFDB_API}/{uniprot_acc.upper()}"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data[0] if data else None
        return data
    except requests.RequestException:
        return None


def fetch_plddt_for_uniprot(
    uniprot_acc: str,
    target_sequence: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    session: Optional[requests.Session] = None,
) -> Optional[np.ndarray]:
    """
    Fetch and cache pLDDT for a UniProt accession, aligned to target_sequence.
    """
    if not uniprot_acc:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{uniprot_acc.upper()}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("target_sequence") == target_sequence and "plddt" in cached:
            return np.asarray(cached["plddt"], dtype=np.float32)

    sess = session or requests.Session()
    meta = fetch_afdb_metadata(uniprot_acc)
    if not meta:
        return None

    pdb_url = meta.get("pdbUrl") or meta.get("cifUrl", "").replace(".cif", ".pdb")
    if not pdb_url:
        return None

    try:
        pdb_resp = sess.get(pdb_url, timeout=REQUEST_TIMEOUT)
        pdb_resp.raise_for_status()
    except requests.RequestException:
        return None

    plddt, af_seq = parse_plddt_from_pdb(pdb_resp.text)
    api_seq = meta.get("uniprotSequence") or meta.get("sequence") or af_seq
    aligned = align_plddt_to_sequence(plddt, api_seq or af_seq, target_sequence)
    if aligned is None:
        aligned = align_plddt_to_sequence(plddt, af_seq, target_sequence)
    if aligned is None:
        return None

    with open(cache_path, "w") as f:
        json.dump({
            "uniprot_acc": uniprot_acc.upper(),
            "target_sequence": target_sequence,
            "plddt": aligned.tolist(),
            "af_sequence_len": len(af_seq),
        }, f)
    return aligned


def fetch_plddt_batch(
    proteins: list,
    cache_dir: str = DEFAULT_CACHE_DIR,
    sleep_s: float = 0.1,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Fetch pLDDT for all proteins with UniProt accessions.

    Returns dict mapping protein id → aligned pLDDT array.
    """
    results: dict[str, np.ndarray] = {}
    session = requests.Session()
    iterator = proteins
    if verbose:
        iterator = tqdm(proteins, desc="Fetching AlphaFold pLDDT")

    for p in iterator:
        acc = p.get("uniprot_acc", "")
        if not acc:
            continue
        plddt = fetch_plddt_for_uniprot(
            acc, p["sequence"], cache_dir=cache_dir, session=session,
        )
        if plddt is not None:
            results[p["id"]] = plddt
        if sleep_s > 0:
            time.sleep(sleep_s)
    return results
