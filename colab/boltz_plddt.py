"""
Boltz-2 per-residue confidence (pLDDT) ingestion.

Boltz writes ``plddt_*.npz`` (often 0–1) and mmCIF with B-factors.
We normalize to 0–100 and align to the DisProt sequence — same contract as AF2/AF3.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Optional

import numpy as np

from colab.af_plddt import align_plddt_to_sequence
from colab.af3_plddt import parse_plddt_from_mmcif

DEFAULT_BOLTZ_CACHE_DIR = "boltz_plddt_cache"


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", (name or "").strip())


def normalize_plddt_scale(plddt: np.ndarray) -> np.ndarray:
    """Boltz may emit 0–1 floats; DisorderNet expects 0–100."""
    arr = np.asarray(plddt, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.nanmax(finite)) <= 1.5:
        arr = arr * 100.0
    return arr


def parse_plddt_from_boltz_npz(path: str) -> np.ndarray:
    """Load per-token pLDDT from Boltz ``plddt_*.npz``."""
    data = np.load(path)
    # Common keys: "plddt", "confidence", or single array
    if isinstance(data, np.lib.npyio.NpzFile):
        if "plddt" in data:
            arr = data["plddt"]
        elif "confidence" in data:
            arr = data["confidence"]
        else:
            key = list(data.keys())[0]
            arr = data[key]
    else:
        arr = np.asarray(data)
    return normalize_plddt_scale(np.asarray(arr, dtype=np.float32).ravel())


def find_boltz_prediction_dir(
    output_root: str,
    protein_id: str,
    uniprot_acc: str = "",
) -> Optional[str]:
    """
    Locate Boltz predictions folder for a protein.

    Layout::
      {output_root}/predictions/{job_name}/plddt_{job_name}_model_0.npz
      {output_root}/{job_name}/...   (flat)
    """
    if not output_root or not os.path.isdir(output_root):
        return None

    keys = [
        k for k in {
            protein_id,
            uniprot_acc,
            uniprot_acc.upper() if uniprot_acc else "",
            _sanitize_name(protein_id),
            _sanitize_name(uniprot_acc),
        }
        if k
    ]

    search_roots = [output_root]
    pred_root = os.path.join(output_root, "predictions")
    if os.path.isdir(pred_root):
        search_roots.insert(0, pred_root)

    for root in search_roots:
        for key in keys:
            for candidate in (
                os.path.join(root, key),
                os.path.join(root, key.lower()),
                os.path.join(root, key.upper()),
                os.path.join(root, _sanitize_name(key)),
            ):
                if os.path.isdir(candidate) and _has_boltz_outputs(candidate):
                    return candidate

        # fuzzy: folder name contains key
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        for entry in entries:
            el = entry.lower()
            if any(k.lower() in el for k in keys):
                path = os.path.join(root, entry)
                if os.path.isdir(path) and _has_boltz_outputs(path):
                    return path
    return None


def _has_boltz_outputs(job_dir: str) -> bool:
    return bool(
        glob.glob(os.path.join(job_dir, "plddt_*.npz"))
        or glob.glob(os.path.join(job_dir, "*_model_*.cif"))
        or glob.glob(os.path.join(job_dir, "*.cif"))
    )


def load_boltz_plddt_from_dir(job_dir: str) -> tuple[np.ndarray, str]:
    """
    Extract (plddt, sequence) from a Boltz prediction folder.

    Prefer ``plddt_*_model_0.npz`` + matching CIF for sequence; fall back to CIF B-factors.
    """
    npz_files = sorted(glob.glob(os.path.join(job_dir, "plddt_*_model_0.npz")))
    if not npz_files:
        npz_files = sorted(glob.glob(os.path.join(job_dir, "plddt_*.npz")))

    cif_files = sorted(glob.glob(os.path.join(job_dir, "*_model_0.cif")))
    if not cif_files:
        cif_files = sorted(glob.glob(os.path.join(job_dir, "*_model_*.cif")))
    if not cif_files:
        cif_files = sorted(glob.glob(os.path.join(job_dir, "*.cif")))

    af_seq = ""
    if cif_files:
        with open(cif_files[0]) as f:
            cif_plddt, af_seq = parse_plddt_from_mmcif(f.read())
        cif_plddt = normalize_plddt_scale(cif_plddt)
    else:
        cif_plddt = np.asarray([], dtype=np.float32)

    if npz_files:
        plddt = parse_plddt_from_boltz_npz(npz_files[0])
        if af_seq and len(plddt) != len(af_seq):
            # Token vs residue mismatch — prefer CIF residue pLDDT when lengths disagree
            if len(cif_plddt) == len(af_seq) and len(cif_plddt) > 0:
                return cif_plddt, af_seq
        if not af_seq:
            af_seq = "X" * len(plddt)
        return plddt, af_seq

    if len(cif_plddt) > 0:
        return cif_plddt, af_seq
    raise FileNotFoundError(f"No Boltz pLDDT / CIF outputs in {job_dir}")


def load_boltz_plddt_for_protein(
    protein_id: str,
    target_sequence: str,
    output_root: str,
    uniprot_acc: str = "",
    cache_dir: str = DEFAULT_BOLTZ_CACHE_DIR,
) -> Optional[np.ndarray]:
    """Load Boltz pLDDT aligned to target sequence, with JSON disk cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{protein_id}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("target_sequence") == target_sequence and "plddt" in cached:
            return np.asarray(cached["plddt"], dtype=np.float32)

    job_dir = find_boltz_prediction_dir(output_root, protein_id, uniprot_acc)
    if not job_dir:
        return None

    try:
        plddt, af_seq = load_boltz_plddt_from_dir(job_dir)
    except (OSError, FileNotFoundError, ValueError):
        return None

    aligned = align_plddt_to_sequence(plddt, af_seq, target_sequence)
    if aligned is None:
        # If sequence unknown (all X), try length-only align
        if len(plddt) == len(target_sequence):
            aligned = np.asarray(plddt, dtype=np.float32)
        else:
            return None

    with open(cache_path, "w") as f:
        json.dump({
            "protein_id": protein_id,
            "uniprot_acc": uniprot_acc.upper() if uniprot_acc else "",
            "target_sequence": target_sequence,
            "plddt": aligned.tolist(),
            "boltz_sequence_len": len(af_seq),
            "source": "boltz2",
            "source_dir": job_dir,
        }, f)
    return aligned


def load_boltz_plddt_batch(
    proteins: list,
    output_root: str,
    cache_dir: str = DEFAULT_BOLTZ_CACHE_DIR,
    verbose: bool = True,
    max_workers: int = 8,
) -> dict[str, np.ndarray]:
    """Load Boltz pLDDT for all proteins (threaded; same cache contract)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm.auto import tqdm

    results: dict[str, np.ndarray] = {}

    def _one(p: dict):
        plddt = load_boltz_plddt_for_protein(
            protein_id=p["id"],
            target_sequence=p["sequence"],
            output_root=output_root,
            uniprot_acc=p.get("uniprot_acc", ""),
            cache_dir=cache_dir,
        )
        return p["id"], plddt

    workers = max(1, min(max_workers, len(proteins) or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, p) for p in proteins]
        iterator = as_completed(futs)
        if verbose:
            iterator = tqdm(iterator, total=len(futs), desc="Loading Boltz pLDDT")
        for fut in iterator:
            pid, plddt = fut.result()
            if plddt is not None:
                results[pid] = plddt
    return results


def load_boltz_plddt_sample_stack(
    job_dir: str,
    max_samples: int = 8,
) -> Optional[np.ndarray]:
    """
    Stack per-sample pLDDT arrays from a Boltz prediction folder.

    Returns shape (n_samples, L) on 0–100 scale, or None if <2 samples.
    Cheap ensemble / dynamics *proxy* — not a physical conformational ensemble.
    """
    npz_files = sorted(glob.glob(os.path.join(job_dir, "plddt_*_model_*.npz")))
    if len(npz_files) < 2:
        # Try CIF B-factors across models
        cif_files = sorted(glob.glob(os.path.join(job_dir, "*_model_*.cif")))
        if len(cif_files) < 2:
            return None
        rows: list[np.ndarray] = []
        length = None
        for cif in cif_files[:max_samples]:
            with open(cif) as f:
                plddt, _ = parse_plddt_from_mmcif(f.read())
            plddt = normalize_plddt_scale(plddt)
            if length is None:
                length = len(plddt)
            if len(plddt) != length:
                continue
            rows.append(plddt)
        if len(rows) < 2:
            return None
        return np.stack(rows, axis=0)

    rows = []
    length = None
    for path in npz_files[:max_samples]:
        plddt = parse_plddt_from_boltz_npz(path)
        if length is None:
            length = len(plddt)
        if len(plddt) != length:
            continue
        rows.append(plddt)
    if len(rows) < 2:
        return None
    return np.stack(rows, axis=0)


def boltz_plddt_variance_from_dir(
    job_dir: str,
    max_samples: int = 8,
) -> Optional[np.ndarray]:
    """Per-residue std of Boltz multi-sample pLDDT (ensemble proxy)."""
    stack = load_boltz_plddt_sample_stack(job_dir, max_samples=max_samples)
    if stack is None:
        return None
    return np.nanstd(stack, axis=0).astype(np.float32)


def load_boltz_variance_for_protein(
    protein_id: str,
    target_sequence: str,
    output_root: str,
    uniprot_acc: str = "",
    cache_dir: str = "boltz_variance_cache",
    max_samples: int = 8,
) -> Optional[np.ndarray]:
    """Disk-cached multi-sample pLDDT std (ensemble proxy)."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{protein_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if (
            cached.get("target_sequence") == target_sequence
            and cached.get("max_samples") == max_samples
            and "std" in cached
        ):
            return np.asarray(cached["std"], dtype=np.float32)

    job_dir = find_boltz_prediction_dir(output_root, protein_id, uniprot_acc)
    if not job_dir:
        return None
    std = boltz_plddt_variance_from_dir(job_dir, max_samples=max_samples)
    if std is None:
        return None
    L = len(target_sequence)
    if len(std) > L:
        std = std[:L]
    elif len(std) < L:
        out = np.full(L, np.nan, dtype=np.float32)
        out[: len(std)] = std
        std = out
    with open(cache_path, "w") as f:
        json.dump({
            "protein_id": protein_id,
            "target_sequence": target_sequence,
            "max_samples": max_samples,
            "std": std.tolist(),
            "source": "boltz2_multisample",
        }, f)
    return std


def load_boltz_variance_batch(
    proteins: list,
    output_root: str,
    max_samples: int = 8,
    verbose: bool = True,
    cache_dir: str = "boltz_variance_cache",
    max_workers: int = 8,
) -> dict[str, np.ndarray]:
    """Load Boltz multi-sample pLDDT std keyed by protein id (threaded + cached)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm.auto import tqdm

    results: dict[str, np.ndarray] = {}

    def _one(p: dict):
        std = load_boltz_variance_for_protein(
            p["id"], p["sequence"], output_root,
            uniprot_acc=p.get("uniprot_acc", ""),
            cache_dir=cache_dir,
            max_samples=max_samples,
        )
        return p["id"], std

    workers = max(1, min(max_workers, len(proteins) or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, p) for p in proteins]
        iterator = as_completed(futs)
        if verbose:
            iterator = tqdm(iterator, total=len(futs), desc="Boltz pLDDT variance")
        for fut in iterator:
            pid, std = fut.result()
            if std is not None:
                results[pid] = std
    return results


def load_boltz_structure_features(
    proteins: list,
    output_root: str,
    *,
    plddt_cache_dir: str = DEFAULT_BOLTZ_CACHE_DIR,
    variance_cache_dir: str = "boltz_variance_cache",
    max_samples: int = 8,
    max_workers: int = 8,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Overlap pLDDT + multi-sample variance loads (I/O bound).

    Returns ``(plddt_by_id, variance_by_id)``. Accuracy-identical to sequential loads;
    cache hits dominate after the first pass.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_plddt = ex.submit(
            load_boltz_plddt_batch,
            proteins,
            output_root,
            cache_dir=plddt_cache_dir,
            verbose=verbose,
            max_workers=max_workers,
        )
        fut_var = ex.submit(
            load_boltz_variance_batch,
            proteins,
            output_root,
            max_samples=max_samples,
            verbose=verbose,
            cache_dir=variance_cache_dir,
            max_workers=max_workers,
        )
        return fut_plddt.result(), fut_var.result()


def select_proteins_for_boltz(
    proteins: list,
    output_root: str,
) -> tuple[list, list]:
    """Split proteins into (done, pending) based on existing Boltz prediction dirs."""
    done: list = []
    pending: list = []
    for p in proteins:
        job = find_boltz_prediction_dir(
            output_root, p["id"], p.get("uniprot_acc", ""),
        )
        if job:
            done.append(p)
        else:
            pending.append(p)
    return done, pending
