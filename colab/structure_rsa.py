"""Relative solvent accessibility from AlphaFold structures.

Why this exists
---------------
CAID3 ranks **AlphaFold-rsa 3rd on Disorder-PDB at AUC 0.950**, above every
dedicated predictor except the two PUNCH2 variants, and far above
AlphaFold-pLDDT at rank 11. This project used pLDDT and not rsa.

Measured here on 304 of the 319 CAID3 targets (the rest have no AlphaFold entry):

    raw rsa, per residue      AUC 0.8688
    rsa smoothed over w=21    AUC 0.9459   <- reproduces the published 0.950
    -pLDDT, per residue       AUC 0.9431
    max-ASA table             irrelevant (Tien and Sander&Rost agree to 4 dp)

The window is the whole story: disorder is a regional property and per-residue
solvent accessibility is noisy. Reproducing a published *per-method score* is
much stronger evidence that our CAID3 harness is correct than matching the
benchmark's composition was, and it is why this module pins the window rather
than leaving it a free hyperparameter.

Alignment is checked, not assumed: a structure is used only when its sequence
matches the target exactly. On the CAID3 set that held for 304/304, but this
project has already lost one benchmark to an off-by-mask alignment, so the
check is enforced rather than trusted.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np

# Tien et al. (2013), theoretical maximum ASA in a Gly-X-Gly tripeptide (A^2).
# Sander & Rost (1994) empirical values give the same AUC to four decimals once
# the scores are smoothed, so the choice of table is not load-bearing.
MAX_ASA_TIEN: dict[str, float] = {
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "E": 223, "Q": 225,
    "G": 104, "H": 224, "I": 197, "L": 201, "K": 236, "M": 224, "F": 240,
    "P": 159, "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174,
}

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# Measured on CAID3 Disorder-PDB: 1 -> 0.8688, 9 -> 0.9376, 15 -> 0.9444,
# 21 -> 0.9459, 31 -> 0.9428. Flat near the optimum, so 21 is not overfitted to
# a sharp peak — but it IS chosen on the benchmark, which is why any model that
# consumes rsa must still fit its own weights on training data (see
# structural_fusion.py) rather than treating this as free information.
RSA_SMOOTH_WINDOW = 21

#: Sentinel distinguishing "caller did not say" from an explicit ``None``,
#: which means "do not cache".
_UNSET = "<unset>"


def smooth_window(x: np.ndarray, window: int = RSA_SMOOTH_WINDOW) -> np.ndarray:
    """Centred moving average, always returning one value per input residue.

    ``np.convolve(..., mode="same")`` returns ``max(len(x), len(kernel))``, so a
    protein shorter than the window comes back longer than it went in — a
    20-residue sequence smoothed at w=21 would yield 21 values and shift every
    downstream residue. The pipeline admits proteins from 20 residues up, so the
    window is clamped to the sequence length.
    """
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or x.size == 0:
        return x
    effective = min(int(window), x.size)
    kernel = np.ones(effective) / float(effective)
    out = np.convolve(x, kernel, mode="same")
    assert len(out) == len(x), f"smoothing changed length {len(x)} -> {len(out)}"
    return out


def contact_density(coords: np.ndarray, cutoff: float = 10.0) -> np.ndarray:
    """Neighbours within ``cutoff`` Angstroms of each residue's CB.

    Complementary to solvent accessibility rather than a restatement of it.
    Accessibility is a surface property — an exposed loop on a folded domain
    scores high — whereas contact density measures how much structure a residue
    is embedded in. A disordered residue in an AlphaFold model is typically both
    exposed *and* uncontacted, while an ordered surface residue is exposed and
    densely contacted, so the pair separates cases either alone confuses.

    Counting is O(L^2) in distance but L is capped at 1022 here, so this is
    milliseconds per protein and needs no neighbour structure.
    """
    n = len(coords)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    within = (d < cutoff).sum(axis=1) - 1          # exclude self
    return within.astype(np.float32)


def rsa_from_structure(path: str) -> tuple[
        np.ndarray, str, np.ndarray, np.ndarray, np.ndarray]:
    """Per-residue (rsa, sequence, pLDDT, contacts, CA virtual torsion).

    SASA is Shrake-Rupley via Biopython, so no external DSSP binary is needed.
    AlphaFold stores pLDDT in the B-factor column, so both channels come from
    one parse.
    """
    from Bio.PDB import MMCIFParser
    from Bio.PDB.SASA import ShrakeRupley

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = MMCIFParser(QUIET=True).get_structure("model", path)
        model = next(iter(structure))
        ShrakeRupley().compute(model, level="R")

        rsa: list[float] = []
        seq: list[str] = []
        plddt: list[float] = []
        centres: list[np.ndarray] = []
        ca_centres: list[np.ndarray] = []
        for residue in next(iter(model)):
            aa = THREE_TO_ONE.get(residue.get_resname())
            if aa is None:
                continue
            seq.append(aa)
            rsa.append(float(residue.sasa) / MAX_ASA_TIEN[aa])
            bfactors = [atom.get_bfactor() for atom in residue]
            plddt.append(float(np.mean(bfactors)) if bfactors else float("nan"))
            # CB where present, CA otherwise (glycine has no CB).
            atom = residue["CB"] if "CB" in residue else (
                residue["CA"] if "CA" in residue else None)
            # CA specifically, for the backbone virtual torsion. CB would not
            # do: the torsion is a property of the chain trace, and swapping in
            # CB for non-glycines would make it a different quantity at every
            # glycine.
            ca_atom = residue["CA"] if "CA" in residue else None
            ca_centres.append(
                np.asarray(ca_atom.get_coord(), dtype=np.float32)
                if ca_atom is not None else np.full(3, np.nan, dtype=np.float32))
            centres.append(
                np.asarray(atom.get_coord(), dtype=np.float32)
                if atom is not None else np.full(3, np.nan, dtype=np.float32)
            )

    coords = np.asarray(centres, dtype=np.float32) if centres else np.zeros((0, 3))
    ca = (np.asarray(ca_centres, dtype=np.float32) if ca_centres
          else np.zeros((0, 3)))
    from colab.chirality import ca_virtual_torsion

    return (
        np.asarray(rsa, dtype=np.float32),
        "".join(seq),
        np.asarray(plddt, dtype=np.float32),
        contact_density(coords),
        ca_virtual_torsion(ca),
    )


#: Layout version of the derived-feature cache. An earlier cache under
#: ``rsa_features/`` stored ``rsa``/``plddt``/``seq`` and predates the contacts
#: channel; reading it as if it were current would silently drop a model input.
#: Files without a matching version are ignored rather than interpreted.
RSA_CACHE_VERSION = 3

#: Subdirectory of the structure cache holding derived features.
RSA_CACHE_DIRNAME = "_rsa_v3"


def cif_digest(path: str) -> str:
    """Content fingerprint of an mmCIF, so the cache invalidates itself.

    Keyed on content rather than accession alone: AlphaFold DB has already
    moved v4 -> v6 during this project, and a cache keyed on the accession
    would serve features from the old model forever without anything looking
    wrong. Size and mtime would not do — rsync rewrites both.
    """
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rsa_from_structure_cached(
    path: str, feature_cache: Optional[str] = None,
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, np.ndarray]:
    """``rsa_from_structure`` with the parse memoised on disk.

    Shrake-Rupley in Biopython is pure Python and dominates every run that
    touches structure: on the 22,914-protein union it took **5h04m of the
    18h** wall clock, recomputing the same accessibilities the previous run
    computed. Nothing in that calculation depends on the model, the seed, or
    the task, so it is memoised against the structure's content hash.

    The *raw* rsa is cached, never the smoothed one. The smoothing window is
    chosen on the benchmark (21, from 0.8688 raw to 0.9459 smoothed), so it has
    to stay a parameter a future run can vary; a cache of smoothed values would
    freeze it silently.
    """
    if not feature_cache:
        return rsa_from_structure(path)

    acc = os.path.splitext(os.path.basename(path))[0].upper()
    npz = os.path.join(feature_cache, f"{acc}.npz")
    digest = cif_digest(path)
    if os.path.isfile(npz):
        try:
            with np.load(npz, allow_pickle=False) as d:
                if (int(d["version"]) == RSA_CACHE_VERSION
                        and str(d["cif_digest"]) == digest):
                    return (d["rsa"], str(d["seq"]), d["plddt"],
                            d["contacts"], d["ca_torsion"])
        except Exception:
            pass  # A truncated or foreign file is recomputed, never trusted.

    rsa, seq, plddt, contacts, torsion = rsa_from_structure(path)
    os.makedirs(feature_cache, exist_ok=True)
    # Written through a file object, not a path: np.savez_compressed appends
    # ".npz" to any path lacking it, so a ".part" temp would land beside the
    # name we then try to rename, and every write would fail silently.
    tmp = f"{npz}.{os.getpid()}.part"
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh, version=RSA_CACHE_VERSION, cif_digest=digest,
                rsa=rsa, seq=seq, plddt=plddt, contacts=contacts,
                ca_torsion=torsion)
        os.replace(tmp, npz)          # atomic: many workers share this dir
    except OSError:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return rsa, seq, plddt, contacts, torsion


def fetch_structure(
    uniprot_acc: str, cache_dir: str, timeout: int = 60
) -> Optional[str]:
    """Download the AlphaFold mmCIF for an accession, cached on disk.

    The URL comes from the AlphaFold DB API, never a hardcoded version string:
    the ``model_v4`` path 404s because the database moved to v6, and a pinned
    version fails silently — yielding zero structures and, if unnoticed, a
    "structure features unavailable" result that looks like a modelling finding.
    """
    if not uniprot_acc:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{uniprot_acc.upper()}.cif")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    import requests

    from colab.af_plddt import fetch_afdb_metadata

    meta = fetch_afdb_metadata(uniprot_acc)
    url = (meta or {}).get("cifUrl")
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        tmp = path + ".part"
        with open(tmp, "w") as fh:
            fh.write(resp.text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except (requests.RequestException, OSError):
        if os.path.exists(path + ".part"):
            os.unlink(path + ".part")
        return None
    return path


def structure_features(
    uniprot_acc: str,
    target_sequence: str,
    cache_dir: str,
    window: int = RSA_SMOOTH_WINDOW,
    allow_fetch: bool = True,
    feature_cache: Optional[str] = _UNSET,
) -> Optional[dict]:
    """Smoothed rsa and pLDDT aligned to ``target_sequence``, or None.

    Returns None rather than a misaligned array whenever the structure's
    sequence differs from the target. An AlphaFold entry is keyed by UniProt
    accession and may correspond to a different isoform than the benchmark
    target; scoring one against the other is the same class of error that made
    this project's CAID3 numbers wrong for its whole history.

    ``feature_cache`` memoises the SASA parse. It defaults to a subdirectory of
    ``cache_dir``, which makes the fast path the one that runs by default;
    pass ``None`` to recompute from the mmCIF every time. Either way the answer
    is the same array — the cache carries the structure's content hash and
    recomputes on any mismatch — so this is a speed setting, not a science one.
    """
    if not uniprot_acc or not target_sequence:
        return None
    path = os.path.join(cache_dir, f"{uniprot_acc.upper()}.cif")
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        if not allow_fetch:
            return None
        path = fetch_structure(uniprot_acc, cache_dir)
        if path is None:
            return None
    if feature_cache is _UNSET:
        feature_cache = os.path.join(cache_dir, RSA_CACHE_DIRNAME)
    try:
        rsa, seq, plddt, contacts, torsion = rsa_from_structure_cached(
            path, feature_cache)
    except Exception:
        return None
    if seq != target_sequence:
        return None
    # Contacts are smoothed on the same window as rsa: both are regional
    # properties, and an unsmoothed count is as noisy per-residue as raw
    # accessibility was (0.8688 against 0.9459 smoothed).
    return {
        "rsa": smooth_window(rsa, window).astype(np.float32),
        "rsa_raw": rsa,
        "plddt": plddt,
        # Scaled by a typical globular-core count so the channel arrives at
        # roughly unit range, like rsa and pLDDT/100.
        "contacts": (smooth_window(contacts, window) / 20.0).astype(np.float32),
        # Backbone handedness. Left unsmoothed *and* left signed: smoothing a
        # signed quantity across a helix boundary cancels it, and every other
        # structural channel here is mirror-invariant, so this is the only one
        # that can tell a structure from its reflection.
        "ca_torsion": np.asarray(torsion, dtype=np.float32),
        "handedness": np.sin(np.radians(
            np.asarray(torsion, dtype=np.float64))).astype(np.float32),
        "window": int(window),
        "uniprot_acc": uniprot_acc.upper(),
    }
