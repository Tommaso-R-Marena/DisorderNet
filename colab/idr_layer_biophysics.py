"""
AF-blind IDR biophysics cues for the biology layer (no MD / no ensembles).

Cheap sequence-derived patterning signals that often correlate with condensate
propensity and binding — interpretation aids only; they do not change logits.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np

_POS = set("KRH")
_NEG = set("DE")
_AROM = set("FWY")
_HYDRO = set("AILMFVW")


def _charges(seq: str) -> np.ndarray:
    out = np.zeros(len(seq), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in _POS:
            out[i] = 1.0
        elif aa in _NEG:
            out[i] = -1.0
    return out


def net_charge_per_residue(seq: str) -> float:
    if not seq:
        return 0.0
    c = _charges(seq)
    return float(c.sum() / len(seq))


def fraction_charged_residues(seq: str) -> float:
    if not seq:
        return 0.0
    return float(sum(1 for a in seq if a in _POS or a in _NEG) / len(seq))


def sequence_complexity_entropy(seq: str) -> float:
    """Shannon entropy over amino-acid frequencies (bits), normalized to [0,1]."""
    if not seq:
        return 0.0
    counts = Counter(seq)
    n = len(seq)
    ent = -sum((c / n) * math.log2(c / n) for c in counts.values() if c)
    # Max entropy for 20 AA alphabet
    return float(ent / math.log2(min(20, n) or 1))


def scd_lite(seq: str) -> float:
    """
    Sequence Charge Decoration (lite).

    Classic SCD ≈ Σ_{i<j} q_i q_j √|i-j| / N. Positive SCD → same-sign blocks
    separated in sequence (stretched / repulsive patterning); negative →
    opposite-charge mixing (compact / sticky electrostatics).
    Truncated O(N·W) form with window W for efficiency on long IDRs.
    """
    if len(seq) < 4:
        return 0.0
    q = _charges(seq)
    n = len(q)
    # Windowed approximation — accuracy-preserving enough for triage cues
    w = min(40, n - 1)
    total = 0.0
    for i in range(n):
        qi = float(q[i])
        if qi == 0.0:
            continue
        j_hi = min(n, i + w + 1)
        for j in range(i + 1, j_hi):
            qj = float(q[j])
            if qj == 0.0:
                continue
            total += qi * qj * math.sqrt(j - i)
    return float(total / n)


def kappa_lite(seq: str, blob: int = 5) -> Optional[float]:
    """
    Charge patterning κ-lite in [0, 1]-ish: local vs global charge asymmetry.

    High κ → blocky same-charge stretches (often condensate / binding relevant);
    low κ → well-mixed charges. Returns None if FCR too low to pattern.
    """
    if len(seq) < blob * 2:
        return None
    q = _charges(seq)
    fcr = float(np.mean(np.abs(q)))
    if fcr < 0.1:
        return None
    # Global: fraction of positions that are + among charged, vs −
    charged = q[q != 0]
    if len(charged) < 4:
        return None
    # Local blob sigma of net charge density
    nets = []
    for i in range(0, len(q) - blob + 1):
        nets.append(float(q[i:i + blob].sum()) / blob)
    if not nets:
        return None
    local_var = float(np.var(nets))
    # Normalize roughly into [0, 1]
    kappa = local_var / (fcr * fcr + 1e-8)
    return float(min(1.0, max(0.0, kappa)))


def compute_idr_biophysics_cues(sequence: str, start: int, end: int) -> dict:
    """
    Biophysics cue block for one IDR segment (0-based half-open).

    Explicitly *not* a conformational ensemble — sequence patterning only.
    """
    seg = (sequence[start:end] or "").upper()
    L = max(len(seg), 1)
    ncpr = net_charge_per_residue(seg)
    fcr = fraction_charged_residues(seg)
    scd = scd_lite(seg)
    kappa = kappa_lite(seg)
    entropy = sequence_complexity_entropy(seg)
    arom = sum(1 for a in seg if a in _AROM) / L
    hydro = sum(1 for a in seg if a in _HYDRO) / L

    tags: list[str] = []
    if fcr >= 0.3 and kappa is not None and kappa >= 0.35:
        tags.append("blocky_charge_patterning")
    if fcr >= 0.25 and scd < -0.5:
        tags.append("mixed_charge_compact_electrostatics")
    if fcr >= 0.25 and scd > 1.0:
        tags.append("segregated_charge_stretching")
    if arom >= 0.08 and fcr >= 0.2:
        tags.append("aromatic_charged_sticker_spacer")
    if entropy <= 0.55 and L >= 15:
        tags.append("low_complexity_idr")
    if abs(ncpr) >= 0.25 and fcr >= 0.3:
        tags.append("strongly_signed_polyelectrolyte")

    return {
        "ncpr": round(ncpr, 4),
        "fcr": round(fcr, 4),
        "scd_lite": round(scd, 4),
        "kappa_lite": None if kappa is None else round(kappa, 4),
        "composition_entropy": round(entropy, 4),
        "frac_aromatic": round(arom, 4),
        "frac_hydrophobic": round(hydro, 4),
        "cue_tags": tags,
        "note": "Sequence patterning cue — not an MD / conformational ensemble",
    }
