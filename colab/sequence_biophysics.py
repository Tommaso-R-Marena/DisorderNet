"""Charge patterning and sequence complexity — training-free, protein-level.

Our Binding-IDR deficit is entirely between-protein: within-protein AUC matches
the leader (0.7004 against 0.6958) while between-protein sits at 0.4982, chance,
against their 0.6400. So what is missing is a quantity that distinguishes whole
chains, and a per-residue model with a 213-residue field has no channel for it.

Polymer theory supplies exactly such quantities, and they need no training.
Sequence charge decoration and the related patterning measures are the
established sequence-to-ensemble relationship for IDRs: the *arrangement* of
charge along a chain, not merely its composition, sets whether a disordered
region behaves as an expanded coil or a compact globule (Das & Pappu 2013;
Sawle & Ghosh 2015). Those are protein-level statements about disordered
sequence, which is what the between-protein axis is asking about.

Sequence complexity enters for a different reason: low-complexity regions are
long associated with disorder, and Shannon entropy over the composition is the
direct measure of it.

Everything here is a deterministic function of sequence. Nothing is fitted, so
anything these predict is a property of the biophysics rather than of a model
that saw the benchmark.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

#: Kyte-Doolittle hydropathy.
_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
_POS, _NEG = set("KR"), set("DE")


def charges(seq: str) -> np.ndarray:
    """+1 for K/R, −1 for D/E, 0 otherwise. Histidine is left neutral, which is
    the usual convention at physiological pH."""
    return np.array([1.0 if c in _POS else (-1.0 if c in _NEG else 0.0)
                     for c in seq.upper()], dtype=np.float64)


def fcr_ncpr(seq: str) -> tuple[float, float]:
    """Fraction of charged residues, and net charge per residue.

    Composition only — deliberately, so that SCD's extra content (arrangement)
    can be measured against it rather than confounded with it.
    """
    q = charges(seq)
    n = max(len(q), 1)
    return float(np.abs(q).sum() / n), float(q.sum() / n)


def scd(seq: str) -> float:
    """Sequence charge decoration (Sawle & Ghosh 2015).

        SCD = (1/N) Σ_{i<j} q_i q_j √(j − i)

    More negative when like charges segregate into blocks, less negative when
    opposite charges are well mixed — because the opposite-charge pairs, which
    carry the negative sign, sit far apart in a blocky sequence and are weighted
    by the larger √(j−i). Blocky polyampholytes are the compact ones, so SCD
    runs negatively with chain dimension, which is the Sawle-Ghosh result.

    Unlike FCR and NCPR it is sensitive to *order*: two sequences with identical
    composition but different arrangement — the case that separates an expanded
    coil from a compact globule — differ here and nowhere else. KKKKDDDD scores
    −2.02 against KDKDKDKD's −0.45, while both have FCR 1.0 and NCPR 0.0.

    Computed in O(N²) over charged positions only, which is what makes it
    tractable: charged residues are a minority even in a polyampholyte.
    """
    q = charges(seq)
    idx = np.nonzero(q)[0]
    if len(idx) < 2:
        return 0.0
    qi = q[idx]
    total = 0.0
    for a in range(len(idx) - 1):
        sep = np.sqrt(idx[a + 1:] - idx[a])
        total += float(qi[a] * np.dot(qi[a + 1:], sep))
    return total / max(len(q), 1)


def kappa_like(seq: str, window: int = 5) -> float:
    """Charge asymmetry, in the spirit of Das & Pappu's κ.

    The deviation of local charge asymmetry from the whole-sequence value,
    averaged over windows and normalised by the value a fully segregated
    sequence of the same composition would give. Returns NaN when the sequence
    carries too little charge for the quantity to mean anything, rather than 0,
    which would read as "well mixed".
    """
    q = charges(seq)
    n = len(q)
    if n < window:
        return float("nan")
    npos, nneg = float((q > 0).sum()), float((q < 0).sum())
    if npos + nneg < 2:
        return float("nan")

    def asym(sub):
        p, m = float((sub > 0).sum()), float((sub < 0).sum())
        return (p - m) ** 2 / (p + m) if (p + m) else 0.0

    overall = asym(q)
    devs = [(asym(q[i:i + window]) - overall) ** 2
            for i in range(n - window + 1)]
    obs = float(np.mean(devs))
    # Fully segregated reference: all positives, then all negatives.
    seg = np.concatenate([np.ones(int(npos)), -np.ones(int(nneg))])
    if len(seg) < window:
        return float("nan")
    ref = float(np.mean([(asym(seg[i:i + window]) - overall) ** 2
                         for i in range(len(seg) - window + 1)]))
    return obs / ref if ref > 0 else float("nan")


def shannon_entropy(seq: str) -> float:
    """Shannon entropy of the amino-acid composition, in bits.

    Low-complexity sequence is long associated with disorder; this is the direct
    measure. Maximum is log2(20) ≈ 4.32.
    """
    if not seq:
        return float("nan")
    counts = Counter(seq.upper())
    n = sum(counts.values())
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def mean_hydropathy(seq: str) -> float:
    vals = [_HYDROPATHY[c] for c in seq.upper() if c in _HYDROPATHY]
    return float(np.mean(vals)) if vals else float("nan")


def descriptors(seq: str) -> dict[str, float]:
    """Every protein-level descriptor, for one chain."""
    fcr, ncpr = fcr_ncpr(seq)
    return {
        "fcr": fcr,
        "ncpr": ncpr,
        "abs_ncpr": abs(ncpr),
        "scd": scd(seq),
        "kappa_like": kappa_like(seq),
        "entropy": shannon_entropy(seq),
        "hydropathy": mean_hydropathy(seq),
        "length": float(len(seq)),
    }


def local_descriptors(seq: str, window: int = 25) -> dict[str, np.ndarray]:
    """Per-residue versions, for the within-protein axis.

    Windowed so each residue carries the composition of its neighbourhood,
    which is the scale at which low complexity and charge blocks act.
    """
    n = len(seq)
    half = window // 2
    ent = np.full(n, np.nan)
    hyd = np.full(n, np.nan)
    net = np.full(n, np.nan)
    for i in range(n):
        sub = seq[max(0, i - half):min(n, i + half + 1)]
        ent[i] = shannon_entropy(sub)
        hyd[i] = mean_hydropathy(sub)
        net[i] = fcr_ncpr(sub)[1]
    return {"entropy": ent, "hydropathy": hyd, "ncpr": net}
