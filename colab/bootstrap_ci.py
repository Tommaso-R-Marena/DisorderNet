"""Protein-clustered bootstrap confidence intervals for residue-level metrics.

Why clustering matters
----------------------
Disorder metrics are computed over residues, but residues are not independent:
they come in runs, and whole proteins are ordered or disordered together. A
naive residue-level bootstrap treats ~10^6 correlated residues as ~10^6
independent draws and returns an interval that is far too narrow — narrow
enough to make a 0.009 AUC difference look decisive.

The unit of independent sampling here is the **protein**, so that is what gets
resampled (a cluster / block bootstrap). A protein enters or leaves the sample
with all of its residues at once.

This module exists because the go/no-go criteria rest on differences of a few
thousandths of AUC — ``delta_auc_dn_minus_plddt`` was +0.0088 in the 650M run —
and a point estimate at that scale is uninterpretable without an interval.

Paired comparisons use the *same* protein resample for both scores, so the
interval describes the difference rather than the sum of two independent
sampling errors.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def _safe_ap(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, s))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], Optional[float]]] = {
    "auc": _safe_auc,
    "ap": _safe_ap,
}


def _percentile_ci(samples: Sequence[float], ci: float) -> tuple[Optional[float], Optional[float]]:
    vals = [v for v in samples if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None, None
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(vals, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def protein_bootstrap_metric(
    labels_by_protein: Sequence[np.ndarray],
    scores_by_protein: Sequence[np.ndarray],
    *,
    metric: str = "auc",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Cluster-bootstrap CI for a residue-level metric, resampling proteins.

    ``labels_by_protein`` and ``scores_by_protein`` are parallel sequences of
    per-protein arrays. Each bootstrap replicate draws len(proteins) proteins
    with replacement and pools their residues.
    """
    n_prot = len(labels_by_protein)
    if n_prot != len(scores_by_protein):
        raise ValueError("labels and scores must have the same number of proteins")
    fn = METRICS.get(metric)
    if fn is None:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")
    if n_prot < 2:
        return {"point": None, "insufficient_data": True, "n_proteins": n_prot}

    all_y = np.concatenate(list(labels_by_protein))
    all_s = np.concatenate(list(scores_by_protein))
    point = fn(all_y, all_s)

    rng = np.random.default_rng(seed)
    boots: list = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_prot, size=n_prot)
        y = np.concatenate([labels_by_protein[i] for i in idx])
        s = np.concatenate([scores_by_protein[i] for i in idx])
        boots.append(fn(y, s))

    lo, hi = _percentile_ci(boots, ci)
    usable = [b for b in boots if b is not None]
    return {
        "point": point,
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": ci,
        "n_boot": n_boot,
        "n_effective_boot": len(usable),
        "n_proteins": n_prot,
        "n_residues": int(len(all_y)),
        "se": float(np.std(usable, ddof=1)) if len(usable) > 1 else None,
        "resampling_unit": "protein",
        "insufficient_data": False,
    }


def paired_protein_bootstrap_delta(
    labels_by_protein: Sequence[np.ndarray],
    scores_a_by_protein: Sequence[np.ndarray],
    scores_b_by_protein: Sequence[np.ndarray],
    *,
    metric: str = "auc",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """CI for ``metric(a) - metric(b)`` on the same residues.

    Both methods are evaluated on each bootstrap replicate's identical protein
    sample, so the interval reflects the paired difference. ``crosses_zero``
    answers the question the go/no-go criterion actually asks: is A better than
    B, or is the observed gap within sampling noise?
    """
    n_prot = len(labels_by_protein)
    fn = METRICS.get(metric)
    if fn is None:
        raise ValueError(f"unknown metric {metric!r}")
    if n_prot < 2:
        return {"delta": None, "insufficient_data": True, "n_proteins": n_prot}

    all_y = np.concatenate(list(labels_by_protein))
    a_point = fn(all_y, np.concatenate(list(scores_a_by_protein)))
    b_point = fn(all_y, np.concatenate(list(scores_b_by_protein)))
    delta_point = (
        None if a_point is None or b_point is None else float(a_point - b_point)
    )

    rng = np.random.default_rng(seed)
    deltas: list = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_prot, size=n_prot)
        y = np.concatenate([labels_by_protein[i] for i in idx])
        a = fn(y, np.concatenate([scores_a_by_protein[i] for i in idx]))
        b = fn(y, np.concatenate([scores_b_by_protein[i] for i in idx]))
        if a is not None and b is not None:
            deltas.append(a - b)

    lo, hi = _percentile_ci(deltas, ci)
    crosses = None if lo is None or hi is None else bool(lo <= 0.0 <= hi)
    # Two-sided bootstrap p-value: how often the replicate delta falls on the
    # opposite side of zero from the point estimate.
    p_value = None
    if deltas and delta_point is not None:
        arr = np.asarray(deltas, dtype=float)
        tail = float(np.mean(arr <= 0.0) if delta_point > 0 else np.mean(arr >= 0.0))
        p_value = float(min(1.0, 2.0 * tail))

    return {
        "metric": metric,
        "a": a_point,
        "b": b_point,
        "delta": delta_point,
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": ci,
        "crosses_zero": crosses,
        "p_value_bootstrap": p_value,
        "n_boot": n_boot,
        "n_effective_boot": len(deltas),
        "n_proteins": n_prot,
        "n_residues": int(len(all_y)),
        "resampling_unit": "protein",
        "insufficient_data": False,
        "interpretation": (
            "crosses_zero=True means the observed difference is within "
            "protein-level sampling noise; do not describe it as an improvement."
        ),
    }


def summarize_ci(result: dict, name: str = "metric") -> str:
    """One-line human summary, safe on partial/failed results."""
    if result.get("insufficient_data"):
        return f"{name}: insufficient data"
    point = result.get("point", result.get("delta"))
    lo, hi = result.get("ci_low"), result.get("ci_high")
    if point is None or lo is None:
        return f"{name}: n/a"
    base = f"{name} = {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
    if result.get("crosses_zero"):
        base += "  (CI includes 0 — not distinguishable from no difference)"
    n_prot = result.get("n_proteins")
    if n_prot:
        base += f"  n={n_prot} proteins"
    return base
