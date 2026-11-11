"""Fuse the model with AlphaFold structural signal, without touching the benchmark.

The measurement that motivates this, on 304 CAID3 Disorder-PDB targets:

    DisorderNet-Lite (model)   AUC 0.9215   APS 0.8567
    AlphaFold -pLDDT           AUC 0.9431   APS 0.9062
    AlphaFold rsa (w=21)       AUC 0.9459   APS 0.9168

Two free structural baselines beat the trained model. A grid search over fusion
weights reached AUC 0.9633 — above the CAID3 leader (PUNCH2, 0.9550) — but
those weights were chosen on the residues being scored, so that number is an
upper bound and not a result.

This module exists to make the same combination honestly. Weights are fit on
**out-of-fold training predictions** and then frozen; the benchmark contributes
nothing to them. That is the difference between "we beat SOTA" and "we appear
to", and it is the only version worth reporting.

What must be reported alongside any fused score
-----------------------------------------------
``rsa + pLDDT`` with no model at all reaches 0.9584 on that same data. A fused
result is only interesting to the extent it beats *that*, not to the extent it
beats the model alone — so :func:`fusion_report` always carries the
structure-only baseline next to the fused figure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Fitting on standardised inputs keeps the learned weights interpretable as
# relative contributions rather than as an artefact of each channel's scale.
EPS = 1e-9


def _standardise(x: np.ndarray, stats: Optional[tuple[float, float]] = None):
    x = np.asarray(x, dtype=np.float64)
    if stats is None:
        mean = float(np.nanmean(x)) if x.size else 0.0
        sd = float(np.nanstd(x)) if x.size else 1.0
        stats = (mean, sd if sd > EPS else 1.0)
    mean, sd = stats
    return (x - mean) / sd, stats


def fit_fusion(
    labels: np.ndarray,
    model: np.ndarray,
    rsa: Optional[np.ndarray] = None,
    plddt: Optional[np.ndarray] = None,
    l2: float = 1.0,
) -> dict:
    """Fit fusion weights on training out-of-fold predictions.

    ``labels``/``model`` are the pooled OOF arrays from cross-validation;
    ``rsa``/``plddt`` are the structural channels for the same residues.
    Logistic regression on standardised inputs, so the coefficients read as
    relative contributions.

    The returned dict is everything needed to apply the fusion later, including
    the standardisation statistics — recomputing those at apply time would leak
    the target distribution into the transform.
    """
    from sklearn.linear_model import LogisticRegression

    labels = np.asarray(labels)
    channels: list[str] = ["model"]
    cols = [np.asarray(model, dtype=np.float64)]
    if rsa is not None:
        channels.append("rsa")
        cols.append(np.asarray(rsa, dtype=np.float64))
    if plddt is not None:
        channels.append("neg_plddt")
        cols.append(-np.asarray(plddt, dtype=np.float64))

    stats: list[tuple[float, float]] = []
    std_cols = []
    for col in cols:
        z, s = _standardise(col)
        std_cols.append(z)
        stats.append(s)
    X = np.column_stack(std_cols)

    finite = np.isfinite(X).all(axis=1) & np.isfinite(labels)
    X, y = X[finite], labels[finite]
    if len(np.unique(y)) < 2:
        raise ValueError("fusion fit needs both classes in the training labels")

    clf = LogisticRegression(C=1.0 / max(l2, EPS), max_iter=2000)
    clf.fit(X, y)
    return {
        "channels": channels,
        "coefficients": [float(c) for c in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "standardisation": [[float(m), float(s)] for m, s in stats],
        "n_train_residues": int(len(y)),
        "fitted_on": "training out-of-fold predictions",
        "note": (
            "Weights come from training data only. Applying them to a benchmark "
            "uses no information from it, which is what separates a reportable "
            "fused score from a grid search run on the scored residues."
        ),
    }


def apply_fusion(
    fusion: dict,
    model: np.ndarray,
    rsa: Optional[np.ndarray] = None,
    plddt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply frozen fusion weights. Returns a probability per residue."""
    available = {"model": np.asarray(model, dtype=np.float64)}
    if rsa is not None:
        available["rsa"] = np.asarray(rsa, dtype=np.float64)
    if plddt is not None:
        available["neg_plddt"] = -np.asarray(plddt, dtype=np.float64)

    missing = [c for c in fusion["channels"] if c not in available]
    if missing:
        raise ValueError(
            f"fusion was fit with channels {fusion['channels']} but "
            f"{missing} were not supplied; refusing to score a different model "
            "than the one whose weights these are"
        )

    z = fusion["intercept"]
    for name, coef, (mean, sd) in zip(
        fusion["channels"], fusion["coefficients"], fusion["standardisation"]
    ):
        z = z + coef * ((available[name] - mean) / (sd if sd > EPS else 1.0))
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def fusion_report(
    labels: np.ndarray,
    model: np.ndarray,
    fused: np.ndarray,
    rsa: Optional[np.ndarray] = None,
    plddt: Optional[np.ndarray] = None,
    structure_only: Optional[np.ndarray] = None,
) -> dict:
    """Score a fusion against the baselines that make it interpretable.

    A fused number means little on its own. What matters is whether it beats
    the structure-only combination, because rsa and pLDDT are free — no
    training, no GPU — and on CAID3 they reach 0.9584 together.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    def score(name: str, s: Optional[np.ndarray]) -> Optional[dict]:
        if s is None:
            return None
        s = np.asarray(s, dtype=np.float64)
        ok = np.isfinite(s) & np.isfinite(labels)
        if ok.sum() < 10 or len(np.unique(labels[ok])) < 2:
            return None
        return {
            "name": name,
            "auc": float(roc_auc_score(labels[ok], s[ok])),
            "aps": float(average_precision_score(labels[ok], s[ok])),
            "n_residues": int(ok.sum()),
        }

    labels = np.asarray(labels)
    out = {
        "model_only": score("model", model),
        "rsa_only": score("rsa", rsa),
        "neg_plddt_only": score("-pLDDT", None if plddt is None else -np.asarray(plddt)),
        "structure_only": score("rsa+pLDDT", structure_only),
        "fused": score("fused", fused),
    }

    fused_s, struct_s = out.get("fused"), out.get("structure_only")
    if fused_s and struct_s:
        out["gain_over_structure_only"] = {
            "auc": round(fused_s["auc"] - struct_s["auc"], 4),
            "aps": round(fused_s["aps"] - struct_s["aps"], 4),
        }
        out["model_earns_its_place"] = fused_s["auc"] > struct_s["auc"]
    if fused_s and out.get("model_only"):
        out["gain_over_model_only"] = {
            "auc": round(fused_s["auc"] - out["model_only"]["auc"], 4),
            "aps": round(fused_s["aps"] - out["model_only"]["aps"], 4),
        }
    out["interpretation"] = (
        "A fused score is only evidence for the model if it beats the "
        "structure-only baseline: rsa and pLDDT require no training and reach "
        "AUC 0.9584 together on CAID3 Disorder-PDB."
    )
    return out
