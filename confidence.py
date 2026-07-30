"""Per-residue confidence for disorder prediction — calibration + conformal sets.

Most intrinsic-disorder predictors emit a single uncalibrated score per residue.
This module adds two capabilities they typically lack:

1. **Probability calibration** (isotonic) + **Expected Calibration Error (ECE)**,
   so a reported p(disorder)=0.8 actually means ~80% of such residues are disordered.
   Calibration is monotonic, so it does not change ranking metrics (AUC/AP).

2. **Split-conformal prediction sets** with a finite-sample coverage guarantee.
   For a user risk level ``alpha`` (e.g. 0.1 → 90% coverage), each residue gets a
   prediction *set* over {ordered, disordered}:
       {disordered}         -> confident disorder
       {ordered}            -> confident order
       {ordered,disordered} -> abstain (uncertain)
   The true label is contained in the set with probability >= 1 - alpha
   (Vovk et al.; standard split-conformal). Class-conditional (Mondrian) variants
   control coverage within each class separately, which matters for the imbalanced
   disorder setting.

All functions are pure and dependency-light (numpy + scikit-learn isotonic).
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """Binned ECE: sum_b (n_b/N) * |acc_b - conf_b|."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"y_true and y_prob must have the same length, got {y_true.shape[0]} and {y_prob.shape[0]}"
        )
    n = y_true.shape[0]
    if n == 0:
        return 0.0
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    # (n_b/N)*|acc_b - conf_b| == |sum(y_true_b) - sum(y_prob_b)| / N, so the
    # whole binned sum is two bincounts — no per-bin masking pass, and empty
    # bins contribute zero on their own.
    conf_sum = np.bincount(idx, weights=y_prob, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=y_true, minlength=n_bins)
    return float(np.abs(acc_sum - conf_sum).sum() / n)


def fit_calibrator(cal_prob, cal_true) -> IsotonicRegression:
    """Fit an isotonic calibrator on held-out (out-of-fold) predictions."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(cal_prob, dtype=np.float64), np.asarray(cal_true, dtype=np.float64))
    return iso


def apply_calibrator(iso: IsotonicRegression, prob) -> np.ndarray:
    return np.asarray(iso.predict(np.asarray(prob, dtype=np.float64)), dtype=np.float32)


# ---------------------------------------------------------------------------
# Conformal prediction sets
# ---------------------------------------------------------------------------
def conformal_quantile(cal_prob, cal_true, alpha: float, class_conditional: bool = True):
    """Compute the conformal nonconformity threshold(s).

    Nonconformity for a residue with true class c is ``1 - s_c`` where
    ``s_disorder = p`` and ``s_order = 1 - p``. Returns a threshold ``q`` such that
    a class is included in the prediction set iff its score ``s_c >= 1 - q``.

    class_conditional=True returns a dict {0: q_order, 1: q_disorder} (Mondrian).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    cal_prob = np.asarray(cal_prob, dtype=np.float64)
    cal_true = np.asarray(cal_true, dtype=np.int64)
    s_true = np.where(cal_true == 1, cal_prob, 1.0 - cal_prob)
    nonconf = 1.0 - s_true

    def _q(scores):
        m = len(scores)
        if m == 0:
            return 1.0
        # finite-sample corrected quantile level
        level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
        return float(np.quantile(scores, level, method="higher"))

    if class_conditional:
        return {
            0: _q(nonconf[cal_true == 0]),
            1: _q(nonconf[cal_true == 1]),
        }
    return _q(nonconf)


def conformal_sets(prob, q) -> dict:
    """Build prediction sets given probabilities and threshold(s) from
    :func:`conformal_quantile`.

    Returns a dict with boolean arrays ``has_order`` / ``has_disorder`` and derived
    ``decision`` codes: 1=disorder, 0=order, -1=abstain(both), 2=empty(neither).
    """
    prob = np.asarray(prob, dtype=np.float64)
    if isinstance(q, dict):
        q_ord, q_dis = q[0], q[1]
    else:
        q_ord = q_dis = q
    has_disorder = prob >= (1.0 - q_dis)
    has_order = (1.0 - prob) >= (1.0 - q_ord)

    decision = np.full(len(prob), -1, dtype=np.int64)  # both -> abstain
    decision[has_disorder & ~has_order] = 1
    decision[has_order & ~has_disorder] = 0
    decision[~has_order & ~has_disorder] = 2  # empty (rare)
    return {"has_order": has_order, "has_disorder": has_disorder, "decision": decision}


def conformal_report(prob, true, q) -> dict:
    """Empirical coverage / abstention diagnostics for a conformal configuration."""
    true = np.asarray(true, dtype=np.int64)
    sets = conformal_sets(prob, q)
    has_order, has_disorder = sets["has_order"], sets["has_disorder"]
    decision = sets["decision"]

    covered = np.where(true == 1, has_disorder, has_order)
    singleton = np.isin(decision, (0, 1))

    out = {
        "coverage": float(covered.mean()),
        "abstain_rate": float((decision == -1).mean()),
        "empty_rate": float((decision == 2).mean()),
        "confident_rate": float(singleton.mean()),
        "avg_set_size": float((has_order.astype(int) + has_disorder.astype(int)).mean()),
    }
    if singleton.any():
        pred = decision[singleton]
        out["selective_accuracy"] = float((pred == true[singleton]).mean())
    forced = forced_prediction(prob)
    out["forced_accuracy"] = float((forced["y_hat"] == true).mean())
    out["mean_confidence_pct"] = float(forced["confidence_pct"].mean())
    for c, name in ((1, "disorder"), (0, "order")):
        m = true == c
        if m.any():
            cov_c = (has_disorder[m] if c == 1 else has_order[m]).mean()
            out[f"coverage_{name}"] = float(cov_c)
    return out


# ---------------------------------------------------------------------------
# Forced / always-on confidence (honest best-guess + %)
# ---------------------------------------------------------------------------
def forced_prediction(prob) -> dict:
    """Always emit a best-guess label + percent confidence (no abstain).

    This does **not** weaken the conformal guarantee: the selective ``decision``
    gate remains available separately. Forced mode answers "if I must call every
    residue, what is my best calibrated guess and how sure am I?"

    ``confidence_pct`` is ``100 * max(p, 1-p)`` on the calibrated probability —
    so 50% means coin-flip and 95% means the calibrated model is very sure.
    """
    prob = np.asarray(prob, dtype=np.float64)
    y_hat = (prob >= 0.5).astype(np.int64)
    conf = np.maximum(prob, 1.0 - prob)
    return {
        "y_hat": y_hat,
        "y_hat_labels": np.where(y_hat == 1, "disorder", "order"),
        "confidence_pct": (100.0 * conf).astype(np.float32),
        "p_calibrated": prob.astype(np.float32),
    }


def annotate_confidence(prob, q) -> dict:
    """Combine selective conformal decisions with forced best-guess confidence."""
    prob = np.asarray(prob, dtype=np.float64)
    sets = conformal_sets(prob, q)
    forced = forced_prediction(prob)
    decision = sets["decision"]
    # On confident singleton calls, confidence_pct still reflects calibrated mass
    # on the predicted class; on abstain, it is the best-guess margin.
    return {
        "p_calibrated": forced["p_calibrated"],
        "decision": decision,
        "has_order": sets["has_order"],
        "has_disorder": sets["has_disorder"],
        "y_hat_forced": forced["y_hat"],
        "confidence_pct": forced["confidence_pct"],
        "mode": "selective+forced",
    }
