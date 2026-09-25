"""Distribution-free coverage for disorder prediction: which residues are callable.

Every disorder predictor in CAID reports a ranking statistic — AUC, APS, MCC.
None of them tells a biologist the thing a biologist needs: *for this residue,
can I act on this call, and how often will that be wrong?* A probability of
0.62 is not an answer to that unless the probabilities are calibrated, and
`Calibration.risk_decomposition` in the accompanying Lean development is exact
about what calibration buys and what it cannot:

    risk = calError + resolution

A perfectly calibrated model has paid the first term and nothing else; its
remaining error is the contextual variation its internal code conflated, and
**no recalibration can touch it** (`risk_eq_resolution_of_calibrated`). So
calibration is necessary and is provably not sufficient.

Split conformal prediction gives the missing guarantee, and gives it without
assuming anything about the model, the data distribution, or the calibration of
the probabilities. Given a calibration sample exchangeable with the test
sample, the prediction sets satisfy

    P(Y_test in C(X_test)) >= 1 - alpha

in finite samples, exactly. The model can be arbitrarily bad; the guarantee
holds. What a bad model costs is *informativeness*, not validity — which is the
same law `PredictionSets.validity_is_free` states for conformational ensembles:
the whole library covers at every level, so the only content of a prediction set
is its size.

Here the label space is {ordered, disordered}, so a prediction set is one of

    {ordered}      a call, with the guarantee in force
    {disordered}   a call, with the guarantee in force
    {}             **no label is plausible at this level** — the residue is
                   atypical under the model, and this is the abstention
    {both}         neither label can be ruled out

Which of the two uninformative outcomes occurs is decided by the threshold, and
it is worth being exact because it is easy to state backwards. The score is
`1 - p_y`. A class `c` enters the set when `1 - p_c <= q`. With `q < 0.5` no
residue can admit both classes, since that needs `p_1 >= 1-q > 0.5` and
`p_0 >= 1-q > 0.5` at once; the uninformative outcome is then the **empty
set**, and it occupies the band `q < p_1 < 1-q` — the model's uncertain middle.
Only at `q >= 0.5`, a badly-performing model or a very high confidence level,
does `{both}` appear.

So the quantity worth reporting is the **singleton rate**: the fraction of
residues that receive a definite call at all with the guarantee in force. That
number, not AUC, says how much of a proteome is decidable. Empty sets count as
misses against the coverage guarantee, as they must.

Two variants, both implemented, because the difference matters at 27%
prevalence:

**Marginal** — one threshold, coverage guaranteed averaged over residues. Cheap
and weak: with a skewed prevalence a set that almost always contains the
majority class satisfies it while saying nothing about the minority.

**Class-conditional (Mondrian)** — a separate threshold per true class, so
coverage is guaranteed *within* ordered and *within* disordered residues
separately. This is the one to report for disorder, where the minority class is
the one anybody cares about.

Exchangeability is the only assumption, and it is not free here: residues within
a chain are not exchangeable with each other. Calibration and test are therefore
split **by chain**, never by residue, and the realised coverage is measured on
the test half and reported beside the nominal level. A gap between them is
evidence the assumption is strained, and it is shown rather than hidden.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample corrected quantile, ceil((n+1)(1-alpha))/n.

    The `(n+1)` is what makes the guarantee exact rather than asymptotic. With
    the plain empirical quantile the coverage is `1-alpha` only in the limit;
    with this correction it is at least `1-alpha` for every `n`, which is the
    whole reason to prefer conformal over a held-out threshold.

    Returns `+inf` when the sample is too small for the level to be achievable
    — with `n` calibration points no level above `1 - 1/(n+1)` can be
    guaranteed, and returning infinity makes every set the full label space,
    which is the correct (vacuous, valid) answer rather than a silent failure.
    """
    n = int(scores.size)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def calibrate(probs: np.ndarray, labels: np.ndarray, alpha: float,
              class_conditional: bool = True) -> dict:
    """Thresholds from a calibration sample.

    The conformity score is `1 - p_y`, the model's probability *of the true
    label*. Low means confident and correct; high means the truth was
    surprising.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int8)
    p_true = np.where(labels == 1, probs, 1.0 - probs)
    scores = 1.0 - p_true

    out = {"alpha": alpha, "n_calibration": int(labels.size),
           "class_conditional": bool(class_conditional)}
    if class_conditional:
        out["q"] = {
            int(c): conformal_quantile(scores[labels == c], alpha)
            for c in (0, 1)
        }
        out["n_per_class"] = {int(c): int((labels == c).sum()) for c in (0, 1)}
    else:
        out["q"] = {0: conformal_quantile(scores, alpha),
                    1: conformal_quantile(scores, alpha)}
        out["n_per_class"] = {int(c): int((labels == c).sum()) for c in (0, 1)}
    return out


def prediction_sets(probs: np.ndarray, cal: dict) -> np.ndarray:
    """(n, 2) boolean membership: column c is True when class c is in the set."""
    probs = np.asarray(probs, dtype=np.float64)
    q0, q1 = cal["q"][0], cal["q"][1]
    # class c is included iff 1 - p_c <= q_c
    return np.stack([(1.0 - (1.0 - probs)) <= q0,
                     (1.0 - probs) <= q1], axis=1)


def evaluate_sets(sets: np.ndarray, labels: np.ndarray) -> dict:
    """Realised coverage and informativeness.

    Coverage is reported marginally *and* per class, because a marginal number
    at 27% prevalence is dominated by the majority and can look excellent while
    the minority class is never covered.
    """
    labels = np.asarray(labels).astype(np.int8)
    covered = sets[np.arange(labels.size), labels]
    size = sets.sum(axis=1)
    out = {
        "n": int(labels.size),
        "coverage": float(covered.mean()),
        "singleton_rate": float((size == 1).mean()),
        "undetermined_rate": float((size == 2).mean()),
        "empty_rate": float((size == 0).mean()),
        "mean_set_size": float(size.mean()),
    }
    for c in (0, 1):
        m = labels == c
        if m.any():
            out[f"coverage_class{c}"] = float(covered[m].mean())
            out[f"singleton_rate_class{c}"] = float((size[m] == 1).mean())
            out[f"n_class{c}"] = int(m.sum())
    # Among residues that got a call, how often was the call right? Not a
    # guaranteed quantity — reported because it is what a reader will want and
    # would otherwise compute wrongly from the coverage.
    single = size == 1
    if single.any():
        called = sets[single].argmax(axis=1)
        out["singleton_accuracy"] = float((called == labels[single]).mean())
    return out


def split_by_chain(chain_ids, rng, fraction: float = 0.5):
    """Split chains, never residues.

    Residues within a chain are not exchangeable with each other — they share a
    protein, a fold, a construct and an experiment — so a residue-level split
    would inflate the calibration sample's agreement with the test sample and
    the guarantee would be about a population that does not exist.
    """
    uniq = np.array(sorted(set(chain_ids)))
    perm = rng.permutation(len(uniq))
    n_cal = max(1, int(round(fraction * len(uniq))))
    cal_ids = set(uniq[perm[:n_cal]].tolist())
    mask = np.array([c in cal_ids for c in chain_ids], dtype=bool)
    return mask, ~mask


# ── Conformal risk control, at the level where exchangeability actually holds ──
#
# The split-conformal guarantee above is per residue, and residues are not the
# exchangeable unit: they share a chain, a fold, a construct and an experiment.
# Splitting by chain is the right thing to do and it does *not* rescue the
# per-residue guarantee — it makes calibration and test two random halves of a
# set of chains, so what is exchangeable is the chain.
#
# Measured on the temporal holdout, the shortfall is real: at a 90% target the
# realised per-residue coverage is 0.876 for one model and, on disordered
# residues alone, 0.527 for AlphaFold-rsa. Reporting those as guarantees would
# be false.
#
# Conformal risk control (Angelopoulos, Bates, Fisch, Lei, Schuster 2023) gives
# a valid guarantee at the level where the assumption holds. Take any per-chain
# loss that is bounded and monotone in a threshold, and choose the threshold by
#
#     lambda_hat = inf { t : (n/(n+1)) * mean_i L_i(t) + B/(n+1) <= alpha }
#
# Then E[L_test(lambda_hat)] <= alpha over a fresh chain. The expectation is
# over chains, which is exactly the object the split respects.


def chain_miss_rate(prob: np.ndarray, labels: np.ndarray,
                    threshold: float) -> float:
    """Fraction of a chain's disordered residues the call at ``threshold``
    misses. Bounded in [0, 1] and non-increasing as the threshold falls, which
    is what conformal risk control requires."""
    pos = np.asarray(labels).astype(np.int8) == 1
    if not pos.any():
        return 0.0
    return float((np.asarray(prob)[pos] < threshold).mean())


def control_chain_risk(probs_by_chain, labels_by_chain, alpha: float,
                       grid: np.ndarray | None = None) -> dict:
    """The largest threshold whose per-chain expected miss rate is under alpha.

    Returns the threshold and the calibration curve. A threshold of 0 calls
    every residue disordered and has loss 0, so the search always succeeds;
    what varies is how much of the chain that costs, which is reported as the
    call rate rather than assumed away.
    """
    if grid is None:
        grid = np.linspace(1.0, 0.0, 501)
    n = len(labels_by_chain)
    if n == 0:
        return {"threshold": 0.0, "reason": "no calibration chains"}
    # The bound is (n/(n+1))*mean + 1/(n+1) and the mean is non-negative, so
    # no risk below 1/(n+1) can be certified at all, whatever the model does.
    # On CAID3 Linker that is 1/16 = 0.0625 with 31 targets split in half, so
    # an alpha of 0.05 is unachievable by *sample size* — and the search
    # exhausts its grid and returns "flag everything", which reads as a fact
    # about the methods when it is a fact about n. Say so instead.
    if alpha < 1.0 / (n + 1.0):
        return {"threshold": 0.0, "alpha": alpha, "n_chains": n,
                "achievable": False,
                "min_achievable_alpha": 1.0 / (n + 1.0),
                "reason": (f"alpha={alpha} is below 1/(n+1)={1.0/(n+1.0):.4f} "
                           f"with {n} calibration chains; no method can "
                           f"certify it")}
    curve = []
    chosen = 0.0
    for t in grid:                       # descending: loss is non-increasing
        losses = [chain_miss_rate(p, y, t)
                  for p, y in zip(probs_by_chain, labels_by_chain)]
        bound = (n / (n + 1.0)) * float(np.mean(losses)) + 1.0 / (n + 1.0)
        curve.append((float(t), float(np.mean(losses)), float(bound)))
        if bound <= alpha:
            chosen = float(t)
            break
    return {"threshold": chosen, "alpha": alpha, "n_chains": n,
            "achievable": True, "curve": curve[-5:]}


def evaluate_chain_risk(probs_by_chain, labels_by_chain,
                        threshold: float) -> dict:
    """Realised per-chain miss rate and call rate at a chosen threshold."""
    losses, called, n_pos = [], [], 0
    for p, y in zip(probs_by_chain, labels_by_chain):
        losses.append(chain_miss_rate(p, y, threshold))
        called.append(float((np.asarray(p) >= threshold).mean()))
        n_pos += int((np.asarray(y) == 1).sum())
    return {
        "threshold": threshold,
        "mean_chain_miss_rate": float(np.mean(losses)) if losses else None,
        "median_chain_miss_rate": float(np.median(losses)) if losses else None,
        "mean_fraction_called_disordered": float(np.mean(called))
        if called else None,
        "n_chains": len(losses),
        "n_disordered_residues": n_pos,
    }


# ── Two ways to spend a guarantee, and the gap between them ───────────────────
#
# `control_chain_risk` picks one global threshold. That makes the operating cost
# **calibration-sensitive**: a per-protein recalibration moves residues across a
# global cut, so a method whose advantage is protein-level calibration keeps
# that advantage here.
#
# The alternative is to flag a fixed *quantile* of each protein — the top q
# fraction of that chain's own scores. That is invariant under any per-protein
# strictly monotone recalibration, exactly as `AUC_within` is
# (`auc_within_strictMono_invariant`), because it depends only on the ordering
# inside each chain.
#
# So the two costs bracket the same guarantee from the two sides the AUC
# decomposition already separates:
#
#     global threshold      cost uses discrimination AND calibration
#     per-protein quantile  cost uses discrimination ALONE
#
# and their difference is what calibration is worth operationally. A method
# whose leaderboard position comes from protein-level calibration — the CAID3
# winners that place 21st to 25th on the within-protein axis — should pay much
# more under the per-protein rule than under the global one. That is a
# prediction, and it is testable on the published field.


def chain_quantile_miss_rate(prob: np.ndarray, labels: np.ndarray,
                             q: float) -> float:
    """Miss rate when the top ``q`` fraction of *this chain's own* scores is
    flagged. Invariant under any strictly increasing per-chain recalibration,
    since it depends only on the within-chain ordering."""
    labels = np.asarray(labels).astype(np.int8)
    pos = labels == 1
    if not pos.any():
        return 0.0
    prob = np.asarray(prob, dtype=np.float64)
    n = prob.size
    k = int(np.ceil(q * n))
    if k <= 0:
        return 1.0
    if k >= n:
        return 0.0
    # rank 0 = highest score; flag ranks < k
    order = np.argsort(np.argsort(-prob, kind="stable"), kind="stable")
    return float((order[pos] >= k).mean())


def control_chain_risk_quantile(probs_by_chain, labels_by_chain, alpha: float,
                                grid: np.ndarray | None = None) -> dict:
    """The smallest per-chain flagged fraction whose expected miss rate is
    under alpha. Same conformal risk control bound, different knob."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 501)
    n = len(labels_by_chain)
    if n == 0:
        return {"q": 1.0, "reason": "no calibration chains"}
    if alpha < 1.0 / (n + 1.0):
        return {"q": 1.0, "alpha": alpha, "n_chains": n, "achievable": False,
                "min_achievable_alpha": 1.0 / (n + 1.0),
                "reason": (f"alpha={alpha} is below 1/(n+1)={1.0/(n+1.0):.4f} "
                           f"with {n} calibration chains")}
    for q in grid:                       # ascending: loss is non-increasing
        losses = [chain_quantile_miss_rate(p, y, q)
                  for p, y in zip(probs_by_chain, labels_by_chain)]
        bound = (n / (n + 1.0)) * float(np.mean(losses)) + 1.0 / (n + 1.0)
        if bound <= alpha:
            return {"q": float(q), "alpha": alpha, "n_chains": n,
                    "achievable": True}
    return {"q": 1.0, "alpha": alpha, "n_chains": n, "achievable": False,
            "reason": "grid exhausted"}


def evaluate_chain_risk_quantile(probs_by_chain, labels_by_chain,
                                 q: float) -> dict:
    losses = [chain_quantile_miss_rate(p, y, q)
              for p, y in zip(probs_by_chain, labels_by_chain)]
    return {
        "q": q,
        "mean_chain_miss_rate": float(np.mean(losses)) if losses else None,
        "mean_fraction_called_disordered": float(q),
        "n_chains": len(losses),
    }
