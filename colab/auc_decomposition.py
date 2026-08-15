"""Split a pooled AUC into its within-protein and between-protein parts.

CAID scores by pooling every residue of every protein into one AUC. That number
is the probability that a randomly drawn positive residue outranks a randomly
drawn negative one, and every such pair is either **within** one protein or
**between** two. The metric therefore decomposes exactly:

    AUC_pooled = w_within * AUC_within + w_between * AUC_between

with weights equal to the fraction of positive-negative pairs of each kind:

    P_within  = sum_k  pos_k * neg_k
    P_between = sum_k  pos_k * (N_neg - neg_k)        counted once per pair

This is not an approximation and needs no resampling. It is an identity about
how the Mann-Whitney statistic counts pairs, and it separates two abilities that
a single AUC conflates:

**Within-protein** — given this chain, which residues bind? A local question,
and the one a per-residue model with a bounded receptive field is built for.

**Between-protein** — does this chain have more binding than that one? A global
question, and one a purely local model has no mechanism to answer: a 213-residue
receptive field over a 1,000-residue protein never sees the protein.

**The decomposition itself is elementary, and saying otherwise would oversell
it.** Partitioning a Mann-Whitney pair count by whether the pair is within or
between groups is textbook, and the weight has an obvious closed form: for n
proteins of roughly equal size only about 1/n of pairs fall inside one, which is
exactly what CAID3 shows — 0.0029 against 1/319, 0.0326 against 1/31. "97% of
the metric is between-protein" is therefore close to a restatement of "there are
many proteins", not a discovery.

What is not arithmetic is the empirical divergence it exposes. On Binding our
within-protein AUC is 0.8683 against the leader's 0.8049 while our pooled score
is *lower*; on Binding-IDR our within-protein ability matches the leader and the
entire 0.14 pooled deficit is between-protein. Two methods can be ordered one
way on the question the benchmark is understood to ask and the other way on the
number it reports. That is a fact about these predictors, not about pair
counting, and it is what makes the split worth computing.

The decomposition was written to size a specific gap. On Binding-IDR our
predictions win 21 targets and lose 21 against a leader ahead of us by 0.14
pooled, and per-protein rank normalisation — which destroys between-protein
information — *raises* our score by 0.074 while lowering theirs by 0.059. That
says the deficit is entirely between-protein. This measures it rather than
inferring it.
"""

from __future__ import annotations

import numpy as np


def _rank_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """Mann-Whitney AUC via ranks, ties averaged."""
    from scipy.stats import rankdata

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def decompose_auc(labels_by_target: list[np.ndarray],
                  scores_by_target: list[np.ndarray]) -> dict:
    """Exact within/between split of the pooled AUC.

    ``AUC_within`` is the pair-weighted mean of per-protein AUCs — weighted by
    each protein's own pair count, not by protein, so a chain contributing one
    pair does not count as much as one contributing ten thousand.

    ``AUC_between`` is recovered from the identity rather than computed by
    enumerating cross-protein pairs, which would be quadratic in the residue
    count. The identity makes that unnecessary and exact.
    """
    y_all = np.concatenate(labels_by_target)
    s_all = np.concatenate(scores_by_target)
    pooled = _rank_auc(y_all, s_all)
    if pooled is None:
        return {"pooled": None, "reason": "one class overall"}

    n_pos_total = int(y_all.sum())
    n_neg_total = int(len(y_all) - n_pos_total)
    total_pairs = n_pos_total * n_neg_total

    within_pairs = 0
    within_weighted = 0.0
    usable = 0
    for y, s in zip(labels_by_target, scores_by_target):
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        pairs = n_pos * n_neg
        if pairs == 0:
            continue
        a = _rank_auc(y, s)
        if a is None:
            continue
        usable += 1
        within_pairs += pairs
        within_weighted += a * pairs

    if within_pairs == 0:
        return {"pooled": pooled, "reason": "no protein has both classes"}

    auc_within = within_weighted / within_pairs
    between_pairs = total_pairs - within_pairs
    w_within = within_pairs / total_pairs
    w_between = between_pairs / total_pairs
    # From the identity. Exact, and far cheaper than enumerating the pairs.
    auc_between = ((pooled - w_within * auc_within) / w_between
                   if between_pairs else None)

    return {
        "pooled": pooled,
        "auc_within": auc_within,
        "auc_between": auc_between,
        "w_within": w_within,
        "w_between": w_between,
        "within_pairs": int(within_pairs),
        "between_pairs": int(between_pairs),
        "n_targets_with_both_classes": usable,
    }


def compare_decompositions(ours: dict, theirs: dict) -> dict:
    """Attribute a pooled difference to its within and between parts.

    Only valid when both were computed on the same targets, so the weights
    match; otherwise the two decompositions are of different quantities and the
    attribution is meaningless. That is checked rather than assumed.
    """
    if ours.get("pooled") is None or theirs.get("pooled") is None:
        return {"reason": "a decomposition is unavailable"}
    if abs(ours["w_within"] - theirs["w_within"]) > 1e-12:
        raise ValueError(
            f"pair weights differ ({ours['w_within']} vs {theirs['w_within']}); "
            f"the decompositions describe different target sets and their "
            f"difference cannot be attributed")
    w_in, w_bt = ours["w_within"], ours["w_between"]
    d_within = ours["auc_within"] - theirs["auc_within"]
    d_between = ours["auc_between"] - theirs["auc_between"]
    return {
        "delta_pooled": ours["pooled"] - theirs["pooled"],
        "delta_within": d_within,
        "delta_between": d_between,
        "contribution_within": w_in * d_within,
        "contribution_between": w_bt * d_between,
    }
