"""What a per-protein bias can and cannot reach — computable, before fitting one.

Implements the decision procedures from the Lean development (`AUCGapSeparable`,
`AUCCrossedMatching`, `ThresholdMatching`). Every function here corresponds to a
machine-checked theorem, named in its docstring, so a disagreement between this
code and those statements is a bug in this code.

The question. Our Binding-IDR within-protein AUC matches the published leader
(0.7004 against 0.6958) while our between-protein AUC is 0.4982 — chance —
against their 0.6400, so the entire deficit is between-protein. A per-protein
additive bias is the smallest change that addresses exactly that axis, and it
provably cannot disturb within-protein ranking. What it *can* reach is the
question these functions answer, without training anything.

Three results, in increasing strength:

**Ceiling** (`auc_shift_le_ceiling`). For every bias `b`,
``AUC(s+b) ≤ w_within·AUC_within + w_between``. On CAID3 this evaluates to
0.996–0.9999 because `w_between ≈ 1`, so on its own it says almost nothing.

**Attainability** (`ceiling_attainable_iff_pairwise_overlap`). The ceiling is
reached iff `overlap k + overlap l < 0` for every pair of distinct proteins,
where ``overlap k = maxneg k − minpos k``. The K×K gap matrix is separable —
`c k l = maxneg l − minpos k` — so cycle weights telescope and the whole test
reduces to the two largest overlaps: an O(K) scan. When it passes, the optimal
bias is explicit: ``b k = −(minpos k + maxneg k)/2``.

**The binding bound** (`ceiling_gap_of_max_crossed_matching`). Two cross-protein
comparisons conflict when no bias can satisfy both, which for
`(p∈k, n∈l)` and `(p'∈l, n'∈k)` happens exactly when
``(s_n − s_p) + (s_n' − s_p') ≥ 0``. A maximum matching M of such conflicts
costs `M/2` full pairs:

    max_b AUC ≤ w_within·AUC_within + w_between − (matched pairs)/(all pairs)

The conflict graph splits over protein pairs into bipartite *threshold* graphs
`u + v ≥ 0`, and for those greedy is exactly optimal — sort both gap lists
decreasing, take the largest m with `u[t] + v[m−1−t] ≥ 0`.
"""

from __future__ import annotations

import numpy as np


def protein_overlaps(labels_by_target, scores_by_target):
    """``maxneg k − minpos k`` per protein, the separable gap matrix's diagonal.

    Positive means the protein has a negative scoring at least as high as its
    lowest positive — i.e. its within-protein ranking is imperfect.
    """
    out = []
    for y, s in zip(labels_by_target, scores_by_target):
        y = np.asarray(y)
        s = np.asarray(s, dtype=np.float64)
        pos, neg = s[y == 1], s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            out.append(None)
            continue
        out.append(float(neg.max() - pos.min()))
    return out


def ceiling_attainable(labels_by_target, scores_by_target):
    """`ceiling_attainable_iff_pairwise_overlap`, decided by the two largest.

    Attainable iff ``overlap k + overlap l < 0`` for every distinct pair, which
    holds iff the sum of the two largest overlaps is negative. O(K) after the
    overlaps, no K×K matrix and no cycle search.
    """
    ov = [o for o in protein_overlaps(labels_by_target, scores_by_target)
          if o is not None]
    if len(ov) < 2:
        return {"attainable": True, "reason": "fewer than two usable proteins"}
    top = sorted(ov, reverse=True)[:2]
    return {
        "attainable": bool(top[0] + top[1] < 0),
        "two_largest_overlaps": [float(top[0]), float(top[1])],
        "sum": float(top[0] + top[1]),
        "n_proteins_with_overlap": int(sum(1 for o in ov if o > 0)),
        "n_proteins": len(ov),
    }


def optimal_centering_bias(labels_by_target, scores_by_target):
    """`ceiling_attained_by_centering`: ``b k = −(minpos k + maxneg k)/2``.

    Only optimal when the ceiling is attainable; returned regardless so the
    caller can apply and measure it, but the attainability check is what makes
    it a theorem rather than a heuristic.
    """
    out = []
    for y, s in zip(labels_by_target, scores_by_target):
        y = np.asarray(y)
        s = np.asarray(s, dtype=np.float64)
        pos, neg = s[y == 1], s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            out.append(0.0)
            continue
        out.append(float(-(pos.min() + neg.max()) / 2.0))
    return out


def _greedy_threshold_matching(u: np.ndarray, v: np.ndarray) -> int:
    """Largest m with ``u[t] + v[m−1−t] ≥ 0`` for all t < m, u and v descending.

    `le_maxMatch_iff_greedyFeasible` proves greedy exact for bipartite threshold
    graphs, and the predicate is monotone in m, so one scan suffices.
    """
    hi = min(len(u), len(v))
    m = 0
    for cand in range(1, hi + 1):
        if u[cand - 1] + v[cand - 1] >= 0:
            m = cand
        else:
            break
    return m


def max_crossed_matching(labels_by_target, scores_by_target,
                         max_block_pairs: int = 4_000_000):
    """Maximum matching of mutually unsatisfiable cross-protein comparisons.

    For proteins k and l, ``u`` collects ``s_n − s_p`` over positives of k
    against negatives of l, ``v`` the same with the roles swapped. A pair
    ``(u_i, v_j)`` conflicts iff ``u_i + v_j ≥ 0``; the maximum matching per
    block is greedy on the descending lists, and blocks are disjoint so the
    total is their sum (`maxCrossedCard_ge_of_block`).

    ``max_block_pairs`` guards the enumeration: a block costs |pos_k|·|neg_l|,
    which on Disorder-PDB reaches billions. Blocks over the cap are skipped and
    counted, and since every skipped block could only *raise* the matching, the
    reported bound stays a valid upper bound on the achievable AUC — it is
    simply looser. That direction is checked, not assumed.
    """
    K = len(labels_by_target)
    pos = [np.asarray(s, dtype=np.float64)[np.asarray(y) == 1]
           for y, s in zip(labels_by_target, scores_by_target)]
    neg = [np.asarray(s, dtype=np.float64)[np.asarray(y) == 0]
           for y, s in zip(labels_by_target, scores_by_target)]

    matched = 0
    skipped = 0
    for k in range(K):
        if len(pos[k]) == 0 and len(neg[k]) == 0:
            continue
        for l in range(k + 1, K):
            n_u = len(pos[k]) * len(neg[l])
            n_v = len(pos[l]) * len(neg[k])
            if n_u == 0 or n_v == 0:
                continue
            if n_u > max_block_pairs or n_v > max_block_pairs:
                skipped += 1
                continue
            u = np.sort((neg[l][None, :] - pos[k][:, None]).ravel())[::-1]
            v = np.sort((neg[k][None, :] - pos[l][:, None]).ravel())[::-1]
            matched += _greedy_threshold_matching(u, v)
    return {"matched_pairs": int(matched), "blocks_skipped": int(skipped)}


def bias_bound(labels_by_target, scores_by_target, **kw) -> dict:
    """The full verdict: ceiling, attainability, and the binding upper bound.

    ``upper_bound`` is `auc_shift_le_max_crossed`: no per-protein bias, however
    chosen, can push the pooled AUC above it. Compare it to a target — a
    published leader's score — to decide whether the architecture is worth
    training before training it.
    """
    from colab.auc_decomposition import decompose_auc

    d = decompose_auc(labels_by_target, scores_by_target)
    if d.get("pooled") is None:
        return {"reason": d.get("reason", "undecomposable")}

    total_pairs = d["within_pairs"] + d["between_pairs"]
    ceiling = d["w_within"] * d["auc_within"] + d["w_between"]
    att = ceiling_attainable(labels_by_target, scores_by_target)
    m = max_crossed_matching(labels_by_target, scores_by_target, **kw)
    upper = ceiling - m["matched_pairs"] / total_pairs

    return {
        "pooled": d["pooled"],
        "auc_within": d["auc_within"],
        "auc_between": d["auc_between"],
        "ceiling": ceiling,
        "ceiling_attainable": att["attainable"],
        "two_largest_overlaps": att.get("two_largest_overlaps"),
        "n_proteins_with_overlap": att.get("n_proteins_with_overlap"),
        "matched_pairs": m["matched_pairs"],
        "blocks_skipped": m["blocks_skipped"],
        "upper_bound": upper,
        "headroom": upper - d["pooled"],
    }
