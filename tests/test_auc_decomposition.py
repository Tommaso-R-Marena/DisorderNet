"""The within/between AUC split is an identity, so it can be checked exactly.

Every claim built on this decomposition rests on it being arithmetic rather than
approximation, so these tests check it against brute-force pair counting on
small cases where enumerating every positive-negative pair is feasible.
"""

from __future__ import annotations

import numpy as np
import pytest

from colab.auc_decomposition import compare_decompositions, decompose_auc


def brute_force(labels, scores):
    """Count every positive-negative pair directly, splitting by protein.

    O(n^2) and only usable on toy inputs, which is the point: it depends on no
    identity and so can referee the one under test.
    """
    flat = [(k, y, s) for k, (ys, ss) in enumerate(zip(labels, scores))
            for y, s in zip(ys, ss)]
    pos = [(k, s) for k, y, s in flat if y == 1]
    neg = [(k, s) for k, y, s in flat if y == 0]
    win = btw = 0.0
    n_win = n_btw = 0
    for kp, sp in pos:
        for kn, sn in neg:
            v = 1.0 if sp > sn else (0.5 if sp == sn else 0.0)
            if kp == kn:
                win += v
                n_win += 1
            else:
                btw += v
                n_btw += 1
    total = n_win + n_btw
    return {
        "pooled": (win + btw) / total,
        "auc_within": win / n_win if n_win else None,
        "auc_between": btw / n_btw if n_btw else None,
        "w_within": n_win / total, "w_between": n_btw / total,
    }


def case(seed, n_targets=5, length=12):
    rng = np.random.default_rng(seed)
    labels, scores = [], []
    for _ in range(n_targets):
        y = rng.integers(0, 2, length)
        if y.sum() in (0, length):     # keep both classes present
            y[0], y[1] = 0, 1
        labels.append(y.astype(int))
        scores.append(rng.random(length))
    return labels, scores


class TestItIsAnIdentity:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_matches_brute_force_pair_counting(self, seed):
        labels, scores = case(seed)
        got = decompose_auc(labels, scores)
        want = brute_force(labels, scores)
        for key in ("pooled", "auc_within", "auc_between", "w_within",
                    "w_between"):
            assert got[key] == pytest.approx(want[key], abs=1e-9), key

    @pytest.mark.parametrize("seed", [7, 8])
    def test_the_parts_reconstruct_the_pooled_value(self, seed):
        labels, scores = case(seed, n_targets=8, length=20)
        d = decompose_auc(labels, scores)
        assert (d["w_within"] * d["auc_within"]
                + d["w_between"] * d["auc_between"]) == pytest.approx(
            d["pooled"], abs=1e-12)

    def test_weights_sum_to_one(self):
        d = decompose_auc(*case(11))
        assert d["w_within"] + d["w_between"] == pytest.approx(1.0)

    def test_pair_counts_are_exact(self):
        labels = [np.array([1, 0, 0]), np.array([1, 1, 0])]
        scores = [np.array([0.9, 0.2, 0.1]), np.array([0.8, 0.7, 0.3])]
        d = decompose_auc(labels, scores)
        # protein 1: 1 pos x 2 neg = 2; protein 2: 2 pos x 1 neg = 2
        assert d["within_pairs"] == 4
        # 3 positives x 3 negatives total = 9
        assert d["within_pairs"] + d["between_pairs"] == 9


class TestItSeparatesTheTwoAbilities:
    def test_perfect_within_and_inverted_between(self):
        """Each protein ranked perfectly, but the protein with more positives
        scored lower overall. Within should be 1.0 and between poor."""
        labels = [np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0])]
        scores = [np.array([0.4, 0.35, 0.2, 0.1]),      # high-scoring chain
                  np.array([0.9, 0.85, 0.7, 0.6])]      # but all shifted up
        d = decompose_auc(labels, scores)
        assert d["auc_within"] == pytest.approx(1.0)
        assert d["auc_between"] < d["auc_within"]

    def test_a_constant_offset_changes_only_the_between_part(self):
        labels, scores = case(3, n_targets=4, length=16)
        base = decompose_auc(labels, scores)
        shifted = [s + (5.0 if k == 0 else 0.0)
                   for k, s in enumerate(scores)]
        after = decompose_auc(labels, shifted)
        assert after["auc_within"] == pytest.approx(base["auc_within"])
        assert after["auc_between"] != pytest.approx(base["auc_between"])

    def test_per_protein_normalisation_destroys_between_information(self):
        """Rank-normalising within each protein is exactly what the Binding-IDR
        probe did; it must leave within untouched and flatten between."""
        from scipy.stats import rankdata

        labels, scores = case(5, n_targets=6, length=20)
        base = decompose_auc(labels, scores)
        normed = [rankdata(s) / (len(s) + 1.0) for s in scores]
        after = decompose_auc(labels, normed)
        assert after["auc_within"] == pytest.approx(base["auc_within"])
        assert abs(after["auc_between"] - 0.5) < abs(
            base["auc_between"] - 0.5) + 0.5


class TestDegenerateInputs:
    def test_a_single_protein_has_no_between_pairs(self):
        d = decompose_auc([np.array([1, 0, 1, 0])],
                          [np.array([0.9, 0.1, 0.8, 0.2])])
        assert d["between_pairs"] == 0
        assert d["w_within"] == pytest.approx(1.0)
        assert d["auc_within"] == pytest.approx(d["pooled"])

    def test_single_class_overall_is_reported_not_crashed(self):
        d = decompose_auc([np.array([1, 1])], [np.array([0.5, 0.6])])
        assert d["pooled"] is None
        assert "reason" in d

    def test_targets_with_one_class_contribute_no_within_pairs(self):
        labels = [np.array([1, 1, 1]), np.array([1, 0, 0])]
        scores = [np.array([0.9, 0.8, 0.7]), np.array([0.6, 0.2, 0.1])]
        d = decompose_auc(labels, scores)
        assert d["n_targets_with_both_classes"] == 1
        assert d["within_pairs"] == 2


class TestAttribution:
    def test_it_refuses_mismatched_target_sets(self):
        a = decompose_auc(*case(1, n_targets=4))
        b = decompose_auc(*case(2, n_targets=6))
        if a["w_within"] != b["w_within"]:
            with pytest.raises(ValueError, match="different target sets"):
                compare_decompositions(a, b)

    def test_contributions_sum_to_the_pooled_difference(self):
        labels, scores = case(9, n_targets=6, length=18)
        other = [s[::-1].copy() for s in scores]
        a, b = decompose_auc(labels, scores), decompose_auc(labels, other)
        c = compare_decompositions(a, b)
        assert (c["contribution_within"] + c["contribution_between"]
                ) == pytest.approx(c["delta_pooled"], abs=1e-12)
