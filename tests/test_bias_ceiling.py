"""The per-protein bias bound — checked against brute force where feasible.

Each function mirrors a machine-checked Lean theorem, so the risk is not that
the mathematics is wrong but that this implementation does not compute it.
Small cases are therefore checked against exhaustive search over biases.
"""

from __future__ import annotations

import numpy as np
import pytest

from colab.bias_ceiling import (bias_bound, ceiling_attainable,
                                max_crossed_matching, optimal_centering_bias,
                                protein_overlaps)
from colab.auc_decomposition import decompose_auc


def brute_force_best(labels, scores, grid=None):
    """Best pooled AUC over a grid of per-protein biases. Exponential, so tiny
    inputs only — but it depends on no theorem."""
    import itertools
    grid = grid if grid is not None else np.linspace(-3, 3, 13)
    best = -1.0
    for b in itertools.product(grid, repeat=len(labels)):
        shifted = [s + bb for s, bb in zip(scores, b)]
        d = decompose_auc(labels, shifted)
        if d.get("pooled") is not None:
            best = max(best, d["pooled"])
    return best


class TestOverlaps:
    def test_a_perfectly_ranked_protein_has_negative_overlap(self):
        ov = protein_overlaps([np.array([1, 1, 0, 0])],
                              [np.array([0.9, 0.8, 0.2, 0.1])])
        assert ov[0] == pytest.approx(0.2 - 0.8)

    def test_an_imperfect_protein_has_positive_overlap(self):
        ov = protein_overlaps([np.array([1, 0])], [np.array([0.1, 0.9])])
        assert ov[0] > 0

    def test_a_single_class_protein_is_none(self):
        assert protein_overlaps([np.array([1, 1])], [np.array([0.5, 0.6])])[0] is None


class TestCeilingAttainability:
    def test_two_perfectly_separated_proteins_attain_it(self):
        labels = [np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0])]
        scores = [np.array([0.9, 0.85, 0.2, 0.1]), np.array([0.9, 0.85, 0.2, 0.1])]
        r = ceiling_attainable(labels, scores)
        assert r["attainable"]
        assert r["sum"] < 0

    def test_one_imperfect_protein_pair_breaks_it(self):
        labels = [np.array([1, 0]), np.array([1, 0])]
        scores = [np.array([0.1, 0.9]), np.array([0.1, 0.9])]
        assert not ceiling_attainable(labels, scores)["attainable"]

    def test_the_verdict_uses_only_the_two_largest(self):
        labels = [np.array([1, 1, 0, 0])] * 4
        scores = [np.array([0.9, 0.8, 0.2, 0.1])] * 3 + [np.array([0.1, 0.05, 0.9, 0.8])]
        r = ceiling_attainable(labels, scores)
        ov = sorted(o for o in protein_overlaps(labels, scores) if o is not None)
        assert r["two_largest_overlaps"] == [pytest.approx(ov[-1]),
                                             pytest.approx(ov[-2])]

    def test_centering_bias_attains_the_ceiling_when_it_should(self):
        labels = [np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0])]
        scores = [np.array([0.9, 0.85, 0.2, 0.1]),
                  np.array([5.9, 5.85, 5.2, 5.1])]
        assert ceiling_attainable(labels, scores)["attainable"]
        b = optimal_centering_bias(labels, scores)
        shifted = [s + bb for s, bb in zip(scores, b)]
        d = decompose_auc(labels, shifted)
        assert d["pooled"] == pytest.approx(1.0, abs=1e-9)


class TestCrossedMatching:
    def test_no_conflicts_when_proteins_are_separable(self):
        labels = [np.array([1, 0]), np.array([1, 0])]
        scores = [np.array([0.9, 0.1]), np.array([0.9, 0.1])]
        assert max_crossed_matching(labels, scores)["matched_pairs"] == 0

    def test_a_mutually_inverted_pair_conflicts(self):
        """Each protein's negative outranks the other's positive, so no bias
        can fix both directions."""
        labels = [np.array([1, 0]), np.array([1, 0])]
        scores = [np.array([0.0, 10.0]), np.array([0.0, 10.0])]
        assert max_crossed_matching(labels, scores)["matched_pairs"] >= 1

    def test_matching_never_exceeds_the_smaller_side(self):
        rng = np.random.default_rng(0)
        labels = [rng.integers(0, 2, 6) for _ in range(3)]
        scores = [rng.random(6) for _ in range(3)]
        for i in range(3):
            if labels[i].sum() in (0, 6):
                labels[i][0], labels[i][1] = 0, 1
        m = max_crossed_matching(labels, scores)["matched_pairs"]
        d = decompose_auc(labels, scores)
        assert 0 <= m <= d["between_pairs"]


class TestTheBoundHolds:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_no_bias_beats_the_upper_bound(self, seed):
        """The claim that matters: brute-force search over biases must never
        exceed the computed bound."""
        rng = np.random.default_rng(seed)
        labels, scores = [], []
        for _ in range(3):
            y = rng.integers(0, 2, 4)
            if y.sum() in (0, 4):
                y[0], y[1] = 0, 1
            labels.append(y.astype(int))
            scores.append(rng.random(4))
        bound = bias_bound(labels, scores)["upper_bound"]
        best = brute_force_best(labels, scores)
        assert best <= bound + 1e-9, (best, bound)

    @pytest.mark.parametrize("seed", [7, 8, 9])
    def test_the_bound_is_at_least_the_unshifted_score(self, seed):
        """b = 0 is feasible, so the bound cannot fall below the current AUC."""
        rng = np.random.default_rng(seed)
        labels, scores = [], []
        for _ in range(4):
            y = rng.integers(0, 2, 5)
            if y.sum() in (0, 5):
                y[0], y[1] = 0, 1
            labels.append(y.astype(int))
            scores.append(rng.random(5))
        r = bias_bound(labels, scores)
        assert r["upper_bound"] >= r["pooled"] - 1e-9

    def test_skipped_blocks_only_loosen_the_bound(self):
        """A skipped block can only reduce the matching, which raises the
        bound — so the guard never makes the bound unsound."""
        rng = np.random.default_rng(3)
        labels = [rng.integers(0, 2, 8).astype(int) for _ in range(3)]
        for y in labels:
            y[0], y[1] = 0, 1
        scores = [rng.random(8) for _ in range(3)]
        full = bias_bound(labels, scores)
        capped = bias_bound(labels, scores, max_block_pairs=1)
        assert capped["blocks_skipped"] > 0
        assert capped["upper_bound"] >= full["upper_bound"] - 1e-12
