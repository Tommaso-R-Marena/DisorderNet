"""The package must reproduce the paper's numbers, or it is not the paper's tool.

Each test names the figure or table the number appears in, so a change to the
library that moves a published number fails loudly rather than quietly.
"""

from __future__ import annotations

import numpy as np
import pytest

from disordernet import (
    assess, capacity, capacity_over_range, imbalance_factor,
    pairwise_capacity, pairwise_noise_bound, rank, rates, rates_by_group,
    target_auc, unresolvable_comparisons,
)


class TestCapacityReproducesThePaper:
    @pytest.mark.parametrize("name,eps,expected", [
        ("QuickDraw", 0.1012, 5), ("CAID3", 0.0801, 7),
        ("CIFAR-100", 0.0585, 9), ("ImageNet", 0.0583, 9),
        ("Amazon", 0.0390, 13), ("IMDB", 0.0290, 18),
        ("Caltech-256", 0.0154, 33), ("AudioSet", 0.0135, 38),
        ("20news", 0.0109, 46), ("CIFAR-10", 0.0054, 93),
        ("MNIST", 0.0015, 334),
    ])
    def test_table_s17(self, name, eps, expected):
        assert capacity(eps) == expected, name

    def test_imagenet_over_its_published_range(self):
        """Fig. 2d note: the realisable figure is smaller than the bound."""
        assert capacity_over_range(0.0583, 0.55, 0.92) == 4

    def test_more_items_do_not_help(self):
        """`benchCapacity_noise_only`: the ceiling is free of n."""
        assert {capacity(0.0801, n=n) for n in (31, 233, 10_000)} == {7}

    def test_a_positive_delta_only_lowers_capacity(self):
        assert capacity(0.0801, delta=0.5) <= capacity(0.0801)

    def test_rejects_an_impossible_rate(self):
        for bad in (0.0, 1.0, -0.1):
            with pytest.raises(ValueError):
                capacity(bad)


class TestTheEscape:
    def test_caid3_capacity_rises_from_8_to_51(self):
        """Fig. 2b, from the measured rates."""
        assert capacity(0.0651) == 8
        assert pairwise_capacity(0.0651, eps_pair=0.00996) == 51

    def test_kappa_is_one_exactly_when_balanced(self):
        """`kappa_eq_one_iff_balanced`."""
        assert imbalance_factor(500, 500) == pytest.approx(1.0)
        assert imbalance_factor(240506, 555402) == pytest.approx(1.1856, abs=1e-4)
        assert imbalance_factor(1, 999) > 1.0

    def test_the_bound_needs_no_balance_hypothesis(self):
        """`nuPair_le_imbalanced` holds for any non-empty agreement classes."""
        for kappa in (1.0, 1.1856, 12.0, 250.0):
            assert pairwise_noise_bound(0.0651, kappa) > 0

    def test_balanced_case_recovers_the_published_bound(self):
        """`nuPair_le_two_eps_sq_of_kappa` at kappa = 1, eps <= 1/4."""
        eps = 0.05
        assert pairwise_noise_bound(eps, 1.0) <= 2 * eps ** 2


class TestCountingConverse:
    def test_unresolvable_count_117(self):
        """`unresolvable_count_117`: at least 798 of 6,786."""
        assert unresolvable_comparisons(117, 8) == 798
        assert 117 * 116 // 2 == 6786

    def test_a_field_within_capacity_leaves_nothing_undecidable(self):
        assert unresolvable_comparisons(5, 100) == 0


class TestTheProtocol:
    def test_a_constant_predictor_scores_exactly_one_half(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        assert target_auc(y, np.zeros(6)) == pytest.approx(0.5)

    def test_single_class_targets_are_skipped_not_scored(self):
        assert target_auc(np.ones(8), np.arange(8.0)) is None
        assert target_auc(np.zeros(8), np.arange(8.0)) is None

    def test_calibration_invariance(self):
        """`auc_target_strictMono_invariant`, the protocol's central property."""
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        s = rng.normal(size=200) + y
        base = target_auc(y, s)
        for f in (np.exp, lambda x: x ** 3, lambda x: 1 / (1 + np.exp(-x)),
                  lambda x: 7 * x - 3):
            assert target_auc(y, f(s)) == pytest.approx(base)

    def test_declining_a_target_makes_a_method_ineligible(self):
        """Step 1 is a gate, not a covariate."""
        rng = np.random.default_rng(1)
        ref = {f"t{i}": rng.integers(0, 2, 40) for i in range(12)}
        full = {t: rng.normal(size=40) for t in ref}
        partial = {t: v for t, v in list(full.items())[:8]}
        lb = rank(ref, {"full": full, "declines": partial})
        assert [r["method"] for r in lb.rows] == ["full"]
        assert lb.n_entered == 2 and lb.n_eligible == 1

    def test_a_better_method_ranks_first_and_separates(self):
        rng = np.random.default_rng(2)
        ref = {f"t{i}": rng.integers(0, 2, 80) for i in range(50)}
        good = {t: y + rng.normal(0, 0.5, y.size) for t, y in ref.items()}
        weak = {t: y + rng.normal(0, 3.0, y.size) for t, y in ref.items()}
        lb = rank(ref, {"weak": weak, "good": good}, eps_pair=0.01)
        assert lb.rows[0]["method"] == "good"
        assert lb.rows[1]["separated"]
        assert lb.capacity == 50


class TestNoiseRates:
    def test_discordance_is_the_flip_product(self):
        """`discordant_eq_flip_product`: discordant = 2*d*u, exactly."""
        rng = np.random.default_rng(3)
        for _ in range(50):
            t = rng.integers(0, 2, 60).astype(bool)
            l = rng.integers(0, 2, 60).astype(bool)
            r = rates(t, l)
            d = int((t & ~l).sum())
            u = int((~t & l).sum())
            assert r.discordant_pairs == 2 * d * u

    def test_comparable_pairs_are_counted_exactly(self):
        """`card_comparablePairs` = 2*(a*e + d*u), and no larger set."""
        rng = np.random.default_rng(4)
        t = rng.integers(0, 2, 100).astype(bool)
        l = rng.integers(0, 2, 100).astype(bool)
        r = rates(t, l)
        a = int((t & l).sum()); e = int((~t & ~l).sum())
        d = int((t & ~l).sum()); u = int((~t & l).sum())
        assert r.comparable_pairs == 2 * (a * e + d * u)

    def test_perfect_agreement_gives_zero_noise(self):
        t = np.array([1, 0, 1, 0, 1])
        r = rates(t, t)
        assert r.eps_label == 0.0 and r.eps_pair == 0.0

    def test_pair_noise_is_second_order(self):
        """The whole point: the pairwise rate is far below the label rate."""
        rng = np.random.default_rng(5)
        t = rng.random(4000) < 0.3
        flip = rng.random(4000) < 0.06
        l = np.where(flip, ~t, t)
        r = rates(t, l)
        assert r.eps_pair < r.eps_label
        assert r.ratio > 5

    def test_grouped_rates_pool_within_groups_only(self):
        rng = np.random.default_rng(6)
        t = {f"g{i}": rng.random(200) < 0.3 for i in range(10)}
        l = {k: np.where(rng.random(200) < 0.05, ~v, v) for k, v in t.items()}
        r = rates_by_group(t, l)
        assert r.n_items == 2000
        assert 0 < r.eps_pair < r.eps_label


class TestTheVerdict:
    def test_caid3_verdict(self):
        v = assess(117, 0.0801, eps_pair=0.00996)
        assert (v.capacity, v.pairwise_capacity) == (7, 51)
        assert v.total_comparisons == 6786
        assert v.over_capacity_by == pytest.approx(117 / 7, abs=1e-6)

    def test_it_prints_something_a_person_can_read(self):
        assert "capacity" in str(assess(117, 0.0801))
