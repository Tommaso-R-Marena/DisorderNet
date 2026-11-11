"""The distributional loss is the theorem's quantity, not an approximation.

`DistributionVerdict.distributional_design_law`, from the Lean development:

1. averages never suffice — for any finite panel of observables and any error
   budget, two ensembles reproduce every average exactly and stay further apart
   than the budget;
2. along a measured coordinate the transport distance **equals** the L1
   distance between the cumulative distributions
   (`transportCost_line_eq_cdfL1`);
3. and it **dominates** the gap between the means (`mean_gap_le_cdfL1`).

Binary cross-entropy fits each residue's mean, which is (1). This loss is (2),
in the closed form that holds for two empirical distributions with the same
number of atoms. Both identities are checked numerically here against a direct
CDF integration rather than asserted from the docstring, because "we implemented
the Wasserstein distance" is easy to write and easy to get subtly wrong.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from colab.lite_head import (  # noqa: E402
    distribution_matching_loss,
    masked_multitask_loss,
)


def direct_cdf_l1(p: torch.Tensor, y: torch.Tensor) -> float:
    """The L1 distance between empirical CDFs, integrated on the value line.

    Deliberately computed a different way from the implementation — by walking
    the merged support and summing |F_p - F_y| times each gap — so agreement is
    evidence rather than a restatement.
    """
    grid = torch.cat([p, y]).sort().values
    total = 0.0
    for a, b in zip(grid[:-1], grid[1:]):
        fp = (p <= a).double().mean()
        fy = (y <= a).double().mean()
        total += float((fp - fy).abs() * (b - a))
    return total


def sample(n=40, prev=0.35, seed=0):
    g = torch.Generator().manual_seed(seed)
    logit = torch.randn(1, n, generator=g)
    y = (torch.rand(1, n, generator=g) < prev).float()
    mask = torch.ones(1, n, dtype=torch.bool)
    return logit, y, mask


class TestItEqualsTheTransportDistance:
    def test_the_sorted_form_matches_a_direct_cdf_integration(self):
        for seed in range(8):
            logit, y, mask = sample(seed=seed)
            got = float(distribution_matching_loss(logit, y, mask))
            ref = direct_cdf_l1(torch.sigmoid(logit[0]).double(),
                                y[0].double())
            assert got == pytest.approx(ref, abs=1e-6), seed

    def test_it_dominates_the_gap_between_the_means(self):
        """`mean_gap_le_cdfL1`. If this failed the loss would be weaker than
        simply matching the predicted disorder fraction, and there would be no
        reason to prefer it."""
        for seed in range(12):
            logit, y, mask = sample(n=60, prev=0.5, seed=seed)
            w1 = float(distribution_matching_loss(logit, y, mask))
            gap = float((torch.sigmoid(logit[0]).mean() - y[0].mean()).abs())
            assert w1 >= gap - 1e-9, (seed, w1, gap)

    def test_it_is_zero_exactly_when_the_distributions_match(self):
        n, k = 20, 7
        y = torch.zeros(1, n)
        y[0, :k] = 1.0
        big, small = 40.0, -40.0                    # sigmoid -> 1 and 0
        logit = torch.full((1, n), small)
        logit[0, 3:3 + k] = big                     # right count, wrong places
        mask = torch.ones(1, n, dtype=torch.bool)
        assert float(distribution_matching_loss(logit, y, mask)) < 1e-6

    def test_it_is_blind_to_which_residues(self):
        """The division of labour: this term constrains the distribution, BCE
        constrains the placement. A permutation of the predictions must not
        change it."""
        logit, y, mask = sample(seed=3)
        a = float(distribution_matching_loss(logit, y, mask))
        perm = torch.randperm(logit.shape[1])
        b = float(distribution_matching_loss(logit[:, perm], y, mask))
        assert a == pytest.approx(b, abs=1e-6)

    def test_bce_is_not_blind_to_which_residues(self):
        """Guard the guard: if BCE were permutation-invariant too, the two
        terms would be redundant and the split above would be empty."""
        logit, y, mask = sample(seed=3)
        base = masked_multitask_loss({"t": logit}, {"t": y}, {"t": mask})[0]
        perm = torch.randperm(logit.shape[1])
        moved = masked_multitask_loss({"t": logit[:, perm]}, {"t": y},
                                      {"t": mask})[0]
        assert abs(float(base) - float(moved)) > 1e-3

    def test_a_wrong_count_costs_more_than_a_wrong_placement(self):
        n, k = 20, 7
        y = torch.zeros(1, n)
        y[0, :k] = 1.0
        mask = torch.ones(1, n, dtype=torch.bool)
        right_count = torch.full((1, n), -40.0)
        right_count[0, 5:5 + k] = 40.0            # k positives, all misplaced
        wrong_count = torch.full((1, n), -40.0)
        wrong_count[0, :k + 6] = 40.0             # 6 too many, correctly placed
        assert float(distribution_matching_loss(right_count, y, mask)) < \
            float(distribution_matching_loss(wrong_count, y, mask))


class TestItBehavesInsideTheTrainingLoss:
    def test_it_is_off_by_default(self):
        """Turning it on changes what every existing checkpoint would compute,
        so it must be opt-in."""
        logit, y, mask = sample(seed=1)
        off, parts = masked_multitask_loss({"t": logit}, {"t": y}, {"t": mask})
        assert not any(k.endswith("/W1") for k in parts)
        on, parts_on = masked_multitask_loss(
            {"t": logit}, {"t": y}, {"t": mask}, distribution_weight=0.5)
        assert "t/W1" in parts_on
        assert float(on) > float(off)

    def test_the_weight_scales_it_linearly(self):
        logit, y, mask = sample(seed=2)
        base = float(masked_multitask_loss({"t": logit}, {"t": y},
                                           {"t": mask})[0])
        a = float(masked_multitask_loss({"t": logit}, {"t": y}, {"t": mask},
                                        distribution_weight=0.1)[0])
        b = float(masked_multitask_loss({"t": logit}, {"t": y}, {"t": mask},
                                        distribution_weight=0.2)[0])
        assert (b - base) == pytest.approx(2.0 * (a - base), rel=1e-5)

    def test_it_is_per_protein_not_pooled(self):
        """A distribution pooled across proteins is a different object — the
        one CAID already reports. Two proteins with opposite errors must not
        cancel."""
        n = 30
        y = torch.zeros(2, n)
        y[0, :20] = 1.0                        # 2/3 disordered
        y[1, :3] = 1.0                         # 1/10 disordered
        mask = torch.ones(2, n, dtype=torch.bool)
        # Each protein predicted at the *other's* prevalence: pooled means
        # agree, per-protein distributions do not.
        logit = torch.full((2, n), -40.0)
        logit[0, :3] = 40.0
        logit[1, :20] = 40.0
        assert float(distribution_matching_loss(logit, y, mask)) > 0.4

    def test_it_reaches_the_parameters(self):
        logit, y, mask = sample(seed=4)
        logit = logit.clone().requires_grad_(True)
        loss = distribution_matching_loss(logit, y, mask)
        loss.backward()
        assert logit.grad is not None
        assert torch.count_nonzero(logit.grad) > 0

    def test_a_protein_with_too_few_residues_is_skipped_not_crashed(self):
        logit = torch.randn(1, 5)
        y = torch.zeros(1, 5)
        mask = torch.zeros(1, 5, dtype=torch.bool)
        mask[0, 0] = True
        assert distribution_matching_loss(logit, y, mask) is None

    def test_masked_residues_are_excluded(self):
        n = 20
        y = torch.zeros(1, n)
        y[0, :5] = 1.0
        full = torch.ones(1, n, dtype=torch.bool)
        logit = torch.full((1, n), -40.0)
        logit[0, :5] = 40.0
        assert float(distribution_matching_loss(logit, y, full)) < 1e-6
        # Mask out the positives; both distributions become all-zero, still 0.
        part = full.clone()
        part[0, :5] = False
        assert float(distribution_matching_loss(logit, y, part)) < 1e-6
