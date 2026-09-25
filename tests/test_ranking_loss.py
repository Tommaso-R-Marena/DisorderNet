"""A loss for the part of the metric no recalibration can change.

`AUC_pooled = w_within * AUC_within + w_between * AUC_between`, and
`auc_within_strictMono_invariant` identifies `AUC_within` as the
calibration-invariant part — the irreducible skill, which a per-protein
recalibration can neither buy nor lose.

Nothing in the standard recipe optimises it. Binary cross-entropy fits each
residue's mean; `distribution_matching_loss` fits the protein's distribution.
Neither targets ordering *inside* a chain. This does, and these tests check
that it targets that and nothing else.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from colab.auc_decomposition import decompose_auc  # noqa: E402
from colab.lite_head import (  # noqa: E402
    masked_multitask_loss,
    within_protein_ranking_loss,
)


def batch(n_prot=16, n_res=100, prevalence=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(n_prot, n_res, generator=g) < prevalence).float()
    # guarantee both classes per protein
    y[:, 0], y[:, 1] = 1.0, 0.0
    mask = torch.ones(n_prot, n_res, dtype=torch.bool)
    return y, mask


class TestItOptimisesWhatItClaims:
    def test_minimising_it_drives_auc_within_to_one(self):
        """The direct check: descend on this loss alone and watch the quantity
        it is a surrogate for."""
        y, mask = batch(seed=1)
        logit = torch.randn(y.shape, requires_grad=True)
        opt = torch.optim.Adam([logit], lr=0.1)

        def auc_within():
            ys = [y[i].numpy().astype(np.int8) for i in range(y.shape[0])]
            ss = [logit[i].detach().numpy().astype(np.float64)
                  for i in range(y.shape[0])]
            return decompose_auc(ys, ss)["auc_within"]

        before = auc_within()
        for _ in range(120):
            opt.zero_grad()
            loss = within_protein_ranking_loss(logit, y, mask, n_pairs=256)
            loss.backward()
            opt.step()
        after = auc_within()
        assert before < 0.6, before
        assert after > 0.95, after

    def test_it_is_invariant_to_per_protein_shifts(self):
        """The property that makes it the right surrogate: adding a constant
        to a whole chain cannot change within-chain ordering, so it cannot
        change this loss."""
        y, mask = batch(seed=2)
        logit = torch.randn(y.shape)
        g = torch.Generator().manual_seed(0)
        shifted = logit + torch.randn(y.shape[0], 1, generator=g) * 5.0
        gen_a = torch.Generator().manual_seed(7)
        gen_b = torch.Generator().manual_seed(7)
        a = within_protein_ranking_loss(logit, y, mask, n_pairs=512,
                                        generator=gen_a)
        b = within_protein_ranking_loss(shifted, y, mask, n_pairs=512,
                                        generator=gen_b)
        assert float(a) == pytest.approx(float(b), abs=1e-5)

    def test_bce_is_not_invariant_to_per_protein_shifts(self):
        """Guard the guard: if BCE were shift-invariant too, this loss would be
        redundant and the decomposition it follows would be empty."""
        y, mask = batch(seed=2)
        logit = torch.randn(y.shape)
        g = torch.Generator().manual_seed(0)
        shifted = logit + torch.randn(y.shape[0], 1, generator=g) * 5.0
        a = masked_multitask_loss({"t": logit}, {"t": y}, {"t": mask})[0]
        b = masked_multitask_loss({"t": shifted}, {"t": y}, {"t": mask})[0]
        assert abs(float(a) - float(b)) > 0.1

    def test_a_perfect_within_protein_ordering_costs_least(self):
        y, mask = batch(n_prot=8, n_res=60, seed=3)
        good = (y * 20.0) - 10.0                  # perfectly ordered
        bad = -good
        gen_a = torch.Generator().manual_seed(1)
        gen_b = torch.Generator().manual_seed(1)
        assert float(within_protein_ranking_loss(good, y, mask, 512, gen_a)) < \
            float(within_protein_ranking_loss(bad, y, mask, 512, gen_b))

    def test_pairs_never_cross_protein_boundaries(self):
        """A cross-protein pair would make this the pooled AUC surrogate, which
        is the thing the decomposition says is dominated by calibration."""
        n_res = 40
        y = torch.zeros(2, n_res)
        y[0, :10] = 1.0                       # protein 0 has positives
        y[1, 20:] = 1.0                       # protein 1's are elsewhere
        mask = torch.ones(2, n_res, dtype=torch.bool)
        logit = torch.zeros(2, n_res, requires_grad=True)
        loss = within_protein_ranking_loss(logit, y, mask, n_pairs=512)
        loss.backward()
        g = logit.grad
        # Every gradient must sit on a residue of the protein it belongs to;
        # with all logits equal each protein contributes independently.
        assert torch.count_nonzero(g[0, 10:]) > 0     # protein 0 negatives
        assert torch.count_nonzero(g[1, :20]) > 0     # protein 1 negatives
        assert torch.count_nonzero(g[0, :10]) > 0
        assert torch.count_nonzero(g[1, 20:]) > 0


class TestItBehavesInTheTrainingLoss:
    def test_it_is_off_by_default(self):
        y, mask = batch(seed=4)
        logit = torch.randn(y.shape)
        _off, parts = masked_multitask_loss({"t": logit}, {"t": y},
                                            {"t": mask})
        assert not any(k.endswith("/rank") for k in parts)
        _on, parts_on = masked_multitask_loss({"t": logit}, {"t": y},
                                              {"t": mask}, ranking_weight=0.5)
        assert "t/rank" in parts_on

    def test_the_three_terms_can_be_combined(self):
        y, mask = batch(seed=5)
        logit = torch.randn(y.shape)
        _l, parts = masked_multitask_loss(
            {"t": logit}, {"t": y}, {"t": mask},
            distribution_weight=0.1, ranking_weight=0.1)
        assert {"t", "t/W1", "t/rank"} <= set(parts)

    def test_a_single_class_protein_contributes_nothing(self):
        y = torch.zeros(2, 30)
        y[0, :5] = 1.0                        # only protein 0 has both
        mask = torch.ones(2, 30, dtype=torch.bool)
        logit = torch.randn(2, 30, requires_grad=True)
        loss = within_protein_ranking_loss(logit, y, mask, n_pairs=128)
        loss.backward()
        assert torch.count_nonzero(logit.grad[1]) == 0
        assert torch.count_nonzero(logit.grad[0]) > 0

    def test_an_all_single_class_batch_returns_none(self):
        y = torch.zeros(2, 30)
        mask = torch.ones(2, 30, dtype=torch.bool)
        assert within_protein_ranking_loss(torch.randn(2, 30), y, mask) is None

    def test_masked_residues_are_excluded(self):
        y = torch.zeros(1, 20)
        y[0, :5] = 1.0
        mask = torch.ones(1, 20, dtype=torch.bool)
        mask[0, :5] = False                   # mask out every positive
        assert within_protein_ranking_loss(torch.randn(1, 20), y, mask) is None
