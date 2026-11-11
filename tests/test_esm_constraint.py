"""Constraint from the masked language model — the mechanics, not the biology.

The hypothesis is that binding residues inside a disordered region carry higher
pseudo-likelihood than the sequence around them, because motifs are conserved
and their surroundings drift. Whether that is true of proteins is measured on
CAID3, not asserted here.

What is testable without a GPU is that the machinery computes what it claims:
every position masked exactly once, the residue's own identity never visible
when its probability is read, and the local baseline actually removing a
regional offset. Each of those, wrong, would produce a plausible number.
"""

from __future__ import annotations

import numpy as np
import pytest

from colab.esm_constraint import local_constraint, protein_constraint


class FakeAlphabet:
    mask_idx = 32


class FakeESM:
    """Records what it was asked, and answers deterministically.

    The token at a masked position must be unreadable, so this returns a
    distribution that depends only on position — any dependence on the true
    token would let a broken implementation score itself.
    """

    def __init__(self, vocab=33):
        self.vocab = vocab
        self.seen_masks = []

    def __call__(self, tokens, repr_layers=None, return_contacts=False):
        import torch

        b, ln = tokens.shape
        for row in range(b):
            self.seen_masks.append(
                {int(i) for i in (tokens[row] == FakeAlphabet.mask_idx)
                 .nonzero().flatten()})
        logits = torch.zeros(b, ln, self.vocab)
        for i in range(ln):
            logits[:, i, i % self.vocab] = 5.0
        return {"logits": logits}


def fake_converter(pairs):
    import torch

    seq = pairs[0][1]
    toks = torch.zeros(1, len(seq) + 2, dtype=torch.long)
    for i, ch in enumerate(seq):
        toks[0, i + 1] = (ord(ch) % 25) + 4
    return None, None, toks


class TestMaskingMechanics:
    @pytest.mark.parametrize("n,groups", [(10, 4), (32, 16), (100, 16),
                                          (5, 16), (1, 16)])
    def test_every_position_is_masked_exactly_once(self, n, groups):
        from colab.esm_constraint import pseudo_log_likelihood

        esm = FakeESM()
        seq = "ACDEFGHIKLMNPQRSTVWY"[:1] * n if n < 20 else "ACDEFGHIKLMNPQRSTVWY" * (n // 20 + 1)
        seq = seq[:n]
        pll = pseudo_log_likelihood(esm, fake_converter, FakeAlphabet(), seq,
                                    device="cpu", n_groups=groups)
        assert len(pll) == n
        assert np.isfinite(pll).all()

        counts = np.zeros(n, dtype=int)
        for masked in esm.seen_masks:
            for tok in masked:
                pos = tok - 1                       # undo the BOS offset
                if 0 <= pos < n:
                    counts[pos] += 1
        assert (counts == 1).all(), (
            f"positions masked {sorted(set(counts.tolist()))} times, want all 1")

    def test_masked_positions_are_spread_apart(self):
        """Two masks adjacent in one pass would each be predicting from a
        neighbour that is itself hidden."""
        from colab.esm_constraint import pseudo_log_likelihood

        esm = FakeESM()
        pseudo_log_likelihood(esm, fake_converter, FakeAlphabet(),
                              "ACDEFGHIKL" * 10, device="cpu", n_groups=16)
        for masked in esm.seen_masks:
            positions = sorted(masked)
            gaps = [b - a for a, b in zip(positions, positions[1:])]
            assert all(g >= 16 for g in gaps), gaps

    def test_an_empty_sequence_returns_empty(self):
        from colab.esm_constraint import pseudo_log_likelihood

        out = pseudo_log_likelihood(FakeESM(), fake_converter, FakeAlphabet(),
                                    "", device="cpu")
        assert out.shape == (0,)

    def test_cost_is_the_group_count_not_the_length(self):
        """The whole point of the stride: 16 passes, not one per residue."""
        from colab.esm_constraint import pseudo_log_likelihood

        esm = FakeESM()
        pseudo_log_likelihood(esm, fake_converter, FakeAlphabet(),
                              "ACDEFGHIKL" * 50, device="cpu", n_groups=16,
                              batch_size=8)
        assert len(esm.seen_masks) == 16, len(esm.seen_masks)


class TestLocalConstraint:
    def test_a_constant_signal_becomes_zero(self):
        out = local_constraint(np.full(50, -2.5))
        assert np.allclose(out, 0.0, atol=1e-9)

    def test_a_regional_offset_is_removed(self):
        """Composition bias: one half of the chain simply more predictable."""
        pll = np.concatenate([np.full(60, -1.0), np.full(60, -4.0)])
        out = local_constraint(pll, window=21)
        interior = np.r_[out[10:50], out[70:110]]
        assert np.abs(interior).max() < 0.05

    def test_a_local_spike_survives(self):
        """A motif's excess constraint is exactly what must not be removed."""
        pll = np.full(100, -3.0)
        pll[48:53] = -0.5
        out = local_constraint(pll, window=31)
        assert out[50] > 1.0
        assert out[10] < 0.1

    def test_it_preserves_length(self):
        for n in (1, 2, 5, 31, 32, 200):
            assert len(local_constraint(np.random.randn(n))) == n

    def test_a_short_sequence_is_only_centred(self):
        out = local_constraint(np.array([1.0, 3.0]))
        assert out.sum() == pytest.approx(0.0)


class TestProteinConstraint:
    def test_it_averages_over_the_disordered_residues_only(self):
        pll = np.array([-1.0, -1.0, -5.0, -5.0])
        mask = np.array([True, True, False, False])
        assert protein_constraint(pll, mask) == pytest.approx(-1.0)

    def test_no_disordered_residues_gives_nan_not_zero(self):
        """Zero would read as a measurement; NaN says there was nothing to
        measure."""
        v = protein_constraint(np.array([-1.0, -2.0]), np.array([False, False]))
        assert np.isnan(v)

    def test_it_distinguishes_two_proteins(self):
        m = np.array([True, True, True])
        a = protein_constraint(np.array([-0.5, -0.6, -0.4]), m)
        b = protein_constraint(np.array([-4.0, -3.8, -4.2]), m)
        assert a > b
