"""DisorderNet-Lite: frozen backbone, low-capacity head.

Motivated by measurement, not preference. On homology-split DisProt:

  ESM-2 650M + LoRA (69.9M trainable)   0.7454
  v6 physics GBDT                       0.7804
  AlphaFold pLDDT alone                 0.7906

A 650M PLM losing to a GBDT indicates a capacity/data mismatch — 69.9M trainable
parameters on ~1M residues from 2,340 proteins, with ten active regularisers all
tuned under a leaking evaluation. These tests pin the properties that make the
alternative sample-efficient, so a future change cannot quietly reintroduce the
same failure.
"""

from __future__ import annotations

import pytest
import torch

from colab.lite_head import (
    DilatedResidualBlock,
    DisorderNetLite,
    ScalarMix,
    freeze_backbone,
)

ULTRA_TRAINABLE = 69_930_000  # measured: LoRA r=128 x20 layers + FFN + tail


class TestScalarMix:
    def test_is_a_convex_combination(self):
        mix = ScalarMix(8)
        w = mix.layer_weights()
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
        assert bool((w >= 0).all())

    def test_costs_almost_nothing(self):
        """One weight per layer plus a scale — depth becomes learnable for ~free."""
        mix = ScalarMix(33)
        assert sum(p.numel() for p in mix.parameters()) == 34

    def test_uniform_at_init(self):
        """No layer is privileged before training."""
        w = ScalarMix(5).layer_weights()
        assert bool(torch.allclose(w, torch.full((5,), 0.2), atol=1e-6))

    def test_rejects_wrong_layer_count(self):
        mix = ScalarMix(3)
        with pytest.raises(ValueError):
            mix([torch.randn(1, 4, 8)] * 2)

    def test_gradients_reach_the_weights(self):
        mix = ScalarMix(4)
        out = mix([torch.randn(2, 10, 16) for _ in range(4)])
        out.sum().backward()
        assert mix.weights.grad is not None
        assert float(mix.weights.grad.abs().sum()) > 0


class TestCapacity:
    def test_is_far_smaller_than_the_lora_configuration(self):
        m = DisorderNetLite(embed_dim=1280, n_layers_mixed=6, physics_dim=32)
        assert m.n_trainable() < ULTRA_TRAINABLE / 10, (
            f"{m.n_trainable():,} trainable — the point of this architecture is "
            "sample efficiency on ~1M residues"
        )

    def test_receptive_field_covers_a_disorder_segment(self):
        """Disorder segments are tens of residues; the head must see that context."""
        m = DisorderNetLite(embed_dim=64, n_layers_mixed=2, n_blocks=4)
        assert m.receptive_field >= 30

    def test_capacity_scales_with_hidden_not_backbone(self):
        small = DisorderNetLite(embed_dim=1280, n_layers_mixed=6, hidden=128)
        large = DisorderNetLite(embed_dim=1280, n_layers_mixed=6, hidden=512)
        assert large.n_trainable() > small.n_trainable()
        # Backbone width must not dominate the parameter count.
        wide = DisorderNetLite(embed_dim=2560, n_layers_mixed=6, hidden=128)
        assert wide.n_trainable() < 2 * small.n_trainable()


class TestForward:
    def test_output_is_one_logit_per_residue(self):
        m = DisorderNetLite(embed_dim=32, n_layers_mixed=3, physics_dim=8)
        out = m([torch.randn(2, 40, 32) for _ in range(3)], physics=torch.randn(2, 40, 8))
        assert out.shape == (2, 40)

    def test_missing_required_channel_raises(self):
        m = DisorderNetLite(embed_dim=32, n_layers_mixed=2, physics_dim=8)
        with pytest.raises(ValueError, match="physics"):
            m([torch.randn(1, 10, 32) for _ in range(2)])

    def test_handles_variable_length(self):
        m = DisorderNetLite(embed_dim=32, n_layers_mixed=2)
        for L in (16, 101, 512):
            out = m([torch.randn(1, L, 32) for _ in range(2)])
            assert out.shape == (1, L)

    def test_optional_channels_can_be_omitted(self):
        m = DisorderNetLite(embed_dim=32, n_layers_mixed=2, physics_dim=0, plddt_dim=0)
        assert m([torch.randn(1, 20, 32) for _ in range(2)]).shape == (1, 20)


class TestNormalisationChoice:
    def test_head_uses_groupnorm_not_batchnorm(self):
        """Batches are a few proteins of very different lengths, so batch
        statistics are unstable — and BatchNorm running buffers are exactly what
        went missing in this project's checkpoint round-trip."""
        m = DisorderNetLite(embed_dim=32, n_layers_mixed=2)
        kinds = {type(mod) for mod in m.modules()}
        assert torch.nn.GroupNorm in kinds
        assert torch.nn.BatchNorm1d not in kinds

    def test_block_is_residual(self):
        """A residual block must pass its input through when the branch is zeroed."""
        blk = DilatedResidualBlock(8, dilation=2)
        for p in blk.conv2.parameters():
            torch.nn.init.zeros_(p)
        blk.eval()
        x = torch.randn(1, 8, 20)
        assert torch.allclose(blk(x), x, atol=1e-5)


class TestFrozenBackbone:
    def test_freeze_reports_and_disables_grads(self):
        net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
        n = freeze_backbone(net)
        assert n == 4  # two weights + two biases
        assert not any(p.requires_grad for p in net.parameters())

    def test_freezing_removes_the_lost_weight_failure_mode(self):
        """With a frozen backbone the only trainable state is the head, so a
        checkpoint cannot silently omit fine-tuned backbone weights — the defect
        that cost this project 0.11 AUC on reload."""
        net = torch.nn.Linear(4, 4)
        freeze_backbone(net)
        head = DisorderNetLite(embed_dim=4, n_layers_mixed=1)
        trainable = [n for n, p in head.named_parameters() if p.requires_grad]
        assert trainable, "head must be trainable"
        assert not [n for n, p in net.named_parameters() if p.requires_grad]
