"""Tests for SOTA training modules."""

from __future__ import annotations

import torch

from colab.disordernet_gpu import TrainConfig
from colab.sota_losses import batch_mean_dice_loss, composite_disorder_loss, soft_dice_from_logits
from colab.sota_heads import DisorderSOTAHead


class TestSOTALosses:
    def test_dice_perfect(self):
        logits = torch.tensor([5.0, 5.0, -5.0, -5.0])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        assert soft_dice_from_logits(logits, labels).item() < 0.1

    def test_batch_dice(self):
        logits = torch.tensor([[5.0, 5.0], [-5.0, -5.0]])
        labels = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        mask = torch.ones(2, 2, dtype=torch.bool)
        loss = batch_mean_dice_loss(logits, labels, mask)
        assert loss.item() >= 0.0

    def test_composite_with_dice(self):
        cfg = TrainConfig(use_dice_loss=True, dice_loss_weight=0.3, use_focal_loss=True)
        logits = torch.randn(2, 8)
        labels = (torch.rand(2, 8) > 0.7).float()
        mask = torch.ones(2, 8, dtype=torch.bool)
        loss = composite_disorder_loss(logits, labels, mask, None, None, cfg)
        assert loss.item() > 0


class TestSOTAHead:
    def test_forward_with_mask(self):
        head = DisorderSOTAHead(in_dim=64, d_model=32, n_transformer_layers=1, n_heads=4)
        x = torch.randn(2, 16, 64)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[0, 12:] = False
        out = head(x, pad_mask=mask)
        assert out.shape == (2, 16)


class TestSOTAProfile:
    def test_sota_profile_fields(self):
        cfg = TrainConfig.from_profile("sota")
        assert cfg.lora_rank == 64
        assert cfg.head_type == "sota"
        assert cfg.use_dice_loss is True
        assert cfg.compact_checkpoints is True
        assert cfg.use_rdrop is True
        assert cfg.use_tversky_loss is True
        assert cfg.use_swa is True
        assert cfg.use_v6_distill is True
