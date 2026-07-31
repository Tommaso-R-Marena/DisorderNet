"""Tests for SOTA training modules."""

from __future__ import annotations

import torch

from colab.disordernet_gpu import TrainConfig
from colab.sota_losses import (
    batch_mean_dice_loss,
    batch_mean_tversky_loss,
    composite_disorder_loss,
    soft_dice_from_logits,
)
from colab.sota_heads import DisorderSOTAHead


def _ref_batch_dice(logits, labels, mask, smooth=1.0):
    """Per-sequence reference: loop over rows, slice out the valid residues."""
    losses = [
        soft_dice_from_logits(logits[b][mask[b]], labels[b][mask[b]], smooth=smooth)
        for b in range(logits.shape[0])
        if mask[b].sum() >= 2
    ]
    return torch.stack(losses).mean() if losses else torch.zeros((), dtype=logits.dtype)


def _ref_batch_tversky(logits, labels, mask, alpha=0.3, beta=0.7, smooth=1.0):
    losses = []
    for b in range(logits.shape[0]):
        m = mask[b]
        if m.sum() < 2:
            continue
        probs = torch.sigmoid(logits[b][m])
        y = labels[b][m].float()
        tp = (probs * y).sum()
        fn = (y * (1.0 - probs)).sum()
        fp = ((1.0 - y) * probs).sum()
        losses.append(1.0 - (tp + smooth) / (tp + alpha * fn + beta * fp + smooth))
    return torch.stack(losses).mean() if losses else torch.zeros((), dtype=logits.dtype)


def _ragged_batch(seed=0, batch=6, length=40):
    torch.manual_seed(seed)
    logits = torch.randn(batch, length) * 2
    labels = (torch.rand(batch, length) < 0.4).float()
    mask = torch.zeros(batch, length, dtype=torch.bool)
    for b in range(batch):
        mask[b, : int(torch.randint(2, length + 1, (1,)))] = True
    mask[0, :] = False       # fully padded sequence
    mask[1, :] = False
    mask[1, :1] = True       # single valid residue -> below the >=2 cutoff
    return logits, labels, mask


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

    def test_batched_dice_matches_per_sequence_reference(self):
        logits, labels, mask = _ragged_batch(seed=1)
        assert torch.allclose(
            batch_mean_dice_loss(logits, labels, mask),
            _ref_batch_dice(logits, labels, mask),
            atol=1e-6,
        )

    def test_batched_tversky_matches_per_sequence_reference(self):
        logits, labels, mask = _ragged_batch(seed=2)
        assert torch.allclose(
            batch_mean_tversky_loss(logits, labels, mask),
            _ref_batch_tversky(logits, labels, mask),
            atol=1e-6,
        )

    def test_region_losses_are_zero_when_everything_is_padding(self):
        logits, labels, _ = _ragged_batch(seed=3)
        empty = torch.zeros_like(labels, dtype=torch.bool)
        assert batch_mean_dice_loss(logits, labels, empty).item() == 0.0
        assert batch_mean_tversky_loss(logits, labels, empty).item() == 0.0

    def test_composite_weighted_focal_matches_explicit_formula(self):
        cfg = TrainConfig(
            use_focal_loss=True, focal_gamma=2.0, label_smoothing=0.0,
            use_dice_loss=False, use_tversky_loss=False,
        )
        logits, labels, mask = _ragged_batch(seed=4)
        sample_weight = torch.rand_like(labels) + 0.5
        pos_weight = torch.tensor([3.0])

        flat_logits, flat_labels = logits[mask], labels[mask]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            flat_logits, flat_labels, pos_weight=pos_weight, reduction="none",
        )
        probs = torch.sigmoid(flat_logits)
        pt = torch.where(flat_labels > 0.5, probs, 1.0 - probs)
        w = sample_weight[mask]
        expected = (bce * ((1.0 - pt) ** 2.0) * w).sum() / w.sum().clamp(min=1.0)

        got = composite_disorder_loss(
            logits, labels, mask, pos_weight, sample_weight, cfg,
        )
        assert torch.allclose(got, expected, atol=1e-6)

    def test_composite_unweighted_focal_is_a_plain_mean(self):
        cfg = TrainConfig(
            use_focal_loss=True, focal_gamma=2.0, label_smoothing=0.0,
            use_dice_loss=False, use_tversky_loss=False,
        )
        logits, labels, mask = _ragged_batch(seed=5)
        flat_logits, flat_labels = logits[mask], labels[mask]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            flat_logits, flat_labels, reduction="none",
        )
        probs = torch.sigmoid(flat_logits)
        pt = torch.where(flat_labels > 0.5, probs, 1.0 - probs)
        expected = (bce * ((1.0 - pt) ** 2.0)).mean()

        got = composite_disorder_loss(logits, labels, mask, None, None, cfg)
        assert torch.allclose(got, expected, atol=1e-6)


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

    def test_ultra_profile_fields(self):
        cfg = TrainConfig.from_profile("ultra")
        assert cfg.lora_rank == 128
        assert cfg.use_rich_features is True
        assert cfg.fusion_type == "attention"
        assert cfg.lora_on_ffn is True
        assert cfg.lora_on_out_proj is True
        assert cfg.unfreeze_last_layers == 2
