"""Tests for inference TTA and multi-seed blend."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from colab.inference_tta import mc_dropout_forward_logits
from colab.multi_seed_blend import average_fold_results_multi_seed
from colab.calibration import calibrate_fold_results


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.5)
        self.lin = nn.Linear(4, 1)

    def forward(self, x):
        return self.drop(x).mean(dim=-1)


class TestMCdropoutTTA:
    def test_averages_passes(self):
        model = TinyHead()
        tokens = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3, dtype=torch.bool)

        def fwd(m, t, a, mk, r):
            return m(t)

        out1 = mc_dropout_forward_logits(model, tokens, None, mask, None, 1, fwd)
        out8 = mc_dropout_forward_logits(model, tokens, None, mask, None, 8, fwd)
        assert out1.shape == out8.shape


class TestMultiSeedBlend:
    def test_blend_two_seeds(self):
        proteins = [
            {"id": "P1", "length": 4, "labels": [0, 1, 1, 0]},
            {"id": "P2", "length": 4, "labels": [1, 0, 0, 1]},
        ]
        fr42 = [
            {"fold": 1, "val_probs": np.array([0.2, 0.8, 0.7, 0.3], np.float32),
             "val_labels": np.array([0, 1, 1, 0], np.float32)},
            {"fold": 2, "val_probs": np.array([0.9, 0.1, 0.2, 0.8], np.float32),
             "val_labels": np.array([1, 0, 0, 1], np.float32)},
        ]
        fr43 = [
            {"fold": 1, "val_probs": np.array([0.3, 0.7, 0.8, 0.2], np.float32),
             "val_labels": np.array([0, 1, 1, 0], np.float32)},
            {"fold": 2, "val_probs": np.array([0.85, 0.15, 0.25, 0.75], np.float32),
             "val_labels": np.array([1, 0, 0, 1], np.float32)},
        ]
        blended, report = average_fold_results_multi_seed(
            proteins, {42: fr42, 43: fr43}, n_folds=2,
        )
        assert report["n_seeds"] == 2
        assert len(blended) == 2


class TestChainedCalibration:
    def test_temperature_then_isotonic(self):
        fold_results = [
            {
                "fold": 1,
                "val_probs": np.linspace(0.05, 0.95, 50, dtype=np.float32),
                "val_labels": (np.linspace(0.05, 0.95, 50) > 0.5).astype(np.float32),
            },
        ]
        updated, report = calibrate_fold_results(fold_results, method="temperature_then_isotonic")
        assert report["method"] == "temperature_then_isotonic"
        assert updated[0].get("calibrated") is True
