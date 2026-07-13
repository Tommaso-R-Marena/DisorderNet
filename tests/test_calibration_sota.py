"""Tests for OOF calibration and SOTA training losses."""

from __future__ import annotations

import numpy as np
import torch

from colab.calibration import (
    apply_temperature,
    calibrate_fold_results,
    fit_temperature_scaling,
)
from colab.model_swa import ModelSWA
from colab.sota_losses import (
    batch_mean_tversky_loss,
    rdrop_symmetric_kl,
    v6_distillation_loss,
)


class TestCalibration:
    def test_temperature_improves_or_neutral_on_skewed_probs(self):
        rng = np.random.default_rng(42)
        n = 500
        labels = (rng.random(n) > 0.65).astype(np.float32)
        logits = labels * 2.5 + (1 - labels) * -2.5 + rng.normal(0, 0.3, n)
        probs = 1.0 / (1.0 + np.exp(-logits))
        report = fit_temperature_scaling(labels, probs.astype(np.float32))
        assert not report["insufficient_data"]
        assert report["temperature"] > 0
        assert report["auc_after"] >= report["auc_before"] - 1e-6

    def test_apply_temperature_identity(self):
        p = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        out = apply_temperature(p, 1.0)
        np.testing.assert_allclose(out, p)

    def test_calibrate_fold_results(self):
        fold_results = [
            {
                "fold": 1,
                "val_probs": np.array([0.9, 0.1, 0.85, 0.15], dtype=np.float32),
                "val_labels": np.array([1, 0, 1, 0], dtype=np.float32),
            },
            {
                "fold": 2,
                "val_probs": np.array([0.8, 0.2, 0.75, 0.25], dtype=np.float32),
                "val_labels": np.array([1, 0, 1, 0], dtype=np.float32),
            },
        ]
        updated, report = calibrate_fold_results(fold_results, method="temperature")
        assert len(updated) == 2
        assert updated[0].get("calibrated") is True
        assert "temperature" in report


class TestSOTALossesAdvanced:
    def test_tversky_loss_non_negative(self):
        logits = torch.randn(2, 8)
        labels = (torch.rand(2, 8) > 0.7).float()
        mask = torch.ones(2, 8, dtype=torch.bool)
        loss = batch_mean_tversky_loss(logits, labels, mask)
        assert 0.0 <= loss.item() <= 1.0

    def test_rdrop_kl_zero_for_identical_logits(self):
        logits = torch.randn(2, 6)
        mask = torch.ones(2, 6, dtype=torch.bool)
        kl = rdrop_symmetric_kl(logits, logits, mask)
        assert kl.item() < 1e-5

    def test_v6_distillation_loss(self):
        logits = torch.tensor([[2.0, -2.0]])
        teacher = torch.tensor([[0.9, 0.1]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        loss = v6_distillation_loss(logits, teacher, mask, temperature=1.0)
        assert loss.item() >= 0.0


class TestModelSWA:
    def test_swa_running_average(self):
        m = torch.nn.Linear(2, 1)
        swa = ModelSWA(m)
        with torch.no_grad():
            m.weight.fill_(1.0)
        swa.update(m)
        with torch.no_grad():
            m.weight.fill_(3.0)
        swa.update(m)
        assert swa.swa_state["weight"].mean().item() == 2.0
