"""Tests for colab/quick_screen.py (CPU-only)."""

from __future__ import annotations

import numpy as np

from colab.disordernet_gpu import TrainConfig, process_disprot
from colab.quick_screen import (
    SCREEN_MODES,
    assess_breakthrough_potential,
    expected_ultra_uplift,
    subsample_proteins_stratified,
)


class TestSubsample:
    def test_no_subsample_when_small(self, sample_disprot_entries):
        cfg = TrainConfig(min_seq_len=20, min_disorder=3, min_order=3)
        proteins, _ = process_disprot(sample_disprot_entries, cfg)
        out, meta = subsample_proteins_stratified(proteins, n_target=100, seed=42)
        assert len(out) == len(proteins)
        assert meta["subsampled"] is False

    def test_stratified_subsample(self, sample_disprot_entries):
        cfg = TrainConfig(min_seq_len=20, min_disorder=3, min_order=3)
        proteins, _ = process_disprot(sample_disprot_entries, cfg)
        # duplicate to have enough proteins
        big = proteins * 20
        for i, p in enumerate(big):
            p = dict(p)
            p["id"] = f"{p['id']}_{i}"
            big[i] = p
        out, meta = subsample_proteins_stratified(big, n_target=10, seed=7)
        assert len(out) == 10
        assert meta["subsampled"] is True


class TestVerdict:
    def test_high_tier(self):
        v = assess_breakthrough_potential(
            0.86, 0.89, v6_pooled_auc=0.84, fold_aucs=[0.88, 0.89, 0.90], mode="standard",
        )
        assert v.tier == "HIGH"
        assert v.proceed_full_ultra is True

    def test_stop_tier(self):
        # Weak stacked under mode-aware uplift still STOPS
        v = assess_breakthrough_potential(0.72, 0.75, v6_pooled_auc=0.71, mode="standard")
        assert v.tier == "STOP"
        assert v.proceed_full_ultra is False

    def test_moderate_tier(self):
        v = assess_breakthrough_potential(
            0.82, 0.86, v6_pooled_auc=0.83, fold_aucs=[0.85, 0.86, 0.87], mode="standard",
        )
        assert v.tier in ("MODERATE", "HIGH")

    def test_flash_cannot_alone_proceed(self):
        # Even a strong flash score must not alone green-light 24h ultra
        v = assess_breakthrough_potential(
            0.86, 0.89, v6_pooled_auc=0.84, fold_aucs=[0.88, 0.89, 0.90], mode="flash",
        )
        assert v.proceed_full_ultra is False

    def test_mode_aware_uplift(self):
        assert expected_ultra_uplift("flash")[1] > expected_ultra_uplift("standard")[1]
        assert expected_ultra_uplift("standard")[1] >= expected_ultra_uplift("paradigm")[1]

    def test_legacy_weak_screen_score_not_false_stop_on_projection_floor(self):
        # With standard uplift, 0.80 stacked projects into LOW/MODERATE band, not STOP
        v = assess_breakthrough_potential(0.78, 0.80, v6_pooled_auc=0.79, mode="standard")
        assert v.tier in ("LOW", "MODERATE")
        assert v.projected_full_auc_high >= 0.85


class TestScreenProfiles:
    def test_screen_profile_exists(self):
        cfg = TrainConfig.from_profile("screen", n_folds=3)
        assert cfg.num_epochs == 8
        assert cfg.head_type == "cnn"
        assert cfg.use_v6_distill is False

    def test_screen_plus_profile_is_mini_ultra(self):
        cfg = TrainConfig.from_profile("screen_plus")
        assert cfg.use_rich_features is True
        assert cfg.head_type == "sota"
        assert cfg.lora_on_ffn is True
        assert cfg.split_method == "homology"
        assert cfg.num_epochs >= 12
        assert cfg.early_stop_mode == "composite"

    def test_modes_defined(self):
        assert "flash" in SCREEN_MODES
        assert "standard" in SCREEN_MODES
        assert "paradigm" in SCREEN_MODES

    def test_standard_uses_screen_plus(self):
        assert SCREEN_MODES["standard"]["train_profile"] == "screen_plus"
        assert SCREEN_MODES["flash"]["train_profile"] == "screen"
        assert SCREEN_MODES["paradigm"]["train_profile"] == "screen_plus"
