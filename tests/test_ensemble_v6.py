"""Tests for colab/ensemble_v6.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from colab.disordernet_gpu import TrainConfig, process_disprot
from colab.ensemble_v6 import (
    apply_gpu_v6_ensemble,
    build_v6_features,
    find_optimal_blend_weight,
    run_v6_lite_oof,
)


class TestV6Features:
    def test_build_v6_features_shape(self):
        seq = "ACDEFGHIKLMNPQRSTVWY" * 3
        feats = build_v6_features(seq)
        assert feats.shape[0] == len(seq)
        assert feats.shape[1] > 50


class TestV6LiteOOF:
    def test_oof_alignment(self, sample_disprot_entries):
        cfg = TrainConfig(min_seq_len=20, min_disorder=3, min_order=3)
        proteins, _ = process_disprot(sample_disprot_entries, cfg)
        if len(proteins) < 2:
            pytest.skip("Need proteins")

        n_folds = min(2, len(proteins))
        oof_probs, oof_labels, fold_meta = run_v6_lite_oof(proteins, n_folds=n_folds)
        expected_len = sum(p["length"] for p in proteins)
        assert len(oof_probs) == expected_len
        assert len(oof_labels) == expected_len
        assert len(fold_meta) == n_folds


class TestBlendWeight:
    def test_find_optimal_weight(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1], dtype=np.int8)
        gpu = np.linspace(0.1, 0.9, len(labels), dtype=np.float32)
        v6 = np.linspace(0.15, 0.85, len(labels), dtype=np.float32)
        result = find_optimal_blend_weight(labels, gpu, v6)
        assert 0.0 <= result["best_weight"] <= 1.0
        assert result["best_auc"] is not None


class TestGpuV6Ensemble:
    def test_ensemble_updates_fold_results(self, sample_disprot_entries):
        cfg = TrainConfig(min_seq_len=20, min_disorder=3, min_order=3)
        proteins, _ = process_disprot(sample_disprot_entries, cfg)
        if len(proteins) < 2:
            pytest.skip("Need proteins")

        n_folds = min(2, len(proteins))
        gkf = GroupKFold(n_splits=n_folds)
        groups = np.arange(len(proteins))
        fold_results = []
        for fold_idx, (_, val_idx) in enumerate(gkf.split(groups, groups=groups)):
            val_proteins = [proteins[i] for i in val_idx]
            labels = np.concatenate([np.array(p["labels"], dtype=np.float32) for p in val_proteins])
            probs = np.clip(labels * 0.3 + 0.65 * (1 - labels), 0, 1).astype(np.float32)
            fold_results.append({
                "fold": fold_idx + 1,
                "val_probs": probs,
                "val_labels": labels,
                "best_auc": 0.8,
            })

        report, ensembled, v6_probs = apply_gpu_v6_ensemble(
            proteins, fold_results, n_folds=n_folds, weight=0.3,
        )
        assert report["ensemble_weight"] == 0.3
        assert ensembled[0].get("ensembled") is True
        assert len(v6_probs) == len(proteins)
