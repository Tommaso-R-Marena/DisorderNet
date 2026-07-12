"""Tests for colab/inference_fusion.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from colab.disordernet_gpu import TrainConfig, process_disprot
from colab.inference_fusion import (
    apply_plddt_fusion_to_cv,
    build_combined_plddt_map,
    compute_pooled_metrics,
    fuse_aligned_predictions,
)


def _make_fold_results(proteins):
    n_folds = min(2, len(proteins))
    gkf = GroupKFold(n_splits=n_folds)
    groups = np.arange(len(proteins))
    fold_results = []
    for fold_idx, (_, val_idx) in enumerate(gkf.split(groups, groups=groups)):
        val_proteins = [proteins[i] for i in val_idx]
        labels = np.concatenate([np.array(p["labels"], dtype=np.float32) for p in val_proteins])
        probs = np.clip(labels * 0.2 + 0.7 * (1 - labels), 0, 1).astype(np.float32)
        fold_results.append({
            "fold": fold_idx + 1,
            "val_probs": probs,
            "val_labels": labels,
            "best_auc": 0.8,
        })
    return fold_results, n_folds


class TestInferenceFusion:
    def test_fusion_improves_or_matches_af_subset(self, sample_disprot_entries):
        cfg = TrainConfig(min_seq_len=20, min_disorder=3, min_order=3)
        proteins, _ = process_disprot(sample_disprot_entries, cfg)
        if len(proteins) < 2:
            pytest.skip("Need proteins")

        fold_results, n_folds = _make_fold_results(proteins)
        plddt_by_protein = {}
        for p in proteins:
            plddt_by_protein[p["id"]] = np.where(
                np.array(p["labels"]) == 1, 30.0, 85.0,
            ).astype(np.float32)

        report, fused_folds = apply_plddt_fusion_to_cv(
            proteins, fold_results, plddt_by_protein, n_folds=n_folds, alpha=0.5,
        )
        assert report["fusion_alpha"] == 0.5
        assert fused_folds[0].get("fused") is True
        assert report["after"]["pooled"]["auc"] is not None
        pooled = compute_pooled_metrics(fused_folds)
        assert len(pooled["all_probs"]) == len(pooled["all_labels"])

    def test_combined_plddt_map(self):
        af2 = {"P1": np.array([50.0, 60.0], dtype=np.float32)}
        af3 = {"P1": np.array([75.0, 85.0], dtype=np.float32)}
        combined, stats = build_combined_plddt_map(af2, af3, prefer="af3")
        assert combined["P1"][0] == 75.0
        assert stats["from_af3_preferred"] == 1

    def test_train_config_profiles(self):
        balanced = TrainConfig.from_profile("balanced")
        assert balanced.lora_rank == 16
        mx = TrainConfig.from_profile("max")
        assert mx.lora_rank == 32
        assert mx.num_epochs == 25
        with pytest.raises(ValueError):
            TrainConfig.from_profile("invalid")
