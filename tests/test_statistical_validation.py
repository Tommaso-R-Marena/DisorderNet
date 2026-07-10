"""Tests for colab/statistical_validation.py."""

from __future__ import annotations

import numpy as np
import pytest

from colab.statistical_validation import (
    run_cv_fold_stability_report,
    run_full_statistical_validation,
    run_per_fold_paired_comparison,
    sign_test_two_sided,
)


class TestSignTest:
    def test_tie(self):
        r = sign_test_two_sided(0, 0)
        assert r["insufficient_data"]

    def test_clear_winner(self):
        r = sign_test_two_sided(5, 0)
        assert r["p_value"] < 0.1
        assert r["favors"] == "a"


class TestCvStability:
    def test_fold_summary(self):
        fold_results = [
            {"fold": i, "best_auc": 0.82 + 0.01 * i}
            for i in range(5)
        ]
        r = run_cv_fold_stability_report(fold_results)
        assert r["mean_auc"] is not None
        assert len(r["fold_aucs"]) == 5
        assert r["bootstrap_fold_auc"]["point"] is not None


class TestPairedComparison:
    def test_per_fold_vs_baseline(self, sample_disprot_entries):
        from sklearn.model_selection import GroupKFold

        from colab.disordernet_gpu import TrainConfig, process_disprot

        cfg = TrainConfig(min_seq_len=20, min_disorder=3, min_order=3)
        proteins, _ = process_disprot(sample_disprot_entries, cfg)
        if len(proteins) < 2:
            pytest.skip("Need proteins")

        n_folds = min(2, len(proteins))
        gkf = GroupKFold(n_splits=n_folds)
        groups = np.arange(len(proteins))
        fold_results = []
        plddt_by_protein = {}

        for fold_idx, (_, val_idx) in enumerate(gkf.split(groups, groups=groups)):
            val_proteins = [proteins[i] for i in val_idx]
            labels = np.concatenate([np.array(p["labels"], dtype=np.float32) for p in val_proteins])
            probs = np.clip(labels * 0.2 + 0.7 * (1 - labels), 0, 1).astype(np.float32)
            fold_results.append({"fold": fold_idx + 1, "val_probs": probs, "val_labels": labels})

        for p in proteins:
            plddt_by_protein[p["id"]] = np.where(
                np.array(p["labels"]) == 1, 35.0, 85.0,
            ).astype(np.float32)

        r = run_per_fold_paired_comparison(proteins, fold_results, plddt_by_protein, n_folds=n_folds)
        valid = [f for f in r["per_fold"] if not f.get("insufficient_data")]
        assert len(valid) >= 1
        assert r["summary"]["mean_delta_auc"] is not None

    def test_full_validation(self, sample_disprot_entries):
        from sklearn.model_selection import GroupKFold

        from colab.disordernet_gpu import TrainConfig, process_disprot

        cfg = TrainConfig(min_seq_len=20, min_disorder=3, min_order=3)
        proteins, _ = process_disprot(sample_disprot_entries, cfg)
        if not proteins:
            pytest.skip("No proteins")

        n_folds = min(2, len(proteins))
        gkf = GroupKFold(n_splits=n_folds)
        groups = np.arange(len(proteins))
        fold_results = []
        for _, val_idx in gkf.split(groups, groups=groups):
            val_proteins = [proteins[i] for i in val_idx]
            labels = np.concatenate([np.array(p["labels"], dtype=np.float32) for p in val_proteins])
            probs = np.clip(labels * 0.3 + 0.5, 0, 1).astype(np.float32)
            fold_results.append({"fold": 1, "best_auc": 0.8, "val_probs": probs, "val_labels": labels})

        r = run_full_statistical_validation(proteins, fold_results, plddt_by_protein=None)
        assert "cv_fold_stability" in r
        assert r["paired_af_baseline"]["insufficient_data"]
