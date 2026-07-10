"""Tests for colab/caid_reporting.py."""

from __future__ import annotations

import numpy as np
import pytest

from colab.caid_reporting import (
    compute_caid_metrics,
    compute_f1_max,
    run_full_caid_report,
    run_stratified_caid_report,
    save_caid_report,
)
from colab.disordernet_gpu import TrainConfig, process_disprot


class TestF1Max:
    def test_perfect_scores(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int8)
        probs = np.array([0.0, 0.1, 0.9, 0.95], dtype=np.float32)
        r = compute_f1_max(labels, probs)
        assert r["f1_max"] == 1.0

    def test_insufficient_classes(self):
        labels = np.zeros(5, dtype=np.int8)
        probs = np.ones(5, dtype=np.float32) * 0.5
        r = compute_f1_max(labels, probs)
        assert r["f1_max"] is None


class TestCaidMetrics:
    def test_full_metrics(self):
        labels = np.array([0, 0, 1, 1, 1, 0], dtype=np.int8)
        probs = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3], dtype=np.float32)
        m = compute_caid_metrics(labels, probs)
        assert not m["insufficient_data"]
        assert m["auc"] == 1.0
        assert m["f1_max"] is not None
        assert m["mcc_at_f1_max"] is not None


class TestStratifiedReport:
    def test_stratified_bins(self, sample_disprot_entries):
        cfg = TrainConfig(min_seq_len=20, min_disorder=3, min_order=3)
        proteins, _ = process_disprot(sample_disprot_entries, cfg)
        if len(proteins) < 2:
            pytest.skip("Need proteins")

        from sklearn.model_selection import GroupKFold

        n_folds = min(2, len(proteins))
        gkf = GroupKFold(n_splits=n_folds)
        groups = np.arange(len(proteins))
        fold_results = []
        for _, val_idx in gkf.split(groups, groups=groups):
            val_proteins = [proteins[i] for i in val_idx]
            labels = np.concatenate([np.array(p["labels"], dtype=np.float32) for p in val_proteins])
            probs = np.clip(labels * 0.3 + 0.5, 0, 1).astype(np.float32)
            fold_results.append({"fold": len(fold_results) + 1, "val_probs": probs, "val_labels": labels})

        report = run_stratified_caid_report(proteins, fold_results, n_folds=n_folds)
        assert not report["pooled"]["insufficient_data"]
        assert "by_disorder_fraction" in report
        assert "by_length" in report
        assert "by_organism" in report
        assert "Homo sapiens" in report["by_organism"] or "unknown" in report["by_organism"]

    def test_full_report_save(self, sample_disprot_entries, tmp_path):
        cfg = TrainConfig(min_seq_len=20, min_disorder=3, min_order=3)
        proteins, _ = process_disprot(sample_disprot_entries, cfg)
        if not proteins:
            pytest.skip("No proteins")

        from sklearn.model_selection import GroupKFold

        n_folds = min(2, len(proteins))
        gkf = GroupKFold(n_splits=n_folds)
        groups = np.arange(len(proteins))
        fold_results = []
        for _, val_idx in gkf.split(groups, groups=groups):
            val_proteins = [proteins[i] for i in val_idx]
            labels = np.concatenate([np.array(p["labels"], dtype=np.float32) for p in val_proteins])
            probs = np.clip(labels * 0.3 + 0.5, 0, 1).astype(np.float32)
            fold_results.append({"fold": 1, "val_probs": probs, "val_labels": labels})

        full = run_full_caid_report(proteins, fold_results, n_folds=n_folds)
        assert "per_fold" in full
        path = save_caid_report(full, str(tmp_path / "caid.json"))
        assert (tmp_path / "caid.json").exists()
