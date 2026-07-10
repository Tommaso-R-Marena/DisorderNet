"""Tests for colab/biological_utility.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from colab.biological_utility import (
    align_fold_predictions,
    compute_functional_enrichment,
    compute_segment_metrics,
    compute_transition_metrics,
    interval_iou,
    intervals_from_binary,
    run_biological_utility_report,
    save_biological_utility_report,
)
from colab.disordernet_gpu import TrainConfig, process_disprot


class TestIntervals:
    def test_intervals_from_binary(self):
        arr = np.array([0, 1, 1, 1, 0, 1, 0])
        assert intervals_from_binary(arr) == [(1, 4), (5, 6)]

    def test_interval_min_len(self):
        arr = np.array([0, 1, 0, 1, 1, 1])
        assert intervals_from_binary(arr, min_len=2) == [(3, 6)]

    def test_interval_iou(self):
        assert interval_iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
        assert interval_iou((0, 5), (10, 15)) == 0.0


class TestSegmentMetrics:
    def test_perfect_prediction(self):
        y = np.array([0, 0, 1, 1, 1, 0, 0])
        m = compute_segment_metrics(y, y, min_region_len=1)
        assert m.segment_f1 == 1.0
        assert m.mdr_recall == 1.0

    def test_no_prediction(self):
        y = np.array([0, 0, 1, 1, 1, 0, 0])
        p = np.zeros_like(y)
        m = compute_segment_metrics(y, p, min_region_len=1)
        assert m.segment_precision == 0.0
        assert m.segment_recall == 0.0


class TestFunctionalEnrichment:
    def test_binding_recall(self):
        length = 20
        regions = [{"start": 6, "end": 10, "term_norm": "protein binding"}]
        y_pred = np.zeros(length, dtype=np.int8)
        y_pred[5:10] = 1  # indices 5–9, matches 1-based residues 6–10
        probs = y_pred.astype(float)
        labels = np.zeros(length, dtype=np.int8)
        result = compute_functional_enrichment(y_pred, probs, regions, labels)
        assert "protein binding" in result
        assert result["protein binding"]["recall_at_function"] == 1.0


class TestTransitionMetrics:
    def test_transition_subset(self):
        tmask = np.array([0, 0] + [1] * 10 + [0])
        probs = np.linspace(0.05, 0.95, len(tmask))
        labels = (probs >= 0.5).astype(int)
        m = compute_transition_metrics(probs, labels, tmask)
        assert m["n_residues"] == 10
        assert not m["insufficient_data"]
        assert m["auc"] == 1.0


class TestAlignFoldPredictions:
    def test_aligns_lengths(self, sample_disprot_entries):
        cfg = TrainConfig(min_seq_len=20, min_disorder=3, min_order=3)
        proteins, _ = process_disprot(sample_disprot_entries, cfg)
        if len(proteins) < 2:
            pytest.skip("Need at least 2 proteins")

        from sklearn.model_selection import GroupKFold
        n_folds = min(2, len(proteins))
        gkf = GroupKFold(n_splits=n_folds)
        groups = np.arange(len(proteins))
        fold_results = []
        for _, val_idx in gkf.split(groups, groups=groups):
            val_proteins = [proteins[i] for i in val_idx]
            labels = np.concatenate([np.array(p["labels"], dtype=np.float32) for p in val_proteins])
            fold_results.append({
                "fold": len(fold_results) + 1,
                "val_probs": np.random.rand(len(labels)).astype(np.float32),
                "val_labels": labels,
            })

        aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)
        assert len(aligned) == len(proteins)
        assert sum(a["probs"].shape[0] for a in aligned) == sum(p["length"] for p in proteins)


class TestFullReport:
    def test_report_structure(self, sample_disprot_entries, tmp_path: Path):
        cfg = TrainConfig(min_seq_len=20, min_disorder=3, min_order=3)
        proteins, _ = process_disprot(sample_disprot_entries, cfg)
        if not proteins:
            pytest.skip("No proteins after processing")

        # Build mock 2-fold results
        from sklearn.model_selection import GroupKFold
        n_folds = min(2, len(proteins))
        gkf = GroupKFold(n_splits=n_folds)
        groups = np.arange(len(proteins))
        fold_results = []
        for fold_idx, (_, val_idx) in enumerate(gkf.split(groups, groups=groups)):
            val_proteins = [proteins[i] for i in val_idx]
            labels = np.concatenate([np.array(p["labels"], dtype=np.float32) for p in val_proteins])
            # Mixed labels so sklearn metrics are well-defined
            probs = np.clip(labels * 0.6 + np.random.rand(len(labels)) * 0.4, 0, 1)
            fold_results.append({
                "fold": fold_idx + 1,
                "val_probs": probs.astype(np.float32),
                "val_labels": labels,
            })

        report = run_biological_utility_report(
            proteins, fold_results, threshold=0.5, n_folds=n_folds,
        )
        assert "segment_metrics" in report
        assert "functional_enrichment" in report
        assert "transition_zones" in report
        assert 0 <= report["segment_metrics"]["segment_f1"] <= 1

        out = save_biological_utility_report(report, str(tmp_path / "bio.json"))
        assert Path(out).exists()
        loaded = json.loads(Path(out).read_text())
        assert loaded["threshold"] == 0.5
