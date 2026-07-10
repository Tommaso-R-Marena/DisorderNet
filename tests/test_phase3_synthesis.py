"""Tests for colab/phase3_synthesis.py (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest

from colab.phase3_synthesis import (
    bootstrap_ci,
    build_benchmark_ranking,
    calibrate_plddt,
    compute_calibration_metrics,
    find_optimal_fusion_alpha,
    fuse_disorder_score,
    mcnemar_test,
    run_phase3_integrated_report,
    run_structure_calibration_report,
    save_phase3_report,
)


class TestFusion:
    def test_alpha_extremes(self):
        dn = np.array([0.9, 0.1], dtype=np.float32)
        plddt = np.array([90.0, 30.0], dtype=np.float32)
        pure_dn = fuse_disorder_score(dn, plddt, alpha=1.0)
        assert list(pure_dn) == pytest.approx(list(dn))
        pure_base = fuse_disorder_score(dn, plddt, alpha=0.0)
        assert pure_base[0] == pytest.approx(0.1)
        assert pure_base[1] == pytest.approx(0.7)

    def test_calibrate_reduces_high_confidence_idr(self):
        plddt = np.array([85.0, 90.0], dtype=np.float32)
        dn = np.array([0.8, 0.9], dtype=np.float32)
        cal = calibrate_plddt(plddt, dn)
        assert cal[0] < plddt[0]
        assert cal[1] < plddt[1]


class TestAlphaSearch:
    def test_finds_reasonable_alpha(self):
        labels = np.array([0, 0, 1, 1, 1, 0], dtype=np.int8)
        dn = np.array([0.1, 0.2, 0.9, 0.85, 0.8, 0.15], dtype=np.float32)
        plddt = np.array([90.0, 85.0, 35.0, 40.0, 30.0, 80.0], dtype=np.float32)
        result = find_optimal_fusion_alpha(labels, dn, plddt)
        assert 0.0 <= result["best_alpha"] <= 1.0
        assert result["best_auc"] is not None
        assert len(result["curve"]) > 0


class TestCalibrationMetrics:
    def test_fusion_beats_or_matches_baseline(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=np.int8)
        dn = np.array([0.05, 0.1, 0.95, 0.9, 0.85, 0.2, 0.8, 0.15], dtype=np.float32)
        plddt = np.array([92.0, 88.0, 75.0, 80.0, 35.0, 85.0, 40.0, 90.0], dtype=np.float32)
        m = compute_calibration_metrics(labels, dn, plddt)
        assert not m["insufficient_data"]
        assert m["fusion"]["auc"] is not None
        assert m["disordernet"]["auc"] is not None
        assert m["hallucination_reduction"]["raw_n_hallucinated"] >= 1


class TestBootstrap:
    def test_ci_brackets_point(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1], dtype=np.int8)
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.6, 0.85, 0.4, 0.75], dtype=np.float32)
        b = bootstrap_ci(labels, scores, n_boot=100, seed=0)
        assert not b["insufficient_data"]
        assert b["ci_low"] <= b["point"] <= b["ci_high"]


class TestMcNemar:
    def test_detects_difference(self):
        labels = np.array([1, 1, 0, 0, 1, 0], dtype=np.int8)
        pred_a = np.array([1, 0, 0, 1, 1, 0], dtype=np.int8)
        pred_b = np.array([0, 0, 0, 0, 1, 1], dtype=np.int8)
        m = mcnemar_test(pred_a, pred_b, labels)
        assert m["b"] + m["c"] > 0
        assert 0.0 <= m["p_value"] <= 1.0


class TestBenchmark:
    def test_ranking(self):
        r = build_benchmark_ranking(0.85)
        assert r["rank_among_published"] >= 1
        assert r["beats_af3_plddt"]
        assert r["comparable_head_to_head"] is False
        assert r["delta_vs_af3"] == pytest.approx(0.103)


class TestIntegratedReport:
    def test_full_report(self, sample_disprot_entries, tmp_path):
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
        for _, val_idx in gkf.split(groups, groups=groups):
            val_proteins = [proteins[i] for i in val_idx]
            labels = np.concatenate([np.array(p["labels"], dtype=np.float32) for p in val_proteins])
            probs = np.clip(labels * 0.2 + 0.7 * (1 - labels), 0, 1).astype(np.float32)
            fold_results.append({"fold": len(fold_results) + 1, "val_probs": probs, "val_labels": labels})

        plddt_by_protein = {}
        for p in proteins:
            plddt_by_protein[p["id"]] = np.where(
                np.array(p["labels"]) == 1, 35.0, 85.0,
            ).astype(np.float32)

        cal = run_structure_calibration_report(
            proteins, fold_results, plddt_by_protein, n_folds=n_folds, n_boot=50,
        )
        assert not cal["insufficient_data"]

        bio = {"segment_metrics": {"segment_f1": 0.5}, "transition_zones": {"auc": 0.8}}
        af = {
            "insufficient_data": False,
            "pooled": {"hallucination_rate": 0.2, "rescue_rate": 0.6},
            "delta_auc_vs_plddt_baseline": 0.1,
        }
        report = run_phase3_integrated_report(
            cv_pooled={"auc": 0.88, "ap": 0.85, "f1": 0.7, "mcc": 0.5, "opt_threshold": 0.5},
            bio_report=bio,
            af_report=af,
            calibration_report=cal,
        )
        assert not report["insufficient_data"]
        assert "headline" in report
        assert report["benchmark_ranking"]["beats_af3_plddt"]
        path = save_phase3_report(report, str(tmp_path / "p3.json"))
        assert (tmp_path / "p3.json").exists()
