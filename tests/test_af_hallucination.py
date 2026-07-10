"""Tests for colab/af_hallucination.py (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from colab.af_hallucination import (
    compute_hallucination_metrics,
    compute_plddt_baseline_auc,
    run_af2_af3_comparison_report,
    run_af_rescue_report,
    save_af_rescue_report,
)
from colab.disordernet_gpu import TrainConfig, process_disprot


class TestHallucinationMetrics:
    def test_hallucination_and_rescue(self):
        labels = np.array([1, 1, 1, 0, 0], dtype=np.int8)
        probs = np.array([0.9, 0.2, 0.8, 0.1, 0.9], dtype=np.float32)
        plddt = np.array([85.0, 40.0, 75.0, 90.0, 30.0], dtype=np.float32)
        m = compute_hallucination_metrics(labels, probs, plddt, threshold=0.5, high_plddt_threshold=70.0)
        # disordered: idx 0,1,2 — hallucinated: 0,2 (plddt>=70); rescued: 0,2 (probs>=0.5)
        assert m["n_hallucinated"] == 2
        assert m["hallucination_rate"] == pytest.approx(2 / 3)
        assert m["rescue_rate"] == 1.0
        assert m["n_rescued"] == 2

    def test_no_disordered(self):
        labels = np.zeros(5, dtype=np.int8)
        probs = np.ones(5, dtype=np.float32) * 0.5
        plddt = np.ones(5) * 80
        m = compute_hallucination_metrics(labels, probs, plddt)
        assert m["hallucination_rate"] == 0.0


class TestPlddtBaseline:
    def test_baseline_auc(self):
        labels = np.array([0, 0, 1, 1, 1, 0], dtype=np.int8)
        plddt = np.array([90.0, 80.0, 30.0, 40.0, 20.0, 85.0], dtype=np.float32)
        m = compute_plddt_baseline_auc(labels, plddt)
        assert m["auc"] == 1.0


class TestAfRescueReport:
    def test_full_report(self, sample_disprot_entries, tmp_path):
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
            probs = np.clip(labels * 0.3 + 0.6 * (1 - labels) + 0.1, 0, 1).astype(np.float32)
            fold_results.append({"fold": len(fold_results) + 1, "val_probs": probs, "val_labels": labels})

        plddt_by_protein = {}
        for p in proteins:
            plddt = np.where(np.array(p["labels"]) == 1, 35.0, 85.0).astype(np.float32)
            plddt_by_protein[p["id"]] = plddt

        report = run_af_rescue_report(
            proteins, fold_results, plddt_by_protein, threshold=0.5, n_folds=n_folds,
        )
        assert not report["insufficient_data"]
        assert report["pooled"]["n_residues"] > 0
        assert report["plddt_baseline"]["auc"] is not None
        assert report["delta_auc_vs_plddt_baseline"] is not None

        path = save_af_rescue_report(report, str(tmp_path / "af.json"))
        assert (tmp_path / "af.json").exists()


class TestAf2Af3Comparison:
    def test_comparison_delta(self):
        af2 = {
            "insufficient_data": False,
            "source": "AF2",
            "proteins_with_plddt": 10,
            "pooled": {"hallucination_rate": 0.4, "rescue_rate": 0.5},
            "plddt_baseline": {"auc": 0.7},
            "disordernet_on_af_subset": {"auc": 0.8},
            "delta_auc_vs_plddt_baseline": 0.1,
        }
        af3 = {
            "insufficient_data": False,
            "source": "AF3",
            "proteins_with_plddt": 8,
            "pooled": {"hallucination_rate": 0.35, "rescue_rate": 0.6},
            "plddt_baseline": {"auc": 0.72},
            "disordernet_on_af_subset": {"auc": 0.82},
            "delta_auc_vs_plddt_baseline": 0.1,
        }
        cmp = run_af2_af3_comparison_report(af2, af3)
        assert not cmp["insufficient_data"]
        assert cmp["delta_hallucination_af3_minus_af2"] == pytest.approx(-0.05)
        assert cmp["delta_rescue_af3_minus_af2"] == pytest.approx(0.1)
