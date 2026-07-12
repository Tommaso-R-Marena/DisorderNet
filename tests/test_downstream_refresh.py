"""Tests for colab/downstream_refresh.py and combined pLDDT fusion."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from colab.disordernet_gpu import TrainConfig, merge_plddt_for_training, process_disprot
from colab.inference_fusion import build_combined_plddt_map


class TestCombinedPlddt:
    def test_prefer_af3(self):
        af2 = {"P1": np.array([50.0, 60.0], dtype=np.float32)}
        af3 = {"P1": np.array([70.0, 80.0], dtype=np.float32), "P2": np.array([40.0], dtype=np.float32)}
        combined, stats = build_combined_plddt_map(af2, af3, prefer="af3")
        assert combined["P1"][0] == 70.0
        assert "P2" in combined
        assert stats["from_af3_preferred"] == 1
        assert stats["from_af3_only"] == 1

    def test_merge_plddt_for_training(self):
        af2 = {"A": np.ones(3, dtype=np.float32) * 50}
        af3 = {"A": np.ones(3, dtype=np.float32) * 90, "B": np.ones(2, dtype=np.float32) * 40}
        merged = merge_plddt_for_training(af2, af3, prefer_af3=True)
        assert merged["A"][0] == 90.0
        assert "B" in merged


class TestDownstreamRefresh:
    def test_refresh_metrics(self, sample_disprot_entries):
        from colab.downstream_refresh import refresh_downstream_metrics

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
            probs = np.clip(labels * 0.3 + 0.65 * (1 - labels), 0, 1).astype(np.float32)
            fold_results.append({"val_probs": probs, "val_labels": labels})

        out = refresh_downstream_metrics(
            proteins, fold_results, n_folds=n_folds, print_reports=False,
        )
        assert out["our_auc"] is not None
        assert "caid_report" in out
        assert "bio_report" in out
