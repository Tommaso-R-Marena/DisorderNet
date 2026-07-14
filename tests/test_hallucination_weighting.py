"""Tests for AF hallucination hard-negative weighting in training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from colab.disordernet_gpu import (
    TrainConfig,
    DisProtDataset,
    _hallucination_weight_mask,
    process_disprot,
)


class TestHallucinationWeights:
    def test_hallucination_mask_upweights_disordered_high_plddt(self):
        cfg = TrainConfig(
            use_hallucination_weighting=True,
            hallucination_weight=3.0,
            high_plddt_threshold=70.0,
        )
        labels = [0, 1, 1, 0]
        plddt = np.array([50.0, 85.0, 40.0, 90.0], dtype=np.float32)
        weights = _hallucination_weight_mask(labels, plddt, cfg)
        assert weights == [1.0, 3.0, 1.0, 1.0]

    def test_disabled_returns_ones(self):
        cfg = TrainConfig(use_hallucination_weighting=False)
        labels = [0, 1]
        plddt = np.array([85.0, 85.0], dtype=np.float32)
        weights = _hallucination_weight_mask(labels, plddt, cfg)
        assert weights == [1.0, 1.0]

    def test_dataset_bakes_combined_weights(self, sample_disprot_entries, mock_batch_converter, tmp_path):
        cfg = TrainConfig(
            min_seq_len=20,
            min_disorder=3,
            min_order=3,
            use_hallucination_weighting=True,
            hallucination_weight=4.0,
            high_plddt_threshold=70.0,
            boundary_weight=2.0,
            af_plddt_cache_dir=str(tmp_path),
        )
        proteins, _ = process_disprot(sample_disprot_entries, cfg)
        p = next(x for x in proteins if x["id"] == "DP_TEST01")

        cache_path = tmp_path / "P12345.json"
        plddt = np.where(np.array(p["labels"]) == 1, 85.0, 30.0).astype(np.float32)
        cache_path.write_text(json.dumps({
            "target_sequence": p["sequence"],
            "plddt": plddt.tolist(),
        }))

        ds = DisProtDataset([p], mock_batch_converter, cfg=cfg)
        _, labels, mask, _, sample_weight, _, _, _ = ds[0]

        dis_mask = labels[mask] == 1
        dis_weights = sample_weight[mask][dis_mask]
        assert (dis_weights >= 4.0).any()
