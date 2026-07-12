"""Tests for colab/segment_postprocess.py."""

from __future__ import annotations

import numpy as np

from colab.segment_postprocess import (
    composite_early_stop_score,
    pooled_segment_f1,
    postprocess_binary,
    probs_to_postprocessed_binary,
)


class TestPostprocessBinary:
    def test_closes_short_gaps(self):
        binary = np.array([1, 1, 0, 1, 1], dtype=np.int8)
        out = postprocess_binary(binary, min_len=1, max_gap=1)
        assert out.tolist() == [1, 1, 1, 1, 1]

    def test_removes_short_runs(self):
        binary = np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0], dtype=np.int8)
        out = postprocess_binary(binary, min_len=5, max_gap=0)
        assert out[1] == 0
        assert out[5] == 1

    def test_probs_to_postprocessed(self):
        probs = np.array([0.1, 0.6, 0.55, 0.4, 0.7, 0.8, 0.2], dtype=np.float32)
        out = probs_to_postprocessed_binary(probs, threshold=0.5, min_len=2, max_gap=1)
        assert out.dtype == np.int8


class TestSegmentF1:
    def test_pooled_segment_f1_single_protein(self):
        proteins = [{
            "id": "P1",
            "length": 10,
            "labels": [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        }]
        labels = np.array(proteins[0]["labels"], dtype=np.float32)
        probs = np.array([0.1, 0.2, 0.8, 0.85, 0.9, 0.88, 0.87, 0.2, 0.1, 0.1], dtype=np.float32)
        f1 = pooled_segment_f1(proteins, probs, labels, apply_postprocess=True)
        assert 0.0 <= f1 <= 1.0

    def test_composite_score_weights(self):
        score = composite_early_stop_score(0.8, 0.7, 0.3)
        assert abs(score - (0.5 * 0.8 + 0.3 * 0.7 + 0.2 * 0.3)) < 1e-6
