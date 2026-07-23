"""Tests for run_v7 pure helper functions (feature build, smoothing, metrics)."""
from __future__ import annotations

import numpy as np

from run_v7 import build_features, smooth, evaluate


def test_build_features_dimension():
    ph = np.zeros((40, 118), dtype=np.float32)
    ep = np.random.RandomState(0).randn(40, 96).astype(np.float32)
    X = build_features(ph, ep)
    # physics(118) + ep(96) + 4 window means(4*96) + 2 variances(2*96)
    # + gmean32 + gstd32 + enorm1
    assert X.shape == (40, 118 + 96 + 4 * 96 + 2 * 96 + 32 + 32 + 1)
    assert X.dtype == np.float32


def test_smooth_preserves_length_and_averages():
    v = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    out = smooth(v, 3)
    assert len(out) == len(v)
    # window=1 is a no-op
    assert np.array_equal(smooth(v, 1), v)
    # smoothing pulls values toward the local mean (reduces variance)
    assert out.var() < v.var()


def test_evaluate_perfect_and_keys():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.2, 0.7, 0.99])
    m = evaluate(y, p)
    assert set(m) == {"auc_roc", "avg_precision", "f1", "mcc", "precision",
                      "recall", "balanced_acc"}
    assert m["auc_roc"] == 1.0
    assert 0.0 <= m["f1"] <= 1.0
