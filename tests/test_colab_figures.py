"""Tests for colab/colab_figures.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from colab.colab_figures import generate_all_figures, optimal_threshold


@pytest.fixture
def synthetic_cv_results():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 5000)
    probs = np.clip(labels * 0.65 + rng.random(5000) * 0.35, 0, 1)
    fold_results = []
    for i in range(5):
        sl = slice(i * 1000, (i + 1) * 1000)
        fold_results.append({
            "fold": i + 1,
            "best_auc": 0.85 + i * 0.01,
            "best_ap": 0.50 + i * 0.02,
            "val_probs": probs[sl],
            "val_labels": labels[sl],
            "history": [
                {"epoch": 1, "train_loss": 0.5, "val_loss": 0.45, "val_auc": 0.84},
                {"epoch": 2, "train_loss": 0.4, "val_loss": 0.40, "val_auc": 0.86},
            ],
        })
    return fold_results, labels, probs


class TestOptimalThreshold:
    def test_returns_valid_range(self):
        labels = np.array([0, 0, 1, 1, 1, 0])
        probs = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])
        thresh, f1 = optimal_threshold(labels, probs)
        assert 0.0 <= thresh <= 1.0
        assert 0.0 <= f1 <= 1.0


class TestGenerateFigures:
    def test_writes_all_outputs(self, synthetic_cv_results, tmp_path: Path):
        fold_results, labels, probs = synthetic_cv_results
        prefix = str(tmp_path / "out_")
        metrics = generate_all_figures(
            fold_results, labels, probs, our_auc=0.88, our_ap=0.55, prefix=prefix
        )
        assert "opt_thresh" in metrics
        assert "f1" in metrics
        assert "mcc" in metrics
        for i in range(1, 5):
            assert (tmp_path / f"out_fig{i}_roc_pr.png").exists() or i > 1
        assert (tmp_path / "out_fig1_roc_pr.png").exists()
        assert (tmp_path / "out_fig2_benchmark.png").exists()
        assert (tmp_path / "out_fig3_progression.png").exists()
        assert (tmp_path / "out_fig4_stability.png").exists()
