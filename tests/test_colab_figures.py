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


class TestBiologicalUtilityFigure:
    def test_writes_fig5(self, tmp_path: Path):
        from colab.colab_figures import generate_biological_utility_figure

        report = {
            "functional_enrichment": {
                "protein binding": {
                    "recall_at_function": 0.82,
                    "precision_at_idr": 0.61,
                    "enrichment_vs_disorder_rate": 2.3,
                    "n_residues": 500,
                },
            },
            "segment_metrics": {
                "segment_precision": 0.71,
                "segment_recall": 0.76,
                "segment_f1": 0.73,
                "mdr_recall": 0.69,
                "mean_segment_iou": 0.66,
            },
        }
        generate_biological_utility_figure(report, prefix=str(tmp_path / "out_"))
        assert (tmp_path / "out_fig5_biological_utility.png").exists()
        assert (tmp_path / "out_fig5_biological_utility.pdf").exists()


class TestAfRescueFigure:
    def test_writes_fig6(self, tmp_path: Path):
        from colab.colab_figures import generate_af_rescue_figure

        report = {
            "insufficient_data": False,
            "pooled": {
                "hallucination_rate": 0.22,
                "rescue_rate": 0.75,
                "rescue_of_disordered": 0.18,
            },
            "plddt_baseline": {"auc": 0.74},
            "disordernet_on_af_subset": {"auc": 0.86},
        }
        generate_af_rescue_figure(report, prefix=str(tmp_path / "out_"))
        assert (tmp_path / "out_fig6_af_rescue.png").exists()


class TestAf2Af3ComparisonFigure:
    def test_writes_fig7(self, tmp_path: Path):
        from colab.colab_figures import generate_af2_af3_comparison_figure

        comparison = {
            "insufficient_data": False,
            "af2": {
                "hallucination_rate": 0.3,
                "rescue_rate": 0.5,
                "plddt_baseline_auc": 0.74,
                "disordernet_auc": 0.85,
            },
            "af3": {
                "hallucination_rate": 0.25,
                "rescue_rate": 0.55,
                "plddt_baseline_auc": 0.76,
                "disordernet_auc": 0.87,
            },
        }
        generate_af2_af3_comparison_figure(comparison, prefix=str(tmp_path / "out_"))
        assert (tmp_path / "out_fig7_af2_af3_comparison.png").exists()


class TestPhase3Figure:
    def test_writes_fig8(self, tmp_path: Path):
        from colab.colab_figures import generate_phase3_figure

        report = {
            "insufficient_data": False,
            "headline": "GPU AUC 0.880 ranks #3/12",
            "benchmark_ranking": {
                "our_auc": 0.88,
                "rank_among_published": 3,
                "n_methods": 12,
                "delta_vs_af3": 0.133,
                "delta_vs_sota_esmdispred": -0.015,
                "table": [
                    {"method": "AF3-pLDDT", "auc": 0.747},
                    {"method": "DisorderNet v6", "auc": 0.831},
                    {"method": "ESM2_650M-LoRA", "auc": 0.88},
                ],
            },
            "phase_summaries": {
                "phase1_biological_utility": {"segment_f1": 0.55},
                "phase2_af_rescue": {"available": True, "hallucination_rate": 0.22, "rescue_rate": 0.7},
                "phase3_calibration": {"fusion_auc": 0.87, "fusion_alpha": 0.6},
            },
            "structure_calibration": {
                "insufficient_data": False,
                "calibration": {
                    "plddt_baseline": {"auc": 0.74},
                    "disordernet": {"auc": 0.86},
                    "fusion": {"auc": 0.87},
                    "calibrated_plddt": {"auc": 0.75},
                    "hallucination_reduction": {
                        "raw_n_hallucinated": 100,
                        "calibrated_n_hallucinated": 40,
                    },
                },
            },
        }
        generate_phase3_figure(report, prefix=str(tmp_path / "out_"))
        assert (tmp_path / "out_fig8_phase3_synthesis.png").exists()
