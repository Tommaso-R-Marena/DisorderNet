"""Which statistic reproduces across identical reruns?

Four runs of the same configuration (ultra / 650M / DisProt / homology, same
seed, same code) exist in this project's results tree:

      run     GPU pooled  mean-folds
  j29683091       0.7203      0.7491
  j29683038       0.7137      0.7514
  j29682871       0.7454      0.7551
  j29683037       0.7234      0.7405

Pooled AUC moves by 0.0317 across those reruns; mean-of-folds by 0.0147. That
is the cross-fold calibration effect showing up as reproducibility: pooled AUC
ranks residues scored by five separately-calibrated models, so it inherits
their calibration variance, while mean-of-folds never compares across folds.
"""

from __future__ import annotations

import json

import pytest

from rockfish.rerun_stability import collect_runs, summarize

# The four measured runs.
MEASURED = [
    {"run": "j29683091", "gpu_pooled": 0.7203, "mean_folds": 0.7491,
     "v6_pooled": 0.7804, "stacked": 0.7876},
    {"run": "j29683038", "gpu_pooled": 0.7137, "mean_folds": 0.7514,
     "v6_pooled": 0.7804, "stacked": 0.7909},
    {"run": "j29682871", "gpu_pooled": 0.7454, "mean_folds": 0.7551,
     "v6_pooled": 0.7804, "stacked": 0.7999},
    {"run": "j29683037", "gpu_pooled": 0.7234, "mean_folds": 0.7405,
     "v6_pooled": 0.7804, "stacked": 0.7920},
]


class TestOnTheMeasuredRuns:
    def test_pooled_auc_is_less_reproducible_than_mean_of_folds(self):
        s = summarize(MEASURED)["statistics"]
        assert s["gpu_pooled"]["sd"] > s["mean_folds"]["sd"]

    def test_mean_of_folds_is_recommended_as_the_headline(self):
        assert summarize(MEASURED)["recommended_headline"]["statistic"] == "mean_folds"

    def test_the_gbdt_beats_the_neural_model_in_every_run(self):
        """The observation the lite architecture exists to explain."""
        adv = summarize(MEASURED)["gbdt_advantage"]
        assert adv["gbdt_wins_every_run"]
        assert adv["mean"] > 0.03

    def test_both_previously_quoted_gpu_figures_are_real_runs(self):
        """0.7454 and 0.7203 are the same configuration measured twice, not an
        error in either — they differ by about the rerun noise floor."""
        vals = [r["gpu_pooled"] for r in MEASURED]
        assert 0.7454 in vals and 0.7203 in vals
        assert abs(0.7454 - 0.7203) < 0.03

    def test_reported_spread_matches_the_docstring(self):
        s = summarize(MEASURED)["statistics"]
        assert s["gpu_pooled"]["range"] == pytest.approx(0.0317, abs=1e-4)
        assert s["mean_folds"]["range"] == pytest.approx(0.0147, abs=1e-4)


class TestCollection:
    def _run_dir(self, root, name, fold_aucs, gpu, v6, stacked, split="homology"):
        d = root / name
        d.mkdir()
        (d / "cv_summary.json").write_text(json.dumps({
            "fold_aucs": fold_aucs, "stacked_pooled_auc": stacked,
            "config": {"split_method": split},
        }))
        (d / "gpu_v6_ensemble_report.json").write_text(json.dumps({
            "weight_search": {"curve": [{"weight": 0.0, "auc": gpu},
                                        {"weight": 1.0, "auc": v6}]},
        }))
        return d

    def test_reads_components_from_the_ensemble_curve_endpoints(self, tmp_path):
        """The only place the components appear unmixed."""
        self._run_dir(tmp_path, "runA", [0.75, 0.76], 0.72, 0.78, 0.79)
        rows = collect_runs([str(tmp_path)])
        assert rows[0]["gpu_pooled"] == 0.72
        assert rows[0]["v6_pooled"] == 0.78

    def test_ignores_protein_split_runs(self, tmp_path):
        """Legacy protein-split numbers are not comparable and must not be
        averaged in with homology-split ones."""
        self._run_dir(tmp_path, "legacy", [0.81], 0.80, 0.83, 0.84, split="protein")
        assert collect_runs([str(tmp_path)]) == []

    def test_ignores_incomplete_runs(self, tmp_path):
        (tmp_path / "partial").mkdir()
        assert collect_runs([str(tmp_path)]) == []

    def test_prefers_stacked_key_over_the_legacy_pooled_field(self, tmp_path):
        """Older summaries stored the stacked score under 'pooled_auc'."""
        d = tmp_path / "legacy_field"
        d.mkdir()
        (d / "cv_summary.json").write_text(json.dumps({
            "fold_aucs": [0.75], "pooled_auc": 0.79,
            "config": {"split_method": "homology"},
        }))
        (d / "gpu_v6_ensemble_report.json").write_text(json.dumps({
            "weight_search": {"curve": [{"weight": 0.0, "auc": 0.72}]},
        }))
        assert collect_runs([str(tmp_path)])[0]["stacked"] == 0.79
