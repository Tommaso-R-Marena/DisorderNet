"""Protein-clustered bootstrap CIs.

Residues within a protein are strongly correlated, so a residue-level bootstrap
treats ~10^6 dependent observations as independent and returns intervals far too
narrow — narrow enough to make a 0.009 AUC difference look decisive. The
resampling unit must be the protein.
"""

from __future__ import annotations

import numpy as np
import pytest

from colab.bootstrap_ci import (
    paired_protein_bootstrap_delta,
    protein_bootstrap_metric,
    summarize_ci,
)


def _segmented_proteins(n_proteins=120, seed=0, dn_edge=0.30, pl_edge=0.29,
                        heterogeneity=0.18):
    """Proteins with contiguous disordered segments and per-protein difficulty.

    Two sources of within-protein correlation, both real: disorder comes in
    runs, and whole proteins differ in how well the model does on them. The
    second is what makes protein-level variance dominate residue-level variance
    — omit it and a residue bootstrap looks deceptively adequate.
    """
    rng = np.random.default_rng(seed)
    ys, dn, inv = [], [], []
    for _ in range(n_proteins):
        n = int(rng.integers(120, 400))
        y = np.zeros(n, dtype=np.int8)
        for _ in range(int(rng.integers(1, 4))):
            s = int(rng.integers(0, max(1, n - 40)))
            L = int(rng.integers(20, 80))
            y[s:s + L] = 1
        # Per-protein random effect on discriminability, shared by both methods.
        skill = float(rng.normal(0.0, heterogeneity))
        ys.append(y)
        dn.append(np.clip(
            0.5 + (dn_edge + skill) * (y - 0.5) + rng.normal(0, 0.2, n), 0, 1
        ).astype(np.float32))
        inv.append(np.clip(
            0.5 + (pl_edge + skill) * (y - 0.5) + rng.normal(0, 0.2, n), 0, 1
        ).astype(np.float32))
    return ys, dn, inv


class TestPointAndInterval:
    def test_point_matches_pooled_metric(self):
        from sklearn.metrics import roc_auc_score

        ys, dn, _ = _segmented_proteins(60)
        r = protein_bootstrap_metric(ys, dn, n_boot=100)
        direct = roc_auc_score(np.concatenate(ys), np.concatenate(dn))
        assert r["point"] == pytest.approx(direct, abs=1e-9)

    def test_interval_brackets_the_point(self):
        ys, dn, _ = _segmented_proteins(60)
        r = protein_bootstrap_metric(ys, dn, n_boot=200)
        assert r["ci_low"] <= r["point"] <= r["ci_high"]

    def test_resampling_unit_is_recorded(self):
        ys, dn, _ = _segmented_proteins(20)
        r = protein_bootstrap_metric(ys, dn, n_boot=50)
        assert r["resampling_unit"] == "protein"
        assert r["n_proteins"] == 20

    def test_is_deterministic_for_a_given_seed(self):
        ys, dn, _ = _segmented_proteins(30)
        a = protein_bootstrap_metric(ys, dn, n_boot=100, seed=7)
        b = protein_bootstrap_metric(ys, dn, n_boot=100, seed=7)
        assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]

    def test_handles_too_few_proteins(self):
        r = protein_bootstrap_metric([np.array([0, 1])], [np.array([0.1, 0.9])], n_boot=10)
        assert r["insufficient_data"] is True


class TestClusteringMatters:
    def test_protein_bootstrap_is_wider_than_residue_bootstrap(self):
        """The whole reason this module exists."""
        from sklearn.metrics import roc_auc_score

        ys, dn, _ = _segmented_proteins(80, seed=5)
        clustered = protein_bootstrap_metric(ys, dn, n_boot=300, seed=1)
        width_clustered = clustered["ci_high"] - clustered["ci_low"]

        all_y, all_s = np.concatenate(ys), np.concatenate(dn)
        rng = np.random.default_rng(1)
        naive = []
        for _ in range(300):
            i = rng.integers(0, len(all_y), len(all_y))
            if len(np.unique(all_y[i])) > 1:
                naive.append(roc_auc_score(all_y[i], all_s[i]))
        lo, hi = np.percentile(naive, [2.5, 97.5])
        assert width_clustered > (hi - lo), (
            "protein-clustered CI must be wider than the residue bootstrap; "
            f"got {width_clustered:.5f} vs {hi - lo:.5f}"
        )


class TestPairedDelta:
    def test_detects_a_real_difference(self):
        ys, dn, inv = _segmented_proteins(150, seed=2, dn_edge=0.40, pl_edge=0.20)
        d = paired_protein_bootstrap_delta(ys, dn, inv, n_boot=300)
        assert d["delta"] > 0
        assert d["crosses_zero"] is False
        assert d["p_value_bootstrap"] < 0.05

    def test_reports_no_difference_when_methods_are_equivalent(self):
        """A delta inside sampling noise must not be presented as an improvement."""
        ys, dn, inv = _segmented_proteins(80, seed=4, dn_edge=0.30, pl_edge=0.30)
        d = paired_protein_bootstrap_delta(ys, dn, inv, n_boot=300)
        assert d["crosses_zero"] is True
        assert d["p_value_bootstrap"] > 0.05

    def test_delta_equals_difference_of_points(self):
        ys, dn, inv = _segmented_proteins(40)
        d = paired_protein_bootstrap_delta(ys, dn, inv, n_boot=100)
        assert d["delta"] == pytest.approx(d["a"] - d["b"], abs=1e-9)

    def test_summary_flags_a_zero_crossing(self):
        ys, dn, inv = _segmented_proteins(60, dn_edge=0.30, pl_edge=0.30)
        d = paired_protein_bootstrap_delta(ys, dn, inv, n_boot=200)
        s = summarize_ci(d, "delta")
        if d["crosses_zero"]:
            assert "includes 0" in s


class TestWiredIntoDistrustBenchmark:
    def test_benchmark_helper_filters_and_returns_a_ci(self):
        from colab.hallucination_benchmark import _bootstrap_distrust_delta

        rng = np.random.default_rng(11)
        ys, dns, plds = [], [], []
        for _ in range(40):
            n = 200
            y = np.zeros(n, dtype=np.int8); y[50:120] = 1
            ys.append(y)
            dns.append(np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, n), 0, 1).astype(np.float32))
            pl = np.clip(100 * (1 - (0.5 + 0.25 * (y - 0.5))), 0, 100).astype(np.float32)
            pl[:10] = np.nan  # unmatched residues must be excluded
            plds.append(pl)

        r = _bootstrap_distrust_delta(ys, dns, plds, n_boot=150)
        assert r["n_proteins"] == 40
        assert r["n_residues"] == 40 * 190, "NaN pLDDT residues must be dropped"
        assert r["ci_low"] is not None and r["ci_high"] is not None


class TestCaid3ReportsAnInterval:
    def test_evaluation_attaches_a_protein_clustered_ci(self):
        """A benchmark AUC quoted against a literature figure needs an interval."""
        import numpy as np

        from colab.caid3_eval import evaluate_caid_predictions

        rng = np.random.default_rng(21)
        refs, preds = [], {}
        for i in range(40):
            n = 150
            lab = np.zeros(n, dtype=np.int8)
            lab[40:90] = 1
            refs.append({
                "id": f"C{i}", "sequence": "A" * n, "length": n,
                "labels": lab.tolist(), "eval_mask": [True] * n,
            })
            preds[f"C{i}"] = np.clip(
                0.5 + 0.3 * (lab - 0.5) + rng.normal(0, 0.2, n), 0, 1
            ).astype(np.float32)

        rep = evaluate_caid_predictions(refs, preds)
        ci = rep["auc_ci"]
        assert ci["resampling_unit"] == "protein"
        assert ci["n_proteins"] == 40
        assert ci["ci_low"] <= rep["pooled"]["auc"] <= ci["ci_high"]
        assert isinstance(rep["ci_reaches_esmdispred"], bool)
