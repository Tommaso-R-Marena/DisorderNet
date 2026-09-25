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
        # Anchored on the real CAID3 leader (PUNCH2 0.955), not on the
        # ESMDisPred abstract figure this repo used to treat as SOTA.
        assert isinstance(rep["ci_reaches_sota"], bool)
        assert isinstance(rep["ci_exceeds_sota"], bool)
        assert rep["sota_reference_auc"] == 0.955


class TestRescueNeedsControls:
    """A bare rescue rate is not a result.

    Rescue is recall restricted to hallucinated residues, so a method that
    simply predicts disorder more liberally rescues more of them. The reported
    0.465 says nothing on its own; it needs baselines at a matched prediction
    budget, and a chance floor.
    """

    @staticmethod
    def _data(n=40000, seed=0):
        import numpy as np

        rng = np.random.default_rng(seed)
        y = (rng.random(n) < 0.25).astype(np.int8)
        pld = np.where(y == 1, rng.normal(45, 15, n), rng.normal(85, 10, n)).astype(np.float32)
        hi = rng.choice(np.flatnonzero(y == 1), size=int(0.3 * (y == 1).sum()), replace=False)
        pld[hi] = rng.normal(85, 5, len(hi))
        pld = np.clip(pld, 0, 100).astype(np.float32)
        return y, pld, rng

    def test_constant_scorer_cannot_win_by_flagging_everything(self):
        """The failure mode the matched budget exists to prevent."""
        import numpy as np

        from colab.hallucination_benchmark import compare_rescue_baselines

        y, pld, rng = self._data()
        n = len(y)
        dn = np.clip(0.5 + 0.30 * (y - 0.5) + rng.normal(0, 0.18, n), 0, 1).astype(np.float32)
        r = compare_rescue_baselines(
            y, pld,
            {"disordernet": dn, "always_disorder": np.ones(n, dtype=np.float32)},
        )
        always = r["methods"]["always_disorder"]
        assert always["n_flagged"] == r["matched_budget_residues"], (
            "a constant scorer must be held to the same budget, not allowed to "
            "flag every residue"
        )
        assert always["rescue_rate"] < r["methods"]["disordernet"]["rescue_rate"]

    def test_inverse_plddt_is_structurally_poor_at_rescue(self):
        """The load-bearing control: hallucinations are high-pLDDT by definition,
        so a structure-confidence score cannot rank them highly."""
        import numpy as np

        from colab.af_plddt import plddt_to_disorder_score
        from colab.hallucination_benchmark import compare_rescue_baselines

        y, pld, rng = self._data()
        n = len(y)
        dn = np.clip(0.5 + 0.30 * (y - 0.5) + rng.normal(0, 0.18, n), 0, 1).astype(np.float32)
        r = compare_rescue_baselines(
            y, pld,
            {"disordernet": dn, "inverse_plddt": plddt_to_disorder_score(pld)},
        )
        assert r["methods"]["inverse_plddt"]["rescue_rate"] < 0.4
        assert r["delta_vs_best_competing"] > 0

    def test_reports_precision_alongside_rescue(self):
        """A method could match the budget but spend it on ordered residues."""
        import numpy as np

        from colab.hallucination_benchmark import compare_rescue_baselines

        y, pld, rng = self._data()
        n = len(y)
        dn = np.clip(0.5 + 0.30 * (y - 0.5) + rng.normal(0, 0.18, n), 0, 1).astype(np.float32)
        r = compare_rescue_baselines(y, pld, {"disordernet": dn})
        m = r["methods"]["disordernet"]
        assert 0.0 <= m["precision_on_flagged"] <= 1.0
        assert m["n_rescued"] <= r["n_hallucinated"]

    def test_equal_methods_yield_no_advantage(self):
        """If two methods are the same, the delta must be ~0 — otherwise the
        comparison manufactures a difference."""
        import numpy as np

        from colab.hallucination_benchmark import compare_rescue_baselines

        y, pld, rng = self._data()
        n = len(y)
        dn = np.clip(0.5 + 0.30 * (y - 0.5) + rng.normal(0, 0.18, n), 0, 1).astype(np.float32)
        r = compare_rescue_baselines(y, pld, {"disordernet": dn, "twin": dn.copy()})
        assert abs(r["delta_vs_best_competing"]) < 0.01


class TestRescueVerdictIsExplicit:
    """Measured on the real 650M run: DisorderNet rescues 8.7% of hallucinations
    where random selection at the same budget rescues 25.1%. The report must
    state that plainly rather than leave a reader to compare two JSON fields."""

    def test_below_random_floor_is_flagged_and_explained(self):
        import numpy as np

        from colab.hallucination_benchmark import compare_rescue_baselines

        rng = np.random.default_rng(5)
        n = 20000
        y = (rng.random(n) < 0.25).astype(np.int8)
        pld = np.where(y == 1, rng.normal(45, 15, n), rng.normal(85, 10, n)).astype(np.float32)
        hi = rng.choice(np.flatnonzero(y == 1), size=int(0.3 * (y == 1).sum()), replace=False)
        pld[hi] = rng.normal(85, 5, len(hi))
        pld = np.clip(pld, 0, 100).astype(np.float32)

        # A scorer that mirrors pLDDT agreement — confident exactly where the
        # structure is confident, so it misses hallucinations by construction.
        agrees_with_structure = (1 - pld / 100).astype(np.float32)
        r = compare_rescue_baselines(
            y, pld,
            {
                "disordernet": agrees_with_structure,
                "random_floor": rng.random(n).astype(np.float32),
            },
        )
        assert r["below_random_floor"] is True
        assert "REFUTED" in r["verdict"]

    def test_genuine_detector_is_not_flagged(self):
        import numpy as np

        from colab.hallucination_benchmark import compare_rescue_baselines

        rng = np.random.default_rng(6)
        n = 20000
        y = (rng.random(n) < 0.25).astype(np.int8)
        pld = np.where(y == 1, rng.normal(45, 15, n), rng.normal(85, 10, n)).astype(np.float32)
        hi = rng.choice(np.flatnonzero(y == 1), size=int(0.3 * (y == 1).sum()), replace=False)
        pld[hi] = rng.normal(85, 5, len(hi))
        pld = np.clip(pld, 0, 100).astype(np.float32)

        # Scores the label directly: a real detector.
        oracle = (y + rng.normal(0, 0.1, n)).astype(np.float32)
        r = compare_rescue_baselines(
            y, pld,
            {"disordernet": oracle, "random_floor": rng.random(n).astype(np.float32)},
        )
        assert r["below_random_floor"] is False
        assert "supported" in r["verdict"]
