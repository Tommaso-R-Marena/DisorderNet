"""The within-protein leaderboard makes a claim about the whole field.

If it is wrong it is wrong about 115 published methods at once, so the pieces
that decide who moves — which targets count, which methods are eligible, and
what a protein-level score is worth — are checked directly.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load():
    """Import the analysis script by path; it lives under results/, not a
    package, because it is a record of one analysis rather than a library."""
    path = os.path.join(REPO, "results", "caid3",
                        "within_protein_leaderboard.py")
    spec = importlib.util.spec_from_file_location("wpl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wpl = _load()


def ref_entry(seq, lab):
    return (seq, lab)


class TestEligibleTargets:
    def test_single_class_targets_are_excluded(self):
        """A target with no positive contributes no within-protein pair, so
        including it would only make the two rankings look more alike."""
        ref = {
            "both": ref_entry("AAAA", "0011"),
            "all_neg": ref_entry("AAAA", "0000"),
            "all_pos": ref_entry("AAAA", "1111"),
        }
        assert wpl.two_class_targets(ref) == ["both"]

    def test_unevaluated_residues_do_not_create_a_second_class(self):
        """'-' marks a residue CAID does not score. A target that is all
        negatives plus masked residues is still single-class."""
        ref = {"masked": ref_entry("AAAA", "00--")}
        assert wpl.two_class_targets(ref) == []

    def test_a_target_is_kept_on_evaluated_residues_only(self):
        ref = {"t": ref_entry("AAAAAA", "0-1-0-")}
        assert wpl.two_class_targets(ref) == ["t"]


class TestEligibleMethods:
    @staticmethod
    def _ref():
        return {"a": ref_entry("AAAA", "0011"),
                "b": ref_entry("AAAAA", "00111")}

    @staticmethod
    def _write(tmp_path, name, preds):
        lines = []
        for tid, (seq, scores) in preds.items():
            lines.append(f">{tid}")
            for i, (aa, s) in enumerate(zip(seq, scores), 1):
                lines.append(f"{i}\t{aa}\t{s}")
        (tmp_path / f"{name}.caid").write_text("\n".join(lines) + "\n")

    def test_a_method_missing_a_target_is_not_eligible(self, tmp_path):
        self._write(tmp_path, "partial", {"a": ("AAAA", [0.1, 0.2, 0.8, 0.9])})
        assert "partial" not in wpl.full_coverage_methods(self._ref(),
                                                          str(tmp_path))

    def test_a_length_mismatch_is_not_eligible(self, tmp_path):
        self._write(tmp_path, "short", {
            "a": ("AAAA", [0.1, 0.2, 0.8, 0.9]),
            "b": ("AAAA", [0.1, 0.2, 0.8, 0.9]),      # reference has 5
        })
        assert "short" not in wpl.full_coverage_methods(self._ref(),
                                                        str(tmp_path))

    def test_a_complete_method_is_eligible(self, tmp_path):
        self._write(tmp_path, "full", {
            "a": ("AAAA", [0.1, 0.2, 0.8, 0.9]),
            "b": ("AAAAA", [0.1, 0.2, 0.8, 0.9, 0.95]),
        })
        got = wpl.full_coverage_methods(self._ref(), str(tmp_path))
        assert "full" in got
        assert set(got["full"]) == {"a", "b"}


class TestAProteinLevelScoreHasNoWithinProteinResolution:
    def test_constant_per_protein_scores_give_within_auc_exactly_half(self):
        """The hydropathy baseline assigns one number per chain. Its
        within-protein AUC must be 0.5 by construction — if it were not, the
        'purely between-protein' reading of its pooled score would be false."""
        rng = np.random.default_rng(0)
        ys, ss = [], []
        for k in range(12):
            n = int(rng.integers(20, 60))
            y = (rng.random(n) < 0.3).astype(np.int8)
            y[0], y[1] = 1, 0                      # guarantee both classes
            ys.append(y)
            ss.append(np.full(n, float(k), dtype=np.float64))
        d = wpl.decompose_auc(ys, ss)
        assert d["auc_within"] == pytest.approx(0.5, abs=1e-12)

    def test_the_hydropathy_baseline_is_between_protein_only(self):
        ref = {
            "polar": ref_entry("DDEEKKRRSS" * 4, ("1" * 20) + ("0" * 20)),
            "greasy": ref_entry("IIVVLLFFMM" * 4, ("0" * 30) + ("1" * 10)),
            "mixed": ref_entry("ADEIKLVSRT" * 4, ("0" * 25) + ("1" * 15)),
        }
        d = wpl.hydropathy_between_auc(ref, list(ref))
        assert d is not None
        assert d["auc_within"] == pytest.approx(0.5, abs=1e-12)
        # The identity must still close on real inputs.
        recomposed = (d["w_within"] * d["auc_within"]
                      + d["w_between"] * d["auc_between"])
        assert recomposed == pytest.approx(d["pooled"], abs=1e-12)


class TestRankAgreement:
    def test_spearman_matches_scipy(self):
        from scipy.stats import spearmanr

        rng = np.random.default_rng(7)
        a = rng.normal(size=40)
        b = 0.6 * a + 0.8 * rng.normal(size=40)
        assert wpl.spearman(a, b) == pytest.approx(
            float(spearmanr(a, b).statistic), abs=1e-12)

    def test_spearman_handles_ties_the_same_way_scipy_does(self):
        from scipy.stats import spearmanr

        a = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 4.0])
        b = np.array([2.0, 1.0, 2.0, 5.0, 4.0, 4.0])
        assert wpl.spearman(a, b) == pytest.approx(
            float(spearmanr(a, b).statistic), abs=1e-12)

    def test_a_perfectly_reversed_ranking_is_minus_one(self):
        a = np.arange(10, dtype=float)
        assert wpl.spearman(a, -a) == pytest.approx(-1.0)


class TestTheBootstrapClustersOnProteins:
    def test_it_resamples_proteins_and_returns_an_interval(self):
        rng = np.random.default_rng(3)
        rows = []
        for skill in (0.0, 0.5, 1.5):
            ys, ss = [], []
            for _ in range(15):
                n = 40
                y = (rng.random(n) < 0.35).astype(np.int8)
                y[0], y[1] = 1, 0
                s = rng.normal(size=n) + skill * y
                ys.append(y)
                ss.append(s)
            d = wpl.decompose_auc(ys, ss)
            rows.append({"method": f"m{skill}", "ys": ys, "ss": ss, **d})

        lo, hi = wpl.clustered_spearman_ci(rows, 15, np.random.default_rng(1),
                                           n_boot=40)
        assert lo is not None and hi is not None
        assert -1.0 <= lo <= hi <= 1.0

    def test_the_resample_indexes_every_method_identically(self):
        """Each method must be re-scored on the *same* redrawn proteins, or the
        correlation is between two different populations."""
        src = open(os.path.join(REPO, "results", "caid3",
                                "within_protein_leaderboard.py")).read()
        body = src[src.index("def clustered_spearman_ci"):
                   src.index("def hydropathy_between_auc")]
        assert "take = rng.integers" in body
        assert body.index("take = rng.integers") < body.index("for r in rows")


class TestOurSubmissionsFaceTheSameRule:
    @staticmethod
    def _ref():
        return {"a": ("AAAA", "0011"), "b": ("AAAAA", "00111")}

    @staticmethod
    def _write(path, preds):
        lines = []
        for tid, (seq, scores) in preds.items():
            lines.append(f">{tid}")
            for i, (aa, s) in enumerate(zip(seq, scores), 1):
                lines.append(f"{i}\t{aa}\t{s}")
        open(path, "w").write("\n".join(lines) + "\n")

    def test_a_complete_submission_is_included(self, tmp_path):
        self._write(str(tmp_path / "DisorderNet-linker.caid"), {
            "a": ("AAAA", [0.1, 0.2, 0.8, 0.9]),
            "b": ("AAAAA", [0.1, 0.2, 0.8, 0.9, 0.95]),
        })
        got = wpl.extra_methods(self._ref(), "linker",
                                f"{tmp_path}:DisorderNet")
        assert set(got) == {"DisorderNet"}

    def test_our_partial_submission_is_excluded_like_anyone_elses(self, tmp_path):
        """Comparing our within-protein AUC on a subset against methods scored
        on everything is the exact cherry-pick this analysis exists to expose."""
        self._write(str(tmp_path / "DisorderNet-linker.caid"), {
            "a": ("AAAA", [0.1, 0.2, 0.8, 0.9]),
        })
        assert wpl.extra_methods(self._ref(), "linker",
                                 f"{tmp_path}:DisorderNet") == {}

    def test_a_missing_file_is_reported_not_fatal(self, tmp_path):
        assert wpl.extra_methods(self._ref(), "binding",
                                 f"{tmp_path}:DisorderNet") == {}

    def test_several_submission_directories_may_be_named(self, tmp_path):
        for name in ("A", "B"):
            d = tmp_path / name
            d.mkdir()
            self._write(str(d / f"{name}-linker.caid"), {
                "a": ("AAAA", [0.1, 0.2, 0.8, 0.9]),
                "b": ("AAAAA", [0.1, 0.2, 0.8, 0.9, 0.95]),
            })
        got = wpl.extra_methods(
            self._ref(), "linker",
            f"{tmp_path / 'A'}:A,{tmp_path / 'B'}:B")
        assert set(got) == {"A", "B"}

    def test_the_output_says_the_ranks_are_not_caids(self):
        """These ranks are recomputed on full-coverage methods and two-class
        targets. Printing them beside CAID's own without saying so would be
        the most quotable mistake in the file."""
        src = open(os.path.join(REPO, "results", "caid3",
                                "within_protein_leaderboard.py")).read()
        assert "NOT CAID" in src and "published ranks" in src
        assert "recomputed on this subset" in src

    def test_two_checkpoints_sharing_a_filename_get_distinct_labels(self, tmp_path):
        """Both checkpoints archive as 'DisorderNet-<task>.caid'. Without a
        separate label the second would overwrite the first and the table would
        contain one model under two apparent identities."""
        for run in ("windowed", "pbias"):
            d = tmp_path / run
            d.mkdir()
            self._write(str(d / "DisorderNet-linker.caid"), {
                "a": ("AAAA", [0.1, 0.2, 0.8, 0.9]),
                "b": ("AAAAA", [0.1, 0.2, 0.8, 0.9, 0.95]),
            })
        got = wpl.extra_methods(
            self._ref(), "linker",
            f"{tmp_path / 'windowed'}:DisorderNet:DN-windowed,"
            f"{tmp_path / 'pbias'}:DisorderNet:DN-pbias")
        assert set(got) == {"DN-windowed", "DN-pbias"}

    def test_a_repeated_label_is_refused(self, tmp_path):
        for run in ("x", "y"):
            d = tmp_path / run
            d.mkdir()
            self._write(str(d / "DisorderNet-linker.caid"), {
                "a": ("AAAA", [0.1, 0.2, 0.8, 0.9]),
                "b": ("AAAAA", [0.1, 0.2, 0.8, 0.9, 0.95]),
            })
        with pytest.raises(SystemExit):
            wpl.extra_methods(
                self._ref(), "linker",
                f"{tmp_path / 'x'}:DisorderNet:DN,"
                f"{tmp_path / 'y'}:DisorderNet:DN")
