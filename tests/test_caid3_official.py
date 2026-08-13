"""The official CAID3 references, and what may be claimed from them.

Four of the five references used to be reconstructed from DisProt. Only Linker
reconstructed exactly; Disorder-NOX came out as 319 targets against a true 204.
CAID serves all five, plus every entrant's per-residue predictions, so nothing
needs reconstructing and every comparison can be paired.

These tests guard the two things that can silently go wrong with that: scoring
against a reference that is not the benchmark, and reading more into a
difference than the target counts support.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from colab.caid3_official import (
    EXPECTED,
    LEADERS,
    TASKS,
    ReferenceCompositionError,
    composition,
    evaluated_mask,
    paired_bootstrap,
    per_target_arrays,
    read_caid_predictions,
    read_reference,
    score_method,
    verify_composition,
)

CACHE = os.environ.get("CAID3_OFFICIAL_DIR", "")
PREDS = os.environ.get("CAID3_PREDICTIONS_DIR", "")
has_refs = bool(CACHE) and os.path.isdir(CACHE)
has_preds = bool(PREDS) and os.path.isdir(PREDS)


def write_ref(tmp_path, records):
    p = tmp_path / "ref.fasta"
    p.write_text("".join(f">{i}\n{s}\n{l}\n" for i, s, l in records))
    return str(p)


def write_caid(tmp_path, name, records):
    p = tmp_path / f"{name}.caid"
    out = []
    for tid, scores in records:
        out.append(f">{tid}\n")
        for i, v in enumerate(scores, 1):
            txt = "" if v is None else f"{v:.3f}"
            out.append(f"{i}\tA\t{txt}\t0\n")
        p.write_text("".join(out))
    return str(p)


class TestReferenceParsing:
    def test_round_trips_a_three_line_record(self, tmp_path):
        path = write_ref(tmp_path, [("T1", "ACDE", "01-1")])
        ref = read_reference(path)
        assert ref["T1"] == ("ACDE", "01-1")

    def test_sequence_and_label_length_must_agree(self, tmp_path):
        path = write_ref(tmp_path, [("T1", "ACDE", "011")])
        with pytest.raises(ValueError, match="4 residues but 3 labels"):
            read_reference(path)

    def test_truncated_record_is_an_error(self, tmp_path):
        p = tmp_path / "r.fasta"
        p.write_text(">T1\nACDE\n")
        with pytest.raises(ValueError, match="truncated"):
            read_reference(str(p))

    def test_dash_marks_the_unevaluated_residues(self):
        m = evaluated_mask("01-1-")
        assert m.tolist() == [True, True, False, True, False]

    def test_composition_counts_the_three_classes(self, tmp_path):
        path = write_ref(tmp_path, [("A", "AAAA", "01-1"), ("B", "AA", "00")])
        assert composition(read_reference(path)) == (2, 2, 3, 1)


class TestCompositionIsEnforced:
    """A reference that is not the benchmark must never be scored against."""

    def test_wrong_composition_raises_with_both_numbers(self, tmp_path):
        path = write_ref(tmp_path, [("T1", "ACDE", "0101")])
        with pytest.raises(ReferenceCompositionError) as e:
            verify_composition("linker", path)
        msg = str(e.value)
        assert "(1, 2, 2, 0)" in msg
        assert str(EXPECTED["linker"]) in msg

    def test_every_task_has_an_expected_composition(self):
        assert set(EXPECTED) == set(TASKS)
        for task, want in EXPECTED.items():
            assert len(want) == 4, task
            assert want[0] > 0 and want[1] > 0, task

    def test_expected_prevalences_match_the_published_challenge(self):
        """Published CAID3 prevalences, to the digit the site reports."""
        want = {"disorder_pdb": 0.316, "disorder_nox": 0.264,
                "binding": 0.106, "binding_idr": 0.386, "linker": 0.067}
        for task, (_n, pos, neg, _u) in EXPECTED.items():
            assert round(pos / (pos + neg), 3) == want[task], task


class TestScoringAndCoverage:
    def test_unevaluated_residues_are_excluded(self, tmp_path):
        ref = read_reference(write_ref(tmp_path, [("T1", "AAAA", "0-11")]))
        pred = read_caid_predictions(
            write_caid(tmp_path, "m", [("T1", [0.1, 0.9, 0.8, 0.7])]))
        ys, _ss, _k, _c = per_target_arrays(ref, pred)
        assert ys[0].tolist() == [0, 1, 1]

    def test_missing_targets_are_counted_not_hidden(self, tmp_path):
        ref = read_reference(write_ref(
            tmp_path, [("T1", "AA", "01"), ("T2", "AA", "01")]))
        pred = read_caid_predictions(
            write_caid(tmp_path, "m", [("T1", [0.1, 0.9])]))
        r = score_method(ref, pred)
        assert r["n_scored_targets"] == 1
        assert r["missing_targets"] == ["T2"]
        assert r["coverage"] == pytest.approx(0.5)

    def test_length_mismatch_is_reported_separately(self, tmp_path):
        ref = read_reference(write_ref(
            tmp_path, [("T1", "AAAA", "0101"), ("T2", "AA", "01")]))
        pred = read_caid_predictions(write_caid(
            tmp_path, "m", [("T1", [0.1, 0.9]), ("T2", [0.2, 0.8])]))
        r = score_method(ref, pred)
        assert r["length_mismatch_targets"] == ["T1"]
        assert r["n_scored_targets"] == 1

    def test_declined_residues_become_nan_and_are_dropped(self, tmp_path):
        """FoldUnfold and NeProc leave the score field blank."""
        ref = read_reference(write_ref(tmp_path, [("T1", "AAAA", "0011")]))
        pred = read_caid_predictions(
            write_caid(tmp_path, "m", [("T1", [0.1, None, 0.8, 0.9])]))
        assert np.isnan(pred["T1"][1])
        ys, ss, _k, _c = per_target_arrays(ref, pred)
        assert len(ys[0]) == 3
        assert np.isfinite(ss[0]).all()

    def test_a_perfect_ranker_scores_one(self, tmp_path):
        ref = read_reference(write_ref(tmp_path, [("T1", "AAAA", "0011")]))
        pred = read_caid_predictions(
            write_caid(tmp_path, "m", [("T1", [0.1, 0.2, 0.8, 0.9])]))
        assert score_method(ref, pred)["auc"] == pytest.approx(1.0)


class TestPairedComparison:
    """A difference is only a difference if the interval says so."""

    def _two_methods(self, tmp_path, n=40, sep=0.0, seed=0):
        rng = np.random.default_rng(seed)
        recs, a_rows, b_rows = [], [], []
        for i in range(n):
            lab = rng.integers(0, 2, 12)
            recs.append((f"T{i}", "A" * 12, "".join(map(str, lab))))
            a_rows.append((f"T{i}", (lab * (0.5 + sep) + rng.normal(0, 0.3, 12)).tolist()))
            b_rows.append((f"T{i}", (lab * 0.5 + rng.normal(0, 0.3, 12)).tolist()))
        refs = tmp_path / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "linker.fasta").write_text(
            "".join(f">{i}\n{s}\n{l}\n" for i, s, l in recs))
        preds = tmp_path / "preds"
        preds.mkdir(exist_ok=True)
        for name, rows in (("A", a_rows), ("B", b_rows)):
            (preds / f"{name}.caid").write_text("".join(
                f">{t}\n" + "".join(f"{j}\tA\t{v:.4f}\t0\n"
                                    for j, v in enumerate(s, 1))
                for t, s in rows))
        return str(refs), str(preds)

    def test_identical_methods_give_zero_delta_and_a_large_p(self, tmp_path):
        refs, preds = self._two_methods(tmp_path, sep=0.0, seed=1)
        import shutil
        shutil.copy(os.path.join(preds, "A.caid"), os.path.join(preds, "A2.caid"))
        r = paired_bootstrap("linker", refs, preds, "A", "A2", n_boot=200)
        assert r["delta_auc"] == pytest.approx(0.0, abs=1e-12)
        # Every resampled delta is exactly zero. Both tails must contain the
        # ties, or a perfect null comes back as p=0.
        assert r["p_two_sided"] == pytest.approx(1.0)

    def test_p_value_never_reaches_zero(self, tmp_path):
        """B resamples cannot resolve below 1/(B+1); reporting 0 overstates it."""
        refs, preds = self._two_methods(tmp_path, n=60, sep=3.0, seed=9)
        r = paired_bootstrap("linker", refs, preds, "A", "B", n_boot=200)
        assert r["p_two_sided"] > 0.0
        assert r["p_two_sided"] <= 2.0 / 201.0 + 1e-12

    def test_a_clearly_better_method_is_detected(self, tmp_path):
        refs, preds = self._two_methods(tmp_path, n=60, sep=1.2, seed=2)
        r = paired_bootstrap("linker", refs, preds, "A", "B", n_boot=500)
        assert r["delta_auc"] > 0
        assert r["delta_ci"][0] > 0
        assert r["p_two_sided"] < 0.05

    def test_comparison_uses_only_targets_both_predicted(self, tmp_path):
        refs, preds = self._two_methods(tmp_path, n=20, seed=3)
        text = open(os.path.join(preds, "B.caid")).read()
        keep = text.split(">T15")[0]
        open(os.path.join(preds, "B.caid"), "w").write(keep)
        r = paired_bootstrap("linker", refs, preds, "A", "B", n_boot=100)
        assert r["n_common_targets"] == 15

    def test_bootstrap_resamples_proteins_not_residues(self, tmp_path):
        """Residues within a protein are correlated; resampling them gives an
        interval far too narrow, which is how a null result looks significant."""
        refs, preds = self._two_methods(tmp_path, n=25, sep=0.0, seed=4)
        r = paired_bootstrap("linker", refs, preds, "A", "B", n_boot=400)
        width = r["delta_ci"][1] - r["delta_ci"][0]
        # 25 proteins of 12 residues: a protein-clustered interval is wide.
        assert width > 0.02, width
        assert r["n_common_targets"] == 25


@pytest.mark.skipif(not (has_refs and has_preds),
                    reason="set CAID3_OFFICIAL_DIR and CAID3_PREDICTIONS_DIR")
class TestAgainstTheRealChallenge:
    """Run only where the downloaded challenge data is present."""

    @pytest.mark.parametrize("task", TASKS)
    def test_downloaded_reference_matches_caid_composition(self, task):
        verify_composition(task, os.path.join(CACHE, f"{task}.fasta"))

    @pytest.mark.parametrize("task", TASKS)
    def test_published_leader_reproduces(self, task):
        leader, pub_auc, _pub_aps = LEADERS[task]
        ref = read_reference(os.path.join(CACHE, f"{task}.fasta"))
        pred = read_caid_predictions(os.path.join(PREDS, f"{leader}.caid"))
        got = score_method(ref, pred)
        assert got is not None, leader
        assert abs(got["auc"] - pub_auc) <= 0.0015, (task, got["auc"], pub_auc)

    def test_the_caid3_winner_is_not_separable_from_a_structural_baseline(self):
        """PUNCH2 ranks 1st on Disorder-PDB; AlphaFold-rsa, which is training
        free, ranks 3rd. On all 319 targets with protein-clustered resampling
        the difference does not clear zero. Recorded because a project aiming to
        beat the leaderboard needs to know the leaderboard's own top margin is
        not resolvable."""
        r = paired_bootstrap("disorder_pdb", CACHE, PREDS,
                             "PUNCH2", "AlphaFold-rsa", n_boot=1000)
        assert r["n_common_targets"] == 319
        assert r["delta_auc"] > 0
        assert r["delta_ci"][0] < 0 < r["delta_ci"][1]

    def test_skipped_targets_are_harder_than_average(self):
        """SPOT-Disorder2 declines 21 of 319. PUNCH2, which declines none,
        scores higher on the 298 that remain than on the full set — so the
        declined targets are the hard ones, and a method is rewarded for
        skipping them."""
        ref = read_reference(os.path.join(CACHE, "disorder_pdb.fasta"))
        punch = read_caid_predictions(os.path.join(PREDS, "PUNCH2.caid"))
        full = score_method(ref, punch)
        r = paired_bootstrap("disorder_pdb", CACHE, PREDS,
                             "PUNCH2", "SPOT-Disorder2", n_boot=200)
        assert r["n_common_targets"] < full["n_scored_targets"]
        assert r["auc_a"] > full["auc"]
