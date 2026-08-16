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
        path = os.path.join(CACHE, f"{task}.fasta")
        verify_composition(task, path)          # raises on mismatch
        assert composition(read_reference(path)) == EXPECTED[task]

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


class TestCoverageBias:
    """Declining targets is worth AUC, and the report has to show it."""

    def _setup(self, tmp_path):
        """Two targets: one easy and short, one hard and long. The 'skipper'
        predicts only the easy one, which is the whole phenomenon in miniature."""
        rng = np.random.default_rng(0)
        easy_lab = np.array([0, 0, 1, 1] * 6)
        hard_lab = np.array([0, 1] * 40)
        recs = [("EASY", "A" * len(easy_lab), "".join(map(str, easy_lab))),
                ("HARD", "A" * len(hard_lab), "".join(map(str, hard_lab)))]
        refs = tmp_path / "refs"
        refs.mkdir()
        (refs / "linker.fasta").write_text(
            "".join(f">{i}\n{s}\n{l}\n" for i, s, l in recs))
        preds = tmp_path / "preds"
        preds.mkdir()

        def emit(name, rows):
            (preds / f"{name}.caid").write_text("".join(
                f">{t}\n" + "".join(f"{j}\tA\t{v:.4f}\t0\n"
                                    for j, v in enumerate(s, 1))
                for t, s in rows))

        # Yardstick answers both: sharp on EASY, near-random on HARD.
        emit("yardstick", [
            ("EASY", easy_lab * 2.0 + rng.normal(0, 0.1, len(easy_lab))),
            ("HARD", hard_lab * 0.05 + rng.normal(0, 1.0, len(hard_lab))),
        ])
        # Skipper answers only EASY.
        emit("skipper", [("EASY", easy_lab * 2.0 + rng.normal(0, 0.1, len(easy_lab)))])
        return str(refs), str(preds)

    def test_it_finds_the_skipper_and_prices_the_skip(self, tmp_path):
        from colab.caid3_official import coverage_bias_report

        refs, preds = self._setup(tmp_path)
        rep = coverage_bias_report("linker", refs, preds, "yardstick",
                                   min_skipped=1)
        rows = {m["method"]: m for m in rep["methods"]}
        assert "skipper" in rows
        r = rows["skipper"]
        assert r["n_skipped"] == 1
        assert r["yardstick_on_attempted"] > r["yardstick_on_skipped"]
        assert r["skipping_worth"] > 0

    def test_a_fully_covering_method_is_not_reported(self, tmp_path):
        from colab.caid3_official import coverage_bias_report

        refs, preds = self._setup(tmp_path)
        rep = coverage_bias_report("linker", refs, preds, "yardstick",
                                   min_skipped=1)
        assert "yardstick" not in {m["method"] for m in rep["methods"]}

    def test_it_records_the_length_of_what_was_skipped(self, tmp_path):
        """The mechanism is length: skipped targets are the long ones."""
        from colab.caid3_official import coverage_bias_report

        refs, preds = self._setup(tmp_path)
        rep = coverage_bias_report("linker", refs, preds, "yardstick",
                                   min_skipped=1)
        r = next(m for m in rep["methods"] if m["method"] == "skipper")
        assert r["median_length_skipped"] == 80
        assert r["median_length_all"] in (24, 52, 80)


@pytest.mark.skipif(not (has_refs and has_preds),
                    reason="set CAID3_OFFICIAL_DIR and CAID3_PREDICTIONS_DIR")
class TestRankFusionOnTheRealChallenge:
    def test_fusion_leaves_a_single_input_unchanged(self):
        """Ranking is monotone on the pooled vector, so one input in, same AUC
        out. If this drifts, the fusion is altering the metric rather than the
        prediction — which is what per-target normalisation did."""
        from colab.caid3_official import rank_fuse

        ref = read_reference(os.path.join(CACHE, "disorder_nox.fasta"))
        p = read_caid_predictions(os.path.join(PREDS, "AlphaFold-rsa.caid"))
        alone = score_method(ref, p)
        fused = rank_fuse([p], ref)
        assert score_method(ref, fused)["auc"] == pytest.approx(alone["auc"],
                                                               abs=1e-6)

    def test_fusion_is_order_independent(self):
        from colab.caid3_official import rank_fuse

        ref = read_reference(os.path.join(CACHE, "linker.fasta"))
        a = read_caid_predictions(os.path.join(PREDS, "AlphaFold-rsa.caid"))
        b = read_caid_predictions(os.path.join(PREDS, "AlphaFold3-rsa.caid"))
        ab = score_method(ref, rank_fuse([a, b], ref))["auc"]
        ba = score_method(ref, rank_fuse([b, a], ref))["auc"]
        assert ab == pytest.approx(ba, abs=1e-9)


class TestFamilyWiseCorrection:
    """Sixteen comparisons were run against these references, and the two that
    cleared 0.05 were reported. Holm is what makes that honest."""

    def test_the_smallest_p_gets_the_full_bonferroni_factor(self):
        from colab.caid3_official import holm_bonferroni

        r = holm_bonferroni({"a": 0.01, "b": 0.2, "c": 0.5, "d": 0.9})
        assert r["a"]["p_adjusted"] == pytest.approx(0.04)
        assert r["a"]["rank"] == 1

    def test_adjusted_values_are_monotone(self):
        """Without enforcing this, a later test can appear more significant
        than one ranked above it."""
        from colab.caid3_official import holm_bonferroni

        r = holm_bonferroni({"a": 0.02, "b": 0.021, "c": 0.5, "d": 0.9})
        adj = [v["p_adjusted"] for v in sorted(r.values(),
                                               key=lambda z: z["rank"])]
        assert adj == sorted(adj)

    def test_it_is_never_more_conservative_than_bonferroni(self):
        from colab.caid3_official import holm_bonferroni

        ps = {"a": 0.001, "b": 0.01, "c": 0.02, "d": 0.3}
        r = holm_bonferroni(ps)
        for k, p in ps.items():
            assert r[k]["p_adjusted"] <= min(1.0, len(ps) * p) + 1e-12

    def test_adjusted_p_never_exceeds_one(self):
        from colab.caid3_official import holm_bonferroni

        r = holm_bonferroni({f"t{i}": 0.9 for i in range(20)})
        assert all(v["p_adjusted"] <= 1.0 for v in r.values())

    def test_a_single_test_is_unadjusted(self):
        """A pre-registered primary endpoint of one needs no correction."""
        from colab.caid3_official import holm_bonferroni

        r = holm_bonferroni({"only": 0.03})
        assert r["only"]["p_adjusted"] == pytest.approx(0.03)
        assert r["only"]["significant"]

    def test_the_real_disorder_pdb_family_is_reproduced(self):
        """The published correction, so a change to holm_bonferroni that would
        revive those claims fails loudly."""
        from colab.caid3_official import holm_bonferroni

        r = holm_bonferroni({
            "fused-PUNCH2": 0.006, "ours-AFrsa": 0.013, "ours-PUNCH2": 0.2049,
            "b1": 0.001, "b2": 0.003, "b3": 0.008, "b4": 0.011, "b5": 0.022,
            "b6": 0.041, "b7": 0.3758, "b8": 0.4608, "b9": 0.6257,
            "b10": 0.6857, "b11": 0.7156, "b12": 0.7326, "b13": 0.8866,
        })
        assert not r["ours-AFrsa"]["significant"]
        assert not r["fused-PUNCH2"]["significant"]
        assert r["ours-AFrsa"]["p_adjusted"] == pytest.approx(0.143, abs=0.002)


class TestPreregisteredConstants:
    """The pre-registration is only binding if the code agrees with it."""

    def test_the_primary_family_is_exactly_two_tests(self):
        from rockfish.eval_caid3_official import (PRIMARY_OPPONENTS,
                                                  PRIMARY_TASK)

        assert PRIMARY_TASK == "disorder_pdb"
        assert len(PRIMARY_OPPONENTS) == 2
        assert set(PRIMARY_OPPONENTS) == {"AlphaFold-rsa", "PUNCH2"}

    @staticmethod
    def _doc(name):
        return os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "results", "caid3", name)

    def test_the_floors_match_the_active_registration(self):
        """There are two pre-registrations with different floors, because the
        second was set from corrected measurements. The code must follow the
        active one — this test caught it following neither cleanly."""
        from rockfish.eval_caid3_official import NON_INFERIORITY_FLOORS

        text = open(self._doc("PREREGISTRATION_2.md")).read()
        assert len(NON_INFERIORITY_FLOORS) == 4, NON_INFERIORITY_FLOORS
        for task, floor in NON_INFERIORITY_FLOORS.items():
            assert f"{floor:.4f}" in text, (
                f"{task} floor {floor} is not stated in PREREGISTRATION_2.md")

    def test_the_superseded_floor_is_still_the_first_registration(self):
        """The first registration is history, not a live constraint, and its
        floor must stay as it was written — rewriting it would erase what the
        first run was actually held to."""
        assert "0.9553" in open(self._doc("PREREGISTRATION.md")).read()

    def test_the_first_registration_is_marked_concluded(self):
        text = open(self._doc("PREREGISTRATION.md")).read()
        assert "CONCLUDED" in text, (
            "a completed registration must say so, or it reads as a live "
            "constraint the code is violating")

    def test_both_documents_exist_and_name_their_opponents(self):
        for name in ("PREREGISTRATION.md", "PREREGISTRATION_2.md"):
            text = open(self._doc(name)).read()
            assert "AlphaFold-rsa" in text or "bindEmbed21IDR" in text, name
            assert "Holm" in text, name


class TestSupersededPathsSaySo:
    """The reconstructions produced numbers that were reported as CAID3 results.

    Binding read 0.8389 against a published leader of 0.776 — apparently a
    decisive win — where the official 52-target reference gives 0.7649 and rank
    11. Anyone reaching for those modules has to meet that fact first.
    """

    def test_the_reconstruction_module_is_marked(self):
        import colab.caid3_references as m

        doc = m.__doc__ or ""
        assert doc.lstrip().startswith("SUPERSEDED"), doc[:80]
        assert "caid3_official" in doc

    def test_the_old_evaluator_is_marked(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rockfish", "eval_caid3_tasks.py")
        head = open(path).read()[:2000]
        assert "SUPERSEDED" in head
        assert "eval_caid3_official.py" in head

    def test_the_replacement_exists_and_is_not_marked(self):
        import colab.caid3_official as m

        assert "SUPERSEDED" not in (m.__doc__ or "")


class TestCrossRoundLeakGuard:
    """A CAID3-filtered checkpoint must never be scored on CAID2.

    The two rounds share exactly one protein, so a CAID3-only filter leaves 307
    of CAID2's 348 Disorder-PDB targets in the training union. That model would
    score *better* on CAID2 than an honest one, which is why this cannot be
    left to the operator remembering.
    """

    @staticmethod
    def _ckpt(tmp_path, refs):
        import json as _json

        d = tmp_path / "run"
        d.mkdir()
        payload = {"caid_leak_filter": {"reference": refs}} if refs is not None else {}
        (d / "multitask_results.json").write_text(_json.dumps(payload))
        return str(d)

    def test_a_caid3_only_checkpoint_is_refused_on_caid2(self, tmp_path):
        from rockfish.eval_caid3_official import assert_filtered_against

        ckpt = self._ckpt(tmp_path, ["/refs/caid3_official/disorder_pdb.fasta"])
        with pytest.raises(SystemExit) as exc:
            assert_filtered_against("caid2", ckpt)
        assert "memorisation" in str(exc.value)

    def test_a_dual_filtered_checkpoint_passes_both_rounds(self, tmp_path):
        from rockfish.eval_caid3_official import assert_filtered_against

        ckpt = self._ckpt(tmp_path, [
            "/refs/caid3_official/disorder_pdb.fasta",
            "/refs/caid2_official/disorder_pdb.fasta",
            "/refs/caid2_official/binding.fasta",
        ])
        assert_filtered_against("caid2", ckpt) is None
        assert_filtered_against("caid3", ckpt) is None

    def test_a_single_string_reference_is_accepted_not_iterated(self, tmp_path):
        """Older runs recorded one path as a bare string. Treating a string as
        a list of characters would make every check pass on 'c' in 'caid2'."""
        from rockfish.eval_caid3_official import assert_filtered_against

        ckpt = self._ckpt(tmp_path, "/refs/caid3_official/disorder_pdb.fasta")
        assert_filtered_against("caid3", ckpt) is None
        with pytest.raises(SystemExit):
            assert_filtered_against("caid2", ckpt)

    def test_a_missing_or_empty_filter_record_is_refused(self, tmp_path):
        from rockfish.eval_caid3_official import assert_filtered_against

        with pytest.raises(SystemExit) as exc:
            assert_filtered_against("caid3", self._ckpt(tmp_path, None))
        assert "none recorded" in str(exc.value)

    def test_a_checkpoint_without_metadata_is_refused(self, tmp_path):
        from rockfish.eval_caid3_official import assert_filtered_against

        d = tmp_path / "bare"
        d.mkdir()
        with pytest.raises(SystemExit) as exc:
            assert_filtered_against("caid3", str(d))
        assert "cannot be established" in str(exc.value)

    def test_the_evaluator_calls_the_guard_before_loading_weights(self):
        """A guard that runs after the model loads still runs, but a guard that
        runs after *scoring* would not — so pin the order."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rockfish", "eval_caid3_official.py")
        src = open(path).read()
        call = src.index("assert_filtered_against(args.benchmark")
        load = src.index('torch.load(ckpt')
        assert call < load, "the leak guard must precede loading the model"


class TestCaid2Composition:
    def test_the_two_rounds_have_different_published_compositions(self):
        """If these ever coincide the guard that distinguishes them is inert."""
        from colab.caid3_official import EXPECTED, EXPECTED_CAID2

        shared = set(EXPECTED) & set(EXPECTED_CAID2)
        assert shared, "the rounds share no task name"
        for task in shared:
            assert EXPECTED[task] != EXPECTED_CAID2[task], task

    def test_caid2_has_no_binding_idr_task(self):
        """CAID2 scored `binding`; `binding_idr` is a CAID3 addition. Scoring
        our binding_idr head against CAID2's binding reference would silently
        compare two different label definitions."""
        from colab.caid3_official import TASKS_CAID2

        assert "binding_idr" not in TASKS_CAID2
        assert "binding" in TASKS_CAID2

    def test_caid3_floors_are_not_applied_to_caid2(self):
        """CAID2 shares four task names with CAID3 and none of their values.
        Applying a CAID3 floor to a CAID2 AUC would print a PASS or FAIL about
        a comparison nobody registered."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rockfish", "eval_caid3_official.py")
        src = open(path).read()
        assert "active_floors" in src
        assert 'NON_INFERIORITY_FLOORS if args.benchmark == "caid3"' in src
        assert "for task, floor in active_floors.items():" in src
        assert "for task, floor in NON_INFERIORITY_FLOORS.items():" not in src

    def test_no_caid3_table_is_consulted_on_another_round(self):
        """`leader = LEADERS[task][0]` named PUNCH2 on CAID2, where PUNCH2 was
        not an entrant, and the fused comparison then opened a file that does
        not exist and took the whole evaluation down after 14 minutes of GPU.
        The leader must come from the round being scored."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rockfish", "eval_caid3_official.py")
        src = open(path).read()
        assert 'if args.benchmark == "caid3":\n            leader, leader_auc = LEADERS[task]' in src
        assert 'leader, leader_auc = top["method"], top["auc"]' in src
        assert '"leader_auc": LEADERS[task][1]' not in src
        assert src.count("LEADERS[task]") == 1, (
            "every use of the CAID3 leader table must be behind the round check")

    def test_every_opponent_file_is_checked_before_it_is_opened(self):
        """The paired loop guarded, the fused comparison did not."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rockfish", "eval_caid3_official.py")
        src = open(path).read()
        block = src[src.index("fused_better = [r for r in board"):
                    src.index("results[task] = {")]
        guard = block.index('os.path.exists(os.path.join(staged, f"{leader}.caid")')
        call = block.index("paired_bootstrap")
        assert guard < call, "the fused comparison opens the leader unguarded"

    def test_the_binding_idr_crossover_is_conditioned_on_the_file(self):
        """CAID2 has no binding_idr reference; reading it unconditionally
        crashes the round after every task has already been scored."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rockfish", "eval_caid3_official.py")
        src = open(path).read()
        assert 'os.path.isfile(idr_path)' in src
        assert 'read_reference(os.path.join(args.refs, "binding_idr.fasta"))' \
            not in src
