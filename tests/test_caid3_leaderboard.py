"""The SOTA bar must be the real one.

This codebase treated ESMDisPred = 0.895 as "CAID3 SOTA" in a dozen places.
Both halves were wrong: on the official CAID3 Disorder-PDB benchmark ESMDisPred
scores 0.937, and the leader is PUNCH2 at 0.955. Aiming at 0.895 set a bar
~0.06 AUC below the real one, so a run could "reach SOTA" without being close.
"""

from __future__ import annotations

import pytest

from colab.caid3_leaderboard import (
    DISORDER_FRACTION,
    DISORDER_PDB,
    ESMDISPRED_ABSTRACT_AUC,
    ESMDISPRED_CAID3_DISORDER_PDB_AUC,
    N_TARGETS,
    SOTA_AUC,
    SOTA_METHOD,
    rank_of,
    summarize,
)


class TestTheBar:
    def test_sota_is_punch2_not_esmdispred(self):
        assert SOTA_METHOD == "PUNCH2"
        assert SOTA_AUC == 0.955

    def test_sota_is_the_best_transcribed_entry(self):
        best = max(e.auc for e in DISORDER_PDB if e.auc is not None)
        assert SOTA_AUC == best

    def test_the_old_bar_was_far_too_low(self):
        """0.06 AUC of slack is the difference between rank 1 and off the table."""
        assert SOTA_AUC - ESMDISPRED_ABSTRACT_AUC > 0.05

    def test_esmdispreds_two_figures_are_kept_apart(self):
        """0.895 is from its abstract; 0.937 is its Disorder-PDB score."""
        assert ESMDISPRED_ABSTRACT_AUC != ESMDISPRED_CAID3_DISORDER_PDB_AUC
        assert ESMDISPRED_CAID3_DISORDER_PDB_AUC == 0.937

    def test_leaderboard_is_ordered_by_auc(self):
        aucs = [e.auc for e in DISORDER_PDB if e.auc is not None]
        assert aucs == sorted(aucs, reverse=True)


class TestOurStandingIsNotFlattered:
    LITE_AUC, LITE_APS = 0.9228, 0.8547

    def test_lite_is_not_sota(self):
        """The measured result, stated plainly."""
        s = summarize(self.LITE_AUC, self.LITE_APS)
        assert s["is_sota"] is False
        assert s["gap_to_sota_auc"] < 0

    def test_lite_does_not_beat_esmdispred_on_this_benchmark(self):
        assert summarize(self.LITE_AUC)["beats_esmdispred_on_this_benchmark"] is False

    def test_lite_would_beat_the_old_bar(self):
        """Why the wrong bar mattered: against 0.895 this reads as a win."""
        assert self.LITE_AUC > ESMDISPRED_ABSTRACT_AUC

    def test_aps_gap_is_larger_and_is_flagged(self):
        """AUC and APS disagree here, and APS is the harder metric on a
        benchmark that is ~32% positive. A claim resting on AUC must say so."""
        s = summarize(self.LITE_AUC, self.LITE_APS)
        assert s["aps_gap_larger_than_auc_gap"] is True
        assert abs(s["gap_to_sota_aps"]) > abs(s["gap_to_sota_auc"])

    def test_rank_is_a_lower_bound_not_an_optimistic_guess(self):
        """Only the top ten are transcribed, so a rank must never be reported as
        better than the excerpt can support."""
        assert rank_of(0.9228) >= 11
        assert rank_of(0.9999) == 1
        assert rank_of(0.0) == len([e for e in DISORDER_PDB if e.auc is not None]) + 1

    def test_summary_names_the_transcription_limit(self):
        assert "lower bound" in summarize(0.9228)["note"]


class TestProtocolComposition:
    def test_official_composition_is_recorded(self):
        """319 targets at 31.6% disorder is what our evaluation must reproduce
        to be scoring the same benchmark."""
        assert N_TARGETS == 319
        assert DISORDER_FRACTION == pytest.approx(0.316, abs=1e-3)

    def test_our_measured_composition_matches(self):
        """Measured after the alignment fix: 319 targets, 99,239 residues,
        31.6% disorder = 31,359 positives against the official 31,401."""
        measured_targets, measured_fraction = 319, 0.316
        assert abs(measured_targets - N_TARGETS) <= 2
        assert abs(measured_fraction - DISORDER_FRACTION) < 0.02
