"""The quick-screen verdict ladder must live on the post-leakage-fix scale.

The old ladder used absolute cuts (HIGH at proj_hi>=0.90 and stacked>=0.84)
calibrated against baselines of 0.817/0.831 — numbers produced by random protein
splits with a fold soup and in-sample thresholds. After the homology-split fixes
the same pipeline measures ~0.75-0.79, so every screen graded STOP no matter what
it showed, and the tool silently stopped discriminating.
"""

from __future__ import annotations

from colab.quick_screen import (
    ESMDISPRED_REFERENCE,
    MEASURED_CAID3,
    MEASURED_GPU_CV_AUC,
    MEASURED_STACKED_CV_AUC,
    MEASURED_V6_CV_AUC,
    NOISE_FLOOR_AUC,
    assess_breakthrough_potential,
)


def verdict(stacked, gpu=None, **kw):
    gpu = MEASURED_GPU_CV_AUC if gpu is None else gpu
    return assess_breakthrough_potential(
        gpu_pooled_auc=gpu, stacked_pooled_auc=stacked, **kw
    )


class TestScale:
    def test_baselines_are_the_homology_split_measurements(self):
        """Not the pre-fix 0.817/0.831, which no current run reproduces."""
        assert MEASURED_GPU_CV_AUC < 0.80
        assert MEASURED_V6_CV_AUC < 0.80
        assert MEASURED_STACKED_CV_AUC < 0.80

    def test_the_gbdt_still_beats_the_language_model(self):
        """The finding that motivated the lite architecture. If this inverts,
        the constants were edited without a rerun behind them."""
        assert MEASURED_V6_CV_AUC > MEASURED_GPU_CV_AUC

    def test_caid3_and_cv_scales_are_kept_apart(self):
        """Comparing a CV AUC to ESMDisPred's CAID3 number is the category error
        this project already had to correct once."""
        assert MEASURED_CAID3 > MEASURED_STACKED_CV_AUC
        assert ESMDISPRED_REFERENCE > MEASURED_CAID3


class TestLadder:
    def test_baseline_performance_is_not_a_result(self):
        v = verdict(MEASURED_STACKED_CV_AUC)
        assert v.tier == "LOW"
        assert not v.proceed_full_ultra

    def test_a_sub_noise_gain_is_not_a_result(self):
        """The failure the old ladder could not express: +0.02 looks like a win
        and is smaller than the gap between two identical reruns."""
        v = verdict(MEASURED_STACKED_CV_AUC + 0.9 * NOISE_FLOOR_AUC)
        assert v.tier == "LOW"
        assert not v.proceed_full_ultra
        assert "noise floor" in v.headline

    def test_clearing_one_noise_floor_is_moderate_not_a_green_light(self):
        v = verdict(MEASURED_STACKED_CV_AUC + 1.2 * NOISE_FLOOR_AUC)
        assert v.tier == "MODERATE"
        assert not v.proceed_full_ultra
        assert "seeds" in v.recommendation

    def test_two_noise_floors_green_lights_the_full_run(self):
        v = verdict(MEASURED_STACKED_CV_AUC + 2.5 * NOISE_FLOOR_AUC, gpu=0.70)
        assert v.tier == "HIGH"
        assert v.proceed_full_ultra

    def test_regression_below_baseline_stops(self):
        v = verdict(MEASURED_STACKED_CV_AUC - 2 * NOISE_FLOOR_AUC)
        assert v.tier == "STOP"
        assert not v.proceed_full_ultra

    def test_a_strong_screen_is_not_graded_stop(self):
        """The concrete symptom of the stale scale: a screen well above every
        measured baseline used to fall through to STOP."""
        assert verdict(0.86, gpu=0.80).tier != "STOP"

    def test_ladder_is_monotonic_in_the_stacked_auc(self):
        order = {"STOP": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}
        tiers = [
            order[verdict(MEASURED_STACKED_CV_AUC + d, gpu=0.70).tier]
            for d in (-0.10, -0.03, 0.0, 0.03, 0.06, 0.10)
        ]
        assert tiers == sorted(tiers), tiers


class TestHonesty:
    def test_the_projection_is_labelled_unvalidated(self):
        """`expected_ultra_uplift` was fitted under the leaking evaluation and
        has never been checked against a homology-split full run."""
        v = verdict(MEASURED_STACKED_CV_AUC + 3 * NOISE_FLOOR_AUC, gpu=0.70)
        assert "unvalidated" in v.recommendation.lower()

    def test_headline_states_the_margin_against_a_named_baseline(self):
        v = verdict(MEASURED_STACKED_CV_AUC + 0.05, gpu=0.70)
        assert f"{MEASURED_STACKED_CV_AUC:.3f}" in v.headline
        assert "+0.050" in v.headline
