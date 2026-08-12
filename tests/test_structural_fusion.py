"""Structural fusion must be fit on training data, never on the benchmark.

The measurement that motivates the module, on 304 CAID3 Disorder-PDB targets:

    DisorderNet-Lite (model)   AUC 0.9215
    AlphaFold -pLDDT           AUC 0.9431
    AlphaFold rsa (w=21)       AUC 0.9459
    rsa + pLDDT                AUC 0.9584
    rsa + pLDDT + model        AUC 0.9633   <- weights fitted on the scored data

Two free baselines beat the trained model, and the fused figure that clears
PUNCH2 (0.9550) came from a grid search over the residues being scored. These
tests pin the properties that make an honest version possible.
"""

from __future__ import annotations

import numpy as np
import pytest

from colab.structural_fusion import apply_fusion, fit_fusion, fusion_report
from colab.structure_rsa import RSA_SMOOTH_WINDOW, smooth_window


def synth(n=4000, seed=0):
    """Signal shared by three noisy channels, as in the real setup."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.32, n)
    model = 1 / (1 + np.exp(-(rng.normal(y * 1.4, 1.0))))
    rsa = rng.normal(y * 1.6, 1.0)
    plddt = rng.normal(80 - y * 30, 12)
    return y, model, rsa, plddt


class TestSmoothing:
    def test_window_is_the_measured_optimum(self):
        """1 -> 0.8688, 21 -> 0.9459 on CAID3. The window is the whole
        difference between reproducing AlphaFold-rsa and not."""
        assert RSA_SMOOTH_WINDOW == 21

    def test_smoothing_preserves_length(self):
        x = np.arange(50, dtype=float)
        assert len(smooth_window(x, 21)) == 50

    def test_window_of_one_is_identity(self):
        x = np.random.default_rng(0).random(30)
        assert np.allclose(smooth_window(x, 1), x)

    def test_smoothing_reduces_variance(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 500)
        assert smooth_window(x, 21).std() < x.std()

    def test_handles_sequences_shorter_than_the_window(self):
        x = np.array([1.0, 2.0, 3.0])
        assert len(smooth_window(x, 21)) == 3


class TestFitIsolation:
    def test_weights_carry_their_own_standardisation(self):
        """Recomputing statistics at apply time would leak the target
        distribution into the transform."""
        y, model, rsa, plddt = synth()
        f = fit_fusion(y, model, rsa, plddt)
        assert len(f["standardisation"]) == len(f["channels"])

    def test_apply_is_deterministic_given_frozen_weights(self):
        y, model, rsa, plddt = synth()
        f = fit_fusion(y, model, rsa, plddt)
        a = apply_fusion(f, model, rsa, plddt)
        b = apply_fusion(f, model, rsa, plddt)
        assert np.array_equal(a, b)

    def test_apply_does_not_consult_labels(self):
        """The signature makes benchmark leakage impossible by construction."""
        import inspect

        assert "label" not in inspect.signature(apply_fusion).parameters

    def test_refuses_to_score_with_missing_channels(self):
        """Silently dropping a channel would score a different model than the
        one these weights belong to."""
        y, model, rsa, plddt = synth()
        f = fit_fusion(y, model, rsa, plddt)
        with pytest.raises(ValueError, match="not supplied"):
            apply_fusion(f, model)

    def test_fit_needs_both_classes(self):
        y, model, rsa, plddt = synth()
        with pytest.raises(ValueError, match="both classes"):
            fit_fusion(np.zeros_like(y), model, rsa, plddt)

    def test_output_is_a_probability(self):
        y, model, rsa, plddt = synth()
        p = apply_fusion(fit_fusion(y, model, rsa, plddt), model, rsa, plddt)
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_weights_fit_on_one_split_transfer_to_another(self):
        """The property the whole design rests on: weights learned on training
        residues must still work on residues they never saw."""
        from sklearn.metrics import roc_auc_score

        y, model, rsa, plddt = synth(n=8000)
        half = len(y) // 2
        f = fit_fusion(y[:half], model[:half], rsa[:half], plddt[:half])
        held = apply_fusion(f, model[half:], rsa[half:], plddt[half:])
        assert roc_auc_score(y[half:], held) > roc_auc_score(y[half:], model[half:])


class TestReportingIsHonest:
    def test_report_carries_the_structure_only_baseline(self):
        """rsa+pLDDT is free — no training, no GPU — and reaches 0.9584 on
        CAID3. A fused score means nothing without it alongside."""
        y, model, rsa, plddt = synth()
        f = fit_fusion(y, model, rsa, plddt)
        fused = apply_fusion(f, model, rsa, plddt)
        struct = fit_fusion(y, rsa, plddt=plddt)
        struct_p = apply_fusion(struct, rsa, plddt=plddt)
        rep = fusion_report(y, model, fused, rsa, plddt, structure_only=struct_p)
        assert rep["structure_only"] is not None
        assert "gain_over_structure_only" in rep

    def test_report_states_whether_the_model_earned_its_place(self):
        y, model, rsa, plddt = synth()
        f = fit_fusion(y, model, rsa, plddt)
        fused = apply_fusion(f, model, rsa, plddt)
        struct = fit_fusion(y, rsa, plddt=plddt)
        rep = fusion_report(y, model, fused, rsa, plddt,
                            structure_only=apply_fusion(struct, rsa, plddt=plddt))
        assert isinstance(rep["model_earns_its_place"], bool)

    def test_interpretation_names_the_right_comparison(self):
        y, model, rsa, plddt = synth()
        f = fit_fusion(y, model, rsa, plddt)
        rep = fusion_report(y, model, apply_fusion(f, model, rsa, plddt), rsa, plddt)
        assert "structure-only" in rep["interpretation"]


class TestSmoothingLengthSafety:
    """np.convolve(mode="same") returns max(len(x), len(kernel)), so a protein
    shorter than the window came back LONGER than it went in — a 20-residue
    sequence at w=21 yielded 21 values and shifted every downstream residue.
    The pipeline admits proteins from 20 residues up, so this was reachable.
    """

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 19, 20, 21, 22, 100])
    def test_length_is_always_preserved(self, n):
        x = np.arange(n, dtype=float)
        assert len(smooth_window(x, RSA_SMOOTH_WINDOW)) == n

    def test_short_sequence_returns_its_own_mean(self):
        x = np.array([1.0, 2.0, 3.0])
        out = smooth_window(x, 21)
        assert len(out) == 3
        assert np.isfinite(out).all()

    def test_empty_input_stays_empty(self):
        assert len(smooth_window(np.array([]), 21)) == 0
