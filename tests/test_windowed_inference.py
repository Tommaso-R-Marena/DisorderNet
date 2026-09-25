"""Long proteins must be scored inside the regime the model was trained on.

ESM-2 was trained at 1024 tokens; the head was trained with --max-len 1022, so
no protein longer than that appeared in training at all. Scoring a 1500-residue
chain in a single pass put both far outside that regime, and the failure was not
graceful: on CAID3 Disorder-NOX targets above 1500 residues the model scored
0.5183 — chance — while AlphaFold-rsa scored 0.8967 on the same targets. The
signal was there and the model was discarding it.

These cover the windowing itself, which is easy to get subtly wrong in ways that
show up as a quietly worse number rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from rockfish.eval_caid3_official import WINDOW, _taper, _windows


class TestWindowCoverage:
    @pytest.mark.parametrize("n", [1, 10, 500, 1021, 1022, 1023, 1500, 2048,
                                   3088, 5000])
    def test_every_residue_is_covered(self, n):
        covered = np.zeros(n, dtype=int)
        for a, b in _windows(n):
            covered[a:b] += 1
        assert (covered > 0).all(), f"{int((covered == 0).sum())} uncovered at n={n}"

    @pytest.mark.parametrize("n", [1, 500, 1022, 1023, 3088])
    def test_no_window_exceeds_the_model_limit(self, n):
        assert all(b - a <= WINDOW for a, b in _windows(n))

    @pytest.mark.parametrize("n", [1, 300, 1022])
    def test_short_sequences_are_a_single_window(self, n):
        """Anything within the trained range must behave exactly as before."""
        assert _windows(n) == [(0, n)]

    def test_the_last_window_reaches_the_end(self, n=1500):
        assert _windows(n)[-1][1] == n

    def test_windows_are_ordered_and_overlapping(self, n=3000):
        w = _windows(n)
        assert all(w[i][0] < w[i + 1][0] for i in range(len(w) - 1))
        assert all(w[i + 1][0] < w[i][1] for i in range(len(w) - 1)), \
            "a gap between windows would leave residues with one-sided context"

    def test_a_long_protein_needs_several_windows(self):
        assert len(_windows(3088)) >= 4


class TestTaper:
    def test_weight_is_positive_everywhere(self):
        """A zero weight anywhere divides by zero when the overlaps are
        normalised, or silently drops the residue."""
        for n in (1, 2, 5, 100, WINDOW):
            assert (_taper(n) > 0).all(), n

    def test_the_centre_outweighs_the_edges(self):
        w = _taper(1000)
        assert w[500] > w[0] and w[500] > w[-1]

    def test_it_is_symmetric(self):
        w = _taper(101)
        assert np.allclose(w, w[::-1])


class TestWeightedAverageIsAnAverage:
    """The stitching must not shift the scores it combines."""

    def test_constant_predictions_survive_stitching(self):
        n = 3000
        acc = np.zeros(n)
        wsum = np.zeros(n)
        for a, b in _windows(n):
            w = _taper(b - a)
            acc[a:b] += 0.73 * w
            wsum[a:b] += w
        assert np.allclose(acc / wsum, 0.73)

    def test_a_ramp_survives_stitching(self):
        n = 2500
        truth = np.linspace(0.0, 1.0, n)
        acc, wsum = np.zeros(n), np.zeros(n)
        for a, b in _windows(n):
            w = _taper(b - a)
            acc[a:b] += truth[a:b] * w
            wsum[a:b] += w
        assert np.allclose(acc / wsum, truth)
