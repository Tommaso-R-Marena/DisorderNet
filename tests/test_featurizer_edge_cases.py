"""Edge-case and equivalence tests for the vectorised featurisers.

These pin down behaviour that the previous per-residue implementations either
got wrong (empty sequences, an empty `windows` list) or computed with enough
float32 cumulative-sum error to produce negative variances.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import features
import features_fast
from confidence import conformal_quantile, expected_calibration_error
from features import compute_features_for_protein, get_feature_names
from features_fast import compute_features_fast
from run_v6_mem import PHYS_DIM, phys

AA = "ACDEFGHIKLMNPQRSTVWY"


def _random_sequence(length: int, seed: int = 0, alphabet: str = AA) -> str:
    rng = np.random.RandomState(seed)
    return "".join(rng.choice(list(alphabet), length))


# ---------------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------------
def test_features_empty_sequence_keeps_two_dimensions():
    out = compute_features_for_protein("")
    assert out.shape == (0, features.n_features())
    assert out.dtype == np.float32


def test_features_with_no_windows_does_not_raise():
    """The global block references the disorder-promoting set, which used to be
    defined only inside the per-window loop (NameError when `windows` is empty)."""
    out = compute_features_for_protein("ACDEFGHIK", windows=[])
    assert out.shape == (9, features.n_features([]))
    assert np.isfinite(out).all()


def test_features_column_count_matches_names_for_custom_windows():
    for windows in ([], [5], (7, 21), (5, 11, 21, 41)):
        n = features.n_features(windows)
        assert len(get_feature_names(windows)) == n
        assert compute_features_for_protein("MAEPRQEFEV", windows=windows).shape[1] == n


def test_features_global_block_is_constant_down_the_sequence():
    out = compute_features_for_protein(_random_sequence(60, seed=3))
    global_block = out[:, -3:]
    assert np.allclose(global_block, global_block[0])


def test_features_windowed_composition_sums_to_one():
    seq = _random_sequence(80, seed=4)
    out = compute_features_for_protein(seq)
    # first window block starts right after one-hot(20) + props(8) + position(2)
    comp = out[:, 30:50]
    assert np.allclose(comp.sum(axis=1), 1.0, atol=1e-5)


def test_features_unknown_residues_have_zero_onehot_and_properties():
    out = compute_features_for_protein("AXC")
    assert out[1, :20].sum() == 0.0
    assert np.allclose(out[1, 20:28], 0.0)


def test_features_variances_are_non_negative():
    seq = _random_sequence(400, seed=5)
    out = compute_features_for_protein(seq)
    for w_start in (30, 30 + 42, 30 + 84, 30 + 126):
        variances = out[:, w_start + 28:w_start + 36]
        assert (variances >= 0).all()


# ---------------------------------------------------------------------------
# features_fast.py
# ---------------------------------------------------------------------------
def test_features_fast_empty_sequence():
    out = compute_features_fast("")
    assert out.shape == (0, features_fast.n_features())


def test_features_fast_width_tracks_window_count():
    for windows in ([5], (7, 15), (7, 15, 31)):
        expected = features_fast.n_features(windows)
        assert compute_features_fast("MAEPRQEFEV", windows=windows).shape[1] == expected


def test_features_fast_single_residue():
    out = compute_features_fast("M")
    assert out.shape == (1, 162)
    assert np.isfinite(out).all()


def test_features_fast_composition_and_onehot_agree_at_window_one():
    seq = "MAEPRQEFEV"
    out = compute_features_fast(seq, windows=[1])
    assert np.allclose(out[:, :20], out[:, 30:50])


def test_features_fast_variances_are_non_negative():
    out = compute_features_fast(_random_sequence(500, seed=6))
    for w_start in (30, 30 + 42, 30 + 84):
        assert (out[:, w_start + 28:w_start + 36] >= 0).all()


def test_features_fast_handles_unknown_residues():
    out = compute_features_fast("ACXDE")
    assert out[2, :20].sum() == 0.0
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# run_v6_mem.phys
# ---------------------------------------------------------------------------
def test_phys_empty_sequence():
    out = phys("")
    assert out.shape == (0, PHYS_DIM)
    assert out.dtype == np.float32


@pytest.mark.parametrize("length", [1, 2, 5, 120])
def test_phys_shape_and_finiteness(length):
    out = phys(_random_sequence(length, seed=length))
    assert out.shape == (length, PHYS_DIM)
    assert np.isfinite(out).all()


def test_phys_tolerates_non_standard_residues():
    out = phys("MAXUZ" * 5)
    assert out.shape == (25, PHYS_DIM)
    assert np.isfinite(out).all()


def test_phys_distinct_residue_columns_match_naive_count():
    """Columns 110/111 are len(set(window)) / min(len(window), 20).

    Layout: props 7 | position 2 | dis/ord 2 | 5x11 window means 55
            | 3x2 hydro/disprop variance 6 | 3x12 key-residue means 36
            | Pro/Gly 2 | distinct 2 | hydro variance 2 | disprop delta 1 | global 3
    """
    seq = _random_sequence(150, seed=7, alphabet=AA + "X")
    out = phys(seq)
    length = len(seq)
    for column, half in ((110, 10), (111, 25)):
        want = [
            len(set(seq[max(0, i - half):min(length, i + half + 1)]))
            / min(min(length, i + half + 1) - max(0, i - half), 20)
            for i in range(length)
        ]
        assert np.allclose(out[:, column], want, atol=1e-6)


# ---------------------------------------------------------------------------
# confidence.py
# ---------------------------------------------------------------------------
def _naive_ece(y_true, y_prob, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += (mask.sum() / len(y_true)) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(total)


def test_ece_matches_binned_reference():
    rng = np.random.RandomState(8)
    prob = rng.rand(5000)
    true = (rng.rand(5000) < prob).astype(np.float64)
    assert expected_calibration_error(true, prob) == pytest.approx(_naive_ece(true, prob), abs=1e-12)


def test_ece_edge_cases():
    assert expected_calibration_error([], []) == 0.0
    # perfectly calibrated deterministic predictions
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.5])
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.5, 0.5], n_bins=0)


def test_conformal_quantile_rejects_out_of_range_alpha():
    prob = np.linspace(0.0, 1.0, 20)
    true = (prob > 0.5).astype(int)
    for bad_alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            conformal_quantile(prob, true, alpha=bad_alpha)
    assert conformal_quantile(prob, true, alpha=0.1) is not None


# ---------------------------------------------------------------------------
# interval extraction
# ---------------------------------------------------------------------------
def _naive_intervals(binary, min_len=1):
    out, start = [], None
    for i, v in enumerate(binary):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(binary) - start >= min_len:
        out.append((start, len(binary)))
    return out


def test_intervals_from_binary_matches_reference():
    from colab.biological_utility import intervals_from_binary

    rng = np.random.RandomState(9)
    for _ in range(50):
        arr = (rng.rand(rng.randint(1, 120)) < 0.4).astype(np.int8)
        for min_len in (1, 3, 7):
            assert intervals_from_binary(arr, min_len=min_len) == _naive_intervals(arr, min_len)


def test_intervals_from_binary_edge_cases():
    from colab.biological_utility import intervals_from_binary

    assert intervals_from_binary(np.array([], dtype=np.int8)) == []
    assert intervals_from_binary(np.zeros(5, dtype=np.int8)) == []
    assert intervals_from_binary(np.ones(5, dtype=np.int8)) == [(0, 5)]
    # boolean input and 2-D input both work
    assert intervals_from_binary(np.array([True, True, False, True])) == [(0, 2), (3, 4)]
    assert intervals_from_binary(np.array([[1, 1], [0, 1]], dtype=np.int8)) == [(0, 2), (3, 4)]
    # values other than 1 still count as "on"
    assert intervals_from_binary(np.array([0, 5, 5, 0])) == [(1, 3)]


# ---------------------------------------------------------------------------
# biophysics
# ---------------------------------------------------------------------------
def test_scd_lite_matches_naive_double_sum():
    from colab.idr_layer_biophysics import _charges, scd_lite

    def naive(seq):
        if len(seq) < 4:
            return 0.0
        q = _charges(seq).astype(np.float64)
        n = len(q)
        w = min(40, n - 1)
        total = 0.0
        for i in range(n):
            for j in range(i + 1, min(n, i + w + 1)):
                total += float(q[i]) * float(q[j]) * math.sqrt(j - i)
        return float(total / n)

    for seq in ("KKKEEE", "K" * 10 + "E" * 10, _random_sequence(300, seed=11), "AAAA", "KE"):
        assert scd_lite(seq) == pytest.approx(naive(seq), rel=1e-9, abs=1e-9)


def test_charges_lookup_matches_membership_test():
    from colab.idr_layer_biophysics import _charges

    seq = _random_sequence(200, seed=12, alphabet=AA + "X")
    want = [1.0 if c in "KRH" else (-1.0 if c in "DE" else 0.0) for c in seq]
    assert _charges(seq).tolist() == want
    assert _charges("").shape == (0,)
