"""Tests for the shared sliding-window primitives in window_stats.py.

Each windowed statistic is checked against a direct per-window reference so the
prefix-sum implementations cannot drift, plus the degenerate shapes the
featurisers actually hit (empty sequences, single residues, windows wider than
the sequence).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from window_stats import (
    SymbolWindows,
    build_index_table,
    encode_sequence,
    moving_average,
    moving_variance,
    prefix_sums,
    window_bounds,
)

AA = "ACDEFGHIKLMNPQRSTVWY"


def _slices(length: int, half: int):
    return [(max(0, i - half), min(length, i + half + 1)) for i in range(length)]


# ---------------------------------------------------------------------------
# window_bounds / prefix_sums
# ---------------------------------------------------------------------------
def test_window_bounds_matches_naive_slices():
    for length in (0, 1, 2, 9, 33):
        for half in (0, 1, 4, 50):
            start, end, win_len = window_bounds(length, half)
            assert list(zip(start.tolist(), end.tolist())) == _slices(length, half)
            assert np.array_equal(win_len, (end - start).astype(np.float32))


def test_window_bounds_arrays_are_readonly_and_cached():
    a = window_bounds(17, 3)
    b = window_bounds(17, 3)
    assert a[0] is b[0]  # cache hit, not a fresh allocation
    with pytest.raises(ValueError):
        a[0][0] = 99


def test_window_bounds_rejects_negative_half_width():
    with pytest.raises(ValueError):
        window_bounds(10, -1)


def test_prefix_sums_shape_and_values():
    v = np.arange(12, dtype=np.float32).reshape(4, 3)
    cs = prefix_sums(v)
    assert cs.shape == (5, 3)
    assert np.allclose(cs[0], 0.0)
    assert np.allclose(cs[-1], v.sum(axis=0))
    assert prefix_sums(np.zeros((0, 3))).shape == (1, 3)


# ---------------------------------------------------------------------------
# moving_average / moving_variance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half", [0, 1, 5, 40])
def test_moving_average_matches_naive(half):
    rng = np.random.RandomState(0)
    for length in (1, 2, 7, 60):
        v = rng.randn(length, 4).astype(np.float32)
        want = np.array([v[s:e].astype(np.float64).mean(axis=0) for s, e in _slices(length, half)])
        assert np.allclose(moving_average(v, half), want, atol=1e-5)
        # 1-D behaves like a single column
        assert np.allclose(moving_average(v[:, 0], half), want[:, 0], atol=1e-5)


@pytest.mark.parametrize("half", [0, 1, 5, 40])
def test_moving_variance_matches_naive(half):
    rng = np.random.RandomState(1)
    for length in (1, 2, 7, 60):
        v = rng.randn(length, 3).astype(np.float32) * 30.0
        want = np.array([v[s:e].astype(np.float64).var(axis=0) for s, e in _slices(length, half)])
        assert np.allclose(moving_variance(v, half), want, atol=1e-3)


def test_moving_variance_never_negative_on_large_near_constant_values():
    """E[x^2]-E[x]^2 cancels catastrophically in float32; the result must not go negative."""
    v = np.full(4000, 204.2, dtype=np.float32)
    v[::997] = 204.3
    var = moving_variance(v, 25)
    assert (var >= 0).all()
    assert not np.isnan(np.sqrt(var)).any()


def test_moving_stats_preserve_dtype_and_handle_empty():
    assert moving_average(np.zeros((0, 3), dtype=np.float32), 4).shape == (0, 3)
    assert moving_variance(np.zeros(0, dtype=np.float32), 4).shape == (0,)
    assert moving_average(np.ones((5, 2), dtype=np.float32), 1).dtype == np.float32
    assert moving_average(np.ones((5, 2), dtype=np.float64), 1).dtype == np.float64


def test_moving_average_of_constant_is_constant():
    v = np.full((50, 3), 7.5, dtype=np.float32)
    assert np.allclose(moving_average(v, 9), 7.5)
    assert np.allclose(moving_variance(v, 9), 0.0)


# ---------------------------------------------------------------------------
# index tables
# ---------------------------------------------------------------------------
def test_build_index_table_and_encode_sequence():
    table = build_index_table(AA, default=-1)
    assert encode_sequence("", table).shape == (0,)
    idx = encode_sequence("ACXY", table)
    assert idx.tolist() == [0, 1, -1, 19]
    # non-latin-1 input still yields one index per character
    assert encode_sequence("AéM", table).shape == (3,)


# ---------------------------------------------------------------------------
# SymbolWindows
# ---------------------------------------------------------------------------
def _naive_entropy(window: str) -> float:
    n = len(window)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in Counter(window).values())


@pytest.mark.parametrize("half", [0, 2, 10, 100])
def test_symbol_windows_match_naive(half):
    rng = np.random.RandomState(2)
    for length in (1, 3, 25, 120):
        seq = "".join(rng.choice(list(AA + "XU"), length))
        sym = SymbolWindows(seq)
        windows = [seq[s:e] for s, e in _slices(length, half)]

        assert sym.distinct(half).tolist() == [float(len(set(w))) for w in windows]
        assert np.allclose(sym.entropy(half), [_naive_entropy(w) for w in windows], atol=1e-6)
        assert sym.char_counts("P", half).tolist() == [float(w.count("P")) for w in windows]
        assert sym.member_counts("AEGKPQRS", half).tolist() == [
            float(sum(c in "AEGKPQRS" for c in w)) for w in windows
        ]


def test_symbol_windows_totals():
    seq = "AACCCGT"
    sym = SymbolWindows(seq)
    assert int(sym.total_counts().sum()) == len(seq)
    assert sym.total_entropy() == pytest.approx(_naive_entropy(seq))


def test_symbol_windows_empty_sequence():
    sym = SymbolWindows("")
    assert sym.length == 0
    assert sym.distinct(5).shape == (0,)
    assert sym.entropy(5).shape == (0,)
    assert sym.char_counts("P", 5).shape == (0,)
    assert sym.member_counts("AEG", 5).shape == (0,)
    assert sym.total_entropy() == 0.0


def test_symbol_windows_absent_character_is_zero():
    sym = SymbolWindows("AAAA")
    assert sym.char_counts("W", 1).tolist() == [0.0] * 4
    assert sym.member_counts("WYF", 1).tolist() == [0.0] * 4


def test_symbol_windows_counts_are_memoised_and_readonly():
    sym = SymbolWindows("ACDEFACDEF")
    first = sym.counts(2)
    assert sym.counts(2) is first
    with pytest.raises(ValueError):
        first[0, 0] = 5
