"""Shared primitives for centred sliding-window sequence statistics.

Every windowed feature in DisorderNet (moving averages/variances of amino-acid
property scales, windowed composition, Shannon entropy, distinct-symbol counts)
is a prefix-sum difference over a fixed half-width window truncated at the
sequence termini. Collecting those primitives here keeps three separate
featurisers (``features.py``, ``features_fast.py``, ``run_v6_mem.py``)
numerically consistent and removes the per-residue Python loops they used to
carry.

Two numerical decisions are deliberate:

* **Prefix sums accumulate in float64.** A float32 cumulative sum over a few
  thousand residues of the bulkiness/molecular-weight scales reaches ~1e7,
  where the float32 spacing is ~1. Differences of such sums keep only a couple
  of significant digits, which is fatal for ``E[x^2] - E[x]^2``.
* **Moving variances are clipped at zero.** Even in float64 the two moments can
  cross for near-constant windows; a negative "variance" then becomes a NaN the
  moment anything downstream takes a square root.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "window_bounds",
    "prefix_sums",
    "moving_average",
    "moving_variance",
    "build_index_table",
    "encode_sequence",
    "SymbolWindows",
]


def build_index_table(alphabet: str, default: int = 0, dtype=np.int32) -> np.ndarray:
    """256-entry byte lookup table mapping ``alphabet[i]`` to ``i``.

    Unlisted bytes map to ``default``. Use with :func:`encode_sequence` to turn
    a residue string into an index array without a per-character dict lookup.
    """
    table = np.full(256, default, dtype=dtype)
    letters = np.frombuffer(alphabet.encode("latin-1", "replace"), dtype=np.uint8)
    table[letters] = np.arange(len(alphabet), dtype=dtype)
    return table


def encode_sequence(sequence: str, table: np.ndarray) -> np.ndarray:
    """Map a sequence to indices via a :func:`build_index_table` lookup table."""
    if not sequence:
        return np.zeros(0, dtype=table.dtype)
    raw = np.frombuffer(sequence.encode("latin-1", "replace"), dtype=np.uint8)
    return table[raw]

# Bounds only depend on (length, half-width). Featurisers ask for the same few
# pairs many times per protein, and for the same lengths across a batch.
_BOUNDS_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_BOUNDS_CACHE_MAX = 512


def window_bounds(length: int, half_width: int):
    """``(start, end_exclusive, window_length)`` arrays for centred windows.

    Window ``i`` spans ``[max(i - half_width, 0), min(i + half_width, L - 1) + 1)``.
    The returned arrays are cached and read-only; do not mutate them.
    """
    if half_width < 0:
        raise ValueError(f"half_width must be non-negative, got {half_width}")
    key = (length, half_width)
    cached = _BOUNDS_CACHE.get(key)
    if cached is not None:
        return cached

    i = np.arange(length)
    start = np.maximum(i - half_width, 0)
    end = np.minimum(i + half_width, length - 1) + 1
    win_len = (end - start).astype(np.float32)
    for arr in (start, end, win_len):
        arr.flags.writeable = False

    if len(_BOUNDS_CACHE) >= _BOUNDS_CACHE_MAX:
        _BOUNDS_CACHE.clear()
    _BOUNDS_CACHE[key] = (start, end, win_len)
    return start, end, win_len


def prefix_sums(values: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Zero-prefixed cumulative sums along axis 0 (shape ``(L + 1, ...)``)."""
    values = np.asarray(values)
    out = np.zeros((values.shape[0] + 1,) + values.shape[1:], dtype=dtype)
    if values.shape[0]:
        np.cumsum(values, axis=0, dtype=dtype, out=out[1:])
    return out


def _result_dtype(values: np.ndarray):
    return np.float64 if np.asarray(values).dtype == np.float64 else np.float32


def _broadcast_len(win_len: np.ndarray, ndim: int) -> np.ndarray:
    return win_len if ndim == 1 else win_len[:, None]


def moving_average(values, half_width: int) -> np.ndarray:
    """Centred moving mean over axis 0, truncated at the termini."""
    values = np.asarray(values)
    length = values.shape[0]
    out_dtype = _result_dtype(values)
    if length == 0:
        return np.zeros(values.shape, dtype=out_dtype)

    start, end, win_len = window_bounds(length, half_width)
    csum = prefix_sums(values)
    total = csum[end] - csum[start]
    total /= _broadcast_len(win_len, values.ndim)
    return total.astype(out_dtype, copy=False)


def moving_variance(values, half_width: int) -> np.ndarray:
    """Centred moving variance ``E[x^2] - E[x]^2`` over axis 0, clipped at 0."""
    values = np.asarray(values)
    length = values.shape[0]
    out_dtype = _result_dtype(values)
    if length == 0:
        return np.zeros(values.shape, dtype=out_dtype)

    start, end, win_len = window_bounds(length, half_width)
    v64 = values.astype(np.float64, copy=False)
    csum = prefix_sums(v64)
    csum_sq = prefix_sums(v64 * v64)
    denom = _broadcast_len(win_len, values.ndim)
    mean = (csum[end] - csum[start]) / denom
    mean_sq = (csum_sq[end] - csum_sq[start]) / denom
    var = mean_sq - mean * mean
    np.maximum(var, 0.0, out=var)
    return var.astype(out_dtype, copy=False)


class SymbolWindows:
    """Windowed symbol statistics for one sequence.

    Builds the alphabet actually present in the sequence (usually <= 21 symbols)
    once, then answers window queries with prefix-sum differences instead of
    rebuilding a ``set``/``Counter`` per residue. Results are memoised per
    half-width because callers typically ask for counts, entropy and distinct
    counts at the same width.
    """

    __slots__ = (
        "sequence", "length", "codes", "n_symbols",
        "_counts_prefix", "_counts", "_byte_to_column",
    )

    def __init__(self, sequence: str):
        self.sequence = sequence
        self.length = len(sequence)
        self._counts: dict[int, np.ndarray] = {}
        # -1 marks a byte that does not occur in this sequence.
        self._byte_to_column = np.full(256, -1, dtype=np.int64)

        if self.length == 0:
            self.codes = np.zeros(0, dtype=np.int64)
            self.n_symbols = 0
            self._counts_prefix = np.zeros((1, 0), dtype=np.int32)
            return

        # latin-1 with 'replace' keeps one byte per character for any input, so
        # the code array stays aligned with residue positions.
        raw = np.frombuffer(sequence.encode("latin-1", "replace"), dtype=np.uint8)
        symbols, codes = np.unique(raw, return_inverse=True)
        codes = np.asarray(codes).reshape(-1)
        self.codes = codes
        self.n_symbols = symbols.size
        self._byte_to_column[symbols] = np.arange(self.n_symbols)

        onehot = np.zeros((self.length, self.n_symbols), dtype=np.int32)
        onehot[np.arange(self.length), codes] = 1
        self._counts_prefix = prefix_sums(onehot, dtype=np.int32)

    def counts(self, half_width: int) -> np.ndarray:
        """``(L, n_symbols)`` integer symbol counts per window."""
        cached = self._counts.get(half_width)
        if cached is not None:
            return cached
        if self.length == 0:
            result = np.zeros((0, self.n_symbols), dtype=np.int32)
        else:
            start, end, _ = window_bounds(self.length, half_width)
            result = self._counts_prefix[end] - self._counts_prefix[start]
        result.flags.writeable = False
        self._counts[half_width] = result
        return result

    def total_counts(self) -> np.ndarray:
        """``(n_symbols,)`` symbol counts over the whole sequence."""
        return self._counts_prefix[-1]

    def total_entropy(self) -> float:
        """Shannon entropy (bits) of the whole-sequence symbol distribution."""
        if self.length == 0:
            return 0.0
        counts = self.total_counts().astype(np.float64)
        counts = counts[counts > 0]
        p = counts / counts.sum()
        return float(-(p * np.log2(p)).sum())

    def window_lengths(self, half_width: int) -> np.ndarray:
        if self.length == 0:
            return np.zeros(0, dtype=np.float32)
        return window_bounds(self.length, half_width)[2]

    def distinct(self, half_width: int) -> np.ndarray:
        """Number of distinct symbols per window (float32)."""
        if self.length == 0:
            return np.zeros(0, dtype=np.float32)
        return (self.counts(half_width) > 0).sum(axis=1).astype(np.float32)

    def entropy(self, half_width: int) -> np.ndarray:
        """Shannon entropy (bits) of the symbol distribution per window."""
        if self.length == 0:
            return np.zeros(0, dtype=np.float32)
        counts = self.counts(half_width).astype(np.float64)
        win_len = self.window_lengths(half_width).astype(np.float64)
        # H = log2(n) - (1/n) * sum_k c_k log2(c_k), with 0*log2(0) := 0.
        clogc = np.zeros_like(counts)
        np.log2(counts, out=clogc, where=counts > 0)
        clogc *= counts
        ent = np.log2(win_len) - clogc.sum(axis=1) / win_len
        np.maximum(ent, 0.0, out=ent)
        return ent.astype(np.float32)

    def char_counts(self, char: str, half_width: int) -> np.ndarray:
        """Occurrences of a single character per window (float32)."""
        if self.length == 0:
            return np.zeros(0, dtype=np.float32)
        byte = np.frombuffer(char.encode("latin-1", "replace"), dtype=np.uint8)[0]
        column = int(self._byte_to_column[byte])
        if column < 0:  # character absent from this sequence
            return np.zeros(self.length, dtype=np.float32)
        return self.counts(half_width)[:, column].astype(np.float32)

    def member_counts(self, chars, half_width: int) -> np.ndarray:
        """Occurrences of any character in ``chars`` per window (float32)."""
        if self.length == 0:
            return np.zeros(0, dtype=np.float32)
        wanted = np.frombuffer("".join(sorted(set(chars))).encode("latin-1", "replace"),
                               dtype=np.uint8)
        columns = self._byte_to_column[wanted]
        columns = columns[columns >= 0]
        if columns.size == 0:
            return np.zeros(self.length, dtype=np.float32)
        return self.counts(half_width)[:, columns].sum(axis=1).astype(np.float32)
