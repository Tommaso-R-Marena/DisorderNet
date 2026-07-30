"""Optimized feature engineering using vectorized numpy operations.

Layout (per residue, for the default three windows) is 162 columns:

    one-hot (20) | per-residue properties (8) | position (2)
    | per window: composition (20), mean properties (8), property variance (8),
      [entropy, disorder frac, order frac, net charge, |charge|, complexity] (6)
    | local Pro/Gly enrichment (2) | low-complexity flag (1) | global (3)

Every windowed block is a prefix-sum difference (see :mod:`window_stats`), so
the cost is O(L * n_scales) numpy work rather than the O(L * window) Python
loops this module used to run despite its name.
"""
import numpy as np

from window_stats import (
    SymbolWindows,
    build_index_table,
    encode_sequence,
    moving_average,
    moving_variance,
)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Property arrays indexed by AA_TO_IDX
_HYDRO = np.array([1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                    1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3])
_CHARGE = np.array([0, 0, -1, -1, 0, 0, 0.1, 0, 1, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
_FLEX = np.array([0.984, 0.906, 1.068, 1.094, 0.915, 1.031, 0.950, 0.927, 1.102, 0.935,
                  0.952, 1.048, 1.049, 1.037, 1.008, 1.046, 0.997, 0.931, 0.904, 0.929])
_DISPROP = np.array([0.06, -0.02, 0.192, 0.736, -0.697, 0.166, 0.303, -0.486, 0.586, -0.326,
                     -0.397, 0.007, 0.987, 0.318, 0.18, 0.341, 0.059, -0.121, -0.884, -0.510])
_BETA = np.array([0.83, 1.19, 0.54, 0.37, 1.38, 0.75, 0.87, 1.60, 0.74, 1.30,
                  1.05, 0.89, 0.55, 1.10, 0.93, 0.75, 1.19, 1.70, 1.37, 1.47])
_ALPHA = np.array([1.42, 0.70, 1.01, 1.51, 1.13, 0.57, 1.00, 1.08, 1.16, 1.21,
                   1.45, 0.67, 0.57, 1.11, 0.98, 0.77, 0.83, 1.06, 1.08, 0.69])
_BULK = np.array([11.50, 13.46, 11.68, 13.57, 19.80, 3.40, 13.69, 21.40, 15.71, 21.40,
                  16.25, 12.82, 17.43, 14.45, 14.28, 9.47, 15.77, 21.57, 21.67, 18.03])
_MW = np.array([89.1, 121.2, 133.1, 147.1, 165.2, 75.0, 155.2, 131.2, 146.2, 131.2,
                149.2, 132.1, 115.1, 146.1, 174.2, 105.1, 119.1, 117.1, 204.2, 181.2])

ALL_PROPS = np.stack([_HYDRO, _CHARGE, _FLEX, _DISPROP, _BETA, _ALPHA, _BULK, _MW])  # (8, 20)
_PROPS_T = ALL_PROPS.T.astype(np.float32)  # (20, 8), one row per amino acid

# Disorder/order promoting sets
_DISORDER_PROMOTING = set("AEGKPQRS")
_ORDER_PROMOTING = set("CFILMVWY")

DEFAULT_WINDOWS = (7, 15, 31)

# Hard-coded scales for the trailing blocks (kept for layout compatibility).
_PG_HALF = 10
_LOW_COMPLEXITY_HALF = 15
_LOW_COMPLEXITY_MAX_UNIQUE = 8

# Unknown residues map to -1 so they contribute nothing to one-hot/composition.
_AA_LUT = build_index_table("".join(AMINO_ACIDS), default=-1, dtype=np.int32)

# Fixed-size blocks: one-hot(20) + props(8) + position(2) + pg(2) + lc(1) + global(3)
_STATIC_DIM = 36
_PER_WINDOW_DIM = 42


def n_features(windows=DEFAULT_WINDOWS) -> int:
    """Number of columns produced by :func:`compute_features_fast`."""
    return _STATIC_DIM + _PER_WINDOW_DIM * len(tuple(windows))


def seq_to_indices(sequence):
    """Convert sequence to index array. Unknown = -1 mapped to all zeros."""
    return encode_sequence(sequence, _AA_LUT)


def compute_features_fast(sequence, windows=DEFAULT_WINDOWS):
    """Vectorized feature computation. Returns (seq_len, n_features) array."""
    windows = tuple(windows)
    L = len(sequence)
    if L == 0:
        return np.zeros((0, n_features(windows)), dtype=np.float32)

    idx = seq_to_indices(sequence)
    valid = idx >= 0
    sym = SymbolWindows(sequence)
    features_list = []

    # 1. One-hot encoding (20)
    onehot = np.zeros((L, 20), dtype=np.float32)
    rows = np.flatnonzero(valid)
    onehot[rows, idx[rows]] = 1.0
    features_list.append(onehot)

    # 2. Per-residue properties (8) — unknown residues stay all-zero
    props = np.zeros((L, 8), dtype=np.float32)
    props[rows] = _PROPS_T[idx[rows]]
    features_list.append(props)

    # 3. Position features (2)
    positions = np.arange(L, dtype=np.float32)
    denom = np.float32(max(L - 1, 1))
    features_list.append(
        np.stack([positions / denom, np.minimum(positions, (L - 1) - positions) / denom], axis=1)
    )

    # Per-residue charge (0 for unknown residues), shared by every window scale.
    charge = props[:, 1]
    abs_charge = np.abs(charge)
    disorder_mask = np.isin(idx, [AA_TO_IDX[c] for c in sorted(_DISORDER_PROMOTING)])
    order_mask = np.isin(idx, [AA_TO_IDX[c] for c in sorted(_ORDER_PROMOTING)])
    extras = np.stack([disorder_mask, order_mask], axis=1).astype(np.float32)
    charges = np.stack([charge, abs_charge], axis=1)

    # 4. Multi-scale windowed features
    for w in windows:
        half = w // 2
        win_len = sym.window_lengths(half)

        # 4a. Windowed composition (20)
        features_list.append(moving_average(onehot, half))

        # 4b. Windowed mean / variance of properties (8 + 8)
        features_list.append(moving_average(props, half))
        features_list.append(moving_variance(props, half))

        # 4c. entropy, disorder frac, order frac, net charge, |charge|, complexity (6)
        frac = moving_average(extras, half)
        charge_stats = moving_average(charges, half)
        complexity = sym.distinct(half) / np.minimum(win_len, 20.0)
        features_list.append(
            np.stack([
                sym.entropy(half),
                frac[:, 0], frac[:, 1],
                charge_stats[:, 0], charge_stats[:, 1],
                complexity,
            ], axis=1)
        )

    # 5. Proline/Glycine enrichment (2)
    pg_len = sym.window_lengths(_PG_HALF)
    features_list.append(
        np.stack([
            sym.char_counts("P", _PG_HALF) / pg_len,
            sym.char_counts("G", _PG_HALF) / pg_len,
        ], axis=1)
    )

    # 6. Low complexity (1)
    lc = (sym.distinct(_LOW_COMPLEXITY_HALF) <= _LOW_COMPLEXITY_MAX_UNIQUE)
    features_list.append(lc.astype(np.float32).reshape(-1, 1))

    # 7. Global features (3)
    global_dp = float(disorder_mask.sum()) / L
    global_entropy = sym.total_entropy()
    global_feats = np.empty((L, 3), dtype=np.float32)
    global_feats[:, 0] = global_dp
    global_feats[:, 1] = global_entropy / 4.32
    global_feats[:, 2] = np.log(L) / 10.0
    features_list.append(global_feats)

    return np.concatenate(features_list, axis=1)


if __name__ == "__main__":
    seq = "MAEPRQEFEVMEDHAGTYGLGK" * 5
    f = compute_features_fast(seq)
    print(f"Seq len: {len(seq)}, Features: {f.shape}")
