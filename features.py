"""
Feature engineering for protein disorder prediction.
Combines physicochemical properties, sequence composition, complexity,
and multi-scale contextual features.
"""
import numpy as np
from collections import Counter

from window_stats import (
    SymbolWindows,
    build_index_table,
    encode_sequence,
    moving_average,
    moving_variance,
)

# ============================================================
# AMINO ACID PROPERTY SCALES
# ============================================================

# Kyte-Doolittle hydrophobicity
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Charge at pH 7
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}

# Molecular weight (Da)
MW = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'Q': 146.1, 'E': 147.1, 'G': 75.0, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1,
}

# Flexibility index (Vihinen & Torkkila, 1994)
FLEXIBILITY = {
    'A': 0.984, 'R': 1.008, 'N': 1.048, 'D': 1.068, 'C': 0.906,
    'Q': 1.037, 'E': 1.094, 'G': 1.031, 'H': 0.950, 'I': 0.927,
    'L': 0.935, 'K': 1.102, 'M': 0.952, 'F': 0.915, 'P': 1.049,
    'S': 1.046, 'T': 0.997, 'W': 0.904, 'Y': 0.929, 'V': 0.931,
}

# Disorder propensity (Top-IDP scale, Campen et al. 2008)
DISORDER_PROPENSITY = {
    'A': 0.06, 'R': 0.18, 'N': 0.007, 'D': 0.192, 'C': -0.02,
    'Q': 0.318, 'E': 0.736, 'G': 0.166, 'H': 0.303, 'I': -0.486,
    'L': -0.326, 'K': 0.586, 'M': -0.397, 'F': -0.697, 'P': 0.987,
    'S': 0.341, 'T': 0.059, 'W': -0.884, 'Y': -0.510, 'V': -0.121,
}

# Beta-sheet propensity (Chou-Fasman)
BETA_PROPENSITY = {
    'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
    'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
    'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
    'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70,
}

# Alpha-helix propensity (Chou-Fasman)
ALPHA_PROPENSITY = {
    'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
    'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06,
}

# Bulkiness
BULKINESS = {
    'A': 11.50, 'R': 14.28, 'N': 12.82, 'D': 11.68, 'C': 13.46,
    'Q': 14.45, 'E': 13.57, 'G': 3.40, 'H': 13.69, 'I': 21.40,
    'L': 21.40, 'K': 15.71, 'M': 16.25, 'F': 19.80, 'P': 17.43,
    'S': 9.47, 'T': 15.77, 'W': 21.67, 'Y': 18.03, 'V': 21.57,
}

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

ALL_SCALES = {
    'hydrophobicity': HYDROPHOBICITY,
    'charge': CHARGE,
    'mw': MW,
    'flexibility': FLEXIBILITY,
    'disorder_propensity': DISORDER_PROPENSITY,
    'beta_propensity': BETA_PROPENSITY,
    'alpha_propensity': ALPHA_PROPENSITY,
    'bulkiness': BULKINESS,
}

# Residue classes used for both the windowed and the protein-level fractions.
# These live at module scope because the global block needs them even when
# `windows` is empty.
DISORDER_PROMOTING = frozenset("AEGKPQRS")
ORDER_PROMOTING = frozenset("CFILMVWY")

DEFAULT_WINDOWS = (5, 11, 21, 41)
PRO_GLY_WINDOW = 21      # local Pro/Gly enrichment window (odd, centred)
LOW_COMPLEXITY_HALF = 25  # low-complexity flag looks at i-25 .. i+25
LOW_COMPLEXITY_MAX_UNIQUE = 8
_MAX_ENTROPY_BITS = 4.32  # log2(20), used to normalise the global entropy

# Fixed blocks: one-hot(20) + properties(8) + position(2)
#               + Pro/Gly(2) + low-complexity(1) + global(3)
_STATIC_FEATURE_DIM = 36
_PER_WINDOW_FEATURE_DIM = 42

# (20, 8) property matrix in ALL_SCALES order; unknown residues read as zeros.
_SCALE_MATRIX = np.array(
    [[scale.get(aa, 0.0) for scale in ALL_SCALES.values()] for aa in "ACDEFGHIKLMNPQRSTVWY"],
    dtype=np.float64,
)
_AA_LUT = build_index_table("ACDEFGHIKLMNPQRSTVWY", default=-1, dtype=np.int32)


def n_features(windows=DEFAULT_WINDOWS) -> int:
    """Number of columns produced by :func:`compute_features_for_protein`."""
    return _STATIC_FEATURE_DIM + _PER_WINDOW_FEATURE_DIM * len(tuple(windows))


def get_residue_properties(aa):
    """Get physicochemical property vector for an amino acid."""
    props = []
    for scale_name, scale in ALL_SCALES.items():
        props.append(scale.get(aa, 0.0))
    return props


def shannon_entropy(window):
    """Calculate Shannon entropy of amino acid composition in a window."""
    if len(window) == 0:
        return 0.0
    counts = Counter(window)
    total = len(window)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy


def sequence_complexity(window):
    """Wootton-Federhen sequence complexity."""
    if len(window) <= 1:
        return 0.0
    n = len(window)
    counts = Counter(window)
    
    # SEG-like complexity
    complexity = 0.0
    for count in counts.values():
        if count > 0:
            complexity += count * np.log2(count)
    
    if n > 0:
        complexity = (n * np.log2(n) - complexity) / (n * np.log2(min(n, 20)))
    
    return complexity


def compute_features_for_protein(sequence, windows=DEFAULT_WINDOWS):
    """
    Compute feature matrix for a protein sequence.

    Returns: numpy array of shape (seq_len, num_features)

    Fully vectorised: each windowed block is a prefix-sum difference over the
    sequence (see :mod:`window_stats`) rather than a per-residue Python loop,
    and the protein-level block is computed once instead of once per residue.
    """
    windows = tuple(windows)
    seq_len = len(sequence)
    if seq_len == 0:
        return np.zeros((0, n_features(windows)), dtype=np.float32)

    idx = encode_sequence(sequence, _AA_LUT)
    known = np.flatnonzero(idx >= 0)
    sym = SymbolWindows(sequence)
    blocks = []

    # 1. One-hot encoding (20 features)
    onehot = np.zeros((seq_len, 20), dtype=np.float64)
    onehot[known, idx[known]] = 1.0
    blocks.append(onehot)

    # 2. Physicochemical properties (8 features); unknown residues stay zero
    props = np.zeros((seq_len, 8), dtype=np.float64)
    props[known] = _SCALE_MATRIX[idx[known]]
    blocks.append(props)

    # 3. Relative position features (2 features)
    positions = np.arange(seq_len, dtype=np.float64)
    denom = float(max(seq_len - 1, 1))
    blocks.append(
        np.stack([positions / denom,
                  np.minimum(positions, (seq_len - 1) - positions) / denom], axis=1)
    )

    # Per-residue class/charge indicators shared across window scales.
    is_disorder = np.isin(idx, [AA_TO_IDX[c] for c in sorted(DISORDER_PROMOTING)])
    is_order = np.isin(idx, [AA_TO_IDX[c] for c in sorted(ORDER_PROMOTING)])
    charge = props[:, 1]  # 'charge' is the second entry of ALL_SCALES
    residue_stats = np.stack(
        [is_disorder, is_order, charge, np.abs(charge)], axis=1
    ).astype(np.float64)

    # 4. Multi-scale windowed features (42 per window)
    for w in windows:
        half_w = w // 2
        window_len = sym.window_lengths(half_w).astype(np.float64)
        entropy = sym.entropy(half_w).astype(np.float64)
        # Wootton-Federhen complexity reduces exactly to H / log2(min(n, 20)).
        with np.errstate(divide="ignore", invalid="ignore"):
            complexity = np.where(window_len > 1,
                                  entropy / np.log2(np.minimum(window_len, 20.0)), 0.0)

        stats = moving_average(residue_stats, half_w)
        blocks.append(moving_average(onehot, half_w))          # 4a composition (20)
        blocks.append(moving_average(props, half_w))           # 4b mean properties (8)
        blocks.append(moving_variance(props, half_w))          # 4c variances (8)
        blocks.append(np.stack([entropy, complexity], axis=1))  # 4d (2)
        blocks.append(stats[:, :2])                            # 4e disorder/order frac (2)
        blocks.append(stats[:, 2:])                            # 4f net / |charge| (2)

    # 5. Proline and glycine enrichment (2 features) — strong disorder indicators
    pg_half = PRO_GLY_WINDOW // 2
    pg_len = sym.window_lengths(pg_half).astype(np.float64)
    blocks.append(
        np.stack([sym.char_counts('P', pg_half) / pg_len,
                  sym.char_counts('G', pg_half) / pg_len], axis=1)
    )

    # 6. Low complexity indicator (1 feature)
    low_complexity = sym.distinct(LOW_COMPLEXITY_HALF) <= LOW_COMPLEXITY_MAX_UNIQUE
    blocks.append(low_complexity.astype(np.float64).reshape(-1, 1))

    # 7. Protein-level global features (3 features) — constant down the sequence
    global_disorder_frac = float(is_disorder.sum()) / seq_len
    global_entropy = sym.total_entropy()
    log_length = float(np.log(seq_len))
    blocks.append(
        np.broadcast_to(
            np.array([global_disorder_frac,
                      global_entropy / _MAX_ENTROPY_BITS,
                      log_length / 10.0]),
            (seq_len, 3),
        )
    )

    return np.concatenate(blocks, axis=1).astype(np.float32)


def get_feature_names(windows=DEFAULT_WINDOWS):
    """Get descriptive names for all features."""
    names = []
    
    # One-hot
    for aa in AMINO_ACIDS:
        names.append(f"onehot_{aa}")
    
    # Properties
    for scale_name in ALL_SCALES:
        names.append(f"prop_{scale_name}")
    
    # Position
    names.extend(["rel_position", "dist_to_terminus"])
    
    # Multi-scale
    for w in windows:
        for aa in AMINO_ACIDS:
            names.append(f"w{w}_comp_{aa}")
        for scale_name in ALL_SCALES:
            names.append(f"w{w}_avg_{scale_name}")
        for scale_name in ALL_SCALES:
            names.append(f"w{w}_var_{scale_name}")
        names.extend([f"w{w}_entropy", f"w{w}_complexity"])
        names.extend([f"w{w}_disorder_frac", f"w{w}_order_frac"])
        names.extend([f"w{w}_net_charge", f"w{w}_charge_asym"])
    
    # Extra features
    names.extend(["local_proline", "local_glycine", "low_complexity",
                   "global_disorder_frac", "global_entropy_norm", "log_length"])
    
    return names


if __name__ == "__main__":
    # Quick test
    test_seq = "MAEPRQEFEVMEDHAGTY"
    features = compute_features_for_protein(test_seq)
    names = get_feature_names()
    print(f"Sequence length: {len(test_seq)}")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Number of features: {len(names)}")
    print(f"Feature names sample: {names[:5]}...{names[-5:]}")
