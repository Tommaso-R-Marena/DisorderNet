"""
Optimized feature engineering using vectorized numpy operations.
"""
import numpy as np
from collections import Counter

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

# Disorder/order promoting sets
_DISORDER_PROMOTING = set("AEGKPQRS")
_ORDER_PROMOTING = set("CFILMVWY")


def seq_to_indices(sequence):
    """Convert sequence to index array. Unknown = -1 mapped to all zeros."""
    indices = np.array([AA_TO_IDX.get(aa, -1) for aa in sequence], dtype=np.int32)
    return indices


def compute_features_fast(sequence, windows=[7, 15, 31]):
    """Vectorized feature computation. Returns (seq_len, n_features) array."""
    L = len(sequence)
    idx = seq_to_indices(sequence)
    valid = idx >= 0
    
    features_list = []
    
    # 1. One-hot encoding (20)
    onehot = np.zeros((L, 20), dtype=np.float32)
    for i in range(L):
        if valid[i]:
            onehot[i, idx[i]] = 1.0
    features_list.append(onehot)
    
    # 2. Per-residue properties (8)
    props = np.zeros((L, 8), dtype=np.float32)
    for i in range(L):
        if valid[i]:
            props[i] = ALL_PROPS[:, idx[i]]
    features_list.append(props)
    
    # 3. Position features (2)
    positions = np.arange(L, dtype=np.float32)
    rel_pos = positions / max(L - 1, 1)
    dist_term = np.minimum(positions, L - 1 - positions) / max(L - 1, 1)
    features_list.append(np.stack([rel_pos, dist_term], axis=1))
    
    # 4. Multi-scale windowed features (using convolution-like operations)
    for w in windows:
        half = w // 2
        
        # 4a. Windowed composition (20 per window)
        # Use cumsum for efficient windowed counting
        comp = np.zeros((L, 20), dtype=np.float32)
        for i in range(L):
            s, e = max(0, i - half), min(L, i + half + 1)
            window_idx = idx[s:e]
            window_valid = valid[s:e]
            wlen = e - s
            for j in range(len(window_idx)):
                if window_valid[j]:
                    comp[i, window_idx[j]] += 1.0 / wlen
        features_list.append(comp)
        
        # 4b. Windowed average properties (8 per window)
        avg_props = np.zeros((L, 8), dtype=np.float32)
        var_props = np.zeros((L, 8), dtype=np.float32)
        for i in range(L):
            s, e = max(0, i - half), min(L, i + half + 1)
            window_props = props[s:e]
            avg_props[i] = window_props.mean(axis=0)
            var_props[i] = window_props.var(axis=0)
        features_list.append(avg_props)
        features_list.append(var_props)
        
        # 4c. Windowed entropy + complexity + disorder/order fracs + charge (6 per window)
        extra = np.zeros((L, 6), dtype=np.float32)
        for i in range(L):
            s, e = max(0, i - half), min(L, i + half + 1)
            window_seq = sequence[s:e]
            wlen = len(window_seq)
            
            # Entropy
            counts = Counter(window_seq)
            entropy = 0.0
            for c in counts.values():
                p = c / wlen
                if p > 0:
                    entropy -= p * np.log2(p)
            extra[i, 0] = entropy
            
            # Disorder/order promoting fractions
            dp = sum(1 for c in window_seq if c in _DISORDER_PROMOTING) / wlen
            op = sum(1 for c in window_seq if c in _ORDER_PROMOTING) / wlen
            extra[i, 1] = dp
            extra[i, 2] = op
            
            # Net charge and charge asymmetry
            charges = np.array([_CHARGE[AA_TO_IDX[c]] if c in AA_TO_IDX else 0 for c in window_seq])
            extra[i, 3] = charges.mean()
            extra[i, 4] = np.abs(charges).mean()
            
            # Complexity
            unique = len(set(window_seq))
            extra[i, 5] = unique / min(wlen, 20)
        
        features_list.append(extra)
    
    # 5. Proline/Glycine enrichment (2)
    pg = np.zeros((L, 2), dtype=np.float32)
    for i in range(L):
        s, e = max(0, i - 10), min(L, i + 11)
        window_seq = sequence[s:e]
        wlen = len(window_seq)
        pg[i, 0] = window_seq.count('P') / wlen
        pg[i, 1] = window_seq.count('G') / wlen
    features_list.append(pg)
    
    # 6. Low complexity (1)
    lc = np.zeros((L, 1), dtype=np.float32)
    for i in range(L):
        s, e = max(0, i - 15), min(L, i + 16)
        unique = len(set(sequence[s:e]))
        lc[i, 0] = 1.0 if unique <= 8 else 0.0
    features_list.append(lc)
    
    # 7. Global features (3)
    global_dp = sum(1 for c in sequence if c in _DISORDER_PROMOTING) / L
    global_entropy = 0.0
    gc = Counter(sequence)
    for c in gc.values():
        p = c / L
        if p > 0:
            global_entropy -= p * np.log2(p)
    
    global_feats = np.zeros((L, 3), dtype=np.float32)
    global_feats[:, 0] = global_dp
    global_feats[:, 1] = global_entropy / 4.32
    global_feats[:, 2] = np.log(L) / 10.0
    features_list.append(global_feats)
    
    return np.concatenate(features_list, axis=1)


if __name__ == "__main__":
    seq = "MAEPRQEFEVMEDHAGTYGLGK" * 5
    f = compute_features_fast(seq)
    print(f"Seq len: {len(seq)}, Features: {f.shape}")
