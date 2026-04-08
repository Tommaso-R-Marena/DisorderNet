"""
Feature engineering for protein disorder prediction.
Combines physicochemical properties, sequence composition, complexity,
and multi-scale contextual features.
"""
import numpy as np
from collections import Counter

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


def compute_features_for_protein(sequence, windows=[5, 11, 21, 41]):
    """
    Compute feature matrix for a protein sequence.
    
    Returns: numpy array of shape (seq_len, num_features)
    """
    seq_len = len(sequence)
    all_features = []
    
    for i in range(seq_len):
        aa = sequence[i]
        feat = []
        
        # 1. One-hot encoding (20 features)
        onehot = [0.0] * 20
        if aa in AA_TO_IDX:
            onehot[AA_TO_IDX[aa]] = 1.0
        feat.extend(onehot)
        
        # 2. Physicochemical properties (8 features)
        feat.extend(get_residue_properties(aa))
        
        # 3. Relative position features (2 features)
        rel_pos = i / max(seq_len - 1, 1)
        dist_to_terminus = min(i, seq_len - 1 - i) / max(seq_len - 1, 1)
        feat.extend([rel_pos, dist_to_terminus])
        
        # 4. Multi-scale windowed features
        for w in windows:
            half_w = w // 2
            start = max(0, i - half_w)
            end = min(seq_len, i + half_w + 1)
            window_seq = sequence[start:end]
            
            # 4a. Amino acid composition in window (20 features per window)
            aa_counts = Counter(window_seq)
            window_len = len(window_seq)
            composition = [aa_counts.get(aa, 0) / window_len for aa in AMINO_ACIDS]
            feat.extend(composition)
            
            # 4b. Average properties in window (8 features per window)
            for scale_name, scale in ALL_SCALES.items():
                avg_val = np.mean([scale.get(c, 0.0) for c in window_seq])
                feat.append(avg_val)
            
            # 4c. Property variances in window (8 features per window)
            for scale_name, scale in ALL_SCALES.items():
                var_val = np.var([scale.get(c, 0.0) for c in window_seq])
                feat.append(var_val)
            
            # 4d. Shannon entropy and complexity (2 features per window)
            feat.append(shannon_entropy(window_seq))
            feat.append(sequence_complexity(window_seq))
            
            # 4e. Disorder-promoting residue fraction (1 feature per window)
            disorder_promoting = set("AEGKPQRS")
            order_promoting = set("CFILMVWY")
            dp_frac = sum(1 for c in window_seq if c in disorder_promoting) / window_len
            op_frac = sum(1 for c in window_seq if c in order_promoting) / window_len
            feat.extend([dp_frac, op_frac])
            
            # 4f. Charge pattern features (2 features per window)
            charges = [CHARGE.get(c, 0) for c in window_seq]
            net_charge = sum(charges) / window_len
            charge_asym = sum(abs(c) for c in charges) / window_len
            feat.extend([net_charge, charge_asym])
        
        # 5. Proline and glycine enrichment (2 features)
        # These are strong disorder indicators
        local_w = 21
        half = local_w // 2
        s, e = max(0, i - half), min(seq_len, i + half + 1)
        local = sequence[s:e]
        feat.append(local.count('P') / len(local))
        feat.append(local.count('G') / len(local))
        
        # 6. Low complexity indicator (1 feature)
        # Is this residue in a low-complexity region?
        lc_window = sequence[max(0, i-25):min(seq_len, i+26)]
        unique_aa = len(set(lc_window))
        feat.append(1.0 if unique_aa <= 8 else 0.0)
        
        # 7. Protein-level global features (3 features)
        global_disorder_frac = sum(1 for c in sequence if c in disorder_promoting) / seq_len
        global_entropy = shannon_entropy(sequence)
        log_length = np.log(seq_len)
        feat.extend([global_disorder_frac, global_entropy / 4.32, log_length / 10.0])
        
        all_features.append(feat)
    
    return np.array(all_features, dtype=np.float32)


def get_feature_names(windows=[5, 11, 21, 41]):
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
