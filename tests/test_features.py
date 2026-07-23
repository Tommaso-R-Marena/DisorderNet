"""Tests for the physicochemical feature module (features.py)."""
from __future__ import annotations

import math

import numpy as np

from features import (
    AMINO_ACIDS,
    compute_features_for_protein,
    get_feature_names,
    get_residue_properties,
    sequence_complexity,
    shannon_entropy,
)


def test_residue_properties_length_and_unknown():
    props = get_residue_properties("A")
    assert len(props) == 8
    # unknown residue -> zeros (scale.get default 0.0)
    assert get_residue_properties("X") == [0.0] * 8


def test_shannon_entropy_bounds():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("AAAA") == 0.0  # single symbol -> 0 bits
    # two equally frequent symbols -> 1 bit
    assert abs(shannon_entropy("ABAB") - 1.0) < 1e-9


def test_sequence_complexity_range():
    assert sequence_complexity("A") == 0.0
    low = sequence_complexity("AAAAAAAA")
    high = sequence_complexity("ACDEFGHI")
    assert 0.0 <= low < high


def test_feature_matrix_shape_matches_names():
    seq = "MAEPRQEFEVMEDHAGTYGKLPQRS"
    feats = compute_features_for_protein(seq)
    names = get_feature_names()
    assert feats.shape == (len(seq), len(names))
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_onehot_block_is_correct():
    seq = "ACD"
    feats = compute_features_for_protein(seq)
    # first 20 features are the one-hot encoding
    for i, aa in enumerate(seq):
        onehot = feats[i, :20]
        assert onehot.sum() == 1.0
        assert onehot[AMINO_ACIDS.index(aa)] == 1.0


def test_feature_names_unique_and_counted():
    names = get_feature_names()
    assert len(names) == len(set(names))
    assert names[0].startswith("onehot_")
    assert "log_length" in names
