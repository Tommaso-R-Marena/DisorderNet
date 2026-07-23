"""Tests for the unified embedding extractor's pure helpers."""
from __future__ import annotations

import pytest

from extract_embeddings import BACKBONE_DIM, repr_layer_for, _iter_length_batches


@pytest.mark.parametrize("name,layer", [
    ("esm2_t6_8M_UR50D", 6),
    ("esm2_t12_35M_UR50D", 12),
    ("esm2_t30_150M_UR50D", 30),
    ("esm2_t33_650M_UR50D", 33),
    ("esm2_t36_3B_UR50D", 36),
])
def test_repr_layer_parsing(name, layer):
    assert repr_layer_for(name) == layer


def test_repr_layer_matches_registry_keys():
    for name in BACKBONE_DIM:
        assert repr_layer_for(name) >= 6


def test_repr_layer_bad_name():
    with pytest.raises(ValueError):
        repr_layer_for("not-a-model")


def test_length_batches_respect_token_budget_and_cover_all():
    items = [(f"P{i}", "A" * (10 + i)) for i in range(20)]
    batches = list(_iter_length_batches(items, batch_tokens=60))
    seen = [idx for b in batches for (idx, _, _) in b]
    assert sorted(seen) == list(range(20))  # every item covered exactly once
    for b in batches:
        toks = sum(len(seq) + 2 for (_, _, seq) in b)
        # a batch may exceed budget only when it is a single (long) item
        assert toks <= 60 or len(b) == 1


def test_length_batches_sorted_by_length():
    items = [("A", "AAAAA"), ("B", "A"), ("C", "AAA")]
    first = list(_iter_length_batches(items, batch_tokens=1000))[0]
    lengths = [len(seq) for (_, _, seq) in first]
    assert lengths == sorted(lengths)
