"""Tests for colab/cv_splits.py."""

from __future__ import annotations

import numpy as np

from colab.cv_splits import (
    config_fingerprint,
    get_cv_splits,
    get_fold_val_protein_ids,
    proteins_fingerprint,
    sort_proteins_deterministic,
)
from colab.disordernet_gpu import TrainConfig


def _make_proteins(n: int = 6) -> list:
    return [
        {
            "id": f"DP{i:04d}",
            "sequence": "A" * 30 + "C" * 10,
            "length": 40,
            "labels": [0] * 30 + [1] * 10,
        }
        for i in range(n)
    ]


class TestCvSplits:
    def test_sort_is_deterministic(self):
        proteins = _make_proteins(5)
        shuffled = list(reversed(proteins))
        assert [p["id"] for p in sort_proteins_deterministic(shuffled)] == [
            p["id"] for p in proteins
        ]

    def test_splits_stable_after_sort(self):
        proteins = sort_proteins_deterministic(_make_proteins(10))
        shuffled = sort_proteins_deterministic(list(reversed(proteins)))
        splits_a = get_cv_splits(proteins, n_folds=5)
        splits_b = get_cv_splits(shuffled, n_folds=5)
        for (_, va), (_, vb) in zip(splits_a, splits_b):
            assert list(va) == list(vb)

    def test_fold_val_ids_cover_all_proteins_once(self):
        proteins = sort_proteins_deterministic(_make_proteins(8))
        fold_ids = get_fold_val_protein_ids(proteins, n_folds=4)
        flat = [pid for fold in fold_ids for pid in fold]
        assert sorted(flat) == sorted(p["id"] for p in proteins)
        assert len(flat) == len(proteins)

    def test_config_fingerprint_changes_with_hyperparams(self):
        cfg_a = TrainConfig(seed=42, lora_rank=16)
        cfg_b = TrainConfig(seed=42, lora_rank=32)
        assert config_fingerprint(cfg_a) != config_fingerprint(cfg_b)

    def test_proteins_fingerprint_order_sensitive(self):
        a = sort_proteins_deterministic(_make_proteins(3))
        b = list(reversed(a))
        assert proteins_fingerprint(a) != proteins_fingerprint(b)
