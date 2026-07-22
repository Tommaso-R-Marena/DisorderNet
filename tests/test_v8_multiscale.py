"""Tests for the multi-scale ensemble helpers (run_v8_multiscale.py)."""
from __future__ import annotations

import numpy as np

from run_v8_multiscale import residue_fold_labels, evaluate


def _proteins(lengths):
    return [{"disprot_id": f"P{i}", "length": L} for i, L in enumerate(lengths)]


def test_residue_fold_labels_partition():
    proteins = _proteins([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    fold = residue_fold_labels(proteins)
    assert len(fold) == sum(p["length"] for p in proteins)
    assert set(np.unique(fold)) == {0, 1, 2, 3, 4}
    # each protein's residues belong to exactly one fold
    off = 0
    for p in proteins:
        block = fold[off:off + p["length"]]
        assert len(set(block.tolist())) == 1
        off += p["length"]


def test_residue_fold_labels_deterministic():
    proteins = _proteins([15, 25, 35, 45, 55, 65])
    assert np.array_equal(residue_fold_labels(proteins), residue_fold_labels(proteins))


def test_equal_weight_ensemble_beats_members():
    """Averaging two conditionally-independent noisy scorers improves AUC."""
    rng = np.random.RandomState(0)
    n = 20000
    y = rng.randint(0, 2, n)
    signal = y * 1.2
    p1 = 1 / (1 + np.exp(-(signal + rng.randn(n))))
    p2 = 1 / (1 + np.exp(-(signal + rng.randn(n))))
    ens = (p1 + p2) / 2
    a1, a2 = evaluate(y, p1)["auc_roc"], evaluate(y, p2)["auc_roc"]
    ae = evaluate(y, ens)["auc_roc"]
    assert ae > max(a1, a2)


def test_evaluate_keys():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate(y, p)
    assert set(m) == {"auc_roc", "avg_precision", "f1", "mcc", "balanced_acc"}
    assert m["auc_roc"] == 1.0
