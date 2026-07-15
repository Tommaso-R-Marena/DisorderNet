"""Regression tests for out-of-fold prediction alignment.

Reproduces the Quick Screen crash:

    AssertionError: Prediction length mismatch in fold alignment

Root cause: profiles like ``screen_plus``/``ultra`` train with
``split_method="homology"``, but ``align_fold_predictions`` used to re-derive the
CV folds with the default ``"protein"`` method. The reconstructed val folds then
disagreed with the folds actually used for training, so the concatenated
``val_probs`` no longer matched the per-protein lengths.

The fix records ``val_ids`` (the exact validation protein order) on each fold
result and aligns against that, independent of the split method.
"""

from __future__ import annotations

import numpy as np
import pytest

from colab.biological_utility import align_fold_predictions
from colab.cv_splits import get_cv_splits


def _make_proteins(n=9):
    # Distinct lengths so any change in fold membership changes per-fold totals.
    proteins = []
    for i in range(n):
        length = 10 + 3 * i
        labels = [(j % 2) for j in range(length)]
        proteins.append(
            {
                "id": f"DP{i:03d}",
                "sequence": "A" * length,
                "length": length,
                "labels": labels,
            }
        )
    return proteins


def _expected_probs(p):
    # Deterministic, protein-specific probability signature.
    return np.full(p["length"], int(p["id"][2:]) / 100.0, dtype=np.float32)


def _fold_results_from_grouping(proteins, grouping, include_val_ids=True):
    by_id = {p["id"]: p for p in proteins}
    fold_results = []
    for ids in grouping:
        probs = np.concatenate([_expected_probs(by_id[i]) for i in ids])
        labels = np.concatenate(
            [np.asarray(by_id[i]["labels"], dtype=np.float32) for i in ids]
        )
        fold = {"val_probs": probs, "val_labels": labels}
        if include_val_ids:
            fold["val_ids"] = list(ids)
        fold_results.append(fold)
    return fold_results


def _default_grouping(proteins, n_folds):
    return [
        [proteins[i]["id"] for i in val_idx]
        for _, val_idx in get_cv_splits(proteins, n_folds)
    ]


def _homology_like_grouping(proteins, n_folds):
    """A partition that deliberately differs from the default protein split."""
    default = _default_grouping(proteins, n_folds)
    # Move one protein from fold 0 to fold 1 (changes both folds' residue totals).
    moved = [list(g) for g in default]
    moved[1].append(moved[0].pop())
    assert moved != default
    return moved


def test_alignment_uses_val_ids_when_split_differs():
    proteins = _make_proteins()
    grouping = _homology_like_grouping(proteins, n_folds=3)
    fold_results = _fold_results_from_grouping(proteins, grouping, include_val_ids=True)

    aligned = align_fold_predictions(proteins, fold_results, n_folds=3)

    by_id = {a["id"]: a for a in aligned}
    assert set(by_id) == {p["id"] for p in proteins}
    for p in proteins:
        np.testing.assert_allclose(by_id[p["id"]]["probs"], _expected_probs(p))
        np.testing.assert_allclose(
            by_id[p["id"]]["labels"], np.asarray(p["labels"], dtype=np.float32)
        )


def test_alignment_without_val_ids_reproduces_crash():
    """Old behavior: no val_ids + non-default grouping -> length mismatch."""
    proteins = _make_proteins()
    grouping = _homology_like_grouping(proteins, n_folds=3)
    fold_results = _fold_results_from_grouping(proteins, grouping, include_val_ids=False)

    with pytest.raises(AssertionError, match="Prediction length mismatch"):
        align_fold_predictions(proteins, fold_results, n_folds=3)


def test_alignment_fallback_when_grouping_matches_default():
    """Backward compat: old cached folds (no val_ids) still align when the
    grouping matches the default protein split."""
    proteins = _make_proteins()
    grouping = _default_grouping(proteins, n_folds=3)
    fold_results = _fold_results_from_grouping(proteins, grouping, include_val_ids=False)

    aligned = align_fold_predictions(proteins, fold_results, n_folds=3)

    by_id = {a["id"]: a for a in aligned}
    assert set(by_id) == {p["id"] for p in proteins}
    for p in proteins:
        np.testing.assert_allclose(by_id[p["id"]]["probs"], _expected_probs(p))
