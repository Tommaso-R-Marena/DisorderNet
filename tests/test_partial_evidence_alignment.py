"""Fold alignment must survive partial label evidence.

`val_probs` holds one value per *evidenced* residue — that is what eval_epoch
scores. `align_fold_predictions` sliced it by `p["length"]`, i.e. one value per
residue, which is correct only when every residue is labelled. DisProt is ~100%
evidenced so it never showed; PDB-derived labels are 75-82% evidenced, and a
14-GPU-hour pdb_missing run died at the stack stage, after cross-validation had
already finished, with:

    aligned 1606012 residues across 3885 proteins vs 1319757 predicted

Aligned items are now full-length with NaN probs / -1 labels at unlabelled
positions, because position-indexed consumers (pLDDT fusion, bedgraph export)
need sequence indexing. Anything that pools residues must drop the sentinels.
"""

from __future__ import annotations

import numpy as np
import pytest

from colab.biological_utility import (
    align_fold_predictions,
    evidenced,
    pool_evidenced,
)


def make_case(evidence_fraction: float, n_proteins: int = 6, length: int = 20):
    """Proteins with a known evidence pattern, and fold_results built the way
    eval_epoch builds them: evidenced residues only, in protein order."""
    rng = np.random.default_rng(0)
    proteins, probs, labels, ids = [], [], [], []
    for i in range(n_proteins):
        ev = np.zeros(length, dtype=bool)
        n_ev = max(1, int(length * evidence_fraction))
        ev[rng.choice(length, n_ev, replace=False)] = True
        lab = rng.integers(0, 2, length)
        proteins.append({
            "id": f"P{i}", "length": length,
            "sequence": "A" * length,
            "labels": lab.tolist(),
            "label_evidence": ev.tolist(),
        })
        ids.append(f"P{i}")
        probs.append(rng.random(n_ev).astype(np.float32))
        labels.append(lab[ev].astype(np.float32))
    fold = {
        "fold": 1, "val_ids": ids,
        "val_probs": np.concatenate(probs),
        "val_labels": np.concatenate(labels),
    }
    return proteins, [fold]


class TestAlignmentUnderPartialEvidence:
    def test_partial_evidence_no_longer_raises(self):
        """The exact failure: 80% evidence used to trip the length assertion."""
        proteins, folds = make_case(0.8)
        aligned = align_fold_predictions(proteins, folds, n_folds=1)
        assert len(aligned) == len(proteins)

    def test_full_evidence_still_works(self):
        proteins, folds = make_case(1.0)
        aligned = align_fold_predictions(proteins, folds, n_folds=1)
        assert len(aligned) == len(proteins)

    def test_items_are_full_length_for_position_indexed_consumers(self):
        """pLDDT fusion indexes probs by residue; the array must span the
        sequence even where labels are absent."""
        proteins, folds = make_case(0.6)
        for item in align_fold_predictions(proteins, folds, n_folds=1):
            assert len(item["probs"]) == item["protein"]["length"]
            assert len(item["labels"]) == item["protein"]["length"]

    def test_unlabelled_positions_carry_droppable_sentinels(self):
        proteins, folds = make_case(0.6)
        for item in align_fold_predictions(proteins, folds, n_folds=1):
            m = evidenced(item)
            assert np.all(np.isnan(np.asarray(item["probs"])[~m]))
            assert np.all(np.asarray(item["labels"])[~m] < 0)

    def test_predictions_land_on_the_residues_they_were_made_for(self):
        """The values must be scattered back to their own positions, not
        packed into a prefix."""
        proteins, folds = make_case(0.5)
        aligned = align_fold_predictions(proteins, folds, n_folds=1)
        offset = 0
        for item, p in zip(aligned, proteins):
            ev = np.asarray(p["label_evidence"], dtype=bool)
            n = int(ev.sum())
            expected = np.asarray(folds[0]["val_probs"])[offset:offset + n]
            assert np.allclose(np.asarray(item["probs"])[ev], expected)
            offset += n

    def test_labels_match_the_source_labels_at_evidenced_positions(self):
        proteins, folds = make_case(0.7)
        for item, p in zip(align_fold_predictions(proteins, folds, n_folds=1), proteins):
            ev = np.asarray(p["label_evidence"], dtype=bool)
            src = np.asarray(p["labels"], dtype=np.float32)
            assert np.array_equal(np.asarray(item["labels"])[ev], src[ev])


class TestPooling:
    def test_pool_drops_every_sentinel(self):
        proteins, folds = make_case(0.6)
        labels, probs = pool_evidenced(align_fold_predictions(proteins, folds, n_folds=1))
        assert not np.isnan(probs).any()
        assert (labels >= 0).all()

    def test_pooled_count_equals_the_evidenced_count(self):
        proteins, folds = make_case(0.6)
        aligned = align_fold_predictions(proteins, folds, n_folds=1)
        labels, _ = pool_evidenced(aligned)
        assert len(labels) == len(folds[0]["val_probs"])

    def test_pooling_is_metric_safe(self):
        """sklearn would reject NaN scores or a -1 class; this is what keeps a
        partial-evidence run from failing at the metric instead of the mask."""
        from sklearn.metrics import roc_auc_score

        proteins, folds = make_case(0.6)
        labels, probs = pool_evidenced(align_fold_predictions(proteins, folds, n_folds=1))
        roc_auc_score(labels, probs)


class TestFusionRoundTrip:
    def test_written_back_val_probs_keep_the_evidenced_contract(self):
        """val_probs must stay one value per evidenced residue, or it desyncs
        from val_labels the moment evidence is partial."""
        from colab.inference_fusion import write_fused_probs_to_fold_results

        proteins, folds = make_case(0.7)
        aligned = align_fold_predictions(proteins, folds, n_folds=1)
        updated = write_fused_probs_to_fold_results(proteins, folds, aligned, n_folds=1)
        assert len(updated[0]["val_probs"]) == len(folds[0]["val_probs"])
        assert len(updated[0]["val_probs"]) == len(folds[0]["val_labels"])

    def test_round_trip_preserves_values(self):
        from colab.inference_fusion import write_fused_probs_to_fold_results

        proteins, folds = make_case(0.7)
        aligned = align_fold_predictions(proteins, folds, n_folds=1)
        updated = write_fused_probs_to_fold_results(proteins, folds, aligned, n_folds=1)
        assert np.allclose(updated[0]["val_probs"], folds[0]["val_probs"], equal_nan=False)


@pytest.mark.parametrize("frac", [0.3, 0.5, 0.75, 0.82, 1.0])
def test_alignment_holds_across_evidence_levels(frac):
    proteins, folds = make_case(frac)
    aligned = align_fold_predictions(proteins, folds, n_folds=1)
    labels, probs = pool_evidenced(aligned)
    assert len(labels) == len(folds[0]["val_probs"])


class TestNoConsumerLeaksTheSentinel:
    """Find sentinel leaks by inspection, not by crashing one stage at a time.

    align_fold_predictions returns full-length arrays with -1 labels and NaN
    probs where a label source evaluated nothing. Every consumer that *pools*
    residues must drop those first. Each one I missed cost a multi-hour run:

      ensemble_v6 / meta_ensemble     found by reading the contract
      caid_reporting pooled call      found by reading the contract
      caid_reporting strata           crashed at 1h12 with
                                      "Target is multiclass but average='binary'"
      sota_ensemble, af_hallucination found by this test

    A -1 in y_true makes sklearn treat the problem as multiclass; a NaN in
    y_score raises. Both are loud, but only after the GPU time is spent.
    """

    CONSUMERS = [
        ("colab/ensemble_v6.py", 'labels_concat.append'),
        ("colab/meta_ensemble.py", 'chunks_y.append'),
        ("colab/sota_ensemble.py", 'label_chunks.append'),
        ("colab/caid_reporting.py", 'by_disorder['),
        ("colab/af_hallucination.py", 'labels_list.append'),
    ]

    def _source(self, rel):
        from pathlib import Path
        return (Path(__file__).resolve().parents[1] / rel).read_text()

    @pytest.mark.parametrize("path,marker", CONSUMERS)
    def test_pooling_sites_reference_the_evidence_mask(self, path, marker):
        """Every file that pools aligned residues must consult `evidenced`."""
        src = self._source(path)
        assert marker in src, f"{path}: pooling site {marker!r} moved; re-audit"
        assert "evidenced" in src or "pool_evidenced" in src, (
            f"{path} pools aligned residues without consulting the evidence "
            "mask — sentinels would be scored as real labels"
        )

    def test_helpers_are_exported_for_consumers(self):
        from colab.biological_utility import evidenced, pool_evidenced

        assert callable(evidenced) and callable(pool_evidenced)

    def test_sentinels_would_actually_break_sklearn(self):
        """Justifies the whole guard: this is the failure it prevents."""
        from sklearn.metrics import roc_auc_score

        y = np.array([0, 1, -1, 1], dtype=float)
        s = np.array([0.1, 0.9, np.nan, 0.8])
        with pytest.raises(Exception):
            roc_auc_score(y, s)
