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
    """Every consumer of aligned predictions must drop the sentinels.

    align_fold_predictions returns full-length arrays with -1 labels and NaN
    probs where a label source evaluated nothing. Any consumer that pools those
    residues must mask them first, or sklearn rejects the NaN / treats -1 as a
    third class — always loudly, always after the GPU time is spent.

    This test ENUMERATES the importers rather than listing them. A hand-written
    list is exactly how colab/inference_fusion.py was missed: five files were
    audited, it was the sixth, and it cost another 27-minute run.
    """

    @staticmethod
    def _consumers():
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "colab"
        out = []
        for path in sorted(root.glob("*.py")):
            src = path.read_text()
            if "align_fold_predictions(" not in src:
                continue
            if path.name == "biological_utility.py":     # defines it
                continue
            out.append((path.name, src))
        return out

    def test_consumers_are_discovered(self):
        names = [n for n, _ in self._consumers()]
        assert len(names) >= 5, f"expected several consumers, found {names}"

    def test_every_consumer_consults_the_evidence_mask(self):
        """Fails on any NEW file that aligns predictions and pools them without
        masking — which is the case a fixed list cannot catch."""
        offenders = []
        for name, src in self._consumers():
            uses_evidence = "evidenced" in src or "pool_evidenced" in src
            # Only READS count. `new_item["probs"] = ...` is a substitution,
            # not pooling, and its write-back path masks separately — counting
            # assignments made fold_model_soup a false positive.
            pools = False
            for line in src.splitlines():
                body = line.split("#", 1)[0]
                for pat in ('item["labels"]', 'item["probs"]',
                            '["labels"] for item', '["probs"] for item'):
                    if pat not in body:
                        continue
                    lhs = body.split("=", 1)[0] if "=" in body else ""
                    if pat in lhs and "==" not in body:
                        continue          # assignment target, not a read
                    pools = True
            if pools and not uses_evidence:
                offenders.append(name)
        assert not offenders, (
            f"{offenders} pool aligned residues without consulting the evidence "
            "mask; sentinels would be scored as real labels"
        )

    def test_helpers_are_exported_for_consumers(self):
        from colab.biological_utility import evidenced, pool_evidenced

        assert callable(evidenced) and callable(pool_evidenced)

    def test_sentinels_would_actually_break_sklearn(self):
        """Justifies the guard: this is the failure it prevents."""
        from sklearn.metrics import roc_auc_score

        y = np.array([0, 1, -1, 1], dtype=float)
        s = np.array([0.1, 0.9, np.nan, 0.8])
        with pytest.raises(Exception):
            roc_auc_score(y, s)


class TestMasksAreNotComposedTwice:
    """Two masks over the same array live in different spaces once one is applied.

    Fixing the sentinel leak in af_hallucination introduced exactly that: labels
    were first selected by label evidence (329 residues) and then re-indexed by
    a pLDDT-validity mask still built over the full 340. NumPy caught it, but
    only after 27 minutes of a recovery run:

        IndexError: boolean index did not match indexed array along axis 0;
        size of axis is 329 but size of corresponding boolean axis is 340

    The rule is one combined mask, applied once, to every array.
    """

    def test_two_stage_masking_is_a_length_error(self):
        """The failure mode itself, so the reason for the rule is on record."""
        labels = np.arange(10, dtype=float)
        evidence = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        plddt_valid = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
        kept = labels[evidence]                      # 6 elements
        with pytest.raises(IndexError):
            _ = kept[plddt_valid]                    # 10-element mask

    def test_combined_mask_is_consistent(self):
        labels = np.arange(10, dtype=float)
        evidence = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        plddt_valid = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
        keep = evidence & plddt_valid
        assert len(labels[keep]) == len(np.arange(10.0)[keep]) == int(keep.sum())

    @pytest.mark.parametrize("fn", [
        "run_af_rescue_report",
        "run_labeled_distrust_benchmark",
    ])
    def test_hallucination_sites_build_one_mask(self, fn):
        """Each site must combine evidence with pLDDT validity before indexing,
        never index twice."""
        import inspect

        from colab import af_hallucination

        target = getattr(af_hallucination, fn, None)
        if target is None:
            pytest.skip(f"{fn} not present")
        src = inspect.getsource(target)
        if "evidenced(" not in src:
            pytest.skip("site does not consume aligned items")
        # A combined mask contains both terms on one line.
        assert any(
            "evidenced(item)" in line and ("isnan" in line or "&" in line)
            for line in src.splitlines()
        ), f"{fn} applies evidence separately from pLDDT validity"


class TestTheRealPipelineSurvivesPartialEvidence:
    """Run the actual entry points, not a syntactic scan of them.

    Seven sentinel leaks were found here, one crash at a time, each costing a
    multi-hour GPU run. Six were direct reads of ``item["probs"]`` and a
    file-enumerating audit caught them. The seventh was not: in
    ``apply_plddt_fusion_to_cv`` the NaN was masked only against pLDDT validity,
    then handed to ``find_optimal_fusion_alpha`` as a plain array. Once the
    sentinel crosses a function boundary it stops looking like a sentinel, and
    no amount of grepping for ``item["probs"]`` will see it.

    So drive the real functions with partially-evidenced input. A leak anywhere
    downstream — however many frames deep, however the array was renamed —
    surfaces here in milliseconds instead of after the queue wait.
    """

    @staticmethod
    def _plddt_for(proteins):
        rng = np.random.default_rng(7)
        return {p["id"]: rng.uniform(20.0, 95.0, p["length"]).astype(np.float32)
                for p in proteins}

    def test_plddt_fusion_runs_end_to_end(self):
        from colab.inference_fusion import apply_plddt_fusion_to_cv

        proteins, folds = make_case(0.75, n_proteins=12, length=40)
        report, fused = apply_plddt_fusion_to_cv(
            proteins, folds, self._plddt_for(proteins), n_folds=1,
        )
        assert np.isfinite(report["before"]["pooled"]["auc"])
        assert np.isfinite(report["after"]["pooled"]["auc"])
        assert np.isfinite(report["fusion_alpha"])
        for f in fused:
            assert np.isfinite(np.asarray(f["val_probs"], dtype=float)).all()

    def test_alpha_search_scores_only_evidenced_residues(self):
        """The α grid must be optimised on real labels, not on -1 sentinels."""
        from colab.inference_fusion import apply_plddt_fusion_to_cv

        proteins, folds = make_case(0.5, n_proteins=12, length=40)
        report, _ = apply_plddt_fusion_to_cv(
            proteins, folds, self._plddt_for(proteins), n_folds=1,
        )
        n_ev = sum(int(np.sum(p["label_evidence"])) for p in proteins)
        assert report["before"]["pooled"]["n_residues"] == n_ev

    @pytest.mark.parametrize("frac", [0.3, 0.6, 0.9, 1.0])
    def test_fusion_holds_across_evidence_levels(self, frac):
        from colab.inference_fusion import apply_plddt_fusion_to_cv

        proteins, folds = make_case(frac, n_proteins=12, length=40)
        report, _ = apply_plddt_fusion_to_cv(
            proteins, folds, self._plddt_for(proteins), n_folds=1,
        )
        assert np.isfinite(report["after"]["pooled"]["auc"])


class TestTheMetricBoundaryNamesTheProblem:
    """sklearn's "Input contains NaN" says nothing about which array or caller.

    That message appeared three separate times, each after hours of compute, and
    each time cost a full stack-trace bisect to attribute. The choke point now
    states what leaked and what the caller should have done.
    """

    def test_nan_scores_are_rejected_by_name(self):
        from colab.phase3_synthesis import _safe_auc_ap

        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
        scores = np.array([0.1, 0.9, np.nan, 0.8, 0.2, 0.7], dtype=np.float32)
        with pytest.raises(ValueError, match="unevaluated residues reached a metric"):
            _safe_auc_ap(labels, scores)

    def test_sentinel_labels_are_rejected_by_name(self):
        from colab.phase3_synthesis import _safe_auc_ap

        labels = np.array([0, 1, -1, 1, 0, 1], dtype=np.int8)
        scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2, 0.7], dtype=np.float32)
        with pytest.raises(ValueError, match="unevaluated residues reached a metric"):
            _safe_auc_ap(labels, scores)

    def test_the_message_says_how_to_fix_it(self):
        from colab.phase3_synthesis import _safe_auc_ap

        labels = np.array([0, 1, 0, 1], dtype=np.int8)
        scores = np.array([0.1, np.nan, 0.3, 0.4], dtype=np.float32)
        with pytest.raises(ValueError) as e:
            _safe_auc_ap(labels, scores)
        assert "evidenced(item)" in str(e.value)
        # and must not suggest masking here, which would move the denominator
        assert "inflate" in str(e.value)

    def test_clean_input_is_untouched(self):
        from colab.phase3_synthesis import _safe_auc_ap

        labels = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3], dtype=np.float32)
        auc, ap = _safe_auc_ap(labels, scores)
        assert auc == pytest.approx(1.0)
        assert 0.0 < ap <= 1.0
