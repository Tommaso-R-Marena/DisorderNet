"""CAID3 labels and predictions must refer to the same residue.

`parse_caid_reference_fasta` appended to `labels` only at unmasked positions
while `eval_mask` spanned the sequence, so the two arrays indexed different
spaces. `evaluate_caid_predictions` then computed labels[:n][mask[:n]] with
n = min(len(probs), len(labels), len(mask)) — predictions indexed by sequence
position, labels by labelled-position index. They agree only when every '-' is
trailing.

Measured on the real CAID3 Disorder-PDB reference before the fix:
  - 239 of 319 targets had len(labels) != len(eval_mask)
  - 127 of the 200 scored targets were misaligned
  - 69.8% of scored residues were compared against another residue's prediction
  - correcting it moved pooled AUC from 0.8733 to 0.9042 (+0.031)
"""

from __future__ import annotations

import numpy as np
import pytest

from colab.caid3_eval import evaluate_caid_predictions, parse_caid_reference_fasta


def write_fasta(tmp_path, entries):
    p = tmp_path / "ref.fasta"
    p.write_text("".join(f">{i}\n{s}\n{lab}\n" for i, s, lab in entries))
    return str(p)


class TestParserAlignment:
    def test_labels_and_mask_span_the_sequence(self, tmp_path):
        """The invariant the whole evaluation rests on."""
        path = write_fasta(tmp_path, [("T1", "AAAACCCCGGGG", "--11--0000--")])
        p = parse_caid_reference_fasta(path)[0]
        assert len(p["labels"]) == len(p["eval_mask"]) == len(p["sequence"])

    def test_mask_marks_exactly_the_dashes(self, tmp_path):
        path = write_fasta(tmp_path, [("T1", "AAAACCCC", "-1-0-1-0")])
        p = parse_caid_reference_fasta(path)[0]
        assert p["eval_mask"] == [False, True, False, True, False, True, False, True]

    def test_label_values_sit_at_their_own_positions(self, tmp_path):
        """Position 3 must carry the label written at position 3."""
        path = write_fasta(tmp_path, [("T1", "AAAACCCC", "--11--00")])
        p = parse_caid_reference_fasta(path)[0]
        lab = np.asarray(p["labels"])
        assert lab[2] == 1 and lab[3] == 1
        assert lab[6] == 0 and lab[7] == 0

    def test_interior_dashes_do_not_shorten_labels(self, tmp_path):
        """The defect in one assertion: interior '-' used to shift every
        subsequent label one slot left relative to its prediction."""
        path = write_fasta(tmp_path, [("T1", "A" * 20, "1" * 5 + "-" * 10 + "0" * 5)])
        p = parse_caid_reference_fasta(path)[0]
        assert len(p["labels"]) == 20
        assert sum(p["eval_mask"]) == 10


class TestEvaluationAlignment:
    def _ref_and_perfect_preds(self, tmp_path, entries):
        path = write_fasta(tmp_path, entries)
        ref = parse_caid_reference_fasta(path)
        preds = {
            p["id"]: np.where(np.asarray(p["labels"]) == 1, 0.99, 0.01).astype(np.float32)
            for p in ref
        }
        return ref, preds

    def test_a_perfect_predictor_scores_one(self, tmp_path):
        """Impossible under the old indexing whenever a '-' was interior."""
        ref, preds = self._ref_and_perfect_preds(tmp_path, [
            ("T1", "AAAACCCCGGGG", "--11--0000--"),
            ("T2", "AAAACCCC", "11110000"),
        ])
        rep = evaluate_caid_predictions(ref, preds)
        assert rep["pooled"]["auc"] == pytest.approx(1.0, abs=1e-9)

    def test_masked_positions_are_ignored(self, tmp_path):
        """Garbage at '-' positions must not move any metric."""
        ref, preds = self._ref_and_perfect_preds(tmp_path, [
            ("T1", "AAAACCCCGGGG", "--11--0000--"),
            ("T2", "AAAACCCC", "11110000"),
        ])
        clean = evaluate_caid_predictions(ref, preds)["pooled"]["auc"]
        for p in ref:
            preds[p["id"]] = np.where(
                np.asarray(p["eval_mask"]), preds[p["id"]], 0.5
            ).astype(np.float32)
        assert evaluate_caid_predictions(ref, preds)["pooled"]["auc"] == clean

    def test_an_inverted_predictor_scores_zero(self, tmp_path):
        """Guards against a fix that accidentally scores the mask itself."""
        ref, preds = self._ref_and_perfect_preds(tmp_path, [
            ("T1", "AAAACCCCGGGG", "--11--0000--"),
            ("T2", "AAAACCCC", "11110000"),
        ])
        for k in preds:
            preds[k] = 1.0 - preds[k]
        assert evaluate_caid_predictions(ref, preds)["pooled"]["auc"] == pytest.approx(
            0.0, abs=1e-9
        )


class TestProtocolIsRecorded:
    def _mixed(self, tmp_path):
        # T3 is entirely disordered: no per-target AUC, but real pooled evidence.
        path = write_fasta(tmp_path, [
            ("T1", "AAAACCCC", "11110000"),
            ("T2", "AAAACCCC", "11000011"),
            ("T3", "AAAACCCC", "11111111"),
        ])
        ref = parse_caid_reference_fasta(path)
        preds = {
            p["id"]: np.where(np.asarray(p["labels"]) == 1, 0.9, 0.1).astype(np.float32)
            for p in ref
        }
        return evaluate_caid_predictions(ref, preds)

    def test_single_class_targets_are_pooled_but_not_per_target_scored(self, tmp_path):
        rep = self._mixed(tmp_path)
        assert rep["n_scored"] == 3
        assert rep["n_scored_two_class"] == 2

    def test_both_protocols_are_reported(self, tmp_path):
        """Which residue set a quoted AUC came from must be recoverable."""
        rep = self._mixed(tmp_path)
        assert rep["pooled"]["auc"] is not None
        assert rep["pooled_two_class_only"]["auc"] is not None

    def test_sota_comparison_uses_the_stricter_protocol(self, tmp_path):
        """Pooling single-class targets adds mostly fully-disordered ones, which
        makes the metric easier rather than the model better."""
        rep = self._mixed(tmp_path)
        assert rep["sota_comparison_protocol"] == "two_class_targets_only"

    def test_reaches_and_exceeds_are_distinct_claims(self, tmp_path):
        rep = self._mixed(tmp_path)
        assert "ci_reaches_esmdispred" in rep
        assert "ci_exceeds_esmdispred" in rep

    def test_comparison_note_flags_the_unknown_protocol(self, tmp_path):
        """The error this project already had to correct once."""
        note = self._mixed(tmp_path)["comparison_note"]
        assert "protocol is not known" in note
