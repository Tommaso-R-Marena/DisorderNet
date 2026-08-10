"""Tests for multi-source labels and per-residue evidence.

The invariant that matters: a residue with no label evidence must behave exactly
as if it were absent — never as a negative. PDB-derived disorder only informs
residues some structure covers, so calling never-crystallised residues "ordered"
would train and score the model against a labelling artefact.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from colab.disordernet_gpu import TrainConfig, _disorder_loss
from colab.label_sources import (
    LabelSource,
    build_labelled_set,
    labels_from_record,
    regions_to_mask,
    to_pipeline_proteins,
)


def _record(n=10):
    return {
        "acc": "P00001",
        "length": n,
        "sequence": "A" * n,
        "curated-disorder-priority": {"regions": [[1, 3]]},
        "derived-missing_residues-th_90": {"regions": [[8, 10]]},
        "derived-observed-priority": {"regions": [[4, 7]]},
    }


class TestRegionConversion:
    def test_one_indexed_inclusive_to_zero_indexed_half_open(self):
        m = regions_to_mask([[1, 3], [10, 10]], 12)
        assert m[0] and m[1] and m[2] and not m[3]
        assert m[9] and not m[8] and not m[10]

    def test_clamps_out_of_range(self):
        assert not regions_to_mask([[99, 120]], 5).any()
        assert regions_to_mask([[0, 2]], 5)[:2].all()

    def test_tolerates_inverted_and_malformed(self):
        assert regions_to_mask([[5, 1]], 6)[0:5].all()
        assert not regions_to_mask([["x", None], [], None], 5).any()

    def test_empty_input(self):
        assert not regions_to_mask(None, 4).any()
        assert not regions_to_mask([], 4).any()


class TestLabelSemantics:
    def test_curated_covers_whole_chain(self):
        lp = labels_from_record(_record(), LabelSource.MOBIDB_CURATED)
        assert lp.evidence.all(), "curation annotates the whole chain"
        assert lp.labels[0:3].all() and not lp.labels[3:].any()

    def test_pdb_missing_marks_uncovered_residues_unknown(self):
        lp = labels_from_record(_record(), LabelSource.PDB_MISSING)
        # observed 4-7 and missing 8-10 are evidenced; 1-3 were never crystallised
        assert lp.n_evidenced == 7
        assert not lp.evidence[0:3].any(), (
            "never-crystallised residues must be UNKNOWN, not ordered"
        )
        assert lp.labels[7:10].all(), "missing residues are the positives"
        assert not lp.labels[3:7].any(), "observed residues are the negatives"

    def test_union_takes_positives_from_either_definition(self):
        lp = labels_from_record(_record(), LabelSource.UNION)
        assert lp.labels[0:3].all() and lp.labels[7:10].all()

    def test_returns_none_rather_than_all_zero_protein(self):
        bare = {"acc": "X", "length": 5, "sequence": "AAAAA"}
        assert labels_from_record(bare, LabelSource.PDB_MISSING) is None
        assert labels_from_record(bare, LabelSource.MOBIDB_CURATED) is None

    def test_rejects_prediction_sources_as_labels(self):
        """Predictor outputs are not ground truth; training on them is distillation."""
        with pytest.raises(ValueError):
            labels_from_record(_record(), LabelSource.DISPROT)


class TestSetBuilding:
    def test_reports_skip_reasons_and_evidence_stats(self):
        kept, stats = build_labelled_set(
            [_record()], LabelSource.PDB_MISSING, min_len=5, min_evidence_fraction=0.1
        )
        assert stats["n_proteins"] == 1
        assert stats["n_evidenced_residues"] == 7
        assert stats["evidence_fraction"] == pytest.approx(0.7)
        assert stats["disorder_fraction_of_evidenced"] == pytest.approx(3 / 7, abs=1e-3)

    def test_drops_proteins_below_evidence_floor(self):
        rec = _record(200)
        rec["derived-observed-priority"] = {"regions": [[1, 4]]}
        rec["derived-missing_residues-th_90"] = {"regions": [[5, 6]]}
        kept, stats = build_labelled_set(
            [rec], LabelSource.PDB_MISSING, min_len=5, min_evidence_fraction=0.5
        )
        assert kept == []
        assert stats["skipped"].get("no_evidence_for_source") == 1

    def test_output_order_is_deterministic(self):
        recs = [dict(_record(), acc=a) for a in ("P3", "P1", "P2")]
        kept, _ = build_labelled_set(recs, LabelSource.PDB_MISSING, min_len=5)
        assert [p.id for p in kept] == ["P1", "P2", "P3"]

    def test_pipeline_adapter_shape(self):
        kept, _ = build_labelled_set([_record()], LabelSource.PDB_MISSING, min_len=5)
        d = to_pipeline_proteins(kept)[0]
        assert d["length"] == len(d["labels"]) == len(d["label_evidence"]) == 10
        assert d["n_dis"] == 3
        assert d["label_source"] == "pdb_missing"


class TestEvidenceReachesTheLoss:
    """A zero-weight residue must be indistinguishable from an absent one."""

    @staticmethod
    def _cfg():
        return TrainConfig.from_profile("ultra")

    def test_masked_residues_behave_as_absent(self):
        cfg = self._cfg()
        logits = torch.tensor([[10.0, -10.0, 10.0, -10.0]])
        labels = torch.tensor([[1.0, 0.0, 0.0, 1.0]])  # residues 2,3 mislabelled
        full = torch.tensor([[True] * 4])

        masked = _disorder_loss(
            logits, labels, None, torch.tensor([[1.0, 1.0, 0.0, 0.0]]), cfg, mask=full
        )
        absent = _disorder_loss(
            logits[:, :2], labels[:, :2], None,
            torch.tensor([[1.0, 1.0]]), cfg, mask=torch.tensor([[True, True]]),
        )
        assert masked.item() == pytest.approx(absent.item(), abs=1e-5)

    def test_treating_unknown_as_ordered_is_catastrophically_different(self):
        """Guards against a future change that drops the evidence weighting."""
        cfg = self._cfg()
        logits = torch.tensor([[10.0, -10.0, 10.0, -10.0]])
        labels = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        full = torch.tensor([[True] * 4])

        masked = _disorder_loss(
            logits, labels, None, torch.tensor([[1.0, 1.0, 0.0, 0.0]]), cfg, mask=full
        )
        as_ordered = _disorder_loss(
            logits, labels, None, torch.tensor([[1.0] * 4]), cfg, mask=full
        )
        assert as_ordered.item() > 100 * masked.item()

    def test_uniform_reweighting_is_scale_invariant(self):
        """Weighted mean, not mean-of-weighted: scaling all weights must not
        rescale the loss (and hence the effective learning rate)."""
        cfg = self._cfg()
        logits = torch.tensor([[2.0, -1.0, 0.5, -3.0]])
        labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        full = torch.tensor([[True] * 4])
        a = _disorder_loss(logits, labels, None, torch.tensor([[1.0] * 4]), cfg, mask=full)
        b = _disorder_loss(logits, labels, None, torch.tensor([[7.0] * 4]), cfg, mask=full)
        assert a.item() == pytest.approx(b.item(), rel=1e-5)

    def test_dice_and_tversky_also_respect_evidence(self):
        """Overlap losses take a mask, not weights, so they need explicit handling."""
        cfg = self._cfg()
        assert cfg.use_dice_loss and cfg.use_tversky_loss, "fixture assumes both on"
        logits = torch.tensor([[8.0, -8.0, 8.0, 8.0]])
        labels = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # residues 2,3 wrong
        full = torch.tensor([[True] * 4])

        masked = _disorder_loss(
            logits, labels, None, torch.tensor([[1.0, 1.0, 0.0, 0.0]]), cfg, mask=full
        )
        absent = _disorder_loss(
            logits[:, :2], labels[:, :2], None,
            torch.tensor([[1.0, 1.0]]), cfg, mask=torch.tensor([[True, True]]),
        )
        # If dice/tversky ignored evidence, the two wrong residues would still
        # contribute overlap error and these would diverge.
        assert masked.item() == pytest.approx(absent.item(), abs=1e-5)


class TestEvidenceReachesTheDataset:
    def test_dataset_zeroes_weight_for_unknown_residues(self):
        from colab.label_sources import LabelledProtein

        lp = LabelledProtein(
            id="P1",
            sequence="A" * 10,
            labels=np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8),
            evidence=np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool),
            source="pdb_missing",
        )
        d = to_pipeline_proteins([lp])[0]
        assert d["label_evidence"][:3] == [False, False, False]
        assert all(d["label_evidence"][3:])


class TestCaidFilterAppliesToEveryLabelSource:
    """The CAID leak-free filter must not be reachable from only one code path.

    It originally lived inline in the DisProt branch of _load_proteins, and the
    MobiDB branch returned before reaching it. The pdb_missing arm therefore
    trained unfiltered on 19,819 proteins — while being scored on CAID3, whose
    Disorder-PDB references share the same missing-residue definition and have
    PDB structures by construction. That would have measured memorisation.
    """

    def test_both_label_paths_call_the_shared_filter(self):
        import inspect

        from rockfish import run_disordernet as rd

        disprot_src = inspect.getsource(rd._load_proteins)
        mobidb_src = inspect.getsource(rd._load_proteins_mobidb)
        for name, src in (("disprot", disprot_src), ("mobidb", mobidb_src)):
            assert "_apply_caid_leak_free_filter" in src, (
                f"the {name} label path does not apply the CAID leak-free filter"
            )

    def test_filter_is_a_noop_when_disabled(self):
        from rockfish.run_disordernet import _apply_caid_leak_free_filter

        class _Args:
            caid_leak_free_train = False

        class _Cfg:
            checkpoint_dir = "/nonexistent"

        proteins = [{"id": "P1", "sequence": "AAAA", "length": 4}]
        out, meta = _apply_caid_leak_free_filter(proteins, {"x": 1}, _Cfg(), _Args())
        assert out is proteins and meta == {"x": 1}

    def test_filter_removes_flagged_proteins_and_records_the_audit(self, tmp_path, monkeypatch):
        import rockfish.run_disordernet as rd

        class _Args:
            caid_leak_free_train = True
            caid3_reference = None
            caid_leak_identity = 0.40
            label_source = "pdb_missing"

        class _Cfg:
            checkpoint_dir = str(tmp_path)

        proteins = [
            {"id": "KEEP", "sequence": "A" * 40, "length": 40},
            {"id": "LEAK", "sequence": "C" * 40, "length": 40},
        ]
        monkeypatch.setattr(
            "colab.caid_challenge.resolve_caid3_references", lambda *a, **k: {"r": "x.fasta"}
        )
        monkeypatch.setattr(
            "colab.caid_leakage.load_caid_refs_for_audit",
            lambda paths: [{"id": "LEAK", "sequence": "C" * 40}],
        )

        out, meta = rd._apply_caid_leak_free_filter(proteins, {}, _Cfg(), _Args())
        kept = {p["id"] for p in out}
        assert "LEAK" not in kept, "a CAID-identical training protein must be dropped"
        assert "KEEP" in kept
        assert meta["caid_leak_free_train"]["n_removed"] == 1
        assert (tmp_path / "caid_leakage_audit.json").exists()

    def test_audit_records_which_label_source_it_filtered(self, tmp_path, monkeypatch):
        import json

        import rockfish.run_disordernet as rd

        class _Args:
            caid_leak_free_train = True
            caid3_reference = None
            caid_leak_identity = 0.40
            label_source = "pdb_missing"

        class _Cfg:
            checkpoint_dir = str(tmp_path)

        monkeypatch.setattr(
            "colab.caid_challenge.resolve_caid3_references", lambda *a, **k: {"r": "x"}
        )
        monkeypatch.setattr(
            "colab.caid_leakage.load_caid_refs_for_audit",
            lambda paths: [{"id": "Z", "sequence": "D" * 40}],
        )
        rd._apply_caid_leak_free_filter(
            [{"id": "A", "sequence": "A" * 40, "length": 40}], {}, _Cfg(), _Args()
        )
        audit = json.loads((tmp_path / "caid_leakage_audit.json").read_text())
        assert audit["label_source"] == "pdb_missing"
