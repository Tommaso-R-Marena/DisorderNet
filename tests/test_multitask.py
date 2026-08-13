"""One model, five CAID3 benchmarks.

CAID3 is won by five different specialists — PUNCH2 for Disorder-PDB,
ESMDisPred for Disorder-NOX, IPA-AF2-Linker for Linker, DisoFLAG-PB for
Binding, bindEmbed21IDR for Binding-IDR. None of them answers another's
question. A single model that answers all five from one forward pass is only
affordable with a frozen backbone, and only *safe* with a low-capacity head,
because the small tasks are very small:

    disorder   336,014 positive residues
    binding     88,761
    linker      15,683

These tests pin the properties that make the shared trunk defensible rather
than a way for a 15k-positive task to overfit its own private capacity.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from colab.caid_tasks import (
    BINDING_TERMS,
    DISORDER_TERMS,
    LINKER_TERMS,
    MASK,
    TASKS,
    build_task_dataset,
    task_labels,
)
from colab.lite_head import MultiTaskLiteHead, masked_multitask_loss


def entry(length=40, regions=()):
    return {
        "disprot_id": "DP0001", "acc": "P00001",
        "sequence": "A" * length, "length": length,
        "regions": [
            {"term_id": t, "start": s, "end": e} for t, s, e in regions
        ],
    }


DISORDER = sorted(DISORDER_TERMS)[0]
LINKER = sorted(LINKER_TERMS)[0]
BINDING = "GO:0005515"


class TestLabelSemantics:
    def test_nox_calls_unannotated_residues_ordered(self):
        """The convention that keeps AlphaFold baselines out of the NOX top ten:
        absence of annotation is a negative, not a masked residue."""
        lab = task_labels(entry(regions=[(DISORDER, 1, 10)]), "disorder_nox")
        assert lab[:10].sum() == 10
        assert (lab[10:] == 0).all()
        assert (lab != MASK).all()

    def test_pdb_style_masks_unannotated_residues(self):
        """Disorder-PDB ignores them instead, which is why a structural signal
        can top that benchmark and not the other."""
        lab = task_labels(entry(regions=[(DISORDER, 1, 10)]), "disorder_pdb")
        assert (lab[:10] == 1).all()
        assert (lab[10:] == MASK).all()

    def test_linker_positives_come_from_the_linker_term(self):
        lab = task_labels(entry(regions=[(LINKER, 5, 12)]), "linker")
        assert lab[4:12].sum() == 8
        assert lab.sum() == 8

    def test_binding_idr_is_scored_only_inside_disorder(self):
        """Its leader sits at AUC 0.641 because the easy signal is masked away
        by construction: the question is not 'is this disordered' but 'does
        this IDR bind'."""
        e = entry(regions=[(DISORDER, 1, 20), (BINDING, 5, 10)])
        lab = task_labels(e, "binding_idr")
        assert (lab[20:] == MASK).all(), "outside IDRs must be masked"
        assert (lab[4:10] == 1).all(), "binding inside an IDR is positive"
        assert (lab[10:20] == 0).all(), "non-binding IDR residues are negative"

    def test_binding_outside_disorder_still_counts_for_plain_binding(self):
        e = entry(regions=[(BINDING, 30, 35)])
        assert task_labels(e, "binding")[29:35].sum() == 6

    def test_entries_without_the_annotation_are_dropped(self):
        """Returning an all-negative sequence would train on a fiction."""
        e = entry(regions=[(DISORDER, 1, 10)])
        assert task_labels(e, "linker") is None
        assert task_labels(e, "binding") is None

    def test_coordinates_are_one_based_inclusive(self):
        lab = task_labels(entry(regions=[(DISORDER, 1, 1)]), "disorder_nox")
        assert lab[0] == 1 and lab[1] == 0

    def test_unknown_task_is_rejected(self):
        with pytest.raises(ValueError, match="unknown task"):
            task_labels(entry(), "solubility")

    def test_every_declared_task_is_implemented(self):
        e = entry(regions=[(DISORDER, 1, 20), (BINDING, 5, 10), (LINKER, 25, 30)])
        for t in TASKS:
            task_labels(e, t)   # must not raise


class TestDatasetBuild:
    def test_masked_residues_are_carried_as_evidence_not_labels(self):
        e = entry(regions=[(DISORDER, 1, 20), (BINDING, 5, 10)])
        rows = build_task_dataset([e], "binding_idr")
        assert len(rows) == 1
        ev = np.asarray(rows[0]["label_evidence"])
        lab = np.asarray(rows[0]["labels"])
        assert not ev[20:].any(), "non-IDR residues must not be evaluated"
        assert set(np.unique(lab)) <= {0, 1}, "labels must not leak the sentinel"

    def test_row_shape_matches_the_pipeline_contract(self):
        rows = build_task_dataset([entry(regions=[(DISORDER, 1, 10)])], "disorder_nox")
        r = rows[0]
        for key in ("id", "sequence", "length", "labels", "label_evidence", "n_dis"):
            assert key in r
        assert len(r["labels"]) == len(r["label_evidence"]) == r["length"]


class TestSharedTrunk:
    def test_an_extra_task_is_nearly_free(self):
        """The economic argument for one model over five: a task costs one 1x1
        read-out over a trunk the backbone already paid for."""
        one = MultiTaskLiteHead(in_dim=128, tasks=("disorder",))
        five = MultiTaskLiteHead(in_dim=128, tasks=TASKS)
        extra = five.n_trainable() - one.n_trainable()
        assert extra < one.n_trainable() * 0.5
        assert five.cost_per_extra_task() < 1000

    def test_all_tasks_come_from_one_forward_pass(self):
        head = MultiTaskLiteHead(in_dim=64, tasks=("disorder", "linker", "binding"))
        out = head(torch.randn(2, 30, 64))
        assert set(out) == {"disorder", "linker", "binding"}
        for v in out.values():
            assert v.shape == (2, 30)

    def test_per_task_heads_stay_linear(self):
        """A task with 15k positives must not grow private capacity."""
        head = MultiTaskLiteHead(in_dim=64, tasks=("linker",))
        assert isinstance(head.out["linker"], torch.nn.Conv1d)
        assert head.out["linker"].kernel_size == (1,)

    def test_duplicate_tasks_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            MultiTaskLiteHead(in_dim=32, tasks=("linker", "linker"))

    def test_tasks_share_the_trunk_not_the_output(self):
        head = MultiTaskLiteHead(in_dim=64, tasks=("disorder", "linker"))
        x = torch.randn(2, 25, 64)
        out = head(x)
        assert not torch.allclose(out["disorder"], out["linker"])


class TestMaskedLoss:
    def test_masked_residues_do_not_contribute(self):
        """Binding-IDR is defined as 'ignore everything outside an IDR'.
        Scoring those residues would train on the wrong question."""
        logits = {"binding_idr": torch.zeros(1, 10)}
        labels = {"binding_idr": torch.zeros(1, 10)}
        ev_all = {"binding_idr": torch.ones(1, 10, dtype=torch.bool)}
        half = torch.zeros(1, 10, dtype=torch.bool); half[:, :5] = True
        loss_all, _ = masked_multitask_loss(logits, labels, ev_all)
        loss_half, _ = masked_multitask_loss(logits, labels, {"binding_idr": half})
        assert torch.allclose(loss_all, loss_half)

    def test_changing_a_masked_label_changes_nothing(self):
        logits = {"t": torch.randn(1, 8)}
        ev = torch.zeros(1, 8, dtype=torch.bool); ev[:, :4] = True
        a = torch.zeros(1, 8); b = a.clone(); b[:, 4:] = 1.0
        la, _ = masked_multitask_loss(logits, {"t": a}, {"t": ev})
        lb, _ = masked_multitask_loss(logits, {"t": b}, {"t": ev})
        assert torch.allclose(la, lb)

    def test_breakdown_exposes_each_task(self):
        """A multi-task loss that collapses onto the biggest task is worth
        catching early, so the per-task values are reported."""
        logits = {"a": torch.randn(1, 6), "b": torch.randn(1, 6)}
        labels = {"a": torch.zeros(1, 6), "b": torch.ones(1, 6)}
        _, parts = masked_multitask_loss(logits, labels)
        assert set(parts) == {"a", "b"}

    def test_task_weights_apply(self):
        logits = {"a": torch.zeros(1, 4)}
        labels = {"a": torch.ones(1, 4)}
        base, _ = masked_multitask_loss(logits, labels)
        doubled, _ = masked_multitask_loss(logits, labels, weights={"a": 2.0})
        assert torch.allclose(doubled, 2 * base)

    def test_all_masked_raises_rather_than_returning_zero(self):
        """A silent zero loss would look like perfect convergence."""
        logits = {"t": torch.randn(1, 5)}
        with pytest.raises(ValueError, match="every mask was empty"):
            masked_multitask_loss(
                logits, {"t": torch.zeros(1, 5)},
                {"t": torch.zeros(1, 5, dtype=torch.bool)},
            )

    def test_gradients_reach_every_task_head(self):
        head = MultiTaskLiteHead(in_dim=32, tasks=("disorder", "linker"))
        out = head(torch.randn(2, 20, 32))
        loss, _ = masked_multitask_loss(
            out, {"disorder": torch.ones(2, 20), "linker": torch.zeros(2, 20)}
        )
        loss.backward()
        for t in ("disorder", "linker"):
            g = head.out[t].weight.grad
            assert g is not None and g.abs().sum() > 0


class TestStructureAwareness:
    """Structure as an INPUT, not as something to average with afterwards.

    Post-hoc fusion failed: weights fit on training data made rsa+pLDDT worse,
    0.9581 -> 0.9554, because averaging lets a task-agnostic signal override the
    model wherever they disagree, including where the model is right. The top
    CAID3 methods are structure-aware rather than structure-ensembled, and the
    difference is that a learned gate can decide *when* to trust structure.

    This matters most on Disorder-NOX, where no AlphaFold baseline reaches the
    top ten: NOX counts unannotated residues as ordered, so the structural
    shortcut inverts and a model must learn to discount it.
    """

    def _head(self, **kw):
        from colab.lite_head import MultiTaskLiteHead
        return MultiTaskLiteHead(in_dim=64, tasks=("disorder_nox", "linker"), **kw)

    def test_structure_channels_are_cheap(self):
        plain = self._head()
        aware = self._head(structure_dim=16)
        assert aware.n_trainable() - plain.n_trainable() < 10_000

    def test_absent_structure_is_representable_not_imputed(self):
        """An absent AlphaFold entry is not 'buried and confident'. A model that
        cannot tell absence from low accessibility will read absence as order,
        and 109 of 2,340 training proteins have no AlphaFold entry."""
        head = self._head(structure_dim=16)
        x = torch.randn(2, 30, 64)
        zeros = torch.zeros(2, 30)
        present = head(x, rsa=zeros, plddt=zeros, structure_available=torch.ones(2, 30))
        absent = head(x, rsa=zeros, plddt=zeros, structure_available=zeros)
        assert not torch.allclose(present["disorder_nox"], absent["disorder_nox"])

    def test_structure_changes_the_prediction(self):
        head = self._head(structure_dim=16)
        x = torch.randn(2, 30, 64)
        ones = torch.ones(2, 30)
        buried = head(x, rsa=torch.zeros(2, 30), plddt=ones * 95,
                      structure_available=ones)
        exposed = head(x, rsa=ones, plddt=ones * 30, structure_available=ones)
        assert not torch.allclose(buried["disorder_nox"], exposed["disorder_nox"])

    def test_missing_channels_raise_rather_than_pass_a_constant(self):
        """Silently feeding zeros would make the learned gate meaningless while
        still producing plausible output."""
        head = self._head(structure_dim=16)
        with pytest.raises(ValueError, match="structural channels"):
            head(torch.randn(1, 20, 64))

    def test_sequence_only_head_ignores_structure_arguments(self):
        head = self._head()
        out = head(torch.randn(1, 20, 64))
        assert set(out) == {"disorder_nox", "linker"}

    def test_skip_path_spans_the_structural_channels(self):
        """rsa alone scores 0.9459 on Disorder-PDB, so the linear fallback each
        task degrades to should include structure, not sequence alone."""
        head = self._head(structure_dim=16)
        assert head.skip["linker"].in_channels == 64 + 16

    def test_gradients_reach_the_structure_encoder(self):
        head = self._head(structure_dim=16)
        ones = torch.ones(2, 30)
        out = head(torch.randn(2, 30, 64), rsa=ones * 0.5, plddt=ones * 70,
                   structure_available=ones)
        out["linker"].sum().backward()
        g = head.structure.encode[0].weight.grad
        assert g is not None and g.abs().sum() > 0

    def test_rsa_gradient_channel_is_length_preserving(self):
        """Disorder boundaries appear as accessibility transitions, so the
        local rsa gradient is supplied — and must not shift the sequence."""
        from colab.lite_head import StructureChannels

        block = StructureChannels.assemble(
            torch.rand(2, 37), torch.rand(2, 37) * 100, torch.ones(2, 37),
            length=37, batch=2, device=torch.device("cpu"),
        )
        assert block.shape == (2, StructureChannels.N_CHANNELS, 37)

    def test_contact_density_is_a_separate_channel_from_accessibility(self):
        """An exposed loop on a folded domain is accessible AND densely
        contacted; a disordered residue is accessible and uncontacted. Either
        channel alone confuses those two, and they are exactly the false
        positives a disorder predictor makes."""
        from colab.lite_head import StructureChannels

        ones = torch.ones(1, 20)
        loop = StructureChannels.assemble(ones, ones * 80, ones, 20, 1,
                                          torch.device("cpu"), contacts=ones)
        idr = StructureChannels.assemble(ones, ones * 80, ones, 20, 1,
                                         torch.device("cpu"),
                                         contacts=torch.zeros(1, 20))
        assert not torch.allclose(loop, idr)

    def test_wide_receptive_field_covers_a_long_idr(self):
        """IDRs frequently run past 100 residues; the default field is 61."""
        from colab.lite_head import DEFAULT_DILATIONS, WIDE_DILATIONS, receptive_field

        assert receptive_field(DEFAULT_DILATIONS, 4) == 61
        assert receptive_field(WIDE_DILATIONS, 4) > 200

    def test_wide_field_costs_no_extra_parameters(self):
        """Dilation buys context, not weights."""
        from colab.lite_head import MultiTaskLiteHead, WIDE_DILATIONS

        narrow = MultiTaskLiteHead(in_dim=64, tasks=("linker",))
        wide = MultiTaskLiteHead(in_dim=64, tasks=("linker",),
                                 dilations=WIDE_DILATIONS)
        assert narrow.n_trainable() == wide.n_trainable()
