"""The soft target must reach the loss and never reach a metric.

PREREGISTRATION_11 changes the training target on ~8% of residues. Two things
have to hold, and the second is the one that would fail silently: the fractional
target must be what the loss sees, and the hard label must be what every AUC,
class-balance statistic and stratified split sees. An AUC computed against a
target of 0.4 is not an AUC.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rockfish.train_multitask import hard_labels, load_soft_labels  # noqa: E402


class TestTheAccessor:
    def test_it_prefers_the_hard_label_when_one_exists(self):
        row = {"task_labels": {"disorder_pdb": np.array([0.4, 0.6, 1.0])},
               "task_labels_hard": {"disorder_pdb": np.array([0, 1, 1])}}
        assert hard_labels(row, "disorder_pdb").tolist() == [0, 1, 1]

    def test_it_falls_back_when_no_soft_target_was_used(self):
        """A hard-label run must behave exactly as it did before."""
        row = {"task_labels": {"disorder_pdb": np.array([0, 1, 1])}}
        assert hard_labels(row, "disorder_pdb").tolist() == [0, 1, 1]

    def test_a_metric_computed_through_it_is_never_fractional(self):
        row = {"task_labels": {"disorder_pdb": np.linspace(0, 1, 11)},
               "task_labels_hard":
                   {"disorder_pdb": (np.linspace(0, 1, 11) > 0.5).astype(int)}}
        y = hard_labels(row, "disorder_pdb")
        assert set(np.unique(y)) <= {0, 1}


class TestTheCache:
    def test_no_path_means_no_change(self):
        assert load_soft_labels(None) == {}
        assert load_soft_labels("") == {}

    def test_a_missing_file_is_refused_not_ignored(self, tmp_path):
        """A registered soft-label arm must not quietly become a control."""
        with pytest.raises(SystemExit) as exc:
            load_soft_labels(str(tmp_path / "absent.json"))
        assert "must not silently fall back" in str(exc.value)

    def test_it_loads_indices_and_values(self, tmp_path):
        p = tmp_path / "soft.json"
        p.write_text(json.dumps({"proteins": {
            "P1": {"index": [0, 3, 7], "soft": [0.0, 0.5, 1.0],
                   "length": 10, "n_structures": 4}}}))
        got = load_soft_labels(str(p))
        idx, val = got["P1"]
        assert idx.tolist() == [0, 3, 7]
        assert val.tolist() == pytest.approx([0.0, 0.5, 1.0])


class TestTheTargetItself:
    def test_a_fractional_target_is_a_valid_bce_target(self):
        """The whole change rests on this: BCE already accepts a soft target."""
        torch = pytest.importorskip("torch")
        logit = torch.zeros(5, requires_grad=True)
        soft = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, soft)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logit.grad).all()

    def test_a_half_target_has_its_minimum_at_a_half(self):
        """Which is the mechanism: an ambiguous residue stops pulling to 0 or 1."""
        torch = pytest.importorskip("torch")
        soft = torch.tensor([0.5])
        losses = {p: float(torch.nn.functional.binary_cross_entropy_with_logits(
            torch.logit(torch.tensor([p])), soft)) for p in (0.1, 0.5, 0.9)}
        assert losses[0.5] < losses[0.1] and losses[0.5] < losses[0.9]
