"""Tests for colab/sota_ensemble.py and compact checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn

from colab.compact_checkpoint import extract_trainable_state_dict, is_trainable_key
from colab.sota_ensemble import build_physics_disorder_prior, _search_three_way_blend


class TestCompactCheckpoint:
    def test_trainable_key_filter(self):
        assert is_trainable_key("head.merge.0.weight")
        assert is_trainable_key("esm.layers.0.lora_A")
        assert not is_trainable_key("esm.layers.0.self_attn.q_proj.weight")

    def test_extract_subset(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(4, 1)
                self.frozen = nn.Linear(4, 1)
                for p in self.frozen.parameters():
                    p.requires_grad = False

        m = M()
        state = extract_trainable_state_dict(m)
        assert any("head" in k for k in state)
        assert not any("frozen" in k for k in state)


class TestSOTAEnsemble:
    def test_physics_prior_range(self):
        proteins = [{
            "id": "P1",
            "sequence": "ACDEFGHIKLMNPQRSTVWY" * 2,
            "length": 40,
        }]
        priors = build_physics_disorder_prior(proteins)
        p = priors["P1"]
        assert len(p) == 40
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_three_way_search(self):
        n = 200
        labels = (torch.rand(n) > 0.7).numpy().astype("float32")
        gpu = labels * 0.8 + 0.1
        v6 = labels * 0.7 + 0.15
        phys = labels * 0.6 + 0.2
        r = _search_three_way_blend(labels, gpu, v6, phys, step=0.25)
        assert "best_weights" in r
        assert r["best_auc"] > 0.5
