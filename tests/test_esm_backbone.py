"""Tests for ESM-2 backbone registry."""

from __future__ import annotations

import pytest

from colab.disordernet_gpu import TrainConfig
from colab.esm_backbone import (
    BACKBONE_REGISTRY,
    auto_batch_for_backbone,
    get_backbone_spec,
)


class TestBackboneRegistry:
    def test_650m_spec(self):
        spec = get_backbone_spec("650M")
        assert spec.embed_dim == 1280
        assert spec.n_layers == 33

    def test_3b_spec(self):
        spec = get_backbone_spec("3B")
        assert spec.embed_dim == 2560
        assert spec.params_m == 3000

    def test_alias(self):
        assert get_backbone_spec("esm2_3b").key == "3B"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_backbone_spec("99B")

    def test_auto_batch_3b_conservative(self):
        bs, acc = auto_batch_for_backbone(40.0, "3B")
        assert bs <= 3
        assert acc >= 4

    def test_auto_batch_650m(self):
        bs, acc = auto_batch_for_backbone(40.0, "650M")
        assert bs >= 4


class TestUltra3bProfile:
    def test_ultra3b_backbone(self):
        cfg = TrainConfig.from_profile("ultra3b")
        assert cfg.esm_backbone == "3B"
        assert cfg.esm_embed_dim == 2560
        assert cfg.lora_rank == 64

    def test_all_backbones_registered(self):
        assert len(BACKBONE_REGISTRY) >= 5
