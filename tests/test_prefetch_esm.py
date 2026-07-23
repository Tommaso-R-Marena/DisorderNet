"""Tests for the login-node ESM weight prefetch helper."""
from __future__ import annotations

from rockfish.prefetch_esm import DEFAULT_MODELS, esm_ckpt_targets


def test_targets_are_model_and_regression():
    targets = esm_ckpt_targets("esm2_t33_650M_UR50D", "/cache")
    urls = [u for u, _ in targets]
    dests = [d for _, d in targets]
    assert urls == [
        "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
        "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt",
    ]
    assert dests == [
        "/cache/esm2_t33_650M_UR50D.pt",
        "/cache/esm2_t33_650M_UR50D-contact-regression.pt",
    ]


def test_default_models_cover_v8_backbones():
    assert DEFAULT_MODELS == [
        "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D",
    ]


def test_dest_uses_basename_of_url():
    for url, dest in esm_ckpt_targets("esm2_t12_35M_UR50D", "/x/y"):
        assert dest.startswith("/x/y/")
        assert dest.endswith(url.rsplit("/", 1)[1])
