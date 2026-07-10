"""Shared pytest fixtures for DisorderNet Colab tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def sample_disprot_entries() -> list:
    """Synthetic DisProt-like entries covering filter edge cases."""
    seq_a = "A" * 30 + "D" * 10 + "E" * 20  # 60 residues
    seq_b = "M" * 25  # too short after filters if min_seq_len=20 but few disorder
    seq_c = "G" * 50  # binding-only annotation, no disorder term
    seq_d = "K" * 30  # order term only

    return [
        {
            "disprot_id": "DP_TEST01",
            "acc": "P12345",
            "organism": "Homo sapiens",
            "sequence": seq_a,
            "regions": [
                {"start": 31, "end": 40, "term_name": "disorder"},
                {"start": 35, "end": 38, "term_name": "protein binding"},
                {"start": 50, "end": 55, "term_name": "disorder to order"},
            ],
        },
        {
            "disprot_id": "DP_TEST02",
            "sequence": seq_b,
            "regions": [{"start": 1, "end": 5, "term_name": "disorder"}],
        },
        {
            "disprot_id": "DP_TEST03",
            "sequence": seq_c,
            "regions": [{"start": 10, "end": 20, "term_name": "protein binding"}],
        },
        {
            "disprot_id": "DP_TEST04",
            "sequence": seq_d,
            "regions": [{"start": 1, "end": 10, "term_name": "order"}],
        },
        {
            "disprot_id": "DP_TEST05",
            "sequence": "",
            "regions": [],
        },
        {
            "disprot_id": "DP_TEST06",
            "acc": "P99999",
            "organism": "Mus musculus",
            "sequence": seq_a,
            "regions": [{"start": 1, "end": 5, "term_name": "flexible linker"}],
        },
    ]


@pytest.fixture
def disprot_cache_file(tmp_path: Path, sample_disprot_entries: list) -> str:
    path = tmp_path / "disprot_raw.json"
    path.write_text(json.dumps(sample_disprot_entries))
    return str(path)


class _MockAttn(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)


class _MockLayer(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.self_attn = _MockAttn(dim)


class MockESM(nn.Module):
    """Minimal ESM-2 stand-in for unit tests (no 650M download)."""

    def __init__(self, n_layers: int = 4, dim: int = 1280):
        super().__init__()
        self.layers = nn.ModuleList([_MockLayer(dim) for _ in range(n_layers)])
        self._dim = dim

    def forward(self, tokens, repr_layers=None, return_contacts=False):
        batch, length = tokens.shape
        hidden = torch.randn(batch, length, self._dim, device=tokens.device)
        layer_idx = repr_layers[0] if repr_layers else len(self.layers)
        return {"representations": {layer_idx: hidden}}


@pytest.fixture
def mock_esm() -> MockESM:
    return MockESM(n_layers=4, dim=1280)


@pytest.fixture
def mock_batch_converter():
    def converter(data):
        # data: list of (id, sequence)
        max_len = max(len(seq) for _, seq in data) + 2
        tokens = torch.zeros(len(data), max_len, dtype=torch.long)
        for i, (_, seq) in enumerate(data):
            length = len(seq) + 2
            tokens[i, :length] = torch.arange(length)
        return None, None, tokens

    return converter
