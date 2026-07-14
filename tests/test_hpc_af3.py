"""Tests for HPC efficiency helpers and AF3 Rockfish wiring."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from colab.hpc_efficiency import (
    apply_hpc_runtime_settings,
    save_disk_token_cache,
    load_disk_token_cache,
    token_cache_key,
)
from rockfish.af3_rockfish import default_af3_root, setup_af3_for_rockfish
from rockfish.run_disordernet import build_parser


class TestHpcEfficiency:
    def test_runtime_settings(self):
        report = apply_hpc_runtime_settings(verbose=False)
        assert isinstance(report, dict)

    def test_token_disk_roundtrip(self, tmp_path):
        proteins = [{"id": "P1", "sequence": "ACDE"}]
        payload = {"tokens": torch.arange(6), "aa_idx": torch.arange(4)}
        path = save_disk_token_cache("P1", "ACDE", payload, str(tmp_path))
        assert path and os.path.isfile(path)
        loaded = load_disk_token_cache(proteins, str(tmp_path))
        assert "P1" in loaded
        assert torch.equal(loaded["P1"]["tokens"], payload["tokens"])

    def test_token_cache_key_stable(self):
        assert token_cache_key("A", "ACDE") == token_cache_key("A", "ACDE")
        assert token_cache_key("A", "ACDE") != token_cache_key("A", "ACDF")


class TestAf3Rockfish:
    def test_setup_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DISORDERNET_AF3_ROOT", str(tmp_path / "af3"))
        cfg = setup_af3_for_rockfish(mode="off", af3_root=str(tmp_path / "af3"))
        assert cfg.get("skipped") or cfg.get("mode") == "off"

    def test_setup_ingest_missing(self, tmp_path):
        root = tmp_path / "af3"
        root.mkdir()
        cfg = setup_af3_for_rockfish(mode="ingest", af3_root=str(root), clone_repo=False)
        assert cfg["ready"] is False  # no outputs yet

    def test_cli_af3_stage(self):
        args = build_parser().parse_args(["af3", "--af3-mode", "ingest"])
        assert args.stage == "af3"
        assert args.af3_mode == "ingest"

    def test_default_root(self, monkeypatch):
        monkeypatch.delenv("DISORDERNET_AF3_ROOT", raising=False)
        monkeypatch.delenv("AF3_ROOT", raising=False)
        assert "af3" in default_af3_root()
