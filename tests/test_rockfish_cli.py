"""Tests for rockfish/run_disordernet.py (no GPU required)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from rockfish.run_disordernet import _resolve_workdir, build_parser  # noqa: E402


class TestRockfishCLI:
    def test_parser_defaults(self):
        args = build_parser().parse_args(["cv"])
        assert args.stage == "cv"
        assert args.profile == "ultra"
        assert args.backbone == "650M"
        assert args.n_folds == 5
        assert args.num_workers == 4
        assert args.calibration_method == "temperature_then_isotonic"

    def test_parser_screen_mode(self):
        args = build_parser().parse_args(
            ["screen", "--screen-mode", "paradigm", "--backbone", "3B", "--profile", "ultra3b"]
        )
        assert args.stage == "screen"
        assert args.screen_mode == "paradigm"
        assert args.backbone == "3B"
        assert args.profile == "ultra3b"

    def test_parser_all_stages(self):
        for stage in ("screen", "cv", "stack", "postprocess", "full"):
            args = build_parser().parse_args([stage])
            assert args.stage == stage

    def test_resolve_workdir_explicit(self, tmp_path):
        wd = _resolve_workdir(str(tmp_path / "runs"))
        assert wd == str(tmp_path / "runs")
        assert os.path.isdir(wd)

    def test_resolve_workdir_cwd_fallback(self, monkeypatch):
        monkeypatch.delenv("SLURM_TMPDIR", raising=False)
        monkeypatch.delenv("TMPDIR", raising=False)
        monkeypatch.delenv("SCRATCH", raising=False)
        wd = _resolve_workdir(None)
        assert os.path.isdir(wd)
