"""Boltz must not write outside the run by default.

An eval job invoked outside the usual sbatch wrapper (which sets
BOLTZ_MODE=ingest, read-only) inherited the argparse default --boltz-mode auto,
ran Boltz-2, and wrote into ~/boltz — a directory shared by everything that
account runs. It left a truncated 620 MB mols.tar and a zero-byte
boltz2_conf.ckpt there, which Boltz then finds present and fails on, and which
consumed the home quota that had already killed three training jobs.

The default root is now inside the run. A shared location is opt-in.
"""

from __future__ import annotations

import os
from unittest import mock

from colab.boltz_runner import default_boltz_root, resolve_boltz_paths


def _env(**kw):
    base = {k: v for k, v in os.environ.items()
            if k not in ("DISORDERNET_BOLTZ_ROOT", "BOLTZ_ROOT",
                         "DISORDERNET_WORKDIR", "BOLTZ_CACHE")}
    base.update({k: v for k, v in kw.items() if v is not None})
    return mock.patch.dict(os.environ, base, clear=True)


class TestDefaultRootIsContained:
    def test_defaults_under_the_workdir(self, tmp_path):
        with _env(DISORDERNET_WORKDIR=str(tmp_path)):
            assert default_boltz_root() == os.path.join(str(tmp_path), "boltz")

    def test_never_defaults_to_the_home_directory(self, tmp_path):
        """The specific regression: ~/boltz is shared, and a partial download
        there breaks the account's other work."""
        with _env(DISORDERNET_WORKDIR=str(tmp_path)):
            assert default_boltz_root() != os.path.join(os.path.expanduser("~"), "boltz")

    def test_falls_back_to_cwd_not_home(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with _env():
            root = default_boltz_root()
        assert root == os.path.join(str(tmp_path), "boltz")
        assert not root.startswith(os.path.join(os.path.expanduser("~"), "boltz"))


class TestSharedLocationIsOptIn:
    def test_explicit_env_still_wins(self, tmp_path):
        shared = str(tmp_path / "shared_boltz")
        with _env(DISORDERNET_BOLTZ_ROOT=shared, DISORDERNET_WORKDIR=str(tmp_path)):
            assert default_boltz_root() == shared

    def test_legacy_env_name_still_works(self, tmp_path):
        shared = str(tmp_path / "legacy")
        with _env(BOLTZ_ROOT=shared, DISORDERNET_WORKDIR=str(tmp_path)):
            assert default_boltz_root() == shared

    def test_explicit_root_beats_workdir(self, tmp_path):
        """An operator who set the variable meant it."""
        with _env(DISORDERNET_BOLTZ_ROOT="/explicit", DISORDERNET_WORKDIR=str(tmp_path)):
            assert default_boltz_root() == "/explicit"


class TestPathsDeriveFromRoot:
    def test_inputs_outputs_and_cache_stay_under_the_root(self, tmp_path):
        with _env(DISORDERNET_WORKDIR=str(tmp_path)):
            paths = resolve_boltz_paths()
        root = paths["boltz_root"]
        for key in ("input_dir", "output_dir", "cache_dir", "manifest_path"):
            assert paths[key].startswith(root), f"{key} escapes the boltz root"

    def test_cache_can_be_redirected_independently(self, tmp_path):
        """Weights are large and genuinely shareable; inputs/outputs are not."""
        cache = str(tmp_path / "weights")
        with _env(DISORDERNET_WORKDIR=str(tmp_path), BOLTZ_CACHE=cache):
            paths = resolve_boltz_paths()
        assert paths["cache_dir"] == cache
        assert paths["input_dir"].startswith(paths["boltz_root"])
