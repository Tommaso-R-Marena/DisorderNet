"""Tests for the portable CPU-pipeline path configuration."""

from __future__ import annotations

import importlib

import disordernet_paths

_ENV_KEYS = (
    "DISORDERNET_HOME",
    "DISORDERNET_DATA_DIR",
    "DISORDERNET_EMB_DIR",
    "DISORDERNET_RESULTS_ROOT",
)


def _reload(monkeypatch, **env):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(disordernet_paths)


def test_defaults_are_repo_local(monkeypatch):
    dp = _reload(monkeypatch)
    assert dp.DATA_DIR == dp.REPO_ROOT / "data"
    assert dp.EMB_DIR == dp.REPO_ROOT / "data" / "embeddings"
    assert dp.DISPROT_JSON == dp.REPO_ROOT / "data" / "disprot_processed.json"
    assert dp.results_dir("results_v6") == dp.REPO_ROOT / "results_v6"


def test_home_override(monkeypatch, tmp_path):
    dp = _reload(monkeypatch, DISORDERNET_HOME=str(tmp_path))
    assert dp.DATA_DIR == tmp_path / "data"
    assert dp.EMB_DIR == tmp_path / "data" / "embeddings"
    assert dp.DISPROT_JSON == tmp_path / "data" / "disprot_processed.json"
    assert dp.results_dir("results_v6") == tmp_path / "results_v6"


def test_finegrained_overrides_take_precedence(monkeypatch, tmp_path):
    data = tmp_path / "d"
    emb = tmp_path / "e"
    results = tmp_path / "r"
    dp = _reload(
        monkeypatch,
        DISORDERNET_HOME=str(tmp_path / "home"),
        DISORDERNET_DATA_DIR=str(data),
        DISORDERNET_EMB_DIR=str(emb),
        DISORDERNET_RESULTS_ROOT=str(results),
    )
    assert dp.DATA_DIR == data
    assert dp.EMB_DIR == emb
    assert dp.DISPROT_JSON == data / "disprot_processed.json"
    assert dp.results_dir("results") == results / "results"


def test_results_dir_create_makes_directory(monkeypatch, tmp_path):
    dp = _reload(monkeypatch, DISORDERNET_HOME=str(tmp_path))
    out = dp.results_dir("results_v6", create=True)
    assert out.is_dir()
    # Idempotent: calling again does not raise.
    assert dp.results_dir("results_v6", create=True) == out


def teardown_module(module):  # noqa: ARG001 - restore defaults for other tests
    importlib.reload(disordernet_paths)
