"""Tests for concurrent I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from colab.async_io import fetch_disprot_concurrent, mirror_files_parallel, run_overlapped


class TestAsyncIO:
    def test_run_overlapped(self):
        a, b = run_overlapped(lambda: 1, lambda: 2, "a", "b")
        assert a == 1 and b == 2

    def test_mirror_parallel(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.mkdir()
        files = []
        for i in range(5):
            p = src / f"f{i}.json"
            p.write_text("{}")
            files.append(str(p))
        copied = mirror_files_parallel(files, str(dest), max_workers=4)
        assert len(copied) == 5
        assert (dest / "f0.json").exists()

    def test_fetch_disprot_concurrent_mocked(self, tmp_path):
        cache = tmp_path / "disprot.json"
        page0 = {"data": [{"id": "a"}], "total": 2}
        page1 = {"data": [{"id": "b"}], "total": 2}

        def fake_get(url, params=None, timeout=60):
            page = params["page"]
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = page0 if page == 0 else page1
            return resp

        with patch("colab.async_io.requests.get", side_effect=fake_get):
            entries = fetch_disprot_concurrent(cache_path=str(cache), max_workers=2, per_page=1)
        assert len(entries) == 2
        assert cache.exists()
        # second call hits cache
        entries2 = fetch_disprot_concurrent(cache_path=str(cache))
        assert entries2 == entries
