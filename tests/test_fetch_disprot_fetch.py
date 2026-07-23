"""Tests for fetch_disprot network path using a mocked requests session."""
from __future__ import annotations

import importlib


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _load(monkeypatch, tmp_path):
    monkeypatch.setenv("DISORDERNET_HOME", str(tmp_path))
    import disordernet_paths
    importlib.reload(disordernet_paths)
    import fetch_disprot
    return importlib.reload(fetch_disprot)


def test_fetch_disprot_entries_single_page(monkeypatch, tmp_path):
    fd = _load(monkeypatch, tmp_path)
    entries = [{"disprot_id": f"DP{i}", "sequence": "A" * 40,
                "regions": [{"type": "disorder", "start": 1, "end": 10}]}
               for i in range(3)]

    def fake_get(url, params=None, timeout=None):
        return _FakeResp({"data": entries, "total": len(entries)})

    monkeypatch.setattr(fd.requests, "get", fake_get)
    got = fd.fetch_disprot_entries()
    assert len(got) == 3
    assert got[0]["disprot_id"] == "DP0"


def test_fetch_disprot_entries_handles_error(monkeypatch, tmp_path):
    fd = _load(monkeypatch, tmp_path)

    def boom(url, params=None, timeout=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(fd.requests, "get", boom)
    # gracefully returns [] rather than raising
    assert fd.fetch_disprot_entries() == []


def test_main_writes_processed_json(monkeypatch, tmp_path):
    fd = _load(monkeypatch, tmp_path)
    entries = [{"disprot_id": "DP1", "acc": "P1", "name": "x", "sequence": "A" * 50,
                "regions": [{"type": "disorder", "start": 5, "end": 25}]}]
    monkeypatch.setattr(fd.requests, "get",
                        lambda *a, **k: _FakeResp({"data": entries, "total": 1}))
    fd.main()
    import os, json
    out = os.path.join(str(fd.OUTPUT_DIR), "disprot_processed.json")
    assert os.path.exists(out)
    data = json.load(open(out))
    assert data[0]["disprot_id"] == "DP1" and sum(data[0]["disorder_labels"]) == 21
