"""Unit tests for the root CPU ``fetch_disprot.process_entries`` labeling logic.

These are pure-function tests (no network). The DISORDERNET_HOME override keeps the
module's import-time ``makedirs`` inside a temp dir instead of the repo.
"""

from __future__ import annotations

import importlib


def _load_fetch_disprot(monkeypatch, tmp_path):
    monkeypatch.setenv("DISORDERNET_HOME", str(tmp_path))
    import disordernet_paths

    importlib.reload(disordernet_paths)
    import fetch_disprot

    return importlib.reload(fetch_disprot)


def test_labels_single_region_1based_inclusive(monkeypatch, tmp_path):
    fd = _load_fetch_disprot(monkeypatch, tmp_path)
    entries = [
        {
            "disprot_id": "DP1",
            "acc": "P1",
            "name": "protein one",
            "sequence": "A" * 60,
            "regions": [{"type": "disorder", "start": 11, "end": 20}],
        }
    ]
    out = fd.process_entries(entries)
    assert len(out) == 1
    p = out[0]
    assert p["length"] == 60
    labels = p["disorder_labels"]
    # DisProt 1-based inclusive [11, 20] -> 0-based indices 10..19 (10 residues).
    assert sum(labels) == 10
    assert labels[10] == 1 and labels[19] == 1
    assert labels[9] == 0 and labels[20] == 0
    assert abs(p["disorder_fraction"] - 10 / 60) < 1e-9
    assert p["num_disorder_regions"] == 1


def test_skips_short_and_regionless_entries(monkeypatch, tmp_path):
    fd = _load_fetch_disprot(monkeypatch, tmp_path)
    entries = [
        {  # sequence shorter than the 20-residue minimum
            "disprot_id": "SHORT",
            "sequence": "A" * 10,
            "regions": [{"type": "disorder", "start": 1, "end": 5}],
        },
        {  # no regions at all
            "disprot_id": "NOREG",
            "sequence": "A" * 40,
            "regions": [],
        },
    ]
    assert fd.process_entries(entries) == []


def test_multiple_regions_and_end_clamp(monkeypatch, tmp_path):
    fd = _load_fetch_disprot(monkeypatch, tmp_path)
    entries = [
        {
            "disprot_id": "MULTI",
            "sequence": "A" * 30,
            "regions": [
                {"type": "disorder", "start": 1, "end": 5},
                {"type": "disorder", "start": 28, "end": 100},  # end past sequence
            ],
        }
    ]
    p = fd.process_entries(entries)[0]
    labels = p["disorder_labels"]
    # [1,5] -> idx 0..4 (5) ; [28,100] clamped to len 30 -> idx 27..29 (3) => 8.
    assert sum(labels) == 8
    assert labels[0] == 1 and labels[4] == 1 and labels[5] == 0
    assert labels[27] == 1 and labels[29] == 1
    assert p["num_disorder_regions"] == 2


def teardown_module(module):  # noqa: ARG001 - restore default paths for other tests
    import disordernet_paths

    importlib.reload(disordernet_paths)
