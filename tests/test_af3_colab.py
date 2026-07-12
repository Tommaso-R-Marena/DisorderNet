"""Tests for colab/af3_colab.py."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

from colab.af3_colab import (
    build_protein_json,
    resolve_af3_paths,
    select_proteins_for_af3,
    setup_af3_for_colab,
    verify_af3_weights,
    write_af3_input_jsons,
)


def test_resolve_paths():
    paths = resolve_af3_paths("/tmp/af3")
    assert paths["weights_path"].endswith("af3.bin")
    assert paths["output_dir"].endswith("outputs")
    assert paths["model_dir"] == "/tmp/af3"


def test_build_protein_json():
    p = {"id": "dp1", "sequence": "ACDE", "uniprot_acc": "P99999"}
    j = build_protein_json(p)
    assert j["name"] == "P99999"
    assert j["sequences"][0]["protein"]["sequence"] == "ACDE"
    assert j["dialect"] == "alphafold3"
    assert j["version"] == 1


def test_build_protein_json_msa_free():
    p = {"id": "dp1", "sequence": "ACDE", "uniprot_acc": "P99999"}
    j = build_protein_json(p, msa_free=True)
    prot = j["sequences"][0]["protein"]
    assert prot["unpairedMsa"] == ""
    assert prot["pairedMsa"] == ""
    assert prot["templates"] == []


def test_write_input_jsons(tmp_path):
    proteins = [{"id": "a", "sequence": "AAA", "uniprot_acc": "P1"}]
    paths = write_af3_input_jsons(proteins, str(tmp_path))
    assert len(paths) == 1
    with open(paths[0]) as f:
        data = json.load(f)
    assert data["sequences"][0]["protein"]["id"] == ["A"]


def test_verify_weights_missing(tmp_path):
    paths = resolve_af3_paths(str(tmp_path))
    ok, msg = verify_af3_weights(paths)
    assert not ok
    assert "af3.bin" in msg


def test_verify_weights_present(tmp_path):
    paths = resolve_af3_paths(str(tmp_path))
    os.makedirs(paths["drive_root"], exist_ok=True)
    with open(paths["weights_path"], "wb") as f:
        f.write(b"x" * 1024)
    ok, msg = verify_af3_weights(paths)
    assert ok


def test_setup_off_mode():
    cfg = setup_af3_for_colab(mode="off", mount_drive=False, clone_repo=False)
    assert cfg["ready"]
    assert cfg["mode"] == "off"


def test_setup_auto_mode_ready_with_weights(tmp_path):
    paths = resolve_af3_paths(str(tmp_path))
    os.makedirs(paths["drive_root"], exist_ok=True)
    with open(paths["weights_path"], "wb") as f:
        f.write(b"x" * 1024)

    with patch("colab.af3_colab.resolve_af3_paths", return_value=paths), \
         patch("colab.af3_colab.clone_alphafold3_repo", return_value=(True, "ok")), \
         patch("colab.af3_colab.docker_available", return_value=False):
        cfg = setup_af3_for_colab(
            mode="auto",
            drive_root=str(tmp_path),
            mount_drive=False,
            clone_repo=True,
        )
    assert cfg["weights_ok"]
    assert cfg["ready"]


def test_select_proteins_pending(tmp_path):
    paths = resolve_af3_paths(str(tmp_path))
    os.makedirs(paths["output_dir"], exist_ok=True)
    proteins = [
        {"id": "DP1", "uniprot_acc": "P11111", "sequence": "ACDE"},
        {"id": "DP2", "uniprot_acc": "P22222", "sequence": "FGHI"},
    ]
    done, pending = select_proteins_for_af3(proteins, paths["output_dir"])
    assert len(done) == 0
    assert len(pending) == 2
