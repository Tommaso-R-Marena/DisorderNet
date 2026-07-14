"""Tests for Boltz-2 pLDDT ingest, runner helpers, and Rockfish CLI defaults."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from colab.boltz_plddt import (
    find_boltz_prediction_dir,
    load_boltz_plddt_for_protein,
    normalize_plddt_scale,
    parse_plddt_from_boltz_npz,
    select_proteins_for_boltz,
)
from colab.boltz_runner import (
    PINNED_BOLTZ_VERSION,
    job_name_for_protein,
    resolve_boltz_paths,
    write_boltz_yaml,
)
from colab.disordernet_gpu import merge_plddt_for_training
from colab.inference_fusion import build_combined_plddt_map
from rockfish.boltz_rockfish import default_boltz_root, setup_boltz_for_rockfish
from rockfish.run_disordernet import build_parser


class TestBoltzPlddt:
    def test_normalize_01_to_100(self):
        arr = normalize_plddt_scale(np.array([0.2, 0.9], dtype=np.float32))
        assert float(arr.max()) > 50

    def test_normalize_keeps_0_100(self):
        arr = normalize_plddt_scale(np.array([20.0, 90.0], dtype=np.float32))
        assert abs(float(arr[1]) - 90.0) < 1e-3

    def test_parse_npz(self, tmp_path):
        path = tmp_path / "plddt_job_model_0.npz"
        np.savez(path, plddt=np.array([0.1, 0.5, 0.9], dtype=np.float32))
        out = parse_plddt_from_boltz_npz(str(path))
        assert out.shape == (3,)
        assert float(out[2]) > 50

    def test_find_and_load(self, tmp_path):
        root = tmp_path / "outputs"
        job = root / "predictions" / "P12345"
        job.mkdir(parents=True)
        seq = "ACDEFGHIKL"
        # Fake CIF with CA atoms
        cif_lines = ["data_model", "loop_", "_atom_site.group_PDB", "_atom_site.label_atom_id",
                     "_atom_site.label_comp_id", "_atom_site.label_asym_id",
                     "_atom_site.label_seq_id", "_atom_site.B_iso_or_equiv"]
        aa = list("ACDEFGHIKL")
        aa3 = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU"]
        for i, (a1, a3) in enumerate(zip(aa, aa3), start=1):
            cif_lines.append(f"ATOM CA {a3} A {i} {50.0 + i}")
        (job / "P12345_model_0.cif").write_text("\n".join(cif_lines) + "\n")
        np.savez(job / "plddt_P12345_model_0.npz", plddt=np.linspace(0.2, 0.9, 10))

        found = find_boltz_prediction_dir(str(root), "DP1", "P12345")
        assert found == str(job)

        cache = tmp_path / "cache"
        plddt = load_boltz_plddt_for_protein(
            "DP1", seq, str(root), uniprot_acc="P12345", cache_dir=str(cache),
        )
        assert plddt is not None
        assert len(plddt) == 10
        assert (cache / "DP1.json").exists()

    def test_select_pending(self, tmp_path):
        proteins = [
            {"id": "a", "uniprot_acc": "A1", "sequence": "AAAA"},
            {"id": "b", "uniprot_acc": "B1", "sequence": "BBBB"},
        ]
        done, pending = select_proteins_for_boltz(proteins, str(tmp_path))
        assert len(done) == 0 and len(pending) == 2


class TestBoltzRunner:
    def test_pinned_version(self):
        assert PINNED_BOLTZ_VERSION == "2.2.1"

    def test_write_yaml_msa_empty(self, tmp_path):
        p = {"id": "DP1", "uniprot_acc": "P12345", "sequence": "ACDE"}
        path = tmp_path / "P12345.yaml"
        name = write_boltz_yaml(p, str(path), msa_mode="empty")
        assert name == "P12345"
        text = path.read_text()
        assert "msa: empty" in text
        assert "ACDE" in text

    def test_job_name(self):
        assert job_name_for_protein({"id": "x", "uniprot_acc": "P1"}) == "P1"

    def test_resolve_paths(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DISORDERNET_BOLTZ_ROOT", str(tmp_path / "boltz"))
        paths = resolve_boltz_paths()
        assert paths["output_dir"].endswith("outputs")


class TestMergePreferBoltz:
    def test_boltz_preferred(self):
        af2 = {"a": np.ones(3, dtype=np.float32) * 10}
        boltz = {"a": np.ones(3, dtype=np.float32) * 80}
        combined, stats = build_combined_plddt_map(af2, None, prefer="boltz", plddt_boltz=boltz)
        assert float(combined["a"][0]) == 80.0
        assert stats["from_boltz"] == 1

    def test_merge_training_helper(self):
        af2 = {"a": np.ones(2, dtype=np.float32) * 10}
        boltz = {"a": np.ones(2, dtype=np.float32) * 70}
        out = merge_plddt_for_training(af2, None, plddt_boltz=boltz, prefer="boltz")
        assert float(out["a"][0]) == 70.0


class TestBoltzRockfishCLI:
    def test_defaults_prefer_boltz(self):
        args = build_parser().parse_args(["cv"])
        assert args.structure_backend == "boltz"
        assert args.boltz_mode == "auto"
        assert args.af3_mode == "off"

    def test_boltz_stage(self):
        args = build_parser().parse_args(
            ["boltz", "--boltz-mode", "ingest", "--structure-backend", "boltz"],
        )
        assert args.stage == "boltz"
        assert args.boltz_mode == "ingest"

    def test_diffusion_samples_cli(self):
        args = build_parser().parse_args(
            ["boltz", "--boltz-diffusion-samples", "5"],
        )
        assert args.boltz_diffusion_samples == 5

    def test_setup_off(self, tmp_path):
        cfg = setup_boltz_for_rockfish(
            mode="off", boltz_root=str(tmp_path / "b"), ensure_install=False,
        )
        assert cfg.get("skipped") is True

    def test_default_root(self, monkeypatch):
        monkeypatch.delenv("DISORDERNET_BOLTZ_ROOT", raising=False)
        monkeypatch.delenv("BOLTZ_ROOT", raising=False)
        assert "boltz" in default_boltz_root()
