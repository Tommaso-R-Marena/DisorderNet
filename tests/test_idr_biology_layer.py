"""Tests for the post-structure IDR biology layer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from colab.boltz_plddt import (  # noqa: E402
    boltz_plddt_variance_from_dir,
    load_boltz_plddt_sample_stack,
    normalize_plddt_scale,
)
from colab.idr_biology_layer import (  # noqa: E402
    LAYER_VERSION,
    build_protein_idr_layer,
    build_proteome_idr_layer,
    export_idr_layer_jsonl,
)
from rockfish.run_disordernet import build_parser  # noqa: E402


class TestIdrBiologyLayer:
    def test_protein_layer_segments_and_roles(self):
        seq = "A" * 40
        dis = np.concatenate([np.ones(20), np.zeros(20)]).astype(np.float32) * 0.9
        fn = np.zeros((40, 5), dtype=np.float32)
        fn[:20, 0] = 0.85  # protein binding
        fn[5:15, 3] = 0.9  # condensate
        plddt = np.full(40, 40.0, dtype=np.float32)
        plddt[8:12] = 85.0  # hallucinated stretch inside IDR

        rec = build_protein_idr_layer(
            protein_id="P1",
            sequence=seq,
            disorder_probs=dis,
            plddt=plddt,
            function_probs=fn,
            boltz_plddt_std=np.linspace(1, 20, 40).astype(np.float32),
        )
        assert rec["layer_version"] == LAYER_VERSION
        assert rec["n_idr_segments"] >= 1
        roles = {r["group"] for s in rec["idr_segments"] for r in s["predicted_roles"]}
        assert "protein binding" in roles
        assert "condensate / assembly" in roles
        assert rec["hallucination"]["n_hallucinated"] > 0
        assert "distrust_structure_model_in_hallucination_regions" in rec["actions"]
        assert rec["ensemble_proxy"] is not None

    def test_proteome_export_jsonl(self, tmp_path):
        proteins = [
            {"id": "a", "sequence": "AAAAAA", "length": 6, "uniprot_acc": "A1"},
            {"id": "b", "sequence": "TTTTTT", "length": 6, "uniprot_acc": "B1"},
        ]
        preds = {
            "a": np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1], dtype=np.float32),
            "b": np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32),
        }
        report = build_proteome_idr_layer(proteins, preds, max_proteins_in_summary=10)
        assert report["n_proteins"] == 2
        assert "full_md_conformational_ensembles" in report["non_goals"]
        path = export_idr_layer_jsonl(report["proteins"], str(tmp_path / "layer.jsonl"))
        lines = Path(path).read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["protein_id"] in ("a", "b")


class TestBoltzVariance:
    def test_sample_stack_and_std(self, tmp_path):
        job = tmp_path / "pred"
        job.mkdir()
        for i in range(3):
            np.savez(
                job / f"plddt_x_model_{i}.npz",
                plddt=np.array([0.2 + 0.1 * i, 0.5, 0.8 - 0.05 * i], dtype=np.float32),
            )
        stack = load_boltz_plddt_sample_stack(str(job))
        assert stack is not None and stack.shape[0] == 3
        std = boltz_plddt_variance_from_dir(str(job))
        assert std is not None and std.shape == (3,)
        assert float(std[0]) > 0


class TestIdrLayerCLI:
    def test_idr_layer_stage(self):
        args = build_parser().parse_args(["idr-layer", "--run-idr-layer"])
        assert args.stage == "idr-layer"
        assert args.run_idr_layer is True
