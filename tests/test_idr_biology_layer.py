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
    load_boltz_structure_features,
    load_boltz_variance_for_protein,
)
from colab.function_predict import split_function_oof_by_lengths  # noqa: E402
from colab.idr_biology_layer import (  # noqa: E402
    LAYER_VERSION,
    build_idr_layer_package,
    build_protein_idr_layer,
    build_proteome_idr_layer,
    compute_idr_sequence_cues,
    detect_boundary_transition_regions,
    export_idr_layer_bed,
    export_idr_layer_jsonl,
    export_idr_triage_tsv,
    score_protein_triage,
)
from rockfish.run_disordernet import build_parser  # noqa: E402


def _write_multi_sample_job(job: Path, n_res: int = 5, n_samples: int = 3) -> None:
    job.mkdir(parents=True, exist_ok=True)
    for i in range(n_samples):
        np.savez(
            job / f"plddt_x_model_{i}.npz",
            plddt=np.linspace(0.2 + 0.05 * i, 0.8 - 0.02 * i, n_res).astype(np.float32),
        )


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
        assert LAYER_VERSION.startswith("1.1")
        assert rec["n_idr_segments"] >= 1
        roles = {r["group"] for s in rec["idr_segments"] for r in s["predicted_roles"]}
        assert "protein binding" in roles
        assert "condensate / assembly" in roles
        assert rec["hallucination"]["n_hallucinated"] > 0
        assert "distrust_structure_model_in_hallucination_regions" in rec["actions"]
        assert rec["ensemble_proxy"] is not None
        assert rec["conditional_disorder"] is not None
        assert rec["triage"]["score"] > 0
        assert rec["role_hallucination_intersections"]  # halluc on role-bearing IDR

    def test_sequence_cues_corroborate_condensate(self):
        # Charged + aromatic + RGG motif IDR
        seq = ("RGGFY" * 4 + "A" * 20)
        dis = np.concatenate([np.ones(20), np.zeros(20)]).astype(np.float32) * 0.95
        fn = np.zeros((40, 5), dtype=np.float32)
        fn[:20, 3] = 0.9  # condensate
        rec = build_protein_idr_layer(
            protein_id="C1",
            sequence=seq,
            disorder_probs=dis,
            function_probs=fn,
        )
        seg = rec["idr_segments"][0]
        assert "sequence_cues" in seg
        assert seg["sequence_cues"]["motifs"] or seg["sequence_cues"]["cue_tags"]
        role = seg["predicted_roles"][0]
        assert role["group"] == "condensate / assembly"
        assert "sequence_cue_agrees" in (role.get("evidence") or [])
        assert "sequence_cues_corroborate_role_calls" in rec["actions"]

    def test_boundary_transition_and_annotated_mask(self):
        dis = np.array([0.1] * 10 + [0.9] * 10 + [0.1] * 10, dtype=np.float32)
        tmask = np.zeros(30, dtype=np.float32)
        tmask[8:13] = 1
        out = detect_boundary_transition_regions(dis, transition_mask=tmask)
        assert out["n_predicted_boundary_residues"] > 0
        assert out["n_annotated_transition_residues"] == 5
        assert out["annotated_transition_regions"]

    def test_composition_cues_basic(self):
        cues = compute_idr_sequence_cues("RRRGGQQQQQAAAA", 0, 14)
        assert cues["composition"]["frac_basic"] > 0
        assert any(m["motif"] in ("RGG", "polyQ") for m in cues["motifs"])

    def test_proteome_package_exports(self, tmp_path):
        proteins = [
            {"id": "a", "sequence": "A" * 12, "length": 12, "uniprot_acc": "A1"},
            {"id": "b", "sequence": "T" * 12, "length": 12, "uniprot_acc": "B1"},
        ]
        preds = {
            "a": np.array([0.9] * 8 + [0.1] * 4, dtype=np.float32),
            "b": np.array([0.2] * 12, dtype=np.float32),
        }
        package = build_idr_layer_package(proteins, preds, max_proteins_in_summary=10)
        report = package["report"]
        records = package["records"]
        assert report["n_proteins"] == 2
        assert "full_md_conformational_ensembles" in report["non_goals"]
        assert "top_priority_proteins" in report
        assert len(records) == 2
        # triage ranks higher-disorder protein first
        assert records[0]["protein_id"] == "a"

        path = export_idr_layer_jsonl(records, str(tmp_path / "layer.jsonl"))
        lines = Path(path).read_text().strip().splitlines()
        assert len(lines) == 2

        bed = export_idr_layer_bed(records, str(tmp_path / "layer.bed"))
        bed_txt = Path(bed).read_text()
        assert "track name=" in bed_txt
        assert "A1" in bed_txt

        tsv = export_idr_triage_tsv(records, str(tmp_path / "triage.tsv"))
        tsv_lines = Path(tsv).read_text().strip().splitlines()
        assert tsv_lines[0].startswith("protein_id")
        assert len(tsv_lines) == 3

    def test_build_proteome_uses_package(self):
        proteins = [{"id": "a", "sequence": "AAAAAA", "length": 6}]
        preds = {"a": np.ones(6, dtype=np.float32) * 0.8}
        report = build_proteome_idr_layer(proteins, preds)
        assert report["layer_version"] == LAYER_VERSION
        assert report["n_proteins"] == 1

    def test_function_oof_split_matches_disorder_lengths(self):
        ids = ["p1", "p2"]
        lengths = {"p1": 4, "p2": 3}
        flat = np.arange(7 * 5, dtype=np.float32).reshape(7, 5)
        by_id = split_function_oof_by_lengths(flat, ids, lengths)
        assert by_id["p1"].shape == (4, 5)
        assert by_id["p2"].shape == (3, 5)
        np.testing.assert_array_equal(by_id["p1"], flat[:4])
        np.testing.assert_array_equal(by_id["p2"], flat[4:])

    def test_function_oof_split_rejects_mismatch(self):
        try:
            split_function_oof_by_lengths(
                np.zeros((5, 5), dtype=np.float32),
                ["a"],
                {"a": 4},
            )
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_triage_scores_intersection_higher(self):
        base = {
            "disorder_fraction": 0.5,
            "n_role_assignments": 1,
            "idr_segments": [],
            "hallucination": {"n_hallucinated": 0},
            "ensemble_proxy": None,
            "conditional_disorder": {"n_predicted_boundary_residues": 0},
            "role_hallucination_intersections": [],
        }
        low = score_protein_triage(base)
        high = score_protein_triage({
            **base,
            "role_hallucination_intersections": [{"x": 1}],
            "hallucination": {"n_hallucinated": 10},
        })
        assert high["score"] > low["score"]


class TestBoltzVariance:
    def test_sample_stack_and_std(self, tmp_path):
        job = tmp_path / "pred"
        _write_multi_sample_job(job, n_res=3, n_samples=3)
        stack = load_boltz_plddt_sample_stack(str(job))
        assert stack is not None and stack.shape[0] == 3
        std = boltz_plddt_variance_from_dir(str(job))
        assert std is not None and std.shape == (3,)
        assert float(std[0]) > 0

    def test_variance_disk_cache(self, tmp_path):
        root = tmp_path / "outputs"
        job = root / "predictions" / "P1"
        seq = "ACDE"
        _write_multi_sample_job(job, n_res=4, n_samples=3)
        cache = tmp_path / "var_cache"
        std1 = load_boltz_variance_for_protein(
            "DP1", seq, str(root), uniprot_acc="P1",
            cache_dir=str(cache), max_samples=3,
        )
        assert std1 is not None and len(std1) == 4
        assert (cache / "DP1.json").exists()
        import shutil
        shutil.rmtree(job)
        std2 = load_boltz_variance_for_protein(
            "DP1", seq, str(root), uniprot_acc="P1",
            cache_dir=str(cache), max_samples=3,
        )
        np.testing.assert_allclose(std1, std2)

    def test_variance_batch_and_structure_features(self, tmp_path):
        root = tmp_path / "outputs"
        seq = "ACDEF"
        job = root / "predictions" / "ACC1"
        _write_multi_sample_job(job, n_res=5, n_samples=2)
        proteins = [{"id": "x", "sequence": seq, "uniprot_acc": "ACC1", "length": 5}]
        plddt_cache = tmp_path / "plddt_cache"
        var_cache = tmp_path / "var_cache"
        plddt_cache.mkdir()
        with open(plddt_cache / "x.json", "w") as f:
            json.dump({
                "protein_id": "x",
                "target_sequence": seq,
                "plddt": [40.0, 50.0, 60.0, 70.0, 80.0],
                "source": "test",
            }, f)

        plddt_by_id, var_by_id = load_boltz_structure_features(
            proteins,
            str(root),
            plddt_cache_dir=str(plddt_cache),
            variance_cache_dir=str(var_cache),
            max_samples=2,
            verbose=False,
        )
        assert "x" in plddt_by_id and len(plddt_by_id["x"]) == 5
        assert "x" in var_by_id and len(var_by_id["x"]) == 5
        assert float(plddt_by_id["x"][0]) == 40.0
        assert (var_cache / "x.json").exists()


class TestIdrLayerCLI:
    def test_idr_layer_stage(self):
        args = build_parser().parse_args(["idr-layer", "--run-idr-layer"])
        assert args.stage == "idr-layer"
        assert args.run_idr_layer is True

    def test_pipeline_idr_layer_default_on(self):
        args = build_parser().parse_args(["pipeline"])
        assert args.run_idr_layer is True
        assert args.no_idr_layer is False

    def test_pipeline_can_skip_idr_layer(self):
        args = build_parser().parse_args(["pipeline", "--no-idr-layer"])
        assert args.no_idr_layer is True

    def test_export_idr_layer_and_diffusion_samples(self):
        args = build_parser().parse_args([
            "predict", "--fasta", "q.fa",
            "--export-idr-layer",
            "--boltz-diffusion-samples", "7",
        ])
        assert args.export_idr_layer is True
        assert args.boltz_diffusion_samples == 7
