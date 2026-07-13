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
from colab.function_predict import (  # noqa: E402
    calibrate_function_probs,
    split_function_oof_by_lengths,
    tune_function_threshold,
)
from colab.idr_biology_layer import (  # noqa: E402
    LAYER_VERSION,
    annotate_role_call_confidence,
    build_idr_layer_package,
    build_protein_idr_layer,
    build_proteome_idr_layer,
    compute_idr_sequence_cues,
    compute_quality_flags,
    detect_boundary_transition_regions,
    evaluate_role_calls_against_annotations,
    export_idr_disorder_bedgraph,
    export_idr_layer_bed,
    export_idr_layer_bundle,
    export_idr_layer_jsonl,
    export_idr_role_bedgraphs,
    export_idr_triage_tsv,
    score_protein_triage,
)
from colab.idr_layer_biophysics import (  # noqa: E402
    compute_idr_biophysics_cues,
    kappa_lite,
    scd_lite,
)
from colab.idr_layer_io import (  # noqa: E402
    ligand_binding_support,
    load_disorder_preds_from_dir,
    load_function_preds_from_dir,
    load_ligand_map,
    load_partner_map,
    partner_binding_support,
)
from colab.idr_layer_ops import (  # noqa: E402
    compare_idr_layer_jsonl,
    export_idr_layer_gff3,
    filter_layer_records,
    load_idr_layer_jsonl,
    proteome_landscape_summary,
    resume_protein_ids_from_jsonl,
    validate_idr_layer_record,
    validate_idr_layer_records,
    write_idr_layer_html,
    write_idr_layer_markdown,
    write_idr_run_manifest,
    write_triage_protein_cards,
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
        assert LAYER_VERSION.startswith("1.6")
        assert rec["n_idr_segments"] >= 1
        assert "quality" in rec
        assert rec["quality"]["severity"] in ("ok", "review", "quarantine")
        roles = {r["group"] for s in rec["idr_segments"] for r in s["predicted_roles"]}
        assert "protein binding" in roles
        assert "condensate / assembly" in roles
        assert rec["hallucination"]["n_hallucinated"] > 0
        assert "distrust_structure_model_in_hallucination_regions" in rec["actions"]
        assert rec["ensemble_proxy"] is not None
        assert rec["conditional_disorder"] is not None
        assert rec["triage"]["score"] > 0
        assert rec["role_hallucination_intersections"]  # halluc on role-bearing IDR
        assert validate_idr_layer_record(rec) == []

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

    def test_partner_context_and_role_validation(self, tmp_path):
        # Charged IDR so complementary-charge partner cue clears the ≥0.4 bar
        proteins = [{
            "id": "p1",
            "sequence": ("K" * 10) + ("A" * 10) + ("E" * 10),
            "length": 30,
            "uniprot_acc": "U1",
            "functional_regions": [
                {"start": 1, "end": 10, "term_norm": "protein binding"},
            ],
        }]
        preds = {"p1": np.array([0.9] * 10 + [0.1] * 20, dtype=np.float32)}
        fn = np.zeros((30, 5), dtype=np.float32)
        fn[:10, 0] = 0.9
        package = build_idr_layer_package(
            proteins, preds,
            function_probs_by_id={"p1": fn},
            partners_by_id={"p1": ["D" * 12 + "I" * 8]},
        )
        rec = package["records"][0]
        assert any(s.get("partner_context") for s in rec["idr_segments"])
        roles = rec["idr_segments"][0]["predicted_roles"]
        assert roles[0].get("conditioned_prob") is not None
        assert "partner_context_supports_binding" in (roles[0].get("evidence") or [])
        val = package["report"]["role_validation"]
        assert val["enabled"] is True
        assert val["micro"]["f1"] == 1.0
        bg = export_idr_disorder_bedgraph(proteins, preds, str(tmp_path / "dis.bedgraph"))
        assert "U1" in Path(bg).read_text()

    def test_load_preds_and_partners(self, tmp_path):
        proteins = [{"id": "DP1", "sequence": "ACDE", "uniprot_acc": "P1", "length": 4}]
        (tmp_path / "DP1.tsv").write_text(
            "position\tresidue\tscore\n1\tA\t0.1\n2\tC\t0.9\n3\tD\t0.8\n4\tE\t0.2\n"
        )
        loaded = load_disorder_preds_from_dir(str(tmp_path), proteins=proteins)
        assert "DP1" in loaded and abs(float(loaded["DP1"][1]) - 0.9) < 1e-5
        partner_json = tmp_path / "partners.json"
        partner_json.write_text(json.dumps({"DP1": ["KKKKEEEE"]}))
        assert load_partner_map(str(partner_json))["DP1"] == ["KKKKEEEE"]
        assert partner_binding_support("DDDDAAAA", ["KKKKIIII"])["support"] > 0

    def test_ligand_context_and_bundle(self, tmp_path):
        from colab.idr_biology_layer import export_idr_layer_bundle

        proteins = [{
            "id": "p1",
            "sequence": ("AILMFV" * 3) + ("G" * 12),
            "length": 30,
            "uniprot_acc": "U1",
        }]
        preds = {"p1": np.array([0.95] * 18 + [0.1] * 12, dtype=np.float32)}
        fn = np.zeros((30, 5), dtype=np.float32)
        fn[:18, 4] = 0.85  # lipid / small molecule
        package = build_idr_layer_package(
            proteins, preds,
            function_probs_by_id={"p1": fn},
            ligands_by_id={"p1": ["lipid", {"type": "metal", "name": "Zn"}]},
            max_workers=2,
        )
        rec = package["records"][0]
        assert any(s.get("ligand_context") for s in rec["idr_segments"])
        assert package["report"]["structure_distrust"] is not None
        assert package["report"]["n_segments_with_ligand_context"] >= 1
        lb = ligand_binding_support("AILMFVWAA", [{"type": "lipid"}])
        assert lb["support"] > 0 and "lipid / small molecule binding" in lb["target_roles"]

        paths = export_idr_layer_bundle(
            out_dir=str(tmp_path / "out"),
            report=package["report"],
            records=package["records"],
            proteins=proteins,
            disorder_probs_by_id=preds,
            gzip_jsonl=True,
        )
        assert Path(paths["jsonl"]).exists() and paths["jsonl"].endswith(".gz")
        assert Path(paths["roles"]).exists()
        lig_path = tmp_path / "ligands.json"
        lig_path.write_text(json.dumps({"p1": ["nucleotide"]}))
        assert load_ligand_map(str(lig_path))["p1"][0]["type"] == "nucleic"

    def test_v14_cache_tune_compare_markdown(self, tmp_path):
        proteins = [{
            "id": "p1",
            "sequence": "A" * 20,
            "length": 20,
            "uniprot_acc": "U1",
            "functional_regions": [
                {"start": 1, "end": 10, "term_norm": "protein binding"},
            ],
        }]
        preds = {"p1": np.array([0.9] * 12 + [0.1] * 8, dtype=np.float32)}
        fn = np.zeros((20, 5), dtype=np.float32)
        fn[:10, 0] = 0.9
        cache = tmp_path / "cache"
        pkg1 = build_idr_layer_package(
            proteins, preds, function_probs_by_id={"p1": fn},
            cache_dir=str(cache), max_workers=1,
        )
        assert pkg1["report"]["cache"]["hits"] == 0
        assert pkg1["report"]["landscape"]["n_proteins"] == 1
        pkg2 = build_idr_layer_package(
            proteins, preds, function_probs_by_id={"p1": fn},
            cache_dir=str(cache), max_workers=1,
        )
        assert pkg2["report"]["cache"]["hits"] == 1

        yt = np.zeros((100, 5), dtype=np.float32)
        yt[:30, 0] = 1
        yp = yt.copy()
        yp[:30, 0] = 0.75
        yp[30:, 0] = 0.2
        tune = tune_function_threshold(yt, yp)
        assert tune["enabled"] is True

        paths = export_idr_layer_bundle(
            out_dir=str(tmp_path / "out"),
            report=pkg1["report"],
            records=pkg1["records"],
            proteins=proteins,
            disorder_probs_by_id=preds,
        )
        assert Path(paths["markdown"]).exists()
        assert "DisorderNet" in Path(paths["markdown"]).read_text()
        assert Path(paths["caid_dir"]).is_dir()
        cmp = compare_idr_layer_jsonl(paths["jsonl"], paths["jsonl"])
        assert cmp["n_shared"] == 1
        assert abs(cmp["mean_abs_triage_delta"]) < 1e-9
        land = proteome_landscape_summary(pkg1["records"])
        assert land["n_proteins_with_roles"] >= 1

    def test_export_idr_layer_and_diffusion_samples(self):
        args = build_parser().parse_args([
            "predict", "--fasta", "q.fa",
            "--export-idr-layer",
            "--boltz-diffusion-samples", "7",
            "--idr-partners", "partners.json",
            "--idr-ligands", "ligands.json",
            "--idr-preds-dir", "preds",
            "--idr-workers", "8",
            "--idr-gzip",
            "--idr-auto-threshold",
            "--idr-calibrate-function",
            "--idr-cache",
            "--idr-compare", "old.jsonl",
            "--idr-resume", "partial.jsonl",
        ])
        assert args.idr_auto_threshold is True
        assert args.idr_calibrate_function is True
        assert args.idr_cache is True
        assert args.idr_compare == "old.jsonl"
        assert args.idr_resume == "partial.jsonl"

    def test_v15_quality_calibration_resume_html(self, tmp_path):
        # Quality: missing structure + short chain + idr-rich no roles
        short = build_protein_idr_layer(
            protein_id="S1",
            sequence="A" * 20,
            disorder_probs=np.ones(20, dtype=np.float32) * 0.9,
        )
        assert "missing_structure_plddt" in short["quality"]["flags"]
        assert "short_chain" in short["quality"]["flags"]
        assert short["quality"]["severity"] == "review"

        # Calibration on synthetic OOF
        yt = np.zeros((200, 5), dtype=np.float32)
        yt[:80, 0] = 1
        yp = np.clip(yt + np.random.RandomState(0).normal(0, 0.15, yt.shape), 0.01, 0.99)
        yp = yp.astype(np.float32)
        cal_probs, cal_rep = calibrate_function_probs(yt, yp, min_positives=10)
        assert cal_rep["enabled"] is True
        assert cal_probs.shape == (200, 5)
        assert "protein binding" in cal_rep["per_group"]

        proteins = [
            {"id": "a", "sequence": "A" * 40, "length": 40, "uniprot_acc": "A1"},
            {"id": "b", "sequence": "T" * 40, "length": 40, "uniprot_acc": "B1"},
        ]
        preds = {
            "a": np.array([0.9] * 25 + [0.1] * 15, dtype=np.float32),
            "b": np.array([0.2] * 40, dtype=np.float32),
        }
        fn = {
            "a": np.zeros((40, 5), dtype=np.float32),
            "b": np.zeros((40, 5), dtype=np.float32),
        }
        fn["a"][:20, 0] = 0.85
        package = build_idr_layer_package(
            proteins, preds, function_probs_by_id=fn, max_workers=2,
        )
        assert package["report"]["quality"]["n_ok"] + package["report"]["quality"]["n_review"] >= 1
        schema = validate_idr_layer_records(package["records"])
        assert schema["ok"] is True

        # First export, then resume skipping "a"
        paths = export_idr_layer_bundle(
            out_dir=str(tmp_path / "out1"),
            report=package["report"],
            records=package["records"],
            proteins=proteins,
            disorder_probs_by_id=preds,
            function_probs_by_id=fn,
        )
        assert Path(paths["html"]).exists()
        assert "DisorderNet" in Path(paths["html"]).read_text()
        assert Path(paths["role_bedgraphs"]).is_dir()
        html_path = write_idr_layer_html(package["report"], str(tmp_path / "x.html"))
        assert Path(html_path).stat().st_size > 100

        prior = load_idr_layer_jsonl(paths["jsonl"])
        skip = resume_protein_ids_from_jsonl(paths["jsonl"])
        assert "a" in skip and "b" in skip
        resumed = build_idr_layer_package(
            proteins, preds, function_probs_by_id=fn,
            skip_protein_ids={"a"},
            prior_records=[r for r in prior if r["protein_id"] == "a"],
        )
        assert resumed["report"]["n_resumed"] == 1
        assert resumed["report"]["n_built_this_run"] == 1
        assert {r["protein_id"] for r in resumed["records"]} == {"a", "b"}

        role_paths = export_idr_role_bedgraphs(proteins, fn, str(tmp_path / "roles_bg"))
        assert len(role_paths) == 5
        assert "track type=bedGraph" in Path(next(iter(role_paths.values()))).read_text()

        q = compute_quality_flags(short)
        assert q["n_flags"] >= 2

    def test_v16_biophysics_gff_cards_function_reload(self, tmp_path):
        # Blocky charged IDR → patterning cues
        seq = ("KKKK" + "EEEE") * 4 + "A" * 10
        bio = compute_idr_biophysics_cues(seq, 0, 32)
        assert "fcr" in bio and bio["fcr"] > 0.5
        assert bio["kappa_lite"] is not None or bio["scd_lite"] != 0.0
        assert scd_lite("K" * 10 + "E" * 10) != 0.0
        assert kappa_lite("KE" * 20) is not None or kappa_lite("KKKKKEEEEE" * 3) is not None

        cues = compute_idr_sequence_cues("RGGFYKEN" + "A" * 20, 0, 28)
        assert "biophysics" in cues
        assert any(m["motif"] in ("RGG", "KEN_box") for m in cues["motifs"])

        roles = annotate_role_call_confidence([
            {"group": "protein binding", "mean_prob": 0.8, "max_prob": 0.9, "evidence": ["sequence_cue_agrees"]},
            {"group": "nucleic acid binding", "mean_prob": 0.75, "max_prob": 0.85},
            {"group": "condensate / assembly", "mean_prob": 0.72, "max_prob": 0.8},
        ])
        assert all("confidence" in r for r in roles)
        assert any(r.get("multi_role_conflict") for r in roles)

        proteins = [
            {"id": "a", "sequence": seq, "length": len(seq), "uniprot_acc": "A1"},
            {"id": "b", "sequence": "T" * 40, "length": 40, "uniprot_acc": "B1"},
        ]
        preds = {
            "a": np.array([0.95] * 32 + [0.1] * 10, dtype=np.float32),
            "b": np.ones(40, dtype=np.float32) * 0.1,
        }
        fn = {"a": np.zeros((42, 5), dtype=np.float32), "b": np.zeros((40, 5), dtype=np.float32)}
        fn["a"][:32, 3] = 0.88
        package = build_idr_layer_package(proteins, preds, function_probs_by_id=fn)
        rec = next(r for r in package["records"] if r["protein_id"] == "a")
        assert rec["idr_segments"][0]["predicted_roles"][0].get("confidence") is not None
        assert "biophysics" in rec["idr_segments"][0]["sequence_cues"]

        paths = export_idr_layer_bundle(
            out_dir=str(tmp_path / "bundle"),
            report=package["report"],
            records=package["records"],
            proteins=proteins,
            disorder_probs_by_id=preds,
            function_probs_by_id=fn,
            cards_top_n=2,
        )
        assert Path(paths["gff3"]).exists()
        assert "##gff-version 3" in Path(paths["gff3"]).read_text()
        assert Path(paths["cards_dir"]).is_dir()
        assert Path(paths["manifest"]).exists()
        assert int(paths["cards_n"]) >= 1
        gff = export_idr_layer_gff3(package["records"], str(tmp_path / "x.gff3"))
        assert "IDR_segment" in Path(gff).read_text()
        cards = write_triage_protein_cards(package["records"], str(tmp_path / "cards"), top_n=1)
        assert len(cards) == 1
        write_idr_run_manifest(str(tmp_path / "man.json"), layer_version=LAYER_VERSION)
        filtered = filter_layer_records(package["records"], min_triage_score=0.0)
        assert len(filtered) == len(package["records"])

        # Function pred reload
        fn_dir = tmp_path / "fn"
        fn_dir.mkdir()
        np.save(fn_dir / "a.npy", fn["a"])
        loaded = load_function_preds_from_dir(str(fn_dir), proteins=proteins)
        assert "a" in loaded and loaded["a"].shape[1] == 5


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
            "--idr-partners", "partners.json",
            "--idr-ligands", "ligands.json",
            "--idr-preds-dir", "preds",
            "--idr-workers", "8",
            "--idr-gzip",
            "--idr-disorder-threshold", "0.55",
            "--idr-auto-threshold",
            "--idr-cache",
        ])
        assert args.export_idr_layer is True
        assert args.boltz_diffusion_samples == 7
        assert args.idr_ligands == "ligands.json"
        assert args.idr_workers == 8
        assert args.idr_gzip is True
        assert abs(args.idr_disorder_threshold - 0.55) < 1e-9
        assert args.idr_auto_threshold is True
        assert args.idr_cache is True

    def test_v15_flags(self):
        args = build_parser().parse_args([
            "idr-layer",
            "--idr-calibrate-function",
            "--idr-resume", "partial.jsonl.gz",
            "--idr-no-html",
            "--idr-no-role-bedgraphs",
            "--idr-function-preds-dir", "fn/",
            "--idr-min-triage", "2.5",
            "--idr-quarantine-only",
            "--idr-no-gff",
            "--idr-no-cards",
            "--idr-cards-top-n", "12",
        ])
        assert args.idr_calibrate_function is True
        assert args.idr_resume == "partial.jsonl.gz"
        assert args.idr_no_html is True
        assert args.idr_no_role_bedgraphs is True
        assert args.idr_function_preds_dir == "fn/"
        assert abs(args.idr_min_triage - 2.5) < 1e-9
        assert args.idr_quarantine_only is True
        assert args.idr_no_gff is True
        assert args.idr_no_cards is True
        assert args.idr_cards_top_n == 12
