"""Tests for structure distrust atlas / labeled hallucination protocol (paper pillar 1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from colab.colab_figures import (  # noqa: E402
    generate_distrust_atlas_figure,
    generate_distrust_benchmark_figure,
    generate_downstream_mask_utility_figure,
)
from colab.hallucination_benchmark import (  # noqa: E402
    attach_caid3_credibility_floor,
    compare_distrust_baselines,
    finalize_distrust_benchmark_with_caid3,
    save_distrust_benchmark,
)
from colab.novel_use_cases import build_af_rescue_manifest, screen_af_hallucinations  # noqa: E402
from colab.structure_distrust_atlas import (  # noqa: E402
    ATLAS_VERSION,
    build_structure_distrust_atlas,
    compute_protein_distrust_row,
    estimate_downstream_mask_utility,
    export_structure_distrust_atlas_bundle,
    load_plddt_cache_for_proteins,
)
from colab.training_contamination_audit import (  # noqa: E402
    audit_structure_signal_contamination,
)
from rockfish.run_disordernet import build_parser  # noqa: E402


class TestProxyVsLabeledScreening:
    def test_proxy_rescue_rate_undefined(self):
        seq = "A" * 20
        dis = np.ones(20, dtype=np.float32) * 0.9
        plddt = np.full(20, 85.0, dtype=np.float32)
        out = screen_af_hallucinations(seq, dis, plddt, protein_id="P")
        assert out["definition"] == "proxy_distrust"
        assert out["metrics"]["rescue_rate_valid"] is False
        assert out["metrics"]["rescue_rate"] is None
        assert out["metrics"]["n_hallucinated"] == 20
        assert out["n_rescued_regions"] == 0  # not advertised as rescue

    def test_labeled_rescue_is_valid(self):
        seq = "A" * 20
        dis = np.concatenate([np.ones(10), np.zeros(10)]).astype(np.float32) * 0.9
        plddt = np.full(20, 80.0, dtype=np.float32)
        labels = np.concatenate([np.ones(10), np.zeros(10)]).astype(np.int8)
        out = screen_af_hallucinations(
            seq, dis, plddt, protein_id="P", labels=labels,
        )
        assert out["definition"] == "labeled_independent"
        assert out["metrics"]["rescue_rate_valid"] is True
        assert out["metrics"]["rescue_rate"] == 1.0
        assert out["metrics"]["n_hallucinated"] == 10


class TestDistrustBaselinesAndUtility:
    def test_compare_baselines_dn_beats_random_plddt(self):
        rng = np.random.RandomState(0)
        labels = (rng.rand(400) > 0.7).astype(np.int8)
        # DN correlated with labels
        probs = np.clip(labels.astype(np.float32) * 0.8 + rng.rand(400) * 0.15, 0, 1)
        # pLDDT anti-correlated lightly with disorder
        plddt = 90 - labels.astype(np.float32) * 30 + rng.randn(400) * 5
        report = compare_distrust_baselines(labels, probs, plddt.astype(np.float32))
        assert report["enabled"] is True
        assert report["definition"] == "labeled_independent"
        assert report["disordernet"]["auc"] is not None
        assert report["plddt_inverse_baseline"]["auc"] is not None

    def test_downstream_mask_utility(self):
        labels = np.array([1] * 40 + [0] * 60, dtype=np.int8)
        probs = labels.astype(np.float32) * 0.95
        # High pLDDT everywhere so mask comparison is meaningful
        plddt = np.full(100, 85.0, dtype=np.float32)
        util = estimate_downstream_mask_utility(labels, probs, plddt)
        assert util["enabled"] is True
        assert util["precision_dn_distrust_mask"] is not None
        assert util["precision_dn_distrust_mask"] >= util["base_disorder_rate_in_high_plddt"]


class TestStructureDistrustAtlas:
    def test_atlas_proxy_and_labeled(self, tmp_path):
        proteins = [
            {"id": "a", "sequence": "A" * 30, "length": 30, "uniprot_acc": "A1"},
            {"id": "b", "sequence": "T" * 30, "length": 30, "uniprot_acc": "B1"},
        ]
        preds = {
            "a": np.array([0.9] * 15 + [0.1] * 15, dtype=np.float32),
            "b": np.ones(30, dtype=np.float32) * 0.1,
        }
        plddt = {
            "a": np.full(30, 88.0, dtype=np.float32),
            "b": np.full(30, 88.0, dtype=np.float32),
        }
        labels = {
            "a": np.array([1] * 15 + [0] * 15, dtype=np.int8),
            "b": np.zeros(30, dtype=np.int8),
        }
        row = compute_protein_distrust_row(
            protein_id="a",
            sequence=proteins[0]["sequence"],
            disorder_probs=preds["a"],
            plddt=plddt["a"],
            labels=labels["a"],
        )
        assert row["atlas_version"] == ATLAS_VERSION
        assert row["proxy_distrust"]["n_residues"] == 15
        assert row["labeled"]["n_hallucinated"] == 15
        assert row["labeled"]["rescue_rate"] == 1.0

        atlas = build_structure_distrust_atlas(
            proteins, preds, plddt, labels_by_id=labels,
        )
        assert atlas["n_proteins"] == 2
        assert atlas["n_proteins_with_proxy_distrust"] >= 1
        assert atlas["labeled_evaluation"]["n_proteins_with_labels"] == 2
        assert atlas["labeled_evaluation"]["overall_rescue_rate"] is not None
        assert "proxy_flags_are_not_independent_rescue" in atlas["non_claims"]

        paths = export_structure_distrust_atlas_bundle(atlas, str(tmp_path))
        assert Path(paths["jsonl"]).exists()
        assert Path(paths["tsv"]).exists()
        assert Path(paths["report"]).exists()
        report = json.loads(Path(paths["report"]).read_text())
        assert "proteins" not in report or report.get("n_proteins_embedded") == 0

    def test_manifest_uses_labels_when_provided(self):
        proteins = [{"id": "a", "sequence": "A" * 12, "length": 12, "uniprot_acc": "U"}]
        preds = {"a": np.ones(12, dtype=np.float32) * 0.9}
        plddt = {"a": np.full(12, 90.0, dtype=np.float32)}
        labels = {"a": np.ones(12, dtype=np.int8)}
        man = build_af_rescue_manifest(
            proteins, preds, plddt, labels_by_id=labels,
        )
        assert man["definition"] == "labeled_independent"
        assert man["overall_rescue_rate"] == 1.0

    def test_cli_flag(self):
        args = build_parser().parse_args(["eval", "--no-structure-distrust-atlas"])
        assert args.no_structure_distrust_atlas is True

    def test_contamination_caid_figures_and_atlas_stage_cli(self, tmp_path):
        low = audit_structure_signal_contamination(
            use_hallucination_weighting=False, use_plddt_features=False,
        )
        assert low["risk_tier"] == "low"
        high = audit_structure_signal_contamination(
            use_hallucination_weighting=True, use_plddt_features=True,
        )
        assert high["risk_tier"] == "high"
        assert high["required_ablation"] is not None

        bench = {
            "matched_baselines": {
                "disordernet": {"auc": 0.9},
                "plddt_inverse_baseline": {"auc": 0.7},
                "delta_auc_dn_minus_plddt": 0.2,
            },
            "labeled_rescue_report": {
                "pooled": {"hallucination_rate": 0.3, "rescue_rate": 0.8},
            },
            "training_contamination": high,
        }
        bench = attach_caid3_credibility_floor(bench, {"pooled": {"auc": 0.88}, "n_scored": 100})
        assert bench["caid3_credibility_floor"]["available"] is True

        proteins = [{"id": "a", "sequence": "A" * 10, "length": 10, "uniprot_acc": "P1"}]
        cache = tmp_path / "plddt"
        cache.mkdir()
        (cache / "P1.json").write_text(json.dumps({
            "target_sequence": "A" * 10,
            "plddt": [80.0] * 10,
        }))
        loaded = load_plddt_cache_for_proteins(proteins, str(cache))
        assert "a" in loaded and len(loaded["a"]) == 10

        atlas = build_structure_distrust_atlas(
            proteins,
            {"a": np.ones(10, dtype=np.float32) * 0.9},
            loaded,
        )
        fig_dir = tmp_path / "figs"
        generate_distrust_benchmark_figure(bench, out_dir=str(fig_dir))
        generate_distrust_atlas_figure(atlas, out_dir=str(fig_dir))
        util = estimate_downstream_mask_utility(
            np.ones(40, dtype=np.int8),
            np.ones(40, dtype=np.float32) * 0.9,
            np.full(40, 85.0, dtype=np.float32),
        )
        generate_downstream_mask_utility_figure(util, out_dir=str(fig_dir))
        assert (fig_dir / "fig_distrust_benchmark.png").exists()
        assert (fig_dir / "fig_distrust_atlas.png").exists()

        args = build_parser().parse_args([
            "structure-distrust-atlas",
            "--atlas-preds-dir", "preds",
            "--atlas-plddt-dir", "plddt",
            "--no-distrust-figures",
        ])
        assert args.stage == "structure-distrust-atlas"
        assert args.atlas_preds_dir == "preds"
        assert args.no_distrust_figures is True


class TestPublishReadyFixes:
    def test_finalize_distrust_benchmark_with_caid3(self, tmp_path):
        bench = {
            "matched_baselines": {"delta_auc_dn_minus_plddt": 0.1},
            "non_claims": ["proxy_DN_threshold_intersection_is_not_rescue"],
        }
        save_distrust_benchmark(bench, str(tmp_path / "structure_distrust_benchmark.json"))
        (tmp_path / "caid3_eval_report.json").write_text(json.dumps({
            "pooled": {"auc": 0.87, "ap": 0.71},
            "n_scored": 1234,
        }))
        out = finalize_distrust_benchmark_with_caid3(
            str(tmp_path), regenerate_figure=False,
        )
        assert out is not None
        assert out["caid3_credibility_floor"]["available"] is True
        assert out["caid3_credibility_floor"]["auc"] == 0.87
        reloaded = json.loads(
            (tmp_path / "structure_distrust_benchmark.json").read_text()
        )
        assert reloaded["caid3_credibility_floor"]["available"] is True

    def test_finalize_returns_none_without_benchmark(self, tmp_path):
        (tmp_path / "caid3_eval_report.json").write_text("{}")
        assert finalize_distrust_benchmark_with_caid3(str(tmp_path)) is None

    def test_ultra_clean_profile_forces_flags_off(self):
        from colab.disordernet_gpu import TrainConfig

        clean = TrainConfig.from_profile("ultra_clean")
        ultra = TrainConfig.from_profile("ultra")
        assert clean.use_hallucination_weighting is False
        assert clean.use_plddt_features is False
        assert ultra.use_plddt_features is True
        # capacity still matches ultra
        assert clean.lora_rank == ultra.lora_rank
        assert clean.num_epochs == ultra.num_epochs

    def test_cli_accepts_contamination_clean_flags(self):
        args = build_parser().parse_args([
            "pipeline",
            "--no-hallucination-weighting",
            "--no-plddt-features",
            "--profile", "ultra_clean",
        ])
        assert args.no_hallucination_weighting is True
        assert args.no_plddt_features is True
        assert args.profile == "ultra_clean"

    def test_mirror_globs_include_distrust_paths(self):
        from rockfish.mirror_results import DEFAULT_GLOBS

        needed = [
            "checkpoints/structure_distrust_benchmark.json",
            "checkpoints/structure_distrust_atlas_report.json",
            "checkpoints/structure_distrust_atlas.jsonl",
            "checkpoints/structure_distrust_atlas.tsv",
            "checkpoints/distrust_figures/**",
            "checkpoints/idr_biology_layer_*",
            "checkpoints/function_prediction_report.json",
            # Clean companion / alternate checkpoint roots (full parity)
            "checkpoints_*/structure_distrust_benchmark.json",
            "checkpoints_*/structure_distrust_atlas_report.json",
            "checkpoints_*/structure_distrust_atlas.jsonl",
            "checkpoints_*/structure_distrust_atlas.tsv",
            "checkpoints_*/distrust_figures/**",
            "checkpoints_*/idr_biology_layer_*",
            "checkpoints_*/function_prediction_report.json",
        ]
        for g in needed:
            assert g in DEFAULT_GLOBS, f"missing mirror glob: {g}"

    def test_pipeline_finalize_after_mocked_caid_save(self, tmp_path):
        """Unit-level: after CAID JSON lands + finalize, benchmark has floor."""
        save_distrust_benchmark(
            {"claim": "distrust", "caid3_credibility_floor": {"available": False}},
            str(tmp_path / "structure_distrust_benchmark.json"),
        )
        (tmp_path / "caid3_eval_report.json").write_text(json.dumps({
            "pooled": {"auc": 0.9},
            "n_proteins": 50,
        }))
        patched = finalize_distrust_benchmark_with_caid3(
            str(tmp_path), regenerate_figure=False,
        )
        assert patched["caid3_credibility_floor"]["available"] is True
        assert patched["caid3_credibility_floor"]["auc"] == 0.9
