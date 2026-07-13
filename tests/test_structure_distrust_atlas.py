"""Tests for structure distrust atlas / labeled hallucination protocol (paper pillar 1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from colab.hallucination_benchmark import (  # noqa: E402
    compare_distrust_baselines,
)
from colab.novel_use_cases import build_af_rescue_manifest, screen_af_hallucinations  # noqa: E402
from colab.structure_distrust_atlas import (  # noqa: E402
    ATLAS_VERSION,
    build_structure_distrust_atlas,
    compute_protein_distrust_row,
    estimate_downstream_mask_utility,
    export_structure_distrust_atlas_bundle,
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
