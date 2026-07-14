"""Tests for rockfish/package_publish_results.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from rockfish.package_publish_results import (  # noqa: E402
    assemble_publish_package,
    build_default_runs,
    extract_run_summary,
)


def _write_ckpt(ckpt: Path, *, auc: float, delta: float, clean: bool = False) -> None:
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "sota_postprocess_report.json").write_text(json.dumps({"pooled_auc": auc}))
    (ckpt / "caid3_eval_report.json").write_text(
        json.dumps({"pooled": {"auc": auc - 0.01}, "n_scored": 100})
    )
    (ckpt / "structure_distrust_benchmark.json").write_text(
        json.dumps(
            {
                "matched_baselines": {
                    "delta_auc_dn_minus_plddt": delta,
                    "disordernet": {"auc": 0.9},
                    "plddt_inverse_baseline": {"auc": 0.9 - delta},
                },
                "labeled_rescue_report": {
                    "pooled": {"rescue_rate": 0.7, "hallucination_rate": 0.25},
                },
                "caid3_credibility_floor": {"available": True, "auc": auc - 0.01},
                "training_contamination": {
                    "risk_tier": "low" if clean else "high",
                },
                "downstream_mask_utility": {"enabled": True},
            }
        )
    )
    (ckpt / "distrust_figures").mkdir(exist_ok=True)
    (ckpt / "distrust_figures" / "fig_distrust_benchmark.png").write_bytes(b"png")


class TestPublishPackage:
    def test_build_default_runs_layout(self, tmp_path):
        runs = build_default_runs(tmp_path)
        assert [r["label"] for r in runs] == [
            "ultra_650M",
            "ultra_clean_650M",
            "ultra3b",
        ]
        slim = build_default_runs(tmp_path, include_clean=False, include_3b=False)
        assert [r["label"] for r in slim] == ["ultra_650M"]

    def test_assemble_package(self, tmp_path):
        root = tmp_path / "bundle"
        _write_ckpt(root / "ultra_650M" / "checkpoints", auc=0.90, delta=0.12)
        _write_ckpt(
            root / "ultra_clean_650M" / "checkpoints_ultra_clean",
            auc=0.88,
            delta=0.08,
            clean=True,
        )
        _write_ckpt(root / "ultra3b" / "checkpoints", auc=0.92, delta=0.15)

        pkg = tmp_path / "publish_package"
        manifest = assemble_publish_package(
            pkg,
            build_default_runs(root),
            package_id="test_pkg",
        )
        assert (pkg / "MANIFEST.json").is_file()
        assert (pkg / "comparison.json").is_file()
        assert (pkg / "PACKAGE_README.md").is_file()
        assert (pkg / "ultra_650M" / "structure_distrust_benchmark.json").is_file()
        assert (pkg / "ultra_clean_650M" / "caid3_eval_report.json").is_file()
        assert (pkg / "ultra3b" / "distrust_figures" / "fig_distrust_benchmark.png").is_file()

        assert manifest["package_id"] == "test_pkg"
        assert len(manifest["runs"]) == 3
        by_label = {r["label"]: r for r in manifest["runs"]}
        assert by_label["ultra_650M"]["pooled_auc"] == 0.90
        assert by_label["ultra_clean_650M"]["contamination_risk_tier"] == "low"
        assert by_label["ultra3b"]["delta_auc_dn_minus_plddt"] == 0.15

    def test_extract_summary_missing_dir(self, tmp_path):
        s = extract_run_summary(
            tmp_path / "missing",
            label="x",
            profile="ultra",
            backbone="650M",
        )
        assert s["pooled_auc"] is None
        assert s["artifacts_present"]["structure_distrust_benchmark.json"] is False
