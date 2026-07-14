"""Tests for rockfish/utils.py and publish_submit CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from rockfish.publish_submit import build_parser, main as publish_main  # noqa: E402
from rockfish.run_disordernet import build_parser as rockfish_parser  # noqa: E402
from rockfish.utils import (  # noqa: E402
    default_publish_root,
    require_account,
    run_specs_3b,
    run_specs_650m,
    sbatch_export_keys,
)


class TestUtils:
    def test_run_specs_650m(self, tmp_path):
        specs = run_specs_650m(tmp_path, include_clean=True)
        assert [s["label"] for s in specs] == ["ultra_650M", "ultra_clean_650M"]
        assert specs[1]["clean"] is True
        assert specs[1]["checkpoint_subdir"] == "checkpoints_ultra_clean"

    def test_run_specs_3b(self, tmp_path):
        specs = run_specs_3b(tmp_path, include_clean=True)
        assert [s["label"] for s in specs] == ["ultra3b", "ultra_clean_3B"]
        assert specs[1]["profile"] == "ultra3b"
        assert specs[1]["clean"] is True

    def test_require_account(self, monkeypatch):
        monkeypatch.delenv("DISORDERNET_ACCOUNT", raising=False)
        assert require_account(None) == "sfried3"
        assert require_account("sfried3") == "sfried3"
        assert require_account("my_gpu") == "my_gpu"
        try:
            require_account("CHANGE_ME_gpu")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_export_keys_include_all(self):
        keys = sbatch_export_keys()
        assert keys.startswith("ALL,")
        assert "PROFILE" in keys
        assert "BUNDLE_KIND" in keys

    def test_default_publish_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DISORDERNET_RESULTS", str(tmp_path))
        root = default_publish_root("650m", stamp="X")
        assert root == tmp_path / "publish_650m_X"


class TestPublishCLI:
    def test_parser_subcommands(self):
        p = build_parser()
        a = p.parse_args(["submit-650m", "--account", "x", "--dry-run"])
        assert a.command == "submit-650m"
        assert a.dry_run is True
        b = p.parse_args(["submit-3b", "--account", "x", "--no-clean"])
        assert b.command == "submit-3b"
        assert b.no_clean is True
        c = p.parse_args(
            ["package", "--root-workdir", "/tmp/x", "--kind", "3b"],
        )
        assert c.command == "package"
        assert c.kind == "3b"

    def test_dry_run_650m(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DISORDERNET_RESULTS", str(tmp_path / "results"))
        rc = publish_main(
            [
                "submit-650m",
                "--account", "test_gpu",
                "--dry-run",
                "--stamp", "TESTSTAMP",
                "--root-workdir", str(tmp_path / "bundle"),
            ]
        )
        assert rc == 0
        summary = json.loads((tmp_path / "bundle" / "submit_summary.json").read_text())
        assert summary["kind"] == "650m"
        assert "ultra_650M" in summary["job_ids"]
        assert "ultra_clean_650M" in summary["job_ids"]
        assert "package" in summary["job_ids"]
        assert summary["dry_run"] is True

    def test_dry_run_3b_no_clean(self, tmp_path):
        rc = publish_main(
            [
                "submit-3b",
                "--account", "test_gpu",
                "--dry-run",
                "--no-clean",
                "--root-workdir", str(tmp_path / "b3"),
            ]
        )
        assert rc == 0
        summary = json.loads((tmp_path / "b3" / "submit_summary.json").read_text())
        assert list(summary["job_ids"]) == ["ultra3b", "package"]

    def test_package_kind_650m(self, tmp_path):
        from rockfish.package_publish_results import assemble_publish_package
        from rockfish.utils import run_specs_650m

        root = tmp_path / "bundle"
        for spec in run_specs_650m(root, include_clean=True):
            ckpt = Path(spec["checkpoint_dir"])
            ckpt.mkdir(parents=True)
            (ckpt / "sota_postprocess_report.json").write_text('{"pooled_auc": 0.9}')
            (ckpt / "structure_distrust_benchmark.json").write_text(
                '{"matched_baselines":{"delta_auc_dn_minus_plddt":0.1},'
                '"training_contamination":{"risk_tier":"low"},'
                '"caid3_credibility_floor":{"available":false}}'
            )
        pkg = tmp_path / "out"
        rc = publish_main(
            ["package", "--root-workdir", str(root), "--kind", "650m",
             "--package-dir", str(pkg)]
        )
        assert rc == 0
        assert (pkg / "MANIFEST.json").is_file()
        assert (pkg / "ultra_650M" / "sota_postprocess_report.json").is_file()

    def test_rockfish_runner_publish_stages(self):
        p = rockfish_parser()
        a = p.parse_args(["publish-650m", "--account", "x", "--dry-run"])
        assert a.stage == "publish-650m"
        assert a.dry_run is True
        b = p.parse_args(
            ["package-publish", "--publish-root", "/tmp/x", "--bundle-kind", "3b"],
        )
        assert b.stage == "package-publish"
        assert b.bundle_kind == "3b"
