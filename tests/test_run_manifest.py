"""Tests for colab/run_manifest.py."""

from __future__ import annotations

from colab.disordernet_gpu import TrainConfig
from colab.run_manifest import (
    build_run_manifest,
    colab_drive_mirror_basenames,
    mirror_files_to_drive,
    save_run_manifest,
)
from rockfish.utils import ARTIFACT_FILES, REQUIRED_ARTIFACTS_STRICT


class TestRunManifest:
    def test_build_and_save(self, tmp_path):
        cfg = TrainConfig(seed=42, n_folds=5)
        proteins = [{"id": f"P{i}", "length": 40} for i in range(4)]
        manifest = build_run_manifest(
            cfg,
            proteins,
            cv_summary={"pooled_auc": 0.82, "fold_aucs": [0.8, 0.84]},
            disprot_meta={"n_entries": 1000, "content_sha256": "abc"},
            extra={"run_timestamp": "test"},
        )
        assert manifest["n_proteins"] == 4
        assert manifest["cv_summary"]["pooled_auc"] == 0.82
        assert manifest["proteins_fingerprint"]
        assert "git_revision" in manifest
        path = save_run_manifest(manifest, str(tmp_path / "run_manifest.json"))
        assert (tmp_path / "run_manifest.json").exists()
        assert path.endswith("run_manifest.json")

    def test_colab_mirror_basenames_align_with_rockfish(self):
        names = colab_drive_mirror_basenames()
        assert "sota_postprocess_report.json" in names
        assert "structure_distrust_benchmark.json" in names
        for req in REQUIRED_ARTIFACTS_STRICT:
            assert req in names
        assert set(names) == set(ARTIFACT_FILES)

    def test_mirror_skips_missing_drive(self, tmp_path):
        f = tmp_path / "cv_results.json"
        f.write_text("{}")
        assert mirror_files_to_drive([str(f)], "/nonexistent/drive") == []

    def test_mirror_copies_to_drive(self, tmp_path):
        drive = tmp_path / "drive"
        drive.mkdir()
        src = tmp_path / "cv_results.json"
        src.write_text('{"ok": true}')
        copied = mirror_files_to_drive([str(src)], str(drive), run_subdir="run1")
        assert len(copied) == 1
        assert copied[0].endswith("cv_results.json")
        assert (drive / "results" / "run1" / "cv_results.json").exists()
