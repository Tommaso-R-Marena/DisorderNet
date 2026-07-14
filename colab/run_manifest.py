"""
Reproducibility manifest and Colab output helpers.

Captures git revision, dataset fingerprints, and config hashes so every
GPU run can be audited and compared across sessions.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Any, Optional

from colab.cv_splits import config_fingerprint, proteins_fingerprint


def get_git_revision(repo_path: str = ".") -> Optional[str]:
    """Return short git commit hash if available (shared with rockfish.utils)."""
    try:
        from rockfish.utils import git_revision

        return git_revision(repo_path)
    except ImportError:
        try:
            return subprocess.check_output(
                ["git", "-C", repo_path, "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None


def colab_drive_mirror_basenames() -> list[str]:
    """
    High-value Colab result filenames to mirror to Drive.

    Prefer rockfish.utils.ARTIFACT_FILES when available so Colab and Rockfish
    stay aligned; fall back to a minimal Colab set otherwise.
    """
    try:
        from rockfish.utils import ARTIFACT_FILES

        return list(ARTIFACT_FILES)
    except ImportError:
        return [
            "run_manifest.json",
            "cv_results.json",
            "cv_summary.json",
            "sota_postprocess_report.json",
            "caid_evaluation_report.json",
            "caid3_eval_report.json",
            "biological_utility_report.json",
            "statistical_validation_report.json",
            "phase3_integrated_report.json",
            "structure_distrust_benchmark.json",
            "structure_distrust_atlas_report.json",
            "function_prediction_report.json",
            "idr_biology_layer_report.json",
        ]


def build_run_manifest(
    cfg: Any,
    proteins: list,
    cv_summary: Optional[dict] = None,
    disprot_meta: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Assemble reproducibility metadata for a Colab GPU run."""
    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": get_git_revision(),
        "n_proteins": len(proteins),
        "n_residues": int(sum(p["length"] for p in proteins)),
        "proteins_fingerprint": proteins_fingerprint(proteins),
        "config_fingerprint": config_fingerprint(cfg),
        "seed": getattr(cfg, "seed", None),
        "n_folds": getattr(cfg, "n_folds", None),
        "quality_profile": getattr(cfg, "_profile_name", "custom"),
        "early_stop_mode": getattr(cfg, "early_stop_mode", "composite"),
        "checkpoint_dir": getattr(cfg, "checkpoint_dir", "checkpoints"),
        "data_cache": getattr(cfg, "data_cache", "disprot_raw.json"),
        "disprot_meta": disprot_meta,
    }
    if cv_summary:
        manifest["cv_summary"] = {
            "pooled_auc": cv_summary.get("pooled_auc"),
            "pooled_ap": cv_summary.get("pooled_ap"),
            "fold_aucs": cv_summary.get("fold_aucs"),
            "mean_auc": cv_summary.get("mean_auc"),
            "std_auc": cv_summary.get("std_auc"),
            "total_cv_hours": cv_summary.get("total_cv_hours"),
            "proteins_fingerprint": cv_summary.get("proteins_fingerprint"),
            "config_fingerprint": cv_summary.get("config_fingerprint"),
        }
    if extra:
        manifest.update(extra)
    return manifest


def save_run_manifest(manifest: dict, path: str = "run_manifest.json") -> str:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def mirror_files_to_drive(
    paths: list[str],
    drive_root: str,
    run_subdir: Optional[str] = None,
) -> list[str]:
    """
    Copy local report/checkpoint files to Google Drive.

    Returns list of destination paths that were written.
    """
    if not drive_root or not os.path.isdir(drive_root):
        return []

    dest_root = os.path.join(drive_root, "results", run_subdir or "")
    os.makedirs(dest_root, exist_ok=True)
    copied: list[str] = []

    for src in paths:
        if not src or not os.path.isfile(src):
            continue
        dest = os.path.join(dest_root, os.path.basename(src))
        shutil.copy2(src, dest)
        copied.append(dest)

    return copied
