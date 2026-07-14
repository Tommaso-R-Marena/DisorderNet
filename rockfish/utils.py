"""Shared Rockfish helpers: artifacts, run specs, paths, and Slurm submit utilities."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = Path(__file__).resolve().parent / "slurm"
PHASE_SBATCH = SLURM_DIR / "pipeline_phase.sbatch"
PACKAGE_SBATCH = SLURM_DIR / "package_results.sbatch"

# Standard bundle subdirs
LABEL_ULTRA_650M = "ultra_650M"
LABEL_CLEAN_650M = "ultra_clean_650M"
LABEL_ULTRA_3B = "ultra3b"
LABEL_CLEAN_3B = "ultra_clean_3B"

# Exact files copied into publish packages / mirrored from checkpoint roots
ARTIFACT_FILES: tuple[str, ...] = (
    "cv_summary.json",
    "cv_progress.json",
    "run_manifest.json",
    "sota_postprocess_report.json",
    "sota_stack_report.json",
    "gpu_v6_ensemble_report.json",
    "eval_summary.json",
    "caid3_eval_report.json",
    "phase3_integrated_report.json",
    "statistical_validation_report.json",
    "af_rescue_report.json",
    "boltz_rescue_report.json",
    "af_rescue_manifest.json",
    "af3_rescue_report.json",
    "af2_af3_comparison.json",
    "inference_fusion_report.json",
    "multi_seed_blend_report.json",
    "structure_distrust_benchmark.json",
    "structure_distrust_atlas_report.json",
    "structure_distrust_atlas.jsonl",
    "structure_distrust_atlas.tsv",
    "function_prediction_report.json",
    "idr_biology_layer_report.json",
    "idr_biology_layer_report.md",
    "idr_biology_layer_report.html",
    "idr_biology_layer.jsonl",
)

# Report/figure globs relative to a checkpoint directory (for packages + mirrors)
ARTIFACT_REPORT_GLOBS: tuple[str, ...] = (
    "idr_biology_layer_*",
    "quick_screen_*.json",
    "distrust_figures/**",
)

# Large weight globs — mirrored for resume, not copied into publish packages
ARTIFACT_WEIGHT_GLOBS: tuple[str, ...] = (
    "fold_*_compact.pt",
)

# All basename-level globs (reports + weights)
ARTIFACT_GLOBS: tuple[str, ...] = ARTIFACT_REPORT_GLOBS + ARTIFACT_WEIGHT_GLOBS

# Required for --strict packaging (go/no-go inputs)
REQUIRED_ARTIFACTS_STRICT: tuple[str, ...] = (
    "sota_postprocess_report.json",
    "structure_distrust_benchmark.json",
)

# Surfaced in package run summaries (presence flags; not all required)
ARTIFACTS_PRESENT_KEYS: tuple[str, ...] = (
    "sota_postprocess_report.json",
    "structure_distrust_benchmark.json",
    "caid3_eval_report.json",
    "structure_distrust_atlas_report.json",
)


class RunSpec(TypedDict):
    label: str
    profile: str
    backbone: str
    checkpoint_dir: str
    checkpoint_subdir: str
    clean: bool


def ensure_repo_on_path(repo: Optional[Path] = None) -> Path:
    """Idempotently put the repo root on sys.path for script entrypoints."""
    root = Path(repo) if repo is not None else REPO_ROOT
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_revision(repo: Optional[Path] = None) -> Optional[str]:
    """Return short HEAD SHA, or None if git is unavailable / not a repo."""
    root = Path(repo) if repo is not None else REPO_ROOT
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return (out.stdout or "").strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def default_results_root() -> Path:
    return Path(os.environ.get("DISORDERNET_RESULTS", str(Path.home() / "disordernet_runs")))


def default_publish_root(kind: str, stamp: Optional[str] = None) -> Path:
    """kind: '650m' | '3b'"""
    stamp = stamp or utc_stamp()
    return default_results_root() / f"publish_{kind}_{stamp}"


def require_account(account: Optional[str] = None) -> str:
    acc = account or os.environ.get("DISORDERNET_ACCOUNT", "sfried3")
    if not acc or acc in ("CHANGE_ME_gpu", "CHANGE_ME"):
        raise ValueError(
            "Set DISORDERNET_ACCOUNT (or pass --account) to your Rockfish account "
            "(default: sfried3)"
        )
    return acc


def env_defaults() -> dict[str, str]:
    """Common defaults for publish submitters."""
    repo = os.environ.get("DISORDERNET_REPO", str(REPO_ROOT))
    return {
        "DISORDERNET_REPO": repo,
        "DISORDERNET_VENV": os.environ.get(
            "DISORDERNET_VENV", str(Path.home() / "venvs" / "disordernet")
        ),
        "DISORDERNET_RESULTS": str(default_results_root()),
        "DISORDERNET_PARTITION": os.environ.get("DISORDERNET_PARTITION", "a100"),
        "DISORDERNET_CPU_ACCOUNT": os.environ.get(
            "DISORDERNET_CPU_ACCOUNT",
            os.environ.get("DISORDERNET_ACCOUNT", "sfried3"),
        ),
        "STAGE": "pipeline",
        "RUN_CAID3": os.environ.get("RUN_CAID3", "1"),
        "PREFETCH_AF": os.environ.get("PREFETCH_AF", "1"),
        "BOLTZ_MODE": os.environ.get("BOLTZ_MODE", "ingest"),
        "STRUCTURE_BACKEND": os.environ.get("STRUCTURE_BACKEND", "boltz"),
        "SEED": os.environ.get("SEED", "42"),
        "NUM_WORKERS": os.environ.get("NUM_WORKERS", "4"),
    }


def mirror_glob_patterns() -> list[str]:
    """
    Glob patterns relative to a workdir that contains checkpoint roots.
    Covers both ``checkpoints/`` and ``checkpoints_*/`` (clean companions).
    """
    roots = ("checkpoints", "checkpoints_*")
    patterns: list[str] = []
    for root in roots:
        for name in ARTIFACT_FILES:
            patterns.append(f"{root}/{name}")
        for g in ARTIFACT_GLOBS:
            patterns.append(f"{root}/{g}")
    # preserve order, unique
    seen: set[str] = set()
    out: list[str] = []
    for p in patterns:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def sbatch_export_keys(extra: Sequence[str] = ()) -> str:
    keys = [
        "ALL",
        "DISORDERNET_ACCOUNT",
        "DISORDERNET_REPO",
        "DISORDERNET_VENV",
        "DISORDERNET_RESULTS",
        "DISORDERNET_WORKDIR",
        "PROFILE",
        "BACKBONE",
        "CHECKPOINT_SUBDIR",
        "STAGE",
        "RUN_CAID3",
        "PREFETCH_AF",
        "BOLTZ_MODE",
        "STRUCTURE_BACKEND",
        "RUN_NO_HALLUC_WEIGHT",
        "RUN_NO_PLDDT_FEATURES",
        "BUNDLE_KIND",
        "DISORDERNET_PUBLISH_ROOT",
        "DISORDERNET_PACKAGE_DIR",
        "PACKAGE_ID",
        "INCLUDE_CLEAN",
        "SEED",
        "NUM_WORKERS",
        "PACKAGE_STRICT",
    ]
    if os.environ.get("DISORDERNET_BOLTZ_ROOT"):
        keys.append("DISORDERNET_BOLTZ_ROOT")
    if os.environ.get("BOLTZ_CACHE"):
        keys.append("BOLTZ_CACHE")
    keys.extend(extra)
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return ",".join(out)


def _spec(
    *,
    label: str,
    profile: str,
    backbone: str,
    root: Path,
    checkpoint_subdir: str,
    clean: bool,
) -> RunSpec:
    return {
        "label": label,
        "profile": profile,
        "backbone": backbone,
        "checkpoint_dir": str(root / label / checkpoint_subdir),
        "checkpoint_subdir": checkpoint_subdir,
        "clean": clean,
    }


def run_specs_650m(root: Path | str, *, include_clean: bool = True) -> list[RunSpec]:
    root_p = Path(root)
    runs: list[RunSpec] = [
        _spec(
            label=LABEL_ULTRA_650M,
            profile="ultra",
            backbone="650M",
            root=root_p,
            checkpoint_subdir="checkpoints",
            clean=False,
        )
    ]
    if include_clean:
        runs.append(
            _spec(
                label=LABEL_CLEAN_650M,
                profile="ultra_clean",
                backbone="650M",
                root=root_p,
                checkpoint_subdir="checkpoints_ultra_clean",
                clean=True,
            )
        )
    return runs


def run_specs_3b(root: Path | str, *, include_clean: bool = True) -> list[RunSpec]:
    """3B main + optional contamination-clean companion (clean via CLI flags)."""
    root_p = Path(root)
    runs: list[RunSpec] = [
        _spec(
            label=LABEL_ULTRA_3B,
            profile="ultra3b",
            backbone="3B",
            root=root_p,
            checkpoint_subdir="checkpoints",
            clean=False,
        )
    ]
    if include_clean:
        runs.append(
            _spec(
                label=LABEL_CLEAN_3B,
                profile="ultra3b",
                backbone="3B",
                root=root_p,
                checkpoint_subdir="checkpoints_ultra_clean_3b",
                clean=True,
            )
        )
    return runs


def run_specs_for_kind(
    root: Path | str,
    kind: str,
    *,
    include_clean: bool = True,
) -> list[RunSpec]:
    if kind == "650m":
        return run_specs_650m(root, include_clean=include_clean)
    if kind == "3b":
        return run_specs_3b(root, include_clean=include_clean)
    raise ValueError(f"Unknown kind {kind!r}; expected '650m' or '3b'")


def ensure_bundle_dirs(root: Path, specs: Sequence[RunSpec]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        (root / spec["label"]).mkdir(parents=True, exist_ok=True)
    (root / "publish_package").mkdir(parents=True, exist_ok=True)


def which_sbatch() -> Optional[str]:
    return shutil.which("sbatch")


def submit_sbatch(
    script: Path | str,
    *,
    account: str,
    job_name: str,
    export: str,
    partition: Optional[str] = None,
    dependency: Optional[str] = None,
    dry_run: bool = False,
    extra_args: Sequence[str] = (),
    env: Optional[dict[str, str]] = None,
) -> str:
    """
    Submit a Slurm script. Returns job id (or DRY_RUN placeholder).
    Raises RuntimeError / CalledProcessError on failure.
    """
    script = Path(script)
    if not script.is_file() and not dry_run:
        raise RuntimeError(f"sbatch script not found: {script}")

    cmd = [
        "sbatch",
        "--parsable",
        f"--account={account}",
        f"--job-name={job_name}",
        f"--export={export}",
    ]
    if partition:
        cmd.append(f"--partition={partition}")
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.extend(extra_args)
    cmd.append(str(script))

    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return f"DRYRUN_{job_name}"

    if which_sbatch() is None:
        raise RuntimeError(
            "sbatch not found on PATH. Run on a Rockfish login node, or pass --dry-run."
        )

    run_env = os.environ.copy()
    if env:
        run_env.update({k: str(v) for k, v in env.items() if v is not None})

    result = subprocess.run(
        cmd, check=True, capture_output=True, text=True, env=run_env,
    )
    jid = (result.stdout or "").strip().split(";")[0].strip()
    if not jid:
        raise RuntimeError(f"sbatch returned empty job id. stderr={result.stderr!r}")
    return jid


def push_env(mapping: dict[str, str]) -> dict[str, str]:
    """
    Export mapping into os.environ (string values only).
    Returns the applied mapping (for provenance / tests).
    Prefer passing ``env=`` to submit_sbatch when possible.
    """
    applied: dict[str, str] = {}
    for k, v in mapping.items():
        if v is None:
            continue
        s = str(v)
        os.environ[k] = s
        applied[k] = s
    return applied
