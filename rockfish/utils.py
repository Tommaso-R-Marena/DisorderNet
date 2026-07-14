"""Shared Rockfish helpers: paths, env defaults, and Slurm submit utilities."""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = Path(__file__).resolve().parent / "slurm"
PHASE_SBATCH = SLURM_DIR / "pipeline_phase.sbatch"
PACKAGE_SBATCH = SLURM_DIR / "package_results.sbatch"

# Standard bundle subdirs
LABEL_ULTRA_650M = "ultra_650M"
LABEL_CLEAN_650M = "ultra_clean_650M"
LABEL_ULTRA_3B = "ultra3b"
LABEL_CLEAN_3B = "ultra_clean_3B"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
    ]
    if os.environ.get("DISORDERNET_BOLTZ_ROOT"):
        keys.append("DISORDERNET_BOLTZ_ROOT")
    if os.environ.get("BOLTZ_CACHE"):
        keys.append("BOLTZ_CACHE")
    keys.extend(extra)
    # Unique preserve order
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return ",".join(out)


def run_specs_650m(root: Path | str, *, include_clean: bool = True) -> list[dict]:
    root = Path(root)
    runs = [
        {
            "label": LABEL_ULTRA_650M,
            "profile": "ultra",
            "backbone": "650M",
            "checkpoint_dir": root / LABEL_ULTRA_650M / "checkpoints",
            "checkpoint_subdir": "checkpoints",
            "clean": False,
        }
    ]
    if include_clean:
        runs.append(
            {
                "label": LABEL_CLEAN_650M,
                "profile": "ultra_clean",
                "backbone": "650M",
                "checkpoint_dir": root / LABEL_CLEAN_650M / "checkpoints_ultra_clean",
                "checkpoint_subdir": "checkpoints_ultra_clean",
                "clean": True,
            }
        )
    return runs


def run_specs_3b(root: Path | str, *, include_clean: bool = True) -> list[dict]:
    """3B main + optional contamination-clean companion (ultra3b flags forced off)."""
    root = Path(root)
    runs = [
        {
            "label": LABEL_ULTRA_3B,
            "profile": "ultra3b",
            "backbone": "3B",
            "checkpoint_dir": root / LABEL_ULTRA_3B / "checkpoints",
            "checkpoint_subdir": "checkpoints",
            "clean": False,
        }
    ]
    if include_clean:
        runs.append(
            {
                "label": LABEL_CLEAN_3B,
                "profile": "ultra3b",  # same capacity; clean via CLI flags
                "backbone": "3B",
                "checkpoint_dir": root / LABEL_CLEAN_3B / "checkpoints_ultra_clean_3b",
                "checkpoint_subdir": "checkpoints_ultra_clean_3b",
                "clean": True,
            }
        )
    return runs


def ensure_bundle_dirs(root: Path, specs: Sequence[dict]) -> None:
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
) -> str:
    """
    Submit a Slurm script. Returns job id (or DRY_RUN placeholder).
    Raises RuntimeError if sbatch is missing (unless dry_run).
    """
    script = Path(script)
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

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    jid = (result.stdout or "").strip().split(";")[0].strip()
    if not jid:
        raise RuntimeError(f"sbatch returned empty job id. stderr={result.stderr!r}")
    return jid


def push_env(mapping: dict[str, str]) -> None:
    """Export mapping into os.environ (string values only)."""
    for k, v in mapping.items():
        if v is None:
            continue
        os.environ[k] = str(v)
