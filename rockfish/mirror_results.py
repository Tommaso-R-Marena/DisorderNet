#!/usr/bin/env python3
"""Parallel mirror of checkpoint/report files to durable storage."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from colab.async_io import mirror_files_parallel  # noqa: E402

DEFAULT_GLOBS = [
    "checkpoints/cv_progress.json",
    "checkpoints/cv_summary.json",
    "checkpoints/run_manifest.json",
    "checkpoints/sota_postprocess_report.json",
    "checkpoints/gpu_v6_ensemble_report.json",
    "checkpoints/sota_stack_report.json",
    "checkpoints/eval_summary.json",
    "checkpoints/caid3_eval_report.json",
    "checkpoints/phase3_integrated_report.json",
    "checkpoints/statistical_validation_report.json",
    "checkpoints/af_rescue_report.json",
    "checkpoints/af3_rescue_report.json",
    "checkpoints/af2_af3_comparison.json",
    "checkpoints/inference_fusion_report.json",
    "checkpoints/af_rescue_manifest.json",
    "checkpoints/multi_seed_blend_report.json",
    "checkpoints/quick_screen_*.json",
    "checkpoints/fold_*_compact.pt",
]


def main() -> int:
    p = argparse.ArgumentParser(description="Parallel mirror DisorderNet results")
    p.add_argument("--dest", required=True, help="Destination directory")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cwd", default=".", help="Working directory with checkpoints/")
    args = p.parse_args()

    os.chdir(args.cwd)
    paths: list[str] = []
    for pattern in DEFAULT_GLOBS:
        paths.extend(glob.glob(pattern))
    # unique preserve order
    seen = set()
    uniq = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            uniq.append(path)

    copied = mirror_files_parallel(uniq, args.dest, max_workers=args.workers)
    print(f"Mirrored {len(copied)}/{len(uniq)} files → {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
