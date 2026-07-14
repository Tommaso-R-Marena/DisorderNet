#!/usr/bin/env python3
"""Parallel mirror of checkpoint/report files to durable storage."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from rockfish.utils import ensure_repo_on_path, mirror_glob_patterns  # noqa: E402

ensure_repo_on_path()

from colab.async_io import mirror_files_parallel  # noqa: E402

# Back-compat alias for tests / callers that import DEFAULT_GLOBS
DEFAULT_GLOBS = mirror_glob_patterns()


def collect_mirror_paths(cwd: str | Path = ".") -> list[str]:
    """Resolve unique existing files matching DEFAULT_GLOBS under cwd."""
    cwd = Path(cwd)
    prev = os.getcwd()
    os.chdir(cwd)
    try:
        paths: list[str] = []
        for pattern in mirror_glob_patterns():
            paths.extend(glob.glob(pattern, recursive=True))
        seen: set[str] = set()
        uniq: list[str] = []
        for path in paths:
            if path not in seen and os.path.isfile(path):
                seen.add(path)
                uniq.append(path)
        return uniq
    finally:
        os.chdir(prev)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Parallel mirror DisorderNet results")
    p.add_argument("--dest", required=True, help="Destination directory")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cwd", default=".", help="Working directory with checkpoints/")
    p.add_argument(
        "--require-min-files",
        type=int,
        default=0,
        help="Exit 1 if fewer than N files matched (fail-loud empty mirrors)",
    )
    args = p.parse_args(argv)

    uniq = collect_mirror_paths(args.cwd)
    if args.require_min_files and len(uniq) < args.require_min_files:
        print(
            f"ERROR: mirrored sources {len(uniq)} < --require-min-files "
            f"{args.require_min_files} (cwd={args.cwd})",
            file=sys.stderr,
        )
        return 1

    os.chdir(args.cwd)
    copied = mirror_files_parallel(uniq, args.dest, max_workers=args.workers)
    print(f"Mirrored {len(copied)}/{len(uniq)} files → {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
