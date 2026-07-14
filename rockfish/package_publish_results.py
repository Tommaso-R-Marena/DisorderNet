#!/usr/bin/env python3
"""Assemble an organized Rockfish publish package from run workdirs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from rockfish.utils import (  # noqa: E402
    ARTIFACT_FILES,
    ARTIFACT_REPORT_GLOBS,
    ARTIFACTS_PRESENT_KEYS,
    REQUIRED_ARTIFACTS_STRICT,
    RunSpec,
    ensure_repo_on_path,
    git_revision,
    run_specs_for_kind,
    utc_stamp,
)

ensure_repo_on_path()

# Back-compat: exact filenames only (figures/globs handled separately)
KEEP_NAMES: tuple[str, ...] = ARTIFACT_FILES


class PackageIncompleteError(RuntimeError):
    """Raised when --strict packaging is missing required artifacts."""


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not parse JSON {path}: {exc}", file=sys.stderr)
        return None


def _dig(d: Optional[dict], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def extract_run_summary(ckpt_dir: Path, *, label: str, profile: str, backbone: str) -> dict:
    """Pull headline metrics for side-by-side comparison."""
    post = _load_json(ckpt_dir / "sota_postprocess_report.json") or {}
    cv = _load_json(ckpt_dir / "cv_summary.json") or {}
    caid = _load_json(ckpt_dir / "caid3_eval_report.json") or {}
    bench = _load_json(ckpt_dir / "structure_distrust_benchmark.json") or {}
    util = bench.get("downstream_mask_utility") if isinstance(bench, dict) else None
    matched = bench.get("matched_baselines") if isinstance(bench, dict) else None
    rescue = _dig(bench, "labeled_rescue_report", "pooled")
    floor = bench.get("caid3_credibility_floor") if isinstance(bench, dict) else None
    contam = bench.get("training_contamination") if isinstance(bench, dict) else None

    pooled_auc = (
        post.get("pooled_auc")
        or post.get("auc")
        or _dig(post, "metrics", "auc")
        or cv.get("pooled_auc")
        or _dig(cv, "pooled", "auc")
    )
    caid_auc = _dig(caid, "pooled", "auc") or caid.get("auc") or _dig(floor, "auc")

    return {
        "label": label,
        "profile": profile,
        "backbone": backbone,
        "checkpoint_dir": str(ckpt_dir),
        "pooled_auc": pooled_auc,
        "caid3_auc": caid_auc,
        "delta_auc_dn_minus_plddt": (
            _dig(matched, "delta_auc_dn_minus_plddt") if isinstance(matched, dict) else None
        ),
        "labeled_rescue_rate": (
            _dig(rescue, "rescue_rate") if isinstance(rescue, dict) else None
        ),
        "hallucination_rate": (
            _dig(rescue, "hallucination_rate") if isinstance(rescue, dict) else None
        ),
        "mask_utility_enabled": bool(isinstance(util, dict) and util.get("enabled")),
        "caid3_floor_available": bool(isinstance(floor, dict) and floor.get("available")),
        "contamination_risk_tier": (
            _dig(contam, "risk_tier") if isinstance(contam, dict) else None
        ),
        "artifacts_present": {
            name: (ckpt_dir / name).is_file() for name in ARTIFACTS_PRESENT_KEYS
        },
    }


def missing_required_artifacts(ckpt_dir: Path) -> list[str]:
    return [name for name in REQUIRED_ARTIFACTS_STRICT if not (ckpt_dir / name).is_file()]


def _copy_tree_selective(src_ckpt: Path, dest_dir: Path) -> list[str]:
    """Copy known report artifacts (+ figures dir) into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    if not src_ckpt.is_dir():
        return copied

    for name in ARTIFACT_FILES:
        src = src_ckpt / name
        if src.is_file():
            shutil.copy2(src, dest_dir / name)
            copied.append(name)

    # Reports/figures only — exclude fold_*_compact.pt (mirrored, not packaged)
    for pattern in ARTIFACT_REPORT_GLOBS:
        if pattern.endswith("/**"):
            # directory tree (e.g. distrust_figures/**)
            dirname = pattern[:-3].rstrip("/")
            fig_src = src_ckpt / dirname
            if fig_src.is_dir():
                fig_dst = dest_dir / dirname
                if fig_dst.exists():
                    shutil.rmtree(fig_dst)
                shutil.copytree(fig_src, fig_dst)
                copied.append(f"{dirname}/")
            continue
        for src in src_ckpt.glob(pattern):
            if src.is_file():
                shutil.copy2(src, dest_dir / src.name)
                copied.append(src.name)

    return sorted(set(copied))


def write_package_readme(package_dir: Path, comparison: dict) -> Path:
    runs = comparison.get("runs") or []
    lines = [
        "# DisorderNet Rockfish publish package",
        "",
        f"Generated: {comparison.get('created_utc')}",
        f"Package id: `{comparison.get('package_id')}`",
        f"Git revision: `{comparison.get('git_revision')}`",
        "",
        "## Runs included",
        "",
        "| Label | Profile | Backbone | Pooled AUC | CAID3 AUC | ΔAUC vs inv-pLDDT | Rescue rate | Contamination |",
        "|-------|---------|----------|------------|-----------|-------------------|-------------|---------------|",
    ]
    for r in runs:
        lines.append(
            "| {label} | {profile} | {backbone} | {pooled_auc} | {caid3_auc} | "
            "{delta} | {rescue} | {risk} |".format(
                label=r.get("label"),
                profile=r.get("profile"),
                backbone=r.get("backbone"),
                pooled_auc=r.get("pooled_auc"),
                caid3_auc=r.get("caid3_auc"),
                delta=r.get("delta_auc_dn_minus_plddt"),
                rescue=r.get("labeled_rescue_rate"),
                risk=r.get("contamination_risk_tier"),
            )
        )
    lines += [
        "",
        "## Layout",
        "",
        "```",
        "publish_*/",
        "  MANIFEST.json / PACKAGE_README.md / comparison.json",
        "  <run_label>/   # key reports + distrust_figures/",
        "```",
        "",
        "Submitters: `submit_publish_650m.sh` or `submit_publish_3b.sh`",
        "(CLI: `python rockfish/publish_submit.py submit-650m|submit-3b`).",
        "",
        "## Next step",
        "",
        "Fill `docs/METHODS_CHECKLIST.md` from these files and apply the go/no-go",
        "criteria in `rockfish/README.md` § Publish path.",
        "",
    ]
    path = package_dir / "PACKAGE_README.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def assemble_publish_package(
    package_dir: str | Path,
    runs: Sequence[dict | RunSpec],
    *,
    package_id: Optional[str] = None,
    strict: bool = False,
    kind: Optional[str] = None,
) -> dict:
    """
    Build organized package.

    Each run dict needs: label, profile, backbone, checkpoint_dir.
    If ``strict``, raise PackageIncompleteError when required artifacts are missing.
    """
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    pkg_id = package_id or f"publish_{utc_stamp()}"
    rev = git_revision()

    run_summaries: list[dict] = []
    layout: dict[str, Any] = {}
    strict_errors: list[str] = []

    # Validate first when strict so we do not leave a partial package on disk
    for run in runs:
        label = str(run["label"])
        ckpt = Path(run["checkpoint_dir"])
        missing = missing_required_artifacts(ckpt)
        if missing:
            msg = f"{label}: missing {missing} under {ckpt}"
            if strict:
                strict_errors.append(msg)
            else:
                print(f"Warning: {msg}", file=sys.stderr)

    if strict and strict_errors:
        raise PackageIncompleteError(
            "Strict package incomplete:\n  - " + "\n  - ".join(strict_errors)
        )

    for run in runs:
        label = str(run["label"])
        profile = str(run.get("profile") or "")
        backbone = str(run.get("backbone") or "")
        ckpt = Path(run["checkpoint_dir"])
        missing = missing_required_artifacts(ckpt)

        dest = package_dir / label
        copied = _copy_tree_selective(ckpt, dest)
        summary = extract_run_summary(ckpt, label=label, profile=profile, backbone=backbone)
        summary["copied_artifacts"] = copied
        summary["package_subdir"] = label
        summary["missing_required"] = missing
        run_summaries.append(summary)
        layout[label] = {
            "source_checkpoint_dir": str(ckpt),
            "package_subdir": str(dest),
            "n_copied": len(copied),
            "missing_required": missing,
        }

    comparison = {
        "package_id": pkg_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": rev,
        "kind": kind,
        "strict": strict,
        "runs": run_summaries,
    }
    manifest = {
        **comparison,
        "layout": layout,
        "required_artifacts_strict": list(REQUIRED_ARTIFACTS_STRICT),
        "note": (
            "Organized Rockfish publish package. Prefer "
            "`python rockfish/publish_submit.py package --kind 650m|3b`. "
            "See PACKAGE_README.md."
        ),
    }

    (package_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (package_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_package_readme(package_dir, comparison)
    return manifest


def build_runs_for_kind(
    root_workdir: str | Path,
    kind: str,
    *,
    include_clean: bool = True,
) -> list[dict]:
    specs = run_specs_for_kind(root_workdir, kind, include_clean=include_clean)
    return [
        {
            "label": r["label"],
            "profile": r["profile"],
            "backbone": r["backbone"],
            "checkpoint_dir": r["checkpoint_dir"],
        }
        for r in specs
    ]


def build_default_runs(
    root_workdir: str | Path,
    *,
    include_clean: bool = True,
    include_3b: bool = True,
) -> list[dict]:
    """Deprecated combined layout. Prefer build_runs_for_kind."""
    runs = build_runs_for_kind(root_workdir, "650m", include_clean=include_clean)
    if include_3b:
        runs.extend(build_runs_for_kind(root_workdir, "3b", include_clean=False))
    return runs


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Assemble DisorderNet Rockfish publish package. "
            "Prefer: python rockfish/publish_submit.py package --kind 650m|3b"
        ),
    )
    p.add_argument("--package-dir", required=True, help="Output package directory")
    p.add_argument("--root-workdir", default=None, help="Bundle parent directory")
    p.add_argument(
        "--kind",
        choices=["650m", "3b"],
        default=None,
        help="Bundle kind (recommended). If omitted with --root-workdir, uses legacy 650m+3b.",
    )
    p.add_argument("--no-clean", action="store_true", help="Skip clean companion in package")
    p.add_argument(
        "--no-3b",
        action="store_true",
        help="Legacy: skip ultra3b when --kind is omitted",
    )
    p.add_argument("--package-id", default=None)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if required go/no-go artifacts are missing",
    )
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="Explicit run spec label=profile=backbone=/path/to/checkpoints (repeatable)",
    )
    args = p.parse_args(argv)

    include_clean = not args.no_clean
    runs: list[dict] = []
    kind = args.kind

    if args.run:
        for spec in args.run:
            parts = spec.split("=", 3)
            if len(parts) != 4:
                print(
                    f"ERROR: bad --run {spec!r}; expected label=profile=backbone=/path/to/ckpt",
                    file=sys.stderr,
                )
                return 2
            label, profile, backbone, ckpt = parts
            runs.append(
                {
                    "label": label,
                    "profile": profile,
                    "backbone": backbone,
                    "checkpoint_dir": ckpt,
                }
            )
    elif args.root_workdir and kind:
        runs = build_runs_for_kind(
            args.root_workdir, kind, include_clean=include_clean,
        )
    elif args.root_workdir:
        print(
            "Warning: --kind omitted; using legacy combined 650m+3b layout. "
            "Prefer --kind 650m|3b.",
            file=sys.stderr,
        )
        runs = build_default_runs(
            args.root_workdir,
            include_clean=include_clean,
            include_3b=not args.no_3b,
        )
    else:
        print("ERROR: provide --root-workdir (with --kind) or --run", file=sys.stderr)
        return 2

    try:
        manifest = assemble_publish_package(
            args.package_dir,
            runs,
            package_id=args.package_id,
            strict=args.strict,
            kind=kind,
        )
    except PackageIncompleteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Publish package → {args.package_dir}")
    print(
        json.dumps(
            {
                "package_id": manifest["package_id"],
                "n_runs": len(manifest["runs"]),
                "git_revision": manifest.get("git_revision"),
                "strict": args.strict,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
