#!/usr/bin/env python3
"""Assemble an organized Rockfish publish package from run workdirs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# High-value artifacts to retain under each run folder (relative to checkpoint dir)
KEEP_NAMES = [
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
    "inference_fusion_report.json",
    "structure_distrust_benchmark.json",
    "structure_distrust_atlas_report.json",
    "structure_distrust_atlas.jsonl",
    "structure_distrust_atlas.tsv",
    "function_prediction_report.json",
    "idr_biology_layer_report.json",
    "idr_biology_layer_report.md",
    "idr_biology_layer_report.html",
    "idr_biology_layer.jsonl",
]


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
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
        "delta_auc_dn_minus_plddt": _dig(matched, "delta_auc_dn_minus_plddt")
        if isinstance(matched, dict)
        else None,
        "labeled_rescue_rate": _dig(rescue, "rescue_rate") if isinstance(rescue, dict) else None,
        "hallucination_rate": _dig(rescue, "hallucination_rate") if isinstance(rescue, dict) else None,
        "mask_utility_enabled": bool(isinstance(util, dict) and util.get("enabled")),
        "caid3_floor_available": bool(isinstance(floor, dict) and floor.get("available")),
        "contamination_risk_tier": _dig(contam, "risk_tier") if isinstance(contam, dict) else None,
        "artifacts_present": {
            name: (ckpt_dir / name).is_file()
            for name in (
                "structure_distrust_benchmark.json",
                "caid3_eval_report.json",
                "sota_postprocess_report.json",
                "structure_distrust_atlas_report.json",
            )
        },
    }


def _copy_tree_selective(src_ckpt: Path, dest_dir: Path) -> list[str]:
    """Copy known report artifacts (+ figures dir) into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    if not src_ckpt.is_dir():
        return copied

    for name in KEEP_NAMES:
        src = src_ckpt / name
        if src.is_file():
            shutil.copy2(src, dest_dir / name)
            copied.append(name)

    # Glob extras (idr exports, quick screens)
    for pattern in ("idr_biology_layer_*", "quick_screen_*.json"):
        for src in src_ckpt.glob(pattern):
            if src.is_file():
                shutil.copy2(src, dest_dir / src.name)
                copied.append(src.name)

    fig_src = src_ckpt / "distrust_figures"
    if fig_src.is_dir():
        fig_dst = dest_dir / "distrust_figures"
        if fig_dst.exists():
            shutil.rmtree(fig_dst)
        shutil.copytree(fig_src, fig_dst)
        copied.append("distrust_figures/")

    return sorted(set(copied))


def write_package_readme(package_dir: Path, comparison: dict) -> Path:
    runs = comparison.get("runs") or []
    lines = [
        "# DisorderNet Rockfish publish package",
        "",
        f"Generated: {comparison.get('created_utc')}",
        f"Package id: `{comparison.get('package_id')}`",
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
        "publish_package_*/",
        "  MANIFEST.json          # machine-readable comparison + paths",
        "  PACKAGE_README.md      # this file",
        "  comparison.json        # headline metrics only",
        "  ultra_650M/            # main production reports",
        "  ultra_clean_650M/      # contamination-clean ablation (if run)",
        "  ultra3b/               # ESM-2 3B production reports (if run)",
        "```",
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
    runs: list[dict],
    *,
    package_id: Optional[str] = None,
) -> dict:
    """
    Build organized package.

    Each run dict: {
      label, profile, backbone,
      checkpoint_dir  (path containing reports),
    }
    """
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    pkg_id = package_id or f"publish_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    run_summaries: list[dict] = []
    layout: dict[str, Any] = {}

    for run in runs:
        label = str(run["label"])
        profile = str(run.get("profile") or "")
        backbone = str(run.get("backbone") or "")
        ckpt = Path(run["checkpoint_dir"])
        dest = package_dir / label
        copied = _copy_tree_selective(ckpt, dest)
        summary = extract_run_summary(ckpt, label=label, profile=profile, backbone=backbone)
        summary["copied_artifacts"] = copied
        summary["package_subdir"] = label
        run_summaries.append(summary)
        layout[label] = {
            "source_checkpoint_dir": str(ckpt),
            "package_subdir": str(dest),
            "n_copied": len(copied),
        }

    comparison = {
        "package_id": pkg_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runs": run_summaries,
    }
    manifest = {
        **comparison,
        "layout": layout,
        "note": (
            "Organized Rockfish publish package: main ultra (650M), optional clean "
            "ablation, optional ultra3b. See PACKAGE_README.md."
        ),
    }

    (package_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (package_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_package_readme(package_dir, comparison)
    return manifest


def build_default_runs(
    root_workdir: str | Path,
    *,
    include_clean: bool = True,
    include_3b: bool = True,
) -> list[dict]:
    """Standard layout used by submit_publish_all.sh."""
    root = Path(root_workdir)
    runs = [
        {
            "label": "ultra_650M",
            "profile": "ultra",
            "backbone": "650M",
            "checkpoint_dir": root / "ultra_650M" / "checkpoints",
        }
    ]
    if include_clean:
        runs.append(
            {
                "label": "ultra_clean_650M",
                "profile": "ultra_clean",
                "backbone": "650M",
                "checkpoint_dir": root / "ultra_clean_650M" / "checkpoints_ultra_clean",
            }
        )
    if include_3b:
        runs.append(
            {
                "label": "ultra3b",
                "profile": "ultra3b",
                "backbone": "3B",
                "checkpoint_dir": root / "ultra3b" / "checkpoints",
            }
        )
    return runs


def main() -> int:
    p = argparse.ArgumentParser(description="Assemble DisorderNet Rockfish publish package")
    p.add_argument("--package-dir", required=True, help="Output package directory")
    p.add_argument(
        "--root-workdir",
        default=None,
        help="Parent workdir with ultra_650M / ultra_clean_650M / ultra3b children",
    )
    p.add_argument("--no-clean", action="store_true", help="Skip clean companion in package")
    p.add_argument("--no-3b", action="store_true", help="Skip ultra3b in package")
    p.add_argument("--package-id", default=None)
    # Explicit run overrides: --run label=profile=backbone=ckpt_dir (repeatable)
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="Explicit run spec label=profile=backbone=/path/to/checkpoints (repeatable)",
    )
    args = p.parse_args()

    include_clean = not args.no_clean
    include_3b = not args.no_3b

    runs: list[dict] = []
    if args.run:
        for spec in args.run:
            parts = spec.split("=", 3)
            if len(parts) != 4:
                raise SystemExit(
                    f"Bad --run {spec!r}; expected label=profile=backbone=/path/to/ckpt"
                )
            label, profile, backbone, ckpt = parts
            runs.append(
                {
                    "label": label,
                    "profile": profile,
                    "backbone": backbone,
                    "checkpoint_dir": ckpt,
                }
            )
    elif args.root_workdir:
        runs = build_default_runs(
            args.root_workdir,
            include_clean=include_clean,
            include_3b=include_3b,
        )
    else:
        raise SystemExit("Provide --root-workdir or one/more --run specs")

    manifest = assemble_publish_package(
        args.package_dir,
        runs,
        package_id=args.package_id,
    )
    print(f"Publish package → {args.package_dir}")
    print(json.dumps({"package_id": manifest["package_id"], "n_runs": len(manifest["runs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
