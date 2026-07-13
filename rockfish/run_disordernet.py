#!/usr/bin/env python3
"""
DisorderNet HPC runner — mirrors Colab Pro pipeline for Slurm clusters.

Stages:
  screen      — quick paradigm screen (go/no-go)
  cv          — full N-fold GPU cross-validation
  stack       — GPU+v6 ensemble + SOTA meta-stack (Cells 7b–7c)
  postprocess — fold soup + calibration (Cell 7d)
  full        — cv → stack → postprocess in one job

Example (interactive debug on Rockfish):
  python rockfish/run_disordernet.py cv --profile ultra --backbone 650M

Example (submit via Slurm):
  sbatch rockfish/slurm/train_ultra.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Repo root on PYTHONPATH when launched from rockfish/ or repo root
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _resolve_workdir(path: Optional[str]) -> str:
    """Prefer explicit workdir, then Slurm scratch, then $SCRATCH, else cwd."""
    if path:
        os.makedirs(path, exist_ok=True)
        return path
    for candidate in (
        os.environ.get("SLURM_TMPDIR"),
        os.environ.get("TMPDIR"),
        os.environ.get("SCRATCH"),
        os.environ.get("PI_SCRATCH"),
        f"/scratch/{os.environ.get('USER', '')}" if os.environ.get("USER") else None,
    ):
        if candidate and os.path.isdir(candidate):
            work = os.path.join(candidate, "disordernet")
            os.makedirs(work, exist_ok=True)
            return work
    return os.getcwd()


def _load_proteins(data_cache: str, cfg) -> tuple[list, dict]:
    from colab.disordernet_gpu import fetch_disprot, get_disprot_cache_meta, process_disprot

    entries = fetch_disprot(cache_path=data_cache)
    proteins, disprot_meta = process_disprot(entries, cfg)
    return proteins, disprot_meta


def _build_cfg(args, workdir: str):
    from colab.disordernet_gpu import TrainConfig, setup_environment
    from colab.esm_backbone import apply_backbone_to_config

    cfg = TrainConfig.from_profile(
        args.profile,
        seed=args.seed,
        n_folds=args.n_folds,
        esm_backbone=args.backbone,
        checkpoint_dir=os.path.join(workdir, args.checkpoint_dir),
        data_cache=os.path.join(workdir, args.data_cache)
        if not os.path.isabs(args.data_cache)
        else args.data_cache,
        num_workers=args.num_workers,
    )
    cfg = setup_environment(cfg)
    cfg = apply_backbone_to_config(cfg, args.backbone)
    return cfg


def _load_esm(cfg):
    from colab.esm_backbone import load_esm_backbone

    model, alphabet, converter, spec = load_esm_backbone(
        cfg.device,
        backbone=cfg.esm_backbone,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
    )
    return model, alphabet, converter, spec


def stage_screen(args, cfg, proteins, model, converter) -> dict:
    from colab.quick_screen import (
        print_quick_screen_report,
        run_paradigm_quick_screen,
        save_quick_screen_report,
    )

    report = run_paradigm_quick_screen(
        proteins=proteins,
        esm_backbone=model,
        batch_converter=converter,
        cfg=cfg,
        mode=args.screen_mode,
        seed=args.seed,
        run_ensemble=not args.skip_ensemble,
        use_v6_pro=True,
        backbone=args.backbone,
        checkpoint_subdir="quick_screen",
    )
    print_quick_screen_report(report)
    out = os.path.join(
        cfg.checkpoint_dir,
        f"quick_screen_{args.screen_mode}_{args.backbone}.json",
    )
    save_quick_screen_report(report, out)
    print(f"Screen report saved: {out}")
    return report


def stage_cv(args, cfg, proteins, model, converter, disprot_meta) -> tuple[list, dict]:
    from colab.disordernet_gpu import infer_resume_fold, run_cross_validation
    from colab.run_manifest import build_run_manifest, save_run_manifest

    progress_path = os.path.join(cfg.checkpoint_dir, "cv_progress.json")
    resume = args.resume_fold
    if resume < 0:
        resume = infer_resume_fold(progress_path, cfg, proteins, disprot_meta)

    fold_results, cv_summary = run_cross_validation(
        proteins=proteins,
        esm_backbone=model,
        batch_converter=converter,
        cfg=cfg,
        resume_from_fold=resume,
        prefetch_af_plddt=args.prefetch_af_plddt,
    )

    cv_path = os.path.join(cfg.checkpoint_dir, "cv_summary.json")
    with open(cv_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"CV summary saved: {cv_path}")

    manifest = build_run_manifest(cfg, proteins, cv_summary, disprot_meta, extra={
        "stage": "cv",
        "cluster": os.environ.get("SLURM_CLUSTER_NAME", "local"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
    })
    save_run_manifest(manifest, os.path.join(cfg.checkpoint_dir, "run_manifest.json"))
    return fold_results, cv_summary


def _load_fold_results(checkpoint_dir: str) -> tuple[list, dict]:
    from colab.disordernet_gpu import _deserialize_fold_result

    progress_path = os.path.join(checkpoint_dir, "cv_progress.json")
    if not os.path.isfile(progress_path):
        raise FileNotFoundError(
            f"No cv_progress.json in {checkpoint_dir}. Run stage 'cv' first."
        )
    with open(progress_path) as f:
        payload = json.load(f)
    fold_results = [_deserialize_fold_result(r) for r in payload.get("fold_results", [])]
    summary_path = os.path.join(checkpoint_dir, "cv_summary.json")
    cv_summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            cv_summary = json.load(f)
    return fold_results, cv_summary


def stage_stack(args, cfg, proteins, fold_results, cv_summary) -> tuple[list, dict]:
    from colab.ensemble_v6 import (
        apply_gpu_v6_ensemble,
        print_ensemble_report,
        save_ensemble_report,
    )
    from colab.sota_ensemble import (
        apply_sota_stack,
        print_sota_stack_report,
        save_sota_stack_report,
    )

    v6_cache = os.path.join(cfg.checkpoint_dir, "v6_oof_probs_cache.json")
    ensemble_report, fold_results, v6_probs = apply_gpu_v6_ensemble(
        proteins=proteins,
        fold_results=fold_results,
        n_folds=cfg.n_folds,
        v6_cache_path=v6_cache,
        run_v6_if_missing=True,
        seed=cfg.seed,
        use_v6_pro=True,
    )
    print_ensemble_report(ensemble_report)
    save_ensemble_report(
        ensemble_report,
        os.path.join(cfg.checkpoint_dir, "gpu_v6_ensemble_report.json"),
    )

    sota_report, fold_results, _ = apply_sota_stack(
        proteins=proteins,
        fold_results=fold_results,
        n_folds=cfg.n_folds,
        v6_probs_by_id=v6_probs,
        v6_cache_path=v6_cache,
        run_v6_if_missing=False,
        seed=cfg.seed,
        use_v6_pro=True,
        use_meta_ensemble=True,
    )
    print_sota_stack_report(sota_report)
    save_sota_stack_report(
        sota_report,
        os.path.join(cfg.checkpoint_dir, "sota_stack_report.json"),
    )

    if sota_report.get("after", {}).get("pooled"):
        cv_summary["pooled_auc"] = sota_report["after"]["pooled"]["auc"]
        cv_summary["pooled_ap"] = sota_report["after"]["pooled"]["ap"]
    cv_summary["gpu_v6_ensemble"] = ensemble_report
    cv_summary["sota_stack"] = sota_report

    summary_path = os.path.join(cfg.checkpoint_dir, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    return fold_results, cv_summary


def stage_postprocess(args, cfg, proteins, fold_results, model, converter) -> tuple[dict, list]:
    from colab.sota_postprocess import (
        print_sota_postprocess_report,
        run_sota_postprocess,
        save_sota_postprocess_report,
    )

    plddt_by_id = None
    if cfg.use_hallucination_weighting:
        from colab.disordernet_gpu import build_plddt_cache_for_training

        plddt_by_id = build_plddt_cache_for_training(
            proteins, cache_dir=os.path.join(cfg.checkpoint_dir, cfg.af_plddt_cache_dir),
        )

    report, fold_results = run_sota_postprocess(
        proteins=proteins,
        esm_backbone=model,
        batch_converter=converter,
        cfg=cfg,
        fold_results=fold_results,
        plddt_by_id=plddt_by_id,
        checkpoint_dir=cfg.checkpoint_dir,
        apply_soup=not args.skip_soup,
        soup_mode=args.soup_mode,
        calibrate=not args.skip_calibration,
        calibration_method=args.calibration_method,
    )
    print_sota_postprocess_report(report)
    save_sota_postprocess_report(
        report,
        os.path.join(cfg.checkpoint_dir, "sota_postprocess_report.json"),
    )
    return report, fold_results


def run_pipeline(args) -> int:
    t0 = time.time()
    workdir = _resolve_workdir(args.workdir)
    os.chdir(workdir)
    print(f"Workdir: {workdir}")
    print(f"Stage: {args.stage}  profile={args.profile}  backbone={args.backbone}")

    cfg = _build_cfg(args, workdir)
    proteins, disprot_meta = _load_proteins(cfg.data_cache, cfg)
    print(f"Proteins: {len(proteins):,}  residues: {sum(p['length'] for p in proteins):,}")

    needs_esm = args.stage in ("screen", "cv", "postprocess", "full")
    model = converter = None
    if needs_esm:
        model, _, converter, spec = _load_esm(cfg)
        print(f"ESM backbone: {spec.key}  dim={spec.embed_dim}")

    fold_results: list = []
    cv_summary: dict = {}

    if args.stage == "screen":
        stage_screen(args, cfg, proteins, model, converter)

    elif args.stage == "cv":
        fold_results, cv_summary = stage_cv(
            args, cfg, proteins, model, converter, disprot_meta,
        )

    elif args.stage == "stack":
        fold_results, cv_summary = _load_fold_results(cfg.checkpoint_dir)
        fold_results, cv_summary = stage_stack(args, cfg, proteins, fold_results, cv_summary)

    elif args.stage == "postprocess":
        fold_results, cv_summary = _load_fold_results(cfg.checkpoint_dir)
        stage_postprocess(args, cfg, proteins, fold_results, model, converter)

    elif args.stage == "full":
        fold_results, cv_summary = stage_cv(
            args, cfg, proteins, model, converter, disprot_meta,
        )
        fold_results, cv_summary = stage_stack(args, cfg, proteins, fold_results, cv_summary)
        stage_postprocess(args, cfg, proteins, fold_results, model, converter)

    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    elapsed_h = (time.time() - t0) / 3600
    print(f"\nDone — stage={args.stage}  elapsed={elapsed_h:.2f}h  workdir={workdir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DisorderNet HPC pipeline (Rockfish / Slurm)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "stage",
        choices=["screen", "cv", "stack", "postprocess", "full"],
        help="Pipeline stage to run",
    )
    p.add_argument("--profile", default="ultra", help="TrainConfig profile")
    p.add_argument("--backbone", default="650M", help="ESM-2 backbone key (650M, 3B, …)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--workdir", default=None, help="Working directory (default: Slurm scratch)")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Relative to workdir")
    p.add_argument("--data-cache", default="disprot_raw.json", help="DisProt cache path")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (HPC: 4–8)")
    p.add_argument("--resume-fold", type=int, default=-1, help="Resume CV from fold (auto if -1)")

    # Screen
    p.add_argument("--screen-mode", default="standard", choices=["flash", "standard", "paradigm"])
    p.add_argument("--skip-ensemble", action="store_true", help="Screen: skip v6 ensemble")

    # CV
    p.add_argument("--prefetch-af-plddt", action="store_true", help="Prefetch AF pLDDT cache")

    # Postprocess
    p.add_argument("--skip-soup", action="store_true")
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument(
        "--calibration-method",
        default="temperature_then_isotonic",
        choices=["temperature", "isotonic", "temperature_then_isotonic"],
    )
    p.add_argument("--soup-mode", default="held_out", choices=["held_out", "full_soup"])
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_pipeline(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
