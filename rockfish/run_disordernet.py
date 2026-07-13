#!/usr/bin/env python3
"""
DisorderNet HPC runner — mirrors Colab Pro pipeline for Slurm clusters.

Stages:
  screen           — quick paradigm screen (go/no-go)
  cv               — full N-fold GPU cross-validation
  stack            — GPU+v6 ensemble + SOTA meta-stack (Cells 7b–7c)
  postprocess      — fold soup + calibration (Cell 7d)
  full             — cv → stack → postprocess
  eval             — CAID reports, AF rescue, structure calibration, Phase 3 (Cells 8–11)
  predict          — batch FASTA inference with fold soup
  multi-seed-blend — average OOF from multiple seed checkpoint dirs
  pipeline         — full → eval (complete Rockfish production run)

Example (interactive debug on Rockfish):
  python rockfish/run_disordernet.py cv --profile ultra --backbone 650M
  python rockfish/run_disordernet.py boltz --boltz-mode auto   # Boltz-2 default
  python rockfish/run_disordernet.py cv --profile ultra_fun   # disorder→function

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

    overrides = dict(
        seed=args.seed,
        n_folds=args.n_folds,
        esm_backbone=args.backbone,
        checkpoint_dir=os.path.join(workdir, args.checkpoint_dir),
        data_cache=os.path.join(workdir, args.data_cache)
        if not os.path.isabs(args.data_cache)
        else args.data_cache,
        num_workers=args.num_workers,
    )
    if getattr(args, "function_head", False):
        overrides["use_function_head"] = True
    if getattr(args, "no_function_head", False):
        overrides["use_function_head"] = False
    if getattr(args, "function_loss_weight", None) is not None:
        overrides["function_loss_weight"] = args.function_loss_weight
    cfg = TrainConfig.from_profile(args.profile, **overrides)
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

    # Structure pLDDT for train-time channel (Boltz default > AF3 > AF2)
    plddt_override = None
    backend = getattr(args, "structure_backend", "boltz")
    prefer = backend if backend in ("boltz", "af3", "af2") else "boltz"
    plddt_boltz: dict = {}
    plddt_af3: dict = {}
    af2 = getattr(args, "_prefetched_plddt", None) or {}

    if getattr(args, "boltz_mode", "off") != "off" and backend != "af2":
        boltz_batch = stage_boltz(args, cfg, proteins)
        plddt_boltz = boltz_batch.get("plddt_by_id") or {}
    if getattr(args, "af3_mode", "off") != "off":
        af3_batch = stage_af3(args, cfg, proteins)
        plddt_af3 = af3_batch.get("plddt_by_id") or {}

    if plddt_boltz or plddt_af3:
        from colab.disordernet_gpu import merge_plddt_for_training, build_plddt_cache_for_training
        if not af2 and (args.prefetch_af_plddt or cfg.use_plddt_features or cfg.use_hallucination_weighting):
            af2 = build_plddt_cache_for_training(
                proteins, cache_dir=os.path.join(cfg.checkpoint_dir, cfg.af_plddt_cache_dir),
            )
        plddt_override = merge_plddt_for_training(
            af2, plddt_af3, prefer_af3=(prefer == "af3"),
            plddt_boltz=plddt_boltz, prefer=prefer,
        )
        print(
            f"  Merged pLDDT for training: {len(plddt_override)} proteins "
            f"(prefer={prefer}, boltz={len(plddt_boltz)}, af3={len(plddt_af3)}, af2={len(af2)})"
        )
    elif af2:
        plddt_override = af2

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
        plddt_by_id=plddt_override,
        prefetch_af_plddt=args.prefetch_af_plddt and plddt_override is None,
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



def stage_boltz(args, cfg, proteins) -> dict:
    """Ingest or run Boltz-2 (pinned); return paths + pLDDT dict."""
    from rockfish.boltz_rockfish import run_boltz_on_rockfish, setup_boltz_for_rockfish

    mode = getattr(args, "boltz_mode", "off")
    if mode == "off":
        print("Boltz disabled (boltz_mode=off)")
        return {"skipped": True, "mode": "off"}

    batch = run_boltz_on_rockfish(
        proteins=proteins,
        mode=mode,
        boltz_root=getattr(args, "boltz_root", None),
        max_proteins=getattr(args, "boltz_max_proteins", None),
        msa_free=not getattr(args, "boltz_use_msa_server", False),
        use_msa_server=getattr(args, "boltz_use_msa_server", False),
        timeout_s=getattr(args, "boltz_timeout", 7200),
        shard_index=getattr(args, "boltz_shard_index", None),
        shard_count=getattr(args, "boltz_shard_count", None),
        sampling_steps=getattr(args, "boltz_sampling_steps", 50),
    )
    if batch.get("skipped"):
        return batch

    paths = batch.get("paths") or setup_boltz_for_rockfish(
        mode="ingest", boltz_root=args.boltz_root, ensure_install=False,
    )["paths"]
    out_root = paths["output_dir"]
    cache = os.path.join(cfg.checkpoint_dir, "boltz_plddt_cache")
    try:
        from colab.boltz_plddt import load_boltz_plddt_batch
        plddt = load_boltz_plddt_batch(
            proteins, output_root=out_root, cache_dir=cache, verbose=True,
        )
        batch["n_plddt"] = len(plddt)
        batch["plddt_by_id"] = plddt
        batch["output_dir"] = out_root
        batch["paths"] = paths
        print(f"Boltz pLDDT available for {len(plddt)}/{len(proteins)} proteins")
    except Exception as exc:
        print(f"Boltz pLDDT load failed: {exc}")
        batch["plddt_by_id"] = {}
        batch["paths"] = paths
    return batch


def stage_af3(args, cfg, proteins) -> dict:
    """Ingest or run AF3 on Rockfish; return paths + optional pLDDT dict."""
    from rockfish.af3_rockfish import run_af3_on_rockfish, setup_af3_for_rockfish

    mode = args.af3_mode
    if mode == "off":
        print("AF3 disabled (af3_mode=off)")
        return {"skipped": True, "mode": "off"}

    batch = run_af3_on_rockfish(
        proteins=proteins,
        mode=mode,
        af3_root=args.af3_root,
        max_proteins=args.af3_max_proteins,
        msa_free=not args.af3_use_msa,
        timeout_s=args.af3_timeout,
        prefer_docker=args.af3_docker,
        shard_index=getattr(args, "af3_shard_index", None),
        shard_count=getattr(args, "af3_shard_count", None),
    )
    if batch.get("skipped"):
        return batch
    if not batch.get("success") and not batch.get("n_done") and mode == "ingest":
        # ingest: success may be True with n_done; tolerate partial
        if not batch.get("paths"):
            print(f"AF3 stage warning: {batch.get('error', batch)}")
            return batch

    paths = batch.get("paths") or setup_af3_for_rockfish(mode="ingest", af3_root=args.af3_root)["paths"]
    out_root = paths["output_dir"]
    cache = os.path.join(cfg.checkpoint_dir, "af3_plddt_cache")
    try:
        from colab.af3_plddt import load_af3_plddt_batch
        plddt = load_af3_plddt_batch(proteins, output_root=out_root, cache_dir=cache, verbose=True)
        batch["n_plddt"] = len(plddt)
        batch["plddt_by_id"] = plddt
        batch["output_dir"] = out_root
        batch["paths"] = paths
        print(f"AF3 pLDDT available for {len(plddt)}/{len(proteins)} proteins")
    except Exception as exc:
        print(f"AF3 pLDDT load failed: {exc}")
        batch["plddt_by_id"] = {}
        batch["paths"] = paths
    return batch

def stage_eval(args, cfg, proteins, fold_results, cv_summary) -> dict:
    """Cells 8–11: CAID, biological utility, AF rescue, structure calibration, Phase 3."""
    import json

    from colab.af_hallucination import (
        fetch_and_run_af_rescue_report,
        print_af_rescue_report,
        save_af_rescue_report,
    )
    from colab.downstream_refresh import refresh_downstream_metrics
    from colab.inference_fusion import (
        apply_plddt_fusion_to_cv,
        print_fusion_report,
        save_fusion_report,
    )
    from colab.novel_use_cases import build_af_rescue_manifest, save_novel_use_case_report
    from colab.phase3_synthesis import (
        print_phase3_report,
        run_phase3_integrated_report,
        run_structure_calibration_report,
        save_phase3_report,
    )
    from colab.statistical_validation import (
        print_statistical_validation,
        run_full_statistical_validation,
        save_statistical_validation,
    )

    ckpt = cfg.checkpoint_dir
    downstream = refresh_downstream_metrics(
        proteins, fold_results, n_folds=cfg.n_folds, print_reports=True,
    )

    # Disorder → function (multi-label) OOF report
    from colab.function_predict import (
        print_function_report,
        run_function_prediction_report,
        save_function_report,
        summarize_function_label_coverage,
    )
    func_report = run_function_prediction_report(
        proteins, fold_results, n_folds=cfg.n_folds,
    )
    if not func_report.get("enabled"):
        func_report["label_coverage"] = summarize_function_label_coverage(proteins)
    print_function_report(func_report)
    save_function_report(func_report, os.path.join(ckpt, "function_prediction_report.json"))

    af_cache = os.path.join(ckpt, cfg.af_plddt_cache_dir)
    af_report, plddt_by_id = fetch_and_run_af_rescue_report(
        proteins=proteins,
        fold_results=fold_results,
        n_folds=cfg.n_folds,
        cache_dir=af_cache,
        sleep_s=0.05,
    )
    print_af_rescue_report(af_report)
    save_af_rescue_report(af_report, os.path.join(ckpt, "af_rescue_report.json"))

    # Boltz-2 rescue (default structure backend)
    boltz_report = {"skipped": True}
    plddt_boltz: dict = {}
    if getattr(args, "boltz_mode", "off") != "off":
        from colab.af_hallucination import run_af_rescue_report
        from colab.boltz_plddt import load_boltz_plddt_batch
        from rockfish.boltz_rockfish import setup_boltz_for_rockfish

        if args.boltz_mode in ("run", "auto"):
            stage_boltz(args, cfg, proteins)
        boltz_cfg = setup_boltz_for_rockfish(
            mode="ingest", boltz_root=args.boltz_root, ensure_install=False,
        )
        out_root = boltz_cfg["paths"]["output_dir"]
        plddt_boltz = load_boltz_plddt_batch(
            proteins, output_root=out_root,
            cache_dir=os.path.join(ckpt, "boltz_plddt_cache"), verbose=True,
        )
        if plddt_boltz:
            boltz_report = run_af_rescue_report(
                proteins=proteins,
                fold_results=fold_results,
                plddt_by_protein=plddt_boltz,
                n_folds=cfg.n_folds,
                source="Boltz-2 (pinned auto-download)",
            )
            boltz_report["skipped"] = False
            boltz_report["n_plddt_loaded"] = len(plddt_boltz)
            print_af_rescue_report(boltz_report)
            save_af_rescue_report(boltz_report, os.path.join(ckpt, "boltz_rescue_report.json"))
            # Prefer Boltz over AF2 for downstream fusion when available
            from colab.disordernet_gpu import merge_plddt_for_training
            plddt_by_id = merge_plddt_for_training(
                plddt_by_id, None, plddt_boltz=plddt_boltz, prefer="boltz",
            )

    # Optional AF3 Phase 2b (ingest preferred on Rockfish)
    af3_report = {"skipped": True}
    plddt_af3: dict = {}
    af3_comparison: dict = {"insufficient_data": True}
    if getattr(args, "af3_mode", "off") != "off":
        from colab.af_hallucination import (
            fetch_and_run_af3_rescue_report,
            print_af2_af3_comparison,
            run_af2_af3_comparison_report,
            save_af2_af3_comparison,
        )
        from rockfish.af3_rockfish import setup_af3_for_rockfish

        af3_cfg = setup_af3_for_rockfish(mode=args.af3_mode, af3_root=args.af3_root)
        if args.af3_mode in ("run", "auto"):
            stage_af3(args, cfg, proteins)
            af3_cfg = setup_af3_for_rockfish(mode="ingest", af3_root=args.af3_root)
        if af3_cfg.get("outputs_ok"):
            af3_report, plddt_af3 = fetch_and_run_af3_rescue_report(
                proteins=proteins,
                fold_results=fold_results,
                af3_output_root=af3_cfg["paths"]["output_dir"],
                n_folds=cfg.n_folds,
                cache_dir=os.path.join(ckpt, "af3_plddt_cache"),
            )
            af3_report["skipped"] = False
            print_af_rescue_report(af3_report)
            save_af_rescue_report(af3_report, os.path.join(ckpt, "af3_rescue_report.json"))
            af3_comparison = run_af2_af3_comparison_report(af_report, af3_report)
            print_af2_af3_comparison(af3_comparison)
            save_af2_af3_comparison(af3_comparison, os.path.join(ckpt, "af2_af3_comparison.json"))
            # Prefer AF3 pLDDT for fusion / training-adjacent analysis when available
            from colab.disordernet_gpu import merge_plddt_for_training
            prefer = getattr(args, "structure_backend", "boltz")
            if prefer not in ("boltz", "af3", "af2"):
                prefer = "boltz"
            plddt_by_id = merge_plddt_for_training(
                plddt_by_id, plddt_af3,
                plddt_boltz=plddt_boltz,
                prefer=prefer,
            )

    fusion_report, fold_results = apply_plddt_fusion_to_cv(
        proteins=proteins,
        fold_results=fold_results,
        plddt_by_protein=plddt_by_id,
        n_folds=cfg.n_folds,
    )
    print_fusion_report(fusion_report)
    save_fusion_report(
        fusion_report, os.path.join(ckpt, "inference_fusion_report.json"),
    )

    calibration_report = run_structure_calibration_report(
        proteins=proteins,
        fold_results=fold_results,
        plddt_by_protein=plddt_by_id,
        n_folds=cfg.n_folds,
    )

    stats_report = run_full_statistical_validation(
        proteins, fold_results, plddt_by_protein=plddt_by_id, n_folds=cfg.n_folds,
    )
    print_statistical_validation(stats_report)
    save_statistical_validation(
        stats_report, os.path.join(ckpt, "statistical_validation_report.json"),
    )

    cv_pooled = {
        "auc": downstream["our_auc"],
        "ap": downstream["our_ap"],
        "f1": downstream["our_f1"],
        "mcc": downstream["our_mcc"],
        "opt_threshold": downstream["opt_threshold"],
    }
    phase3 = run_phase3_integrated_report(
        cv_pooled=cv_pooled,
        bio_report=downstream["bio_report"],
        af_report=af_report,
        calibration_report=calibration_report,
        af3_report=af3_report if not af3_report.get("skipped") else None,
        af2_af3_comparison=af3_comparison if not af3_comparison.get("insufficient_data") else None,
        caid_report=downstream["caid_report"],
        statistical_validation=stats_report,
    )
    print_phase3_report(phase3)
    save_phase3_report(phase3, os.path.join(ckpt, "phase3_integrated_report.json"))

    from colab.biological_utility import align_fold_predictions

    preds_by_id = {
        item["id"]: item["probs"]
        for item in align_fold_predictions(proteins, fold_results, n_folds=cfg.n_folds)
    }
    manifest = build_af_rescue_manifest(proteins, preds_by_id, plddt_by_id)
    save_novel_use_case_report(manifest, os.path.join(ckpt, "af_rescue_manifest.json"))

    eval_summary = {
        "pooled_auc": cv_pooled["auc"],
        "pooled_ap": cv_pooled["ap"],
        "phase3_headline": phase3.get("headline"),
        "af_rescue_rate": af_report.get("pooled", {}).get("rescue_rate"),
        "fusion_delta_auc": fusion_report.get("delta_auc_pooled"),
        "function_macro_auc": (
            func_report.get("metrics", {}) or {}
        ).get("macro_auc"),
        "function_enabled": bool(func_report.get("enabled")),
    }
    with open(os.path.join(ckpt, "eval_summary.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"Eval summary saved: {os.path.join(ckpt, 'eval_summary.json')}")
    return eval_summary


def stage_predict(args, cfg, model, converter) -> dict:
    from colab.predict_batch import export_predictions, parse_fasta, predict_fasta_batch

    if not args.fasta:
        raise ValueError("--fasta required for predict stage")

    plddt_by_id = None
    if args.prefetch_af_plddt:
        from colab.disordernet_gpu import build_plddt_cache_for_training
        proteins = parse_fasta(args.fasta)
        plddt_by_id = build_plddt_cache_for_training(
            proteins, cache_dir=os.path.join(cfg.checkpoint_dir, cfg.af_plddt_cache_dir),
        )

    preds = predict_fasta_batch(
        fasta_path=args.fasta,
        esm_backbone=model,
        batch_converter=converter,
        cfg=cfg,
        checkpoint_dir=cfg.checkpoint_dir,
        plddt_by_id=plddt_by_id,
        use_tta=not args.skip_tta,
        tta_passes=cfg.mc_dropout_tta_passes,
        n_folds=cfg.n_folds,
    )
    proteins = parse_fasta(args.fasta)
    out_dir = args.predict_out or os.path.join(cfg.checkpoint_dir, "predictions")
    manifest = export_predictions(proteins, preds, out_dir, formats=("caid", "tsv"))
    print(f"Predictions written to {out_dir} ({manifest['n_scored']} proteins)")
    return manifest


def stage_multi_seed_blend(args, cfg, proteins) -> tuple[list, dict]:
    from colab.multi_seed_blend import (
        average_fold_results_multi_seed,
        print_multi_seed_report,
        save_multi_seed_report,
    )

    seed_dirs = [s.strip() for s in args.seed_dirs.split(",") if s.strip()]
    if len(seed_dirs) < 2:
        raise ValueError("--seed-dirs needs comma-separated checkpoint dirs (≥2)")

    per_seed: dict[int, list] = {}
    for i, sd in enumerate(seed_dirs):
        fr, _ = _load_fold_results(sd)
        per_seed[42 + i] = fr

    blended, report = average_fold_results_multi_seed(
        proteins, per_seed, n_folds=cfg.n_folds,
    )
    print_multi_seed_report(report)
    save_multi_seed_report(report, os.path.join(cfg.checkpoint_dir, "multi_seed_blend_report.json"))
    return blended, report


def stage_caid3_eval(args, cfg, proteins, fold_results, model, converter) -> dict:
    """Score fold-soup model on CAID3 Disorder-PDB reference (fair vs ESMDisPred)."""
    from colab.caid3_eval import (
        evaluate_caid_predictions,
        export_caid_predictions_dir,
        fetch_caid3_reference,
        parse_caid_reference_fasta,
        print_caid3_eval_report,
        save_caid3_eval_report,
    )
    from colab.predict_batch import predict_fasta_batch

    ref_path = args.caid3_reference or fetch_caid3_reference(
        os.path.join(cfg.checkpoint_dir, "caid3_disorder_pdb.fasta"),
    )
    ref_proteins = parse_caid_reference_fasta(ref_path)

    # Write temp FASTA for predict_batch
    tmp_fasta = os.path.join(cfg.checkpoint_dir, "_caid3_query.fasta")
    with open(tmp_fasta, "w") as f:
        for p in ref_proteins:
            f.write(f">{p['id']}\n{p['sequence']}\n")

    preds_by_id = predict_fasta_batch(
        fasta_path=tmp_fasta,
        esm_backbone=model,
        batch_converter=converter,
        cfg=cfg,
        checkpoint_dir=cfg.checkpoint_dir,
        use_tta=not args.skip_tta,
        n_folds=cfg.n_folds,
    )

    out_dir = os.path.join(cfg.checkpoint_dir, "caid3_submission")
    export_caid_predictions_dir(ref_proteins, preds_by_id, out_dir)

    report = evaluate_caid_predictions(ref_proteins, preds_by_id)
    print_caid3_eval_report(report)
    save_caid3_eval_report(report, os.path.join(cfg.checkpoint_dir, "caid3_eval_report.json"))
    return report


def run_pipeline(args) -> int:
    t0 = time.time()
    workdir = _resolve_workdir(args.workdir)
    os.chdir(workdir)
    print(f"Workdir: {workdir}")
    print(f"Stage: {args.stage}  profile={args.profile}  backbone={args.backbone}")

    cfg = _build_cfg(args, workdir)
    proteins, disprot_meta = _load_proteins(cfg.data_cache, cfg)
    print(f"Proteins: {len(proteins):,}  residues: {sum(p['length'] for p in proteins):,}")

    needs_esm = args.stage in (
        "screen", "cv", "postprocess", "full", "predict", "pipeline",
    )
    model = converter = None
    prefetched_plddt = None

    if needs_esm:
        want_prefetch = (
            args.prefetch_af_plddt
            or cfg.use_plddt_features
            or cfg.use_hallucination_weighting
        ) and args.stage in ("cv", "full", "pipeline", "screen")

        def _load():
            return _load_esm(cfg)

        def _prefetch():
            from colab.disordernet_gpu import build_plddt_cache_for_training
            return build_plddt_cache_for_training(
                proteins,
                cache_dir=os.path.join(cfg.checkpoint_dir, cfg.af_plddt_cache_dir),
            )

        if want_prefetch and not getattr(args, "no_overlap_prefetch", False):
            from colab.async_io import run_overlapped
            (model, _, converter, spec), prefetched_plddt = run_overlapped(
                _load, _prefetch,
                primary_name="ESM load",
                background_name="AF2 pLDDT prefetch",
            )
            print(f"ESM backbone: {spec.key}  dim={spec.embed_dim}")
            print(f"  Overlapped pLDDT prefetch: {len(prefetched_plddt)} proteins")
        else:
            model, _, converter, spec = _load_esm(cfg)
            print(f"ESM backbone: {spec.key}  dim={spec.embed_dim}")

        from colab.hpc_efficiency import disable_gradient_checkpointing_for_inference
        if args.stage in ("predict", "postprocess"):
            if disable_gradient_checkpointing_for_inference(model):
                print("  Inference: gradient checkpointing OFF (same logits, faster)")

    # stash for stage_cv when AF3 not providing override
    args._prefetched_plddt = prefetched_plddt  # type: ignore[attr-defined]

    fold_results: list = []
    cv_summary: dict = {}

    if args.stage == "screen":
        stage_screen(args, cfg, proteins, model, converter)

    elif args.stage == "boltz":
        stage_boltz(args, cfg, proteins)

    elif args.stage == "af3":
        stage_af3(args, cfg, proteins)

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

    elif args.stage == "eval":
        fold_results, cv_summary = _load_fold_results(cfg.checkpoint_dir)
        stage_eval(args, cfg, proteins, fold_results, cv_summary)

    elif args.stage == "predict":
        stage_predict(args, cfg, model, converter)

    elif args.stage == "multi-seed-blend":
        fold_results, _ = stage_multi_seed_blend(args, cfg, proteins)
        with open(os.path.join(cfg.checkpoint_dir, "cv_progress_blended.json"), "w") as f:
            from colab.disordernet_gpu import _serialize_fold_result
            json.dump(
                {"fold_results": [_serialize_fold_result(r) for r in fold_results]},
                f, indent=2,
            )

    elif args.stage == "pipeline":
        fold_results, cv_summary = stage_cv(
            args, cfg, proteins, model, converter, disprot_meta,
        )
        fold_results, cv_summary = stage_stack(args, cfg, proteins, fold_results, cv_summary)
        stage_postprocess(args, cfg, proteins, fold_results, model, converter)
        stage_eval(args, cfg, proteins, fold_results, cv_summary)
        if args.run_caid3_eval:
            stage_caid3_eval(args, cfg, proteins, fold_results, model, converter)

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
        choices=[
            "screen", "cv", "stack", "postprocess", "full",
            "eval", "predict", "multi-seed-blend", "pipeline", "boltz", "af3",
        ],
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
    p.add_argument(
        "--function-head", action="store_true",
        help="Enable disorder→function multi-label head (or use --profile ultra_fun)",
    )
    p.add_argument(
        "--no-function-head", action="store_true",
        help="Disable function head even if profile enables it",
    )
    p.add_argument(
        "--function-loss-weight", type=float, default=None,
        help="Weight for function multi-label BCE (default: profile / 0.35)",
    )

    # Postprocess
    p.add_argument("--skip-soup", action="store_true")
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument(
        "--calibration-method",
        default="temperature_then_isotonic",
        choices=["temperature", "isotonic", "temperature_then_isotonic"],
    )
    p.add_argument("--soup-mode", default="held_out", choices=["held_out", "full_soup"])

    # Eval / CAID3
    p.add_argument("--run-caid3-eval", action="store_true", help="pipeline: also run CAID3 benchmark")
    p.add_argument("--caid3-reference", default=None, help="Path to CAID3 reference FASTA")

    # Predict
    p.add_argument("--fasta", default=None, help="Input FASTA for predict stage")
    p.add_argument("--predict-out", default=None, help="Output directory for predictions")
    p.add_argument("--skip-tta", action="store_true", help="Disable MC-dropout TTA at inference")

    # Multi-seed
    p.add_argument(
        "--seed-dirs",
        default="",
        help="Comma-separated checkpoint dirs for multi-seed-blend",
    )

    # Structure backend (Boltz-2 default — open weights, auto-download)
    p.add_argument(
        "--structure-backend",
        default="boltz",
        choices=["boltz", "af3", "af2", "off"],
        help="Preferred structure pLDDT source for train/eval fusion",
    )
    p.add_argument(
        "--boltz-mode", default="auto",
        choices=["off", "ingest", "run", "auto"],
        help="Boltz-2: auto installs pinned boltz and downloads weights on first run",
    )
    p.add_argument("--boltz-root", default=None, help="Boltz root (inputs/outputs/cache)")
    p.add_argument("--boltz-max-proteins", type=int, default=None)
    p.add_argument("--boltz-timeout", type=int, default=7200)
    p.add_argument(
        "--boltz-use-msa-server", action="store_true",
        help="Use mmseqs MSA server (default: MSA-free single sequence)",
    )
    p.add_argument("--boltz-sampling-steps", type=int, default=50)
    p.add_argument("--boltz-shard-index", type=int, default=None)
    p.add_argument("--boltz-shard-count", type=int, default=None)

    # AF3 (Rockfish) — optional licensed secondary backend
    p.add_argument(
        "--af3-mode", default="off",
        choices=["off", "ingest", "run", "auto"],
        help="AF3 integration: ingest outputs, run missing, or off",
    )
    p.add_argument("--af3-root", default=None, help="AF3 root with af3.bin + outputs/")
    p.add_argument("--af3-max-proteins", type=int, default=None)
    p.add_argument("--af3-timeout", type=int, default=7200)
    p.add_argument("--af3-use-msa", action="store_true", help="Require public_databases MSA (slow)")
    p.add_argument("--af3-docker", action="store_true", help="Prefer Docker over native AF3")
    p.add_argument(
        "--af3-shard-index", type=int, default=None,
        help="AF3 Slurm array shard index (0-based)",
    )
    p.add_argument(
        "--af3-shard-count", type=int, default=None,
        help="AF3 Slurm array shard count",
    )
    p.add_argument(
        "--no-overlap-prefetch", action="store_true",
        help="Disable ESM‖pLDDT overlap (debug)",
    )
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
