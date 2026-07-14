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
    if getattr(args, "no_hallucination_weighting", False):
        overrides["use_hallucination_weighting"] = False
    if getattr(args, "no_plddt_features", False):
        overrides["use_plddt_features"] = False
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
        diffusion_samples=getattr(args, "boltz_diffusion_samples", 5),
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


def stage_idr_layer(args, cfg, proteins, fold_results) -> dict:
    """
    Post-structure IDR biology layer: disorder + roles + hallucination + Boltz variance proxy.
    """
    import numpy as np

    from colab.biological_utility import align_fold_predictions
    from colab.idr_biology_layer import (
        build_idr_layer_package,
        export_idr_layer_bundle,
        print_idr_layer_report,
    )
    from colab.idr_layer_io import (
        load_disorder_preds_from_dir,
        load_function_preds_from_dir,
        load_ligand_map,
        load_partner_map,
    )

    ckpt = cfg.checkpoint_dir
    aligned = align_fold_predictions(proteins, fold_results, n_folds=cfg.n_folds)
    disorder_by_id = {item["id"]: item["probs"] for item in aligned}

    # Optional: overlay / replace with predict-dir scores (re-export path)
    preds_dir = getattr(args, "idr_preds_dir", None)
    if preds_dir:
        loaded = load_disorder_preds_from_dir(preds_dir, proteins=proteins)
        if loaded:
            disorder_by_id.update(loaded)
            print(f"  Loaded disorder scores from {preds_dir}: {len(loaded)} proteins")

    fn_preds_dir = getattr(args, "idr_function_preds_dir", None)
    function_overlay: dict = {}
    if fn_preds_dir:
        function_overlay = load_function_preds_from_dir(fn_preds_dir, proteins=proteins)
        if function_overlay:
            print(f"  Loaded function scores from {fn_preds_dir}: {len(function_overlay)} proteins")

    partners_by_id = load_partner_map(getattr(args, "idr_partners", None))
    if partners_by_id:
        print(f"  Partner-context map: {len(partners_by_id)} proteins")
    ligands_by_id = load_ligand_map(getattr(args, "idr_ligands", None))
    if ligands_by_id:
        print(f"  Ligand-context map: {len(ligands_by_id)} proteins")

    # Function OOF shares the full-sequence pad-mask stream with disorder (same lengths).
    function_by_id: dict = {}
    function_oof_metrics = None
    function_threshold = float(getattr(args, "idr_function_threshold", 0.5))
    threshold_tuning = None
    function_calibration = None
    cache_tag = ""
    y_true = np.zeros((0, 1), dtype=np.float32)
    y_prob = np.zeros((0, 1), dtype=np.float32)
    try:
        from colab.function_predict import (
            align_function_oof,
            calibrate_function_probs,
            run_function_prediction_report,
            split_function_oof_by_lengths,
            tune_function_threshold,
        )
        y_true, y_prob, fn_protein_ids = align_function_oof(
            proteins, fold_results, n_folds=cfg.n_folds,
        )
        total_L = sum(len(item["probs"]) for item in aligned)
        if len(y_prob) == total_L and total_L > 0:
            offset = 0
            for item in aligned:
                L = len(item["probs"])
                function_by_id[item["id"]] = y_prob[offset:offset + L]
                offset += L
            print(f"  Attached function OOF to {len(function_by_id)} proteins")
        elif len(y_prob) > 0 and fn_protein_ids:
            lengths = {p["id"]: int(p.get("length") or len(p["sequence"])) for p in proteins}
            try:
                function_by_id = split_function_oof_by_lengths(y_prob, fn_protein_ids, lengths)
                print(f"  Attached function OOF via protein lengths ({len(function_by_id)})")
            except ValueError:
                print(
                    "  Function OOF length ≠ disorder / protein stream; "
                    "role calls skipped (retrain with function head)."
                )
        if function_by_id:
            if getattr(args, "idr_calibrate_function", False) and len(y_true) > 0:
                from colab.function_predict import (
                    FUNCTION_GROUP_NAMES,
                    apply_function_temperatures,
                )

                calibrated, function_calibration = calibrate_function_probs(y_true, y_prob)
                temps = [
                    float(
                        (function_calibration.get("per_group") or {})
                        .get(g, {})
                        .get("temperature", 1.0)
                    )
                    for g in FUNCTION_GROUP_NAMES
                ]
                if function_calibration.get("enabled"):
                    for pid in list(function_by_id):
                        function_by_id[pid] = apply_function_temperatures(
                            function_by_id[pid], temps,
                        )
                    y_prob = calibrated
                    cache_tag = "calibrated"
                    print(
                        f"  Function temperature calibration mean_T="
                        f"{function_calibration.get('mean_temperature')}"
                    )
            function_oof_metrics = run_function_prediction_report(
                proteins, fold_results, n_folds=cfg.n_folds,
            )
            if getattr(args, "idr_auto_threshold", False) and len(y_true) > 0:
                threshold_tuning = tune_function_threshold(y_true, y_prob)
                if threshold_tuning.get("enabled"):
                    function_threshold = float(threshold_tuning["threshold"])
                    print(
                        f"  OOF-tuned function threshold → {function_threshold} "
                        f"(score={threshold_tuning.get('best_score')})"
                    )
    except Exception as exc:
        print(f"  Function OOF not attached to layer: {exc}")

    if function_overlay:
        function_by_id.update(function_overlay)

    plddt_by_id: dict = {}
    boltz_std: dict = {}
    structure_source = "none"
    if getattr(args, "boltz_mode", "off") != "off":
        from colab.boltz_plddt import load_boltz_structure_features
        from rockfish.boltz_rockfish import setup_boltz_for_rockfish

        if args.boltz_mode in ("run", "auto"):
            stage_boltz(args, cfg, proteins)
        bcfg = setup_boltz_for_rockfish(
            mode="ingest", boltz_root=args.boltz_root, ensure_install=False,
        )
        out_root = bcfg["paths"]["output_dir"]
        plddt_by_id, boltz_std = load_boltz_structure_features(
            proteins,
            out_root,
            plddt_cache_dir=os.path.join(ckpt, "boltz_plddt_cache"),
            variance_cache_dir=os.path.join(ckpt, "boltz_variance_cache"),
            max_samples=getattr(args, "boltz_diffusion_samples", 5),
            verbose=True,
        )
        structure_source = "boltz2"

    if not plddt_by_id:
        try:
            from colab.af_plddt import fetch_plddt_batch
            plddt_by_id = fetch_plddt_batch(
                proteins,
                cache_dir=os.path.join(ckpt, cfg.af_plddt_cache_dir),
                sleep_s=0.05,
                verbose=False,
            )
            if plddt_by_id:
                structure_source = "af2"
                print(f"  AF2 pLDDT fallback: {len(plddt_by_id)} proteins")
        except Exception as exc:
            print(f"  AF2 pLDDT fallback skipped: {exc}")

    cache_dir = getattr(args, "idr_cache_dir", None)
    if cache_dir is None and getattr(args, "idr_cache", False):
        cache_dir = os.path.join(ckpt, "idr_layer_cache")

    skip_ids = set()
    prior_records = None
    resume_path = getattr(args, "idr_resume", None)
    if resume_path:
        from colab.idr_layer_ops import load_idr_layer_jsonl, resume_protein_ids_from_jsonl
        skip_ids = resume_protein_ids_from_jsonl(resume_path)
        prior_records = load_idr_layer_jsonl(resume_path)
        print(f"  Resume: skipping {len(skip_ids)} proteins from {resume_path}")

    package = build_idr_layer_package(
        proteins,
        disorder_by_id,
        plddt_by_id=plddt_by_id,
        function_probs_by_id=function_by_id,
        boltz_std_by_id=boltz_std,
        partners_by_id=partners_by_id,
        ligands_by_id=ligands_by_id,
        disorder_threshold=getattr(args, "idr_disorder_threshold", 0.5),
        function_threshold=function_threshold,
        structure_source=structure_source,
        max_proteins_in_summary=getattr(args, "idr_layer_max_proteins", 100),
        max_workers=getattr(args, "idr_workers", 4),
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        skip_protein_ids=skip_ids,
        prior_records=prior_records,
    )
    report, full_records = package["report"], package["records"]
    if function_oof_metrics is not None:
        report["function_oof_metrics"] = function_oof_metrics
    if function_calibration is not None:
        report["function_calibration"] = function_calibration
    if threshold_tuning is not None:
        report["function_threshold_tuning"] = threshold_tuning
        report["thresholds"]["function_threshold"] = function_threshold
    print_idr_layer_report(report)
    paths = export_idr_layer_bundle(
        out_dir=ckpt,
        report=report,
        records=full_records,
        proteins=proteins,
        disorder_probs_by_id=disorder_by_id,
        function_probs_by_id=function_by_id,
        gzip_jsonl=getattr(args, "idr_gzip", False),
        export_caid=not getattr(args, "idr_no_caid", False),
        export_html=not getattr(args, "idr_no_html", False),
        export_role_bedgraphs=not getattr(args, "idr_no_role_bedgraphs", False),
        export_gff=not getattr(args, "idr_no_gff", False),
        export_cards=not getattr(args, "idr_no_cards", False),
        cards_top_n=getattr(args, "idr_cards_top_n", 20),
        min_triage_score=getattr(args, "idr_min_triage", None),
        quarantine_only=getattr(args, "idr_quarantine_only", False),
        run_args=args,
    )
    compare_path = getattr(args, "idr_compare", None)
    if compare_path:
        from colab.idr_layer_ops import compare_idr_layer_jsonl
        cmp = compare_idr_layer_jsonl(compare_path, paths["jsonl"])
        cmp_out = os.path.join(ckpt, "idr_biology_layer_compare.json")
        with open(cmp_out, "w") as f:
            json.dump(cmp, f, indent=2)
        paths["compare"] = cmp_out
        print(
            f"  Compared vs {compare_path}: shared={cmp['n_shared']}  "
            f"mean|Δtriage|={cmp['mean_abs_triage_delta']}"
        )
    print(f"IDR layer exports: {len(full_records)} proteins → {paths}")
    return report


def stage_structure_distrust_atlas(args, cfg, proteins) -> dict:
    """
    CPU-only structure distrust atlas from pred dir + pLDDT cache.

    Does not require fold OOF (proxy atlas). Optional labels via
    ``--atlas-labels-dir`` of ``*.npy`` arrays keyed by protein id.
    """
    from colab.idr_layer_io import load_disorder_preds_from_dir
    from colab.predict_batch import parse_fasta
    from colab.structure_distrust_atlas import (
        build_structure_distrust_atlas,
        export_structure_distrust_atlas_bundle,
        load_plddt_cache_for_proteins,
        print_distrust_atlas,
    )
    from colab.training_contamination_audit import attach_contamination_flags

    ckpt = cfg.checkpoint_dir
    if getattr(args, "fasta", None):
        proteins = parse_fasta(args.fasta)

    preds_dir = getattr(args, "atlas_preds_dir", None) or getattr(args, "idr_preds_dir", None)
    if not preds_dir:
        raise ValueError("--atlas-preds-dir (or --idr-preds-dir) required for structure-distrust-atlas")
    preds = load_disorder_preds_from_dir(preds_dir, proteins=proteins)
    if not preds:
        raise ValueError(f"No disorder predictions loaded from {preds_dir}")

    plddt_dir = (
        getattr(args, "atlas_plddt_dir", None)
        or os.path.join(ckpt, getattr(cfg, "af_plddt_cache_dir", "af_plddt_cache"))
    )
    plddt_by_id = load_plddt_cache_for_proteins(proteins, plddt_dir)
    if not plddt_by_id:
        raise ValueError(f"No pLDDT cache entries loaded from {plddt_dir}")

    labels_by_id: dict = {}
    labels_dir = getattr(args, "atlas_labels_dir", None)
    if labels_dir and os.path.isdir(labels_dir):
        import numpy as np
        for p in proteins:
            path = os.path.join(labels_dir, f"{p['id']}.npy")
            if os.path.isfile(path):
                labels_by_id[p["id"]] = np.load(path).astype(np.int8)

    structure_source = getattr(args, "atlas_structure_source", None) or "af2"
    atlas = build_structure_distrust_atlas(
        proteins, preds, plddt_by_id,
        labels_by_id=labels_by_id or None,
        disorder_threshold=getattr(args, "idr_disorder_threshold", 0.5),
        structure_source=structure_source,
    )
    atlas = attach_contamination_flags(atlas, cfg)
    print_distrust_atlas(atlas)
    paths = export_structure_distrust_atlas_bundle(atlas, ckpt)
    if not getattr(args, "no_distrust_figures", False):
        try:
            from colab.colab_figures import generate_distrust_atlas_figure
            fig_dir = os.path.join(ckpt, "distrust_figures")
            generate_distrust_atlas_figure(atlas, out_dir=fig_dir)
            paths["figures"] = fig_dir
        except Exception as exc:
            print(f"Distrust figures skipped: {exc}")
    print(f"Structure distrust atlas (CPU) → {paths}")
    return atlas


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
        split_method=getattr(cfg, "split_method", "protein"),
        homology_min_identity=float(getattr(cfg, "homology_min_identity", 0.4)),
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

    aligned = align_fold_predictions(proteins, fold_results, n_folds=cfg.n_folds)
    preds_by_id = {item["id"]: item["probs"] for item in aligned}
    labels_by_id = {item["id"]: item["labels"] for item in aligned}

    manifest = build_af_rescue_manifest(
        proteins, preds_by_id, plddt_by_id, labels_by_id=labels_by_id,
    )
    save_novel_use_case_report(manifest, os.path.join(ckpt, "af_rescue_manifest.json"))

    # Paper pillar 1: labeled distrust benchmark + proteome atlas
    bench = None
    if not getattr(args, "no_structure_distrust_atlas", False):
        import numpy as np

        from colab.hallucination_benchmark import (
            print_distrust_benchmark,
            run_labeled_distrust_benchmark,
            save_distrust_benchmark,
        )
        from colab.structure_distrust_atlas import (
            build_structure_distrust_atlas,
            estimate_downstream_mask_utility,
            export_structure_distrust_atlas_bundle,
            print_distrust_atlas,
        )

        structure_source = "boltz2" if plddt_boltz else ("af3" if plddt_af3 else "af2")
        caid3_report = None
        caid3_path = os.path.join(ckpt, "caid3_eval_report.json")
        if os.path.isfile(caid3_path):
            with open(caid3_path) as f:
                caid3_report = json.load(f)
        bench = run_labeled_distrust_benchmark(
            proteins, fold_results, plddt_by_id,
            n_folds=cfg.n_folds,
            structure_source=structure_source,
            cfg=cfg,
            caid3_report=caid3_report,
        )
        # Attach computational downstream utility on pooled labeled residues
        try:
            ys, ps, plds = [], [], []
            for item in aligned:
                pid = item["id"]
                if pid not in plddt_by_id:
                    continue
                L = len(item["probs"])
                pld = np.asarray(plddt_by_id[pid], dtype=np.float32).ravel()
                if len(pld) < L:
                    continue
                ys.append(np.asarray(item["labels"], dtype=np.int8).ravel()[:L])
                ps.append(np.asarray(item["probs"], dtype=np.float32).ravel()[:L])
                plds.append(pld[:L])
            if ys:
                util = estimate_downstream_mask_utility(
                    np.concatenate(ys), np.concatenate(ps), np.concatenate(plds),
                )
                bench["downstream_mask_utility"] = util
        except Exception as exc:
            bench["downstream_mask_utility"] = {"enabled": False, "error": str(exc)}

        print_distrust_benchmark(bench)
        save_distrust_benchmark(
            bench, os.path.join(ckpt, "structure_distrust_benchmark.json"),
        )

        atlas = build_structure_distrust_atlas(
            proteins, preds_by_id, plddt_by_id,
            labels_by_id=labels_by_id,
            structure_source=structure_source,
        )
        if bench.get("downstream_mask_utility"):
            atlas["downstream_mask_utility"] = bench["downstream_mask_utility"]
        print_distrust_atlas(atlas)
        atlas_paths = export_structure_distrust_atlas_bundle(atlas, ckpt)
        print(f"Structure distrust atlas → {atlas_paths}")
        if not getattr(args, "no_distrust_figures", False):
            try:
                from colab.colab_figures import (
                    generate_distrust_atlas_figure,
                    generate_distrust_benchmark_figure,
                    generate_downstream_mask_utility_figure,
                )
                fig_dir = os.path.join(ckpt, "distrust_figures")
                generate_distrust_benchmark_figure(bench, out_dir=fig_dir)
                generate_distrust_atlas_figure(atlas, out_dir=fig_dir)
                util = bench.get("downstream_mask_utility") or {}
                if util.get("enabled"):
                    generate_downstream_mask_utility_figure(util, out_dir=fig_dir)
                print(f"Distrust figures → {fig_dir}")
            except Exception as exc:
                print(f"Distrust figures skipped: {exc}")

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
        "structure_distrust_rescue_rate": (
            (bench.get("labeled_rescue_report") or {}).get("pooled", {}).get("rescue_rate")
            if bench is not None else None
        ),
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
    boltz_std: dict = {}
    if getattr(args, "boltz_mode", "off") != "off":
        from colab.boltz_plddt import load_boltz_structure_features
        from rockfish.boltz_rockfish import setup_boltz_for_rockfish
        proteins_pre = parse_fasta(args.fasta)
        if args.boltz_mode in ("run", "auto"):
            stage_boltz(args, cfg, proteins_pre)
        bcfg = setup_boltz_for_rockfish(
            mode="ingest", boltz_root=args.boltz_root, ensure_install=False,
        )
        out_root = bcfg["paths"]["output_dir"]
        plddt_by_id, boltz_std = load_boltz_structure_features(
            proteins_pre,
            out_root,
            plddt_cache_dir=os.path.join(cfg.checkpoint_dir, "boltz_plddt_cache"),
            variance_cache_dir=os.path.join(cfg.checkpoint_dir, "boltz_variance_cache"),
            max_samples=getattr(args, "boltz_diffusion_samples", 5),
            verbose=True,
        )
    elif args.prefetch_af_plddt:
        from colab.disordernet_gpu import build_plddt_cache_for_training
        proteins_pre = parse_fasta(args.fasta)
        plddt_by_id = build_plddt_cache_for_training(
            proteins_pre, cache_dir=os.path.join(cfg.checkpoint_dir, cfg.af_plddt_cache_dir),
        )

    want_fn = bool(getattr(cfg, "use_function_head", False)) or getattr(args, "export_idr_layer", False)
    result = predict_fasta_batch(
        fasta_path=args.fasta,
        esm_backbone=model,
        batch_converter=converter,
        cfg=cfg,
        checkpoint_dir=cfg.checkpoint_dir,
        plddt_by_id=plddt_by_id,
        use_tta=not args.skip_tta,
        tta_passes=cfg.mc_dropout_tta_passes,
        n_folds=cfg.n_folds,
        return_function=want_fn,
    )
    if isinstance(result, tuple):
        preds, function_by_id = result
    else:
        preds, function_by_id = result, {}

    proteins = parse_fasta(args.fasta)
    out_dir = args.predict_out or os.path.join(cfg.checkpoint_dir, "predictions")
    manifest = export_predictions(proteins, preds, out_dir, formats=("caid", "tsv"))
    print(f"Predictions written to {out_dir} ({manifest['n_scored']} proteins)")

    if getattr(args, "export_idr_layer", False):
        from colab.idr_biology_layer import (
            build_idr_layer_package,
            export_idr_layer_bundle,
            print_idr_layer_report,
        )
        from colab.idr_layer_io import load_ligand_map, load_partner_map

        partners_by_id = load_partner_map(getattr(args, "idr_partners", None))
        ligands_by_id = load_ligand_map(getattr(args, "idr_ligands", None))
        cache_dir = getattr(args, "idr_cache_dir", None)
        if cache_dir is None and getattr(args, "idr_cache", False):
            cache_dir = os.path.join(out_dir, "idr_layer_cache")
        package = build_idr_layer_package(
            proteins, preds,
            plddt_by_id=plddt_by_id or {},
            function_probs_by_id=function_by_id,
            boltz_std_by_id=boltz_std,
            partners_by_id=partners_by_id,
            ligands_by_id=ligands_by_id,
            disorder_threshold=getattr(args, "idr_disorder_threshold", 0.5),
            function_threshold=getattr(args, "idr_function_threshold", 0.5),
            structure_source=(
                "boltz2"
                if boltz_std or (plddt_by_id and getattr(args, "boltz_mode", "off") != "off")
                else "af2"
            ),
            max_workers=getattr(args, "idr_workers", 4),
            cache_dir=cache_dir,
        )
        report, full = package["report"], package["records"]
        print_idr_layer_report(report)
        paths = export_idr_layer_bundle(
            out_dir=out_dir,
            report=report,
            records=full,
            proteins=proteins,
            disorder_probs_by_id=preds,
            function_probs_by_id=function_by_id,
            gzip_jsonl=getattr(args, "idr_gzip", False),
            export_caid=not getattr(args, "idr_no_caid", False),
            export_html=not getattr(args, "idr_no_html", False),
            export_role_bedgraphs=not getattr(args, "idr_no_role_bedgraphs", False),
            export_gff=not getattr(args, "idr_no_gff", False),
            export_cards=not getattr(args, "idr_no_cards", False),
            cards_top_n=getattr(args, "idr_cards_top_n", 20),
            min_triage_score=getattr(args, "idr_min_triage", None),
            quarantine_only=getattr(args, "idr_quarantine_only", False),
            run_args=args,
        )
        manifest["idr_layer_proteins"] = len(full)
        manifest["idr_layer_paths"] = paths
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
        return_function=False,
    )
    if isinstance(preds_by_id, tuple):
        preds_by_id = preds_by_id[0]

    out_dir = os.path.join(cfg.checkpoint_dir, "caid3_submission")
    export_caid_predictions_dir(ref_proteins, preds_by_id, out_dir)

    report = evaluate_caid_predictions(ref_proteins, preds_by_id)
    print_caid3_eval_report(report)
    save_caid3_eval_report(report, os.path.join(cfg.checkpoint_dir, "caid3_eval_report.json"))

    # Eval typically runs before CAID3; patch the distrust benchmark in-place.
    try:
        from colab.hallucination_benchmark import finalize_distrust_benchmark_with_caid3

        patched = finalize_distrust_benchmark_with_caid3(
            cfg.checkpoint_dir,
            cfg,
            regenerate_figure=not getattr(args, "no_distrust_figures", False),
        )
        if patched and (patched.get("caid3_credibility_floor") or {}).get("available"):
            print("  Attached CAID3 credibility floor → structure_distrust_benchmark.json")
    except Exception as exc:
        print(f"  Warning: could not finalize distrust benchmark with CAID3: {exc}")
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

    elif args.stage == "idr-layer":
        fold_results, cv_summary = _load_fold_results(cfg.checkpoint_dir)
        stage_idr_layer(args, cfg, proteins, fold_results)

    elif args.stage == "structure-distrust-atlas":
        stage_structure_distrust_atlas(args, cfg, proteins)

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
        # Default product export: IDR biology layer (disable with --no-idr-layer)
        if getattr(args, "run_idr_layer", True) and not getattr(args, "no_idr_layer", False):
            stage_idr_layer(args, cfg, proteins, fold_results)
        if args.run_caid3_eval:
            stage_caid3_eval(args, cfg, proteins, fold_results, model, converter)
            # Defense in depth: re-finalize after CAID3 even if stage helper skipped.
            try:
                from colab.hallucination_benchmark import finalize_distrust_benchmark_with_caid3

                finalize_distrust_benchmark_with_caid3(
                    cfg.checkpoint_dir,
                    cfg,
                    regenerate_figure=not getattr(args, "no_distrust_figures", False),
                )
            except Exception as exc:
                print(f"  Warning: pipeline CAID3 finalize skipped: {exc}")

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
            "eval", "idr-layer", "structure-distrust-atlas",
            "predict", "multi-seed-blend", "pipeline", "boltz", "af3",
            "publish-650m", "publish-3b", "package-publish",
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
        "--no-hallucination-weighting",
        action="store_true",
        help="Disable train-time hallucination/pLDDT loss weighting (contamination-clean)",
    )
    p.add_argument(
        "--no-plddt-features",
        action="store_true",
        help="Disable pLDDT input features (contamination-clean ablation)",
    )
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
    p.add_argument(
        "--run-idr-layer", action="store_true", default=True,
        help="pipeline: export IDR biology layer (default on)",
    )
    p.add_argument(
        "--no-idr-layer", action="store_true",
        help="pipeline: skip IDR biology layer export",
    )
    p.add_argument(
        "--no-structure-distrust-atlas", action="store_true",
        help="eval: skip labeled distrust benchmark + proteome atlas export",
    )
    p.add_argument(
        "--no-distrust-figures", action="store_true",
        help="Skip matplotlib distrust paper figures",
    )
    p.add_argument(
        "--atlas-preds-dir", default=None,
        help="structure-distrust-atlas: directory of disorder .tsv/.caid preds",
    )
    p.add_argument(
        "--atlas-plddt-dir", default=None,
        help="structure-distrust-atlas: pLDDT JSON cache directory",
    )
    p.add_argument(
        "--atlas-labels-dir", default=None,
        help="structure-distrust-atlas: optional dir of label .npy arrays",
    )
    p.add_argument(
        "--atlas-structure-source", default="af2",
        help="structure-distrust-atlas: label for structure backend in report",
    )
    p.add_argument(
        "--idr-layer-max-proteins", type=int, default=100,
        help="Max proteins embedded in idr_biology_layer_report.json summary",
    )
    p.add_argument(
        "--export-idr-layer", action="store_true",
        help="predict stage: also write IDR biology layer JSON/JSONL/BED/triage",
    )
    p.add_argument(
        "--idr-preds-dir", default=None,
        help="idr-layer: optional predict-dir of .tsv/.caid scores to overlay",
    )
    p.add_argument(
        "--idr-function-preds-dir", default=None,
        help="idr-layer: optional dir of function .npy/.tsv/JSON to overlay role probs",
    )
    p.add_argument(
        "--idr-partners", default=None,
        help="Optional partner map JSON/TSV for conditioned binding cues",
    )
    p.add_argument(
        "--idr-ligands", default=None,
        help="Optional ligand map JSON/TSV for lipid/NA/metal conditioned cues",
    )
    p.add_argument(
        "--idr-disorder-threshold", type=float, default=0.5,
        help="Disorder probability threshold for IDR segments",
    )
    p.add_argument(
        "--idr-function-threshold", type=float, default=0.5,
        help="Function probability threshold for role calls",
    )
    p.add_argument(
        "--idr-workers", type=int, default=4,
        help="Thread workers for IDR layer proteome build",
    )
    p.add_argument(
        "--idr-gzip", action="store_true",
        help="Write idr_biology_layer.jsonl.gz instead of plain JSONL",
    )
    p.add_argument(
        "--idr-auto-threshold", action="store_true",
        help="idr-layer: tune function role threshold from OOF micro-F1",
    )
    p.add_argument(
        "--idr-cache", action="store_true",
        help="Cache per-protein IDR layer records under checkpoint dir",
    )
    p.add_argument(
        "--idr-cache-dir", default=None,
        help="Explicit IDR layer record cache directory",
    )
    p.add_argument(
        "--idr-compare", default=None,
        help="Compare current JSONL export against a prior JSONL/.gz path",
    )
    p.add_argument(
        "--idr-resume", default=None,
        help="Resume proteome export: skip protein_ids already in this JSONL/.gz",
    )
    p.add_argument(
        "--idr-calibrate-function", action="store_true",
        help="idr-layer: per-group temperature scaling of OOF function probs",
    )
    p.add_argument(
        "--idr-min-triage", type=float, default=None,
        help="Filter triage cards to proteins with triage score ≥ this value",
    )
    p.add_argument(
        "--idr-quarantine-only", action="store_true",
        help="Write triage cards only for quality-quarantined proteins",
    )
    p.add_argument(
        "--idr-cards-top-n", type=int, default=20,
        help="Number of Markdown triage cards to write",
    )
    p.add_argument(
        "--idr-no-html", action="store_true",
        help="Skip HTML summary in the IDR layer bundle",
    )
    p.add_argument(
        "--idr-no-role-bedgraphs", action="store_true",
        help="Skip per-role bedGraph tracks in the IDR layer bundle",
    )
    p.add_argument(
        "--idr-no-gff", action="store_true",
        help="Skip GFF3 IDR segment export",
    )
    p.add_argument(
        "--idr-no-cards", action="store_true",
        help="Skip per-protein Markdown triage cards",
    )
    p.add_argument(
        "--idr-no-caid", action="store_true",
        help="Skip CAID disorder export folder in the IDR layer bundle",
    )
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
    p.add_argument(
        "--boltz-diffusion-samples", type=int, default=5,
        help="Boltz diffusion samples (≥2 enables variance proxy; mean pLDDT uses model_0)",
    )
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

    # Publish submitters (delegate to rockfish/publish_submit.py)
    p.add_argument(
        "--account",
        default=None,
        help="publish-650m / publish-3b: Rockfish _gpu account "
             "(or set DISORDERNET_ACCOUNT)",
    )
    p.add_argument(
        "--publish-root",
        default=None,
        help="publish-*: bundle parent directory (DISORDERNET_PUBLISH_ROOT)",
    )
    p.add_argument(
        "--package-dir",
        default=None,
        help="publish-* / package-publish: output publish_package directory",
    )
    p.add_argument(
        "--bundle-kind",
        default="650m",
        choices=["650m", "3b"],
        help="package-publish: which bundle layout to assemble",
    )
    p.add_argument(
        "--no-clean",
        action="store_true",
        help="publish-*: skip contamination-clean companion",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="publish-*: print sbatch commands without submitting",
    )
    p.add_argument(
        "--partition",
        default=None,
        help="publish-*: GPU partition override (e.g. ica100)",
    )
    return p


def _run_publish_stage(args) -> int:
    """Delegate publish submit / package stages to publish_submit CLI."""
    from rockfish.publish_submit import main as publish_main

    if args.stage == "publish-650m":
        argv = ["submit-650m"]
    elif args.stage == "publish-3b":
        argv = ["submit-3b"]
    elif args.stage == "package-publish":
        if not args.publish_root:
            raise ValueError("package-publish requires --publish-root")
        argv = [
            "package",
            "--root-workdir", args.publish_root,
            "--kind", args.bundle_kind,
        ]
    else:
        raise ValueError(f"Not a publish stage: {args.stage}")

    if getattr(args, "account", None):
        argv.extend(["--account", args.account])
    if getattr(args, "publish_root", None) and args.stage != "package-publish":
        argv.extend(["--root-workdir", args.publish_root])
    if getattr(args, "package_dir", None):
        argv.extend(["--package-dir", args.package_dir])
    if getattr(args, "no_clean", False):
        argv.append("--no-clean")
    if getattr(args, "dry_run", False) and args.stage != "package-publish":
        argv.append("--dry-run")
    if getattr(args, "partition", None) and args.stage != "package-publish":
        argv.extend(["--partition", args.partition])
    return publish_main(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage in ("publish-650m", "publish-3b", "package-publish"):
        try:
            return _run_publish_stage(args)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        except (RuntimeError, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    try:
        return run_pipeline(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
