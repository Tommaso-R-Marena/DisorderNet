"""
Fast paradigm screen — estimate breakthrough potential before full 18–24h CV.

Runs a stratified protein subset with reduced folds/epochs, GPU + v6 ensemble,
and returns a go/no-go verdict for the current ESM-2 650M + LoRA paradigm.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from colab.cv_splits import sort_proteins_deterministic
from colab.ensemble_v6 import apply_gpu_v6_ensemble, run_v6_lite_oof
from colab.inference_fusion import compute_pooled_metrics
from colab.statistical_validation import _bootstrap_mean_ci

BREAKTHROUGH_TARGET = 0.90
ESMDISPRED_REFERENCE = 0.895
VERIFIED_GPU_BASELINE = 0.817
VERIFIED_V6_BASELINE = 0.831

# screen_mode → runtime / rigor trade-off (A100 rough guides)
SCREEN_MODES: dict[str, dict[str, Any]] = {
    "flash": {
        "n_proteins": 120,
        "n_folds": 2,
        "train_profile": "screen",
        "description": "~45–90 min A100 — coarse go/no-go",
    },
    "standard": {
        "n_proteins": 250,
        "n_folds": 3,
        "train_profile": "screen",
        "description": "~2–3 h A100 — recommended screen",
    },
    "paradigm": {
        "n_proteins": 350,
        "n_folds": 3,
        "train_profile": "screen_plus",
        "description": "~4–6 h A100 — mini-ultra paradigm fidelity",
    },
}


@dataclass
class BreakthroughVerdict:
    tier: str
    headline: str
    recommendation: str
    projected_full_auc_low: float
    projected_full_auc_high: float
    proceed_full_ultra: bool


def subsample_proteins_stratified(
    proteins: list,
    n_target: int,
    seed: int = 42,
) -> tuple[list, dict]:
    """
    Deterministic stratified subsample by disorder fraction.

    Preserves low / mid / high disorder-rate proteins for representative screen.
    """
    proteins = sort_proteins_deterministic(proteins)
    if len(proteins) <= n_target:
        return proteins, {
            "n_sampled": len(proteins),
            "n_total": len(proteins),
            "subsampled": False,
            "buckets": {},
        }

    buckets: dict[str, list] = {"low": [], "mid": [], "high": []}
    for p in proteins:
        frac = p["n_dis"] / max(p["length"], 1)
        if frac < 0.12:
            buckets["low"].append(p)
        elif frac < 0.30:
            buckets["mid"].append(p)
        else:
            buckets["high"].append(p)

    rng = np.random.default_rng(seed)
    selected: list = []
    bucket_counts: dict[str, int] = {}
    for name, bucket in buckets.items():
        if not bucket:
            bucket_counts[name] = 0
            continue
        share = max(1, int(round(n_target * len(bucket) / len(proteins))))
        share = min(share, len(bucket))
        idx = rng.choice(len(bucket), size=share, replace=False)
        picked = [bucket[i] for i in sorted(idx)]
        selected.extend(picked)
        bucket_counts[name] = len(picked)

    # Top up or trim to n_target
    if len(selected) > n_target:
        selected = sort_proteins_deterministic(selected)[:n_target]
    elif len(selected) < n_target:
        remaining = [p for p in proteins if p not in selected]
        rng.shuffle(remaining)
        need = n_target - len(selected)
        selected.extend(remaining[:need])
        selected = sort_proteins_deterministic(selected)

    meta = {
        "n_sampled": len(selected),
        "n_total": len(proteins),
        "subsampled": True,
        "buckets": bucket_counts,
        "n_residues": sum(p["length"] for p in selected),
        "seed": seed,
    }
    return selected, meta


def assess_breakthrough_potential(
    gpu_pooled_auc: float,
    stacked_pooled_auc: float,
    v6_pooled_auc: Optional[float] = None,
    fold_aucs: Optional[list[float]] = None,
    mode: str = "standard",
) -> BreakthroughVerdict:
    """
    Map quick-screen metrics to a breakthrough verdict for the current paradigm.

    Uses conservative projection: full ultra + meta-stack typically adds 0.01–0.04
    over a screen run, rarely more unless screen already ≥0.87.
    """
    uplift = stacked_pooled_auc - gpu_pooled_auc
    fold_ci = _bootstrap_mean_ci(np.asarray(fold_aucs or [stacked_pooled_auc]))

    # Conservative projection band for full DisProt 5-fold ultra pipeline
    if stacked_pooled_auc >= 0.87:
        proj_lo = stacked_pooled_auc + 0.01
        proj_hi = min(0.92, stacked_pooled_auc + 0.04)
    elif stacked_pooled_auc >= 0.84:
        proj_lo = stacked_pooled_auc
        proj_hi = stacked_pooled_auc + 0.03
    else:
        proj_lo = max(0.0, stacked_pooled_auc - 0.01)
        proj_hi = stacked_pooled_auc + 0.02

    v6_ceiling_note = ""
    if v6_pooled_auc is not None and v6_pooled_auc < 0.84:
        proj_hi = min(proj_hi, v6_pooled_auc + 0.06)
        v6_ceiling_note = " v6 subset ceiling limits ensemble headroom."

    if stacked_pooled_auc >= 0.88 and uplift >= 0.02:
        tier = "HIGH"
        headline = (
            f"Screen stacked AUC {stacked_pooled_auc:.3f} — paradigm can likely reach "
            f"0.88–0.92 full CV"
        )
        rec = "Proceed with full QUALITY_PROFILE='ultra' 5-fold CV + 7b–7d stack."
        proceed = True
    elif stacked_pooled_auc >= 0.85 or (
        stacked_pooled_auc >= 0.83 and uplift >= 0.025
    ):
        tier = "MODERATE"
        headline = (
            f"Screen stacked AUC {stacked_pooled_auc:.3f} — breakthrough possible "
            f"but not assured (projected {proj_lo:.3f}–{proj_hi:.3f})"
        )
        rec = (
            "Full ultra run is reasonable if you have GPU budget; otherwise tune "
            "ensemble/ultra first." + v6_ceiling_note
        )
        proceed = proj_hi >= 0.88
    elif stacked_pooled_auc >= 0.82:
        tier = "LOW"
        headline = (
            f"Screen stacked AUC {stacked_pooled_auc:.3f} — unlikely to hit "
            f"{BREAKTHROUGH_TARGET:.2f} without changes"
        )
        rec = (
            "Current paradigm probably caps ~0.85–0.87 full CV. Consider backbone "
            "upgrade or CAID3-targeted training before 24h run." + v6_ceiling_note
        )
        proceed = False
    else:
        tier = "STOP"
        headline = (
            f"Screen stacked AUC {stacked_pooled_auc:.3f} — paradigm not competitive "
            f"(baseline GPU {VERIFIED_GPU_BASELINE:.3f})"
        )
        rec = "Do not commit to full ultra CV; fix data, profile, or architecture first."
        proceed = False

    if fold_ci.get("ci_low") is not None and fold_ci["ci_low"] < 0.80 and tier == "HIGH":
        tier = "MODERATE"
        headline += " (wide fold CI — treat as moderate)"
        proceed = False

    return BreakthroughVerdict(
        tier=tier,
        headline=headline,
        recommendation=rec,
        projected_full_auc_low=proj_lo,
        projected_full_auc_high=proj_hi,
        proceed_full_ultra=proceed,
    )


def run_paradigm_quick_screen(
    proteins: list,
    esm_backbone,
    batch_converter,
    cfg,
    mode: str = "standard",
    seed: int = 42,
    run_ensemble: bool = True,
    checkpoint_subdir: str = "quick_screen",
) -> dict:
    """
    End-to-end quick screen: subset CV → optional v6 ensemble → verdict.

    Mutates cfg checkpoint_dir for isolation from full CV runs.
    """
    if mode not in SCREEN_MODES:
        raise ValueError(f"Unknown screen mode '{mode}'. Choose: {list(SCREEN_MODES)}")

    spec = SCREEN_MODES[mode]
    screen_t0 = time.time()

    print(f"\n{'═' * 64}")
    print(f" PARADIGM QUICK SCREEN  │  mode={mode}")
    print(f" {spec['description']}")
    print(f"{'═' * 64}")

    subset, sample_meta = subsample_proteins_stratified(
        proteins, n_target=spec["n_proteins"], seed=seed,
    )
    print(
        f"  Subsample: {sample_meta['n_sampled']}/{sample_meta['n_total']} proteins  "
        f"({sample_meta.get('n_residues', sum(p['length'] for p in subset)):,} residues)",
        flush=True,
    )
    if sample_meta.get("buckets"):
        print(f"  Buckets (low/mid/high): {sample_meta['buckets']}", flush=True)

    from colab.disordernet_gpu import TrainConfig, run_cross_validation

    screen_cfg = TrainConfig.from_profile(
        spec["train_profile"],
        seed=seed,
        n_folds=spec["n_folds"],
        checkpoint_dir=f"{cfg.checkpoint_dir}/{checkpoint_subdir}_{mode}",
    )
    screen_cfg.device = cfg.device
    screen_cfg.amp_dtype = cfg.amp_dtype
    screen_cfg.pin_memory = cfg.pin_memory
    screen_cfg.num_workers = cfg.num_workers
    screen_cfg.data_cache = cfg.data_cache

    print(
        f"\n  GPU CV: {screen_cfg.n_folds} folds  profile={spec['train_profile']}  "
        f"epochs={screen_cfg.num_epochs}  LoRA r={screen_cfg.lora_rank}",
        flush=True,
    )

    fold_results, cv_summary = run_cross_validation(
        proteins=subset,
        esm_backbone=esm_backbone,
        batch_converter=batch_converter,
        cfg=screen_cfg,
        prefetch_af_plddt=False,
    )

    gpu_metrics = compute_pooled_metrics(fold_results)
    gpu_auc = gpu_metrics["auc"]
    fold_aucs = [r["best_auc"] for r in fold_results]

    print(f"\n  GPU screen pooled AUC: {gpu_auc:.4f}  AP: {gpu_metrics['ap']:.4f}", flush=True)

    stacked_auc = gpu_auc
    ensemble_report: dict = {"skipped": True}
    v6_auc: Optional[float] = None

    if run_ensemble:
        print("\n  Running v6-lite ensemble on screen subset…", flush=True)
        v6_cache = f"{screen_cfg.checkpoint_dir}/v6_screen_cache.json"
        oof_probs, oof_labels, _ = run_v6_lite_oof(
            subset, n_folds=screen_cfg.n_folds, seed=seed, verbose=True,
        )
        v6_auc = float(roc_auc_score(oof_labels, oof_probs))
        from colab.ensemble_v6 import aligned_probs_from_oof, save_v6_probs_cache
        v6_by_id = aligned_probs_from_oof(subset, oof_probs)
        save_v6_probs_cache(v6_by_id, v6_cache)

        ensemble_report, fold_results, _ = apply_gpu_v6_ensemble(
            proteins=subset,
            fold_results=fold_results,
            n_folds=screen_cfg.n_folds,
            v6_probs_by_id=v6_by_id,
            v6_cache_path=v6_cache,
            run_v6_if_missing=False,
            seed=seed,
        )

        stacked_metrics = compute_pooled_metrics(fold_results)
        stacked_auc = stacked_metrics["auc"]
        print(
            f"  v6 screen AUC: {v6_auc:.4f}  │  stacked: {stacked_auc:.4f}  "
            f"(Δ={stacked_auc - gpu_auc:+.4f})",
            flush=True,
        )

    verdict = assess_breakthrough_potential(
        gpu_pooled_auc=gpu_auc,
        stacked_pooled_auc=stacked_auc,
        v6_pooled_auc=v6_auc,
        fold_aucs=fold_aucs,
        mode=mode,
    )

    elapsed_h = (time.time() - screen_t0) / 3600
    report = {
        "mode": mode,
        "mode_spec": spec,
        "sample": sample_meta,
        "screen_profile": spec["train_profile"],
        "gpu": {
            "pooled_auc": float(gpu_auc),
            "pooled_ap": float(gpu_metrics["ap"]),
            "fold_aucs": [float(a) for a in fold_aucs],
            "mean_fold_auc": float(np.mean(fold_aucs)),
        },
        "v6_screen_auc": v6_auc,
        "stacked": {
            "pooled_auc": float(stacked_auc),
            "ensemble": ensemble_report,
        },
        "verdict": {
            "tier": verdict.tier,
            "headline": verdict.headline,
            "recommendation": verdict.recommendation,
            "projected_full_auc_low": verdict.projected_full_auc_low,
            "projected_full_auc_high": verdict.projected_full_auc_high,
            "proceed_full_ultra": verdict.proceed_full_ultra,
        },
        "references": {
            "breakthrough_target": BREAKTHROUGH_TARGET,
            "esmdispred_caid3": ESMDISPRED_REFERENCE,
            "verified_gpu_baseline": VERIFIED_GPU_BASELINE,
            "verified_v6_baseline": VERIFIED_V6_BASELINE,
        },
        "elapsed_hours": float(elapsed_h),
        "cv_summary": cv_summary,
    }
    return report


def print_quick_screen_report(report: dict) -> None:
    """Pretty-print screen results and verdict."""
    v = report["verdict"]
    refs = report["references"]
    print(f"\n{'═' * 64}")
    print(" QUICK SCREEN RESULTS")
    print(f"{'═' * 64}")
    print(f"  Mode          : {report['mode']} ({report['mode_spec']['description']})")
    print(f"  Sample        : {report['sample']['n_sampled']} proteins")
    print(f"  GPU AUC       : {report['gpu']['pooled_auc']:.4f}")
    if report.get("v6_screen_auc") is not None:
        print(f"  v6 AUC        : {report['v6_screen_auc']:.4f}")
    print(f"  Stacked AUC   : {report['stacked']['pooled_auc']:.4f}")
    print(f"  Elapsed       : {report['elapsed_hours']:.2f} h")
    print(f"\n── Breakthrough assessment (target {refs['breakthrough_target']:.2f}) ──")
    print(f"  Tier          : {v['tier']}")
    print(f"  {v['headline']}")
    print(f"  Projected full CV: {v['projected_full_auc_low']:.3f} – {v['projected_full_auc_high']:.3f}")
    print(f"  → {v['recommendation']}")
    print(f"\n  Proceed full ultra?  {'YES ✓' if v['proceed_full_ultra'] else 'NO ✗'}")
    print(f"{'═' * 64}\n")


def save_quick_screen_report(report: dict, path: str = "quick_screen_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
