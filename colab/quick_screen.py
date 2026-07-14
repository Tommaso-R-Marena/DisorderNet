"""
Fast paradigm screen — estimate breakthrough potential before full 18–24h CV.

Runs a stratified protein subset with reduced folds/epochs, GPU + v6 ensemble,
and returns a go/no-go verdict for the current ESM-2 650M + LoRA paradigm.

Important: ``standard`` uses the ``screen_plus`` (mini-ultra) training profile so
the go/no-go reflects the ultra stack, not a toy CNN recipe.
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

# Expected uplift from this screen recipe → full ultra + 7b–7d meta-stack.
# Weak recipes (flash) can jump more; mini-ultra (standard) and paradigm less.
_MODE_ULTRA_UPLIFT: dict[str, tuple[float, float]] = {
    "flash": (0.05, 0.12),
    "standard": (0.02, 0.06),
    "paradigm": (0.01, 0.04),
}

# screen_mode → runtime / rigor trade-off
# Timing: ~2–3 h is an A100-40GB guide; A100-80GB often finishes much faster.
SCREEN_MODES: dict[str, dict[str, Any]] = {
    "flash": {
        "n_proteins": 120,
        "n_folds": 2,
        "train_profile": "screen",
        "description": (
            "~45–90 min A100-40GB / ~20–40 min A100-80GB — coarse go/no-go "
            "(toy CNN recipe; not ultra-faithful)"
        ),
    },
    "standard": {
        "n_proteins": 250,
        "n_folds": 3,
        "train_profile": "screen_plus",
        "description": (
            "~2–3 h A100-40GB / ~40–90 min A100-80GB — recommended mini-ultra screen "
            "(SOTA head + rich features; judges ultra paradigm)"
        ),
    },
    "paradigm": {
        "n_proteins": 350,
        "n_folds": 3,
        "train_profile": "screen_plus",
        "train_profile_3b": "ultra3b",
        "description": (
            "~4–6 h A100-40GB — larger mini-ultra subset (highest 650M fidelity before full CV)"
        ),
        "description_3b": "~8–12 h A100 40GB+ — ESM-2 3B paradigm screen",
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


def expected_ultra_uplift(mode: str) -> tuple[float, float]:
    """Return (lo, hi) expected AUC uplift from screen → full ultra stack."""
    return _MODE_ULTRA_UPLIFT.get(mode, (0.02, 0.06))


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
    Map quick-screen metrics to a breakthrough verdict for the ultra paradigm.

    Projection is **mode-aware**: flash (toy recipe) allows more uplift to full
    ultra than standard/paradigm (mini-ultra). Tiers use the projected full-CV
    band, with a stacked-AUC floor so a very weak screen cannot auto-approve.
    """
    uplift = stacked_pooled_auc - gpu_pooled_auc
    fold_ci = _bootstrap_mean_ci(np.asarray(fold_aucs or [stacked_pooled_auc]))
    up_lo, up_hi = expected_ultra_uplift(mode)

    # If the screen itself is already ultra-strong, keep uplift modest.
    if stacked_pooled_auc >= 0.87:
        up_lo, up_hi = min(up_lo, 0.01), min(up_hi, 0.04)
    elif stacked_pooled_auc >= 0.84:
        up_lo, up_hi = min(up_lo, 0.02), min(up_hi, 0.05)

    proj_lo = float(np.clip(stacked_pooled_auc + up_lo, 0.0, 0.95))
    proj_hi = float(np.clip(stacked_pooled_auc + up_hi, 0.0, 0.96))
    if proj_hi < proj_lo:
        proj_hi = proj_lo

    v6_ceiling_note = ""
    # Only apply a soft v6 ceiling on coarse flash screens (subset v6 is noisy).
    if mode == "flash" and v6_pooled_auc is not None and v6_pooled_auc < 0.80:
        capped = min(proj_hi, v6_pooled_auc + 0.08)
        if capped < proj_hi:
            proj_hi = capped
            v6_ceiling_note = " Coarse v6 subset caps ensemble headroom."

    # Tier by projected full-CV band + stacked floor (anti false-approve).
    if proj_hi >= 0.90 and stacked_pooled_auc >= 0.84 and uplift >= 0.01:
        tier = "HIGH"
        headline = (
            f"Screen stacked AUC {stacked_pooled_auc:.3f} — paradigm can likely reach "
            f"0.88–0.92 full CV (projected {proj_lo:.3f}–{proj_hi:.3f})"
        )
        rec = "Proceed with full QUALITY_PROFILE='ultra' 5-fold CV + 7b–7d stack."
        proceed = True
    elif proj_hi >= 0.88 and stacked_pooled_auc >= 0.80:
        tier = "MODERATE"
        headline = (
            f"Screen stacked AUC {stacked_pooled_auc:.3f} — breakthrough possible "
            f"but not assured (projected {proj_lo:.3f}–{proj_hi:.3f})"
        )
        rec = (
            "Full ultra run is reasonable if you have GPU budget; otherwise run "
            "SCREEN_MODE='paradigm' or SCREEN_BACKBONE='3B' first."
            + v6_ceiling_note
        )
        proceed = proj_hi >= 0.88 and stacked_pooled_auc >= 0.82
    elif proj_hi >= 0.85 and stacked_pooled_auc >= 0.78:
        tier = "LOW"
        headline = (
            f"Screen stacked AUC {stacked_pooled_auc:.3f} — unlikely to hit "
            f"{BREAKTHROUGH_TARGET:.2f} without changes "
            f"(projected {proj_lo:.3f}–{proj_hi:.3f})"
        )
        rec = (
            "Current 650M screen projects below breakthrough. Try paradigm mode, "
            "ESM-2 3B, or architecture changes before a 24h ultra run."
            + v6_ceiling_note
        )
        proceed = False
    else:
        tier = "STOP"
        headline = (
            f"Screen stacked AUC {stacked_pooled_auc:.3f} — paradigm not competitive "
            f"(baseline GPU {VERIFIED_GPU_BASELINE:.3f}; "
            f"projected {proj_lo:.3f}–{proj_hi:.3f})"
        )
        rec = (
            "Do not commit to full ultra CV yet. Confirm you used standard/paradigm "
            "(mini-ultra), then fix data/profile/backbone."
        )
        proceed = False

    if fold_ci.get("ci_low") is not None and fold_ci["ci_low"] < 0.78 and tier == "HIGH":
        tier = "MODERATE"
        headline += " (wide fold CI — treat as moderate)"
        proceed = False

    if mode == "flash" and proceed:
        # Flash is too coarse to green-light a 24h run by itself.
        proceed = False
        rec = (
            "Flash looks promising, but re-run SCREEN_MODE='standard' "
            "(mini-ultra) before committing to full ultra."
        )
        if tier == "HIGH":
            tier = "MODERATE"
            headline += " (flash only — confirm with standard)"

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
    use_v6_pro: bool = True,
    backbone: str = "650M",
    checkpoint_subdir: str = "quick_screen",
) -> dict:
    """
    End-to-end quick screen: subset CV → optional v6 ensemble → verdict.

    Mutates cfg checkpoint_dir for isolation from full CV runs.
    """
    if mode not in SCREEN_MODES:
        raise ValueError(f"Unknown screen mode '{mode}'. Choose: {list(SCREEN_MODES)}")

    spec = SCREEN_MODES[mode]
    train_profile = spec["train_profile"]
    if backbone.upper() in ("3B", "ESM2_3B") and "train_profile_3b" in spec:
        train_profile = spec["train_profile_3b"]
        print(f"  Backbone: ESM-2 3B (profile={train_profile})", flush=True)
    else:
        print(f"  Backbone: ESM-2 {backbone}", flush=True)

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
        train_profile,
        seed=seed,
        n_folds=spec["n_folds"],
        esm_backbone=backbone,
        checkpoint_dir=f"{cfg.checkpoint_dir}/{checkpoint_subdir}_{mode}_{backbone}",
    )
    screen_cfg.device = cfg.device
    screen_cfg.amp_dtype = cfg.amp_dtype
    screen_cfg.pin_memory = cfg.pin_memory
    screen_cfg.num_workers = cfg.num_workers
    screen_cfg.data_cache = cfg.data_cache

    print(
        f"\n  GPU CV: {screen_cfg.n_folds} folds  profile={train_profile}  "
        f"epochs={screen_cfg.num_epochs}  LoRA r={screen_cfg.lora_rank}  "
        f"head={screen_cfg.head_type}",
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
        print("\n  Running v6 ensemble on screen subset…", flush=True)
        v6_cache = f"{screen_cfg.checkpoint_dir}/v6_screen_cache.json"
        from colab.ensemble_v6 import aligned_probs_from_oof, save_v6_probs_cache
        if use_v6_pro:
            from colab.v6_pro_ensemble import run_v6_pro_oof
            print("  Using v6-pro (LGB+XGB) for screen ensemble…", flush=True)
            oof_probs, oof_labels, _ = run_v6_pro_oof(
                subset, n_folds=screen_cfg.n_folds, seed=seed, verbose=True,
            )
        else:
            oof_probs, oof_labels, _ = run_v6_lite_oof(
                subset, n_folds=screen_cfg.n_folds, seed=seed, verbose=True,
            )
        v6_auc = float(roc_auc_score(oof_labels, oof_probs))
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
            use_v6_pro=False,
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
    up_lo, up_hi = expected_ultra_uplift(mode)
    report = {
        "mode": mode,
        "mode_spec": spec,
        "sample": sample_meta,
        "backbone": backbone,
        "v6_pro_ensemble": use_v6_pro,
        "screen_profile": train_profile,
        "expected_ultra_uplift": {"lo": up_lo, "hi": up_hi},
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
    print(f"  Profile       : {report.get('screen_profile', '?')}")
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
