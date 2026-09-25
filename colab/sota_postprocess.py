"""
Post-CV SOTA pipeline: fold checkpoint ensemble + probability calibration.

Chains leakage-aware OOF ensemble modes with temperature / isotonic calibration.
"""

from __future__ import annotations

import json
from typing import Optional

from colab.calibration import calibrate_fold_results, save_calibration_report
from colab.fold_model_soup import print_fold_soup_report, run_fold_model_soup
from colab.inference_fusion import compute_pooled_metrics

# A post-processing step may wobble the pooled metric by rounding; anything
# beyond this is a real regression and must not be silently adopted.
SOUP_REGRESSION_TOLERANCE = 0.002


def run_sota_postprocess(
    proteins: list,
    esm_backbone,
    batch_converter,
    cfg,
    fold_results: list,
    plddt_by_id: Optional[dict] = None,
    checkpoint_dir: Optional[str] = None,
    apply_soup: bool = True,
    soup_mode: str = "held_out",
    calibrate: bool = True,
    calibration_method: str = "temperature",
) -> tuple[dict, list]:
    """
    Run fold-model ensemble then OOF calibration on pooled predictions.

    soup_mode:
      held_out   — each fold checkpoint only on its validation proteins (rigorous OOF)
      full_soup  — average all fold models on each val protein (optimistic CV bias)
    """
    report: dict = {"steps": []}
    current = fold_results

    if apply_soup:
        before_soup = compute_pooled_metrics(current)
        soup_report, souped = run_fold_model_soup(
            proteins=proteins,
            esm_backbone=esm_backbone,
            batch_converter=batch_converter,
            cfg=cfg,
            fold_results=current,
            plddt_by_id=plddt_by_id,
            checkpoint_dir=checkpoint_dir,
            mode=soup_mode,
        )
        # An ensembling step that makes the pooled metric worse is a defect, not
        # a result. Adopting its output regardless is how a 0.11 AUC regression
        # reached final_pooled in the 650M run (0.7999 -> 0.6867) with nothing
        # flagged. Keep the better predictions and record the regression loudly
        # so it gets diagnosed rather than published.
        after_soup = compute_pooled_metrics(souped)
        delta = after_soup["auc"] - before_soup["auc"]
        soup_report["delta_auc_pooled"] = delta
        if delta < -SOUP_REGRESSION_TOLERANCE:
            soup_report["regression_detected"] = True
            soup_report["applied"] = False
            soup_report["note"] = (
                f"Fold soup lowered pooled AUC by {abs(delta):.4f} "
                f"({before_soup['auc']:.4f} -> {after_soup['auc']:.4f}). Output "
                "discarded and pre-soup predictions kept. A soup that degrades "
                "the metric indicates the reloaded checkpoints do not reproduce "
                "the predictions they were saved from — diagnose before trusting "
                "any downstream number from this stage."
            )
            print(f"\n  ⚠ FOLD SOUP REGRESSION: {soup_report['note']}\n")
        else:
            soup_report["regression_detected"] = False
            soup_report["applied"] = True
            current = souped
        report["fold_soup"] = soup_report
        report["steps"].append("fold_soup")

    if calibrate:
        before = compute_pooled_metrics(current)
        calibrated, cal_report = calibrate_fold_results(current, method=calibration_method)
        after = compute_pooled_metrics(calibrated)
        cal_report["before"] = {"pooled": {k: before[k] for k in ("auc", "ap", "n_residues")}}
        cal_report["after"] = {"pooled": {k: after[k] for k in ("auc", "ap", "n_residues")}}
        cal_report["delta_auc_pooled"] = after["auc"] - before["auc"]
        cal_report["delta_ap_pooled"] = after["ap"] - before["ap"]
        # Temperature scaling is strictly monotone and cannot move AUC at all;
        # isotonic can, and leave-one-fold-out isotonic can lose ranking
        # resolution by collapsing distinct scores into ties. Calibration exists
        # to fix probability scale, not to cost discrimination — so a material
        # AUC loss here is a regression, and the uncalibrated ranking is kept.
        cal_delta = after["auc"] - before["auc"]
        if cal_delta < -SOUP_REGRESSION_TOLERANCE:
            cal_report["regression_detected"] = True
            cal_report["applied"] = False
            cal_report["note"] = (
                f"Calibration lowered pooled AUC by {abs(cal_delta):.4f} "
                f"({before['auc']:.4f} -> {after['auc']:.4f}); uncalibrated "
                "predictions kept. Probabilities are then uncalibrated — report "
                "AUC/AP, not calibration-dependent metrics, until this is fixed."
            )
            print(f"\n  ⚠ CALIBRATION REGRESSION: {cal_report['note']}\n")
        else:
            cal_report["regression_detected"] = False
            cal_report["applied"] = True
            current = calibrated
        report["calibration"] = cal_report
        report["steps"].append("calibration")

    pooled = compute_pooled_metrics(current)
    report["final_pooled"] = {k: pooled[k] for k in ("auc", "ap", "n_residues")}
    report["benchmark_comparability"] = (
        "DisProt homology-CV pooled AUC. NOT comparable to CAID3 figures such as ESMDisPred 0.895: different label definition (curator-annotated functional disorder vs missing residues in crystal structures), different proteins, different protocol. The comparable measurement is caid3_eval_report.json."
    )
    return report, current


def print_sota_postprocess_report(report: dict) -> None:
    print(f"\n{'═' * 64}")
    print(" SOTA POST-PROCESS (fold soup + calibration)")
    print(f"{'═' * 64}")
    if "fold_soup" in report:
        print_fold_soup_report(report["fold_soup"])
    if "calibration" in report:
        cal = report["calibration"]
        print(f"\n  Calibration ({cal.get('method', 'n/a')})")
        if cal.get("insufficient_data"):
            print("  Skipped — insufficient label diversity")
        else:
            b, a = cal["before"]["pooled"], cal["after"]["pooled"]
            print(f"  Before : AUC={b['auc']:.4f}  AP={b['ap']:.4f}")
            print(f"  After  : AUC={a['auc']:.4f}  AP={a['ap']:.4f}")
            print(f"  Δ AUC  : {cal.get('delta_auc_pooled', 0):+.4f}")
            if "temperature" in cal:
                print(f"  Temperature T = {cal['temperature']:.3f}")
    fp = report.get("final_pooled", {})
    if fp:
        print(f"\n  Final pooled AUC={fp.get('auc', 0):.4f}  AP={fp.get('ap', 0):.4f}")
    print(f"{'═' * 64}")


def save_sota_postprocess_report(report: dict, path: str = "sota_postprocess_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    if "calibration" in report:
        save_calibration_report(report["calibration"], "calibration_report.json")
    return path
