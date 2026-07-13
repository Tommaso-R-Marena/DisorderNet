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
        soup_report, current = run_fold_model_soup(
            proteins=proteins,
            esm_backbone=esm_backbone,
            batch_converter=batch_converter,
            cfg=cfg,
            fold_results=current,
            plddt_by_id=plddt_by_id,
            checkpoint_dir=checkpoint_dir,
            mode=soup_mode,
        )
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
        report["calibration"] = cal_report
        report["steps"].append("calibration")
        current = calibrated

    pooled = compute_pooled_metrics(current)
    report["final_pooled"] = {k: pooled[k] for k in ("auc", "ap", "n_residues")}
    report["gap_to_esmdispred"] = 0.895 - pooled["auc"]
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
        print(f"  Gap→ESMDisPred (0.895): {report.get('gap_to_esmdispred', 0):+.4f}")
    print(f"{'═' * 64}")


def save_sota_postprocess_report(report: dict, path: str = "sota_postprocess_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    if "calibration" in report:
        save_calibration_report(report["calibration"], "calibration_report.json")
    return path
