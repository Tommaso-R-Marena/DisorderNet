#!/usr/bin/env python3
"""Which reported statistic actually reproduces across identical reruns?

Four runs of the same configuration (ultra / 650M / DisProt / homology splits,
same seed, same code) sit in this project's results tree. They give a direct
measurement of how much each headline number moves when nothing changes:

      run     GPU pooled  mean-folds   v6 pooled    stacked
  j29683091       0.7203      0.7491      0.7804     0.7876
  j29683038       0.7137      0.7514      0.7804     0.7909
  j29682871       0.7454      0.7551      0.7804     0.7999
  j29683037       0.7234      0.7405      0.7804     0.7920

  GPU pooled AUC      sd 0.0138   range 0.0317
  GPU mean-of-folds   sd 0.0062   range 0.0147

Pooled AUC is roughly twice as noisy across reruns as mean-of-folds. That is
what the cross-fold calibration analysis predicts: pooled AUC ranks residues
scored by five separately-calibrated fold models, so it carries the variance of
those five calibrations on top of the model's own. Mean-of-folds averages five
AUCs that were each computed within one calibration and never compares across
them. Prefer mean-of-folds as the headline, and quote pooled only alongside its
rank-normalised counterpart.

Caveat, and it matters: v6 is identical to four decimal places in all four runs
because they share one cached OOF prediction file. That is one measurement
reused, not four independent ones, so the GBDT's own reproducibility is
unmeasured here.

The GBDT beats the 69.9M-parameter neural model in every run, by +0.035 to
+0.067 (mean +0.055) — the observation that motivates the `lite` profile.

Usage:
    python rockfish/rerun_stability.py <results_root> [<results_root> ...]
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


def collect_runs(roots: list[str]) -> list[dict]:
    """One row per completed homology-split run found under any root."""
    rows = []
    for root in roots:
        for run_dir in sorted(glob.glob(os.path.join(root, "*/"))):
            summary_path = os.path.join(run_dir, "cv_summary.json")
            ensemble_path = os.path.join(run_dir, "gpu_v6_ensemble_report.json")
            if not (os.path.isfile(summary_path) and os.path.isfile(ensemble_path)):
                continue
            try:
                cv = json.load(open(summary_path))
                ens = json.load(open(ensemble_path))
            except (OSError, ValueError):
                continue
            if not cv.get("fold_aucs"):
                continue
            if (cv.get("config") or {}).get("split_method") != "homology":
                continue
            # The ensemble weight curve's endpoints are the only place the
            # components are reported unmixed: weight 0 is the neural model
            # alone, weight 1 the physics GBDT alone.
            curve = {round(c["weight"], 2): c["auc"]
                     for c in ens.get("weight_search", {}).get("curve", [])}
            rows.append({
                "run": os.path.basename(run_dir.rstrip("/")),
                "gpu_pooled": curve.get(0.0),
                "v6_pooled": curve.get(1.0),
                "mean_folds": float(np.mean(cv["fold_aucs"])),
                "fold_aucs": cv["fold_aucs"],
                "stacked": cv.get("stacked_pooled_auc", cv.get("pooled_auc")),
            })
    return rows


def summarize(rows: list[dict]) -> dict:
    out: dict = {"n_runs": len(rows), "runs": rows, "statistics": {}}
    for key, label in (
        ("gpu_pooled", "neural model, pooled OOF AUC"),
        ("mean_folds", "neural model, mean of fold AUCs"),
        ("v6_pooled", "v6-pro physics GBDT, pooled"),
        ("stacked", "GPU + v6 + meta-stack, pooled"),
    ):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if len(vals) < 2:
            continue
        out["statistics"][key] = {
            "label": label,
            "n": len(vals),
            "mean": round(float(np.mean(vals)), 4),
            "sd": round(float(np.std(vals, ddof=1)), 4),
            "range": round(float(max(vals) - min(vals)), 4),
            "values": [round(v, 4) for v in vals],
        }

    pairs = [(r["v6_pooled"], r["gpu_pooled"]) for r in rows
             if r.get("v6_pooled") is not None and r.get("gpu_pooled") is not None]
    if pairs:
        deltas = [v6 - gpu for v6, gpu in pairs]
        out["gbdt_advantage"] = {
            "per_run": [round(d, 4) for d in deltas],
            "mean": round(float(np.mean(deltas)), 4),
            "min": round(float(min(deltas)), 4),
            "max": round(float(max(deltas)), 4),
            "gbdt_wins_every_run": all(d > 0 for d in deltas),
        }

    stats = out["statistics"]
    if "gpu_pooled" in stats and "mean_folds" in stats:
        pooled_sd, mean_sd = stats["gpu_pooled"]["sd"], stats["mean_folds"]["sd"]
        out["recommended_headline"] = {
            "statistic": "mean_folds" if mean_sd < pooled_sd else "gpu_pooled",
            "reason": (
                f"pooled AUC sd {pooled_sd:.4f} vs mean-of-folds sd {mean_sd:.4f} "
                f"across {stats['gpu_pooled']['n']} identical reruns. Pooled AUC "
                "ranks residues scored by separately-calibrated fold models, so "
                "it carries their calibration variance on top of the model's."
            ),
        }
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    rows = collect_runs(argv[1:])
    if len(rows) < 2:
        print(f"Found {len(rows)} completed homology-split run(s); need >= 2.",
              file=sys.stderr)
        return 1

    report = summarize(rows)
    print(f"{'run':>34} {'GPU pooled':>11} {'mean-folds':>11} "
          f"{'v6 pooled':>10} {'stacked':>9}")
    for r in rows:
        def fmt(x):
            return f"{x:.4f}" if x is not None else "    --"
        print(f"{r['run'][:34]:>34} {fmt(r['gpu_pooled']):>11} "
              f"{fmt(r['mean_folds']):>11} {fmt(r['v6_pooled']):>10} "
              f"{fmt(r['stacked']):>9}")

    print("\nspread across identical reruns:")
    for s in report["statistics"].values():
        print(f"  {s['label']:34s} n={s['n']}  sd={s['sd']:.4f}  range={s['range']:.4f}")

    adv = report.get("gbdt_advantage")
    if adv:
        print(f"\nGBDT advantage over the neural model: mean {adv['mean']:+.4f} "
              f"(range {adv['min']:+.4f} to {adv['max']:+.4f}); "
              f"wins every run: {adv['gbdt_wins_every_run']}")

    rec = report.get("recommended_headline")
    if rec:
        print(f"\nheadline statistic → {rec['statistic']}\n  {rec['reason']}")

    if len({r["v6_pooled"] for r in rows if r.get("v6_pooled")}) == 1 and len(rows) > 1:
        print("\n  NOTE: v6 is identical in every run because they share one cached "
              "OOF file.\n  That is one measurement reused, not several independent "
              "ones — the GBDT's\n  own reproducibility is not measured here.")

    out_path = os.path.join(argv[1], "rerun_stability.json")
    try:
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote {out_path}")
    except OSError as exc:
        print(f"\nCould not write report: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
