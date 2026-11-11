#!/usr/bin/env python3
"""Close the registered endpoints of PREREGISTRATION_7, _8 and _9.

Three variants were trained against a regime-matched control (`mt_control`,
which shares their training union and validation holdout) and their pooled CAID3
scores were recorded, but the *registered* endpoints of two of them are stated
on statistics the CAID3 evaluator does not write:

  PREREG_7 (`mt_motif`)  P1  beat `mt_control` on pooled Binding-IDR
                         P2  within-protein AUC on Binding-IDR >= 0.7620
  PREREG_8 (`mt_wass`)   P1  beat `mt_control` on pooled Disorder-NOX
                         P2  beat `mt_control` on BETWEEN-protein Disorder-NOX
  PREREG_9 (`mt_rank`)   P1  beat `mt_control` on WITHIN-protein AUC
                         P2  within-protein AUC on Disorder-NOX >= 0.8564

A paper that lists these as "rejected" without evaluating the endpoint they were
registered on is asserting an outcome it did not measure. This computes them.

Every comparison is paired within target against the same control, on the
official references, and reports the pooled/within/between decomposition of both
sides so the mechanism claim of each registration can be checked rather than
inferred from the pooled number.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.auc_decomposition import decompose_auc  # noqa: E402
from colab.caid3_official import (  # noqa: E402
    evaluated_mask,
    holm_bonferroni,
    read_caid_predictions,
    read_reference,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("RE_REFS", f"{ROOT}/caid3_official")
OUT = os.environ.get("RE_OUT", "")

#: variant -> (registration, task the endpoint is stated on, statistic, bar)
ENDPOINTS = {
    "multitask_motif": ("PREREGISTRATION_7", "binding_idr", "pooled", 0.7620),
    "multitask_wass": ("PREREGISTRATION_8", "disorder_nox", "between", None),
    "multitask_rank": ("PREREGISTRATION_9", "disorder_nox", "within", 0.8564),
}
CONTROL = "multitask_control"
TASKS = ("disorder_pdb", "disorder_nox", "binding", "binding_idr", "linker")


def load(run, task):
    p = os.path.join(ROOT, run, "caid_submissions",
                     f"DisorderNet-{task}.caid")
    return read_caid_predictions(p) if os.path.isfile(p) else None


def pooled_auc(ref, pred):
    """The official statistic: every evaluated residue of every answered target.

    Not the same as pooling the two-class targets `arrays` keeps. On Binding-IDR
    ten of 52 targets carry one class among evaluated residues, and they carry
    real weight in the pooled number even though they contribute no within-target
    pair: restricting to two-class targets moves `mt_motif` from 0.5281 to
    0.6888. The registered endpoints of PREREG_7 and _8 are stated on the
    official pooled AUC, so that is what is compared against them.
    """
    from sklearn.metrics import roc_auc_score

    ys, ss = [], []
    for tid, (_seq, lab) in ref.items():
        if tid not in pred:
            continue
        m = evaluated_mask(lab)
        y = np.frombuffer(lab.encode(), np.uint8)[m].astype(np.int8) - ord("0")
        s = np.asarray(pred[tid])[m]
        if not np.isfinite(s).all():
            continue
        ys.append(y)
        ss.append(s)
    if not ys:
        return None
    y, s = np.concatenate(ys), np.concatenate(ss)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None


def arrays(ref, pred):
    """Per-target (labels, scores) on evaluated residues, both classes only."""
    ys, ss, ids = [], [], []
    for tid, (_seq, lab) in ref.items():
        if tid not in pred:
            continue
        m = evaluated_mask(lab)
        y = np.frombuffer(lab.encode(), np.uint8)[m].astype(np.int8) - ord("0")
        s = np.asarray(pred[tid])[m]
        if len(np.unique(y)) < 2 or not np.isfinite(s).all():
            continue
        ys.append(y)
        ss.append(s)
        ids.append(tid)
    return ys, ss, ids


def paired(ref, a, b):
    """Per-target AUC differences on the targets both methods answered."""
    from sklearn.metrics import roc_auc_score

    ya, sa, ia = arrays(ref, a)
    yb, sb, ib = arrays(ref, b)
    lookup = dict(zip(ib, zip(yb, sb)))
    d, kept = [], []
    for tid, y, s in zip(ia, ya, sa):
        if tid not in lookup:
            continue
        _yb, s2 = lookup[tid]
        d.append(roc_auc_score(y, s) - roc_auc_score(y, s2))
        kept.append(tid)
    return np.asarray(d), kept


def main() -> int:
    from scipy.stats import wilcoxon

    report, pvals = {}, {}
    for task in TASKS:
        path = os.path.join(REFS, f"{task}.fasta")
        if not os.path.isfile(path):
            continue
        ref = read_reference(path)
        ctrl = load(CONTROL, task)
        if ctrl is None:
            print(f"no control submission for {task}", file=sys.stderr)
            continue
        yc, sc, _ = arrays(ref, ctrl)
        dc = decompose_auc(yc, sc)
        dc["pooled_all_targets"] = pooled_auc(ref, ctrl)

        for run in ENDPOINTS:
            pred = load(run, task)
            if pred is None:
                continue
            yv, sv, _ = arrays(ref, pred)
            dv = decompose_auc(yv, sv)
            dv["pooled_all_targets"] = pooled_auc(ref, pred)
            d_i, kept = paired(ref, pred, ctrl)
            if len(d_i) < 20:
                continue
            w = wilcoxon(d_i, alternative="two-sided", zero_method="wilcox")
            rng = np.random.default_rng(20260820)
            boot = np.array([d_i[rng.integers(0, len(d_i), len(d_i))].mean()
                             for _ in range(10000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])

            key = f"{task}: {run} vs {CONTROL}"
            report[key] = {
                "n_targets": len(d_i),
                "wins": int((d_i > 0).sum()),
                "mean_per_target": float(d_i.mean()),
                "ci": [float(lo), float(hi)],
                "p_wilcoxon": float(w.pvalue),
                "variant": {k: dv.get(k) for k in
                            ("pooled", "pooled_all_targets", "auc_within",
                             "auc_within_unweighted", "auc_between",
                             "w_within")},
                "control": {k: dc.get(k) for k in
                            ("pooled", "pooled_all_targets", "auc_within",
                             "auc_within_unweighted", "auc_between",
                             "w_within")},
            }
            pvals[key] = float(w.pvalue)
            print(f"\n{key}")
            print(f"  pooled   {dv['pooled_all_targets']:.4f} vs "
                  f"{dc['pooled_all_targets']:.4f}   "
                  f"({dv['pooled_all_targets'] - dc['pooled_all_targets']:+.4f})"
                  f"   [all targets, the official statistic]")
            print(f"  pooled   {dv['pooled']:.4f} vs {dc['pooled']:.4f}   "
                  f"({dv['pooled'] - dc['pooled']:+.4f})"
                  f"   [two-class targets only]")
            print(f"  within   {dv['auc_within']:.4f} vs "
                  f"{dc['auc_within']:.4f}   "
                  f"({dv['auc_within'] - dc['auc_within']:+.4f})   "
                  f"unweighted {dv['auc_within_unweighted']:.4f} vs "
                  f"{dc['auc_within_unweighted']:.4f}")
            print(f"  between  {dv['auc_between']:.4f} vs "
                  f"{dc['auc_between']:.4f}   "
                  f"({dv['auc_between'] - dc['auc_between']:+.4f})")
            print(f"  paired per-target {d_i.mean():+.5f} "
                  f"[{lo:+.5f}, {hi:+.5f}]  Wilcoxon p={float(w.pvalue):.5f}  "
                  f"wins {int((d_i > 0).sum())}/{len(d_i)}")

    print(f"\n{'=' * 78}\n Registered endpoints\n{'=' * 78}")
    verdicts = {}
    for run, (reg, task, stat, bar) in ENDPOINTS.items():
        key = f"{task}: {run} vs {CONTROL}"
        if key not in report:
            print(f"{reg:20s} {run:18s} NOT EVALUABLE (missing submission)")
            continue
        r = report[key]
        field = {"pooled": "pooled_all_targets",
                 "within": "auc_within_unweighted",
                 "between": "auc_between"}[stat]
        v, c = r["variant"][field], r["control"][field]
        p1 = v > c
        p2 = (v >= bar) if bar is not None else None
        verdicts[run] = {
            "registration": reg, "task": task, "statistic": stat,
            "variant": v, "control": c, "delta": v - c,
            "P1_beats_control": bool(p1),
            "P2_bar": bar, "P2_meets_bar": (bool(p2) if p2 is not None
                                            else None),
            "p_wilcoxon_per_target": r["p_wilcoxon"],
        }
        print(f"{reg:20s} {run:18s} {task:12s} {stat:8s} "
              f"{v:.4f} vs {c:.4f} ({v - c:+.4f})  "
              f"P1 {'PASS' if p1 else 'FAIL'}"
              + (f"   P2 (>= {bar:.4f}) "
                 f"{'PASS' if p2 else 'FAIL'}" if bar is not None else ""))

    if pvals:
        print(f"\nHolm across all {len(pvals)} paired comparisons")
        for k, v in sorted(holm_bonferroni(pvals).items(),
                           key=lambda kv: kv[1]["rank"]):
            print(f"  {k:52s} p={v['p_raw']:.5f} adj={v['p_adjusted']:.5f} "
                  f"{'significant' if v['significant'] else ''}")
        report["_holm"] = holm_bonferroni(pvals)

    report["_verdicts"] = verdicts
    if OUT:
        with open(OUT, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
