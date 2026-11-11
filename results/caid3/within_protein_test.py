#!/usr/bin/env python3
"""Test the within-protein claim directly, as a paired per-target comparison.

The pooled AUC is 97-99.5% between-protein pairs, and on that statistic
DisorderNet and PUNCH2 are inseparable: +0.0019, p = 0.657. The paper's thesis
is that the *within-protein* axis is the one that answers the residue-level
question, and `auc_within_strictMono_invariant` proves it is the part no
per-protein recalibration can change.

So test it on its own terms. Each target contributes one AUC per method; the
comparison is paired within target, which is the natural design — the same
protein, the same labels, two predictors.

Three tests, because the assumptions differ and a claim this central should not
rest on one:

- **Wilcoxon signed-rank**, distribution-free, on the per-target differences;
- **paired t**, for the mean difference;
- **a target-level bootstrap** of the mean difference, 10,000 resamples.

Exploratory. The registered primary family is on pooled AUC and this is not it.
It is reported as the test the theory implies, not as the test that was
promised.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import (  # noqa: E402
    TASKS,
    evaluated_mask,
    holm_bonferroni,
    read_caid_predictions,
    read_reference,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("WT_REFS", f"{ROOT}/caid3_official")
PREDS = os.environ.get("WT_PREDS", f"{ROOT}/caid3_predictions")
EXTRA = os.environ.get("WT_EXTRA", "")
OPPONENTS = os.environ.get("WT_OPPONENTS", "PUNCH2,AlphaFold-rsa").split(",")
OUT = os.environ.get("WT_OUT", "")


def per_target(ref, a, b):
    from sklearn.metrics import roc_auc_score

    out, kept = [], []
    for tid, (_seq, lab) in ref.items():
        if tid not in a or tid not in b:
            continue
        m = evaluated_mask(lab)
        y = np.frombuffer(lab.encode(), np.uint8)[m].astype(np.int8) - ord("0")
        if len(np.unique(y)) < 2:
            continue
        x, z = np.asarray(a[tid])[m], np.asarray(b[tid])[m]
        if not (np.isfinite(x).all() and np.isfinite(z).all()):
            continue
        out.append(roc_auc_score(y, x) - roc_auc_score(y, z))
        kept.append(tid)
    return np.asarray(out), kept


def main() -> int:
    from scipy.stats import ttest_rel, wilcoxon

    ours = {}
    for item in filter(None, (s.strip() for s in EXTRA.split(","))):
        p = item.split(":")
        pre = p[1] if len(p) > 1 and p[1] else "DisorderNet"
        lab = p[2] if len(p) > 2 and p[2] else pre
        ours[lab] = (p[0], pre)
    if not ours:
        print("set WT_EXTRA", file=sys.stderr)
        return 2

    report, pvals = {}, {}
    for task in TASKS:
        path = os.path.join(REFS, f"{task}.fasta")
        if not os.path.isfile(path):
            continue
        ref = read_reference(path)
        for label, (d, pre) in ours.items():
            f = os.path.join(d, f"{pre}-{task}.caid")
            if not os.path.isfile(f):
                continue
            A = read_caid_predictions(f)
            for opp in [o.strip() for o in OPPONENTS if o.strip()]:
                g = os.path.join(PREDS, f"{opp}.caid")
                if not os.path.isfile(g):
                    continue
                B = read_caid_predictions(g)
                d_i, kept = per_target(ref, A, B)
                if len(d_i) < 20:
                    continue
                mean = float(d_i.mean())
                sd = float(d_i.std(ddof=1))
                t = ttest_rel(d_i, np.zeros_like(d_i))
                w = wilcoxon(d_i, alternative="two-sided",
                             zero_method="wilcox")
                rng = np.random.default_rng(20260819)
                boot = np.array([d_i[rng.integers(0, len(d_i), len(d_i))].mean()
                                 for _ in range(10000)])
                lo, hi = np.percentile(boot, [2.5, 97.5])
                nb = boot.size
                p_lo = (1.0 + np.count_nonzero(boot <= 0)) / (nb + 1.0)
                p_hi = (1.0 + np.count_nonzero(boot >= 0)) / (nb + 1.0)
                p_boot = min(1.0, 2.0 * min(p_lo, p_hi))
                wins = int((d_i > 0).sum())

                key = f"{task}: {label} vs {opp}"
                print(f"\n{key}")
                print(f"  {len(d_i)} targets, wins {wins}/{len(d_i)} "
                      f"({wins/len(d_i):.1%})")
                print(f"  mean per-target AUC difference {mean:+.5f} "
                      f"(sd {sd:.5f})")
                print(f"  bootstrap 95% CI [{lo:+.5f}, {hi:+.5f}]  "
                      f"p={p_boot:.5f}")
                print(f"  paired t          p={float(t.pvalue):.5f}")
                print(f"  Wilcoxon          p={float(w.pvalue):.5f}")
                report[key] = {
                    "n_targets": len(d_i), "wins": wins, "mean": mean,
                    "sd": sd, "ci": [float(lo), float(hi)],
                    "p_bootstrap": float(p_boot),
                    "p_ttest": float(t.pvalue), "p_wilcoxon": float(w.pvalue),
                }
                pvals[key] = float(w.pvalue)

    if pvals:
        print(f"\n{'=' * 84}\n Holm across all {len(pvals)} comparisons "
              f"(Wilcoxon p-values)\n{'=' * 84}")
        for k, v in sorted(holm_bonferroni(pvals).items(),
                           key=lambda kv: kv[1]["rank"]):
            print(f" {k:<52}p={v['p_raw']:.5f} adj={v['p_adjusted']:.5f}  "
                  f"{'SIGNIFICANT' if v['significant'] else '-'}")
        report["_holm"] = holm_bonferroni(pvals)

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
