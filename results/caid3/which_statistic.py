#!/usr/bin/env python3
"""Which CAID statistic predicts what a method actually costs to use?

The field reports pooled AUC. This project has shown that pooled AUC is 97-99.5%
the between-protein question, that `AUC_within` is the calibration-invariant
part (`auc_within_strictMono_invariant`), and that the *operating* cost — how
much of a protein a method must flag to guarantee a miss rate — separates
methods pooled AUC ranks as inseparable.

That leaves a prescriptive question with an empirical answer. If a benchmark is
going to report one number to summarise how usable a method is, which of the
available numbers does the job?

Concretely, across every full-coverage entrant on a reference:

- **pooled AUC** — what CAID reports;
- **AUC_within** — the calibration-invariant component;
- **AUC_between** — the other component;

each correlated against the **calibration-invariant operating cost**: the
fraction of each protein that must be flagged, using a per-protein quantile, to
guarantee a miss rate of `alpha`. That cost is the operational object, it is
invariant to recalibration by construction, and it is what a user pays.

Spearman throughout, because all three are used as rankings. Reported with a
method-level bootstrap interval, and with the *difference* between correlations
tested on the same resamples, since two correlations computed on one set of
methods are not independent and comparing their intervals would be wrong.

    export ANALYSIS_SCRIPT=results/caid3/which_statistic.py
    sbatch rockfish/slurm/analysis_cpu.sbatch
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
CERT = os.environ.get("WS_CERT", f"{ROOT}/certified_caid3.json")
COST = os.environ.get("WS_COST", f"{ROOT}/operating_cost.json")
OUT = os.environ.get("WS_OUT", "")
ALPHA = os.environ.get("WS_ALPHA", "0.05")
N_BOOT = int(os.environ.get("WS_NBOOT", "10000"))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def main() -> int:
    if not (os.path.isfile(CERT) and os.path.isfile(COST)):
        print(f"need both {CERT} and {COST}", file=sys.stderr)
        return 2
    cert = json.load(open(CERT))
    cost = json.load(open(COST))

    report = {"alpha": ALPHA, "n_boot": N_BOOT, "tasks": {}}

    def find_key(entry):
        """Match the alpha by value, not by spelling.

        The cost tables key on `f"alpha_{a}"` with `a` a float, so 0.10 is
        stored as "alpha_0.1" and looked up as "alpha_0.10" — a silent zero
        overlap that reported "0 methods in both" for every benchmark rather
        than failing.
        """
        want = float(ALPHA)
        for k in entry:
            if not k.startswith("alpha_"):
                continue
            try:
                if abs(float(k.split("_", 1)[1]) - want) < 1e-9:
                    return k
            except ValueError:
                continue
        return None

    for task in cert.get("tasks", {}):
        if task not in cost.get("tasks", {}):
            continue
        cm = cert["tasks"][task]["methods"]
        om = cost["tasks"][task]["methods"]
        probe = next((find_key(om[n]) for n in om if find_key(om[n])), None)
        if probe is None:
            print(f"{task}: no cost entry at alpha={ALPHA}; "
                  f"available: {sorted({k for n in om for k in om[n] if k.startswith('alpha_')})}")
            continue
        key = probe
        names = [n for n in cm if n in om
                 and (om[n].get(key) or {}).get("flagged_quantile_median")
                 is not None]
        if len(names) < 20:
            print(f"{task}: only {len(names)} methods usable at {key} — "
                  f"skipped ({len(set(cm) & set(om))} in both analyses)")
            continue

        pooled = np.array([cm[n]["pooled"] for n in names])
        within = np.array([cm[n]["within"] for n in names])
        between = np.array([cm[n]["between"] for n in names])
        # Lower cost is better, so negate to make every predictor
        # "higher is better" and the correlations comparable in sign.
        price = -np.array([om[n][key]["flagged_quantile_median"]
                           for n in names])
        global_price = -np.array([om[n][key]["flagged_median"] for n in names])

        rows = {"pooled": spearman(pooled, price),
                "within": spearman(within, price),
                "between": spearman(between, price)}
        rows_global = {"pooled": spearman(pooled, global_price),
                       "within": spearman(within, global_price),
                       "between": spearman(between, global_price)}

        # Method-level bootstrap. The difference is resampled on the same draws
        # because the two correlations share a method set and are dependent.
        rng = np.random.default_rng(20260819)
        n = len(names)
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            take = rng.integers(0, n, size=n)
            diffs[i] = (spearman(within[take], price[take])
                        - spearman(pooled[take], price[take]))
        diffs = diffs[np.isfinite(diffs)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        m = diffs.size
        p_lo = (1.0 + np.count_nonzero(diffs <= 0.0)) / (m + 1.0)
        p_hi = (1.0 + np.count_nonzero(diffs >= 0.0)) / (m + 1.0)
        pval = min(1.0, 2.0 * min(p_lo, p_hi))

        print(f"\n{'=' * 88}")
        print(f" {task}: {n} methods scored by both analyses")
        print(f" Spearman against the operating cost at risk <= {ALPHA}")
        print("=" * 88)
        print(f" {'predictor':<16}{'vs calibration-invariant cost':>32}"
              f"{'vs global-threshold cost':>28}")
        for k in ("pooled", "within", "between"):
            print(f" {k:<16}{rows[k]:>32.3f}{rows_global[k]:>28.3f}")
        print(f"\n within − pooled, against the calibration-invariant cost: "
              f"{rows['within'] - rows['pooled']:+.3f}")
        print(f"   95% CI [{lo:+.3f}, {hi:+.3f}]   p = {pval:.4f}")

        report["tasks"][task] = {
            "n_methods": n, "methods": names,
            "spearman_vs_quantile_cost": rows,
            "spearman_vs_global_cost": rows_global,
            "within_minus_pooled": rows["within"] - rows["pooled"],
            "ci": [float(lo), float(hi)], "p": float(pval),
        }

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
