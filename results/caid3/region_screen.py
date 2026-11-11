#!/usr/bin/env python3
"""What the unit of testing costs, on this paper's own references.

`RegionScreen.fdp_lift` proves that a region-level report and its residue-level
reading have the *same* false discovery proportion — both numerator and
denominator scale by the region length, and it cancels. `region_screen_fdr_le`
then gives residue-level control at `alpha` from a region-level
Benjamini-Yekutieli screen run at `alpha / H_M`, and `harmonic_gain_log` bounds
the saving below by `log b - 1`.

Under arbitrary dependence the deflation factor is the harmonic number of the
number of hypotheses stated (`selfConsistent_fdr_le_harmonic`), and
`harmonic_factor_sharp` shows it cannot be lowered. So the factor is a design
parameter and nothing else: it is fixed by how many hypotheses the screen
states, and a screen for disordered *regions* need not state one per residue.

This measures the choice. For each CAID3 and CAID2 reference it counts the
evaluated residues and the maximal same-label runs -- the natural candidate
regions, and the unit the annotation is actually constant on -- and reports both
harmonic numbers, the saving, and the ratio of the two testing thresholds.

Nothing here is a screen. It prices the design decision on real references, so
the theorem is instantiated rather than only cited.
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import evaluated_mask, read_reference  # noqa: E402

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = {
    "CAID3": os.environ.get("RS_REFS3", f"{ROOT}/caid3_official"),
    "CAID2": os.environ.get("RS_REFS2", f"{ROOT}/caid2_official"),
}
TASKS = ("disorder_pdb", "disorder_nox", "binding", "binding_idr", "linker")
OUT = os.environ.get("RS_OUT", "")


def harm(m: int) -> float:
    """H_m = sum_{j=1}^{m} 1/j, summed small-to-large to keep it exact enough."""
    return math.fsum(1.0 / j for j in range(m, 0, -1))


def runs(labels: str, mask) -> int:
    """Maximal runs of a constant label among evaluated residues.

    Runs are counted along the chain and broken by an unevaluated residue, so a
    region interrupted by missing evidence is two candidates, not one. That is
    the conservative direction: it states more hypotheses, not fewer.
    """
    n = 0
    prev = None
    for ch, keep in zip(labels, mask):
        if not keep:
            prev = None
            continue
        if ch != prev:
            n += 1
        prev = ch
    return n


def main() -> int:
    report = {}
    for round_, d in REFS.items():
        if not os.path.isdir(d):
            continue
        for task in TASKS:
            path = os.path.join(d, f"{task}.fasta")
            if not os.path.isfile(path):
                continue
            ref = read_reference(path)
            n_res = n_reg = n_pos_reg = 0
            for _tid, (_seq, lab) in ref.items():
                m = evaluated_mask(lab)
                n_res += int(m.sum())
                n_reg += runs(lab, m)
                # positive regions only: the candidates a disorder screen would
                # actually state, if it stated one per predicted region
                n_pos_reg += sum(
                    1 for i, (ch, keep) in enumerate(zip(lab, m))
                    if keep and ch == "1"
                    and (i == 0 or not m[i - 1] or lab[i - 1] != "1"))
            if n_res == 0 or n_reg == 0:
                continue
            h_n, h_m = harm(n_res), harm(n_reg)
            b = n_res / n_reg
            report[f"{round_} {task}"] = {
                "round": round_, "task": task,
                "n_residues": n_res, "n_regions": n_reg,
                "n_disordered_regions": n_pos_reg,
                "mean_region_length": b,
                "H_residues": h_n, "H_regions": h_m,
                "saving": h_n - h_m,
                "log_b_minus_1": math.log(b) - 1.0,
                "threshold_ratio": h_n / h_m,
            }
            r = report[f"{round_} {task}"]
            print(f"{round_} {task:12s} n={n_res:7,d} M={n_reg:6,d} "
                  f"b={b:6.2f}  H_n={h_n:6.3f} H_M={h_m:6.3f}  "
                  f"saving={r['saving']:5.3f} (>= log b - 1 = "
                  f"{r['log_b_minus_1']:5.3f})  "
                  f"threshold x{r['threshold_ratio']:.2f}")

    if report:
        print("\nThe screen may test at a threshold larger by the last column, "
              "for the same residue-level false discovery rate.")
    if OUT:
        with open(OUT, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
