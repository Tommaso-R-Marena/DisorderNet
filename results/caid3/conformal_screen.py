#!/usr/bin/env python3
"""Run the screen the theorems describe, and measure what the unit costs.

`RegionScreen.fdp_lift` proves a region-level report and its residue-level
reading have the same false discovery proportion; `region_screen_fdr_le` gives
residue-level control at `alpha` from a region-level Benjamini-Yekutieli
procedure at `alpha / H_M`; `harmonic_gain_log` bounds the saving below by
`log b - 1`. `region_screen.py` priced that design choice in nats. This runs the
screen and prices it in **discoveries**, which is what a user cares about.

Design
------
Candidates are consecutive blocks of `b` residues -- `stdBlocks`, the partition
the theorem is stated for. A block's score is the mean predicted disorder over
its evaluated residues; a block is *truly* disordered when the majority of those
residues are. The truth and the report are then both unions of whole blocks,
which is the condition `fdp_lift` needs.

P-values are **split conformal**: calibrate on the known-ordered blocks of one
half of the targets, and for a block of the other half report

    p = (1 + #{calibration blocks with score >= s}) / (n_cal + 1).

Under exchangeability of the ordered blocks this is superuniform, which is the
only assumption `selfConsistent_fdr_le_harmonic` makes. It is also the reason
the correction has to survive arbitrary dependence: every candidate is compared
against the *same* calibration set, so the p-values are dependent by
construction. The arbitrary-dependence theorem is the right tool here, not a
conservative one.

Four procedures are compared at the same guaranteed residue-level rate:

  region BY      one hypothesis per block, BH at alpha / H_M      <- proposed
  residue BY     one hypothesis per residue, BH at alpha / H_n
  region BH      one hypothesis per block, BH at alpha            (no guarantee
                 under dependence; included to show what is being paid for)
  region Bonf.   one hypothesis per block, threshold alpha / M

and for each we report the discoveries, the realised residue-level FDP against
the reference, and the power. `fdp_lift` is checked rather than assumed: the
region-level and residue-level FDPs of the same report are printed side by side
and must agree.

Nothing here is a discovery about biology. The reference is a benchmark whose
labels are already public; what is measured is the price of a design decision on
a real prediction, with a guarantee that holds under arbitrary dependence.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import (  # noqa: E402
    evaluated_mask,
    read_caid_predictions,
    read_reference,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("CS_REFS", f"{ROOT}/caid3_official")
TASK = os.environ.get("CS_TASK", "disorder_pdb")
METHOD = os.environ.get("CS_METHOD", "")          # dir:prefix, our submission
ALPHAS = [float(a) for a in os.environ.get("CS_ALPHAS", "0.10,0.05").split(",")]
BLOCKS = [int(b) for b in os.environ.get("CS_BLOCKS", "10,25,50").split(",")]
SPLITS = int(os.environ.get("CS_SPLITS", "50"))
CAL_FRAC = float(os.environ.get("CS_CAL_FRAC", "0.8"))
OUT = os.environ.get("CS_OUT", "")


def harm(m: int) -> float:
    return math.fsum(1.0 / j for j in range(m, 0, -1))


def blocks_of(y: np.ndarray, s: np.ndarray, b: int):
    """`stdBlocks`: consecutive blocks of `b`, scored and labelled by majority.

    A trailing partial block is kept only if it is at least half full, so the
    partition covers the chain without a short block dominated by an edge.
    """
    n = len(y)
    out = []
    for start in range(0, n, b):
        stop = min(start + b, n)
        if stop - start < b // 2:
            continue
        yy, ss = y[start:stop], s[start:stop]
        out.append((float(ss.mean()), bool(yy.mean() > 0.5), stop - start,
                    int(yy.sum())))
    return out


def bh_reject(p: np.ndarray, level: float) -> np.ndarray:
    """Benjamini-Hochberg step-up at `level`; returns the rejection mask."""
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p, kind="mergesort")
    thresh = level * np.arange(1, m + 1) / m
    below = p[order] <= thresh
    if not below.any():
        return np.zeros(m, bool)
    k = int(np.nonzero(below)[0].max()) + 1
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej


def conformal_p(cal: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Split-conformal p-values against a calibration sample of true nulls."""
    n = len(cal)
    cal_sorted = np.sort(cal)
    ge = n - np.searchsorted(cal_sorted, s, side="left")
    return (1.0 + ge) / (n + 1.0)


def main() -> int:
    ref = read_reference(os.path.join(REFS, f"{TASK}.fasta"))
    d, pre = (METHOD.split(":") + ["DisorderNet"])[:2]
    path = os.path.join(d, f"{pre}-{TASK}.caid")
    if not os.path.isfile(path):
        print(f"no submission at {path}", file=sys.stderr)
        return 2
    pred = read_caid_predictions(path)

    chains = []
    for tid, (_seq, lab) in ref.items():
        if tid not in pred:
            continue
        m = evaluated_mask(lab)
        y = np.frombuffer(lab.encode(), np.uint8)[m].astype(np.int8) - ord("0")
        s = np.asarray(pred[tid])[m]
        if len(y) < 40 or not np.isfinite(s).all():
            continue
        chains.append((tid, y, s))
    print(f"{len(chains)} chains, {sum(len(c[1]) for c in chains):,} evaluated "
          f"residues\n")

    rng = np.random.default_rng(20260820)
    report = {}

    for b in BLOCKS:
        per_chain = [(tid, blocks_of(y, s, b), y, s) for tid, y, s in chains]
        per_chain = [c for c in per_chain if c[1]]
        for alpha in ALPHAS:
            acc = {k: [] for k in ("region_BY", "residue_BY", "region_BH",
                                   "region_Bonf")}
            lift_gap, feas = [], []
            for _ in range(SPLITS):
                idx = rng.permutation(len(per_chain))
                cut = int(round(CAL_FRAC * len(idx)))
                cal_i, scr_i = idx[:cut], idx[cut:]

                cal_blocks = np.array(
                    [sc for i in cal_i for sc, tr, _n, _p in per_chain[i][1]
                     if not tr])
                cal_res = np.concatenate(
                    [per_chain[i][3][per_chain[i][2] == 0] for i in cal_i])
                if len(cal_blocks) < 30 or len(cal_res) < 200:
                    continue

                sc = np.array([x[0] for i in scr_i for x in per_chain[i][1]])
                tr = np.array([x[1] for i in scr_i for x in per_chain[i][1]])
                ln = np.array([x[2] for i in scr_i for x in per_chain[i][1]])
                npos = np.array([x[3] for i in scr_i for x in per_chain[i][1]])
                M = len(sc)
                res_s = np.concatenate([per_chain[i][3] for i in scr_i])
                res_y = np.concatenate([per_chain[i][2] for i in scr_i])
                n_res = len(res_s)
                if M < 50:
                    continue

                # A conformal p-value cannot fall below 1/(n_cal+1), and the
                # smallest threshold a step-up rule at level L over m
                # hypotheses ever applies is L/m. A screen is therefore
                # *arithmetically* incapable of a discovery unless
                #
                #     n_cal + 1  >=  m * H_m / alpha,
                #
                # whatever the predictor does. The requirement is linear in the
                # number of hypotheses stated, so the unit of testing sets the
                # calibration budget as well as the harmonic factor -- and this
                # is the constraint that binds first. It is reported rather than
                # discovered by an empty result.
                need_reg = harm(M) * M / alpha - 1.0
                need_res = harm(n_res) * n_res / alpha - 1.0
                feas.append((len(cal_blocks), need_reg,
                             len(cal_res), need_res))

                p_reg = conformal_p(cal_blocks, sc)
                p_res = conformal_p(cal_res, res_s)

                runs = {
                    "region_BY": ("reg", bh_reject(p_reg, alpha / harm(M))),
                    "region_BH": ("reg", bh_reject(p_reg, alpha)),
                    "region_Bonf": ("reg", p_reg <= alpha / M),
                    "residue_BY": ("res", bh_reject(p_res,
                                                    alpha / harm(n_res))),
                }
                for name, (kind, rej) in runs.items():
                    if kind == "reg":
                        n_disc = int(rej.sum())
                        # residue-level reading of the region report
                        res_disc = int(ln[rej].sum())
                        res_false = int((ln[rej] - npos[rej]).sum())
                        # `fdp_lift`: the two readings must agree. The region
                        # form counts a block as false when it is not majority
                        # disordered; the residue form counts its ordered
                        # residues. They coincide when the truth is a union of
                        # whole blocks, which is checked below rather than
                        # assumed.
                        fdp_reg = (1 - tr[rej]).sum() / max(n_disc, 1)
                        fdp_res = res_false / max(res_disc, 1)
                        lift_gap.append(abs(fdp_reg - fdp_res))
                        power = (tr[rej].sum() / max(tr.sum(), 1))
                        acc[name].append((n_disc, res_disc, fdp_res, fdp_reg,
                                          power))
                    else:
                        n_disc = int(rej.sum())
                        false = int((rej & (res_y == 0)).sum())
                        fdp = false / max(n_disc, 1)
                        power = ((rej & (res_y == 1)).sum()
                                 / max((res_y == 1).sum(), 1))
                        acc[name].append((n_disc, n_disc, fdp, float("nan"),
                                          power))

            if not acc["region_BY"]:
                continue
            print(f"block b={b:3d}   alpha={alpha:.2f}   "
                  f"{len(acc['region_BY'])} splits")
            row = {}
            for name in ("region_BY", "residue_BY", "region_BH",
                         "region_Bonf"):
                a = np.array([(x[1], x[2], x[4]) for x in acc[name]])
                row[name] = {"residues_reported": float(np.median(a[:, 0])),
                             "realised_residue_fdp": float(np.median(a[:, 1])),
                             "power": float(np.median(a[:, 2]))}
                print(f"   {name:<13} residues reported "
                      f"{np.median(a[:, 0]):8,.0f}   realised residue FDP "
                      f"{np.median(a[:, 1]):.4f}   power "
                      f"{np.median(a[:, 2]):.3f}")
            f = np.array(feas)
            row["calibration"] = {
                "region_have": float(np.median(f[:, 0])),
                "region_need": float(np.median(f[:, 1])),
                "residue_have": float(np.median(f[:, 2])),
                "residue_need": float(np.median(f[:, 3])),
                "region_shortfall": float(np.median(f[:, 1] / f[:, 0])),
                "residue_shortfall": float(np.median(f[:, 3] / f[:, 2])),
                "budget_ratio": float(np.median(f[:, 3] / f[:, 1])),
            }
            c = row["calibration"]
            print(f"   calibration units: region has {c['region_have']:>9,.0f} "
                  f"needs {c['region_need']:>12,.0f}  "
                  f"({c['region_shortfall']:6.1f}x short)")
            print(f"                      residue has {c['residue_have']:>8,.0f} "
                  f"needs {c['residue_need']:>12,.0f}  "
                  f"({c['residue_shortfall']:6.1f}x short)   "
                  f"region needs {c['budget_ratio']:.0f}x fewer")
            gain = (row["region_BY"]["residues_reported"]
                    / max(row["residue_BY"]["residues_reported"], 1))
            row["region_over_residue"] = gain
            row["fdp_lift_max_gap"] = float(np.max(lift_gap)) if lift_gap else None
            print(f"   region BY reports {gain:.2f}x the residues of residue BY"
                  f" at the same guaranteed rate;  fdp_lift max gap "
                  f"{row['fdp_lift_max_gap']:.2e}\n")
            report[f"b{b}_alpha{alpha}"] = row

    if OUT:
        with open(OUT, "w") as fh:
            json.dump({"task": TASK, "method": METHOD, "n_splits": SPLITS,
                       "results": report}, fh, indent=1)
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
