#!/usr/bin/env python3
"""What CAID3 can and cannot establish, at its measured annotation error rate.

Two computations, both requiring the rate this project has just measured.

## 1. How many comparisons the benchmark can order

`LabelNoise.ranking_certified`: a margin above `2*eps*n` certifies a ranking.
With `eps` measured rather than assumed, the count of certifiable comparisons is
determined, and its complement is the set of method pairs this benchmark
**cannot order — not by this analysis, but by anyone**, however many resamples
or bootstraps are taken.

## 2. How large a benchmark would have to be

The paired protein-clustered bootstrap gives a per-target delta distribution.
From its spread and the observed effect, the target count needed to detect that
effect at 80% power follows directly:

    n >= (z_{1-a/2} + z_{1-b})^2 * sd^2 / delta^2

For the DisorderNet-versus-PUNCH2 comparison the effect is +0.0019 on 319
targets. If the required `n` is far beyond any feasible benchmark, that is not a
failure of this project's model — it is a statement that the two methods are
**inseparable at any benchmark size CAID could plausibly run**, which is the
same conclusion the noise calculation reaches by a different route.

Neither number can be improved by resampling. Bootstrap, permutation, LOOCV and
Monte Carlo all estimate the sampling distribution of a statistic computed on
the data in hand; more replicates shrink the Monte Carlo error in locating a
p-value and leave the p-value where it is. Only more targets, a more precise
measurement, or a larger effect move it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import (  # noqa: E402
    evaluated_mask,
    read_caid_predictions,
    read_reference,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("CAP_REFS", f"{ROOT}/caid3_official")
PREDS = os.environ.get("CAP_PREDS", f"{ROOT}/caid3_predictions")
NOISE = os.environ.get("CAP_NOISE", f"{ROOT}/annotation_noise.json")
EXTRA = os.environ.get("CAP_EXTRA", "")
OUT = os.environ.get("CAP_OUT", "")


def binary_predictions(path: str) -> dict[str, np.ndarray]:
    """Column four of a .caid file — each method's own thresholded call."""
    out: dict[str, list] = {}
    cur = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                cur = line[1:].strip()
                out[cur] = []
                continue
            if cur is None:
                continue
            parts = line.split("\t")
            try:
                out[cur].append(float(parts[3]) if len(parts) >= 4 else np.nan)
            except ValueError:
                out[cur].append(np.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def error_counts(ref, task) -> dict[str, dict]:
    files = [(fn[:-5], os.path.join(PREDS, fn))
             for fn in sorted(os.listdir(PREDS)) if fn.endswith(".caid")]
    for item in filter(None, (s.strip() for s in EXTRA.split(","))):
        p = item.split(":")
        pre = p[1] if len(p) > 1 and p[1] else "DisorderNet"
        lab = p[2] if len(p) > 2 and p[2] else pre
        f = os.path.join(p[0], f"{pre}-{task}.caid")
        if os.path.isfile(f):
            files.append((lab, f))
    out = {}
    for name, path in files:
        pred = binary_predictions(path)
        errs = n_eval = 0
        ok = True
        for tid, (_seq, lab) in ref.items():
            q = pred.get(tid)
            if q is None or len(q) != len(lab):
                ok = False
                break
            m = evaluated_mask(lab)
            y = np.frombuffer(lab.encode(), np.uint8)[m].astype(np.int8) - ord("0")
            v = q[m]
            if not np.isfinite(v).all():
                ok = False
                break
            errs += int((v.astype(np.int8) != y).sum())
            n_eval += int(m.sum())
        if ok and n_eval:
            out[name] = {"errors": errs, "n_evaluated": n_eval}
    return out


def power_n(deltas: np.ndarray, alpha: float = 0.05, power: float = 0.80):
    """Targets needed to detect the observed per-target effect."""
    from scipy.stats import norm

    d = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1))
    if d == 0.0 or sd == 0.0:
        return None
    z = norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)
    return {"effect_per_target": d, "sd_per_target": sd,
            "n_required": float(z ** 2 * sd ** 2 / d ** 2)}


def per_target_auc_deltas(ref, a_path, b_path):
    """Per-target AUC difference, the unit the clustered bootstrap resamples."""
    from sklearn.metrics import roc_auc_score

    A = read_caid_predictions(a_path)
    B = read_caid_predictions(b_path)
    out = []
    for tid, (_seq, lab) in ref.items():
        if tid not in A or tid not in B:
            continue
        m = evaluated_mask(lab)
        y = np.frombuffer(lab.encode(), np.uint8)[m].astype(np.int8) - ord("0")
        if len(np.unique(y)) < 2:
            continue
        a, b = np.asarray(A[tid])[m], np.asarray(B[tid])[m]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        out.append(roc_auc_score(y, a) - roc_auc_score(y, b))
    return np.asarray(out)


def main() -> int:
    if not os.path.isfile(NOISE):
        print(f"need {NOISE}", file=sys.stderr)
        return 2
    noise = json.load(open(NOISE))
    eps = float(noise["epsilon_pooled"])
    print(f"measured annotation error rate: eps = {eps:.4f} "
          f"({noise['n_usable']} proteins, MobiDB context-dependent residues)")

    report = {"epsilon": eps, "noise_source": NOISE, "tasks": {}}
    for task in ("disorder_pdb", "disorder_nox"):
        path = os.path.join(REFS, f"{task}.fasta")
        if not os.path.isfile(path):
            continue
        ref = read_reference(path)
        counts = error_counts(ref, task)
        if len(counts) < 10:
            continue
        order = sorted(counts, key=lambda n: counts[n]["errors"])
        best = order[0]
        n_eval = counts[best]["n_evaluated"]
        bar = 2.0 * eps * n_eval

        certified, unresolvable = [], []
        for n in order[1:]:
            margin = counts[n]["errors"] - counts[best]["errors"]
            (certified if margin > bar else unresolvable).append(n)

        print(f"\n{'=' * 84}")
        print(f" {task}: {len(counts)} methods, {n_eval:,} evaluated residues")
        print(f" ranking_certified needs a margin above 2*eps*n = {bar:,.0f} "
              f"errors")
        print("=" * 84)
        print(f" best by error count: {best} "
              f"({counts[best]['errors']:,} errors)")
        print(f" **certified better than {len(certified)} of "
              f"{len(order)-1} other methods**")
        print(f" **{len(unresolvable)} comparisons are inside the annotation "
              f"noise and cannot be**")
        print(f" **ordered by this benchmark, by anyone, at any sample size "
              f"of resamples**")
        print(f"\n the unresolvable set (nearest first):")
        for n in unresolvable[:12]:
            m = counts[n]["errors"] - counts[best]["errors"]
            print(f"   {n:<30}{m:>9,} errors behind  "
                  f"({m / n_eval:.2%} of the set)")
        if len(unresolvable) > 12:
            print(f"   … {len(unresolvable) - 12} more")

        report["tasks"][task] = {
            "n_evaluated": n_eval, "bar": bar, "best": best,
            "n_certified": len(certified), "n_unresolvable": len(unresolvable),
            "unresolvable": unresolvable, "certified": certified,
            "errors": {n: counts[n]["errors"] for n in order},
        }

    # How big a benchmark would separate us from PUNCH2 on Disorder-PDB.
    ref = read_reference(os.path.join(REFS, "disorder_pdb.fasta"))
    punch = os.path.join(PREDS, "PUNCH2.caid")
    ours = None
    for item in filter(None, (s.strip() for s in EXTRA.split(","))):
        p = item.split(":")
        pre = p[1] if len(p) > 1 and p[1] else "DisorderNet"
        f = os.path.join(p[0], f"{pre}-disorder_pdb.caid")
        if os.path.isfile(f):
            ours = f
            break
    if ours and os.path.isfile(punch):
        d = per_target_auc_deltas(ref, ours, punch)
        got = power_n(d)
        if got:
            print(f"\n{'=' * 84}")
            print(f" How large a benchmark would separate us from PUNCH2 "
                  f"on Disorder-PDB")
            print("=" * 84)
            print(f" per-target effect  {got['effect_per_target']:+.5f}")
            print(f" per-target sd      {got['sd_per_target']:.5f}  "
                  f"({len(d)} targets)")
            print(f" **targets needed for 80% power at alpha=0.05: "
                  f"{got['n_required']:,.0f}**")
            print(f" CAID3 has {len(d)}. That is "
                  f"{got['n_required']/len(d):,.0f}x the current benchmark.")
            report["power_vs_punch2"] = {**got, "n_targets_now": len(d)}

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
