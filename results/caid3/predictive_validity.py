#!/usr/bin/env python3
"""Does the pairwise protocol have more *useful* resolution, or just more?

The capacity argument says the pairwise protocol can order 72 methods where the
residue-level one can order 8. The obvious objection is that extra resolution is
not automatically extra signal: a protocol could resolve more pairs and resolve
them wrongly.

This tests it on held-out data. 57 methods entered **both** CAID2 and CAID3.
Rank them on CAID3 under each protocol, then ask which CAID3 ranking predicts
their **CAID2** ranking better. CAID2 is a different round with different
targets — one protein shared with CAID3 — so it is a genuine held-out ordering
of the same methods.

If the pairwise ranking transfers better, its extra resolution is real. If it
transfers worse, the capacity gain is resolution without validity and the
protocol should not be recommended. Either answer settles the objection.

The comparison of two dependent correlations uses a method-level bootstrap on
the same resamples, since both are computed over one set of methods.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import (  # noqa: E402
    evaluated_mask,
    read_caid_predictions,
    read_reference,
    score_method,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
R3 = os.environ.get("PV_R3", f"{ROOT}/caid3_official")
P3 = os.environ.get("PV_P3", f"{ROOT}/caid3_predictions")
R2 = os.environ.get("PV_R2", f"{ROOT}/caid2_official")
P2 = os.environ.get("PV_P2", f"{ROOT}/caid2_predictions")
OUT = os.environ.get("PV_OUT", "")
N_BOOT = int(os.environ.get("PV_NBOOT", "10000"))
TASKS = ("disorder_pdb", "disorder_nox", "binding", "linker")


def spearman(a, b):
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def score_both(ref, pred):
    """(pooled AUC, pairwise mean AUC) or None if not eligible."""
    from sklearn.metrics import roc_auc_score

    per = []
    for tid, (_seq, lab) in ref.items():
        p = pred.get(tid)
        if p is None or len(p) != len(lab):
            return None
        m = evaluated_mask(lab)
        y = np.frombuffer(lab.encode(), np.uint8)[m].astype(np.int8) - ord("0")
        s = np.asarray(p)[m]
        if not np.isfinite(s).all():
            return None
        if len(np.unique(y)) < 2:
            continue
        per.append(float(roc_auc_score(y, s)))
    if len(per) < 5:
        return None
    pooled = score_method(ref, {k: np.asarray(v) for k, v in pred.items()})
    if not pooled:
        return None
    return pooled["auc"], float(np.mean(per))


def main() -> int:
    both = sorted(
        {f[:-5] for f in os.listdir(P3) if f.endswith(".caid")} &
        {f[:-5] for f in os.listdir(P2) if f.endswith(".caid")})
    print(f"{len(both)} methods entered both rounds")

    report = {"n_both": len(both), "tasks": {}}
    for task in TASKS:
        f3 = os.path.join(R3, f"{task}.fasta")
        f2 = os.path.join(R2, f"{task}.fasta")
        if not (os.path.isfile(f3) and os.path.isfile(f2)):
            continue
        ref3, ref2 = read_reference(f3), read_reference(f2)

        rows = {}
        for name in both:
            a = score_both(ref3, read_caid_predictions(
                os.path.join(P3, f"{name}.caid")))
            b = score_both(ref2, read_caid_predictions(
                os.path.join(P2, f"{name}.caid")))
            if a and b:
                rows[name] = {"c3_pooled": a[0], "c3_pair": a[1],
                              "c2_pooled": b[0], "c2_pair": b[1]}
        if len(rows) < 15:
            print(f"{task}: only {len(rows)} scorable in both — skipped")
            continue

        names = sorted(rows)
        c3p = np.array([rows[n]["c3_pooled"] for n in names])
        c3w = np.array([rows[n]["c3_pair"] for n in names])
        c2p = np.array([rows[n]["c2_pooled"] for n in names])
        c2w = np.array([rows[n]["c2_pair"] for n in names])

        # Predict the held-out round's *pairwise* ordering — the quantity the
        # protocol claims to measure — from each CAID3 protocol in turn.
        r_pair = spearman(c3w, c2w)
        r_pool = spearman(c3p, c2w)
        # And the held-out *pooled* ordering, so neither target is favoured.
        r_pair_p = spearman(c3w, c2p)
        r_pool_p = spearman(c3p, c2p)

        rng = np.random.default_rng(20260819)
        n = len(names)
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            t = rng.integers(0, n, size=n)
            diffs[i] = spearman(c3w[t], c2w[t]) - spearman(c3p[t], c2w[t])
        diffs = diffs[np.isfinite(diffs)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        m = diffs.size
        p_lo = (1.0 + np.count_nonzero(diffs <= 0)) / (m + 1.0)
        p_hi = (1.0 + np.count_nonzero(diffs >= 0)) / (m + 1.0)
        pval = min(1.0, 2.0 * min(p_lo, p_hi))

        print(f"\n{'=' * 84}")
        print(f" {task}: {n} methods scorable in both rounds")
        print("=" * 84)
        print(f" {'CAID3 protocol':<22}{'-> CAID2 pairwise':>20}"
              f"{'-> CAID2 pooled':>18}")
        print(f" {'pairwise':<22}{r_pair:>20.3f}{r_pair_p:>18.3f}")
        print(f" {'pooled (as reported)':<22}{r_pool:>20.3f}{r_pool_p:>18.3f}")
        print(f"\n pairwise − pooled, predicting the held-out pairwise order: "
              f"{r_pair - r_pool:+.3f}")
        print(f"   95% CI [{lo:+.3f}, {hi:+.3f}]   p = {pval:.4f}")

        report["tasks"][task] = {
            "n_methods": n, "methods": names,
            "r_pairwise_to_pairwise": r_pair, "r_pooled_to_pairwise": r_pool,
            "r_pairwise_to_pooled": r_pair_p, "r_pooled_to_pooled": r_pool_p,
            "delta": r_pair - r_pool, "ci": [float(lo), float(hi)],
            "p": float(pval),
        }

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
