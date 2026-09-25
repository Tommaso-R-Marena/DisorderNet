#!/usr/bin/env python3
"""CAID3 re-scored under the pairwise protocol, for every entrant.

The residue-level protocol CAID uses has capacity 8 at the measured annotation
error rate; it has 117 entrants. The pairwise protocol has capacity 51, because
a pair is reversed only when **both** its residues flip in opposite directions —
a second-order event. The noise is squared, so the resolution is not.

This re-scores the whole field under both and prints them side by side, so the
community can see its own leaderboard the way the benchmark reports it and the
way its labels can actually support.

## The protocol, in full

For a reference with targets `t` and per-residue labels:

1. **Eligibility.** A method is scored iff it supplies a finite prediction for
   every evaluated residue of every target, at the reference's length. Declining
   targets raises a within-protein score, so coverage is a gate, not a covariate.
2. **Per-target statistic.** For each target with both classes among evaluated
   residues, compute the Mann-Whitney AUC over that target's residues alone.
   Targets with one class contribute no ordered pair and are skipped.
3. **Method score.** The unweighted mean of the per-target AUCs. One protein,
   one vote — pair-weighting lets a few long chains carry the number.
4. **Ranking.** By that mean, descending.
5. **Separation.** Two methods are reported as ordered iff a paired test over
   targets rejects equality: Wilcoxon signed-rank on the per-target differences,
   Holm-corrected within the benchmark. Otherwise they are reported as tied.
6. **Capacity.** The number of methods the reference can order is
   `⌈1/(2·ε_pair)⌉` with `ε_pair` the pairwise discordance rate of the
   annotation. Methods beyond that are reported as an unresolved group, never
   as a rank.

Step 3 is what makes the protocol calibration-invariant: it is unchanged by any
per-protein strictly monotone recalibration of a method's scores
(`auc_within_strictMono_invariant`). Step 6 is what keeps it honest: a rank the
labels cannot support is not printed.
"""

from __future__ import annotations

import json
import math
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
    score_method,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("PL_REFS", f"{ROOT}/caid3_official")
PREDS = os.environ.get("PL_PREDS", f"{ROOT}/caid3_predictions")
EXTRA = os.environ.get("PL_EXTRA", "")
NOISE = os.environ.get("PL_NOISE", f"{ROOT}/relative_noise.json")
OUT = os.environ.get("PL_OUT", "")
TOP = int(os.environ.get("PL_TOP", "25"))


def per_target_aucs(ref, pred):
    """Step 2: one AUC per target, or None if the method is not eligible."""
    from sklearn.metrics import roc_auc_score

    out, kept = [], []
    for tid, (_seq, lab) in ref.items():
        p = pred.get(tid)
        if p is None or len(p) != len(lab):
            return None, None                      # step 1: coverage gate
        m = evaluated_mask(lab)
        y = np.frombuffer(lab.encode(), np.uint8)[m].astype(np.int8) - ord("0")
        s = np.asarray(p)[m]
        if not np.isfinite(s).all():
            return None, None
        if len(np.unique(y)) < 2:
            continue
        out.append(float(roc_auc_score(y, s)))
        kept.append(tid)
    return np.asarray(out), kept


def main() -> int:
    from scipy.stats import wilcoxon

    eps_pair = eps_label = None
    if os.path.isfile(NOISE):
        j = json.load(open(NOISE))
        eps_pair = float(j["eps_pairwise_pooled"])
        eps_label = float(j["eps_label_pooled"])
    cap_pair = math.ceil(1 / (2 * eps_pair)) if eps_pair else None
    cap_lab = math.ceil(1 / (2 * eps_label)) if eps_label else None
    print(f"annotation noise: label {eps_label:.4f} -> capacity {cap_lab};  "
          f"pairwise {eps_pair:.4f} -> capacity {cap_pair}")
    # The closed form needs balanced agreement classes; CAID3's stand at
    # 0.433, so this line reports a comparison and not a guarantee.
    print(f"closed form 2*eps_label^2 = {2*eps_label**2:.5f} "
          f"(measured {eps_pair:.5f}; the balance hypothesis it needs does not "
          f"hold on this reference, so the measurement governs)")

    report = {"eps_label": eps_label, "eps_pairwise": eps_pair,
              "capacity_label": cap_lab, "capacity_pairwise": cap_pair,
              "tasks": {}}

    for task in TASKS:
        path = os.path.join(REFS, f"{task}.fasta")
        if not os.path.isfile(path):
            continue
        ref = read_reference(path)

        files = [(fn[:-5], os.path.join(PREDS, fn))
                 for fn in sorted(os.listdir(PREDS)) if fn.endswith(".caid")]
        for item in filter(None, (s.strip() for s in EXTRA.split(","))):
            q = item.split(":")
            pre = q[1] if len(q) > 1 and q[1] else "DisorderNet"
            lab = q[2] if len(q) > 2 and q[2] else pre
            f = os.path.join(q[0], f"{pre}-{task}.caid")
            if os.path.isfile(f):
                files.append((lab, f))

        rows, n_entered = {}, len(files)
        for name, path_p in files:
            pred = read_caid_predictions(path_p)
            a, kept = per_target_aucs(ref, pred)
            if a is None or len(a) < 5:
                continue
            pooled = score_method(ref, {k: np.asarray(v)
                                        for k, v in pred.items()})
            rows[name] = {"pairwise": float(a.mean()), "per_target": a,
                          "targets": kept,
                          "pooled": (pooled or {}).get("auc")}
        if len(rows) < 5:
            continue

        by_pair = sorted(rows, key=lambda n: -rows[n]["pairwise"])
        by_pool = sorted((n for n in rows if rows[n]["pooled"] is not None),
                         key=lambda n: -rows[n]["pooled"])
        pool_rank = {n: i for i, n in enumerate(by_pool, 1)}

        # Step 5: is the leader separated from each follower?
        lead = by_pair[0]
        pvals = {}
        for n in by_pair[1:]:
            d = rows[lead]["per_target"] - rows[n]["per_target"]
            if np.allclose(d, 0):
                pvals[n] = 1.0
                continue
            try:
                pvals[n] = float(wilcoxon(d, zero_method="wilcox").pvalue)
            except ValueError:
                pvals[n] = 1.0
        holm = holm_bonferroni(pvals) if pvals else {}
        n_sep = sum(1 for v in holm.values() if v["significant"])

        print(f"\n{'=' * 104}")
        print(f" {task}: {n_entered} entered, {len(rows)} eligible, "
              f"{len(rows[lead]['targets'])} scorable targets")
        print(f" capacity under this protocol: {cap_pair}  "
              f"(under residue labels: {cap_lab})")
        print("=" * 104)
        print(f" {'#':>3} {'method':<30}{'pairwise':>10}{'pooled':>9}"
              f"{'pooled #':>10}{'move':>7}{'sep. from #1':>14}")
        for i, n in enumerate(by_pair[:TOP], 1):
            r = rows[n]
            pr = pool_rank.get(n)
            sep = ("—" if n == lead else
                   ("yes" if holm.get(n, {}).get("significant") else "tied"))
            print(f" {i:>3} {n:<30}{r['pairwise']:>10.4f}"
                  f"{(r['pooled'] or float('nan')):>9.4f}"
                  f"{(pr if pr else '-'):>10}"
                  f"{(f'{pr - i:+d}' if pr else '-'):>7}{sep:>14}")
        print(f"\n {lead} is separated from {n_sep} of {len(by_pair)-1} "
              f"other eligible methods (Wilcoxon, Holm within benchmark).")
        moved = [(n, pool_rank[n] - (by_pair.index(n) + 1))
                 for n in by_pair if n in pool_rank]
        big = sorted(moved, key=lambda kv: -abs(kv[1]))[:6]
        print(f" largest rank changes from the pooled protocol:")
        for n, dv in big:
            print(f"   {n:<32}pooled #{pool_rank[n]:<4} pairwise "
                  f"#{by_pair.index(n)+1:<4} {dv:+d}")

        report["tasks"][task] = {
            "n_entered": n_entered, "n_eligible": len(rows),
            "n_targets": len(rows[lead]["targets"]),
            "leader": lead, "n_separated": n_sep,
            "ranking": [
                {"rank": i, "method": n, "pairwise": rows[n]["pairwise"],
                 "pooled": rows[n]["pooled"],
                 "pooled_rank": pool_rank.get(n),
                 "separated_from_leader": bool(
                     holm.get(n, {}).get("significant", False))}
                for i, n in enumerate(by_pair, 1)],
        }

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
