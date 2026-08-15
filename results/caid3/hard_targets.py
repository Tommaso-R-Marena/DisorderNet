#!/usr/bin/env python3
"""Do we solve the targets the field cannot, and can we show it per-target?

Two things a single pooled AUC per benchmark cannot answer.

**Statistical power.** Five benchmarks give five numbers, and three of them
(31 to 52 targets) cannot resolve any winner. But there are 658 targets across
the suite, and each one supports a per-target AUC. A Wilcoxon signed-rank over
per-target differences asks "do we beat them on more targets, by more" — a
different and often far more powerful question than the pooled bootstrap, and
one that does not care about a benchmark's total size in the same way.

**Difficulty.** Every entrant's predictions are published, so each target has a
consensus difficulty: the median per-target AUC across all 117 methods. That
identifies the targets nobody solves. A model that is merely well-calibrated on
easy targets and a model that cracks hard ones both post the same pooled score;
these do not.

Both analyses use only CAID3, which is genuinely held out for these checkpoints
— the leak filter removed every one of its 319 targets plus homologues.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import (  # noqa: E402
    LEADERS,
    TASKS,
    evaluated_mask,
    read_caid_predictions,
    read_reference,
)

REFS = os.environ.get("CAID3_OFFICIAL_DIR",
                      "/scratch4/sfried3/jbeale3_disordernet/caid3_official")
PREDS = os.environ.get("CAID3_PREDICTIONS_DIR",
                       "/scratch4/sfried3/jbeale3_disordernet/caid3_predictions")
OURS = os.environ.get("OUR_SUBMISSIONS", "")
OUR_NAME = os.environ.get("OUR_NAME", "DisorderNet")


def per_target_auc(ref, pred):
    """AUC within each target. None where the target has one class or no
    usable prediction — never silently zero."""
    from sklearn.metrics import roc_auc_score

    out = {}
    for tid, (_seq, lab) in ref.items():
        v = pred.get(tid)
        if v is None or len(v) != len(lab):
            out[tid] = None
            continue
        m = evaluated_mask(lab)
        y = (np.frombuffer(lab.encode(), dtype=np.uint8)[m] - ord("0")).astype(int)
        s = v[m]
        ok = np.isfinite(s)
        y, s = y[ok], s[ok]
        out[tid] = (roc_auc_score(y, s) if len(y) > 1 and len(np.unique(y)) > 1
                    else None)
    return out


def wilcoxon(diffs):
    """Signed-rank test on paired per-target differences.

    Zeros are dropped (Wilcoxon's own convention) and the normal approximation
    with a tie correction is used, which is appropriate well below 30 pairs and
    avoids depending on a specific scipy version's exact-test behaviour.
    """
    d = np.asarray([x for x in diffs if x is not None and x != 0.0],
                   dtype=np.float64)
    n = len(d)
    if n < 6:
        return {"n": n, "p": None, "reason": "too few non-tied pairs"}
    from scipy.stats import rankdata

    r = rankdata(np.abs(d))
    w_plus = float(r[d > 0].sum())
    w_minus = float(r[d < 0].sum())
    w = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie = (counts ** 3 - counts).sum()
    var = (n * (n + 1) * (2 * n + 1) - tie / 2.0) / 24.0
    if var <= 0:
        return {"n": n, "p": None, "reason": "degenerate variance"}
    from math import erfc, sqrt

    z = (w - mean) / sqrt(var)
    return {"n": n, "w_plus": w_plus, "w_minus": w_minus,
            "n_wins": int((d > 0).sum()), "n_losses": int((d < 0).sum()),
            "median_delta": float(np.median(d)),
            "p": float(erfc(abs(z) / sqrt(2.0)))}


def main() -> int:
    if not OURS:
        print("set OUR_SUBMISSIONS", file=sys.stderr)
        return 2

    all_diffs_vs_leader = []
    print(f"{'benchmark':<14}{'targets':>8}{'wins':>6}{'losses':>7}"
          f"{'median d':>10}{'p (signed-rank)':>17}  opponent")
    for task in TASKS:
        our_file = os.path.join(OURS, f"{OUR_NAME}-{task}.caid")
        if not os.path.isfile(our_file):
            continue
        ref = read_reference(os.path.join(REFS, f"{task}.fasta"))
        ours = per_target_auc(ref, read_caid_predictions(our_file))

        for opp in dict.fromkeys([LEADERS[task][0], "AlphaFold-rsa"]):
            path = os.path.join(PREDS, f"{opp}.caid")
            if not os.path.isfile(path):
                continue
            theirs = per_target_auc(ref, read_caid_predictions(path))
            diffs = [ours[t] - theirs[t] for t in ref
                     if ours.get(t) is not None and theirs.get(t) is not None]
            w = wilcoxon(diffs)
            if w["p"] is None:
                print(f"{task:<14}{len(diffs):>8}  {w['reason']}   {opp}")
                continue
            print(f"{task:<14}{len(diffs):>8}{w['n_wins']:>6}{w['n_losses']:>7}"
                  f"{w['median_delta']:>+10.4f}{w['p']:>17.5f}  {opp}")
            if opp == LEADERS[task][0]:
                all_diffs_vs_leader.extend(diffs)

    if all_diffs_vs_leader:
        w = wilcoxon(all_diffs_vs_leader)
        print(f"\nPooled across all benchmarks, versus each one's published "
              f"leader:\n  {w['n_wins']} wins, {w['n_losses']} losses over "
              f"{w['n']} targets, median {w['median_delta']:+.4f}, "
              f"p={w['p']:.6f}")
        print("  Targets are not independent between the Disorder-PDB and "
              "Disorder-NOX\n  references, which share proteins, so read this "
              "as a summary rather than\n  a sixth independent test.")

    # ── Difficulty, from the field itself ────────────────────────────────
    print(f"\n{'=' * 78}\nHard targets — difficulty is the median per-target "
          f"AUC across all entrants\n{'=' * 78}")
    for task in TASKS:
        our_file = os.path.join(OURS, f"{OUR_NAME}-{task}.caid")
        if not os.path.isfile(our_file):
            continue
        ref = read_reference(os.path.join(REFS, f"{task}.fasta"))
        ours = per_target_auc(ref, read_caid_predictions(our_file))

        field = {}
        for fn in sorted(os.listdir(PREDS)):
            if fn.endswith(".caid"):
                field[fn[:-5]] = per_target_auc(
                    ref, read_caid_predictions(os.path.join(PREDS, fn)))
        difficulty = {}
        for tid in ref:
            vals = [v[tid] for v in field.values() if v.get(tid) is not None]
            if len(vals) >= 20:
                difficulty[tid] = float(np.median(vals))
        if not difficulty:
            continue

        order = sorted(difficulty, key=difficulty.get)
        n_hard = max(len(order) // 4, 5)
        bands = {"hardest quartile": order[:n_hard],
                 "easiest quartile": order[-n_hard:]}
        print(f"\n{task}")
        for name, ids in bands.items():
            mine = [ours[t] for t in ids if ours.get(t) is not None]
            fieldv = [difficulty[t] for t in ids]
            # How many entrants beat us on these targets?
            beat = 0
            for m, v in field.items():
                vals = [v[t] for t in ids if v.get(t) is not None
                        and ours.get(t) is not None]
                if len(vals) < len(mine) * 0.8:
                    continue
                if np.mean(vals) > np.mean(mine):
                    beat += 1
            print(f"  {name:<20}n={len(ids):<4} field median "
                  f"{np.mean(fieldv):.4f}   ours {np.mean(mine):.4f}   "
                  f"entrants above us: {beat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
