#!/usr/bin/env python3
"""Where does each benchmark's difficulty live — within proteins or between?

Sizes the gap before any GPU is spent closing it. For every benchmark, our
pooled AUC and each leader's are split into the two abilities a single AUC
conflates, and the difference is attributed to one or the other.

A per-residue model with a bounded receptive field has a mechanism for the
within-protein part and none at all for the between-protein part: a 213-residue
field over a 1,000-residue chain never sees the chain. So if a deficit is
between-protein, no amount of local modelling will close it, and if it is
within-protein, a global term will not help.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.auc_decomposition import (  # noqa: E402
    compare_decompositions,
    decompose_auc,
)
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


def arrays(ref, pred, targets):
    ys, ss = [], []
    for tid in targets:
        lab = ref[tid][1]
        v = pred.get(tid)
        if v is None or len(v) != len(lab):
            return None, None
        m = evaluated_mask(lab)
        y = (np.frombuffer(lab.encode(), dtype=np.uint8)[m] - ord("0")).astype(int)
        s = np.asarray(v)[m]
        ok = np.isfinite(s)
        ys.append(y[ok])
        ss.append(s[ok])
    return ys, ss


def main() -> int:
    if not OURS:
        print("set OUR_SUBMISSIONS", file=sys.stderr)
        return 2

    print(f"{'benchmark':<14}{'who':<26}{'pooled':>8}{'within':>8}"
          f"{'between':>9}{'w_within':>10}")
    for task in TASKS:
        our_file = os.path.join(OURS, f"{OUR_NAME}-{task}.caid")
        if not os.path.isfile(our_file):
            continue
        ref = read_reference(os.path.join(REFS, f"{task}.fasta"))
        ours_pred = read_caid_predictions(our_file)
        leader = LEADERS[task][0]
        lead_pred = read_caid_predictions(os.path.join(PREDS, f"{leader}.caid"))

        # Same targets for both, or the weights differ and no attribution is
        # possible. The leader may cover fewer, so intersect.
        targets = [t for t in ref
                   if t in ours_pred and len(ours_pred[t]) == len(ref[t][1])
                   and t in lead_pred and len(lead_pred[t]) == len(ref[t][1])]
        if not targets:
            continue

        ys, ss = arrays(ref, ours_pred, targets)
        yl, sl = arrays(ref, lead_pred, targets)
        a, b = decompose_auc(ys, ss), decompose_auc(yl, sl)
        if a.get("pooled") is None or b.get("pooled") is None:
            continue

        print(f"{task:<14}{'ours':<26}{a['pooled']:>8.4f}"
              f"{a['auc_within']:>8.4f}{a['auc_between']:>9.4f}"
              f"{a['w_within']:>10.4f}")
        print(f"{'':<14}{leader:<26}{b['pooled']:>8.4f}"
              f"{b['auc_within']:>8.4f}{b['auc_between']:>9.4f}"
              f"{b['w_within']:>10.4f}")
        c = compare_decompositions(a, b)
        share = ""
        if abs(c["delta_pooled"]) > 1e-9:
            frac = c["contribution_between"] / c["delta_pooled"]
            share = f"   between accounts for {frac:>6.1%} of the gap"
        print(f"{'':<14}{'difference':<26}{c['delta_pooled']:>+8.4f}"
              f"{c['delta_within']:>+8.4f}{c['delta_between']:>+9.4f}{share}")
        print()

    print("w_within is the fraction of positive-negative pairs that fall inside")
    print("one protein. Where it is small, the pooled score is mostly a")
    print("between-protein judgement, and a purely local model has no mechanism")
    print("for it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
