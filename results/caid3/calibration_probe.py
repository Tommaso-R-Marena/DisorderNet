#!/usr/bin/env python3
"""Is Binding-IDR a calibration failure rather than a discrimination failure?

Per-target, our Binding-IDR predictions win 21 targets and lose 21 against a
leader that beats us by 0.14 on pooled AUC. Ranking residues *within* a protein
is therefore roughly as good as theirs; what fails is comparability *between*
proteins, which is exactly what a pooled AUC measures.

If that diagnosis is right, rank-normalising each protein's scores — which
discards all between-protein information and keeps only within-protein order —
should raise Binding-IDR and lower the tasks where between-protein information
is real signal.

Every task is reported. This transform is a single bit of choice per task, and
choosing it per task by looking at this table would be fitting on the benchmark;
the point here is to test a mechanism, and any adoption has to be justified
out-of-sample. Disorder-NOX already shows what happens when the bit is applied
where it does not belong: 0.8160 to 0.6952 in an earlier experiment.
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


def pooled_auc(ref, pred, per_protein_rank=False):
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score

    ys, ss = [], []
    for tid, (_seq, lab) in ref.items():
        v = pred.get(tid)
        if v is None or len(v) != len(lab):
            return None
        m = evaluated_mask(lab)
        y = (np.frombuffer(lab.encode(), dtype=np.uint8)[m] - ord("0")).astype(int)
        s = np.asarray(v)[m]
        ok = np.isfinite(s)
        y, s = y[ok], s[ok]
        if per_protein_rank and len(s):
            # Within-protein order only. Divide by n+1 so no protein reaches
            # exactly 1.0 and long proteins do not dominate the top of the
            # pooled ranking purely by having more residues.
            s = rankdata(s) / (len(s) + 1.0)
        ys.append(y)
        ss.append(s)
    y, s = np.concatenate(ys), np.concatenate(ss)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None


def main() -> int:
    if not OURS:
        print("set OUR_SUBMISSIONS", file=sys.stderr)
        return 2
    print(f"{'benchmark':<14}{'as-is':>9}{'per-protein':>13}{'delta':>9}"
          f"{'leader':>9}  verdict")
    for task in TASKS:
        path = os.path.join(OURS, f"{OUR_NAME}-{task}.caid")
        if not os.path.isfile(path):
            continue
        ref = read_reference(os.path.join(REFS, f"{task}.fasta"))
        pred = read_caid_predictions(path)
        raw = pooled_auc(ref, pred)
        norm = pooled_auc(ref, pred, per_protein_rank=True)
        if raw is None or norm is None:
            continue
        lead = LEADERS[task][1]
        verdict = ("calibration was hurting" if norm > raw + 0.005 else
                   "between-protein signal is real" if norm < raw - 0.005 else
                   "no material difference")
        print(f"{task:<14}{raw:>9.4f}{norm:>13.4f}{norm - raw:>+9.4f}"
              f"{lead:>9.3f}  {verdict}")

    print("\nFor reference, the same transform applied to the published "
          "leaders:")
    print(f"{'benchmark':<14}{'leader':<28}{'as-is':>9}{'per-protein':>13}"
          f"{'delta':>9}")
    for task in TASKS:
        ref = read_reference(os.path.join(REFS, f"{task}.fasta"))
        name = LEADERS[task][0]
        p = os.path.join(PREDS, f"{name}.caid")
        if not os.path.isfile(p):
            continue
        pred = read_caid_predictions(p)
        raw = pooled_auc(ref, pred)
        norm = pooled_auc(ref, pred, per_protein_rank=True)
        if raw is None or norm is None:
            print(f"{task:<14}{name:<28}  partial coverage, skipped")
            continue
        print(f"{task:<14}{name:<28}{raw:>9.4f}{norm:>13.4f}{norm - raw:>+9.4f}")
    print("\nIf the leaders lose as much as we gain, the transform is not a "
          "trick we\nfound but a property of the benchmark — and a claim built "
          "on it would be\nabout the metric rather than the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
