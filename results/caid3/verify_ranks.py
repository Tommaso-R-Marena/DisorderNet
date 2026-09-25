#!/usr/bin/env python3
"""Independently recompute our rank on every CAID3 benchmark.

The rank reported by the evaluator is `1 + count(entrants scoring above us)`.
Three things could make that wrong, and each is checked here rather than
assumed:

1. **The field.** predictions.zip holds 117 files, but CAID's own Disorder-PDB
   assessment lists 71 methods — the rest are binding- or linker-specific
   predictors. Including them can only add methods *below* us, so it cannot
   flatter a rank; but the count in "rank N of M" changes, and M should be
   stated for what it is.

2. **Coverage.** Competitors are scored on whatever targets they returned, and
   declining targets is worth up to +0.118 AUC. Our score is at 100% coverage.
   Ranking us against their inflated numbers is the conservative direction, and
   this prints coverage beside every method so that stays visible.

3. **Ties and arithmetic.** An off-by-one, or a tie handled as a win.

Prints the neighbourhood around our score on each benchmark so the insertion
point can be checked by eye.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from colab.caid3_official import (  # noqa: E402
    LEADERS,
    TASKS,
    official_leaderboard,
    read_caid_predictions,
    read_reference,
    score_method,
)

REFS = os.environ.get("CAID3_OFFICIAL_DIR",
                      "/scratch4/sfried3/jbeale3_disordernet/caid3_official")
PREDS = os.environ.get("CAID3_PREDICTIONS_DIR",
                       "/scratch4/sfried3/jbeale3_disordernet/caid3_predictions")
OURS = os.environ.get("OUR_SUBMISSIONS", "")


def main() -> int:
    if not OURS:
        print("set OUR_SUBMISSIONS to a caid_submissions directory",
              file=sys.stderr)
        return 2

    for task in TASKS:
        ours_file = os.path.join(OURS, f"DisorderNet-{task}.caid")
        if not os.path.isfile(ours_file):
            continue
        ref = read_reference(os.path.join(REFS, f"{task}.fasta"))
        ours = score_method(ref, {k: np.asarray(v) for k, v in
                                  read_caid_predictions(ours_file).items()})
        board = official_leaderboard(task, REFS, PREDS)

        above = [r for r in board if r["auc"] > ours["auc"]]
        tied = [r for r in board if r["auc"] == ours["auc"]]
        rank = len(above) + 1
        full_cov = [r for r in board if r["coverage"] >= 1.0]
        above_full = [r for r in full_cov if r["auc"] > ours["auc"]]

        leader, pub_auc, _ = LEADERS[task]
        print(f"\n{'=' * 84}\n{task}: ours {ours['auc']:.4f} "
              f"(APS {ours['aps']:.4f}, {ours['n_scored_targets']}/"
              f"{ours['n_reference_targets']} targets, coverage "
              f"{ours['coverage']:.2f})\n{'=' * 84}")
        print(f"  rank {rank} of {len(board) + 1} scored entrants (+ us)")
        print(f"  rank {len(above_full) + 1} of {len(full_cov) + 1} among "
              f"full-coverage entrants only")
        print(f"  exact ties: {len(tied)}")
        print(f"  published leader {leader} = {pub_auc}")

        window = [r for r in board if abs(r["auc"] - ours["auc"]) < 0.05]
        window = sorted(window, key=lambda r: -r["auc"])[:10]
        print(f"\n  {'#':>4} {'method':<28}{'AUC':>8}{'cov':>7}"
              f"{'targets':>9}")
        shown = False
        for r in window:
            if not shown and r["auc"] < ours["auc"]:
                print(f"  {'>>':>4} {'DisorderNet (ours)':<28}"
                      f"{ours['auc']:>8.4f}{ours['coverage']:>7.2f}"
                      f"{ours['n_scored_targets']:>9}")
                shown = True
            print(f"  {r['rank']:>4} {r['method']:<28}{r['auc']:>8.4f}"
                  f"{r['coverage']:>7.2f}{r['n_scored_targets']:>9}")
        if not shown:
            print(f"  {'>>':>4} {'DisorderNet (ours)':<28}"
                  f"{ours['auc']:>8.4f}{ours['coverage']:>7.2f}"
                  f"{ours['n_scored_targets']:>9}")

        if above:
            worst = min(above, key=lambda r: r["auc"])
            print(f"\n  closest entrant above us: {worst['method']} "
                  f"{worst['auc']:.4f} at coverage {worst['coverage']:.2f}")
            partial = [r for r in above if r["coverage"] < 1.0]
            if partial:
                print(f"  of the {len(above)} above us, {len(partial)} did not "
                      f"predict every target:")
                for r in sorted(partial, key=lambda r: -r["auc"])[:6]:
                    print(f"      {r['method']:<26}{r['auc']:.4f}  "
                          f"{r['n_scored_targets']}/{r['n_reference_targets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
