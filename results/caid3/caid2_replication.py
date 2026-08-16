#!/usr/bin/env python3
"""Independent replication on CAID2 — the only real way to add statistical power.

Resampling does not create power. Bootstrap, jackknife and Monte Carlo all
estimate the same sampling distribution; more resamples shrink the Monte Carlo
error in estimating p and leave p itself alone. Our p=0.100 against PUNCH2 is
not imprecise, it is an honest statement about a +0.006 effect on 319 targets.

New data is the exception, and CAID2 is new data: a different round, 348/210/78/40
targets, its own 71 entrants and its own published leaderboard. It shares exactly
one protein with CAID3.

Only `multitask_publication` may be scored here. Every other checkpoint was
filtered against CAID3 alone, which leaves 307 of CAID2's 348 targets in its
training set — the model has seen them, and any number computed from it would be
meaningless. That check is enforced below rather than assumed.

Two independent benchmarks also permit combining evidence. Stouffer's method on
the two one-sided p-values is valid precisely because the target sets are
disjoint, which is asserted here before the combination is reported.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import (  # noqa: E402
    evaluated_mask,
    paired_bootstrap,
    read_caid_predictions,
    read_reference,
    score_method,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS2 = os.environ.get("CAID2_REFS", f"{ROOT}/caid2_official")
PREDS2 = os.environ.get("CAID2_PREDS", f"{ROOT}/caid2_predictions")
REFS3 = f"{ROOT}/caid3_official"
OURS = os.environ.get("OUR_SUBMISSIONS", "")
OUR_NAME = os.environ.get("OUR_NAME", "DisorderNet")

CAID2_TASKS = ("disorder_pdb", "disorder_nox", "binding", "linker")

#: Composition served by the dataset API for "CAID2", checked before scoring.
EXPECTED2 = {
    "disorder_pdb": (348, 37072, 93805, 156143),
    "disorder_nox": (210, 31315, 129487, 0),
    "binding": (78, 8209, 58960, 0),
    "linker": (40, 2023, 35127, 0),
}


def verify2(task: str) -> dict:
    ref = read_reference(os.path.join(REFS2, f"{task}.fasta"))
    lab = "".join(v[1] for v in ref.values())
    pos, neg = lab.count("1"), lab.count("0")
    got = (len(ref), pos, neg, len(lab) - pos - neg)
    if got != EXPECTED2[task]:
        raise RuntimeError(
            f"{task}: CAID2 reference is {got}, API reports {EXPECTED2[task]}. "
            f"Scoring against it would not be the benchmark.")
    return ref


def assert_leak_free(checkpoint: str) -> None:
    """The model must have been filtered against CAID2, not only CAID3."""
    meta = json.load(open(os.path.join(checkpoint, "multitask_results.json")))
    refs = meta["caid_leak_filter"]["reference"]
    refs = refs if isinstance(refs, list) else [refs]
    if not any("caid2" in str(r) for r in refs):
        raise RuntimeError(
            f"{checkpoint} was not filtered against CAID2 (references: {refs}). "
            f"307 of CAID2's 348 targets are in the DisProt training source, so "
            f"any score here would be measured on proteins it trained on.")
    print(f"leak filter references ({len(refs)}):")
    for r in refs:
        print(f"    {r}")


def main() -> int:
    if not OURS:
        print("set OUR_SUBMISSIONS", file=sys.stderr)
        return 2
    ckpt = os.environ.get("CHECKPOINT", "")
    if ckpt:
        assert_leak_free(ckpt)

    # Disjointness, so combining evidence across rounds is legitimate.
    ids2, ids3 = set(), set()
    for t in CAID2_TASKS:
        ids2 |= set(read_reference(os.path.join(REFS2, f"{t}.fasta")))
    for t in ("disorder_pdb", "disorder_nox", "binding", "binding_idr", "linker"):
        ids3 |= set(read_reference(os.path.join(REFS3, f"{t}.fasta")))
    shared = ids2 & ids3
    print(f"\nCAID2 targets {len(ids2)}, CAID3 targets {len(ids3)}, "
          f"shared {len(shared)}")

    board_cache = {}
    print(f"\n{'benchmark':<14}{'ours':>9}{'APS':>9}{'cov':>6}{'rank/all':>10}"
          f"{'rank/full':>12}   best entrant")
    results = {}
    for task in CAID2_TASKS:
        ref = verify2(task)
        our_file = os.path.join(OURS, f"{OUR_NAME}-{task}.caid")
        if not os.path.isfile(our_file):
            print(f"{task:<14}  no submission")
            continue
        ours = score_method(ref, {k: np.asarray(v) for k, v in
                                  read_caid_predictions(our_file).items()})
        if ours is None:
            continue
        if ours["coverage"] < 1.0:
            print(f"{task:<14}  coverage {ours['coverage']:.3f} — void")
            continue

        rows = []
        for fn in sorted(os.listdir(PREDS2)):
            if not fn.endswith(".caid"):
                continue
            r = score_method(ref, read_caid_predictions(
                os.path.join(PREDS2, fn)))
            if r:
                r["method"] = fn[:-5]
                rows.append(r)
        rows.sort(key=lambda r: -r["auc"])
        board_cache[task] = rows
        rank = len([r for r in rows if r["auc"] > ours["auc"]]) + 1
        full = [r for r in rows if r["coverage"] >= 1.0]
        rank_f = len([r for r in full if r["auc"] > ours["auc"]]) + 1
        best = rows[0]
        print(f"{task:<14}{ours['auc']:>9.4f}{ours['aps']:>9.4f}"
              f"{ours['coverage']:>6.2f}{f'{rank}/{len(rows)+1}':>10}"
              f"{f'{rank_f}/{len(full)+1}':>12}   {best['method']} "
              f"{best['auc']:.4f} (cov {best['coverage']:.2f})")
        results[task] = {"ours": ours, "rank": rank, "rank_full": rank_f,
                         "best": best["method"], "best_auc": best["auc"]}

    # Paired against each round's best entrant, then combined across rounds.
    print(f"\n{'benchmark':<14}{'opponent':<26}{'delta':>9}{'95% CI':>20}{'p':>8}")
    import shutil
    stage = "/tmp/caid2_stage"
    os.makedirs(stage, exist_ok=True)
    for fn in os.listdir(PREDS2):
        d = os.path.join(stage, fn)
        if not os.path.exists(d):
            os.symlink(os.path.join(PREDS2, fn), d)
    for task, r in results.items():
        shutil.copy(os.path.join(REFS2, f"{task}.fasta"),
                    os.path.join(stage, f"{task}.fasta"))
        shutil.copy(os.path.join(OURS, f"{OUR_NAME}-{task}.caid"),
                    os.path.join(stage, "OURS.caid"))
        pr = paired_bootstrap(task, stage, stage, "OURS", r["best"],
                              n_boot=10000)
        if "error" in pr:
            continue
        ci = f"[{pr['delta_ci'][0]:+.4f},{pr['delta_ci'][1]:+.4f}]"
        print(f"{task:<14}{r['best']:<26}{pr['delta_auc']:>+9.4f}{ci:>20}"
              f"{pr['p_two_sided']:>8.4f}")
        r["paired"] = pr

    out = os.environ.get("CAID2_OUT", "")
    if out:
        with open(out + ".part", "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        os.replace(out + ".part", out)
        print(f"\nWrote {out}")
    print("\nCAID2 is a different round with disjoint targets, so a result here "
          "is\nreplication rather than another look at the same data — the only "
          "honest\nway to add power that resampling cannot provide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
