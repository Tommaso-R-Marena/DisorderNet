#!/usr/bin/env python3
"""Rank-fuse the checkpoints that share the fixed validation holdout.

Registered in `PREREGISTRATION_10.md` before any number from this existed.

Membership is decided on the **holdout** — data none of the members saw and all
of them share — and never on CAID3. Weights are equal and not fitted. Both
choices are fixed in the registration, so neither can be tuned after the fact.

The three older checkpoints trained *on* the holdout chains and are excluded by
a check on their own recorded metadata, not by a list maintained here.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import (  # noqa: E402
    TASKS,
    paired_bootstrap,
    rank_fuse,
    read_caid_predictions,
    read_reference,
    score_method,
    verify_composition_for,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("ENS_REFS", f"{ROOT}/caid3_official")
PREDS = os.environ.get("ENS_PREDS", f"{ROOT}/caid3_predictions")
OUT_DIR = os.environ.get("ENS_SUBMISSIONS", f"{ROOT}/ensemble_holdout")
OUT = os.environ.get("ENS_OUT", "")
#: The registered membership window: a checkpoint joins a task's ensemble iff
#: its holdout AUC is within this of the best holdout AUC for that task.
WINDOW = float(os.environ.get("ENS_WINDOW", "0.02"))
CANDIDATES = os.environ.get(
    "ENS_CANDIDATES",
    "multitask_chiral,multitask_control,multitask_motif,multitask_wass,"
    "multitask_rank").split(",")


def holdout_scores(run: str) -> dict | None:
    """The run's own recorded holdout, or None if it did not reserve one.

    Read from the checkpoint's metadata rather than from a list here, so a
    checkpoint that trained on the holdout chains cannot be included by
    forgetting to update this file.
    """
    path = os.path.join(ROOT, run, "multitask_results.json")
    if not os.path.isfile(path):
        return None
    meta = json.load(open(path))
    vh = meta.get("validation_holdout") or {}
    stats, scores = vh.get("stats") or {}, vh.get("scores") or {}
    if not scores or stats.get("salt") != "disordernet-validation-holdout-v1":
        return None
    return {"stats": stats, "auc": {t: v["auc"] for t, v in scores.items()}}


def main() -> int:
    pool = {}
    for run in [c.strip() for c in CANDIDATES if c.strip()]:
        got = holdout_scores(run)
        if got is None:
            print(f"  excluded {run}: no matching validation holdout recorded")
            continue
        pool[run] = got
    if len(pool) < 2:
        print(f"only {len(pool)} eligible checkpoints — nothing to fuse",
              file=sys.stderr)
        return 1

    shares = {r["stats"].get("holdout_share") for r in pool.values()}
    fracs = {r["stats"].get("fraction_requested") for r in pool.values()}
    if len(fracs) != 1:
        print(f"members reserved different holdout fractions {fracs}; they are "
              f"not comparable", file=sys.stderr)
        return 1
    print(f"{len(pool)} eligible checkpoints, holdout share "
          f"{sorted(shares)}, fraction {fracs.pop()}")

    print(f"\n{'checkpoint':<24}" + "".join(f"{t[:11]:>13}" for t in TASKS))
    for run, r in pool.items():
        print(f"{run:<24}" + "".join(
            f"{r['auc'].get(t, float('nan')):>13.4f}" for t in TASKS))

    os.makedirs(OUT_DIR, exist_ok=True)
    stage = os.path.join(OUT_DIR, "_paired")
    os.makedirs(stage, exist_ok=True)
    for fn in os.listdir(PREDS):
        d = os.path.join(stage, fn)
        if fn.endswith(".caid") and not os.path.exists(d):
            os.symlink(os.path.join(PREDS, fn), d)

    report = {"window": WINDOW, "pool": list(pool), "tasks": {}}
    for task in TASKS:
        path = os.path.join(REFS, f"{task}.fasta")
        verify_composition_for("caid3", task, path)
        ref = read_reference(path)

        have = {r: p["auc"][task] for r, p in pool.items() if task in p["auc"]}
        if not have:
            continue
        best = max(have.values())
        members = sorted(r for r, v in have.items() if v >= best - WINDOW)
        print(f"\n{task}: best holdout {best:.4f}; members within {WINDOW} → "
              f"{', '.join(m.replace('multitask_', '') for m in members)}")
        if len(members) < 2:
            print("  only one member — the ensemble is that checkpoint")

        preds = []
        for m in members:
            p = os.path.join(ROOT, m, "caid_submissions",
                             f"DisorderNet-{task}.caid")
            if not os.path.isfile(p):
                print(f"  missing submission for {m}; excluded")
                continue
            preds.append({k: np.asarray(v) for k, v in
                          read_caid_predictions(p).items()})
        if not preds:
            continue
        fused = rank_fuse(preds, ref) if len(preds) > 1 else preds[0]
        if not fused:
            continue

        out_path = os.path.join(OUT_DIR, f"Ensemble-{task}.caid")
        with open(out_path + ".part", "w") as fh:
            for tid, (seq, _lab) in ref.items():
                v = fused.get(tid)
                if v is None:
                    continue
                fh.write(f">{tid}\n")
                for i, (aa, p) in enumerate(zip(seq, v), 1):
                    fh.write(f"{i}\t{aa}\t{p:.4f}\t{int(p >= 0.5)}\n")
        os.replace(out_path + ".part", out_path)

        scored = score_method(ref, {k: np.asarray(v) for k, v in
                                    read_caid_predictions(out_path).items()})
        if not scored:
            continue
        shutil.copy(out_path, os.path.join(stage, "Ensemble.caid"))
        shutil.copy(path, os.path.join(stage, f"{task}.fasta"))

        from colab.caid3_official import official_leaderboard
        board = official_leaderboard(task, REFS, PREDS)
        rank = len([b for b in board if b["auc"] > scored["auc"]]) + 1
        full = [b for b in board if b["coverage"] >= 1.0]
        rank_f = len([b for b in full if b["auc"] > scored["auc"]]) + 1
        print(f"  ensemble AUC {scored['auc']:.4f}  coverage "
              f"{scored['coverage']:.2f}  rank {rank}/{len(board)+1}  "
              f"full-coverage {rank_f}/{len(full)+1}")

        row = {"members": members, "auc": scored["auc"], "aps": scored["aps"],
               "coverage": scored["coverage"], "rank": rank,
               "rank_full": rank_f, "best_holdout": best, "paired": {}}
        for opp in ("AlphaFold-rsa", "PUNCH2"):
            if not os.path.exists(os.path.join(stage, f"{opp}.caid")):
                continue
            pr = paired_bootstrap(task, stage, stage, "Ensemble", opp,
                                  n_boot=10000)
            if "error" in pr:
                continue
            ci = f"[{pr['delta_ci'][0]:+.4f},{pr['delta_ci'][1]:+.4f}]"
            print(f"    vs {opp:<16}{pr['delta_auc']:>+9.4f}{ci:>21}"
                  f"  p={pr['p_two_sided']:.4f}")
            row["paired"][opp] = pr
        report["tasks"][task] = row

    from colab.caid3_official import holm_bonferroni
    prim = (report["tasks"].get("disorder_pdb") or {}).get("paired") or {}
    fam = {o: prim[o]["p_two_sided"] for o in ("AlphaFold-rsa", "PUNCH2")
           if o in prim}
    if fam:
        print(f"\n{'=' * 78}\n PRE-REGISTERED (PREREGISTRATION_10.md): "
              f"Disorder-PDB, Holm across two\n{'=' * 78}")
        for o, v in sorted(holm_bonferroni(fam).items(),
                           key=lambda kv: kv[1]["rank"]):
            tag = "P1" if o == "AlphaFold-rsa" else "P2"
            print(f" {tag} — beat {o:<16}{prim[o]['delta_auc']:>+9.4f}  "
                  f"p={v['p_raw']:.4f}  adj={v['p_adjusted']:.4f}  "
                  f"{'CONFIRMED' if v['significant'] else 'not significant'}")
        report["preregistered"] = holm_bonferroni(fam)

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
