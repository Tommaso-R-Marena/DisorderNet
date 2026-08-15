#!/usr/bin/env python3
"""Do our two checkpoints combine, and does the combination cost anything?

`mt_full` and `mt_windowed` differ on one training decision — whether proteins
over 1022 residues are kept as overlapping windows or dropped — and they win
different benchmarks. mt_full is better on Disorder-PDB (0.9636 against 0.9595)
and Binding-IDR; mt_windowed is far better on Disorder-NOX (0.8928 against
0.8479), Linker and Binding.

An equal-weight rank fusion of the two is parameter-free and, unlike fusing with
AlphaFold-rsa, entirely self-contained: both models are ours, so the result is
something we could submit.

Every benchmark is reported, not the flattering ones, and the per-model scores
sit beside the ensemble so a loss is as visible as a gain. This is exploratory:
the confirmatory claim belongs to `mt_windowed` alone, under
`PREREGISTRATION.md`.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import (  # noqa: E402
    LEADERS,
    TASKS,
    official_leaderboard,
    rank_fuse,
    read_caid_predictions,
    read_reference,
    score_method,
)

REFS = os.environ.get("CAID3_OFFICIAL_DIR",
                      "/scratch4/sfried3/jbeale3_disordernet/caid3_official")
PREDS = os.environ.get("CAID3_PREDICTIONS_DIR",
                       "/scratch4/sfried3/jbeale3_disordernet/caid3_predictions")
ROOT = "/scratch4/sfried3/jbeale3_disordernet"
MEMBERS = {
    "mt_full": f"{ROOT}/multitask_full/caid_submissions_widefix",
    "mt_windowed": f"{ROOT}/multitask_windowed/caid_submissions_widefix",
}


def load(directory, task, ref):
    path = os.path.join(directory, f"DisorderNet-{task}.caid")
    if not os.path.isfile(path):
        return None
    p = {k: np.asarray(v) for k, v in read_caid_predictions(path).items()}
    if any(t not in p or len(p[t]) != len(ref[t][1]) for t in ref):
        return None
    return p


def main() -> int:
    print(f"{'benchmark':<14}{'mt_full':>9}{'windowed':>10}{'ensemble':>10}"
          f"{'best':>10}{'rank/all':>10}{'rank/full':>11}  leader")
    for task in TASKS:
        ref = read_reference(os.path.join(REFS, f"{task}.fasta"))
        members = {n: load(d, task, ref) for n, d in MEMBERS.items()}
        if any(v is None for v in members.values()):
            print(f"{task:<14}  incomplete submissions, skipped")
            continue

        singles = {n: score_method(ref, p)["auc"] for n, p in members.items()}
        fused = rank_fuse(list(members.values()), ref)
        ens = score_method(ref, fused)

        board = official_leaderboard(task, REFS, PREDS)
        rank = len([r for r in board if r["auc"] > ens["auc"]]) + 1
        full = [r for r in board if r["coverage"] >= 1.0]
        rank_full = len([r for r in full if r["auc"] > ens["auc"]]) + 1

        best_single = max(singles.values())
        mark = "+" if ens["auc"] > best_single else ("=" if
                                                     ens["auc"] == best_single
                                                     else "-")
        leader = LEADERS[task]
        delta = f"{mark}{ens['auc'] - best_single:+.4f}"
        print(f"{task:<14}{singles['mt_full']:>9.4f}"
              f"{singles['mt_windowed']:>10.4f}{ens['auc']:>10.4f}"
              f"{delta:>10}"
              f"{f'{rank}/{len(board) + 1}':>10}{f'{rank_full}/{len(full) + 1}':>11}"
              f"  {leader[0]} {leader[1]}")
        assert ens["coverage"] == 1.0, f"{task}: ensemble coverage < 1"
    print("\nlast two columns are the ensemble's rank; '+' means the ensemble "
          "beat both members")
    return 0




def paired_analysis(out_dir: str) -> int:
    """Write the ensemble as .caid submissions and test it against the leaders.

    Reported as **exploratory**, with disclosure: the ensemble's point estimates
    were seen before these p-values were computed, so this is not a confirmatory
    test in the sense `PREREGISTRATION.md` means. The confirmatory claim belongs
    to `mt_windowed` alone. What is reported here is a family of comparisons
    declared in one go and corrected within itself, not a search.
    """
    from colab.caid3_official import holm_bonferroni, paired_bootstrap

    os.makedirs(out_dir, exist_ok=True)

    scored = {}
    for task in TASKS:
        ref = read_reference(os.path.join(REFS, f"{task}.fasta"))
        members = {n: load(d, task, ref) for n, d in MEMBERS.items()}
        if any(v is None for v in members.values()):
            continue
        fused = rank_fuse(list(members.values()), ref)
        path = os.path.join(out_dir, f"DisorderNet-Ensemble-{task}.caid")
        with open(path + ".part", "w") as fh:
            for tid, (seq, _lab) in ref.items():
                v = fused.get(tid)
                if v is None:
                    continue
                fh.write(f">{tid}\n")
                for i, (aa, p) in enumerate(zip(seq, v), 1):
                    fh.write(f"{i}\t{aa}\t{p:.4f}\t{int(p >= 0.5)}\n")
        os.replace(path + ".part", path)
        scored[task] = score_method(
            ref, {k: np.asarray(v) for k, v in
                  read_caid_predictions(path).items()})

        # One staging directory per task. A single shared filename was reused
        # across tasks, and every task but the last read a stale file: the
        # Disorder-PDB comparison was computed against Linker's predictions and
        # reported a delta of +0.0488 where the truth is +0.0139. The bug was
        # invisible in the output and only surfaced because two of our own
        # numbers disagreed.
        import shutil
        staged = os.path.join(out_dir, f"_paired_{task}")
        os.makedirs(staged, exist_ok=True)
        for fn in os.listdir(PREDS):
            dst = os.path.join(staged, fn)
            if not os.path.exists(dst):
                os.symlink(os.path.join(PREDS, fn), dst)
        shutil.copy(path, os.path.join(staged, "DisorderNet-Ensemble.caid"))
        shutil.copy(os.path.join(REFS, f"{task}.fasta"),
                    os.path.join(staged, f"{task}.fasta"))

        # Verify the staged copy is the file we think it is before trusting any
        # comparison computed from it.
        staged_score = score_method(
            read_reference(os.path.join(staged, f"{task}.fasta")),
            {k: np.asarray(v) for k, v in read_caid_predictions(
                os.path.join(staged, "DisorderNet-Ensemble.caid")).items()})
        if abs(staged_score["auc"] - scored[task]["auc"]) > 1e-9:
            raise RuntimeError(
                f"{task}: staged ensemble scores {staged_score['auc']:.4f} but "
                f"the archived file scores {scored[task]['auc']:.4f}. The "
                f"staging directory is serving a different file; every paired "
                f"comparison from it would be wrong.")

        for opp in dict.fromkeys([LEADERS[task][0], "AlphaFold-rsa"]):
            if not os.path.exists(os.path.join(staged, f"{opp}.caid")):
                continue
            r = paired_bootstrap(task, staged, staged, "DisorderNet-Ensemble",
                                 opp, n_boot=10000)
            if abs(r["auc_a"] - scored[task]["auc"]) > 1e-9:
                raise RuntimeError(
                    f"{task} vs {opp}: paired test scored us at "
                    f"{r['auc_a']:.4f}, archived file gives "
                    f"{scored[task]['auc']:.4f}")
            scored.setdefault("_paired", {})[f"{task}: {opp}"] = r

    ps = {k: v["p_two_sided"] for k, v in scored.get("_paired", {}).items()
          if "p_two_sided" in v}
    holm = holm_bonferroni(ps)
    print(f"\n{'=' * 92}\n ENSEMBLE, paired on shared targets — EXPLORATORY"
          f"\n{'=' * 92}")
    print(f"{'comparison':<44}{'delta':>9}{'95% CI':>20}{'p':>8}{'adj':>8}")
    for name, v in sorted(holm.items(), key=lambda kv: kv[1]["rank"]):
        r = scored["_paired"][name]
        ci = f"[{r['delta_ci'][0]:+.4f},{r['delta_ci'][1]:+.4f}]"
        print(f"{name:<44}{r['delta_auc']:>+9.4f}{ci:>20}"
              f"{v['p_raw']:>8.4f}{v['p_adjusted']:>8.4f}"
              f"{'  sig' if v['significant'] else ''}")
    print(f"\nsubmissions written to {out_dir}")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc == 0 and os.environ.get("ENSEMBLE_OUT"):
        rc = paired_analysis(os.environ["ENSEMBLE_OUT"])
    raise SystemExit(rc)
