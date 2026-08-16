#!/usr/bin/env python3
"""What CAID's headline number is actually measuring, for all 115 entrants.

CAID scores by pooling every residue of every protein into one AUC. Every
positive-negative pair in that statistic is either **within** one protein or
**between** two, and the pooled AUC is exactly their pair-weighted average:

    AUC_pooled = w_within * AUC_within + w_between * AUC_between

The weights are not close to even. For n proteins of comparable size only about
1/n of pairs fall inside a protein, so on a 319-target reference roughly 99.7%
of the metric is the between-protein question — *does this chain have more
disorder than that one* — and roughly 0.3% is the within-protein question,
*which residues in this chain are disordered*.

That is arithmetic, and on its own it proves nothing. The question this script
asks is empirical and is not arithmetic: **does it change who wins?** If the
two rankings agree, the decomposition is a curiosity. If they disagree, then
the ordering of the field depends on which of two different abilities the
metric happens to weight, and the leaderboard is not the summary of
residue-level disorder prediction it is read as.

Fairness, since a within-protein comparison is easy to rig by declining
targets: only methods that predicted **every** reference target are compared,
and all of them are decomposed over the identical set of targets that carry
both classes. Coverage is a scoring choice CAID already allows; here it is
removed as a variable rather than adjusted for.

The final section prices the between-protein axis. If a training-free
descriptor — mean Kyte-Doolittle hydropathy over the chain, no fitting, no
network, no MSA — already answers the between-protein question about as well as
the field does, then most of what separates the leaders on the pooled metric is
not what the pooled metric is understood to reward.

    export ANALYSIS_SCRIPT=results/caid3/within_protein_leaderboard.py
    sbatch rockfish/slurm/analysis_cpu.sbatch
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.auc_decomposition import decompose_auc  # noqa: E402
from colab.caid3_official import (  # noqa: E402
    TASKS,
    TASKS_CAID2,
    evaluated_mask,
    read_caid_predictions,
    read_reference,
    verify_composition_for,
)
from colab.sequence_biophysics import mean_hydropathy  # noqa: E402

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
BENCH = os.environ.get("WPL_BENCHMARK", "caid3")
REFS = os.environ.get("WPL_REFS", f"{ROOT}/{BENCH}_official")
PREDS = os.environ.get("WPL_PREDS", f"{ROOT}/{BENCH}_predictions")
OUT = os.environ.get("WPL_OUT", "")
N_BOOT = int(os.environ.get("WPL_NBOOT", "200"))
SEED = int(os.environ.get("WPL_SEED", "20260816"))


def _aligned(ref, pred, targets):
    """(labels, scores) per target over exactly ``targets``, or None if the
    method cannot supply one of them at the reference's length."""
    ys, ss = [], []
    for tid in targets:
        _seq, lab = ref[tid]
        p = pred.get(tid)
        if p is None or len(p) != len(lab):
            return None
        m = evaluated_mask(lab)
        y = np.frombuffer(lab.encode(), dtype=np.uint8)[m].astype(np.int8) - ord("0")
        s = p[m]
        ok = np.isfinite(s)
        if not ok.all():
            return None
        ys.append(y)
        ss.append(s)
    return ys, ss


def full_coverage_methods(ref, preds_dir):
    """Methods that predicted every reference target at the right length.

    A method that declines targets can raise its within-protein AUC by
    declining the hard ones, so the comparison is restricted rather than
    corrected. This is also the population CAID itself reports as
    full-coverage.
    """
    out = {}
    for fn in sorted(os.listdir(preds_dir)):
        if not fn.endswith(".caid"):
            continue
        pred = read_caid_predictions(os.path.join(preds_dir, fn))
        ok = True
        for tid, (_s, lab) in ref.items():
            p = pred.get(tid)
            if p is None or len(p) != len(lab) or not np.isfinite(p).all():
                ok = False
                break
        if ok:
            out[fn[:-5]] = pred
    return out


def two_class_targets(ref):
    """Targets carrying both classes among evaluated residues.

    A target with one class contributes no within-protein pair at all, so it is
    weightless in AUC_within and would only make the two rankings look more
    alike than they are.
    """
    keep = []
    for tid, (_seq, lab) in ref.items():
        m = evaluated_mask(lab)
        if not m.any():
            continue
        y = np.frombuffer(lab.encode(), dtype=np.uint8)[m].astype(np.int8) - ord("0")
        if 0 < int(y.sum()) < len(y):
            keep.append(tid)
    return keep


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def clustered_spearman_ci(rows, n_targets, rng, n_boot):
    """Protein-clustered bootstrap CI for the pooled-vs-within rank agreement.

    Resampling proteins, not residues: residues inside a chain are anything but
    independent, and a residue bootstrap would return an interval far too
    narrow to mean anything.

    ``n_boot`` is small here by design. Each resample redecomposes every
    full-coverage method over every target, so the cost is n_boot x methods x
    residues; and the quantity this bounds is a rank correlation across ~60-90
    methods, whose interval is set by how many *proteins* there are, not by how
    many times they are redrawn. More resamples would shrink the Monte Carlo
    error in locating the percentile and leave the interval where it is.
    """
    stats = []
    for _ in range(n_boot):
        take = rng.integers(0, n_targets, size=n_targets)
        pooled, within = [], []
        for r in rows:
            ys = [r["ys"][i] for i in take]
            ss = [r["ss"][i] for i in take]
            d = decompose_auc(ys, ss)
            if d.get("pooled") is None or d.get("auc_within") is None:
                break
            pooled.append(d["pooled"])
            within.append(d["auc_within"])
        if len(pooled) == len(rows):
            stats.append(spearman(np.asarray(pooled), np.asarray(within)))
    if not stats:
        return None, None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def hydropathy_between_auc(ref, targets):
    """Between-protein AUC of a training-free protein-level descriptor.

    One number per chain — mean Kyte-Doolittle hydropathy, negated so that more
    polar reads as more disordered — broadcast to every residue in it. It has
    *no* within-protein resolution whatsoever: every residue of a chain gets the
    same score, so AUC_within is exactly 0.5 by construction. Whatever pooled
    AUC it reaches is therefore purely the between-protein axis.
    """
    ys, ss = [], []
    for tid in targets:
        seq, lab = ref[tid]
        m = evaluated_mask(lab)
        y = np.frombuffer(lab.encode(), dtype=np.uint8)[m].astype(np.int8) - ord("0")
        h = mean_hydropathy(seq)
        if not np.isfinite(h):
            continue
        ys.append(y)
        ss.append(np.full(len(y), -h, dtype=np.float64))
    if not ys:
        return None
    return decompose_auc(ys, ss)


def main() -> int:
    tasks = TASKS if BENCH == "caid3" else TASKS_CAID2
    report = {"benchmark": BENCH, "n_boot": N_BOOT, "tasks": {}}
    rng = np.random.default_rng(SEED)

    for task in tasks:
        path = os.path.join(REFS, f"{task}.fasta")
        if not os.path.isfile(path):
            continue
        verify_composition_for(BENCH, task, path)
        ref = read_reference(path)
        targets = two_class_targets(ref)
        methods = full_coverage_methods(ref, PREDS)

        print(f"\n{'=' * 100}")
        print(f" {BENCH.upper()} / {task}: {len(ref)} targets, "
              f"{len(targets)} with both classes, "
              f"{len(methods)} full-coverage methods")
        print("=" * 100)
        if len(methods) < 5 or len(targets) < 5:
            print(" too few to compare")
            continue

        rows = []
        for name, pred in methods.items():
            got = _aligned(ref, pred, targets)
            if got is None:
                continue
            ys, ss = got
            d = decompose_auc(ys, ss)
            if d.get("pooled") is None or d.get("auc_within") is None:
                continue
            rows.append({"method": name, "ys": ys, "ss": ss, **d})
        if len(rows) < 5:
            print(" too few decomposable methods")
            continue

        w_within = rows[0]["w_within"]
        print(f" pairs inside a protein: {w_within:.4%}   "
              f"(1/n = {1/len(targets):.4%})")
        print(f" so {1 - w_within:.2%} of the pooled AUC is the "
              f"between-protein question")

        by_pooled = sorted(rows, key=lambda r: -r["pooled"])
        by_within = sorted(rows, key=lambda r: -r["auc_within"])
        pooled_rank = {r["method"]: i for i, r in enumerate(by_pooled, 1)}
        within_rank = {r["method"]: i for i, r in enumerate(by_within, 1)}

        rho = spearman(np.asarray([r["pooled"] for r in rows]),
                       np.asarray([r["auc_within"] for r in rows]))
        lo, hi = clustered_spearman_ci(rows, len(targets), rng, N_BOOT)
        ci = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else "n/a"
        print(f" Spearman(pooled, within) = {rho:+.3f}  95% CI {ci}")

        print(f"\n {'#':>3} {'method':<28}{'pooled':>9}{'within':>9}"
              f"{'between':>9}{'within #':>10}{'move':>7}")
        for i, r in enumerate(by_pooled[:15], 1):
            wr = within_rank[r["method"]]
            print(f" {i:>3} {r['method']:<28}{r['pooled']:>9.4f}"
                  f"{r['auc_within']:>9.4f}{r['auc_between']:>9.4f}"
                  f"{wr:>10}{i - wr:>+7}")

        movers = sorted(rows,
                        key=lambda r: -abs(pooled_rank[r["method"]]
                                           - within_rank[r["method"]]))[:8]
        print(f"\n largest rank changes when the metric is restricted to the "
              f"within-protein question:")
        for r in movers:
            pr, wr = pooled_rank[r["method"]], within_rank[r["method"]]
            print(f"   {r['method']:<28} pooled #{pr:<4} within #{wr:<4} "
                  f"{pr - wr:+d}")

        hyd = hydropathy_between_auc(ref, targets)
        if hyd and hyd.get("auc_between") is not None:
            best_between = max(r["auc_between"] for r in rows)
            print(f"\n training-free mean-hydropathy baseline "
                  f"(one number per chain, no within-protein resolution):")
            print(f"   pooled {hyd['pooled']:.4f}   "
                  f"within {hyd['auc_within']:.4f} (0.5 by construction)   "
                  f"between {hyd['auc_between']:.4f}")
            print(f"   best entrant's between-protein AUC: {best_between:.4f}")

        report["tasks"][task] = {
            "n_targets_two_class": len(targets),
            "n_full_coverage_methods": len(rows),
            "w_within": w_within,
            "spearman_pooled_vs_within": rho,
            "spearman_ci": [lo, hi],
            "hydropathy_baseline": (
                {k: v for k, v in hyd.items() if k != "reason"} if hyd else None),
            "methods": [
                {"method": r["method"], "pooled": r["pooled"],
                 "auc_within": r["auc_within"], "auc_between": r["auc_between"],
                 "rank_pooled": pooled_rank[r["method"]],
                 "rank_within": within_rank[r["method"]]}
                for r in by_pooled
            ],
        }

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
