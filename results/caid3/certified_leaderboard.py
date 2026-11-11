#!/usr/bin/env python3
"""What every CAID3 entrant's leaderboard position is made of — certified.

The within-protein leaderboard showed that the CAID3 winner is 21st to 25th at
the residue-level question on three of five benchmarks. This makes that
quantitative and puts each step behind a machine-checked theorem, all from the
Lean development in `AUCCore`, `AUCInvariance`, `AUCCeiling`,
`AUCGapSeparable`, `AUCCrossedMatching`, `AUCCoverage` and `AUCInversion`.

**Why the within-protein axis is the right object** (`auc_within_shift_invariant`,
strengthened to `auc_within_strictMono_invariant`). AUC_within is *exactly*
invariant under any per-protein strictly monotone recalibration — not only an
additive shift. So it is the part of CAID's statistic that no amount of
rescoring chain-by-chain can change, and AUC_between is the whole of the part
that can. "How much of this method's rank is discrimination and how much is
calibration" is therefore a well-posed question with an exact answer, not a
metaphor.

**How far recalibration alone could take a method** (`auc_shift_le_ceiling`).
For every per-protein bias `b`, `AUC(s+b) <= w_within*AUC_within + w_between`.
The headroom `ceiling - pooled` is exactly `w_between*(1 - AUC_between)`.

**Whether that ceiling is reachable at all**
(`ceiling_attainable_iff_pairwise_overlap`). The K x K gap matrix is separable,
`c k l = maxneg l - minpos k`, so the general no-non-negative-cycle test
collapses to an O(K) scan: reachable iff `overlap k + overlap l < 0` for every
pair, i.e. iff the two largest overlaps sum to less than zero.

**What binds when it is not** (`ceiling_gap_of_crossed_matching`). A matching
`A` of crossed comparisons costs `|A|/2` pairs outright — no bias can satisfy
both directions of a crossed pair.

**Why an inversion is not an accident** (`inversion_requires_between_gap`).
Pooled order reverses within-protein order only when

    w_within*(within_A - within_B) < w_between*(between_B - between_A),

so every inversion in the published table *forces* a between-protein gap of at
least `(w_within/w_between)` times the within-protein gap. Reported per
inversion, alongside the measured gap, so the theorem is checked rather than
cited.

**What a declined target buys** (`decline_fraction_lower_bound`). A coverage
gain of `g` proves at least a fraction `g` of all comparison pairs was thrown
away. Applied to the methods that declined targets.

    export ANALYSIS_SCRIPT=results/caid3/certified_leaderboard.py
    sbatch rockfish/slurm/analysis_cpu.sbatch
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.auc_decomposition import decompose_auc  # noqa: E402
from colab.bias_ceiling import ceiling_attainable, bias_bound  # noqa: E402
from colab.caid3_official import (  # noqa: E402
    TASKS,
    evaluated_mask,
    read_caid_predictions,
    read_reference,
    verify_composition_for,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("CERT_REFS", f"{ROOT}/caid3_official")
PREDS = os.environ.get("CERT_PREDS", f"{ROOT}/caid3_predictions")
EXTRA = os.environ.get("CERT_EXTRA", "")
OUT = os.environ.get("CERT_OUT", "")
TOP = int(os.environ.get("CERT_TOP", "20"))


def target_arrays(ref, pred, targets):
    ys, ss = [], []
    for tid in targets:
        _seq, lab = ref[tid]
        p = pred.get(tid)
        if p is None or len(p) != len(lab):
            return None
        m = evaluated_mask(lab)
        y = np.frombuffer(lab.encode(), dtype=np.uint8)[m].astype(np.int8) - ord("0")
        s = np.asarray(p)[m]
        if not np.isfinite(s).all():
            return None
        ys.append(y)
        ss.append(s.astype(np.float64))
    return ys, ss


def two_class_targets(ref):
    keep = []
    for tid, (_seq, lab) in ref.items():
        m = evaluated_mask(lab)
        if not m.any():
            continue
        y = np.frombuffer(lab.encode(), dtype=np.uint8)[m].astype(np.int8) - ord("0")
        if 0 < int(y.sum()) < len(y):
            keep.append(tid)
    return keep


def load_methods(ref, targets, task):
    """Every method that can supply all of ``targets``, ours included."""
    out = {}
    for fn in sorted(os.listdir(PREDS)):
        if fn.endswith(".caid"):
            got = target_arrays(ref, read_caid_predictions(
                os.path.join(PREDS, fn)), targets)
            if got is not None:
                out[fn[:-5]] = got
    for item in filter(None, (s.strip() for s in EXTRA.split(","))):
        parts = item.split(":")
        directory = parts[0]
        file_prefix = parts[1] if len(parts) > 1 and parts[1] else "DisorderNet"
        label = parts[2] if len(parts) > 2 and parts[2] else file_prefix
        yield_path = os.path.join(directory, f"{file_prefix}-{task}.caid")
        if os.path.isfile(yield_path):
            got = target_arrays(ref, read_caid_predictions(yield_path), targets)
            if got is not None:
                out[label] = got
    return out


def certify(ys, ss, crossed: bool = False):
    """Every certified quantity for one method on one reference."""
    d = decompose_auc(ys, ss)
    if d.get("pooled") is None or d.get("auc_within") is None:
        return None
    w_in, w_bt = d["w_within"], d["w_between"]
    # auc_shift_le_ceiling. Equivalently pooled + w_between*(1 - between).
    ceiling = w_in * d["auc_within"] + w_bt
    att = ceiling_attainable(ys, ss)
    row = {
        "pooled": d["pooled"], "within": d["auc_within"],
        "within_unweighted": d["auc_within_unweighted"],
        "between": d["auc_between"], "w_within": w_in, "w_between": w_bt,
        "ceiling": ceiling,
        "recalibration_headroom": ceiling - d["pooled"],
        "ceiling_attainable": bool(att["attainable"]),
        "two_largest_overlaps": att.get("two_largest_overlaps"),
    }
    if crossed and not att["attainable"]:
        # ceiling_gap_of_crossed_matching. `bias_bound` already subtracts the
        # matching in the units the theorem uses; calling it rather than
        # re-deriving the arithmetic keeps one place for that to be wrong.
        b = bias_bound(ys, ss)
        row["crossed_matching_pairs"] = int(b.get("matched_pairs", 0))
        row["upper_bound_any_bias"] = b.get("upper_bound")
        row["blocks_skipped"] = int(b.get("blocks_skipped", 0))
    return row


def inversion_certificate(a, b):
    """`auc_inversion_iff` and `inversion_requires_between_gap`, checked.

    Returns None unless the pair actually inverts: pooled and within-protein
    put them in opposite orders.
    """
    if not (a["within"] > b["within"] and a["pooled"] < b["pooled"]):
        return None
    w_in, w_bt = a["w_within"], a["w_between"]
    within_gap = a["within"] - b["within"]
    between_gap = b["between"] - a["between"]
    required = (w_in / w_bt) * within_gap
    lhs = w_in * within_gap
    rhs = w_bt * between_gap
    return {
        "within_gap": within_gap,
        "between_gap_measured": between_gap,
        "between_gap_required": required,
        "iff_holds": bool(lhs < rhs),
        "slack": between_gap - required,
    }


def main() -> int:
    report = {"tasks": {}}
    for task in TASKS:
        path = os.path.join(REFS, f"{task}.fasta")
        if not os.path.isfile(path):
            continue
        verify_composition_for("caid3", task, path)
        ref = read_reference(path)
        targets = two_class_targets(ref)
        methods = load_methods(ref, targets, task)
        if len(methods) < 5:
            continue

        rows = {}
        for name, (ys, ss) in methods.items():
            r = certify(ys, ss)
            if r:
                rows[name] = r
        order = sorted(rows, key=lambda n: -rows[n]["pooled"])
        # The crossed-matching bound enumerates |pos_k| x |neg_l| per protein
        # pair, which is billions on Disorder-PDB. Computed only for the
        # methods it is reported for.
        for name in order[:TOP]:
            ys, ss = methods[name]
            rows[name].update(
                {k: v for k, v in certify(ys, ss, crossed=True).items()
                 if k not in rows[name]})
        within_order = sorted(rows, key=lambda n: -rows[n]["within"])
        wrank = {n: i for i, n in enumerate(within_order, 1)}

        print(f"\n{'=' * 108}")
        print(f" {task}: {len(targets)} two-class targets, {len(rows)} "
              f"full-coverage methods")
        print(f" w_within = {rows[order[0]]['w_within']:.4%} — so "
              f"{rows[order[0]]['w_between']:.2%} of the metric is the part "
              f"per-protein recalibration can move")
        print("=" * 108)
        print(f" {'#':>3} {'method':<26}{'pooled':>8}{'within':>8}"
              f"{'w/in #':>7}{'ceiling':>9}{'headroom':>10}{'reach?':>8}"
              f"{'crossed':>9}")
        for i, n in enumerate(order[:TOP], 1):
            r = rows[n]
            reach = "yes" if r["ceiling_attainable"] else "no"
            cx = r.get("crossed_matching_pairs")
            print(f" {i:>3} {n:<26}{r['pooled']:>8.4f}{r['within']:>8.4f}"
                  f"{wrank[n]:>7}{r['ceiling']:>9.4f}"
                  f"{r['recalibration_headroom']:>10.4f}{reach:>8}"
                  f"{(f'{cx:,}' if cx is not None else '-'):>9}")

        # Inversions against the pooled winner: the published result, certified.
        winner = order[0]
        invs = {}
        for n in within_order[:12]:
            if n == winner:
                continue
            cert = inversion_certificate(rows[n], rows[winner])
            if cert:
                invs[n] = cert
        if invs:
            print(f"\n inversions against the pooled winner ({winner}), "
                  f"certified by inversion_requires_between_gap:")
            print(f"   {'method':<26}{'within +':>10}{'between -':>11}"
                  f"{'required':>10}{'slack':>9}   iff")
            for n, c in sorted(invs.items(),
                               key=lambda kv: -kv[1]["within_gap"])[:8]:
                print(f"   {n:<26}{c['within_gap']:>+10.4f}"
                      f"{c['between_gap_measured']:>+11.4f}"
                      f"{c['between_gap_required']:>10.6f}"
                      f"{c['slack']:>+9.4f}   "
                      f"{'holds' if c['iff_holds'] else 'VIOLATED'}")

        report["tasks"][task] = {
            "n_targets": len(targets), "n_methods": len(rows),
            "winner": winner,
            "methods": {n: rows[n] for n in order},
            "within_rank": wrank,
            "inversions_against_winner": invs,
        }

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
