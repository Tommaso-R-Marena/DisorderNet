#!/usr/bin/env python3
"""CAID's capacity: how many methods it can place in a certified order at all.

`BenchmarkCapacity.card_le_benchCapacity` (Lean 4, sorry-free, standard axioms):
a benchmark with `n` targets, scores averaged into [0,1] and therefore on the
grid of denominator `n`, can place in a certified order at most

    k(n, eps, delta) = n // (floor(c*n) + 1) + 1,    c = max(delta, 2*eps)

methods, **however many are entered**. The factor two on the annotation error
rate is the two-sided price of label noise — the inequality behind
`LabelNoise.ranking_certified` — and `delta` is the smallest difference worth
calling a difference.

`benchCapacity_attained` exhibits a family of exactly `k` methods the benchmark
does resolve, so `k` is the capacity and not an estimate.

`benchCapacity_noise_only`: with `delta = 0`, `k <= ceil(1/(2*eps))` whatever `n`
is. **Collecting more targets does not buy resolution the labels lack.**

The residue-level form `card_le_capacity_labelNoise` gives `R/(2*nu+1)+1` for `R`
scorable residues with `nu` mislabelled, and is computed here as an independent
route to the same number.

And the converse, which is what makes this an impossibility rather than a power
calculation: `unresolvable_pair` *constructs*, for two methods whose measured
scores are close, two truths each within the noise budget of the annotation,
one making each method strictly better. Both are compatible with everything the
benchmark recorded, so the ordering is not a function of the data. No analysis
of those data recovers it; only better labels do.

`eps` here is measured, not assumed: 0.0801, from MobiDB's own per-structure
disagreement (`annotation_noise.py`).
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import (  # noqa: E402
    TASKS,
    evaluated_mask,
    read_reference,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("BC_REFS", f"{ROOT}/caid3_official")
REFS2 = os.environ.get("BC_REFS2", f"{ROOT}/caid2_official")
NOISE = os.environ.get("BC_NOISE", f"{ROOT}/annotation_noise.json")
PREDS = os.environ.get("BC_PREDS", f"{ROOT}/caid3_predictions")
DELTA = float(os.environ.get("BC_DELTA", "0.0"))
OUT = os.environ.get("BC_OUT", "")


def bench_capacity(n: int, eps: float, delta: float = 0.0) -> int:
    """`card_le_benchCapacity`, transcribed. Integer division throughout."""
    c = max(delta, 2.0 * eps)
    if n <= 0:
        return 1
    return n // (math.floor(c * n) + 1) + 1


def capacity_ceiling(eps: float, delta: float = 0.0) -> int:
    """`benchCapacity_le_ceil` / `benchCapacity_noise_only`: the n-free bound."""
    c = max(delta, 2.0 * eps)
    return max(1, math.ceil(1.0 / c)) if c > 0 else 0


def capacity_nat(n_units: int, nu: int) -> int:
    """`card_le_capacity_labelNoise`: R scorable units, nu mislabelled."""
    return n_units // (2 * nu + 1) + 1


def two_class_targets(ref) -> int:
    k = 0
    for _tid, (_seq, lab) in ref.items():
        m = evaluated_mask(lab)
        if not m.any():
            continue
        s = lab.replace("-", "")
        if "0" in s and "1" in s:
            k += 1
    return k


def evaluated_residues(ref) -> int:
    return sum(int(evaluated_mask(lab).sum()) for _s, lab in ref.values())


def n_entrants(task: str) -> int:
    try:
        return sum(1 for f in os.listdir(PREDS) if f.endswith(".caid"))
    except OSError:
        return 0


def main() -> int:
    if not os.path.isfile(NOISE):
        print(f"need {NOISE} — run annotation_noise.py first", file=sys.stderr)
        return 2
    eps = float(json.load(open(NOISE))["epsilon_pooled"])
    print(f"measured annotation error rate  eps = {eps:.4f}")
    print(f"smallest difference worth calling one  delta = {DELTA}")
    print(f"resolution  c = max(delta, 2*eps) = {max(DELTA, 2*eps):.4f}")
    print(f"\ncapacity ceiling, independent of n:  "
          f"k <= ceil(1/(2*eps)) = {capacity_ceiling(eps, DELTA)}")
    print("  (benchCapacity_noise_only — collecting more targets does not buy\n"
          "   resolution the labels lack)")

    report = {"epsilon": eps, "delta": DELTA, "rounds": {}}
    for round_name, refs_dir, tasks in (("CAID3", REFS, TASKS),
                                        ("CAID2", REFS2,
                                         ("disorder_pdb", "disorder_nox",
                                          "binding", "linker"))):
        if not os.path.isdir(refs_dir):
            continue
        print(f"\n{'=' * 88}")
        print(f" {round_name}")
        print(f"{'benchmark':<16}{'targets':>9}{'residues':>11}"
              f"{'k (targets)':>13}{'k (residues)':>14}{'entrants':>10}")
        print("=" * 88)
        rows = {}
        for task in tasks:
            path = os.path.join(refs_dir, f"{task}.fasta")
            if not os.path.isfile(path):
                continue
            ref = read_reference(path)
            n_t = two_class_targets(ref)
            n_r = evaluated_residues(ref)
            k_t = bench_capacity(n_t, eps, DELTA)
            nu = int(round(eps * n_r))
            k_r = capacity_nat(n_r, nu)
            ent = n_entrants(task) if round_name == "CAID3" else 0
            print(f"{task:<16}{n_t:>9,}{n_r:>11,}{k_t:>13}{k_r:>14}"
                  f"{(ent or '—'):>10}")
            rows[task] = {"n_targets": n_t, "n_residues": n_r,
                          "capacity_targets": k_t, "capacity_residues": k_r,
                          "nu": nu, "n_entrants": ent}
        report["rounds"][round_name] = rows

    c3 = report["rounds"].get("CAID3", {})
    if c3:
        worst = min(c3.values(), key=lambda r: r["capacity_targets"])
        ent = max((r["n_entrants"] for r in c3.values()), default=0)
        k = max(r["capacity_targets"] for r in c3.values())
        print(f"\n{'=' * 88}")
        print(f" CAID3 admits {ent} entrants and can place at most {k} of them")
        print(f" in a certified order, at the measured annotation error rate.")
        print("=" * 88)
        print(" Every larger set contains a pair whose ordering is not a")
        print(" function of the data (over_capacity_has_close_pair,")
        print(" unresolvable_pair): two truths exist inside the noise budget,")
        print(" one making each method better, both consistent with everything")
        print(" the benchmark recorded. No analysis of those data decides it.")
        _ = worst

    # What it would take.
    print(f"\n to place {ent or 115} methods in order the annotation error rate")
    print(f" would have to fall to eps <= {1.0/(2.0*(ent or 115)):.5f} "
          f"({100.0/(2.0*(ent or 115)):.3f}%), from the measured "
          f"{100*eps:.2f}%.")
    report["eps_required_for_full_field"] = 1.0 / (2.0 * (ent or 115))

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
