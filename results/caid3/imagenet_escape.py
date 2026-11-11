#!/usr/bin/env python3
"""The escape route, measured on ImageNet — not on proteins.

The capacity bound applies to top-1 accuracy verbatim, and at the validated
label-error rates it leaves ImageNet able to order nine models. The paper's
second result says the escape depends on the *statistic*: an average of per-item
losses has no pairing structure and no escape, while a ranking statistic over
within-group pairs is corrupted only when both members of a pair are
mis-annotated in opposite directions.

That prediction is testable on ImageNet itself, because Northcutt, Athalye and
Mueller released, for every algorithmically flagged item, the given label, the
proposed correction, and the votes of the human annotators who adjudicated it
(arXiv:2103.14749). That is exactly the `(T, L)` pair the discordance identity
needs: `L` the annotation the benchmark ships, `T` the annotation after
correction.

The alternative statistic considered here is the one the theorem points at:
**per-class one-vs-rest AUC, averaged over classes.** Each class is a group;
within it the metric depends only on the ranking of images by their score for
that class. It is a standard metric, it measures the same ability as top-1
accuracy, and it has the pairing structure that top-1 accuracy lacks.

Adjudication rule
-----------------
An item counts as a corrected error when the plurality of annotators chose the
proposed label. Items whose plurality was *neither* or *both* are label errors
with no determinate replacement; they are counted in the label rate and, in the
per-class computation, leave their given class without joining another. That is
the conservative direction for the pairwise rate, and both variants are printed.

What this does and does not show
--------------------------------
It shows that the same test set, re-scored on a statistic with the pairing
structure, has a capacity orders of magnitude larger. It does **not** show that
a benchmark should chase capacity for its own sake: a constant predictor has
infinite capacity and no value. Capacity is necessary, not sufficient. The claim
is narrower and defensible — *among statistics measuring the same ability,
prefer the one whose noise is second-order* — and the point of computing it here
is that on ImageNet the difference is not marginal.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict

SCRATCH = os.environ.get(
    "IE_DIR",
    "/tmp/claude-1000/-home-tommaso-marena-DisorderNet/"
    "c91da682-61d6-4883-a0ab-c3b8dd33dbd5/scratchpad")
OUT = os.environ.get("IE_OUT", "")

#: dataset -> (mturk file, n_classes, test-set size). Class counts are the
#: documented balanced designs of these test sets.
SETS = {
    "ImageNet":   ("imagenet_mturk.json",   1000, 50_000),
    "CIFAR-100":  ("cifar100_mturk.json",    100, 10_000),
    "CIFAR-10":   ("cifar10_mturk.json",      10, 10_000),
    "MNIST":      ("mnist_mturk.json",        10, 10_000),
}


def verdict(rec) -> str:
    """Plurality of the human adjudicators, ties resolved to `given`."""
    m = rec["mturk"]
    order = ("given", "guessed", "neither", "both")
    best = max(order, key=lambda k: (m.get(k, 0), k == "given"))
    return best


def analyse(name, path, n_classes, n_items, drop_indeterminate):
    recs = json.load(open(path))
    #: images leaving class c (annotated c, truth is not c)
    out = defaultdict(int)
    #: images arriving in class c (truth c, annotated something else)
    into = defaultdict(int)
    n_err = n_indet = 0
    for r in recs:
        v = verdict(r)
        if v == "given":
            continue
        n_err += 1
        g = r["given_original_label"]
        if v == "guessed":
            out[g] += 1
            into[r["our_guessed_label"]] += 1
        else:
            n_indet += 1
            if not drop_indeterminate:
                out[g] += 1

    per_class = n_items // n_classes
    disc = comp = 0
    for c in range(n_classes):
        u = out[c]                       # |L_c \ T_c|
        d = into[c]                      # |T_c \ L_c|
        a = per_class - u                # |T_c n L_c|
        e = n_items - (per_class + d)    # |(T_c u L_c)^c|
        if a <= 0 or e <= 0:
            continue
        disc += 2 * d * u
        comp += 2 * (a * e + d * u)

    eps_label = n_err / n_items
    eps_pair = disc / comp if comp else 0.0
    cap = lambda e: max(1, math.ceil(1.0 / (2.0 * e))) if e > 0 else None
    return {
        "benchmark": name, "n_items": n_items, "n_classes": n_classes,
        "n_flagged": len(recs), "n_validated_errors": n_err,
        "n_indeterminate": n_indet,
        "eps_label": eps_label, "eps_pair": eps_pair,
        "capacity_accuracy": cap(eps_label),
        "capacity_per_class_auc": cap(eps_pair) if eps_pair else float("inf"),
        "ratio": (eps_label / eps_pair) if eps_pair else None,
        "discordant_pairs": disc, "comparable_pairs": comp,
        "indeterminate_dropped": drop_indeterminate,
    }


def main() -> int:
    rows = []
    for drop in (True, False):
        for name, (fn, k, n) in SETS.items():
            p = os.path.join(SCRATCH, fn)
            if not os.path.isfile(p):
                continue
            rows.append(analyse(name, p, k, n, drop))

    hdr = (f"{'benchmark':<11}{'items':>8}{'cls':>6}{'errors':>8}"
           f"{'eps_label':>11}{'eps_pair':>12}{'cap(acc)':>10}"
           f"{'cap(AUC)':>11}{'ratio':>10}")
    for drop in (True, False):
        tag = ("indeterminate errors dropped" if drop
               else "indeterminate errors kept as leaving their class")
        print(f"\n=== {tag}\n{hdr}")
        for r in rows:
            if r["indeterminate_dropped"] != drop:
                continue
            print(f"{r['benchmark']:<11}{r['n_items']:>8,}{r['n_classes']:>6}"
                  f"{r['n_validated_errors']:>8,}{r['eps_label']:>11.4f}"
                  f"{r['eps_pair']:>12.2e}{r['capacity_accuracy']:>10,}"
                  f"{r['capacity_per_class_auc']:>11,}"
                  f"{(r['ratio'] or 0):>10,.0f}")

    print("\nThe same test set, the same labels, the same errors. What changes")
    print("is the statistic: top-1 accuracy averages per-item losses and has no")
    print("pairing structure, so the whole error budget can be placed where it")
    print("decides comparisons. Per-class one-vs-rest AUC ranks within a class,")
    print("and a pair reverses only when both its members flip in opposite")
    print("directions -- an event that is second-order and, on a benchmark with")
    print("many negatives per class, rare.")
    print("\nCapacity is necessary, not sufficient: a constant predictor has")
    print("infinite capacity and no value. The claim is that among statistics")
    print("measuring the same ability, the one with the pairing structure can")
    print("resolve a field the other cannot.")

    if OUT:
        json.dump({"rows": rows,
                   "source": "arXiv:2103.14749 mturk adjudications",
                   "rule": "plurality of human adjudicators, ties to `given`"},
                  open(OUT, "w"), indent=1)
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
