#!/usr/bin/env python3
"""The capacity bound instantiated outside disorder, on ten ML benchmarks.

`card_le_benchCapacity` is stated for a benchmark whose score is an average over
`n` items, on the grid of denominator `n`, with an annotation wrong on at most a
fraction `eps` of items. Top-1 accuracy on a test set is exactly that: flipping
one label moves accuracy by `1/n`, so a wrong fraction `eps` moves it by at most
`eps`, and two models must differ by more than `2*eps` before the observation
certifies their order. No adaptation of the theorem is needed.

Northcutt, Athalye and Mueller (NeurIPS 2021) measured `eps` for ten of the
benchmarks the field actually uses, by running confident learning over each test
set and having every flagged item checked by human annotators. Their rates are
**lower bounds** -- only algorithmically flagged items were checked -- so the
capacities below are **upper** bounds, which is the conservative direction.

This is the number the field has never computed: given that error rate, how many
models can the benchmark place in a certified order at all?

Two honest caveats, both carried into the output.

* The theorem bounds a *certified* order. It does not say a published ranking is
  wrong; it says the data cannot establish it. If label errors happened to shift
  every model equally the ranking would survive -- but that is an assumption
  about the errors, and Northcutt et al. measured the opposite: correcting the
  labels reorders the leaderboard, and systematically.
* Capacity is computed over the full score range [0,1]. The range anyone
  actually publishes in is far narrower, so the realisable capacity is smaller
  still. Reported as `capacity_observed_range` where a range is known.
"""

from __future__ import annotations

import json
import math
import os

OUT = os.environ.get("BCF_OUT", "")

#: dataset -> (modality, test set size, % label error, published score range)
#: Table 1 of arXiv:2103.14749; rates are the paper's validated estimates.
#: The score range is the span of top-1 accuracies that published models of that
#: benchmark occupy, where a defensible one exists; None where it does not.
NORTHCUTT = {
    "MNIST":          ("image",     10_000,      0.15, (0.985, 0.9987)),
    "CIFAR-10":       ("image",     10_000,      0.54, (0.90, 0.996)),
    "CIFAR-100":      ("image",     10_000,      5.85, (0.65, 0.96)),
    "Caltech-256":    ("image",     29_780,      1.54, None),
    "ImageNet":       ("image",     50_000,      5.83, (0.55, 0.92)),
    "QuickDraw":      ("image", 50_426_266,     10.12, None),
    "20news":         ("text",       7_532,      1.09, None),
    "IMDB":           ("text",      25_000,      2.90, None),
    "Amazon Reviews": ("text",   9_996_437,      3.90, None),
    "AudioSet":       ("audio",     20_371,      1.35, None),
}

#: this paper's own measurement, for comparison on the same axis
OURS = {
    "CAID3 (residue labels)":  ("protein", 99_239, 8.01, None),
    "CAID3 (within-protein pairs)": ("protein", 99_239, 1.00, None),
}


def capacity(eps: float) -> int:
    """`benchCapacity_noise_only`: k <= ceil(1/(2*eps)), free of n."""
    return max(1, math.ceil(1.0 / (2.0 * eps)))


def capacity_over(eps: float, lo: float, hi: float) -> int:
    """The same count restricted to the score range models actually occupy.

    A certified family must be pairwise separated by more than `2*eps`, so on a
    usable range of width `w` at most `floor(w / (2*eps)) + 1` members fit.
    """
    return max(1, int((hi - lo) // (2.0 * eps)) + 1)


def main() -> int:
    rows = []
    for name, (mod, n, pct, rng) in {**NORTHCUTT, **OURS}.items():
        eps = pct / 100.0
        row = {
            "benchmark": name, "modality": mod, "n_items": n,
            "eps_percent": pct, "capacity": capacity(eps),
            "capacity_observed_range": (capacity_over(eps, *rng) if rng
                                        else None),
            "score_range": list(rng) if rng else None,
            "source": ("arXiv:2103.14749 Table 1" if name in NORTHCUTT
                       else "this work"),
        }
        rows.append(row)

    rows.sort(key=lambda r: r["capacity"])
    w = max(len(r["benchmark"]) for r in rows)
    print(f"{'benchmark':<{w}}  {'modality':<8} {'n':>12}  {'eps %':>6}  "
          f"{'capacity':>8}  {'over range':>10}")
    for r in rows:
        cr = r["capacity_observed_range"]
        print(f"{r['benchmark']:<{w}}  {r['modality']:<8} {r['n_items']:>12,}  "
              f"{r['eps_percent']:>6.2f}  {r['capacity']:>8d}  "
              f"{(str(cr) if cr else '-'):>10}")

    print("\nRead this as: at the measured annotation error rate, no analysis of")
    print("the benchmark's own data can place more than `capacity` methods in a")
    print("certified total order -- however many are entered, and whatever")
    print("resampling is applied. The rates are lower bounds, so the capacities")
    print("are upper bounds.")

    if OUT:
        with open(OUT, "w") as fh:
            json.dump({"rows": rows,
                       "note": "eps from arXiv:2103.14749 Table 1 (validated "
                               "estimates, lower bounds); capacity from "
                               "benchCapacity_noise_only"}, fh, indent=1)
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
