"""Command line for the capacity results and the pairwise protocol.

    disordernet capacity --methods 117 --eps 0.0801
    disordernet noise --truth truth.csv --annotation annot.csv
    disordernet rank --reference ref.fasta --predictions preds/ --eps-pair 0.01
    disordernet table
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _capacity(a) -> int:
    from .capacity import assess, capacity_over_range
    v = assess(a.methods, a.eps, delta=a.delta, n_items=a.items,
               kappa=a.kappa, eps_pair=a.eps_pair)
    if a.json:
        print(json.dumps(v.__dict__, indent=1))
        return 0
    print(v)
    if a.range:
        lo, hi = a.range
        print(f"  over the published range [{lo}, {hi}]  "
              f"{capacity_over_range(a.eps, lo, hi):>10,}")
    return 0


def _noise(a) -> int:
    import numpy as np
    from .noise import rates
    t = np.loadtxt(a.truth, delimiter=",").ravel()
    l = np.loadtxt(a.annotation, delimiter=",").ravel()
    r = rates(t, l)
    if a.json:
        print(json.dumps(r.__dict__ | {"ratio": r.ratio, "kappa": r.kappa},
                         indent=1))
        return 0
    print(r)
    from .capacity import capacity, pairwise_capacity
    print(f"capacity as scored {capacity(r.eps_label)}, "
          f"scored on pairs {pairwise_capacity(r.eps_label, eps_pair=r.eps_pair)}")
    return 0


def _rank(a) -> int:
    import numpy as np
    from .protocol import rank

    def read(path):
        out = {}
        name = None
        with open(path) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    name = line[1:].split()[0]
                elif name and line.strip():
                    out[name] = np.array(
                        [float(x) for x in line.replace(",", " ").split()])
                    name = None
        return out

    import os
    reference = {k: v.astype(int) for k, v in read(a.reference).items()}
    preds = {}
    for fn in sorted(os.listdir(a.predictions)):
        if fn.endswith((".txt", ".caid", ".tsv")):
            preds[os.path.splitext(fn)[0]] = read(
                os.path.join(a.predictions, fn))
    lb = rank(reference, preds, eps_pair=a.eps_pair, alpha=a.alpha)
    print(lb if not a.json else json.dumps(lb.rows, indent=1))
    return 0


#: published, validated label-error rates (arXiv:2103.14749 Table 1) plus ours
PUBLISHED = [
    ("QuickDraw", "image", 50_426_266, 0.1012),
    ("CAID3 Disorder-PDB", "protein", 99_239, 0.0801),
    ("CIFAR-100", "image", 10_000, 0.0585),
    ("ImageNet", "image", 50_000, 0.0583),
    ("Amazon Reviews", "text", 9_996_437, 0.0390),
    ("IMDB", "text", 25_000, 0.0290),
    ("Caltech-256", "image", 29_780, 0.0154),
    ("AudioSet", "audio", 20_371, 0.0135),
    ("20news", "text", 7_532, 0.0109),
    ("CIFAR-10", "image", 10_000, 0.0054),
    ("MNIST", "image", 10_000, 0.0015),
]


def _table(a) -> int:
    from .capacity import capacity
    rows = [(n, m, sz, e, capacity(e)) for n, m, sz, e in PUBLISHED]
    if a.json:
        print(json.dumps([{"benchmark": n, "modality": m, "n_items": s,
                           "eps": e, "capacity": c} for n, m, s, e, c in rows],
                         indent=1))
        return 0
    w = max(len(r[0]) for r in rows)
    print(f"{'benchmark':<{w}}  {'modality':<8}{'items':>12}{'eps':>9}"
          f"{'capacity':>10}")
    for n, m, s, e, c in sorted(rows, key=lambda r: r[4]):
        print(f"{n:<{w}}  {m:<8}{s:>12,}{e:>9.4f}{c:>10,}")
    print("\nRates are lower bounds, so capacities are upper bounds.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="disordernet",
        description="What a benchmark can resolve, and the protocol that "
                    "resolves more.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capacity", help="methods a benchmark can order")
    c.add_argument("--methods", type=int, required=True)
    c.add_argument("--eps", type=float, required=True,
                   help="annotation error rate, e.g. 0.0583 for ImageNet")
    c.add_argument("--delta", type=float, default=0.0)
    c.add_argument("--items", type=int, default=None)
    c.add_argument("--kappa", type=float, default=1.0)
    c.add_argument("--eps-pair", type=float, default=None)
    c.add_argument("--range", type=float, nargs=2, metavar=("LO", "HI"))
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=_capacity)

    nz = sub.add_parser("noise", help="both rates from repeat annotations")
    nz.add_argument("--truth", required=True)
    nz.add_argument("--annotation", required=True)
    nz.add_argument("--json", action="store_true")
    nz.set_defaults(fn=_noise)

    r = sub.add_parser("rank", help="score a field under the pairwise protocol")
    r.add_argument("--reference", required=True)
    r.add_argument("--predictions", required=True)
    r.add_argument("--eps-pair", type=float, default=None)
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=_rank)

    t = sub.add_parser("table", help="published benchmarks and their capacities")
    t.add_argument("--json", action="store_true")
    t.set_defaults(fn=_table)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
