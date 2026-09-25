#!/usr/bin/env python3
"""Answer, on its own, the question PREREGISTRATION_11 promised to answer.

The check inside ``train_multitask.py`` compared a set of DisProt ids against a
set of UniProt accessions, so it printed "0 before the filter, 0 surviving" for
every input and could not have failed. The pair 31214999/31215000 was already
training when that was found. Restarting nine hours of GPU time to re-run a
*verification* would be the wrong trade: the leak *prevention* is
``drop_caid_targets``, which is unchanged, regression-tested, and removed 2,039
of 27,412 proteins in that very run. What was missing was the evidence.

So this reproduces the pipeline up to the filter — same inputs, same code path,
no GPU and no training — and reports the two numbers the pre-registration
commits to:

  * how many benchmark accessions carry a soft target *before* the filter
    (expected to be non-zero; the pre-registration says ~141, and a zero here
    means the join is broken again rather than that the cache is clean), and
  * how many survive it (must be 0).

Run it on a compute node, never a login node; it needs BLAST for the homology
pass, exactly as training does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colab.caid3_eval import parse_caid_reference_fasta          # noqa: E402
from rockfish.train_multitask import (                           # noqa: E402
    build_union,
    chunk_long_rows,
    drop_caid_targets,
    load_disprot,
    load_pdb_missing_rows,
    load_soft_labels,
    merge_task_rows,
)

DEFAULT_TASKS = ("disorder_nox", "linker", "binding", "binding_idr")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--disprot", required=True)
    ap.add_argument("--pdb-missing-cache", required=True)
    ap.add_argument("--soft-labels", required=True)
    ap.add_argument("--caid-reference", action="append", required=True)
    ap.add_argument("--leak-identity", type=float, default=0.4)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    entries = load_disprot(args.disprot)
    print(f"DisProt entries: {len(entries):,}")

    tasks = DEFAULT_TASKS
    rows, _cov = build_union(entries, tasks)
    rows = [r for r in rows if r["length"] >= 20]
    print(f"union: {len(rows):,} proteins")

    extra = load_pdb_missing_rows(args.pdb_missing_cache, 10 ** 6, 0,
                                  soft_label_path=args.soft_labels)
    tasks = tasks + ("disorder_pdb",)
    rows = merge_task_rows(rows, extra, tasks)
    print(f"union now {len(rows):,} proteins across {len(tasks)} tasks")

    rows, _stats = chunk_long_rows(rows, args.max_len)
    print(f"after windowing: {len(rows):,} rows")

    soft_accs = set(load_soft_labels(args.soft_labels))

    # The join the original check got wrong: references are DisProt ids, the
    # cache is UniProt accessions.
    dp_to_acc = {}
    for e in entries:
        dp, acc = e.get("disprot_id"), (e.get("acc") or "").strip()
        if dp and acc:
            dp_to_acc[dp] = acc

    ref_accs, ref_seqs, unmapped = set(), set(), set()
    for path in args.caid_reference:
        if not os.path.isfile(path):
            print(f"  missing reference, skipped: {path}")
            continue
        for t in parse_caid_reference_fasta(path):
            ref_seqs.add(t["sequence"])
            acc = dp_to_acc.get(t["id"])
            if acc:
                ref_accs.add(acc)
            elif t["id"].startswith("DP"):
                unmapped.add(t["id"])
            else:
                ref_accs.add(t["id"])

    before = soft_accs & ref_accs
    print(f"\nbenchmark accessions resolved: {len(ref_accs):,} "
          f"({len(unmapped)} DisProt ids unmapped)")
    print(f"of those, carrying a soft target BEFORE the filter: {len(before):,}")
    if not ref_accs:
        print("FAIL: no benchmark accession resolved; the join is broken.")
        return 2
    if not before:
        print("FAIL: zero overlap before the filter. The pre-registration "
              "records ~141; zero means the join is broken, not that the "
              "cache is clean.")
        return 2

    kept, leak = drop_caid_targets(rows, args.caid_reference,
                                   args.leak_identity)
    print(f"drop_caid_targets removed {leak['n_removed']:,} / "
          f"{leak['n_before']:,} ({leak['n_id_overlap']} exact)")

    still_here = {r.get("uniprot_acc") for r in kept
                  if r.get("uniprot_acc") in soft_accs
                  and (r.get("uniprot_acc") in ref_accs
                       or r.get("sequence") in ref_seqs)}
    print(f"surviving the filter: {len(still_here):,}")

    result = {
        "benchmark_accessions_resolved": len(ref_accs),
        "unmapped_reference_ids": sorted(unmapped),
        "soft_targets_on_benchmark_before_filter": len(before),
        "surviving_after_filter": sorted(x for x in still_here if x),
        "removed": leak["n_removed"],
        "n_before": leak["n_before"],
        "exact_hits": leak["n_id_overlap"],
        "pass": not still_here,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out}")

    if still_here:
        print(f"FAIL: {sorted(x for x in still_here if x)[:5]}")
        return 1
    print("\nPASS: no benchmark accession carrying a soft target survives the "
          "filter, and the check was capable of detecting one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
