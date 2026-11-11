#!/usr/bin/env python3
"""Precompute the SASA features for every cached AlphaFold structure.

Shrake-Rupley in Biopython is pure Python, and ``attach_structure`` calls it
once per protein on every run. Measured on the private-trunk run: **5h04m of an
18h job** spent recomputing accessibilities that do not depend on the model, the
seed, the task, or anything else that changed between runs. Over the ten
multi-task runs in this project that is roughly two CPU-days recomputing the
same numbers.

This fills the cache once, in parallel, on CPU nodes — where the work belongs,
since it never touches the GPU that a training job is holding idle while it
runs. Afterwards ``structure_features`` is a disk read.

The cache carries each mmCIF's content hash, so this is a pure speed change: a
structure that is replaced (AlphaFold DB moved v4 -> v6 mid-project) recomputes
on its own, and a cache built from stale files cannot quietly outlive them.

    sbatch rockfish/slurm/build_rsa_cache.sbatch

Idempotent and interruptible. Re-running skips what is already valid, so a job
that hits its wall clock is resumed by submitting it again.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colab.structure_rsa import (  # noqa: E402
    RSA_CACHE_DIRNAME,
    rsa_from_structure_cached,
)

#: What one parse returns, in order. Pinned so a channel added to the parse
#: without updating its consumers fails on the first structure with a message
#: naming the arity, instead of on all 21,878 with "too many values to unpack".
CHANNEL_NAMES = ("rsa", "seq", "plddt", "contacts", "ca_torsion")
EXPECTED_CHANNELS = len(CHANNEL_NAMES)


def _one(args: tuple[str, str]) -> tuple[str, int, str]:
    """Returns (accession, n_residues, error). Never raises: one unparseable
    structure out of 21,879 must not take the other 21,878 with it."""
    path, feature_cache = args
    acc = os.path.splitext(os.path.basename(path))[0]
    try:
        # Unpacked by width, not by name, so an added channel is caught here
        # rather than 21,878 times in a row: adding the CA torsion turned this
        # into a five-tuple and every structure failed with "too many values
        # to unpack" while the tests, which index, passed.
        out = rsa_from_structure_cached(path, feature_cache)
        if len(out) != EXPECTED_CHANNELS:
            return acc, 0, (f"parse returned {len(out)} values, expected "
                            f"{EXPECTED_CHANNELS}")
        rsa, seq = out[0], out[1]
        for i, name in enumerate(CHANNEL_NAMES):
            if i < 2:
                continue
            if len(out[i]) != len(rsa):
                return acc, 0, (f"{name} has {len(out[i])} values against "
                                f"{len(rsa)} residues")
        if len(rsa) != len(seq):
            return acc, 0, f"length mismatch rsa={len(rsa)} seq={len(seq)}"
        return acc, len(rsa), ""
    except Exception as exc:                      # noqa: BLE001 - reported
        return acc, 0, f"{type(exc).__name__}: {exc}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structures", required=True,
                    help="directory of AlphaFold .cif files")
    ap.add_argument("--feature-cache", default=None,
                    help=f"output directory (default: "
                         f"<structures>/{RSA_CACHE_DIRNAME})")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many structures (a smoke test)")
    args = ap.parse_args(argv)

    cache = args.feature_cache or os.path.join(args.structures,
                                               RSA_CACHE_DIRNAME)
    os.makedirs(cache, exist_ok=True)

    cifs = sorted(os.path.join(args.structures, f)
                  for f in os.listdir(args.structures) if f.endswith(".cif"))
    if args.limit:
        cifs = cifs[:args.limit]
    if not cifs:
        print(f"ERROR: no .cif files under {args.structures}", file=sys.stderr)
        return 2
    print(f"{len(cifs):,} structures -> {cache}")
    print(f"{args.workers} workers")

    t0 = time.time()
    done = failed = 0
    residues = 0
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one, (p, cache)) for p in cifs]
        for fut in as_completed(futures):
            acc, n, err = fut.result()
            done += 1
            if err:
                failed += 1
                if len(errors) < 20:
                    errors.append(f"{acc}: {err}")
            else:
                residues += n
            if done % 1000 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                left = (len(cifs) - done) / max(rate, 1e-9)
                print(f"  {done:,}/{len(cifs):,}  {rate:.1f}/s  "
                      f"eta {left/60:.0f}m  failed={failed}", flush=True)

    dt = time.time() - t0
    print(f"\n{done:,} structures, {failed:,} failed, "
          f"{residues:,} residues in {dt/60:.1f} min "
          f"({done/max(dt,1e-9):.1f}/s on {args.workers} workers)")
    if errors:
        print("\nfirst failures:")
        for e in errors:
            print(f"  {e}")
    # A failure rate this side of a few percent is normal (obsolete entries,
    # truncated downloads). A wholesale failure means the parse is broken, and
    # returning 0 would let a training run proceed with an empty cache and
    # silently pay the full cost again.
    if done and failed / done > 0.10:
        print(f"\nERROR: {failed/done:.1%} of structures failed to parse — "
              f"that is not attrition, that is a broken parse.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
