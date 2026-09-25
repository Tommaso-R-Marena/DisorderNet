#!/usr/bin/env python3
"""Build the soft disorder target from MobiDB's per-structure missing calls.

Registered in `results/caid3/PREREGISTRATION_11.md` before this file existed.

`CAPACITY.md` measures that 8.01% of evidenced residues are context-dependent:
missing in some deposited structures of a protein and observed in others. The
training set records each of those as a hard 0 or 1 depending on which structure
was consulted. The fraction is published per structure, so it can go into the
target instead of into the noise:

    soft(residue) = (structures calling it missing) / (structures covering it)

Residues covered by fewer than two structures, or with no MobiDB record, keep
their existing hard label. That is the whole change: no new proteins, no change
to the loss (`binary_cross_entropy_with_logits` already accepts a float target),
no change to the folds or the validation holdout.

Coverage is approximated by the span of each structure's annotated region, the
same convention `relative_noise.py` uses to measure eps -- MobiDB publishes
per-structure *missing* regions and not per-structure coverage, and treating
residues outside a structure's span as undetermined rather than observed is the
conservative direction, since counting them as observed would manufacture
certainty from absence.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

OUT = os.environ.get("SL_OUT", "")
WORKERS = int(os.environ.get("SL_WORKERS", "8"))
ACCS = os.environ.get("SL_ACCS", "")           # newline-separated accessions
MIN_STRUCTURES = int(os.environ.get("SL_MIN_STRUCTURES", "2"))


def regions_to_mask(regions, length: int) -> np.ndarray:
    m = np.zeros(length, dtype=bool)
    for r in regions or []:
        try:
            a, b = int(r[0]), int(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        m[max(a, 1) - 1:min(b, length)] = True
    return m


def soft_for(acc: str):
    """Per-residue (soft target, n covering structures) for one accession."""
    url = f"https://mobidb.org/api/download?acc={acc}&format=json"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as fh:
                rec = json.load(fh)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == 3:
                return None
        except Exception:
            if attempt == 3:
                return None
        time.sleep(1.5 * (attempt + 1))
    else:
        return None
    if isinstance(rec, list):
        rec = rec[0] if rec else {}
    if not isinstance(rec, dict):
        return None
    try:
        length = int(rec.get("length") or 0)
    except (TypeError, ValueError):
        return None
    if length <= 0:
        return None

    missing_count = np.zeros(length, dtype=np.int32)
    cover_count = np.zeros(length, dtype=np.int32)
    n_struct = 0
    for k, v in rec.items():
        if not k.startswith("derived-missing_residues-mobi-"):
            continue
        if not isinstance(v, dict):
            continue
        miss = regions_to_mask(v.get("regions"), length)
        idx = np.nonzero(miss)[0]
        span = np.zeros(length, dtype=bool)
        if idx.size:
            span[idx.min():idx.max() + 1] = True
        else:
            continue
        cover_count += span
        missing_count += (miss & span)
        n_struct += 1
    if n_struct < MIN_STRUCTURES or not (cover_count >= MIN_STRUCTURES).any():
        return None
    usable = cover_count >= MIN_STRUCTURES
    soft = np.zeros(length, dtype=np.float32)
    soft[usable] = missing_count[usable] / cover_count[usable]
    return acc, soft, usable, n_struct


def main() -> int:
    if ACCS and os.path.isfile(ACCS):
        accs = [l.strip() for l in open(ACCS) if l.strip()]
    else:
        print("set SL_ACCS to a file of accessions", file=sys.stderr)
        return 2
    accs = sorted(set(accs))
    print(f"{len(accs)} accessions, {WORKERS} workers", flush=True)

    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, got in enumerate(pool.map(soft_for, accs), 1):
            if got:
                acc, soft, usable, n = got
                out[acc] = {"soft": soft[usable].tolist(),
                            "index": np.nonzero(usable)[0].tolist(),
                            "length": int(soft.size), "n_structures": n}
            if i % 500 == 0:
                print(f"  {i}/{len(accs)}  usable={len(out)}  "
                      f"[{(time.time()-t0)/60:.1f}m]", flush=True)

    if not out:
        print("nothing usable", file=sys.stderr)
        return 1

    n_res = sum(len(v["soft"]) for v in out.values())
    vals = np.concatenate([np.asarray(v["soft"]) for v in out.values()])
    intermediate = ((vals > 0.0) & (vals < 1.0))
    print(f"\n{len(out):,} proteins, {n_res:,} residues with "
          f">= {MIN_STRUCTURES} covering structures")
    print(f"  hard 0            {(vals == 0).mean():>7.2%}")
    print(f"  hard 1            {(vals == 1).mean():>7.2%}")
    print(f"  intermediate      {intermediate.mean():>7.2%}  "
          f"<- the residues this run exists to treat differently")
    if intermediate.any():
        print(f"  their mean value  {vals[intermediate].mean():>7.3f}")

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump({"min_structures": MIN_STRUCTURES,
                       "n_proteins": len(out), "n_residues": n_res,
                       "intermediate_fraction": float(intermediate.mean()),
                       "proteins": out}, fh)
        os.replace(OUT + ".part", OUT)
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
