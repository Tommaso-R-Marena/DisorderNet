#!/usr/bin/env python3
"""How long is a context-dependent block? The constant in the escape route.

The pairwise discordance rate is 9.32x below the label-flip rate, and the
mechanism is that order/disorder flips in **blocks**: a region resolves in one
crystal form and not another, together. Pairs inside a flipping block keep their
relation; only pairs crossing its boundary reverse.

That predicts the ratio from geometry. For a block of length L inside a chain of
length N, the pairs it can reverse are the ones pairing its interior against the
rest — O(L*(N-L)) of them — while the residues it flips are O(L). Working
through, the discordance-to-label ratio goes as the boundary-to-area ratio of
the flipping set, which for compact blocks on a one-dimensional chain is
O(1/L).

So the block-length distribution is the constant in
`correlated_noise_reduces_pair_discordance`. It is also a biophysical quantity
nobody appears to have measured: **the length scale over which order and
disorder are context-dependent.**

Measured from MobiDB's per-structure missing-residue calls: for each pair of
structures of one protein, the maximal runs of residues on which they disagree.
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

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import read_reference  # noqa: E402

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("BL_REFS", f"{ROOT}/caid3_official")
DISPROT = os.environ.get(
    "BL_DISPROT", os.path.expanduser("~/.cache/disordernet/disprot_raw.json"))
OUT = os.environ.get("BL_OUT", "")
WORKERS = int(os.environ.get("BL_WORKERS", "10"))
MAX_PAIRS = int(os.environ.get("BL_MAX_PAIRS", "40"))


def regions_to_mask(regions, length):
    m = np.zeros(length, dtype=bool)
    for r in regions or []:
        try:
            a, b = int(r[0]), int(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        m[max(a, 1) - 1:min(b, length)] = True
    return m


def runs(mask):
    """Lengths of maximal True runs."""
    out, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def fetch(acc):
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
    per = {}
    for k, v in rec.items():
        if k.startswith("derived-missing_residues-mobi-") and isinstance(v, dict):
            per[k.rsplit("-", 1)[-1]] = regions_to_mask(v.get("regions"), length)
    if len(per) < 2:
        return None

    blocks, used = [], 0
    sids = sorted(per)
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            if used >= MAX_PAIRS:
                break
            a, b = per[sids[i]], per[sids[j]]
            ia, ib = np.nonzero(a)[0], np.nonzero(b)[0]
            if ia.size == 0 or ib.size == 0:
                continue
            lo = max(ia.min(), ib.min())
            hi = min(ia.max(), ib.max())
            if hi - lo < 20:
                continue
            dis = (a[lo:hi + 1] != b[lo:hi + 1])
            if not dis.any():
                used += 1
                continue
            blocks.extend(runs(dis))
            used += 1
        if used >= MAX_PAIRS:
            break
    if not blocks:
        return None
    return {"acc": acc, "length": length, "n_structures": len(sids),
            "n_pairs": used, "blocks": blocks}


def main() -> int:
    acc_of = {}
    if os.path.isfile(DISPROT):
        d = json.load(open(DISPROT))
        for e in (d if isinstance(d, list) else d.get("data", [])):
            if isinstance(e, dict) and e.get("disprot_id") and e.get("acc"):
                acc_of[str(e["disprot_id"])] = e["acc"]
    ref = read_reference(os.path.join(REFS, "disorder_pdb.fasta"))
    accs = sorted({acc_of.get(t, t) for t in ref})

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, r in enumerate(pool.map(fetch, accs), 1):
            if r:
                rows.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(accs)} usable={len(rows)} "
                      f"[{(time.time()-t0)/60:.1f}m]", flush=True)

    allb = np.array([b for r in rows for b in r["blocks"]])
    if allb.size < 100:
        print("too few blocks", file=sys.stderr)
        return 1
    print(f"\n{len(rows)} proteins, {allb.size:,} disagreement blocks")
    print(f"\n{'statistic':<28}{'residues':>10}")
    for name, v in (("mean block length", float(allb.mean())),
                    ("median", float(np.median(allb))),
                    ("75th percentile", float(np.percentile(allb, 75))),
                    ("90th percentile", float(np.percentile(allb, 90))),
                    ("max", float(allb.max()))):
        print(f"{name:<28}{v:>10.1f}")
    # Mass-weighted mean: the length of the block a randomly chosen
    # *disagreeing residue* belongs to. This is the one the boundary-to-area
    # argument needs, since it weights by the residues actually flipped.
    mass = float((allb.astype(float) ** 2).sum() / allb.sum())
    print(f"{'mass-weighted mean':<28}{mass:>10.1f}")
    print(f"\n fraction of blocks of length 1: "
          f"{float((allb == 1).mean()):.1%}")
    print(f" fraction of flipped residues in blocks >= 10: "
          f"{float(allb[allb >= 10].sum() / allb.sum()):.1%}")
    print(f"\n predicted discordance/label ratio ~ 1/L with L the "
          f"mass-weighted mean: 1/{mass:.1f} = {1/mass:.4f}")
    print(f" measured ratio (relative_noise.py): 0.0070/0.0651 = 0.1075")

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump({"n_proteins": len(rows), "n_blocks": int(allb.size),
                       "mean": float(allb.mean()),
                       "median": float(np.median(allb)),
                       "mass_weighted_mean": mass,
                       "p90": float(np.percentile(allb, 90)),
                       "hist": np.bincount(allb, minlength=1)[:200].tolist()},
                      fh, indent=2)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
