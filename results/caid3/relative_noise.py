#!/usr/bin/env python3
"""Is there a way around the capacity bound? Measure the noise that AUC sees.

`card_le_benchCapacity` bounds capacity by `⌈1/(2ε)⌉` where `ε` is the rate at
which the annotation gets a **label** wrong. At the measured ε = 0.0801 that is
7 methods, and CAID3 has 117.

But AUC is not a function of labels one at a time. It is a function of **ordered
pairs**: the probability a positive outranks a negative. So the noise that
matters for a ranking statistic is not the label-flip rate — it is the rate at
which the annotation **reverses a pair's order**.

Those are different, and the difference is the escape route. Label noise in
protein structure is not independent across residues: a whole region orders in
one crystal form and not another. When a region flips *together*, every pair
**inside** it keeps its relation (both flip, so the pair stays tied or stays
ordered the same way), and only pairs **crossing** the region's boundary
reverse. If the flipping is spatially correlated, the pairwise discordance rate
is far below the label-flip rate — and the capacity of a pairwise statistic is
correspondingly larger.

This measures both rates on the same data: MobiDB's per-structure
missing-residue calls, pairs of structures of the same protein, restricted to
residues both structures cover.

    eps_label     = P(the two structures disagree about one residue)
    eps_pairwise  = P(the two structures order one pair oppositely)

If `eps_pairwise << eps_label`, the within-protein protocol recovers resolution
the residue-level protocol throws away, and the capacity bound for the pairwise
statistic is the one to quote. If they are comparable, the escape route is
closed and the bound stands as computed.

Either answer is worth having. The first is a prescription; the second closes a
line of attack.
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
REFS = os.environ.get("RN_REFS", f"{ROOT}/caid3_official")
DISPROT = os.environ.get(
    "RN_DISPROT", os.path.expanduser("~/.cache/disordernet/disprot_raw.json"))
OUT = os.environ.get("RN_OUT", "")
WORKERS = int(os.environ.get("RN_WORKERS", "8"))
MAX_STRUCT_PAIRS = int(os.environ.get("RN_MAX_PAIRS", "40"))


def regions_to_mask(regions, length: int) -> np.ndarray:
    """MobiDB regions are inclusive 1-based [start, end] pairs."""
    m = np.zeros(length, dtype=bool)
    for r in regions or []:
        try:
            a, b = int(r[0]), int(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        m[max(a, 1) - 1:min(b, length)] = True
    return m


def fetch(acc: str) -> dict | None:
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

    # Per-structure missing calls, and the aggregated observed regions which
    # bound what any structure could have resolved.
    per = {}
    for k, v in rec.items():
        if not k.startswith("derived-missing_residues-mobi-"):
            continue
        if not isinstance(v, dict):
            continue
        per[k.rsplit("-", 1)[-1]] = regions_to_mask(v.get("regions"), length)
    obs = rec.get("derived-observed-th_90")
    observed = (regions_to_mask(obs.get("regions"), length)
                if isinstance(obs, dict) else np.zeros(length, bool))
    if len(per) < 2:
        return None

    # A structure's coverage: the residues it resolved plus those it declared
    # missing. Approximated by the span between its first and last annotated
    # residue intersected with the protein's observed-or-missing footprint,
    # because MobiDB gives per-structure *missing* regions and not per-structure
    # coverage. Residues outside a structure's span are "not determined by that
    # structure", not "observed by it" — conflating those would manufacture
    # disagreement out of absence.
    footprint = observed.copy()
    for m in per.values():
        footprint |= m

    cover = {}
    for sid, miss in per.items():
        idx = np.nonzero(miss)[0]
        span = np.zeros(length, bool)
        if idx.size:
            span[idx.min():idx.max() + 1] = True
        # Extend the span to the contiguous footprint around it.
        cover[sid] = span | (footprint & span)
        cover[sid] = span
    return {"acc": acc, "length": length, "per": per, "cover": cover,
            "footprint": footprint}


def rates(rec) -> dict | None:
    """Label-flip and pairwise-discordance rates over structure pairs."""
    sids = sorted(rec["per"])
    lab_dis = lab_tot = 0
    pair_dis = pair_tot = 0
    used = 0
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            if used >= MAX_STRUCT_PAIRS:
                break
            a, b = rec["per"][sids[i]], rec["per"][sids[j]]
            both = rec["cover"][sids[i]] & rec["cover"][sids[j]]
            n = int(both.sum())
            if n < 20:
                continue
            ya, yb = a[both].astype(np.int8), b[both].astype(np.int8)
            lab_dis += int((ya != yb).sum())
            lab_tot += n
            # Ordered pairs: a pair (p, q) is strictly ordered by a labelling
            # when exactly one of them is called missing. Two labellings are
            # discordant on that pair when both order it and they disagree.
            pa, na = int(ya.sum()), n - int(ya.sum())
            pb, nb = int(yb.sum()), n - int(yb.sum())
            # concordant-with-a-and-b pairs: (i disordered in both, j ordered
            # in both) etc. Count via the 2x2 contingency of (ya, yb).
            n11 = int(((ya == 1) & (yb == 1)).sum())
            n10 = int(((ya == 1) & (yb == 0)).sum())
            n01 = int(((ya == 0) & (yb == 1)).sum())
            n00 = int(((ya == 0) & (yb == 0)).sum())
            # a orders (p,q) as p>q iff ya[p]=1, ya[q]=0. b orders it the other
            # way iff yb[p]=0, yb[q]=1. So discordant pairs are those with
            # p in {ya=1, yb=0} and q in {ya=0, yb=1}, in either direction.
            pair_dis += 2 * n10 * n01
            # total pairs ordered by *both* labellings, either direction
            pair_tot += 2 * (n10 * n01 + n01 * n10) // 2 + \
                (n11 + n10) * (n00 + n01) + (n11 + n01) * (n00 + n10)
            used += 1
        if used >= MAX_STRUCT_PAIRS:
            break
    if lab_tot == 0 or pair_tot == 0:
        return None
    return {"acc": rec["acc"], "n_structures": len(sids),
            "n_structure_pairs": used,
            "eps_label": lab_dis / lab_tot,
            "eps_pairwise": pair_dis / pair_tot,
            "label_residues": lab_tot, "label_disagree": lab_dis,
            "pair_total": pair_tot, "pair_discordant": pair_dis}


def main() -> int:
    acc_of = {}
    if os.path.isfile(DISPROT):
        d = json.load(open(DISPROT))
        for e in (d if isinstance(d, list) else d.get("data", [])):
            if isinstance(e, dict) and e.get("disprot_id") and e.get("acc"):
                acc_of[str(e["disprot_id"])] = e["acc"]
    ref = read_reference(os.path.join(REFS, "disorder_pdb.fasta"))
    accs = sorted({acc_of.get(t, t) for t in ref})
    print(f"{len(accs)} accessions; keeping those with >=2 structures")

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, rec in enumerate(pool.map(fetch, accs), 1):
            if rec:
                r = rates(rec)
                if r:
                    rows.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(accs)}  usable={len(rows)}  "
                      f"[{(time.time()-t0)/60:.1f}m]", flush=True)
    print(f"\n{len(rows)} proteins with >=2 structures and comparable coverage")
    if len(rows) < 20:
        print("too few", file=sys.stderr)
        return 1

    lab = sum(r["label_disagree"] for r in rows) / sum(
        r["label_residues"] for r in rows)
    prs = sum(r["pair_discordant"] for r in rows) / sum(
        r["pair_total"] for r in rows)
    print(f"\n{'':<34}{'pooled':>12}{'median protein':>17}")
    print(f"{'label-flip rate  eps_label':<34}{lab:>12.4f}"
          f"{np.median([r['eps_label'] for r in rows]):>17.4f}")
    print(f"{'pairwise discordance  eps_pair':<34}{prs:>12.4f}"
          f"{np.median([r['eps_pairwise'] for r in rows]):>17.4f}")
    if prs > 0:
        print(f"\nratio eps_label / eps_pairwise = {lab / prs:.2f}")
    import math
    print(f"\ncapacity ceil(1/(2*eps)):")
    print(f"  from label noise    {max(1, math.ceil(1/(2*lab))) if lab else '-':>6}")
    print(f"  from pairwise noise "
          f"{max(1, math.ceil(1/(2*prs))) if prs else '-':>6}")

    report = {"n_proteins": len(rows), "eps_label_pooled": lab,
              "eps_pairwise_pooled": prs,
              "capacity_label": max(1, math.ceil(1 / (2 * lab))) if lab else None,
              "capacity_pairwise": max(1, math.ceil(1 / (2 * prs))) if prs else None,
              "proteins": rows}
    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
