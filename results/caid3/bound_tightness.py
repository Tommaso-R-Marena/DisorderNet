#!/usr/bin/env python3
"""Does the capacity bound bind, or is it loose? The measurable criterion.

`card_le_benchCapacity` is a **worst-case** statement: it assumes the annotation
errors can fall wherever they most damage the comparison. That is what makes the
converse an impossibility rather than a power calculation, and it is also the
bound's one soft spot. If a benchmark's errors were placed independently of the
methods being compared, they would move both scores by the same amount and
cancel, and the realisable resolution would be far better than the worst case.

So the honest question is not "does the theorem apply" -- it applies to any
benchmark whose score is an average over items -- but "is it tight here". That
is measurable, and this measures it.

A label flip at residue `i` can change the comparison of methods A and B only
when A and B call `i` differently. If they agree, the flip moves both scores
identically and cancels exactly. So the error budget that actually threatens the
A-vs-B comparison is not the total number of ambiguous residues but the number
of ambiguous residues **on which A and B disagree**:

    nu_eff(A, B) = |{ambiguous} & {A != B}|,       nu = |{ambiguous}|

and the ratio to the overall disagreement rate is the diagnostic:

    enrichment = P(A != B | ambiguous) / P(A != B)

  ~1   errors fall independently of the comparison. The worst case is
       unreachable, the bound is loose by roughly 1/P(A != B), and a benchmark
       may reasonably quote a larger effective capacity.
  >>1  ambiguous residues are exactly the residues methods disagree about.
       The error budget sits where the comparison is decided, the worst case is
       close to reachable, and the bound binds as stated.

Ambiguity comes from crystallography (MobiDB's per-structure missing-residue
calls) and disagreement comes from the submitted predictions, so the two are
measured from independent sources -- unlike a confident-learning flag, which is
defined by model disagreement and would make this circular.
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

from colab.caid3_official import (  # noqa: E402
    evaluated_mask,
    read_caid_predictions,
    read_reference,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("BT_REFS", f"{ROOT}/caid3_official")
PREDS = os.environ.get("BT_PREDS", f"{ROOT}/caid3_predictions")
DISPROT = os.environ.get(
    "BT_DISPROT", os.path.expanduser("~/.cache/disordernet/disprot_raw.json"))
TASK = os.environ.get("BT_TASK", "disorder_pdb")
WORKERS = int(os.environ.get("BT_WORKERS", "8"))
OUT = os.environ.get("BT_OUT", "")


def regions_to_mask(regions, length):
    m = np.zeros(length, dtype=bool)
    for r in regions or []:
        try:
            a, b = int(r[0]), int(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        m[max(a, 1) - 1:min(b, length)] = True
    return m


def fetch_ambiguity(acc):
    """Per-residue mask of context-dependent residues, from MobiDB."""
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
    cd = rec.get("derived-missing_residues_context_dependent-th_90")
    if not isinstance(cd, dict):
        return None
    mask = regions_to_mask(cd.get("regions"), length)
    return None if not mask.any() else (acc, mask)


def main() -> int:
    ref = read_reference(os.path.join(REFS, f"{TASK}.fasta"))

    acc_of = {}
    if os.path.isfile(DISPROT):
        d = json.load(open(DISPROT))
        for e in (d if isinstance(d, list) else d.get("data", [])):
            if isinstance(e, dict) and e.get("disprot_id") and e.get("acc"):
                acc_of[str(e["disprot_id"])] = e["acc"]

    targets = sorted(ref)
    accs = {t: acc_of.get(t, t) for t in targets}
    print(f"fetching context-dependence for {len(set(accs.values()))} "
          f"accessions", flush=True)
    amb = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for got in pool.map(fetch_ambiguity, sorted(set(accs.values()))):
            if got:
                amb[got[0]] = got[1]
    print(f"{len(amb)} with a context-dependent record", flush=True)

    # binary calls: column four of each .caid file, the method's own threshold
    calls = {}
    for fn in sorted(os.listdir(PREDS)):
        if not fn.endswith(".caid"):
            continue
        name = fn[:-5]
        try:
            p = read_caid_predictions(os.path.join(PREDS, fn))
        except Exception:
            continue
        if isinstance(p, dict) and len(p) >= 0.9 * len(ref):
            calls[name] = p
    print(f"{len(calls)} methods with usable calls", flush=True)

    # assemble aligned per-residue arrays over the evaluated, ambiguity-known set
    cols, ambig = {}, []
    order = []
    for t in targets:
        a = accs[t]
        if a not in amb:
            continue
        seq, lab = ref[t]
        m = evaluated_mask(lab)
        mask = amb[a]
        n = min(len(lab), mask.size)
        keep = m[:n]
        if not keep.any():
            continue
        order.append((t, n, keep))
        ambig.append(mask[:n][keep])
    if not order:
        print("no overlap between the reference and MobiDB", file=sys.stderr)
        return 1
    ambig = np.concatenate(ambig)

    for name, pred in calls.items():
        chunks, ok = [], True
        for t, n, keep in order:
            v = pred.get(t)
            if v is None or len(v) < n:
                ok = False
                break
            chunks.append(np.asarray(v[:n], dtype=float)[keep])
        if ok:
            arr = np.concatenate(chunks)
            if np.isfinite(arr).all():
                # Binarise at the method's own median over the evaluated set,
                # so every method calls the same number of residues disordered.
                # That removes calibration from the comparison and leaves the
                # discrimination question -- *which* residues -- which is what
                # a within-protein statistic turns on. Using each method's own
                # published threshold instead would confound disagreement about
                # residues with disagreement about how much disorder there is.
                cols[name] = arr > np.median(arr)
    print(f"{len(cols)} methods aligned over {ambig.size:,} evaluated residues; "
          f"{int(ambig.sum()):,} are context-dependent "
          f"({ambig.mean():.2%})\n", flush=True)
    if len(cols) < 5:
        print("too few methods", file=sys.stderr)
        return 1

    names = sorted(cols)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = cols[names[i]], cols[names[j]]
            dis = a != b
            if not dis.any():
                continue
            p_dis = dis.mean()
            p_dis_given_amb = dis[ambig].mean() if ambig.any() else 0.0
            rows.append({
                "a": names[i], "b": names[j],
                "p_disagree": float(p_dis),
                "p_disagree_given_ambiguous": float(p_dis_given_amb),
                "enrichment": float(p_dis_given_amb / p_dis) if p_dis else 0.0,
                "nu_eff_over_nu": float(p_dis_given_amb),
            })

    enr = np.array([r["enrichment"] for r in rows])
    eff = np.array([r["nu_eff_over_nu"] for r in rows])
    pdis = np.array([r["p_disagree"] for r in rows])
    print(f"{len(rows):,} method pairs\n")
    print(f"{'':34}{'median':>10}{'mean':>10}{'p10':>10}{'p90':>10}")
    for label, arr in (("P(disagree)", pdis),
                       ("P(disagree | ambiguous)", eff),
                       ("enrichment", enr)):
        print(f"{label:<34}{np.median(arr):>10.3f}{arr.mean():>10.3f}"
              f"{np.percentile(arr, 10):>10.3f}{np.percentile(arr, 90):>10.3f}")

    print(f"\nfraction of pairs with enrichment > 1: "
          f"{(enr > 1).mean():.1%}")
    print(f"fraction with enrichment > 2: {(enr > 2).mean():.1%}")
    print("\nReading: enrichment near 1 would mean the annotation errors fall")
    print("independently of what the comparison turns on, and the worst-case")
    print("bound would be loose by about 1/P(disagree). Enrichment well above 1")
    print("means the ambiguous residues are the residues the methods disagree")
    print("about, so the error budget sits where the comparison is decided and")
    print("the bound binds close to as stated.")

    if OUT:
        json.dump({"task": TASK, "n_residues": int(ambig.size),
                   "n_ambiguous": int(ambig.sum()),
                   "ambiguity_rate": float(ambig.mean()),
                   "n_methods": len(names), "n_pairs": len(rows),
                   "median_p_disagree": float(np.median(pdis)),
                   "median_p_disagree_given_ambiguous": float(np.median(eff)),
                   "median_enrichment": float(np.median(enr)),
                   "frac_enrichment_gt_1": float((enr > 1).mean()),
                   "frac_enrichment_gt_2": float((enr > 2).mean()),
                   "pairs": rows}, open(OUT, "w"), indent=1)
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
