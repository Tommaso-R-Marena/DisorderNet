#!/usr/bin/env python3
"""Measure the annotation error rate of CAID3 Disorder-PDB, from MobiDB.

`LabelNoise.ranking_certified` turns a margin into a statement about which
method is better *in truth*, given the annotation error rate. This project has
lacked that rate. The first attempt — comparing CAID3's own references — returned
exactly zero disagreements, because Disorder-NOX is derived from the same
assignment and they agree by construction.

MobiDB supplies the real quantity. Alongside the aggregated
`derived-missing_residues-th_90` that CAID3's Disorder-PDB labels follow, it
publishes:

- `derived-missing_residues-mobi-{PDBID}_{CHAIN}` — the missing-residue call of
  each **individual deposited structure**, hundreds per protein for well-studied
  targets (463 for p53);
- `derived-missing_residues_context_dependent-th_90` — the residues that are
  **missing in some structures and observed in others**.

The second is the disagreement set: residues where independent experimental
determinations of the same protein contradict each other about the label the
benchmark scores.

## What this rate is, and is not

It is **not** measurement error in the usual sense. A residue missing in one
crystal and resolved in another is often genuine context dependence — a region
that orders on binding a partner, or in one crystal form and not another. The
distinction does not matter for `ranking_certified` and it makes the finding
stronger rather than weaker: **the label is not a function of the sequence.** A
predictor asked for one number per residue is being scored against a reference
that itself depends on which structure was consulted.

Reported as a distribution over proteins, not a single number, because a
capacity claim built on a mean would hide that the disagreement is concentrated.
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
REFS = os.environ.get("AN_REFS", f"{ROOT}/caid3_official")
DISPROT = os.environ.get(
    "AN_DISPROT", os.path.expanduser("~/.cache/disordernet/disprot_raw.json"))
OUT = os.environ.get("AN_OUT", "")
WORKERS = int(os.environ.get("AN_WORKERS", "8"))

MISSING = "derived-missing_residues-th_90"
CONTEXT = "derived-missing_residues_context_dependent-th_90"
OBSERVED = "derived-observed-th_90"


def _count(rec: dict, key: str) -> int | None:
    v = rec.get(key)
    if not isinstance(v, dict):
        return None
    c = v.get("content_count")
    return int(c) if c is not None else None


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

    n_struct = sum(1 for k in rec
                   if k.startswith("derived-missing_residues-mobi-"))
    missing = _count(rec, MISSING)
    context = _count(rec, CONTEXT)
    observed = _count(rec, OBSERVED)
    if missing is None or observed is None:
        return None
    evidenced = missing + observed
    if evidenced <= 0:
        return None
    return {
        "acc": acc, "length": rec.get("length"),
        "n_structures": n_struct,
        "n_missing": missing, "n_observed": observed,
        "n_context_dependent": context or 0,
        "n_evidenced": evidenced,
        # The disagreement rate over residues the benchmark actually scores.
        "epsilon": (context or 0) / evidenced,
    }


def main() -> int:
    acc_of = {}
    if os.path.isfile(DISPROT):
        d = json.load(open(DISPROT))
        for e in (d if isinstance(d, list) else d.get("data", [])):
            if isinstance(e, dict) and e.get("disprot_id") and e.get("acc"):
                acc_of[str(e["disprot_id"])] = e["acc"]

    ref = read_reference(os.path.join(REFS, "disorder_pdb.fasta"))
    accs = sorted({acc_of.get(t, t) for t in ref})
    print(f"{len(ref)} Disorder-PDB targets -> {len(accs)} accessions")

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, r in enumerate(pool.map(fetch, accs), 1):
            if r:
                rows.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(accs)}  usable={len(rows)}  "
                      f"[{(time.time()-t0)/60:.1f}m]", flush=True)
    print(f"\n{len(rows)} accessions with a usable MobiDB record")
    if len(rows) < 30:
        print("too few to report a rate", file=sys.stderr)
        return 1

    multi = [r for r in rows if r["n_structures"] >= 2]
    eps_all = np.array([r["epsilon"] for r in rows])
    eps_multi = np.array([r["epsilon"] for r in multi])

    print(f"\n{'':<34}{'all':>12}{'>=2 structures':>18}")
    print(f"{'proteins':<34}{len(rows):>12}{len(multi):>18}")
    for label, f in (("mean disagreement rate", np.mean),
                     ("median", np.median),
                     ("90th percentile", lambda x: np.percentile(x, 90))):
        print(f"{label:<34}{f(eps_all):>12.4f}{f(eps_multi):>18.4f}")
    frac0 = float((eps_all == 0).mean())
    print(f"{'proteins with zero disagreement':<34}{frac0:>11.1%}"
          f"{float((eps_multi == 0).mean()):>17.1%}")

    # Pooled over residues, which is the rate `ranking_certified` needs: the
    # benchmark scores residues, not proteins, so a per-protein mean would
    # weight a 60-residue chain like a 2,000-residue one.
    tot_ctx = sum(r["n_context_dependent"] for r in rows)
    tot_ev = sum(r["n_evidenced"] for r in rows)
    eps_pooled = tot_ctx / tot_ev
    print(f"\npooled over residues: {tot_ctx:,} context-dependent of "
          f"{tot_ev:,} evidenced = {eps_pooled:.4f}")
    print(f"structures per protein: median "
          f"{np.median([r['n_structures'] for r in rows]):.0f}, "
          f"max {max(r['n_structures'] for r in rows)}")

    report = {"n_accessions": len(accs), "n_usable": len(rows),
              "epsilon_pooled": eps_pooled,
              "epsilon_mean": float(np.mean(eps_all)),
              "epsilon_median": float(np.median(eps_all)),
              "epsilon_p90": float(np.percentile(eps_all, 90)),
              "n_multi_structure": len(multi),
              "proteins": rows}
    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
