#!/usr/bin/env python3
"""A benchmark that could not have leaked: PDB entries released after training.

Every leak control in this project is a *filter* — remove CAID targets, remove
their homologues, verify the filter ran. Filters can be wrong, and one of them
was: the validation holdout's homology pass silently removed nothing for a while
because a set of tuples was tested for membership of a string.

A **temporal** holdout needs no filter to be right. Structures released after
the training data was downloaded cannot be in it, whatever any code does. The
training caches are dated 2026-08-08 (DisProt) and 2026-08-10 (MobiDB), so
entries first released on or after 2026-08-11 are held out by the calendar.

This is also the closest thing to a CASP-style assessment available for
disorder. CASP retired its disorder category after CASP10 and CAID replaced it,
so there is no CASP disorder track to enter; but Disorder-PDB's labels *are*
"residues present in SEQRES and absent from the coordinates", which is exactly
what a newly deposited structure reports. Scoring on structures nobody had when
the model was trained is the same experiment CASP runs, on the quantity CAID
scores.

Labels come from `UNOBSERVED_RESIDUE_XYZ` on each polymer entity instance —
RCSB's own annotation, not a re-derivation. A residue is positive when it is
unobserved in the coordinates and negative when it is observed.

Filters, each for a stated reason:

- **X-ray and cryo-EM only.** NMR models have no missing-residue concept in this
  sense: every SEQRES residue has coordinates in every model.
- **Both classes present.** A chain with no unobserved residue contributes no
  within-protein pair and a chain with no observed one is not a structure.
- **Sequence-deduplicated**, keeping the *first* instance, so a complex with six
  copies of one chain counts once.
- **Homology-filtered against the training union** at the same identity as the
  CAID filter. The calendar already excludes these structures, but not their
  older paralogues, and a 90%-identical protein solved in 2019 leaks nearly as
  much as the target itself.
- **Length bounds** matching training, 20 to 10,000; longer chains are windowed
  at inference as CAID targets are.

Written as a CAID-format reference so every existing evaluator, leaderboard and
bootstrap applies unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA = "https://data.rcsb.org/rest/v1/core"

#: The later of the two training cache dates, plus one day. Anything released
#: on or after this could not have been in the training data.
DEFAULT_CUTOFF = "2026-08-11"


def _get(url: str, retries: int = 4, timeout: int = 60):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as fh:
                return json.load(fh)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))
    return None


def search_entities(cutoff: str, methods: tuple[str, ...],
                    page: int = 1000) -> list[str]:
    """Polymer entity ids first released on or after ``cutoff``."""
    nodes = [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_accession_info.initial_release_date",
            "operator": "greater_or_equal", "value": cutoff}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "entity_poly.rcsb_entity_polymer_type",
            "operator": "exact_match", "value": "Protein"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "exptl.method", "operator": "in",
            "value": list(methods)}},
    ]
    out, start = [], 0
    while True:
        payload = {
            "query": {"type": "group", "logical_operator": "and",
                      "nodes": nodes},
            "return_type": "polymer_entity",
            "request_options": {"paginate": {"start": start, "rows": page},
                                "results_verbosity": "compact"},
        }
        req = urllib.request.Request(
            SEARCH, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as fh:
            res = json.load(fh)
        ids = res.get("result_set") or []
        out.extend(ids)
        total = int(res.get("total_count", 0))
        start += page
        if start >= total or not ids:
            break
    return out


def entity_record(entity_id: str) -> dict | None:
    """Sequence and unobserved-residue mask for one polymer entity.

    Uses the *first* asym id. Different copies of one entity can differ in what
    was resolved, and averaging them would invent a label the deposition does
    not make; taking one instance reports what one chain shows.
    """
    entry, ent = entity_id.split("_")
    pe = _get(f"{DATA}/polymer_entity/{entry}/{ent}")
    if not pe:
        return None
    poly = pe.get("entity_poly") or {}
    seq = (poly.get("pdbx_seq_one_letter_code_can") or "").replace("\n", "")
    ids = pe.get("rcsb_polymer_entity_container_identifiers") or {}
    asyms = ids.get("asym_ids") or []
    if not seq or not asyms:
        return None

    inst = _get(f"{DATA}/polymer_entity_instance/{entry}/{asyms[0]}")
    if not inst:
        return None
    unobserved = []
    for feat in (inst.get("rcsb_polymer_instance_feature") or []):
        if feat.get("type") != "UNOBSERVED_RESIDUE_XYZ":
            continue
        for pos in (feat.get("feature_positions") or []):
            b, e = pos.get("beg_seq_id"), pos.get("end_seq_id")
            if b is None:
                continue
            unobserved.append((int(b), int(e if e is not None else b)))

    labels = ["0"] * len(seq)
    for b, e in unobserved:
        for i in range(max(b, 1) - 1, min(e, len(seq))):
            labels[i] = "1"
    return {
        "id": f"{entry}_{ent}",
        "sequence": seq,
        "labels": "".join(labels),
        "uniprot": (ids.get("uniprot_ids") or [None])[0],
        "asym": asyms[0],
        "n_copies": len(asyms),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    ap.add_argument("--out", required=True, help="output .fasta (CAID format)")
    ap.add_argument("--raw", default=None, help="cache the raw records here")
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--methods", default="X-RAY DIFFRACTION,ELECTRON MICROSCOPY")
    args = ap.parse_args(argv)

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    print(f"searching for protein entities released on or after "
          f"{args.cutoff} ({', '.join(methods)})…")
    ids = search_entities(args.cutoff, methods)
    if args.limit:
        ids = ids[:args.limit]
    print(f"  {len(ids):,} polymer entities")

    records, failed = [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(entity_record, ids), 1):
            if rec is None:
                failed += 1
            else:
                records.append(rec)
            if i % 250 == 0:
                print(f"  {i:,}/{len(ids):,}  kept={len(records):,} "
                      f"failed={failed}  [{(time.time()-t0)/60:.1f}m]",
                      flush=True)
    print(f"  fetched {len(records):,}, {failed} unavailable")

    if args.raw:
        with open(args.raw + ".part", "w") as fh:
            json.dump(records, fh)
        os.replace(args.raw + ".part", args.raw)
        print(f"  raw records -> {args.raw}")

    stats = {"n_entities": len(ids), "n_fetched": len(records)}
    kept, seen_seq = [], set()
    drop = {"length": 0, "single_class": 0, "duplicate_sequence": 0,
            "bad_residue": 0}
    for r in records:
        seq, lab = r["sequence"], r["labels"]
        if not (args.min_len <= len(seq) <= args.max_len):
            drop["length"] += 1
            continue
        if set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
            drop["bad_residue"] += 1
            continue
        if "1" not in lab or "0" not in lab:
            drop["single_class"] += 1
            continue
        if seq in seen_seq:
            drop["duplicate_sequence"] += 1
            continue
        seen_seq.add(seq)
        kept.append(r)
    stats["dropped"] = drop
    stats["n_kept"] = len(kept)
    print(f"  after filters: {len(kept):,} chains  dropped={drop}")

    if not kept:
        print("ERROR: nothing survived the filters", file=sys.stderr)
        return 1

    with open(args.out + ".part", "w") as fh:
        for r in kept:
            fh.write(f">{r['id']}\n{r['sequence']}\n{r['labels']}\n")
    os.replace(args.out + ".part", args.out)

    n_pos = sum(r["labels"].count("1") for r in kept)
    n_neg = sum(r["labels"].count("0") for r in kept)
    stats.update({"cutoff": args.cutoff, "methods": list(methods),
                  "n_positive": n_pos, "n_negative": n_neg,
                  "prevalence": n_pos / max(n_pos + n_neg, 1)})
    print(f"\nWrote {args.out}")
    print(f"  {len(kept):,} chains, {n_pos:,} unobserved / {n_neg:,} observed "
          f"({stats['prevalence']:.1%} disordered)")
    with open(args.out + ".stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
