"""Measure what each label source actually yields, before spending GPU time.

Answers the questions that decide whether the data-scale plan is viable:
how many trainable proteins per source, how much of each chain carries
evidence, and what the class balance looks like on evidenced residues.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from colab.label_sources import (  # noqa: E402
    LabelSource,
    build_labelled_set,
    fetch_mobidb_proteome,
)

PROTEOME = os.environ.get("DISORDERNET_MOBIDB_PROTEOME", "UP000005640")
# "global" selects every MobiDB entry carrying PDB missing-residue annotation,
# across organisms, rather than one reference proteome.
GLOBAL = os.environ.get("DISORDERNET_MOBIDB_GLOBAL", "0") == "1"
QUERY = {"derived-missing_residues-th_90": "exists"} if GLOBAL else None
LIMIT = int(os.environ.get("DISORDERNET_MOBIDB_LIMIT", "0") or 0) or None
CACHE = os.environ.get(
    "DISORDERNET_MOBIDB_CACHE",
    str(Path.home() / ".cache" / "disordernet" /
        (f"mobidb_pdbcov.ndjson" if GLOBAL else f"mobidb_{PROTEOME}.ndjson")),
)

print(f"selector={'global pdb-coverage' if GLOBAL else PROTEOME} limit={LIMIT or 'all'}")
t0 = time.time()
records = fetch_mobidb_proteome(PROTEOME, CACHE, limit=LIMIT, query=QUERY)
print(f"records: {len(records)}  ({time.time() - t0:.0f}s)\n")

summary = {"proteome": "global_pdb_coverage" if GLOBAL else PROTEOME, "n_records": len(records), "sources": {}}
for source in (LabelSource.MOBIDB_CURATED, LabelSource.PDB_MISSING, LabelSource.UNION):
    t1 = time.time()
    kept, stats = build_labelled_set(
        records, source, min_len=30, max_len=1022, min_evidence_fraction=0.10,
    )
    stats["seconds"] = round(time.time() - t1, 1)
    summary["sources"][source.value] = stats

print("\n" + "=" * 78)
print(f"{'source':<18} {'proteins':>9} {'evid.res':>12} {'evid.frac':>10} {'disorder':>9}")
print("=" * 78)
for name, s in summary["sources"].items():
    print(
        f"{name:<18} {s['n_proteins']:>9,} {s['n_evidenced_residues']:>12,} "
        f"{s['evidence_fraction']:>10.1%} {s['disorder_fraction_of_evidenced']:>9.1%}"
    )

base = 2340  # DisProt after leak-free filtering, for scale reference
print("=" * 78)
for name, s in summary["sources"].items():
    if s["n_proteins"]:
        print(f"  {name:<18} = {s['n_proteins'] / base:5.1f}x the current DisProt set ({base})")

out = Path(os.environ.get("DISORDERNET_RESULTS", str(Path.home()))) / "label_source_probe.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
print(f"\nWrote {out}")
