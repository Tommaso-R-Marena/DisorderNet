"""Time and sanity-check homology clustering on the real DisProt release.

Answers the two questions that decide whether the publish run is viable:
  1. does clustering finish in a sane time on ~3.3k real proteins, and
  2. does it actually cluster (the autojunk bug made it a silent no-op)?
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from colab.disordernet_gpu import TrainConfig, fetch_disprot, process_disprot  # noqa: E402
from colab.homology_splits import (  # noqa: E402
    _blast_available,
    cluster_proteins_by_homology,
    get_homology_cv_splits,
)

CACHE = os.environ.get(
    "DISORDERNET_DISPROT_CACHE",
    str(Path.home() / ".cache" / "disordernet" / "disprot_raw.json"),
)
Path(CACHE).parent.mkdir(parents=True, exist_ok=True)

print(f"BLAST available: {_blast_available()}")
print(f"DisProt cache:   {CACHE}")

raw = fetch_disprot(cache_path=CACHE)
# Use the ultra profile's filters so the protein set matches the publish run.
cfg = TrainConfig.from_profile("ultra") if hasattr(TrainConfig, "from_profile") else TrainConfig()
proteins, skipped = process_disprot(raw, cfg)
print(f"proteins after filtering: {len(proteins)}  (skipped: {dict(skipped)})")
lens = sorted(len(p["sequence"]) for p in proteins)
print(
    f"length: min={lens[0]} median={lens[len(lens) // 2]} "
    f"p90={lens[int(len(lens) * 0.9)]} max={lens[-1]}"
)
print(f"proteins >=200 residues: {sum(1 for L in lens if L >= 200)} / {len(lens)} "
      f"(these are the ones the autojunk bug silenced)")

t0 = time.time()
_, meta = cluster_proteins_by_homology(proteins, min_identity=0.40)
elapsed = time.time() - t0

print("\n=== clustering result ===")
print(json.dumps(meta, indent=2, default=str))

n, k = meta["n_proteins"], meta["n_clusters"]
print(f"\nwall time: {elapsed:.1f}s")
print(f"clusters:  {k} / {n} proteins  ({n - k} proteins absorbed into families)")

if meta.get("degenerate"):
    print("\nFAIL: clustering is degenerate — every protein is its own cluster, so "
          "the homology split is indistinguishable from a protein split.",
          file=sys.stderr)
    sys.exit(1)

splits, split_meta = get_homology_cv_splits(proteins, 5, min_identity=0.40)
sizes = [len(v) for _, v in splits]
print(f"\n5-fold val sizes: {sizes}  (sum={sum(sizes)}, n={n})")
if split_meta.get("fallback"):
    print(f"NOTE: {split_meta['fallback_reason']}")
print("\nCLUSTER_BENCH_OK")
