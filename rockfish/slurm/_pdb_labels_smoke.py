"""End-to-end CPU check of the pdb_missing data path, before any GPU arm.

Validates the three things that would waste a 20-hour ablation arm if wrong:
  1. the loader produces sane protein records under the new label source,
  2. homology clustering scales to ~20k proteins in acceptable time,
  3. the resulting CV folds are disjoint, exhaustive and evidence-bearing.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from colab.cv_splits import get_cv_splits  # noqa: E402
from colab.disordernet_gpu import TrainConfig  # noqa: E402
from colab.homology_splits import cluster_proteins_by_homology_cached  # noqa: E402
from colab.label_sources import (  # noqa: E402
    LabelSource,
    build_labelled_set,
    fetch_mobidb_proteome,
    to_pipeline_proteins,
)

SOURCE = LabelSource(os.environ.get("DISORDERNET_LABEL_SOURCE", "pdb_missing"))
CACHE = os.environ.get(
    "DISORDERNET_MOBIDB_CACHE",
    str(Path.home() / ".cache" / "disordernet" / "mobidb_pdbcov.ndjson"),
)
LIMIT = int(os.environ.get("DISORDERNET_MOBIDB_LIMIT", "0") or 0) or None

cfg = TrainConfig.from_profile("ultra")
print(f"source={SOURCE.value}  min_len={cfg.min_seq_len} max_len={cfg.max_seq_len}")

records = fetch_mobidb_proteome(
    "", CACHE, limit=LIMIT, query={"derived-missing_residues-th_90": "exists"}
)
labelled, stats = build_labelled_set(
    records, SOURCE,
    min_len=cfg.min_seq_len, max_len=cfg.max_seq_len,
    min_evidence_fraction=0.10,
    min_disorder=cfg.min_disorder, min_order=cfg.min_order,
)
proteins = to_pipeline_proteins(labelled)
print(f"\nproteins: {len(proteins)}")

failures: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")
    if not ok:
        failures.append(label)


# --- 1. record sanity -------------------------------------------------------
bad_len = [p for p in proteins if len(p["labels"]) != p["length"]
           or len(p["label_evidence"]) != p["length"]]
check("labels/evidence length matches sequence", not bad_len, f"{len(bad_len)} bad")

no_ev = [p for p in proteins if not any(p["label_evidence"])]
check("every protein has some evidence", not no_ev, f"{len(no_ev)} without")

both = [p for p in proteins
        if 0 < sum(1 for lab, ev in zip(p["labels"], p["label_evidence"]) if ev and lab)
        < sum(p["label_evidence"])]
check("both classes present among evidenced residues", len(both) == len(proteins),
      f"{len(both)}/{len(proteins)}")

ev_frac = np.mean([np.mean(p["label_evidence"]) for p in proteins])
check("mean evidence coverage is partial, not all-ones", 0.05 < ev_frac < 0.999,
      f"{ev_frac:.1%} (all-ones would mean the mask is not doing anything)")

# --- 2. clustering at scale -------------------------------------------------
t0 = time.time()
_, meta = cluster_proteins_by_homology_cached(proteins, min_identity=0.40)
elapsed = time.time() - t0
print(f"\n  clustering: {meta['n_clusters']} clusters from {len(proteins)} "
      f"proteins in {elapsed:.0f}s via {meta.get('backend')}")
check("clustering completes under 30 min", elapsed < 1800, f"{elapsed:.0f}s")
check("clustering is not degenerate", not meta.get("degenerate"),
      f"{meta['n_clusters']} clusters / {len(proteins)} proteins")

# --- 3. folds ---------------------------------------------------------------
t1 = time.time()
splits = get_cv_splits(proteins, 5, split_method="homology", homology_min_identity=0.40)
print(f"  5-fold split in {time.time() - t1:.0f}s (cache hit expected)")
sizes = [len(v) for _, v in splits]
covered = sorted(i for _, v in splits for i in v)
check("folds partition the set exactly", covered == list(range(len(proteins))),
      f"sizes={sizes}")
check("no protein in two validation folds", len(covered) == len(set(covered)))
for tr, va in splits:
    if set(tr) & set(va):
        failures.append("train/val overlap")
        break
else:
    check("train and val disjoint in every fold", True)

summary = {
    "source": SOURCE.value,
    "n_proteins": len(proteins),
    "label_stats": stats,
    "clustering": meta,
    "fold_sizes": sizes,
    "clustering_seconds": round(elapsed, 1),
    "failures": failures,
}
out = Path(os.environ.get("DISORDERNET_RESULTS", str(Path.home()))) / "pdb_labels_smoke.json"
out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
print(f"\nWrote {out}")

if failures:
    print(f"\n{len(failures)} CHECK(S) FAILED: {failures}", file=sys.stderr)
    sys.exit(1)
print("\nPDB_LABELS_SMOKE_OK")
