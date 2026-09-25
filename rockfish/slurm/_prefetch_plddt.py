"""Populate the shared AlphaFold pLDDT cache for the full DisProt protein set.

Run on CPU before the GPU campaign. Reports coverage explicitly: the
structure-distrust claim compares DisorderNet against an inverse-pLDDT baseline
on AF-covered residues, so knowing how many proteins actually have AF entries is
part of the result, not an implementation detail.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from colab.af_plddt import fetch_plddt_batch  # noqa: E402
from colab.disordernet_gpu import TrainConfig, fetch_disprot, process_disprot  # noqa: E402

CACHE = os.environ.get(
    "DISORDERNET_PLDDT_CACHE", str(Path.home() / ".cache" / "disordernet" / "af_plddt")
)
DISPROT = os.environ.get(
    "DISORDERNET_DISPROT_CACHE",
    str(Path.home() / ".cache" / "disordernet" / "disprot_raw.json"),
)
Path(CACHE).mkdir(parents=True, exist_ok=True)

print(f"pLDDT cache : {CACHE}")
print(f"DisProt     : {DISPROT}")

raw = fetch_disprot(cache_path=DISPROT)
cfg = TrainConfig.from_profile("ultra")
proteins, _ = process_disprot(raw, cfg)
with_acc = [p for p in proteins if p.get("uniprot_acc")]
print(f"proteins: {len(proteins)}  with UniProt accession: {len(with_acc)}")

t0 = time.time()
plddt = fetch_plddt_batch(
    proteins,
    cache_dir=CACHE,
    max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    verbose=True,
)
elapsed = time.time() - t0

n_cached = len(list(Path(CACHE).glob("*.json")))
cov = 100.0 * len(plddt) / max(len(proteins), 1)
print(f"\nfetched pLDDT for {len(plddt)} / {len(proteins)} proteins ({cov:.1f}%)")
print(f"cache entries on disk: {n_cached}")
print(f"elapsed: {elapsed / 60:.1f} min")

if not plddt:
    print(
        "\nFAIL: no pLDDT retrieved. The structure channel, hallucination "
        "weighting and structure_distrust_benchmark.json all depend on this; "
        "a strict package job would fail after the GPU runs complete.",
        file=sys.stderr,
    )
    sys.exit(1)

if cov < 50.0:
    print(
        f"\nWARN: only {cov:.1f}% AF coverage. Matched-residue comparisons will "
        "rest on a minority of the set — report the coverage alongside any "
        "distrust claim.",
        file=sys.stderr,
    )

print("\nPLDDT_PREFETCH_OK")
