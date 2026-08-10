"""Multi-source disorder labels with explicit per-residue evidence.

Why this exists
---------------
DisProt curation (~2.3k proteins after filtering) is small enough that the
650M+LoRA model demonstrably memorises it: a screen reached train loss 0.069
while validating at AUC 0.66. Scaling the label set is the highest-leverage
change available, and MobiDB exposes two further sources at 30x the scale.

It also fixes a *task* mismatch. CAID3's Disorder-PDB reference defines disorder
as **residues missing from crystal structures**, not as curator-annotated
functional disorder. Training on DisProt and scoring on Disorder-PDB is a domain
shift; ``LabelSource.PDB_MISSING`` trains on the same definition the benchmark
scores.

The evidence mask is the important part
---------------------------------------
DisProt curates whole proteins, so "not annotated disordered" reasonably means
ordered. **PDB-derived labels do not work that way.** A residue is only
informative if some structure covers it:

    missing from a solved structure  -> disordered   (positive)
    observed in a solved structure   -> ordered      (negative)
    never crystallised               -> UNKNOWN      (must be excluded)

Treating never-crystallised residues as ordered would inject large numbers of
false negatives — exactly the kind of silent label corruption that inflates or
destroys a benchmark number with no visible symptom. Every source here therefore
returns ``(labels, evidence)`` and callers must honour ``evidence``.

``prediction-disorder-*`` fields are deliberately NOT usable as labels. They are
other predictors' outputs; training on them is distillation, and presenting the
result as ground-truth performance would be dishonest.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

import numpy as np

MOBIDB_DOWNLOAD = "https://mobidb.org/api/download"
REQUEST_TIMEOUT = 60

# Fields we request. Keep this list tight: the full record is ~170KB per protein
# (hundreds of per-PDB-chain sub-entries) and the bulk pull is otherwise enormous.
MOBIDB_PROJECTION = (
    "acc,length,sequence,"
    "curated-disorder-priority,"
    "derived-missing_residues-th_90,"
    "derived-observed-priority"
)

CURATED_KEY = "curated-disorder-priority"
MISSING_KEY = "derived-missing_residues-th_90"
OBSERVED_KEY = "derived-observed-priority"


class LabelSource(str, Enum):
    """Which definition of "disordered" to train against."""

    DISPROT = "disprot"                # curator-annotated functional disorder
    MOBIDB_CURATED = "mobidb_curated"  # MobiDB curated consensus (DisProt+IDEAL+…)
    PDB_MISSING = "pdb_missing"        # missing residues — the CAID3 Disorder-PDB definition
    UNION = "union"                    # curated positives OR PDB-missing positives


@dataclass
class LabelledProtein:
    """One protein with per-residue labels and per-residue evidence."""

    id: str
    sequence: str
    labels: np.ndarray            # int8, 1 = disordered
    evidence: np.ndarray          # bool,  True = this residue is informative
    source: str
    uniprot_acc: str = ""
    provenance: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def n_evidenced(self) -> int:
        return int(self.evidence.sum())

    @property
    def n_disordered(self) -> int:
        return int((self.labels[self.evidence] == 1).sum())


def regions_to_mask(regions: Optional[Iterable], length: int) -> np.ndarray:
    """MobiDB regions ([start, end], 1-indexed inclusive) -> boolean mask.

    Out-of-range and inverted regions are clamped rather than trusted; MobiDB
    occasionally carries annotations against a different isoform length.
    """
    mask = np.zeros(length, dtype=bool)
    if not regions:
        return mask
    for reg in regions:
        try:
            start, end = int(reg[0]), int(reg[1])
        except (TypeError, ValueError, IndexError):
            continue
        if end < start:
            start, end = end, start
        lo = max(0, start - 1)          # 1-indexed inclusive -> 0-indexed
        hi = min(length, end)           # inclusive end -> exclusive
        if lo < hi:
            mask[lo:hi] = True
    return mask


def _regions_of(record: dict, key: str) -> Optional[list]:
    node = record.get(key)
    if isinstance(node, dict):
        return node.get("regions")
    return None


def labels_from_record(
    record: dict,
    source: LabelSource,
    *,
    require_evidence_fraction: float = 0.0,
) -> Optional[LabelledProtein]:
    """Build labels + evidence for one MobiDB record under a given definition.

    Returns None when the record cannot support the requested definition (no
    sequence, or no evidence of the required kind), rather than silently
    emitting an all-zero protein.
    """
    seq = (record.get("sequence") or "").strip()
    acc = str(record.get("acc") or "").strip()
    if not seq or not acc:
        return None
    n = len(seq)

    curated = regions_to_mask(_regions_of(record, CURATED_KEY), n)
    missing = regions_to_mask(_regions_of(record, MISSING_KEY), n)
    observed = regions_to_mask(_regions_of(record, OBSERVED_KEY), n)

    if source is LabelSource.MOBIDB_CURATED:
        # Curation covers the whole chain: absence of annotation means ordered.
        labels = curated.astype(np.int8)
        evidence = np.ones(n, dtype=bool)
        if not curated.any():
            return None

    elif source is LabelSource.PDB_MISSING:
        # Only structurally covered residues carry information.
        evidence = missing | observed
        labels = missing.astype(np.int8)
        if not evidence.any():
            return None

    elif source is LabelSource.UNION:
        # Positive under either definition; evidenced under either definition.
        evidence = np.ones(n, dtype=bool) if curated.any() else (missing | observed)
        labels = (curated | missing).astype(np.int8)
        if not evidence.any():
            return None

    else:
        raise ValueError(f"{source} is not sourced from MobiDB records")

    # Guard against structurally-covered-but-uninformative proteins.
    if require_evidence_fraction > 0.0:
        if evidence.mean() < require_evidence_fraction:
            return None

    return LabelledProtein(
        id=acc,
        sequence=seq,
        labels=labels,
        evidence=evidence,
        source=source.value,
        uniprot_acc=acc,
        provenance={
            "n_curated": int(curated.sum()),
            "n_missing": int(missing.sum()),
            "n_observed": int(observed.sum()),
            "evidence_fraction": round(float(evidence.mean()), 4),
        },
    )


def fetch_mobidb_proteome(
    proteome: str,
    cache_path: str,
    *,
    limit: Optional[int] = None,
    force: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """Bulk-download a MobiDB proteome as newline-delimited JSON, with a disk cache.

    The projection matters: a full record carries hundreds of per-PDB-chain
    sub-entries (~170KB each), so an unprojected pull of a 82k-protein proteome
    is many gigabytes for a handful of fields we actually use.
    """
    import requests

    if os.path.exists(cache_path) and not force:
        records = []
        with open(cache_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if verbose:
            print(f"  MobiDB cache hit: {cache_path} ({len(records)} records)", flush=True)
        return records

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    params = {"proteome": proteome, "format": "json", "projection": MOBIDB_PROJECTION}
    if limit:
        params["limit"] = int(limit)

    if verbose:
        print(f"  MobiDB download: proteome={proteome} limit={limit or 'all'}…", flush=True)
    t0 = time.time()
    records: list[dict] = []
    tmp_path = cache_path + ".partial"
    with requests.get(
        MOBIDB_DOWNLOAD, params=params, timeout=REQUEST_TIMEOUT, stream=True
    ) as resp:
        resp.raise_for_status()
        with open(tmp_path, "w") as out:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                records.append(rec)
                out.write(json.dumps(rec) + "\n")
    os.replace(tmp_path, cache_path)  # atomic: a truncated cache is worse than none
    if verbose:
        print(
            f"  MobiDB: {len(records)} records in {time.time() - t0:.0f}s → {cache_path}",
            flush=True,
        )
    return records


def build_labelled_set(
    records: Iterable[dict],
    source: LabelSource,
    *,
    min_len: int = 30,
    max_len: int = 1022,
    min_evidence_fraction: float = 0.10,
    min_disorder: int = 1,
    min_order: int = 1,
    verbose: bool = True,
) -> tuple[list[LabelledProtein], dict]:
    """Filter MobiDB records into a trainable set, reporting why entries were dropped.

    Filters mirror the DisProt path (length bounds, both classes present) but are
    applied to *evidenced* residues only — a protein whose disorder is entirely
    in never-crystallised territory carries no usable signal.
    """
    kept: list[LabelledProtein] = []
    skipped: dict[str, int] = {}

    def drop(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for rec in records:
        seq = (rec.get("sequence") or "").strip()
        if not seq:
            drop("no_sequence")
            continue
        if len(seq) < min_len:
            drop("too_short")
            continue
        if len(seq) > max_len:
            drop("too_long")
            continue

        lp = labels_from_record(rec, source, require_evidence_fraction=min_evidence_fraction)
        if lp is None:
            drop("no_evidence_for_source")
            continue

        ev = lp.evidence
        n_dis = int((lp.labels[ev] == 1).sum())
        n_ord = int((lp.labels[ev] == 0).sum())
        if n_dis < min_disorder:
            drop("too_few_disorder")
            continue
        if n_ord < min_order:
            drop("too_few_order")
            continue
        kept.append(lp)

    kept.sort(key=lambda p: p.id)  # deterministic order for reproducible splits

    total_res = sum(p.length for p in kept)
    evid_res = sum(p.n_evidenced for p in kept)
    dis_res = sum(p.n_disordered for p in kept)
    stats = {
        "source": source.value,
        "n_proteins": len(kept),
        "n_residues": total_res,
        "n_evidenced_residues": evid_res,
        "evidence_fraction": round(evid_res / max(total_res, 1), 4),
        "disorder_fraction_of_evidenced": round(dis_res / max(evid_res, 1), 4),
        "skipped": skipped,
    }
    if verbose:
        print(
            f"  {source.value}: {len(kept)} proteins, {evid_res:,}/{total_res:,} "
            f"evidenced residues ({stats['evidence_fraction']:.1%}), "
            f"disorder={stats['disorder_fraction_of_evidenced']:.1%}",
            flush=True,
        )
        if skipped:
            print(f"    dropped: {skipped}", flush=True)
    return kept, stats


def to_pipeline_proteins(labelled: Iterable[LabelledProtein]) -> list[dict]:
    """Adapt to the dict shape the training pipeline already consumes.

    ``evidence`` rides along so the loss and metrics can exclude uninformative
    residues; consumers that ignore it fall back to whole-sequence behaviour,
    which is correct for the curated sources and wrong for PDB-derived ones.
    """
    out: list[dict] = []
    for p in labelled:
        labels = p.labels.astype(np.int8)
        out.append(
            {
                "id": p.id,
                "uniprot_acc": p.uniprot_acc,
                "sequence": p.sequence,
                "length": p.length,
                "labels": labels.tolist(),
                "label_evidence": p.evidence.astype(bool).tolist(),
                "n_dis": int((labels[p.evidence] == 1).sum()),
                "label_source": p.source,
                "label_provenance": p.provenance,
                "regions": [],
            }
        )
    return out
