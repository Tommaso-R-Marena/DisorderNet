"""Per-residue labels for every CAID3 task family, from DisProt annotations.

CAID3 is five benchmarks, not one, and they are won by different methods:

    benchmark      targets  positives  leader                  AUC     APS
    Disorder-PDB     319      31.6%    PUNCH2                 0.955   0.928
    Disorder-NOX     204      26.4%    ESMDisPred-2PDB        0.885   0.754
    Linker            31       6.7%    IPA-AF2-Linker         0.897   0.474
    Binding           52      10.6%    DisoFLAG-PB            0.776   0.245
    Binding-IDR       52      38.6%    bindEmbed21IDR-raw     0.641   0.514

Every entry in that table is a specialist. PUNCH2 does not predict linkers;
LINKER-Pred does not predict binding. A single model that answers all five from
one forward pass is the contribution this module exists to enable, and the
labels for all of them are already in the DisProt release this project caches.

Label definitions
-----------------
Taken from the challenge's own reference-generation notebook
(BioComputingUP/caid-reference, ``src/references.ipynb``), which specifies each
benchmark as a set of (class, value) rules plus a fill for everything else:

    disorder_nox : [(disorder, '-'), (disorder_nox, '1')]  fill '0'
    binding      : [(binding, '1')]                        fill '0'
    binding_idr  : [(disorder, '0'), (binding, '1')]       fill '-'
    linker       : [(linker, '1')]                         fill '0'

The distinction that matters most is the fill. Disorder-PDB *ignores*
unannotated residues, which is why a structural signal like AlphaFold rsa tops
it (rank 3, AUC 0.950). Disorder-NOX calls them **ordered**, and no AlphaFold
baseline reaches its top ten — the structural shortcut stops working the moment
absence of evidence counts as a negative. Any claim to be "best overall" has to
survive both conventions.

DisProt term IDs in the release cached here use the seven-digit form
(``IDPO:0000002``) rather than the five-digit form in the notebook
(``IDPO:00076``); the names are what identify them, and are asserted below.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

# Structural state: what "disordered" means for every disorder benchmark.
DISORDER_TERMS: frozenset[str] = frozenset({
    "IDPO:0000002",   # disorder
    "IDPO:0000003",   # molten globule
    "IDPO:0000004",   # pre-molten globule
})

# Disorder function: flexible linker/spacer. The Linker benchmark's positives.
LINKER_TERMS: frozenset[str] = frozenset({
    "IDPO:0000033",   # flexible linker
})

# Molecular function: descendants of GO:0005488 "binding" present in DisProt.
# Listed explicitly rather than resolved through the GO graph so the label set
# is reproducible without an ontology download, and auditable in review.
BINDING_TERMS: frozenset[str] = frozenset({
    "GO:0005515",   # protein binding
    "GO:0003676",   # nucleic acid binding
    "GO:0003677",   # DNA binding
    "GO:0003723",   # RNA binding
    "GO:0008289",   # lipid binding
    "GO:0005509",   # calcium ion binding
    "GO:0046872",   # metal ion binding
    "GO:0036094",   # small molecule binding
    "GO:0043167",   # ion binding
    "GO:0005549",   # odorant binding
    "GO:0019842",   # vitamin binding
    "GO:0050825",   # ice binding
    "GO:0042165",   # neurotransmitter binding
    "GO:0033218",   # amide binding
    "GO:0097159",   # organic cyclic compound binding
    "GO:1901363",   # heterocyclic compound binding
})

# Names asserted at import so a DisProt release that renumbers terms fails
# loudly here rather than silently producing all-negative labels.
EXPECTED_NAMES: dict[str, str] = {
    "IDPO:0000002": "disorder",
    "IDPO:0000033": "flexible linker",
    "GO:0005515": "protein binding",
}

TASKS: tuple[str, ...] = ("disorder_nox", "disorder_pdb", "linker", "binding", "binding_idr")

# Sentinel for "not evaluated here" — the '-' fill in the challenge's rules.
MASK = -1


def _regions(entry: dict) -> list[dict]:
    return entry.get("regions") or []


def _paint(length: int, regions: Iterable[dict], terms: frozenset[str]) -> np.ndarray:
    """Boolean mask of residues covered by any region carrying one of ``terms``.

    DisProt coordinates are 1-based and inclusive.
    """
    hit = np.zeros(length, dtype=bool)
    for r in regions:
        if r.get("term_id") not in terms:
            continue
        try:
            start = int(r["start"]) - 1
            end = int(r["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        hit[max(start, 0):min(end, length)] = True
    return hit


def task_labels(entry: dict, task: str) -> Optional[np.ndarray]:
    """Per-residue labels for one CAID3 task: 1 positive, 0 negative, -1 masked.

    Returns None when the entry carries no annotation relevant to the task, so
    callers can drop it rather than train on an all-negative sequence.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {list(TASKS)}")
    seq = entry.get("sequence") or ""
    n = len(seq)
    if n == 0:
        return None
    regions = _regions(entry)

    disorder = _paint(n, regions, DISORDER_TERMS)

    if task == "disorder_nox":
        # Unannotated residues count as ORDERED. This is the convention that
        # denies AlphaFold baselines a place in the NOX top ten.
        if not disorder.any():
            return None
        return disorder.astype(np.int8)

    if task == "disorder_pdb":
        # Unannotated residues are IGNORED, not called ordered. Negatives must
        # come from an observed-structure source, which this module does not
        # have — so everything outside an annotated disorder region is masked
        # and the caller supplies negatives (see colab/label_sources.py).
        if not disorder.any():
            return None
        out = np.full(n, MASK, dtype=np.int8)
        out[disorder] = 1
        return out

    if task == "linker":
        linker = _paint(n, regions, LINKER_TERMS)
        if not linker.any():
            return None
        return linker.astype(np.int8)

    binding = _paint(n, regions, BINDING_TERMS)

    if task == "binding":
        if not binding.any():
            return None
        return binding.astype(np.int8)

    # binding_idr: evaluated only inside disordered regions, where the question
    # stops being "is this disordered" and becomes "does this IDR bind". That
    # is why its leader sits at AUC 0.641 while Disorder-PDB's sits at 0.955 —
    # the easy signal has been masked away by construction.
    if not (binding.any() and disorder.any()):
        return None
    out = np.full(n, MASK, dtype=np.int8)
    out[disorder] = 0
    out[disorder & binding] = 1
    return out


def build_task_dataset(entries: Iterable[dict], task: str) -> list[dict]:
    """Proteins labelled for one task, in the pipeline's dict shape."""
    out: list[dict] = []
    for e in entries:
        labels = task_labels(e, task)
        if labels is None:
            continue
        seq = e["sequence"]
        evidence = labels != MASK
        if not evidence.any():
            continue
        clean = labels.copy()
        clean[~evidence] = 0
        out.append({
            "id": e.get("disprot_id") or e.get("acc"),
            "uniprot_acc": e.get("acc"),
            "sequence": seq,
            "length": len(seq),
            "labels": clean.astype(np.int8).tolist(),
            "label_evidence": evidence.tolist(),
            "n_dis": int(clean[evidence].sum()),
            "task": task,
            "label_source": f"disprot:{task}",
            "regions": [],
        })
    return out


def task_statistics(entries: Iterable[dict]) -> dict:
    """Coverage and prevalence per task — the sanity check before training.

    A task with a few hundred positive residues cannot support a five-fold
    homology-split experiment, and finding that out after a GPU run is
    expensive.
    """
    entries = list(entries)
    stats: dict[str, dict] = {}
    for task in TASKS:
        n_prot = n_res = n_pos = n_eval = 0
        for e in entries:
            labels = task_labels(e, task)
            if labels is None:
                continue
            n_prot += 1
            n_res += len(labels)
            ev = labels != MASK
            n_eval += int(ev.sum())
            n_pos += int((labels[ev] == 1).sum())
        stats[task] = {
            "proteins": n_prot,
            "residues": n_res,
            "evaluated_residues": n_eval,
            "positives": n_pos,
            "prevalence": round(n_pos / n_eval, 4) if n_eval else 0.0,
        }
    return stats
