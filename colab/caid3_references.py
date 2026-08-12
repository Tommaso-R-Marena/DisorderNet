"""Reconstruct the CAID3 reference sets that are not published as FASTA.

Only ``disorder_pdb.fasta`` ships in the CAID demo-data. The other four
benchmarks — Disorder-NOX, Binding, Binding-IDR, Linker — are derivable, and
without them "best overall on CAID3" cannot be measured at all.

The reconstruction is validated by composition, not asserted. Published sizes
against what this module produces from the 319 Disorder-PDB targets plus our
DisProt release:

    benchmark      published            reconstructed       status
    Linker         31 tgt / 1,379 pos   31 / 1,379          EXACT
    Binding        52 tgt / 2,991 pos   49 / 4,673          approximate
    Binding-IDR    52 tgt / 2,991 pos   49 / 4,673          approximate
    Disorder-NOX  204 tgt / 26,367 pos  319 / 31,518        superset

Linker matching to the residue is the strongest evidence available that the
label rules — taken from the challenge's own generation notebook — are being
applied correctly. Only that one is treated as head-to-head; the others are
labelled approximate and must be reported as such, because a benchmark that is
easier or harder than the published one produces a number that cannot be put
beside a published leader.

Why the others differ
---------------------
*Binding* is over-inclusive here: CAID3 applies ``invalid_regions_fullCAID3.tsv``,
a manual validation pass that removes binding regions, which explains 4,673
positives against 2,991. Passing ``invalid_regions`` narrows the gap.

*Disorder-NOX* is a subset selection this module cannot reproduce: CAID3 derives
a separate ``disorder_nox`` class (disorder evidenced by methods other than
X-ray), keeping 204 of the targets. What is produced here is every target with a
disorder annotation, which is a superset and therefore an easier benchmark.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np

from colab.caid_tasks import MASK, task_labels

# Published CAID3 composition, for the checks below.
PUBLISHED: dict[str, dict] = {
    "disorder_pdb": {"targets": 319, "positives": 31401, "leader": "PUNCH2",
                     "auc": 0.955, "aps": 0.928},
    "disorder_nox": {"targets": 204, "positives": 26367, "leader": "ESMDisPred-2PDB",
                     "auc": 0.885, "aps": 0.754},
    "binding": {"targets": 52, "positives": 2991, "leader": "DisoFLAG-PB",
                "auc": 0.776, "aps": 0.245},
    "binding_idr": {"targets": 52, "positives": 2991,
                    "leader": "bindEmbed21IDR-rawGeneral", "auc": 0.641, "aps": 0.514},
    "linker": {"targets": 31, "positives": 1379, "leader": "IPA-AF2-Linker",
               "auc": 0.897, "aps": 0.474},
}

# A reconstruction is head-to-head only if it matches the published target and
# positive counts. Anything else is a different benchmark wearing the same name.
EXACT_TOLERANCE_TARGETS = 0
EXACT_TOLERANCE_POSITIVES = 0


def load_invalid_regions(path: Optional[str]) -> dict[str, list[tuple[int, int]]]:
    """CAID3's manual binding-region exclusions, keyed by DisProt id.

    Without these the Binding benchmark is over-inclusive: 4,673 positives
    against a published 2,991.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            i_id = next(i for i, h in enumerate(header)
                        if "disprot" in h.lower() or h.lower() in ("acc", "id"))
            i_s = next(i for i, h in enumerate(header) if h.lower().startswith("start"))
            i_e = next(i for i, h in enumerate(header) if h.lower().startswith("end"))
        except StopIteration:
            return out
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_id, i_s, i_e):
                continue
            try:
                out.setdefault(parts[i_id], []).append((int(parts[i_s]), int(parts[i_e])))
            except ValueError:
                continue
    return out


def build_reference(
    targets: list[dict],
    entries_by_id: dict[str, dict],
    task: str,
    invalid_regions: Optional[dict[str, list[tuple[int, int]]]] = None,
) -> list[dict]:
    """Reference records for one task over a fixed target list.

    ``targets`` are the official Disorder-PDB entries (id + sequence), which fix
    the target universe so a reconstruction cannot quietly evaluate a different
    or easier protein set.
    """
    out: list[dict] = []
    for t in targets:
        entry = entries_by_id.get(t["id"])
        if entry is None:
            continue
        labels = task_labels(entry, task)
        if labels is None:
            continue
        if len(labels) != len(t["sequence"]):
            # A length disagreement means the DisProt entry is not the same
            # sequence the challenge scored. Skip rather than align by guess.
            continue
        labels = labels.copy()
        for start, end in (invalid_regions or {}).get(t["id"], []):
            labels[max(start - 1, 0):min(end, len(labels))] = MASK
        if not (labels != MASK).any():
            continue
        out.append({
            "id": t["id"],
            "sequence": t["sequence"],
            "labels": labels,
            "eval_mask": labels != MASK,
            "task": task,
        })
    return out


def composition(reference: list[dict]) -> dict:
    """Target count, evaluated residues, positives and prevalence."""
    n_pos = n_eval = 0
    for r in reference:
        lab = np.asarray(r["labels"])
        m = np.asarray(r["eval_mask"])
        n_eval += int(m.sum())
        n_pos += int((lab[m] == 1).sum())
    return {
        "targets": len(reference),
        "evaluated_residues": n_eval,
        "positives": n_pos,
        "prevalence": round(n_pos / n_eval, 4) if n_eval else 0.0,
    }


def validate(task: str, reference: list[dict]) -> dict:
    """Compare a reconstruction against the published composition.

    ``head_to_head`` is the only flag that licenses putting a score beside a
    published leader. Everything else is a differently-composed benchmark, and
    a number from it is not comparable however close it looks.
    """
    comp = composition(reference)
    pub = PUBLISHED.get(task, {})
    d_t = comp["targets"] - pub.get("targets", 0)
    d_p = comp["positives"] - pub.get("positives", 0)
    exact = abs(d_t) <= EXACT_TOLERANCE_TARGETS and abs(d_p) <= EXACT_TOLERANCE_POSITIVES
    return {
        "task": task,
        "reconstructed": comp,
        "published": {k: pub.get(k) for k in ("targets", "positives", "leader",
                                              "auc", "aps")},
        "delta_targets": d_t,
        "delta_positives": d_p,
        "head_to_head": bool(exact),
        "verdict": (
            "EXACT — composition matches the published benchmark; scores are "
            "directly comparable to the published leader"
            if exact else
            "APPROXIMATE — composition differs, so a score here is NOT "
            "comparable to the published leader and must not be reported as if "
            "it were"
        ),
    }


def write_caid_fasta(reference: list[dict], path: str) -> str:
    """Write a reference in CAID's format: header, sequence, label line.

    Labels are '1'/'0' with '-' for residues the benchmark does not evaluate,
    which is the encoding ``parse_caid_reference_fasta`` reads back.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for r in reference:
            lab = np.asarray(r["labels"])
            chars = "".join(
                "-" if v == MASK else ("1" if v == 1 else "0") for v in lab
            )
            fh.write(f">{r['id']}\n{r['sequence']}\n{chars}\n")
    return path


def build_all(
    targets: list[dict],
    entries: Iterable[dict],
    out_dir: str,
    invalid_regions_path: Optional[str] = None,
) -> dict:
    """Build every reconstructable reference and report which are head-to-head."""
    by_id: dict[str, dict] = {}
    by_seq: dict[str, dict] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("disprot_id"):
            by_id[str(e["disprot_id"])] = e
        if e.get("sequence"):
            by_seq[e["sequence"]] = e
    for t in targets:                     # fall back to sequence identity
        if t["id"] not in by_id and t["sequence"] in by_seq:
            by_id[t["id"]] = by_seq[t["sequence"]]

    invalid = load_invalid_regions(invalid_regions_path)
    report: dict[str, dict] = {}
    for task in ("disorder_nox", "binding", "binding_idr", "linker"):
        ref = build_reference(
            targets, by_id, task,
            invalid_regions=invalid if task.startswith("binding") else None,
        )
        if not ref:
            continue
        path = write_caid_fasta(ref, os.path.join(out_dir, f"caid3_{task}.fasta"))
        v = validate(task, ref)
        v["path"] = path
        report[task] = v
    return report
