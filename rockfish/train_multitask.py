#!/usr/bin/env python3
"""Train one frozen-backbone head on every CAID3 task at once.

Why one model instead of five
-----------------------------
CAID3 is five benchmarks won by five specialists, and none answers another's
question. Coverage is the contribution here, not a single-benchmark record — on
Disorder-PDB a two-signal AlphaFold baseline (rsa + pLDDT, no training, no GPU)
already reaches 0.9581 against PUNCH2's 0.9550, so chasing that number head-on
is chasing something a shell script wins.

The other four are different. No AlphaFold baseline reaches the Disorder-NOX
top ten, because NOX calls unannotated residues *ordered* rather than ignoring
them, and the structural shortcut stops working the moment absence of evidence
is a negative. Binding-IDR's leader sits at 0.641.

Why a shared trunk should help rather than merely be cheap
----------------------------------------------------------
The small tasks are very small — linker has 15,683 positive residues and
binding 88,761, against disorder's 336,014. That is the regime where this
project's 1.96M-parameter head beat a 69.9M-parameter LoRA configuration by
+0.074 AUC. A shared trunk carries disorder's data into tasks with a twentieth
of it, and the per-task read-outs stay linear so none of them can grow private
capacity on 15k positives.

Rigor
-----
Splits are homology-clustered (BLASTp), the same machinery the disorder runs
use, and clustering is done once over the union of proteins so a protein
appearing in two tasks cannot land in different folds for each. Metrics are
computed only on residues a task actually evaluates.

Usage:
    python rockfish/train_multitask.py --workdir DIR [--tasks disorder_nox,linker,binding,binding_idr]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colab.caid_tasks import TASKS, build_task_dataset, task_statistics  # noqa: E402
from colab.lite_head import (  # noqa: E402
    WIDE_DILATIONS,
    MultiTaskLiteHead,
    freeze_backbone,
    masked_multitask_loss,
)

DEFAULT_TASKS = ("disorder_nox", "linker", "binding", "binding_idr")


def load_disprot(path: str) -> list[dict]:
    with open(path) as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else data.get("data", [])


def build_union(entries: list[dict], tasks: tuple[str, ...]) -> tuple[list[dict], dict]:
    """One row per protein, carrying a label/evidence vector for each task.

    Proteins are shared across tasks deliberately: that is what lets the trunk
    transfer disorder's data into linker and binding. Anything a task does not
    annotate stays masked for that task and contributes no gradient.
    """
    per_task = {t: {r["id"]: r for r in build_task_dataset(entries, t)} for t in tasks}
    ids: list[str] = []
    seen: set[str] = set()
    for t in tasks:
        for pid in per_task[t]:
            if pid not in seen:
                seen.add(pid)
                ids.append(pid)

    rows = []
    for pid in ids:
        base = next(per_task[t][pid] for t in tasks if pid in per_task[t])
        n = base["length"]
        labels, evidence = {}, {}
        for t in tasks:
            r = per_task[t].get(pid)
            if r is None:
                labels[t] = np.zeros(n, dtype=np.int8)
                evidence[t] = np.zeros(n, dtype=bool)
            else:
                labels[t] = np.asarray(r["labels"], dtype=np.int8)
                evidence[t] = np.asarray(r["label_evidence"], dtype=bool)
        rows.append({
            "id": pid,
            "sequence": base["sequence"],
            "length": n,
            "uniprot_acc": base.get("uniprot_acc"),
            "task_labels": labels,
            "task_evidence": evidence,
        })
    coverage = {
        t: sum(1 for r in rows if r["task_evidence"][t].any()) for t in tasks
    }
    return rows, coverage


def chunk_long_rows(rows: list[dict], max_len: int, stride: int | None = None,
                    min_len: int = 20) -> tuple[list[dict], dict]:
    """Keep proteins longer than the model window, as overlapping windows.

    They were being dropped: 2,440 of them on the pdb_missing source alone, 11%
    of the data, and precisely the hard ones. On CAID3 Disorder-NOX the targets
    the leader declines have a median length of 1292 against 344 overall, and
    the difficulty gap on them is +0.118 AUC. Throwing them out of training and
    then being asked to predict them is how a model ends up at chance above 1500
    residues while a training-free structural feature scores 0.897.

    Each window is an ordinary training row. Windows carry a ``parent`` so the
    fold assignment can keep them together — two windows of one protein in
    different folds is a near-duplicate across the split, which is exactly the
    leak homology clustering exists to prevent.
    """
    stride = stride or max_len // 2
    out, n_chunked, n_windows = [], 0, 0
    for r in rows:
        n = r["length"]
        if n <= max_len:
            out.append({**r, "parent": r.get("parent", r["id"])})
            continue
        n_chunked += 1
        starts = list(range(0, n - max_len + 1, stride))
        if starts[-1] + max_len < n:
            starts.append(n - max_len)
        for k, a in enumerate(starts):
            b = a + max_len
            if b - a < min_len:
                continue
            out.append({
                "id": f"{r['id']}#w{k}",
                "parent": r.get("parent", r["id"]),
                "sequence": r["sequence"][a:b],
                "length": b - a,
                "uniprot_acc": r.get("uniprot_acc"),
                "window_offset": a,
                "task_labels": {t: v[a:b] for t, v in r["task_labels"].items()},
                "task_evidence": {t: v[a:b] for t, v in r["task_evidence"].items()},
                # Structure is attached full-length before chunking, because
                # structure_features matches on the whole sequence. Slice it to
                # the same window or the head sees residue i's embedding beside
                # residue a+i's accessibility.
                **{k: r[k][a:b] for k in
                   ("rsa", "plddt", "contacts", "structure_available")
                   if k in r},
            })
            n_windows += 1
    return out, {"proteins_chunked": n_chunked, "windows_added": n_windows}


def load_pdb_missing_rows(cache_path: str, max_len: int, limit: int = 0) -> list[dict]:
    """Training rows for CAID3 Disorder-PDB's own label definition.

    Everything else here learns DisProt's curated *functional* disorder, while
    Disorder-PDB scores *crystallographic* disorder — residues unobserved in a
    structure. That mismatch is the most likely explanation for the transfer gap
    we measured: the structure-aware model gained on every DisProt CV task and
    then lost on three of four held-out CAID3 benchmarks.

    MobiDB's derived missing-residue annotation is that definition directly, over
    19,421 proteins rather than DisProt's 2,905, with observed regions supplying
    real negatives instead of masked ones.
    """
    from colab.label_sources import LabelSource, build_labelled_set

    records = []
    with open(cache_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit and len(records) >= limit:
                break

    labelled, stats = build_labelled_set(
        records, LabelSource.PDB_MISSING, min_len=20, max_len=max_len,
        min_evidence_fraction=0.10, min_disorder=3, min_order=3,
    )
    print(f"  pdb_missing: {stats['n_proteins']} proteins, "
          f"{stats['n_evidenced_residues']:,} evidenced residues, "
          f"disorder {stats['disorder_fraction_of_evidenced']:.1%}")

    rows = []
    for p in labelled:
        labels = p.labels.astype(np.int8)
        rows.append({
            "id": f"MB:{p.id}",
            "sequence": p.sequence,
            "length": p.length,
            "uniprot_acc": p.uniprot_acc,
            "task_labels": {"disorder_pdb": labels},
            "task_evidence": {"disorder_pdb": p.evidence.astype(bool)},
        })
    return rows


def merge_task_rows(base: list[dict], extra: list[dict],
                    all_tasks: tuple[str, ...]) -> list[dict]:
    """Union two row sets, filling absent tasks with empty evidence.

    A protein present in one source and not the other must contribute no
    gradient to the tasks it has no labels for — otherwise the 19k pdb_missing
    proteins would train the linker head on 19k fabricated negatives against its
    real 15,683 positives.
    """
    by_seq: dict[str, dict] = {}
    for row in base + extra:
        key = row["sequence"]
        if key not in by_seq:
            merged = dict(row)
            merged["task_labels"] = dict(row["task_labels"])
            merged["task_evidence"] = dict(row["task_evidence"])
            by_seq[key] = merged
        else:
            tgt = by_seq[key]
            for t, lab in row["task_labels"].items():
                if t not in tgt["task_labels"]:
                    tgt["task_labels"][t] = lab
                    tgt["task_evidence"][t] = row["task_evidence"][t]

    out = []
    for row in by_seq.values():
        n = row["length"]
        for t in all_tasks:
            if t not in row["task_labels"]:
                row["task_labels"][t] = np.zeros(n, dtype=np.int8)
                row["task_evidence"][t] = np.zeros(n, dtype=bool)
        out.append(row)
    return out


def attach_structure(rows: list[dict], cache_dir: str) -> None:
    """Attach cached AlphaFold rsa/pLDDT to each row, in place.

    Absence is recorded, not imputed. 109 of 2,340 training proteins have no
    AlphaFold entry, and filling those with zeros would tell the model they are
    fully buried — which reads as ordered, exactly backwards for a protein we
    know nothing about.
    """
    from colab.structure_rsa import structure_features

    for r in rows:
        n = r["length"]
        feats = None
        acc = r.get("uniprot_acc")
        if acc:
            feats = structure_features(acc, r["sequence"], cache_dir,
                                       allow_fetch=False)
        if feats is None:
            r["rsa"] = np.zeros(n, dtype=np.float32)
            r["plddt"] = np.zeros(n, dtype=np.float32)
            r["contacts"] = np.zeros(n, dtype=np.float32)
            r["structure_available"] = np.zeros(n, dtype=np.float32)
        else:
            r["rsa"] = np.asarray(feats["rsa"], dtype=np.float32)[:n]
            r["plddt"] = np.asarray(feats["plddt"], dtype=np.float32)[:n]
            r["contacts"] = np.asarray(
                feats.get("contacts", np.zeros(n)), dtype=np.float32)[:n]
            r["structure_available"] = np.ones(n, dtype=np.float32)
            # A structure shorter than the sequence leaves the tail unknown.
            if len(r["rsa"]) < n:
                pad = n - len(r["rsa"])
                r["rsa"] = np.concatenate([r["rsa"], np.zeros(pad, np.float32)])
                r["plddt"] = np.concatenate([r["plddt"], np.zeros(pad, np.float32)])
                r["contacts"] = np.concatenate(
                    [r["contacts"], np.zeros(pad, np.float32)])
                r["structure_available"][n - pad:] = 0.0



def structure_batch(batch, L, device):
    """(rsa, plddt, available, contacts) tensors, or all None."""
    if "rsa" not in batch[0]:
        return None, None, None, None
    import torch as _t
    rsa = _t.zeros(len(batch), L, device=device)
    pl = _t.zeros(len(batch), L, device=device)
    av = _t.zeros(len(batch), L, device=device)
    ct = _t.zeros(len(batch), L, device=device)
    for bi, r in enumerate(batch):
        n = min(r["length"], L)
        rsa[bi, :n] = _t.from_numpy(r["rsa"][:n]).to(device)
        pl[bi, :n] = _t.from_numpy(r["plddt"][:n]).to(device)
        av[bi, :n] = _t.from_numpy(r["structure_available"][:n]).to(device)
        ct[bi, :n] = _t.from_numpy(r["contacts"][:n]).to(device)
    return rsa, pl, av, ct


def drop_caid_targets(
    rows: list[dict], reference_fasta, min_identity: float,
    allow_missing_homology: bool = False,
) -> tuple[list[dict], dict]:
    """Remove benchmark targets and their homologues from the training union.

    Exact id/sequence matches are removed first and unconditionally — a CAID
    target is literally a DisProt entry, so without this the model trains on the
    proteins it will be scored on. Homologues above ``min_identity`` go too,
    since a 90%-identical paralogue leaks nearly as much as the target itself.

    ``reference_fasta`` may be one path or several. Several matters: filtering
    CAID3 alone leaves 307 of CAID2's 348 targets in the training set, because
    the two rounds share exactly one protein. A model filtered only against
    CAID3 therefore cannot be evaluated on CAID2 at all, and a held-out
    benchmark is the difference between one result and a replicated one.
    """
    from colab.caid3_eval import parse_caid_reference_fasta

    paths = ([reference_fasta] if isinstance(reference_fasta, (str, bytes))
             else list(reference_fasta or []))
    paths = [p for p in paths if p and os.path.isfile(p)]
    if not paths:
        return rows, {"n_before": len(rows), "n_removed": 0, "n_id_overlap": 0,
                      "reason": "no benchmark reference available"}

    targets, seen = [], set()
    for path in paths:
        for t in parse_caid_reference_fasta(path):
            if t["id"] in seen:
                continue
            seen.add(t["id"])
            targets.append(t)
    target_ids = {t["id"] for t in targets}
    target_seqs = {t["sequence"] for t in targets}

    kept, n_id = [], 0
    exact_removed = []
    for r in rows:
        if r["id"] in target_ids or r["sequence"] in target_seqs:
            n_id += 1
            exact_removed.append(r["id"])
            continue
        kept.append(r)

    # Homology pass: cluster the survivors against the targets and drop any
    # survivor sharing a cluster with one.
    try:
        from colab.homology_splits import blast_cross_identity_hits

        hits = blast_cross_identity_hits(
            [{"id": r["id"], "sequence": r["sequence"]} for r in kept],
            [{"id": t["id"], "sequence": t["sequence"]} for t in targets],
            min_identity=min_identity,
        )
        if hits is None:
            # BLAST unavailable. The pure-Python fallback cannot finish an
            # all-vs-all here, and silently keeping homologues would leak, so
            # say so rather than proceed as if the filter had run.
            raise RuntimeError("BLAST unavailable; cannot verify homology")
        # Returns (query_index, subject_index, identity) triples.
        homologous = {kept[q]["id"] for q, _s, _i in hits}
        kept = [r for r in kept if r["id"] not in homologous]
    except Exception as exc:                      # pragma: no cover - env dependent
        # A warning was not enough. One run lost its BLAST module and kept 75
        # homologues of CAID3 targets, which both inflated its benchmark
        # numbers and silently confounded a paired A/B against a run that had
        # removed them — two arms differing in the training set as well as the
        # architecture being compared. Fail instead, unless explicitly waived.
        if not allow_missing_homology:
            raise RuntimeError(
            f"homology filtering unavailable ({exc}). Exact id/sequence matches "
            "were removed, but homologues of CAID3 targets would remain in "
            "training, so benchmark numbers would not be leak-free and any "
            "paired comparison would differ in its training set. Load a BLAST "
            "module, or pass --allow-missing-homology-filter to accept that."
        ) from exc
        print(f"  WARNING: homology pass skipped ({exc}) and explicitly "
              "waived; homologues of CAID3 targets remain in training.")
        homologous = set()

    return kept, {
        "n_before": len(rows),
        "n_removed": len(rows) - len(kept),
        "n_id_overlap": n_id,
        "n_homology_removed": len(homologous),
        "min_identity": min_identity,
        "reference": paths if len(paths) > 1 else paths[0],
        "n_targets_filtered": len(targets),
    }


def homology_folds(rows: list[dict], n_folds: int, min_identity: float, seed: int):
    """Homology-clustered folds over the union, computed once for all tasks.

    Clustering per task would let the same protein sit in fold 2 for linker and
    fold 4 for binding, so a shared trunk would train on a protein it is later
    evaluated on.
    """
    from colab.homology_splits import cluster_proteins_by_homology_cached

    # Cluster one representative per parent protein. Windows of the same
    # protein are near-identical, so clustering them individually would burn
    # BLAST time to rediscover that, and any window landing in a different fold
    # from its siblings is a near-duplicate straddling the split.
    rep: dict[str, dict] = {}
    for r in rows:
        pid = r.get("parent", r["id"])
        if pid not in rep or r["length"] > rep[pid]["length"]:
            rep[pid] = {"id": pid, "sequence": r["sequence"],
                        "length": r["length"]}
    parent_ids = list(rep)
    proteins = [rep[p] for p in parent_ids]
    clusters, cluster_meta = cluster_proteins_by_homology_cached(
        proteins, min_identity=min_identity,
    )
    if cluster_meta.get("degenerate"):
        raise SystemExit(
            "Homology clustering produced fewer clusters than folds "
            f"({cluster_meta}). A degenerate split is not a homology split; "
            "refusing to report cross-validation numbers from it."
        )
    cluster_of_parent = {p: int(c) for p, c in zip(parent_ids, clusters)}
    by_cluster: dict[int, list[int]] = {}
    for idx, r in enumerate(rows):
        by_cluster.setdefault(
            cluster_of_parent[r.get("parent", r["id"])], []).append(idx)

    order = sorted(by_cluster.values(), key=len, reverse=True)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for members in order:                      # greedy: largest cluster first
        target = min(range(n_folds), key=lambda f: len(folds[f]))
        folds[target].extend(members)
    rng = np.random.default_rng(seed)
    for f in folds:
        rng.shuffle(f)
    return folds


@torch.no_grad()
def embed(esm, tokens, layer_ids):
    out = esm(tokens, repr_layers=layer_ids, return_contacts=False)
    return [out["representations"][i][:, 1:-1, :] for i in layer_ids]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--disprot", default=None, help="disprot_raw.json (default: workdir)")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--backbone", default="650M")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-identity", type=float, default=0.40)
    ap.add_argument("--fusion-layers", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=1022,
                    help="model window; longer proteins are kept as overlapping "
                         "windows unless --no-window-long-proteins")
    ap.add_argument("--no-private-trunk", action="store_true",
                    help="disable private trunk capacity for the binding tasks "
                         "(the ablation; it is on by default). Without it the "
                         "binding losses shape the shared trunk, which is what "
                         "broke three floors in the protein-bias run.")
    ap.add_argument("--no-protein-bias", action="store_true",
                    help="disable the per-protein bias term (the ablation; it "
                         "is on by default). 96.7%% to 99.7%% of the pairs CAID's "
                         "pooled AUC counts are between proteins, and a purely "
                         "local model has no mechanism for them.")
    ap.add_argument("--no-condition-binding", action="store_true",
                    help="disable the disorder-conditioned binding read-out "
                         "(the ablation arm; conditioning is on by default)")
    ap.add_argument("--no-window-long-proteins", action="store_true",
                    help="drop proteins longer than --max-len instead of "
                         "windowing them (the old behaviour, which discarded "
                         "2,440 of them and left the model at chance above "
                         "1500 residues)")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument(
        "--caid-reference", default=None, action="append",
        help="Benchmark FASTA whose targets must be excluded from training. "
             "Repeatable, and repeating it matters: filtering CAID3 alone "
             "leaves 307 of CAID2's 348 targets in the training set, since the "
             "rounds share one protein.",
    )
    ap.add_argument("--leak-identity", type=float, default=0.40)
    ap.add_argument(
        "--allow-missing-homology-filter", action="store_true",
        help="Proceed when BLAST is unavailable. Exact CAID3 targets are still "
             "removed, but their homologues are not, so benchmark numbers from "
             "such a run are not leak-free and must not be compared against a "
             "run that filtered them.",
    )
    ap.add_argument(
        "--no-caid-filter", action="store_true",
        help="Train on CAID3 targets too. Only for measuring the size of the "
             "leak; any benchmark number from such a run is invalid.",
    )
    ap.add_argument(
        "--final-model", action="store_true",
        help="After CV, fit one head on all (filtered) proteins and save it for "
             "benchmark scoring.",
    )
    ap.add_argument(
        "--pdb-missing-cache", default=None,
        help="MobiDB ndjson (mobidb_pdbcov.ndjson) adding a disorder_pdb task "
             "trained on CAID3 Disorder-PDB's own missing-residue definition "
             "rather than DisProt's curated functional disorder.",
    )
    ap.add_argument("--pdb-missing-limit", type=int, default=0)
    ap.add_argument(
        "--wide-receptive-field", action="store_true",
        help="Dilations 1/4/16/32 give a 213-residue field instead of 61, for "
             "the same parameter count. IDRs frequently run past 100.",
    )
    ap.add_argument(
        "--structure-dim", type=int, default=0,
        help="Width of the AlphaFold rsa/pLDDT input block (0 = sequence only). "
             "Structure as an INPUT, not a post-hoc ensemble: fusing a trained "
             "model with rsa+pLDDT made the baseline worse (0.9581 -> 0.9554), "
             "while the top CAID3 methods are structure-aware.",
    )
    ap.add_argument(
        "--structure-cache",
        default="/scratch4/sfried3/jbeale3_disordernet/af_structures",
        help="Directory of cached AlphaFold mmCIF files.",
    )
    args = ap.parse_args(argv)

    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    bad = [t for t in tasks if t not in TASKS]
    if bad:
        print(f"ERROR: unknown task(s) {bad}; choose from {list(TASKS)}", file=sys.stderr)
        return 2

    os.makedirs(args.workdir, exist_ok=True)
    disprot = args.disprot or os.path.join(args.workdir, "disprot_raw.json")
    if not args.caid_reference:
        args.caid_reference = [
            c for c in (os.path.join(args.workdir, "caid3_disorder_pdb.fasta"),
                        os.path.join(args.workdir, "checkpoints",
                                     "caid3_disorder_pdb.fasta"))
            if os.path.isfile(c)][:1]
    if args.caid_reference:
        print(f"leak filter references ({len(args.caid_reference)}):")
        for c in args.caid_reference:
            print(f"    {c}")
    entries = load_disprot(disprot)
    print(f"DisProt entries: {len(entries):,}")

    stats = task_statistics(entries)
    print(f"\n{'task':<16}{'proteins':>9}{'eval res':>12}{'positives':>11}{'prev':>8}")
    for t in tasks:
        s = stats[t]
        print(f"{t:<16}{s['proteins']:>9}{s['evaluated_residues']:>12,}"
              f"{s['positives']:>11,}{s['prevalence']:>8.1%}")

    if len(set(tasks)) != len(tasks):
        dupes = sorted({t for t in tasks if list(tasks).count(t) > 1})
        raise SystemExit(
            f"duplicate task(s) {dupes} in --tasks. disorder_pdb is appended "
            f"from --pdb-missing-cache and must not also be listed: the "
            f"DisProt-derived version of it is degenerate."
        )

    rows, coverage = build_union(entries, tasks)
    keep_long = not args.no_window_long_proteins
    if keep_long:
        rows = [r for r in rows if r["length"] >= 20]
    else:
        rows = [r for r in rows if 20 <= r["length"] <= args.max_len]
    print(f"\nunion: {len(rows):,} proteins within length bounds")

    if args.pdb_missing_cache and os.path.isfile(args.pdb_missing_cache):
        print("adding disorder_pdb task from MobiDB missing-residue labels…")
        extra = load_pdb_missing_rows(
            args.pdb_missing_cache,
            10 ** 6 if keep_long else args.max_len,
            args.pdb_missing_limit)
        tasks = tasks + ("disorder_pdb",)
        rows = merge_task_rows(rows, extra, tasks)
        coverage["disorder_pdb"] = sum(
            1 for r in rows if r["task_evidence"]["disorder_pdb"].any())
        print(f"  union now {len(rows):,} proteins across {len(tasks)} tasks")

    if args.structure_dim:
        attach_structure(rows, args.structure_cache)

    if keep_long:
        rows, chunk_stats = chunk_long_rows(rows, args.max_len)
        print(f"long proteins kept as windows: "
              f"{chunk_stats['proteins_chunked']:,} proteins -> "
              f"{chunk_stats['windows_added']:,} windows "
              f"(union now {len(rows):,} rows)")
        have = sum(1 for r in rows if r.get("structure_available") is not None
                   and r["structure_available"].any())
        print(f"structural channels: {have}/{len(rows)} proteins have an "
              f"AlphaFold entry ({have/max(len(rows),1):.1%})")
        if have == 0:
            print("  ERROR: no cached structures found — run the structure "
                  "fetch first, or the gate trains on a constant block",
                  file=sys.stderr)
            return 2
    print(f"coverage: { {t: coverage[t] for t in tasks} }")

    # The CAID3 targets ARE DisProt entries, and this cache contains them.
    # Training on the union without filtering means training on the evaluation
    # set — every CAID3 number afterwards would measure memorisation. The
    # disorder pipeline has always applied this filter; the multi-task trainer
    # did not until it was caught, mid-run, by noticing the evaluator would have
    # scored a model trained on its own targets.
    # --stats-only trains nothing, so requiring BLAST there would block a
    # read-only coverage check on any machine without the module.
    if args.stats_only:
        leak = {"skipped": "stats-only"}
    elif not args.no_caid_filter:
        rows, leak = drop_caid_targets(
            rows, args.caid_reference, args.leak_identity,
            allow_missing_homology=args.allow_missing_homology_filter)
        print(f"CAID leak-free: removed {leak['n_removed']} / {leak['n_before']} "
              f"proteins at identity>={args.leak_identity} "
              f"({leak['n_id_overlap']} exact id/sequence hits)")
        if leak["n_removed"] == 0:
            print("  WARNING: nothing removed — is the CAID3 reference resolvable?")
    else:
        print("CAID leak-free filter DISABLED — results are not benchmark-valid")

    if args.stats_only:
        return 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from colab.disordernet_gpu import setup_environment, TrainConfig
    cfg = TrainConfig.from_profile("lite", esm_backbone=args.backbone)
    cfg = setup_environment(cfg)
    device = cfg.device

    from colab.esm_backbone import load_esm_backbone
    esm, _alphabet, batch_converter, spec = load_esm_backbone(
        device, backbone=args.backbone, use_gradient_checkpointing=False,
    )
    n_frozen = freeze_backbone(esm)
    n_layers = len(esm.layers)
    layer_ids = list(range(n_layers - min(args.fusion_layers, n_layers), n_layers))
    print(f"\nbackbone {args.backbone}: {n_frozen} tensors frozen, "
          f"mixing layers {layer_ids[0]}-{layer_ids[-1]}")

    for t in tasks:
        ev = np.concatenate([r["task_evidence"][t] for r in rows])
        lb = np.concatenate([r["task_labels"][t] for r in rows])
        if not ev.any():
            continue
        pos = float(lb[ev].mean())
        if pos >= 0.999 or pos <= 0.001:
            raise SystemExit(
                f"task {t!r} has prevalence {pos:.3%} over {int(ev.sum()):,} "
                f"evidenced residues — effectively one class. A task with no "
                f"negatives contributes a constant gradient and its AUC is "
                f"undefined; it must not be trained or reported."
            )

    folds = homology_folds(rows, args.n_folds, args.min_identity, args.seed)
    print(f"homology folds: {[len(f) for f in folds]}")

    dim = getattr(spec, "embed_dim", getattr(cfg, "esm_embed_dim", 1280))
    results: dict[str, list] = {t: [] for t in tasks}
    t0 = time.time()

    for fold_idx in range(args.n_folds):
        val_idx = set(folds[fold_idx])
        train_rows = [r for i, r in enumerate(rows) if i not in val_idx]
        val_rows = [rows[i] for i in sorted(val_idx)]

        from colab.lite_head import ScalarMix
        mix = ScalarMix(len(layer_ids)).to(device)
        head = MultiTaskLiteHead(in_dim=dim, tasks=tasks,
                                 dropout=cfg.head_dropout,
                                 structure_dim=args.structure_dim,
                                 condition_binding=not args.no_condition_binding,
                                 protein_bias=not args.no_protein_bias,
                                 private_trunk=not args.no_private_trunk,
                                 dilations=(WIDE_DILATIONS
                                            if args.wide_receptive_field else None)).to(device)
        params = list(head.parameters()) + list(mix.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=cfg.weight_decay)
        print(f"\n── fold {fold_idx+1}/{args.n_folds}  train={len(train_rows)} "
              f"val={len(val_rows)}  trainable={sum(p.numel() for p in params):,}")

        for epoch in range(args.epochs):
            head.train()
            perm = np.random.permutation(len(train_rows))
            tot, nb = 0.0, 0
            for s in range(0, len(perm), args.batch_size):
                batch = [train_rows[i] for i in perm[s:s + args.batch_size]]
                data = [(r["id"], r["sequence"]) for r in batch]
                _, _, tokens = batch_converter(data)
                tokens = tokens.to(device)
                feats = mix(embed(esm, tokens, layer_ids))
                L = feats.shape[1]
                sr, sp, sa, sc = structure_batch(batch, L, device)
                logits = head(feats, rsa=sr, plddt=sp, structure_available=sa, contacts=sc)
                lab, ev = {}, {}
                for t in tasks:
                    lab[t] = torch.zeros(len(batch), L, device=device)
                    ev[t] = torch.zeros(len(batch), L, dtype=torch.bool, device=device)
                    for bi, r in enumerate(batch):
                        n = min(r["length"], L)
                        lab[t][bi, :n] = torch.from_numpy(
                            r["task_labels"][t][:n].astype(np.float32)).to(device)
                        ev[t][bi, :n] = torch.from_numpy(
                            r["task_evidence"][t][:n]).to(device)
                try:
                    loss, _ = masked_multitask_loss(logits, lab, ev)
                except ValueError:
                    continue                     # batch had no evaluated residue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                tot += float(loss.detach()); nb += 1
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                print(f"   epoch {epoch+1:>3}/{args.epochs}  loss={tot/max(nb,1):.4f}"
                      f"  [{(time.time()-t0)/60:.1f}m]", flush=True)

        head.eval()
        pooled = {t: ([], []) for t in tasks}
        with torch.no_grad():
            for s in range(0, len(val_rows), args.batch_size):
                batch = val_rows[s:s + args.batch_size]
                data = [(r["id"], r["sequence"]) for r in batch]
                _, _, tokens = batch_converter(data)
                tokens = tokens.to(device)
                feats = mix(embed(esm, tokens, layer_ids))
                sr, sp, sa, sc = structure_batch(batch, feats.shape[1], device)
                logits = head(feats, rsa=sr, plddt=sp, structure_available=sa, contacts=sc)
                for t in tasks:
                    p = torch.sigmoid(logits[t]).float().cpu().numpy()
                    for bi, r in enumerate(batch):
                        n = min(r["length"], p.shape[1])
                        m = r["task_evidence"][t][:n]
                        if not m.any():
                            continue
                        pooled[t][0].append(r["task_labels"][t][:n][m])
                        pooled[t][1].append(p[bi, :n][m])

        from sklearn.metrics import average_precision_score, roc_auc_score
        for t in tasks:
            if not pooled[t][0]:
                continue
            y = np.concatenate(pooled[t][0]); s_ = np.concatenate(pooled[t][1])
            if len(np.unique(y)) < 2:
                continue
            auc = roc_auc_score(y, s_); aps = average_precision_score(y, s_)
            results[t].append({"fold": fold_idx + 1, "auc": float(auc),
                               "aps": float(aps), "n": int(len(y))})
            print(f"   {t:<14} AUC={auc:.4f}  APS={aps:.4f}  n={len(y):,}")

    print(f"\n{'='*60}\n MULTI-TASK RESULTS  ({(time.time()-t0)/60:.1f} min)\n{'='*60}")
    summary = {}
    for t in tasks:
        if not results[t]:
            continue
        aucs = [r["auc"] for r in results[t]]
        apss = [r["aps"] for r in results[t]]
        summary[t] = {
            "mean_auc": float(np.mean(aucs)), "sd_auc": float(np.std(aucs, ddof=1))
            if len(aucs) > 1 else 0.0,
            "mean_aps": float(np.mean(apss)), "folds": results[t],
        }
        print(f"  {t:<14} AUC={np.mean(aucs):.4f} ± {np.std(aucs, ddof=1) if len(aucs)>1 else 0:.4f}"
              f"   APS={np.mean(apss):.4f}")

    out = {
        "tasks": list(tasks), "n_folds": args.n_folds, "epochs": args.epochs,
        "backbone": args.backbone, "split": "homology", "min_identity": args.min_identity,
        "summary": summary, "coverage": coverage,
        "caid3_reference": {
            "disorder_nox": {"leader": "ESMDisPred-2PDB", "auc": 0.885, "aps": 0.754},
            "linker": {"leader": "IPA-AF2-Linker", "auc": 0.897, "aps": 0.474},
            "binding": {"leader": "DisoFLAG-PB", "auc": 0.776, "aps": 0.245},
            "binding_idr": {"leader": "bindEmbed21IDR-rawGeneral", "auc": 0.641, "aps": 0.514},
        },
        "note": (
            "DisProt homology-split CV, not the CAID3 benchmark. Comparable to "
            "the published leaders only in magnitude; the CAID3 targets are a "
            "held-out set with their own composition."
        ),
    }
    out["caid_leak_filter"] = None if args.no_caid_filter else leak
    path = os.path.join(args.workdir, "multitask_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {path}")

    if args.final_model:
        # Benchmark scoring needs one model, not five fold models. Fit on every
        # protein that survived the CAID filter — the targets and their
        # homologues are already gone, so this uses no benchmark information.
        print(f"\nfitting final head on all {len(rows)} filtered proteins…")
        from colab.lite_head import ScalarMix

        mix = ScalarMix(len(layer_ids)).to(device)
        head = MultiTaskLiteHead(in_dim=dim, tasks=tasks,
                                 dropout=cfg.head_dropout,
                                 structure_dim=args.structure_dim,
                                 condition_binding=not args.no_condition_binding,
                                 protein_bias=not args.no_protein_bias,
                                 private_trunk=not args.no_private_trunk,
                                 dilations=(WIDE_DILATIONS
                                            if args.wide_receptive_field else None)).to(device)
        params = list(head.parameters()) + list(mix.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=cfg.weight_decay)
        for epoch in range(args.epochs):
            head.train()
            perm = np.random.permutation(len(rows))
            tot, nb = 0.0, 0
            for s in range(0, len(perm), args.batch_size):
                batch = [rows[i] for i in perm[s:s + args.batch_size]]
                _, _, tokens = batch_converter(
                    [(r["id"], r["sequence"]) for r in batch])
                tokens = tokens.to(device)
                feats = mix(embed(esm, tokens, layer_ids))
                L = feats.shape[1]
                sr, sp, sa, sc = structure_batch(batch, L, device)
                logits = head(feats, rsa=sr, plddt=sp, structure_available=sa, contacts=sc)
                lab, ev = {}, {}
                for t in tasks:
                    lab[t] = torch.zeros(len(batch), L, device=device)
                    ev[t] = torch.zeros(len(batch), L, dtype=torch.bool, device=device)
                    for bi, r in enumerate(batch):
                        n = min(r["length"], L)
                        lab[t][bi, :n] = torch.from_numpy(
                            r["task_labels"][t][:n].astype(np.float32)).to(device)
                        ev[t][bi, :n] = torch.from_numpy(
                            r["task_evidence"][t][:n]).to(device)
                try:
                    loss, _ = masked_multitask_loss(logits, lab, ev)
                except ValueError:
                    continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                tot += float(loss.detach()); nb += 1
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                print(f"   epoch {epoch+1:>3}/{args.epochs}  loss={tot/max(nb,1):.4f}",
                      flush=True)

        from colab.compact_checkpoint import atomic_torch_save
        ckpt = os.path.join(args.workdir, "multitask_head.pt")
        atomic_torch_save({
            "head": head.state_dict(),
            "mix": mix.state_dict(),
            "tasks": list(tasks),
            "layer_ids": layer_ids,
            "backbone": args.backbone,
            "embed_dim": dim,
            "structure_dim": args.structure_dim,
            "wide_receptive_field": args.wide_receptive_field,
            # Recorded so the evaluator builds the same architecture. Absent in
            # checkpoints trained before conditioning existed, and the evaluator
            # defaults it to False for exactly that reason — a head that creates
            # cond parameters cannot strict-load a checkpoint that has none, and
            # the model of record holds three first places.
            "condition_binding": not args.no_condition_binding,
            "protein_bias": not args.no_protein_bias,
            "private_trunk": not args.no_private_trunk,
            "n_train_proteins": len(rows),
            "caid_leak_filter": None if args.no_caid_filter else leak,
        }, ckpt)
        print(f"Wrote {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
