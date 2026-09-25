#!/usr/bin/env python3
"""Is backbone handedness worth a channel? Training-free, measured not asserted.

Every structural input DisorderNet reads is mirror-invariant. Reflect a protein
and its solvent accessibility, its contact counts and AlphaFold's confidence in
it are unchanged; the model cannot tell a structure from its reflection. The
physics is not mirror-invariant — L-amino acids build right-handed alpha-helices
and the conformation that dominates disordered chains, polyproline II, is
left-handed — so there is a real quantity here that the model is blind to.

The standard experimental assay for disorder is itself a chirality measurement,
far-UV circular dichroism. The accompanying Lean development proves it cannot
deliver the distinction wanted: the PPII and statistical-coil basis spectra are
nearly identical, so the PPII/coil split in a deconvolution is free up to the
noise (`Dichroism.equal_basis_split_free`, `near_degenerate_tolerance`). If the
handedness of a disordered region is wanted it has to come from geometry.

**The sign is fixed on training proteins, never on the benchmark.** Which way
handedness runs with disorder is not something to read off CAID3 and then
report as a finding — that is choosing a free parameter on the test set. It is
chosen here from DisProt entries that survived the CAID leak filter, and only
then applied.

    export ANALYSIS_SCRIPT=results/caid3/chirality_probe.py
    sbatch rockfish/slurm/analysis_cpu.sbatch
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.auc_decomposition import decompose_auc  # noqa: E402
from colab.caid3_official import (  # noqa: E402
    TASKS,
    evaluated_mask,
    read_reference,
    verify_composition_for,
)
from colab.structure_rsa import smooth_window, structure_features  # noqa: E402

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("CHIR_REFS", f"{ROOT}/caid3_official")
STRUCTURES = os.environ.get("CHIR_STRUCTURES", f"{ROOT}/af_structures")
DISPROT = os.environ.get("CHIR_DISPROT",
                         os.path.expanduser("~/.cache/disordernet/disprot_raw.json"))
PDB_MISSING = os.environ.get(
    "CHIR_PDB_MISSING",
    os.path.expanduser("~/.cache/disordernet/mobidb_pdbcov.ndjson"))
OUT = os.environ.get("CHIR_OUT", "")
N_TRAIN = int(os.environ.get("CHIR_NTRAIN", "600"))


def accessions() -> dict[str, str]:
    if not os.path.isfile(DISPROT):
        return {}
    with open(DISPROT) as fh:
        data = json.load(fh)
    entries = data if isinstance(data, list) else data.get("data", [])
    return {str(e["disprot_id"]): e["acc"] for e in entries
            if isinstance(e, dict) and e.get("disprot_id") and e.get("acc")}


def channels(acc: str, seq: str, window: int) -> dict | None:
    """Handedness and its achiral control, aligned to ``seq`` or nothing.

    ``structure_features`` returns None unless the structure's sequence matches
    the target exactly, which is the alignment guard the rest of the project
    relies on and is not relaxed here.
    """
    feats = structure_features(acc, seq, STRUCTURES, window=window,
                               allow_fetch=False)
    if feats is None or "handedness" not in feats:
        return None
    h = np.asarray(feats["handedness"], dtype=np.float64)
    if not np.isfinite(h).any():
        return None
    filled = np.nan_to_num(h, nan=0.0)
    return {
        # Signed, smoothed: handedness is a regional property in the same way
        # accessibility is, and the raw per-residue value is as noisy.
        "handedness": smooth_window(filled, window),
        # The achiral control. |sin| discards the sign and keeps everything
        # else, so any advantage the signed channel has over this one is
        # attributable to handedness rather than to torsion magnitude.
        "abs_handedness": smooth_window(np.abs(filled), window),
        "rsa": np.asarray(feats["rsa"], dtype=np.float64),
        "valid": np.isfinite(h),
    }


def fit_sign(window: int) -> dict:
    """Which way handedness runs with disorder, decided on training data.

    Uses the MobiDB missing-residue set — the same source the trainer uses, and
    the one whose proteins carry UniProt accessions and therefore AlphaFold
    structures — with every CAID3 target and every CAID3 sequence removed. The
    quantity fitted is a single bit, the sign, and it is reported with the
    effect it was fitted from so a reader can see whether it was marginal.
    """
    from colab.label_sources import LabelSource, build_labelled_set

    caid_ids, caid_seqs = set(), set()
    for task in TASKS:
        path = os.path.join(REFS, f"{task}.fasta")
        if os.path.isfile(path):
            ref = read_reference(path)
            caid_ids |= set(ref)
            caid_seqs |= {seq for seq, _lab in ref.values()}

    records = []
    with open(PDB_MISSING) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(records) >= 40 * N_TRAIN:
                break
    labelled, _stats = build_labelled_set(
        records, LabelSource.PDB_MISSING, min_len=20, max_len=1022,
        min_evidence_fraction=0.10, min_disorder=3, min_order=3, verbose=False)

    pos, neg, used, n_excluded = [], [], 0, 0
    for p in labelled:
        if used >= N_TRAIN:
            break
        if p.id in caid_ids or p.sequence in caid_seqs:
            n_excluded += 1
            continue
        if not p.uniprot_acc:
            continue
        ch = channels(p.uniprot_acc, p.sequence, window)
        if ch is None:
            continue
        lab = p.labels.astype(np.int8)
        ev = p.evidence.astype(bool) & ch["valid"]
        if not ev.any() or len(np.unique(lab[ev])) < 2:
            continue
        h = ch["handedness"]
        pos.append(h[ev & (lab == 1)])
        neg.append(h[ev & (lab == 0)])
        used += 1

    if used < 20:
        return {"sign": 1.0, "n_train_proteins": used,
                "n_caid_excluded": n_excluded,
                "reason": "too few training proteins with a matching structure"}
    p_mean = float(np.mean(np.concatenate(pos)))
    n_mean = float(np.mean(np.concatenate(neg)))
    return {"sign": 1.0 if p_mean > n_mean else -1.0,
            "n_train_proteins": used, "n_caid_excluded": n_excluded,
            "mean_handedness_disordered": p_mean,
            "mean_handedness_ordered": n_mean,
            "separation": p_mean - n_mean}


def main() -> int:
    window = int(os.environ.get("CHIR_WINDOW", "21"))
    acc_of = accessions()

    print(f"fitting the sign on up to {N_TRAIN} MobiDB proteins, "
          f"CAID3 targets and sequences excluded…")
    fit = fit_sign(window)
    sign = float(fit["sign"])
    print(json.dumps(fit, indent=2, default=float))
    if "reason" in fit:
        print("  sign not established on training data — results below are "
              "exploratory in direction as well as effect", file=sys.stderr)

    report = {"window": window, "sign_fit": fit, "tasks": {}}
    for task in TASKS:
        path = os.path.join(REFS, f"{task}.fasta")
        if not os.path.isfile(path):
            continue
        verify_composition_for("caid3", task, path)
        ref = read_reference(path)

        got = {k: ([], []) for k in ("handedness", "abs_handedness", "rsa")}
        n_struct = 0
        for tid, (seq, lab) in ref.items():
            acc = acc_of.get(tid, tid)
            ch = channels(acc, seq, window)
            if ch is None:
                continue
            m = evaluated_mask(lab) & ch["valid"]
            if not m.any():
                continue
            y = np.frombuffer(lab.encode(), dtype=np.uint8)[m].astype(np.int8) \
                - ord("0")
            if len(np.unique(y)) < 2:
                continue
            n_struct += 1
            for key, flip in (("handedness", sign), ("abs_handedness", 1.0),
                              ("rsa", 1.0)):
                got[key][0].append(y)
                got[key][1].append(flip * ch[key][m])

        print(f"\n{'=' * 84}\n {task}: {n_struct}/{len(ref)} targets with a "
              f"matching AlphaFold structure and both classes\n{'=' * 84}")
        if n_struct < 5:
            print(" too few to measure")
            continue
        row = {}
        print(f" {'channel':<18}{'pooled':>9}{'within':>9}{'between':>10}"
              f"{'1p1v':>9}")
        for key in ("handedness", "abs_handedness", "rsa"):
            d = decompose_auc(got[key][0], got[key][1])
            if d.get("pooled") is None:
                continue
            row[key] = d
            print(f" {key:<18}{d['pooled']:>9.4f}{d['auc_within']:>9.4f}"
                  f"{d['auc_between']:>10.4f}"
                  f"{(d['auc_within_unweighted'] or float('nan')):>9.4f}")
        report["tasks"][task] = {
            "n_targets": n_struct,
            **{k: {kk: vv for kk, vv in v.items() if kk != "reason"}
               for k, v in row.items()},
        }
        h, a = row.get("handedness"), row.get("abs_handedness")
        if h and a:
            print(f"\n signed minus achiral control: "
                  f"within {h['auc_within'] - a['auc_within']:+.4f}, "
                  f"pooled {h['pooled'] - a['pooled']:+.4f}")
            print(" (a positive within-protein difference is the only thing "
                  "here that\n  handedness explains and torsion magnitude does "
                  "not)")

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
