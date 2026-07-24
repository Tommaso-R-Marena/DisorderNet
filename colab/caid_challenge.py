"""CAID challenge suite: CAID3 scoring + CAID4 blind submission + efficiency.

CAID4 (2026) is a *blind* challenge until preliminary results (~Dec 2026): labels
are not public, so we export official ``.caid`` submissions + ``timings.csv`` and
record computational efficiency. When a labeled reference becomes available,
the same evaluate path used for CAID3 applies.

CAID3 Disorder-PDB remains the primary scored credibility floor (vs ESMDisPred).
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np

from colab.caid3_eval import (
    CAID3_DISORDER_PDB_URL,
    evaluate_caid_predictions,
    export_caid_predictions_dir,
    fetch_caid3_reference,
    parse_caid_reference_fasta,
    print_caid3_eval_report,
    save_caid3_eval_report,
    write_caid_prediction_file,
)

# Additional public CAID3 references when present upstream / locally.
CAID3_EXTRA_URLS: dict[str, str] = {
    # Same CAID repo demo-data layout; 404s are tolerated (optional tracks).
    "disorder_nox": (
        "https://raw.githubusercontent.com/BioComputingUP/CAID/master/demo-data/"
        "references/disorder_nox.fasta"
    ),
}


def _try_fetch(url: str, cache_path: str) -> Optional[str]:
    if os.path.isfile(cache_path):
        return cache_path
    try:
        import requests

        resp = requests.get(url, timeout=120)
        if resp.status_code != 200 or len(resp.text) < 100:
            return None
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(resp.text)
        return cache_path
    except Exception:
        return None


def resolve_caid3_references(checkpoint_dir: str, explicit: Optional[str] = None) -> dict[str, str]:
    """Return mapping track → local FASTA path for available CAID3 refs."""
    out: dict[str, str] = {}
    if explicit and os.path.isfile(explicit):
        out["disorder_pdb"] = explicit
    else:
        out["disorder_pdb"] = fetch_caid3_reference(
            os.path.join(checkpoint_dir, "caid3_disorder_pdb.fasta"),
        )
    for name, url in CAID3_EXTRA_URLS.items():
        path = _try_fetch(url, os.path.join(checkpoint_dir, f"caid3_{name}.fasta"))
        if path:
            out[name] = path
    return out


def resolve_caid4_targets(checkpoint_dir: str, explicit: Optional[str] = None) -> Optional[str]:
    """Locate CAID4 *blind* target FASTA (sequences only; no labels yet).

    Set ``CAID4_TARGETS`` / ``--caid4-targets`` to the organizer FASTA once obtained
    from the CAID portal. Until then returns None and the suite records skipped.
    """
    candidates = [
        explicit,
        os.environ.get("CAID4_TARGETS"),
        os.path.join(checkpoint_dir, "caid4_targets.fasta"),
        os.path.expanduser("~/DisorderNet/data/caid4_targets.fasta"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def parse_caid_targets_fasta(path: str) -> list[dict]:
    """Parse sequence-only or labeled CAID FASTA into protein dicts."""
    # Prefer labeled parser; fall back to sequence-only for blind CAID4.
    try:
        labeled = parse_caid_reference_fasta(path)
        if labeled:
            return labeled
    except Exception:
        pass
    proteins: list[dict] = []
    with open(path) as f:
        name, buf = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None and buf:
                    seq = "".join(buf).upper()
                    proteins.append({
                        "id": name,
                        "sequence": seq,
                        "length": len(seq),
                        "labels": [],
                        "eval_mask": [True] * len(seq),
                        "caid_header": name,
                    })
                name = line[1:].split()[0]
                buf = []
            else:
                if set(line) <= set("01-"):
                    continue  # skip label lines without crashing
                buf.append(line)
        if name is not None and buf:
            seq = "".join(buf).upper()
            proteins.append({
                "id": name,
                "sequence": seq,
                "length": len(seq),
                "labels": [],
                "eval_mask": [True] * len(seq),
                "caid_header": name,
            })
    return proteins


def write_timings_csv(path: str, rows: list[tuple[str, float]], header_note: str = "") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        if header_note:
            f.write(f"# {header_note}\n")
        f.write("sequence,milliseconds\n")
        for sid, ms in rows:
            f.write(f"{sid},{ms:.3f}\n")


def predict_with_timings(
    proteins: list[dict],
    *,
    predict_fn,
) -> tuple[dict[str, np.ndarray], list[tuple[str, float]], dict]:
    """Call ``predict_fn(sequence) -> scores`` per protein; record ms timings."""
    preds: dict[str, np.ndarray] = {}
    timings: list[tuple[str, float]] = []
    t_all = time.perf_counter()
    for p in proteins:
        t0 = time.perf_counter()
        scores = predict_fn(p["sequence"], p["id"])
        ms = (time.perf_counter() - t0) * 1000.0
        preds[p["id"]] = np.asarray(scores, dtype=np.float32)
        timings.append((p["id"], ms))
    wall_s = time.perf_counter() - t_all
    n_res = sum(len(p["sequence"]) for p in proteins) or 1
    eff = {
        "n_proteins": len(proteins),
        "n_residues": n_res,
        "wall_seconds": wall_s,
        "ms_per_protein_mean": float(np.mean([t for _, t in timings])) if timings else 0.0,
        "ms_per_1000_residues": (wall_s * 1000.0) / (n_res / 1000.0),
    }
    return preds, timings, eff


def run_caid_challenge_suite(
    *,
    checkpoint_dir: str,
    train_proteins: list[dict],
    preds_factory,
    caid3_reference: Optional[str] = None,
    caid4_targets: Optional[str] = None,
    leak_free_min_identity: float = 0.40,
    skip_tta: bool = False,
) -> dict:
    """Full suite report: leakage audit + CAID3 tracks + optional CAID4 export.

    ``preds_factory(fasta_path) -> dict[id, scores]`` should run the fold-soup model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    report: dict = {
        "suite": "CAID3+CAID4",
        "caid3": {},
        "caid4": {},
        "leakage_audit": {},
        "efficiency": {},
    }

    refs = resolve_caid3_references(checkpoint_dir, caid3_reference)
    caid_for_audit = []
    for track, path in refs.items():
        caid_for_audit.extend(parse_caid_reference_fasta(path))

    from colab.caid_leakage import audit_train_vs_caid, save_leakage_audit

    audit = audit_train_vs_caid(
        train_proteins, caid_for_audit, min_identity=leak_free_min_identity,
    )
    audit_path = os.path.join(checkpoint_dir, "caid_leakage_audit.json")
    save_leakage_audit(audit, audit_path)
    report["leakage_audit"] = {
        "path": audit_path,
        "leak_free": audit.get("leak_free"),
        "n_id_overlap": audit.get("n_id_overlap"),
        "n_homology_hits": audit.get("n_homology_hits"),
        "n_train_flagged_for_exclusion": audit.get("n_train_flagged_for_exclusion"),
        "disclaimer": audit.get("disclaimer"),
    }

    # Primary scored track: Disorder-PDB
    pdb_path = refs.get("disorder_pdb")
    if pdb_path:
        ref_proteins = parse_caid_reference_fasta(pdb_path)
        tmp_fasta = os.path.join(checkpoint_dir, "_caid3_query.fasta")
        with open(tmp_fasta, "w") as f:
            for p in ref_proteins:
                f.write(f">{p['id']}\n{p['sequence']}\n")
        t0 = time.perf_counter()
        preds = preds_factory(tmp_fasta)
        infer_s = time.perf_counter() - t0
        out_dir = os.path.join(checkpoint_dir, "caid3_submission")
        export_caid_predictions_dir(ref_proteins, preds, out_dir)
        caid3_rep = evaluate_caid_predictions(ref_proteins, preds)
        caid3_rep["inference_wall_seconds"] = infer_s
        caid3_rep["reference_path"] = pdb_path
        caid3_rep["leakage_audit_path"] = audit_path
        caid3_rep["train_vs_caid_leak_free"] = bool(audit.get("leak_free"))
        print_caid3_eval_report(caid3_rep)
        save_caid3_eval_report(caid3_rep, os.path.join(checkpoint_dir, "caid3_eval_report.json"))
        report["caid3"]["disorder_pdb"] = {
            "auc": (caid3_rep.get("pooled") or {}).get("auc_roc"),
            "path": os.path.join(checkpoint_dir, "caid3_eval_report.json"),
            "submission_dir": out_dir,
        }
        report["efficiency"]["caid3_disorder_pdb"] = {
            "inference_wall_seconds": infer_s,
            "n_proteins": len(ref_proteins),
        }

    # Optional extra labeled tracks
    for track, path in refs.items():
        if track == "disorder_pdb":
            continue
        ref_proteins = parse_caid_reference_fasta(path)
        if not ref_proteins:
            continue
        tmp_fasta = os.path.join(checkpoint_dir, f"_caid3_{track}_query.fasta")
        with open(tmp_fasta, "w") as f:
            for p in ref_proteins:
                f.write(f">{p['id']}\n{p['sequence']}\n")
        preds = preds_factory(tmp_fasta)
        track_dir = os.path.join(checkpoint_dir, f"caid3_submission_{track}")
        export_caid_predictions_dir(ref_proteins, preds, track_dir)
        trep = evaluate_caid_predictions(ref_proteins, preds)
        trep["benchmark"] = f"CAID3_{track}"
        save_caid3_eval_report(
            trep, os.path.join(checkpoint_dir, f"caid3_eval_report_{track}.json"),
        )
        report["caid3"][track] = {
            "auc": (trep.get("pooled") or {}).get("auc_roc"),
            "path": os.path.join(checkpoint_dir, f"caid3_eval_report_{track}.json"),
        }

    # CAID4 blind targets → submission + timings (no labels yet)
    caid4_path = resolve_caid4_targets(checkpoint_dir, caid4_targets)
    if caid4_path:
        targets = parse_caid_targets_fasta(caid4_path)
        tmp_fasta = os.path.join(checkpoint_dir, "_caid4_query.fasta")
        with open(tmp_fasta, "w") as f:
            for p in targets:
                f.write(f">{p['id']}\n{p['sequence']}\n")
        t0 = time.perf_counter()
        preds = preds_factory(tmp_fasta)
        wall = time.perf_counter() - t0
        out_dir = os.path.join(checkpoint_dir, "caid4_submission", "disorder")
        os.makedirs(out_dir, exist_ok=True)
        # Per-protein timing approximation: proportional to length
        n_res = sum(max(len(p["sequence"]), 1) for p in targets) or 1
        timings = []
        for p in targets:
            pid = p["id"]
            if pid not in preds:
                continue
            write_caid_prediction_file(
                os.path.join(out_dir, f"{pid}.caid"),
                pid, p["sequence"], preds[pid],
            )
            share = len(p["sequence"]) / n_res
            timings.append((pid, wall * share * 1000.0))
        write_timings_csv(
            os.path.join(checkpoint_dir, "caid4_submission", "timings.csv"),
            timings,
            header_note="DisorderNet CAID4 blind submission timings",
        )
        labeled = any(p.get("labels") for p in targets)
        caid4_block = {
            "status": "submission_ready",
            "targets_path": caid4_path,
            "n_targets": len(targets),
            "submission_dir": out_dir,
            "timings_csv": os.path.join(checkpoint_dir, "caid4_submission", "timings.csv"),
            "inference_wall_seconds": wall,
            "labels_available": labeled,
            "note": (
                "CAID4 is blind until ~Dec 2026 preliminary results; "
                "this export is for organizer submission / later scoring."
            ),
        }
        if labeled:
            scored = evaluate_caid_predictions(targets, preds)
            save_caid3_eval_report(
                scored, os.path.join(checkpoint_dir, "caid4_eval_report.json"),
            )
            caid4_block["eval"] = scored
            caid4_block["status"] = "scored"
        report["caid4"] = caid4_block
        report["efficiency"]["caid4"] = {
            "inference_wall_seconds": wall,
            "n_proteins": len(targets),
            "ms_per_1000_residues": (wall * 1000.0) / (n_res / 1000.0),
        }
    else:
        report["caid4"] = {
            "status": "skipped_no_targets",
            "hint": (
                "Place organizer CAID4 target FASTA at data/caid4_targets.fasta "
                "or set CAID4_TARGETS / --caid4-targets. Labels are not public yet."
            ),
        }

    out_path = os.path.join(checkpoint_dir, "caid_challenge_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    report["path"] = out_path
    return report
