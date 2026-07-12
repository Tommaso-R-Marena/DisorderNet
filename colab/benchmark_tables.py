"""
Matched vs literature benchmark tables (Tier 1 evaluation rigor).

Literature rows are reference points from published CAID/DisProt studies — NOT
head-to-head reruns on identical splits. Our rows are from this repository's
DisProt 5-fold protein-grouped CV protocol.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Table A: Literature reference (different protocols / splits — not comparable)
# ---------------------------------------------------------------------------
LITERATURE_REFERENCE_BENCHMARKS: list[dict] = [
    {
        "method": "AF3-pLDDT",
        "auc": 0.747,
        "ap": None,
        "rank": 13,
        "source": "CAID3",
        "protocol": "CAID3 evaluation set",
        "comparable_to_ours": False,
    },
    {
        "method": "AF2-pLDDT",
        "auc": 0.770,
        "ap": None,
        "rank": 11,
        "source": "CAID3",
        "protocol": "CAID3 evaluation set",
        "comparable_to_ours": False,
    },
    {
        "method": "IUPred3",
        "auc": 0.789,
        "ap": None,
        "rank": None,
        "source": "CAID",
        "protocol": "CAID benchmark",
        "comparable_to_ours": False,
    },
    {
        "method": "flDPnn",
        "auc": 0.814,
        "ap": None,
        "rank": None,
        "source": "CAID1/2",
        "protocol": "CAID benchmark",
        "comparable_to_ours": False,
    },
    {
        "method": "SETH (ProtT5+CNN)",
        "auc": 0.830,
        "ap": None,
        "rank": None,
        "source": "Literature",
        "protocol": "Published benchmark",
        "comparable_to_ours": False,
    },
    {
        "method": "flDPnn3a",
        "auc": 0.871,
        "ap": None,
        "rank": None,
        "source": "CAID3",
        "protocol": "CAID3 evaluation set",
        "comparable_to_ours": False,
    },
    {
        "method": "ESM2_35M-LoRA",
        "auc": 0.868,
        "ap": None,
        "rank": None,
        "source": "LoRA-DR",
        "protocol": "CAID-style",
        "comparable_to_ours": False,
    },
    {
        "method": "ESM2_650M-LoRA",
        "auc": 0.880,
        "ap": 0.721,
        "rank": None,
        "source": "LoRA-DR (CAID1)",
        "protocol": "CAID1",
        "comparable_to_ours": False,
    },
    {
        "method": "ESMDisPred",
        "auc": 0.895,
        "ap": 0.778,
        "rank": 1,
        "source": "CAID3 SOTA",
        "protocol": "CAID3 evaluation set",
        "comparable_to_ours": False,
    },
]

# ---------------------------------------------------------------------------
# Table B: Our runs on DisProt (same protocol within each row)
# ---------------------------------------------------------------------------
OUR_DISPROT_CPU_V6 = {
    "method": "DisorderNet v6 (ESM-2 8M + GBDT)",
    "auc": 0.831,
    "ap": 0.537,
    "f1_max": None,
    "mcc": 0.438,
    "source": "This repo (CPU)",
    "protocol": "DisProt, 5-fold protein-grouped CV, n≈1500",
    "comparable_to_ours": True,
    "status": "verified",
    "results_file": "results_v6/metrics.json",
}

OUR_DISPROT_GPU_BASELINE = {
    "method": "DisorderNet GPU (ESM-2 650M + LoRA)",
    "auc": 0.817,
    "ap": None,
    "f1_max": None,
    "mcc": None,
    "source": "This repo (Colab A100)",
    "protocol": "DisProt, 5-fold protein-grouped CV, ESM-2 650M + LoRA v2",
    "comparable_to_ours": True,
    "status": "verified_colab_run",
    "results_file": "disordernet_gpu_results_*.json",
    "notes": "Pooled OOF AUC; AF-fusion subset ~0.831 on AF-covered residues",
}

OUR_DISPROT_GPU_TEMPLATE = {
    "method": "DisorderNet GPU (ESM-2 650M + LoRA)",
    "auc": None,
    "ap": None,
    "f1_max": None,
    "mcc": None,
    "source": "This repo (Colab GPU)",
    "protocol": "DisProt, 5-fold protein-grouped CV, ESM-2 650M + LoRA",
    "comparable_to_ours": True,
    "status": "pending_full_run",
    "results_file": "disordernet_gpu_results_*.json",
}


def build_our_disprot_row(
    auc: float,
    ap: Optional[float] = None,
    f1_max: Optional[float] = None,
    mcc: Optional[float] = None,
    status: str = "verified",
) -> dict:
    """Build a GPU (or updated) row for Table B."""
    row = dict(OUR_DISPROT_GPU_TEMPLATE)
    row["auc"] = float(auc)
    row["ap"] = float(ap) if ap is not None else None
    row["f1_max"] = float(f1_max) if f1_max is not None else None
    row["mcc"] = float(mcc) if mcc is not None else None
    row["status"] = status
    return row


def get_literature_table() -> list[dict]:
    return sorted(LITERATURE_REFERENCE_BENCHMARKS, key=lambda x: x["auc"])


def get_our_disprot_table(gpu_auc: Optional[float] = None, gpu_ap: Optional[float] = None,
                          gpu_f1_max: Optional[float] = None, gpu_mcc: Optional[float] = None) -> list[dict]:
    rows = [dict(OUR_DISPROT_CPU_V6)]
    if gpu_auc is not None:
        rows.append(build_our_disprot_row(gpu_auc, gpu_ap, gpu_f1_max, gpu_mcc))
    else:
        rows.append(dict(OUR_DISPROT_GPU_TEMPLATE))
    return rows


def print_literature_reference_table() -> None:
    """Print Table A with explicit non-comparability disclaimer."""
    rows = get_literature_table()
    print(f"\n{'═' * 72}")
    print(" TABLE A — LITERATURE REFERENCE (NOT head-to-head vs our splits)")
    print(f"{'═' * 72}")
    print("  These AUCs come from published CAID/DisProt evaluations.")
    print("  Protocols and splits differ from our DisProt 5-fold CV.")
    print(f"{'─' * 72}")
    print(f"  {'Method':<28} {'AUC':>7} {'AP':>7}  {'Source':<14} Protocol")
    print(f"{'─' * 72}")
    for r in rows:
        ap_s = f"{r['ap']:.3f}" if r.get("ap") is not None else "    N/A"
        rank_s = f" #{r['rank']}" if r.get("rank") else ""
        print(f"  {r['method']:<28} {r['auc']:>7.3f} {ap_s:>7}  {r['source']:<14} {r['protocol']}{rank_s}")
    print(f"{'═' * 72}")


def print_our_disprot_table(
    gpu_auc: Optional[float] = None,
    gpu_ap: Optional[float] = None,
    gpu_f1_max: Optional[float] = None,
    gpu_mcc: Optional[float] = None,
) -> None:
    """Print Table B — our runs on identical in-repo protocol."""
    rows = get_our_disprot_table(gpu_auc, gpu_ap, gpu_f1_max, gpu_mcc)
    print(f"\n{'═' * 72}")
    print(" TABLE B — OUR DISPROT 5-FOLD PROTEIN-GROUPED CV")
    print(f"{'═' * 72}")
    print("  Directly comparable within this table only.")
    print(f"{'─' * 72}")
    print(f"  {'Method':<36} {'AUC':>7} {'AP':>7} {'F1*':>7} {'MCC':>7}  Status")
    print(f"{'─' * 72}")
    for r in rows:
        auc_s = f"{r['auc']:.3f}" if r.get("auc") is not None else "    TBD"
        ap_s = f"{r['ap']:.3f}" if r.get("ap") is not None else "    N/A"
        f1_s = f"{r['f1_max']:.3f}" if r.get("f1_max") is not None else "    N/A"
        mcc_s = f"{r['mcc']:.3f}" if r.get("mcc") is not None else "    N/A"
        status = r.get("status", "")
        if status == "pending_full_run":
            status = "pending GPU run"
        print(f"  {r['method']:<36} {auc_s:>7} {ap_s:>7} {f1_s:>7} {mcc_s:>7}  {status}")
    print(f"{'─' * 72}")
    print("  * F1_max = max F1 over thresholds (CAID-style)")
    print(f"{'═' * 72}")


def print_matched_benchmark_report(
    gpu_auc: Optional[float] = None,
    gpu_ap: Optional[float] = None,
    gpu_f1_max: Optional[float] = None,
    gpu_mcc: Optional[float] = None,
) -> dict:
    """Print both tables and return structured summary."""
    print_literature_reference_table()
    print_our_disprot_table(gpu_auc, gpu_ap, gpu_f1_max, gpu_mcc)

    if gpu_auc is not None:
        print(f"\n── Context (not head-to-head) ──")
        print(f"  GPU AUC {gpu_auc:.4f} vs literature AF3-pLDDT 0.747 (CAID3, different protocol)")
        print(f"  GPU AUC {gpu_auc:.4f} vs literature ESMDisPred 0.895 (CAID3 SOTA, different protocol)")
        print(f"  GPU vs our v6 CPU: {gpu_auc - OUR_DISPROT_CPU_V6['auc']:+.4f} (same DisProt task, different model)")

    return {
        "literature_reference": get_literature_table(),
        "our_disprot_runs": get_our_disprot_table(gpu_auc, gpu_ap, gpu_f1_max, gpu_mcc),
        "disclaimer": (
            "Literature AUCs are reference points only. "
            "Only rows in Table B share our CV protocol."
        ),
    }


def rank_against_literature(our_auc: float) -> dict:
    """Rank for context; explicitly marked as non-comparable."""
    refs = get_literature_table()
    rank = 1 + sum(1 for r in refs if r["auc"] > our_auc)
    return {
        "our_auc": float(our_auc),
        "literature_rank_if_comparable": rank,
        "n_literature_methods": len(refs),
        "comparable": False,
        "note": "Ranking is contextual only — protocols differ.",
    }
