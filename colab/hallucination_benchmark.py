"""
Labeled structure-distrust / hallucination benchmark (paper architecture 1).

Protocol rule: rescue rates are ONLY valid when disorder labels are independent
of DisorderNet predictions (e.g. DisProt). Proxy screening (DN≥θ ∩ pLDDT≥70)
must never be reported as scientific rescue.

This module is the evaluation spine for the claim:
  "DisorderNet is the default post-structure distrust layer after AF/Boltz."
"""

from __future__ import annotations

import os
import json
from typing import Optional

import numpy as np

from colab.af_hallucination import (
    compute_hallucination_metrics,
    compute_plddt_baseline_auc,
    run_af_rescue_report,
)
from colab.af_plddt import plddt_to_disorder_score
from colab.biological_utility import align_fold_predictions
from sklearn.metrics import average_precision_score, roc_auc_score


PROTOCOL_VERSION = "1.0.0"


def _safe_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(s, dtype=np.float32)
    if len(y) < 5 or len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def _safe_ap(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(s, dtype=np.float32)
    if len(y) < 5 or int(y.sum()) == 0:
        return None
    try:
        return float(average_precision_score(y, s))
    except ValueError:
        return None


def compare_distrust_baselines(
    labels: np.ndarray,
    disorder_probs: np.ndarray,
    plddt: np.ndarray,
    *,
    disorder_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
) -> dict:
    """
    Head-to-head on matched residues with valid pLDDT:

    - DisorderNet disorder probabilities
    - Inverse-pLDDT baseline (structure-only)
    - Distrust-priority score: upweight high-pLDDT disagreement cue

    All metrics use *independent* ``labels`` (never DN hard calls).
    """
    labels = np.asarray(labels, dtype=np.int8).ravel()
    probs = np.asarray(disorder_probs, dtype=np.float32).ravel()
    plddt = np.asarray(plddt, dtype=np.float32).ravel()
    n = min(len(labels), len(probs), len(plddt))
    labels, probs, plddt = labels[:n], probs[:n], plddt[:n]
    valid = ~np.isnan(plddt)
    if int(valid.sum()) < 5:
        return {"enabled": False, "insufficient_data": True, "n_residues": int(valid.sum())}

    y = labels[valid]
    dn = probs[valid]
    pld = plddt[valid]
    inv = plddt_to_disorder_score(pld)
    # Soft distrust score: DN disorder × structure overconfidence (high pLDDT)
    overconf = np.clip(pld / 100.0, 0.0, 1.0)
    distrust_score = dn * overconf

    hall = compute_hallucination_metrics(
        y, dn, pld,
        threshold=disorder_threshold,
        high_plddt_threshold=high_plddt_threshold,
    )
    plddt_base = compute_plddt_baseline_auc(y, pld)

    return {
        "enabled": True,
        "insufficient_data": False,
        "protocol_version": PROTOCOL_VERSION,
        "definition": "labeled_independent",
        "n_residues": int(valid.sum()),
        "disordernet": {
            "auc": _safe_auc(y, dn),
            "ap": _safe_ap(y, dn),
        },
        "plddt_inverse_baseline": plddt_base,
        "distrust_priority_score": {
            "auc": _safe_auc(y, distrust_score),
            "ap": _safe_ap(y, distrust_score),
            "note": "DN_prob × (pLDDT/100) — ranks structure-overconfident disordered sites",
        },
        "hallucination_rescue": hall,
        "delta_auc_dn_minus_plddt": (
            None
            if _safe_auc(y, dn) is None or plddt_base.get("auc") is None
            else round(float(_safe_auc(y, dn) - plddt_base["auc"]), 4)
        ),
        "thresholds": {
            "disorder_threshold": disorder_threshold,
            "high_plddt_threshold": high_plddt_threshold,
        },
    }


def attach_caid3_credibility_floor(
    bench: dict,
    caid3_report: Optional[dict] = None,
) -> dict:
    """Attach optional CAID3 disorder evaluation as credibility floor."""
    bench = dict(bench)
    if not caid3_report:
        bench["caid3_credibility_floor"] = {
            "available": False,
            "note": "No caid3_eval_report.json attached",
        }
        return bench
    pooled = caid3_report.get("pooled") or caid3_report.get("metrics") or {}
    bench["caid3_credibility_floor"] = {
        "available": True,
        "auc": pooled.get("auc") or pooled.get("AUC"),
        "ap": pooled.get("ap") or pooled.get("AP"),
        "n_scored": (
            caid3_report.get("n_scored")
            or pooled.get("n_residues")
            or caid3_report.get("n_proteins")
        ),
        "delta_vs_esmdispred": caid3_report.get("delta_vs_esmdispred"),
        "source_keys": sorted(caid3_report.keys())[:20],
        "note": "Disorder competitiveness floor — not a hallucination metric",
    }
    return bench


def finalize_distrust_benchmark_with_caid3(
    checkpoint_dir: str,
    cfg=None,
    *,
    regenerate_figure: bool = True,
) -> Optional[dict]:
    """
    Patch structure_distrust_benchmark.json in-place after CAID3 lands.

    Eval usually runs before CAID3 in the Rockfish pipeline, so the first
    benchmark write has no credibility floor. Call this after CAID3 to attach
    the floor without re-running the full labeled eval.
    """
    import os

    bench_path = os.path.join(checkpoint_dir, "structure_distrust_benchmark.json")
    caid3_path = os.path.join(checkpoint_dir, "caid3_eval_report.json")
    if not os.path.isfile(bench_path):
        return None
    if not os.path.isfile(caid3_path):
        return None

    with open(bench_path) as f:
        bench = json.load(f)
    with open(caid3_path) as f:
        caid3_report = json.load(f)

    bench = attach_caid3_credibility_floor(bench, caid3_report)
    if cfg is not None:
        try:
            from colab.training_contamination_audit import attach_contamination_flags
            bench = attach_contamination_flags(bench, cfg)
        except Exception:
            pass

    save_distrust_benchmark(bench, bench_path)

    if regenerate_figure:
        try:
            from colab.colab_figures import generate_distrust_benchmark_figure
            fig_dir = os.path.join(checkpoint_dir, "distrust_figures")
            generate_distrust_benchmark_figure(bench, out_dir=fig_dir)
        except Exception:
            pass

    return bench


def compare_rescue_baselines(
    labels: np.ndarray,
    plddt: np.ndarray,
    scores_by_method: dict,
    *,
    high_plddt_threshold: float = 70.0,
    reference_method: str = "disordernet",
) -> dict:
    """Rescue rate per method at a *matched prediction budget*.

    A bare rescue rate is not a result. Rescue is recall restricted to
    hallucinated residues (disordered AND high pLDDT), so a method that simply
    calls more residues disordered rescues more of them — a model predicting
    everything disordered scores 1.0. Reporting 0.465 without a control says
    nothing about whether the model is finding hallucinations or merely
    predicting liberally.

    Every method is therefore thresholded to flag the *same number* of residues
    as the reference method, so rescue rates are directly comparable. Precision
    on the flagged set is reported alongside, since a method could match the
    budget while spending it on ordered residues.

    The inverse-pLDDT row is the load-bearing control: hallucinations are
    high-pLDDT by definition, so a structure-confidence baseline cannot rank
    them highly. Its rescue rate near zero is what makes the task non-trivial —
    and it means the real question is whether DisorderNet beats a *sequence*
    baseline, not whether it beats pLDDT.
    """
    labels = np.asarray(labels, dtype=np.int8).ravel()
    plddt = np.asarray(plddt, dtype=np.float32).ravel()
    valid = ~np.isnan(plddt)
    if int(valid.sum()) < 10:
        return {"enabled": False, "insufficient_data": True}

    y = labels[valid]
    pld = plddt[valid]
    hallucinated = (y == 1) & (pld >= high_plddt_threshold)
    n_halluc = int(hallucinated.sum())
    if n_halluc == 0:
        return {"enabled": False, "no_hallucinations": True}

    prepared: dict = {}
    for name, raw in scores_by_method.items():
        s = np.asarray(raw, dtype=np.float32).ravel()
        if len(s) < len(valid):
            continue
        prepared[name] = s[: len(valid)][valid]
    if reference_method not in prepared:
        return {"enabled": False, "missing_reference": reference_method}

    # Budget = number of residues the reference flags at its own 0.5 threshold.
    budget = int((prepared[reference_method] >= 0.5).sum())
    budget = max(1, min(budget, len(y) - 1))

    rows: dict = {}
    rng = np.random.default_rng(0)
    for name, s in prepared.items():
        # Take exactly `budget` residues by this method's own ranking. A
        # threshold cutoff cannot do this when scores tie: a constant scorer
        # ("everything is disordered") would clear any cutoff and flag the whole
        # set, scoring a perfect 1.0 rescue rate for no skill. Rank with a
        # deterministic random tiebreak so ties are broken arbitrarily rather
        # than in favour of the method.
        order = np.lexsort((rng.random(len(s)), -s))
        flagged = np.zeros(len(s), dtype=bool)
        flagged[order[:budget]] = True
        n_flagged = int(flagged.sum())
        cutoff = float(s[order[budget - 1]])
        rescued = int((flagged & hallucinated).sum())
        rows[name] = {
            "rescue_rate": round(rescued / n_halluc, 4),
            "n_rescued": rescued,
            "n_flagged": n_flagged,
            "precision_on_flagged": round(
                float((flagged & (y == 1)).sum()) / max(n_flagged, 1), 4
            ),
            "score_cutoff": float(cutoff),
        }

    ref = rows[reference_method]["rescue_rate"]
    floor = (rows.get("random_floor") or {}).get("rescue_rate")
    below_chance = None if floor is None else bool(ref < floor)
    best_other = max(
        ((n, r["rescue_rate"]) for n, r in rows.items() if n != reference_method),
        key=lambda kv: kv[1],
        default=(None, None),
    )
    return {
        "enabled": True,
        "definition": (
            "rescue = fraction of hallucinated residues (labelled disordered AND "
            "pLDDT >= threshold) flagged, at a prediction budget matched to the "
            "reference method"
        ),
        "high_plddt_threshold": high_plddt_threshold,
        "n_hallucinated": n_halluc,
        "matched_budget_residues": budget,
        "reference_method": reference_method,
        "methods": rows,
        "below_random_floor": below_chance,
        "verdict": (
            "REFUTED: the model finds fewer hallucinations than random selection "
            "at the same budget. It does not act as a distrust layer — its "
            "confident disorder calls avoid exactly the high-pLDDT residues "
            "where the structure prediction is wrong."
            if below_chance
            else "supported: rescue exceeds the random floor"
        ) if below_chance is not None else None,
        "best_competing_method": best_other[0],
        "delta_vs_best_competing": (
            None if best_other[1] is None else round(ref - best_other[1], 4)
        ),
        "interpretation": (
            "A rescue rate is only meaningful against these controls. If "
            "delta_vs_best_competing is near zero, the model is not detecting "
            "hallucinations better than the alternative — it is predicting "
            "disorder, which the alternative also does."
        ),
    }


def _bootstrap_distrust_delta(
    labels_by_protein: list,
    dn_by_protein: list,
    plddt_by_protein: list,
    *,
    n_boot: int = 1000,
) -> dict:
    """Protein-clustered CI for DisorderNet minus inverse-pLDDT on matched residues.

    Applies the same validity filter ``compare_distrust_baselines`` uses (finite
    pLDDT), per protein, so the interval describes exactly the comparison the
    point estimate reports.
    """
    from colab.bootstrap_ci import paired_protein_bootstrap_delta

    ys, dns, invs = [], [], []
    for y, p, pld in zip(labels_by_protein, dn_by_protein, plddt_by_protein):
        valid = ~np.isnan(pld)
        if int(valid.sum()) < 2 or len(np.unique(y[valid])) < 2:
            continue
        ys.append(y[valid])
        dns.append(p[valid])
        invs.append(plddt_to_disorder_score(pld[valid]))

    if len(ys) < 2:
        return {"insufficient_data": True, "n_proteins": len(ys)}
    return paired_protein_bootstrap_delta(ys, dns, invs, metric="auc", n_boot=n_boot)


def run_labeled_distrust_benchmark(
    proteins: list,
    fold_results: list,
    plddt_by_id: dict[str, np.ndarray],
    *,
    n_folds: int = 5,
    disorder_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    structure_source: str = "af2",
    cfg=None,
    caid3_report: Optional[dict] = None,
    extra_scores_by_id: Optional[dict] = None,
) -> dict:
    """
    Full labeled benchmark: Phase-2 style rescue report + matched baselines.

    Requires DisProt (or equivalent) labels via fold OOF alignments.
    """
    from colab.training_contamination_audit import attach_contamination_flags

    rescue = run_af_rescue_report(
        proteins,
        fold_results,
        plddt_by_id,
        threshold=disorder_threshold,
        high_plddt_threshold=high_plddt_threshold,
        n_folds=n_folds,
        source=structure_source,
    )

    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    all_plddt: list[np.ndarray] = []
    for item in aligned:
        pid = item["id"]
        if pid not in plddt_by_id:
            continue
        L = len(item["probs"])
        pld = np.asarray(plddt_by_id[pid], dtype=np.float32).ravel()
        if len(pld) < L:
            continue
        all_y.append(np.asarray(item["labels"], dtype=np.int8).ravel()[:L])
        all_p.append(np.asarray(item["probs"], dtype=np.float32).ravel()[:L])
        all_plddt.append(pld[:L])

    if not all_y:
        baselines = {"enabled": False, "insufficient_data": True}
    else:
        baselines = compare_distrust_baselines(
            np.concatenate(all_y),
            np.concatenate(all_p),
            np.concatenate(all_plddt),
            disorder_threshold=disorder_threshold,
            high_plddt_threshold=high_plddt_threshold,
        )
        # delta_auc_dn_minus_plddt is go/no-go criterion #1 and came out at
        # +0.0088 in the 650M run. A difference that small is uninterpretable
        # without an interval, and the interval has to resample *proteins*:
        # residues within a protein are strongly correlated, so treating ~10^6
        # of them as independent draws would make almost any gap look decisive.
        baselines["delta_auc_ci"] = _bootstrap_distrust_delta(
            all_y, all_p, all_plddt,
            n_boot=int(os.environ.get("DISORDERNET_CI_BOOT", "1000")),
        )

        # Rescue rate is meaningless without controls at a matched prediction
        # budget — see compare_rescue_baselines. Always include the inverse-pLDDT
        # baseline (which by construction cannot rank high-pLDDT hallucinations)
        # and a random floor, plus any extra stream the caller supplies.
        flat_y = np.concatenate(all_y)
        flat_p = np.concatenate(all_p)
        flat_pl = np.concatenate(all_plddt)
        streams = {
            "disordernet": flat_p,
            "inverse_plddt": plddt_to_disorder_score(np.nan_to_num(flat_pl, nan=50.0)),
            "random_floor": np.random.default_rng(0).random(len(flat_y)).astype(np.float32),
        }
        if extra_scores_by_id:
            extra = []
            for item in aligned:
                pid = item["id"]
                if pid not in plddt_by_id:
                    continue
                L = len(item["probs"])
                arr = extra_scores_by_id.get(pid)
                extra.append(
                    np.asarray(arr, dtype=np.float32).ravel()[:L]
                    if arr is not None and len(np.asarray(arr).ravel()) >= L
                    else np.full(L, 0.5, dtype=np.float32)
                )
            if extra:
                streams["v6_physics"] = np.concatenate(extra)
        baselines["rescue_baselines"] = compare_rescue_baselines(
            flat_y, flat_pl, streams,
            high_plddt_threshold=high_plddt_threshold,
        )

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "claim": (
            "Post-structure distrust layer: independent labels required for "
            "hallucination rescue rates; pLDDT-only is the null baseline"
        ),
        "structure_source": structure_source,
        "labeled_rescue_report": rescue,
        "matched_baselines": baselines,
        "non_claims": [
            "proxy_DN_threshold_intersection_is_not_rescue",
            "not_a_conformational_ensemble_predictor",
            "not_an_alphafold_replacement",
        ],
    }
    report = attach_contamination_flags(report, cfg)
    report = attach_caid3_credibility_floor(report, caid3_report)
    return report


def save_distrust_benchmark(report: dict, path: str) -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def print_distrust_benchmark(report: dict) -> None:
    print(f"\n{'═' * 60}")
    print(" Structure distrust benchmark (labeled protocol)")
    print(f"{'═' * 60}")
    print(f"  protocol={report.get('protocol_version')}  source={report.get('structure_source')}")
    rescue = report.get("labeled_rescue_report") or {}
    pooled = rescue.get("pooled") or {}
    if pooled:
        print(
            f"  halluc_rate={pooled.get('hallucination_rate')}  "
            f"rescue_rate={pooled.get('rescue_rate')}  "
            f"n_halluc={pooled.get('n_hallucinated')}"
        )
    base = report.get("matched_baselines") or {}
    if base.get("enabled"):
        dn = base.get("disordernet") or {}
        pl = base.get("plddt_inverse_baseline") or {}
        print(
            f"  matched AUC  DN={dn.get('auc')}  inv-pLDDT={pl.get('auc')}  "
            f"Δ={base.get('delta_auc_dn_minus_plddt')}"
        )
        # The interval, not just the point estimate: go/no-go criterion #1 turns
        # on a difference of a few thousandths of AUC.
        rb = base.get("rescue_baselines") or {}
        if rb.get("enabled"):
            print("\n  -- rescue at matched prediction budget --")
            for name, m in (rb.get("methods") or {}).items():
                print(f"    {name:<16s} rescue={m['rescue_rate']:.4f}  "
                      f"precision={m['precision_on_flagged']:.4f}")
            if rb.get("below_random_floor"):
                print(f"\n  *** {rb.get('verdict')} ***\n")
        ci = base.get("delta_auc_ci") or {}
        if ci.get("ci_low") is not None:
            verdict = (
                "within protein-level sampling noise"
                if ci.get("crosses_zero") else "excludes zero"
            )
            print(
                f"    95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] "
                f"over {ci.get('n_proteins')} proteins (cluster bootstrap) — {verdict}"
            )
            if ci.get("p_value_bootstrap") is not None:
                print(f"    bootstrap p = {ci['p_value_bootstrap']:.4f}")
    print("  non-claims:", ", ".join(report.get("non_claims") or []))
