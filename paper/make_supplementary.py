#!/usr/bin/env python3
"""Emit the supplementary tables as CSV, from the analysis outputs.

The large tables — full leaderboards over 60–104 methods on five references —
belong in machine-readable form rather than retyped into prose, so each is
written straight from the JSON its job produced.
"""
from __future__ import annotations

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "paper", "figures")
OUT = os.path.join(ROOT, "paper", "supplementary")
TASKS = ["disorder_pdb", "disorder_nox", "binding", "binding_idr", "linker"]
NICE = {"disorder_pdb": "Disorder-PDB", "disorder_nox": "Disorder-NOX",
        "binding": "Binding", "binding_idr": "Binding-IDR", "linker": "Linker"}


def load(name):
    return json.load(open(os.path.join(D, f"{name}.json")))


def write(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name:44s} {len(rows):5d} rows")


def s1_decomposition():
    """Every method's pooled AUC split into its two parts, on every reference."""
    cert = load("certified_caid3")
    rows = []
    for t in TASKS:
        if t not in cert["tasks"]:
            continue
        tk = cert["tasks"][t]
        for m, v in sorted(tk["methods"].items(),
                           key=lambda kv: -kv[1]["pooled"]):
            rows.append([NICE[t], m, f"{v['pooled']:.6f}",
                         f"{v['within']:.6f}", f"{v['within_unweighted']:.6f}",
                         f"{v['between']:.6f}", f"{v['w_within']:.6f}",
                         tk["within_rank"].get(m, ""), tk["n_targets"]])
    write("S1_auc_decomposition.csv",
          ["benchmark", "method", "pooled", "within_pairweighted",
           "within_unweighted", "between", "w_within", "within_rank",
           "n_targets"], rows)


def s2_inversions():
    """Every certified inversion against the pooled winner."""
    cert = load("certified_caid3")
    rows = []
    for t in TASKS:
        tk = cert["tasks"].get(t)
        if not tk:
            continue
        for m, v in sorted(tk["inversions_against_winner"].items(),
                           key=lambda kv: -kv[1]["within_gap"]):
            ratio = v["between_gap_measured"] / v["between_gap_required"]
            rows.append([NICE[t], tk["winner"], m,
                         f"{v['within_gap']:.6f}",
                         f"{v['between_gap_measured']:.6f}",
                         f"{v['between_gap_required']:.8f}",
                         f"{ratio:.1f}", v["iff_holds"]])
    write("S2_inversion_certificates.csv",
          ["benchmark", "pooled_winner", "method", "within_gap",
           "between_gap_measured", "between_gap_required",
           "over_determined_by", "certificate_holds"], rows)


def s3_pairwise():
    """The pairwise leaderboard, complete, for all five references."""
    pl = load("pairwise_leaderboard")
    rows = []
    for t in TASKS:
        tk = pl["tasks"][t]
        for r in tk["ranking"]:
            pr = r.get("pooled_rank")
            rows.append([NICE[t], r["rank"], r["method"],
                         f"{r['pairwise']:.6f}", f"{r['pooled']:.6f}",
                         pr if pr else "", (pr - r["rank"]) if pr else "",
                         r["separated_from_leader"], tk["leader"],
                         tk["n_targets"], tk["n_eligible"], tk["n_entered"]])
    write("S3_pairwise_leaderboard.csv",
          ["benchmark", "pairwise_rank", "method", "pairwise_score",
           "pooled_auc", "pooled_rank", "rank_change",
           "separated_from_leader", "leader", "n_targets", "n_eligible",
           "n_entered"], rows)


def s4_operating_cost():
    """The price of the guarantee, whole field, every tolerated miss rate."""
    oc = load("operating_cost")
    rows = []
    for t in TASKS:
        tk = oc["tasks"].get(t)
        if not tk:
            continue
        for m, v in tk["methods"].items():
            for a in oc["alphas"]:
                k = f"alpha_{a}"
                if k not in v:
                    continue
                e = v[k]
                if "flagged_median" not in e:
                    # a method the protocol declines to price at this alpha,
                    # recorded with its reason rather than dropped
                    rows.append([NICE[t], m, a, "", "", "", "", "", "",
                                 e.get("reason", e.get("n_splits", ""))])
                    continue
                rows.append([NICE[t], m, a,
                             f"{e['flagged_median']:.4f}",
                             f"{e['flagged_iqr'][0]:.4f}",
                             f"{e['flagged_iqr'][1]:.4f}",
                             f"{e['realised_risk_median']:.4f}",
                             f"{e['flagged_quantile_median']:.4f}",
                             f"{e['calibration_credit']:.4f}",
                             e["n_splits"]])
    write("S4_operating_cost.csv",
          ["benchmark", "method", "alpha", "flagged_median", "flagged_iqr_lo",
           "flagged_iqr_hi", "realised_risk_median", "flagged_per_protein",
           "calibration_credit", "n_splits"], rows)


def s5_paired_tests():
    """All thirty per-target paired comparisons, with the Holm adjustment."""
    wt = load("within_protein_test_all")
    holm = wt["_holm"]
    rows = []
    for k, v in wt.items():
        if k.startswith("_"):
            continue
        task, rest = k.split(": ", 1)
        ours, opp = rest.split(" vs ")
        h = holm[k]
        rows.append([NICE.get(task, task), ours, opp, v["n_targets"],
                     v["wins"], f"{v['mean']:.6f}", f"{v['sd']:.6f}",
                     f"{v['ci'][0]:.6f}", f"{v['ci'][1]:.6f}",
                     f"{v['p_bootstrap']:.6f}", f"{v['p_ttest']:.3e}",
                     f"{v['p_wilcoxon']:.3e}", f"{h['p_adjusted']:.3e}",
                     h["significant"], h["rank"], h["n_comparisons"]])
    rows.sort(key=lambda r: float(r[12]))
    write("S5_within_protein_paired_tests.csv",
          ["benchmark", "our_system", "opponent", "n_targets", "wins",
           "mean_delta", "sd", "ci_lo", "ci_hi", "p_bootstrap", "p_ttest",
           "p_wilcoxon", "p_holm_adjusted", "significant_at_0.05", "holm_rank",
           "family_size"], rows)


def s6_capacity():
    """Capacity by benchmark, by all three routes the theorem provides."""
    import math

    cap = load("benchmark_capacity")
    eps, delta = cap["epsilon"], cap["delta"]
    k_noise = math.ceil(1.0 / (2.0 * eps))
    rows = []
    for rnd, tasks in cap["rounds"].items():
        for name, v in tasks.items():
            rows.append([rnd, NICE.get(name, name), v["n_targets"],
                         v["n_residues"], f"{eps:.6f}", delta, v["nu"],
                         v["capacity_targets"], v["capacity_residues"],
                         k_noise, v["n_entrants"] or ""])
    write("S6_capacity.csv",
          ["round", "benchmark", "n_targets", "n_residues", "eps", "delta",
           "noise_budget_residues", "k_from_target_grid", "k_from_residues",
           "k_noise_only", "n_entrants"], rows)


def s7_noise():
    """Per-protein label and pairwise noise, from MobiDB structure pairs."""
    rel = load("relative_noise")
    rows = [[r["acc"], r["n_structures"], r["n_structure_pairs"],
             r["label_residues"], r["label_disagree"], f"{r['eps_label']:.6f}",
             r["pair_total"], r["pair_discordant"],
             f"{r['eps_pairwise']:.6f}", f"{r['eps_pairwise_superseded']:.6f}",
             r["agree_disordered"], r["agree_ordered"],
             r["flip_down"], r["flip_up"],
             r["sp_checked"], r["sp_balanced"], r["sp_hypotheses_ok"],
             r["sp_bound_ok"], r["sp_bound_ok_under_hypotheses"]]
            for r in sorted(rel["proteins"], key=lambda r: -r["pair_total"])]
    write("S7_per_protein_noise.csv",
          ["accession", "n_structures", "n_structure_pairs",
           "residues_compared", "label_disagreements", "eps_label",
           "comparable_pairs", "discordant_pairs", "eps_pairwise",
           "eps_pairwise_superseded", "agree_disordered", "agree_ordered",
           "flip_down_d", "flip_up_u", "structure_pairs_checked",
           "balanced_within_10pct", "both_hypotheses", "bound_holds",
           "bound_holds_under_hypotheses"], rows)


def s8_reproducibility():
    """Cross-round rank correlations, every protocol pairing."""
    pv = load("predictive_validity")
    rows = []
    for t, v in pv["tasks"].items():
        rows.append([NICE.get(t, t), v.get("n_methods", ""),
                     v.get("r_pairwise_to_pairwise", ""),
                     v.get("r_pooled_to_pooled", ""),
                     v.get("r_pairwise_to_pooled", ""),
                     v.get("r_pooled_to_pairwise", "")])
    write("S8_cross_round_reproducibility.csv",
          ["benchmark", "n_shared_methods", "pairwise_to_pairwise",
           "pooled_to_pooled", "pairwise_to_pooled", "pooled_to_pairwise"],
          rows)


def s14_registered_endpoints():
    """The three endpoints computed to close PREREG_7, _8 and _9."""
    re_ = load("registered_endpoints")
    rows = []
    for k, v in re_.items():
        if k.startswith("_"):
            continue
        task, rest = k.split(": ", 1)
        variant, control = rest.split(" vs ")
        for side in ("variant", "control"):
            d = v[side]
            rows.append([NICE.get(task, task),
                         variant if side == "variant" else control, side,
                         d["pooled_all_targets"], d["pooled"],
                         d["auc_within"], d["auc_within_unweighted"],
                         d["auc_between"], d["w_within"],
                         v["n_targets"], v["wins"], v["mean_per_target"],
                         v["ci"][0], v["ci"][1], v["p_wilcoxon"],
                         re_["_holm"][k]["p_adjusted"]])
    write("S14_registered_endpoints.csv",
          ["benchmark", "run", "side", "pooled_all_targets",
           "pooled_two_class_targets", "within_pairweighted",
           "within_unweighted", "between", "w_within", "n_paired_targets",
           "wins", "mean_per_target_delta", "ci_lo", "ci_hi", "p_wilcoxon",
           "p_holm_adjusted"], rows)


def s15_region_screen():
    """What the unit of testing costs, per reference."""
    rs = load("region_screen")
    rows = [[v["round"], NICE.get(v["task"], v["task"]), v["n_residues"],
             v["n_regions"], v["n_disordered_regions"],
             f"{v['mean_region_length']:.2f}", f"{v['H_residues']:.4f}",
             f"{v['H_regions']:.4f}", f"{v['saving']:.4f}",
             f"{v['log_b_minus_1']:.4f}", f"{v['threshold_ratio']:.4f}"]
            for v in rs.values()]
    write("S15_region_screen.csv",
          ["round", "benchmark", "n_residues", "n_regions",
           "n_disordered_regions", "mean_region_length", "H_residues",
           "H_regions", "saving", "log_b_minus_1", "threshold_ratio"], rows)


for f in (s1_decomposition, s2_inversions, s3_pairwise, s4_operating_cost,
          s5_paired_tests, s6_capacity, s7_noise, s8_reproducibility,
          s14_registered_endpoints, s15_region_screen):
    try:
        f()
    except Exception as exc:
        print(f"  {f.__name__} FAILED: {type(exc).__name__}: {exc}")
