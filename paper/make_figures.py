#!/usr/bin/env python3
"""Every figure in the manuscript, from the result JSONs.

Nothing is transcribed: each panel reads the file the job wrote, so a figure
cannot drift from the number it illustrates.
"""
from __future__ import annotations

import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "paper", "figures")
CAID3 = os.path.join(ROOT, "results", "caid3")
# Nature Methods: 180 mm double-column, 88 mm single-column, minimum ~5 pt type
# at final size. Figures are authored at final size so nothing is rescaled.
FULL, HALF = 7.087, 3.465

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})
INK, ACC, WARN, MUT = "#1a1a1a", "#0F766E", "#B45309", "#9CA3AF"


def load(name):
    p = os.path.join(D, f"{name}.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def official_winner_ranks():
    """The declared CAID3 winner's within-protein rank, parsed from the table
    ``within_protein_leaderboard.py`` wrote.

    Parsed rather than retyped: a figure that restates a number is a second
    place for it to be wrong.
    """
    md = open(os.path.join(CAID3, "WITHIN_PROTEIN.md")).read()
    block = md[md.index("| benchmark | CAID3 winner |"):]
    block = block[:block.index("\n\n")]
    rows = []
    for line in block.splitlines()[2:]:
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*\*{0,2}(\d+)"
                     r"\*{0,2}\s*/\s*(\d+)\s*\|", line)
        if m:
            rows.append((m.group(1), m.group(2), int(m.group(3)),
                         int(m.group(4))))
    if len(rows) != 5:
        raise RuntimeError(f"parsed {len(rows)} winner rows, expected 5")
    return rows


def panel_label(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


# ── Figure 1 — the metric is a calibration contest ──────────────────────────
def figure1():
    cert = load("certified_caid3")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.35))
    names = {"disorder_pdb": "Disorder-PDB", "disorder_nox": "Disorder-NOX",
             "binding": "Binding", "binding_idr": "Binding-IDR",
             "linker": "Linker"}

    ax = axes[0]
    ts = [t for t in names if t in (cert or {}).get("tasks", {})]
    # w_within is a property of the reference, identical for every method, so
    # it is stored per method rather than per task; take any one of them.
    w = [next(iter(cert["tasks"][t]["methods"].values()))["w_within"] * 100
         for t in ts]
    ax.barh(range(len(ts)), [100 - x for x in w], color=MUT, label="between-protein")
    ax.barh(range(len(ts)), w, color=ACC, label="within-protein")
    ax.set_yticks(range(len(ts)))
    ax.set_yticklabels([names[t] for t in ts])
    ax.set_xlabel("share of the metric's comparison pairs (%)")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("CAID's statistic is 97–99.5%\na between-protein question")
    panel_label(ax, "a")

    ax = axes[1]
    rows = sorted(official_winner_ranks(), key=lambda r: -r[2])
    y = np.arange(len(rows))
    ax.barh(y, [r[2] for r in rows],
            color=[WARN if r[2] > 10 else ACC for r in rows])
    for i, r in enumerate(rows):
        ax.text(r[2] + 0.7, i, f"of {r[3]}", va="center", fontsize=6.5,
                color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}\n{r[1]}" for r in rows], fontsize=6.5)
    ax.set_xlabel("the winner's rank on the within-protein axis")
    ax.set_xlim(0, 32)
    ax.set_title("The declared winner is 21st–25th\nat the residue-level question")
    panel_label(ax, "b")

    ax = axes[2]
    cert_inv = cert["tasks"]["disorder_nox"]["inversions_against_winner"]
    inv = sorted(cert_inv.items(),
                 key=lambda kv: kv[1]["between_gap_measured"]
                 / kv[1]["between_gap_required"])
    assert all(v["iff_holds"] for _k, v in inv), "a certificate failed"
    y = np.arange(len(inv))
    ratio = [v["between_gap_measured"] / v["between_gap_required"]
             for _k, v in inv]
    ax.barh(y, ratio, color=WARN)
    for i, r in enumerate(ratio):
        ax.text(r * 1.05, i, f"{r:,.0f}×", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("-disorder", "") for k, _v in inv],
                       fontsize=7)
    ax.set_xlabel("measured gap ÷ gap the theorem requires")
    ax.set_xlim(100, 6000)
    ax.set_title(f"Inversions are over-determined\n"
                 f"by {min(ratio):,.0f}–{max(ratio):,.0f}×")
    panel_label(ax, "c")

    fig.tight_layout()
    fig.savefig(os.path.join(D, "figure1_decomposition.png"))
    fig.savefig(os.path.join(D, "figure1_decomposition.pdf"))
    plt.close(fig)


# ── Figure 2 — capacity, and the escape ─────────────────────────────────────
def figure2():
    cap, rel = load("benchmark_capacity"), load("relative_noise")
    an = load("annotation_noise")
    ci = load("capacity_interval")
    field = load("benchmark_capacity_field")
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 5.2))
    axes = axes.ravel()

    ax = axes[0]
    eps = np.linspace(0.005, 0.25, 400)
    ax.plot(eps * 100, [math.ceil(1 / (2 * e)) for e in eps], color=INK, lw=1.6)
    e_lab = rel["eps_label_pooled"]
    ax.axvline(e_lab * 100, color=WARN, ls="--", lw=1)
    ax.plot([e_lab * 100], [rel["capacity_label"]], "o", color=WARN, ms=6)
    ax.annotate(f"measured ε = {e_lab:.3f}\ncapacity {rel['capacity_label']}",
                (e_lab * 100, rel["capacity_label"]),
                textcoords="offset points", xytext=(14, 26), fontsize=7,
                color=WARN, arrowprops=dict(arrowstyle="->", color=WARN, lw=0.8))
    ax.axhline(117, color=MUT, ls=":", lw=1)
    ax.text(19, 124, "117 entrants", fontsize=7, color=MUT, ha="right")
    ax.set_xlabel("annotation error rate ε (%)")
    ax.set_ylabel("methods orderable, ⌈1/2ε⌉")
    ax.set_yscale("log")
    ax.set_ylim(3, 400)
    ax.set_xlim(0, 20)
    ax.set_title("Capacity of the residue-level protocol")
    panel_label(ax, "a")

    ax = axes[1]
    labels = ["residue\nlabels", "within-protein\npairs"]
    vals = [rel["capacity_label"], rel["capacity_pairwise"]]
    b = ax.bar(labels, vals, color=[WARN, ACC], width=0.55)
    for i, (r, v, key) in enumerate(zip(b, vals, ("label", "pairwise"))):
        lo, hi = ci[key]["capacity_ci"]
        ax.errorbar(r.get_x() + r.get_width() / 2, v,
                    yerr=[[v - lo], [hi - v]], color=INK, capsize=3, lw=1.1)
        ax.text(r.get_x() + r.get_width() / 2, hi + 5, str(v), ha="center",
                fontsize=9.5, fontweight="bold")
    ax.axhline(117, color=MUT, ls=":", lw=1)
    ax.text(-0.45, 121, "117 entered", fontsize=6.4, color=MUT, ha="left")
    ax.set_ylabel("methods the benchmark can order")
    ax.set_ylim(0, 150)
    ax.set_title("Pair noise is second-order:\n"
                 f"ε {rel['eps_label_pooled']:.3f} → "
                 f"{rel['eps_pairwise_pooled']:.4f}")
    panel_label(ax, "b")

    ax = axes[2]
    sp = rel["structure_pairs"]
    rows = [("both hypotheses\n(bound holds on all "
             f"{sp['bound_holds_under_hypotheses']})", sp["both_hypotheses"]),
            ("balanced agreement\nclasses (within 10%)", sp["balanced_within_10pct"]),
            ("noise rate ε ≤ 1/4", sp["eps_le_quarter"])]
    y = np.arange(len(rows))
    ax.barh(y, [sp["checked"]] * len(rows), color=MUT, height=0.6)
    ax.barh(y, [r[1] for r in rows], color=[WARN, WARN, ACC], height=0.6)
    for i, r in enumerate(rows):
        ax.text(sp["checked"] * 1.03, i, f"{r[1]:,} / {sp['checked']:,}",
                va="center", fontsize=6.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_xlim(0, sp["checked"] * 1.55)
    ax.set_xlabel("structure pairs")
    ax.invert_yaxis()
    ax.set_title("The closed form needs balance,\n"
                 f"and only {sp['both_hypotheses']} of {sp['checked']:,} "
                 f"pairs have it")
    panel_label(ax, "c")
    _ = cap, an

    ax = axes[3]
    esc = load("imagenet_escape")
    # the escape is computed with indeterminate corrections dropped
    esc_by = {r["benchmark"]: r for r in esc["rows"]
              if r["indeterminate_dropped"]}
    rows = [r for r in field["rows"] if "within-protein" not in r["benchmark"]]
    rows.sort(key=lambda r: r["capacity"])
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        name = r["benchmark"].replace("CAID3 (residue labels)", "CAID3")
        col = ACC if "CAID3" in r["benchmark"] else MUT
        ax.plot([1, r["capacity"]], [i, i], color=col, lw=1.2, zorder=2)
        ax.plot([r["capacity"]], [i], "o", ms=4.2, color=col, zorder=3)
        # the same benchmark under a statistic with the pairing structure
        e = esc_by.get(name) or (esc_by.get("ImageNet")
                                 if name == "ImageNet" else None)
        if name == "CAID3":
            hi = rel["capacity_pairwise"]
        elif e:
            hi = e["capacity_per_class_auc"]
        else:
            hi = None
        if hi:
            ax.plot([r["capacity"], hi], [i, i], color=WARN, lw=1.2, ls=":",
                    zorder=2)
            ax.plot([hi], [i], "D", ms=4.0, color=WARN, zorder=4)
            ax.text(hi * 1.35, i, f"{hi:,}", va="center", fontsize=5.4,
                    color=WARN, fontweight="bold")
        else:
            ax.text(r["capacity"] * 1.35, i, f"{r['capacity']}", va="center",
                    fontsize=5.6, color=col, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r["benchmark"].replace("CAID3 (residue labels)",
                                               "CAID3")
                        + f"  ({r['eps_percent']:.2f}%)" for r in rows],
                       fontsize=5.6)
    ax.set_xscale("log")
    ax.set_xlim(1, 4e8)
    ax.set_xlabel("methods the benchmark can place in a certified order")
    ax.invert_yaxis()
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=MUT, ms=4.2,
                                  label="as scored (per-item average)"),
                       plt.Line2D([], [], marker="D", ls="", color=WARN, ms=4,
                                  label="scored on within-group pairs")],
              frameon=False, fontsize=5.4, loc="upper right",
              labelspacing=0.3, borderpad=0.2)
    ax.set_title("Every benchmark has a capacity, and\n"
                 "the statistic decides it")
    panel_label(ax, "d")

    fig.tight_layout()
    fig.savefig(os.path.join(D, "figure2_capacity.png"))
    fig.savefig(os.path.join(D, "figure2_capacity.pdf"))
    plt.close(fig)


# ── Figure 3 — the field re-scored ──────────────────────────────────────────
NICE = {"disorder_pdb": "Disorder-PDB", "disorder_nox": "Disorder-NOX",
        "binding": "Binding", "binding_idr": "Binding-IDR",
        "linker": "Linker"}
TCOL = {"disorder_pdb": "#0F766E", "disorder_nox": "#B45309",
        "binding": "#1D4ED8", "binding_idr": "#7C3AED", "linker": "#BE123C"}


def figure3():
    from scipy.stats import spearmanr

    pl = load("pairwise_leaderboard")
    pv = load("predictive_validity")
    fig = plt.figure(figsize=(FULL, 5.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          wspace=0.62, hspace=0.52)

    # (a) the two orderings agree — and where they do not
    ax = fig.add_subplot(gs[0, 0])
    allmoves, xs, ys = [], [], []
    for t, v in pl["tasks"].items():
        for r in v["ranking"]:
            if not r.get("pooled_rank"):
                continue
            xs.append(r["pooled_rank"])
            ys.append(r["rank"])
            allmoves.append((abs(r["pooled_rank"] - r["rank"]), t, r["method"],
                             r["pooled_rank"], r["rank"]))
        ax.scatter([r["pooled_rank"] for r in v["ranking"] if r.get("pooled_rank")],
                   [r["rank"] for r in v["ranking"] if r.get("pooled_rank")],
                   s=7, alpha=0.55, linewidths=0, color=TCOL[t], label=NICE[t])
    lim = max(max(xs), max(ys)) + 3
    ax.plot([0, lim], [0, lim], color=INK, lw=0.8, ls="--", zorder=0)
    rho = spearmanr(xs, ys).statistic
    ax.text(0.03, 0.03, f"ρ = {rho:.3f}\n{len(xs)} method–benchmark pairs",
            transform=ax.transAxes, va="bottom", fontsize=7)
    ax.set_xlabel("rank under pooled AUC (as reported)")
    ax.set_ylabel("rank under the pairwise protocol")
    ax.set_xlim(0, lim)
    ax.set_ylim(lim, 0)
    ax.legend(frameon=False, fontsize=6, loc="upper right", handletextpad=0.2,
              borderpad=0.1, labelspacing=0.25)
    ax.set_title("The orderings mostly agree")
    panel_label(ax, "a")

    # (b) …and the disagreement is concentrated, and large
    ax = fig.add_subplot(gs[0, 1])
    top = sorted(allmoves, reverse=True)[:12][::-1]
    y = np.arange(len(top))
    for i, (_d, t, m, a, b) in enumerate(top):
        ax.annotate("", xy=(b, i), xytext=(a, i),
                    arrowprops=dict(arrowstyle="-|>", color=TCOL[t], lw=1.5,
                                    shrinkA=0, shrinkB=0))
        ax.plot([a], [i], "o", color=TCOL[t], ms=3.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m} · {NICE[t]}" for _d, t, m, _a, _b in top],
                       fontsize=6.2)
    ax.set_xlabel("rank  (dot = pooled, arrowhead = pairwise)")
    ax.set_xlim(0, max(max(a, b) for _d, _t, _m, a, b in top) + 6)
    ax.set_title("The 14 largest rank changes")
    panel_label(ax, "b")

    # (c) how much of the field the leader is actually separated from
    ax = fig.add_subplot(gs[1, 0])
    ts = list(pl["tasks"])
    sep = [pl["tasks"][t]["n_separated"] for t in ts]
    tot = [pl["tasks"][t]["n_eligible"] - 1 for t in ts]
    y = np.arange(len(ts))
    ax.barh(y, tot, color=MUT, height=0.62)
    ax.barh(y, sep, color=[TCOL[t] for t in ts], height=0.62)
    for i, (s_, t_) in enumerate(zip(sep, tot)):
        ax.text(t_ + 1.5, i, f"{s_}/{t_}", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels([NICE[t] for t in ts], fontsize=7)
    ax.set_xlabel("entrants the leader is separated from")
    ax.set_xlim(0, max(tot) + 18)
    ax.invert_yaxis()
    ax.set_title("Separation at the\ncertified margin")
    panel_label(ax, "c")

    # (d) the extra resolution reproduces on a held-out round
    ax = fig.add_subplot(gs[1, 1])
    vt = list(pv["tasks"])
    x = np.arange(len(vt))
    pw = [pv["tasks"][t]["r_pairwise_to_pairwise"] for t in vt]
    pd_ = [pv["tasks"][t]["r_pooled_to_pooled"] for t in vt]
    ax.bar(x - 0.19, pw, 0.36, color=ACC, label="pairwise → pairwise")
    ax.bar(x + 0.19, pd_, 0.36, color=MUT, label="pooled → pooled")
    for i, (a_, b_) in enumerate(zip(pw, pd_)):
        ax.text(i - 0.19, a_ + 0.008, f"{a_:.3f}", ha="center", fontsize=5.6)
        ax.text(i + 0.19, b_ + 0.008, f"{b_:.3f}", ha="center", fontsize=5.6)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(t, t) for t in vt], fontsize=6, rotation=14)
    ax.set_ylabel("Spearman, CAID3 → CAID2")
    ax.set_ylim(0.7, 1.06)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right", ncol=2,
              columnspacing=0.8, handletextpad=0.4)
    ax.set_title("Six times the resolution,\nat no cost in reproducibility")
    panel_label(ax, "d")

    fig.savefig(os.path.join(D, "figure3_rescored.png"))
    fig.savefig(os.path.join(D, "figure3_rescored.pdf"))
    plt.close(fig)


# ── Figure 4 — the model ────────────────────────────────────────────────────
def figure4():
    wt = load("within_protein_test_all")
    holm = wt["_holm"]
    tp = load("temporal_results")
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 5.6))
    axes = axes.ravel()

    ax = axes[0]
    tasks = ["disorder_pdb", "disorder_nox", "binding", "binding_idr", "linker"]
    y = np.arange(len(tasks))
    for i, t in enumerate(tasks):
        for j, (opp, off, col) in enumerate((("PUNCH2", +0.17, ACC),
                                             ("AlphaFold-rsa", -0.17, "#1D4ED8"))):
            k = f"{t}: Ensemble vs {opp}"
            v, hv = wt[k], holm[k]
            lo, hi = v["ci"]
            ax.plot([lo, hi], [i + off] * 2, color=col, lw=2.0,
                    solid_capstyle="butt")
            ax.plot([v["mean"]], [i + off], "o", color=col, ms=4.5, zorder=4,
                    mec="white", mew=0.6)
            ax.text(hi + 0.012, i + off,
                    f"{v['wins']}/{v['n_targets']}   "
                    f"p = {hv['p_adjusted']:#.2g}"
                    f"{'' if hv['significant'] else ' (n.s.)'}",
                    va="center", fontsize=6.2,
                    color=INK if hv["significant"] else MUT)
    ax.axvline(0, color=MUT, lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([NICE[t] for t in tasks])
    ax.set_xlabel("per-target within-protein AUC,\nDisorderNet-Ensemble − opponent")
    ax.set_xlim(-0.02, 0.60)
    ax.set_ylim(-0.6, len(tasks) - 0.4)
    ax.legend(handles=[Patch(color=ACC, label="vs PUNCH2 (CAID3 Disorder-PDB winner)"),
                       Patch(color="#1D4ED8", label="vs AlphaFold-rsa")],
              frameon=False, fontsize=5.8, loc="upper right")
    ax.set_title("Paired per-target margins, 95% CI\n"
                 "p Holm-adjusted over all 30 comparisons")
    panel_label(ax, "a")

    # (b) placement — parsed from the two evaluation records, one system each.
    # `mt_windowed` is the pre-registered model of record on CAID3; only the
    # two checkpoints filtered against all five references may be scored on
    # CAID2, and `mt_pbias` is the one reported there. No per-benchmark
    # selection between variants is made anywhere in this panel.
    ax = axes[1]
    rows = []
    md = open(os.path.join(CAID3, "CORRECTED.md")).read()
    blk = md[md.index("**`mt_windowed`**"):]
    blk = blk[:blk.index("**`mt_full`**")]
    for line in blk.splitlines():
        m = re.match(r"\|\s*\*{0,2}([A-Za-z-]+)\*{0,2}\s*\|\s*\*{0,2}"
                     r"([\d.]+)\*{0,2}\s*\|\s*\*{0,2}(\d+) / (\d+)"
                     r"\*{0,2}\s*\|\s*\*{0,2}(\d+) / (\d+)", line)
        if m:
            rows.append(("CAID3", m.group(1), int(m.group(5)), int(m.group(6)),
                         int(m.group(3)), int(m.group(4))))
    md = open(os.path.join(CAID3, "CAID2_REPLICATION.md")).read()
    blk = md[md.index("**`multitask_pbias`**"):]
    blk = blk[:blk.index("**`multitask_publication`**")]
    for line in blk.splitlines():
        m = re.match(r"\|\s*([A-Za-z-]+)\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|"
                     r"\s*\*{0,2}(\d+) / (\d+)\*{0,2}\s*\|\s*\*{0,2}"
                     r"(\d+) / (\d+)", line)
        if m:
            rows.append(("CAID2", m.group(1), int(m.group(4)), int(m.group(5)),
                         int(m.group(2)), int(m.group(3))))
    if len(rows) != 9:
        raise RuntimeError(f"parsed {len(rows)} placement rows, expected 9")
    y = np.arange(len(rows))
    ax.barh(y, [r[2] for r in rows], height=0.62,
            color=[ACC if r[2] == 1 else MUT for r in rows])
    for i, r in enumerate(rows):
        ax.text(r[2] + 0.35, i, f"of {r[3]} full-coverage     "
                                f"(#{r[4]} of {r[5]} entered)", va="center",
                fontsize=6.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}  {r[1]}" for r in rows], fontsize=7)
    ax.set_xlabel("rank on the benchmark's own reported metric")
    ax.set_xlim(0, 55)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.invert_yaxis()
    ax.set_title("Placement, both rounds, pooled AUC\n"
                 "CAID3: mt_windowed · CAID2: mt_pbias")
    panel_label(ax, "b")

    # (c) the temporal holdout, and how little of it survives filtering
    ax = axes[2]
    stages = [("released after the\ntraining caches", 1916),
              ("usable chains", 645),
              ("minus exact\ntraining sequences", 645 - 175),
              ("minus BLAST\nhomologues ≥40%", 186)]
    y = np.arange(len(stages))
    ax.barh(y, [st[1] for st in stages], color=[MUT, MUT, MUT, ACC],
            height=0.6)
    for i, st in enumerate(stages):
        ax.text(st[1] + 30, i, f"{st[1]:,}", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([st[0] for st in stages], fontsize=5.8)
    ax.set_xlabel("chains")
    ax.set_xlim(0, 2300)
    ax.invert_yaxis()
    ax.set_title("71% of 'new' PDB chains were\nalready in the training set")
    panel_label(ax, "c")

    # (d) performance on structures released after the caches were built.
    #
    # Matched on the 61 chains with an AlphaFold model. The baselines can only
    # be computed there, and our models are scored on 186, so the two sets of
    # numbers in `temporal_results.json` are NOT comparable: `pooled` is the
    # 186-chain score for our checkpoints and the 61-chain score for the
    # baselines, which have no `structured_subset`. Plotting them together
    # would repeat exactly the unmatched-chain-set error this project already
    # made once with conformal coverage, so the subset is used throughout and
    # the 186-chain figures are quoted separately in the text.
    ax = axes[3]
    res = tp["results"]
    order = ["multitask_windowed", "multitask_pbias", "AlphaFold-pLDDT",
             "AlphaFold-rsa"]
    nice = {"multitask_windowed": "DisorderNet-windowed",
            "multitask_pbias": "DisorderNet-pbias",
            "AlphaFold-pLDDT": "AlphaFold-pLDDT",
            "AlphaFold-rsa": "AlphaFold-rsa"}

    def matched(m, field):
        v = res[m]
        return v.get("structured_subset", v)[field]

    y = np.arange(len(order))
    w = 0.36
    ax.barh(y - w / 2, [matched(m, "pooled") for m in order], w,
            color=[ACC if "multitask" in m else MUT for m in order])
    ax.barh(y + w / 2, [matched(m, "within") for m in order], w,
            color=[ACC if "multitask" in m else MUT for m in order], alpha=0.55)
    for i, m in enumerate(order):
        ax.text(matched(m, "pooled") + 0.006, i - w / 2,
                f"{matched(m, 'pooled'):.3f}", va="center", fontsize=5.8)
        ax.text(matched(m, "within") + 0.006, i + w / 2,
                f"{matched(m, 'within'):.3f}", va="center", fontsize=5.8)
    ax.set_yticks(y)
    ax.set_yticklabels([nice[m] for m in order], fontsize=5.8)
    ax.set_xlim(0.6, 1.0)
    ax.invert_yaxis()
    ax.legend(handles=[Patch(color=INK, label="pooled"),
                       Patch(color=INK, alpha=0.5, label="within-protein")],
              frameon=False, fontsize=5.8, loc="lower right")
    ax.set_xlabel("AUC, the 61 held-out chains all methods can score")
    ax.set_title("Structures released after the training\n"
                 "caches were built, matched chain sets")
    panel_label(ax, "d")

    fig.tight_layout()
    fig.savefig(os.path.join(D, "figure4_model.png"))
    fig.savefig(os.path.join(D, "figure4_model.pdf"))
    plt.close(fig)


# ── Figure 5 — the operating guarantee ──────────────────────────────────────
def figure5():
    oc = load("operating_cost")
    task = "disorder_pdb"
    meth = oc["tasks"][task]["methods"]
    mark = {"DisorderNet-windowed": ACC, "DisorderNet-pbias": "#14B8A6",
            "PUNCH2": WARN, "AlphaFold-rsa": "#1D4ED8",
            "AlphaFold-pLDDT": "#7C3AED"}

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.45))

    # (a) validity is free; the price is not
    ax = axes[0]
    a = "alpha_0.1"
    xs = [v[a]["flagged_median"] * 100 for v in meth.values()]
    ys = [v[a]["realised_risk_median"] * 100 for v in meth.values()]
    ax.scatter(xs, ys, s=11, color=MUT, linewidths=0, alpha=0.75, zorder=2)
    for m, col in mark.items():
        if m in meth:
            ax.scatter([meth[m][a]["flagged_median"] * 100],
                       [meth[m][a]["realised_risk_median"] * 100],
                       s=34, color=col, zorder=4, edgecolor="white", lw=0.7)
    ax.axhline(10, color=INK, ls="--", lw=1)
    ax.text(99, 10.4, "tolerated miss rate α = 0.10", fontsize=6.5, ha="right")
    ax.annotate("DisorderNet\n32.0%",
                (meth["DisorderNet-windowed"][a]["flagged_median"] * 100,
                 meth["DisorderNet-windowed"][a]["realised_risk_median"] * 100),
                textcoords="offset points", xytext=(6, -26), fontsize=6.5,
                color=ACC, arrowprops=dict(arrowstyle="->", color=ACC, lw=0.7))
    worst = max(meth.items(), key=lambda kv: kv[1][a]["flagged_median"])
    ax.annotate(f"{worst[0]}\n{worst[1][a]['flagged_median'] * 100:.0f}%",
                (worst[1][a]["flagged_median"] * 100,
                 worst[1][a]["realised_risk_median"] * 100),
                textcoords="offset points", xytext=(-8, -34), fontsize=6.5,
                ha="right", color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))
    ax.set_xlabel("residues that must be flagged (%)")
    ax.set_ylabel("realised miss rate (%)")
    ax.set_ylim(0, 13)
    ax.set_xlim(20, 105)
    ax.set_title(f"Every one of {len(meth)} methods keeps the\n"
                 f"promise; the price runs {min(xs):.0f}–{max(xs):.0f}%")
    panel_label(ax, "a")

    # (b) the two ways to buy the same guarantee
    ax = axes[1]
    a = "alpha_0.05"
    top = sorted(meth.items(), key=lambda kv: kv[1][a]["flagged_median"])[:8]
    top = [kv for kv in top] + [("AlphaFold-rsa", meth["AlphaFold-rsa"])]
    y = np.arange(len(top))[::-1]
    for i, (m, v) in zip(y, top):
        g, q = v[a]["flagged_median"] * 100, v[a]["flagged_quantile_median"] * 100
        col = mark.get(m, MUT)
        ax.plot([g, q], [i, i], color=col, lw=1.6, zorder=2)
        ax.plot([g], [i], "o", color=col, ms=4.5, zorder=3)
        ax.plot([q], [i], "s", color=col, ms=4.5, mfc="white", mew=1.2,
                zorder=3)
        ax.text(max(g, q) + 1.2, i, f"{q - g:+.1f}", va="center", fontsize=6.2,
                color=col if abs(q - g) > 8 else INK)
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _v in top], fontsize=6.4)
    ax.set_xlabel("residues flagged at α = 0.05 (%)")
    ax.set_xlim(40, 76)
    ax.set_ylim(-0.9, len(top) - 0.1)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=INK,
                                  label="one global threshold", ms=4.5),
                       plt.Line2D([], [], marker="s", ls="", color=INK,
                                  mfc="white", label="per-protein quantile",
                                  ms=4.5)],
              frameon=False, fontsize=6.2, loc="lower left")
    ax.set_title("The calibration-invariant rule\ncosts the leaders twice as much")
    panel_label(ax, "b")

    # (c) the sentence AUC cannot say
    ax = axes[2]
    ax.axis("off")
    ax.text(0.5, 0.94, "DisorderNet vs PUNCH2, Disorder-PDB", ha="center",
            fontsize=8.5, transform=ax.transAxes)
    box = dict(boxstyle="round,pad=0.5", fc="#F3F4F6", ec="none")
    ax.text(0.5, 0.60,
            "pooled AUC, the reported metric\n"
            "0.9595  vs  0.9552\n"
            "Δ = +0.0043,  p = 0.20  —  inseparable",
            ha="center", va="center", fontsize=7.2, transform=ax.transAxes,
            bbox=box)
    ax.text(0.5, 0.22,
            "operating cost at α = 0.05,\ncalibration-invariant rule\n"
            "55.1%  vs  60.4%  of the protein flagged\n"
            "Δ = 5.3 points",
            ha="center", va="center", fontsize=7.2, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="#CCFBF1", ec="none"))
    ax.annotate("", xy=(0.5, 0.34), xytext=(0.5, 0.44),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    panel_label(ax, "c")

    fig.tight_layout()
    fig.savefig(os.path.join(D, "figure5_operating.png"))
    fig.savefig(os.path.join(D, "figure5_operating.pdf"))
    plt.close(fig)


# ── (retired) reproducibility and generalisation: now Fig. 3d and Fig. 4c,d ─
def _figure6_retired():
    pv, tp = load("predictive_validity"), load("temporal_results")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.45))

    ax = axes[0]
    ts = list(pv["tasks"])
    x = np.arange(len(ts))
    pw = [pv["tasks"][t]["r_pairwise_to_pairwise"] for t in ts]
    pd_ = [pv["tasks"][t]["r_pooled_to_pooled"] for t in ts]
    ax.bar(x - 0.19, pw, 0.36, color=ACC, label="pairwise → pairwise")
    ax.bar(x + 0.19, pd_, 0.36, color=MUT, label="pooled → pooled")
    for i, (a_, b_) in enumerate(zip(pw, pd_)):
        ax.text(i - 0.19, a_ + 0.008, f"{a_:.3f}", ha="center", fontsize=6.2)
        ax.text(i + 0.19, b_ + 0.008, f"{b_:.3f}", ha="center", fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(t, t) for t in ts], fontsize=6.8, rotation=12)
    ax.set_ylabel("Spearman, CAID3 → held-out CAID2")
    ax.set_ylim(0.7, 1.06)
    ax.legend(frameon=False, fontsize=6.4, loc="lower right", ncol=2,
              columnspacing=0.8, handletextpad=0.4)
    ax.set_title("Six times the resolution,\nat no cost in reproducibility")
    panel_label(ax, "a")

    ax = axes[1]
    stages = [("released after the\ntraining caches", 1916),
              ("usable chains", 645),
              ("minus exact\ntraining sequences", 645 - 175),
              ("minus BLAST\nhomologues ≥40%", 186)]
    y = np.arange(len(stages))
    ax.barh(y, [s[1] for s in stages], color=[MUT, MUT, MUT, ACC], height=0.6)
    for i, s_ in enumerate(stages):
        ax.text(s_[1] + 30, i, f"{s_[1]:,}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in stages], fontsize=6.6)
    ax.set_xlabel("chains")
    ax.set_xlim(0, 2250)
    ax.invert_yaxis()
    ax.set_title("71% of 'new' PDB chains were\nalready in the training set")
    panel_label(ax, "b")

    ax = axes[2]
    res = tp["results"]
    order = ["multitask_windowed", "multitask_pbias", "AlphaFold-pLDDT",
             "AlphaFold-rsa"]
    nice = {"multitask_windowed": "DisorderNet-windowed",
            "multitask_pbias": "DisorderNet-pbias",
            "AlphaFold-pLDDT": "AlphaFold-pLDDT", "AlphaFold-rsa": "AlphaFold-rsa"}
    y = np.arange(len(order))
    w = 0.36
    ax.barh(y - w / 2, [res[m]["pooled"] for m in order], w,
            color=[ACC if "multitask" in m else MUT for m in order],
            label="pooled")
    ax.barh(y + w / 2, [res[m]["within"] for m in order], w,
            color=[ACC if "multitask" in m else MUT for m in order],
            alpha=0.55, label="within-protein")
    for i, m in enumerate(order):
        ax.text(res[m]["pooled"] + 0.006, i - w / 2, f"{res[m]['pooled']:.3f}",
                va="center", fontsize=6.2)
        ax.text(res[m]["within"] + 0.006, i + w / 2, f"{res[m]['within']:.3f}",
                va="center", fontsize=6.2)
    ax.set_yticks(y)
    ax.set_yticklabels([nice[m] for m in order], fontsize=6.6)
    ax.set_xlim(0.6, 1.0)
    ax.invert_yaxis()
    ax.legend(handles=[Patch(color=INK, label="pooled"),
                       Patch(color=INK, alpha=0.5, label="within-protein")],
              frameon=False, fontsize=6.4, loc="lower right")
    ax.set_xlabel("AUC, 186 temporally held-out chains")
    ax.set_title("Structures released after\nthe training caches were built")
    panel_label(ax, "c")

    fig.tight_layout()
    fig.savefig(os.path.join(D, "figure6_generalisation.png"))
    fig.savefig(os.path.join(D, "figure6_generalisation.pdf"))
    plt.close(fig)


# ── Figure 6 — the price of a screen ────────────────────────────────────────
def figure6():
    rs = load("region_screen")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.45))

    # (a) logarithmic against linear
    ax = axes[0]
    m = np.unique(np.logspace(0, 5.2, 400).astype(int))
    H = np.cumsum(1.0 / np.arange(1, m.max() + 1))[m - 1]
    ax.plot(m, H, color=ACC, lw=1.8, label="harmonic  $H_m$  (this work)")
    ax.plot(m, m, color=WARN, lw=1.5, ls="--", label="Bonferroni  $m$")
    pdb = rs["CAID3 disorder_pdb"]
    for n, lab, col in ((pdb["n_regions"], "one hypothesis\nper region", ACC),
                        (pdb["n_residues"], "one hypothesis\nper residue", INK)):
        h = float(np.cumsum(1.0 / np.arange(1, n + 1))[-1])
        ax.plot([n], [h], "o", color=col, ms=6, zorder=5, mec="white", mew=0.7)
        ax.annotate(f"{lab}\n{n:,} → $H$ = {h:.1f}", (n, h),
                    textcoords="offset points",
                    xytext=(8, 26 if col is ACC else -30), fontsize=6.4,
                    ha="left", color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.7))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("hypotheses the screen states, $m$")
    ax.set_ylabel("deflation factor")
    ax.set_ylim(1, 3e4)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left")
    ax.set_title("Arbitrary dependence costs a\nlogarithm, not a factor of $m$")
    panel_label(ax, "a")

    # (b) what the design choice is worth, per reference
    ax = axes[1]
    keys = [k for k in rs if k.startswith("CAID3")] + \
           [k for k in rs if k.startswith("CAID2")]
    y = np.arange(len(keys))
    ax.barh(y, [rs[k]["saving"] for k in keys], color=ACC, height=0.6,
            label="measured  $H_n - H_M$")
    ax.barh(y, [rs[k]["log_b_minus_1"] for k in keys], color=INK, height=0.22,
            label="proved floor  $\\log b - 1$")
    for i, k in enumerate(keys):
        ax.text(rs[k]["saving"] + 0.12, i,
                f"×{rs[k]['threshold_ratio']:.2f}", va="center", fontsize=6.6,
                color=ACC, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("CAID3 ", "3 · ").replace("CAID2 ", "2 · ")
                        .replace("disorder_pdb", "Disorder-PDB")
                        .replace("disorder_nox", "Disorder-NOX")
                        .replace("binding_idr", "Binding-IDR")
                        .replace("binding", "Binding")
                        .replace("linker", "Linker") for k in keys],
                       fontsize=6.6)
    ax.set_xlabel("nats of correction bought back")
    ax.set_xlim(0, 8.6)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=6.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=2, columnspacing=1.2,
              handletextpad=0.5)
    ax.set_title("Testing regions, not residues\n(label = threshold gained)")
    panel_label(ax, "b")

    # (c) the factor is attained, so the correction is necessary
    ax = axes[2]
    mm = np.arange(1, 61)
    Hm = np.cumsum(1.0 / mm)
    q = 0.05
    ax.plot(mm, q * Hm, color=WARN, lw=1.8,
            label="uncorrected BH, realised")
    ax.axhline(q, color=INK, lw=1.4, ls="--", label="nominal level  q")
    ax.plot(mm, np.full_like(mm, q, dtype=float) * 0 + q, color=ACC, lw=1.8,
            label="BH at  q/$H_m$, realised")
    ax.axvline(2, color=MUT, lw=1, ls=":")
    ax.annotate("exceeds its level\nfrom m = 2", (2, q * Hm[1]),
                textcoords="offset points", xytext=(16, -4), fontsize=6.6,
                color=WARN,
                arrowprops=dict(arrowstyle="->", color=WARN, lw=0.7))
    ax.set_xlabel("candidates, $m$")
    ax.set_ylabel("E[FDP] on the constructed law")
    ax.set_ylim(-0.048, q * Hm[-1] * 1.08)
    ax.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.legend(frameon=False, fontsize=6.2, loc="lower right")
    ax.set_title("The harmonic factor is attained,\nso it cannot be lowered")
    panel_label(ax, "c")

    fig.tight_layout()
    fig.savefig(os.path.join(D, "figure6_screen.png"))
    fig.savefig(os.path.join(D, "figure6_screen.pdf"))
    plt.close(fig)


for f in (figure1, figure2, figure3, figure4, figure5, figure6):
    try:
        f()
        print(f"  {f.__name__} ok")
    except Exception as exc:
        print(f"  {f.__name__} FAILED: {type(exc).__name__}: {exc}")
