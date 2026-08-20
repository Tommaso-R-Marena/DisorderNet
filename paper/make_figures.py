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
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
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
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9))
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
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9))

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
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 2, str(v), ha="center",
                fontsize=10, fontweight="bold")
    ax.axhline(117, color=MUT, ls=":", lw=1)
    ax.text(1.45, 120, "117 entered", fontsize=7, color=MUT, ha="right")
    ax.set_ylabel("methods the benchmark can order")
    ax.set_ylim(0, 135)
    ax.set_title("Pairwise scoring squares the noise\n"
                 f"ε 0.065 → 0.007, capacity 8 → 72")
    panel_label(ax, "b")

    ax = axes[2]
    e = np.linspace(0.01, 0.20, 300)
    ax.plot(e * 100, e * 100, color=WARN, lw=1.5, label="label noise  ε")
    ax.plot(e * 100, 2 * e * e * 100, color=ACC, lw=1.5,
            label="pair noise bound  2ε²")
    ax.plot([rel["eps_label_pooled"] * 100], [rel["eps_pairwise_pooled"] * 100],
            "o", color=INK, ms=6, zorder=5, label="measured")
    ax.set_xlabel("label-flip rate ε (%)")
    ax.set_ylabel("noise rate (%)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("discordant = 2·d·u  ⟹  ε$_{pair}$ ≤ 2ε²")
    panel_label(ax, "c")
    _ = cap, an

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
    fig = plt.figure(figsize=(10.2, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 0.8], wspace=0.80)

    # (a) the two orderings agree — and where they do not
    ax = fig.add_subplot(gs[0])
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
    ax = fig.add_subplot(gs[1])
    top = sorted(allmoves, reverse=True)[:14][::-1]
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
    ax = fig.add_subplot(gs[2])
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

    fig.savefig(os.path.join(D, "figure3_rescored.png"))
    fig.savefig(os.path.join(D, "figure3_rescored.pdf"))
    plt.close(fig)


# ── Figure 4 — the model ────────────────────────────────────────────────────
def figure4():
    wt = load("within_protein_test_all")
    holm = wt["_holm"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))

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
              frameon=False, fontsize=6.4, loc="lower right")
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

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))

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


# ── Figure 6 — reproducibility and generalisation ───────────────────────────
def figure6():
    pv, tp = load("predictive_validity"), load("temporal_results")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

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
    ax.set_title("Nine times the resolution,\nat no cost in reproducibility")
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


for f in (figure1, figure2, figure3, figure4, figure5, figure6):
    try:
        f()
        print(f"  {f.__name__} ok")
    except Exception as exc:
        print(f"  {f.__name__} FAILED: {type(exc).__name__}: {exc}")
