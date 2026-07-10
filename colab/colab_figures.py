"""Publication figures for DisorderNet GPU Colab runs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)


C_OURS = "#2563EB"
C_AF3 = "#EF4444"
C_SOTA = "#7C3AED"
C_V6 = "#16A34A"
C_OTHERS = "#9CA3AF"


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def optimal_threshold(all_labels: np.ndarray, all_probs: np.ndarray) -> tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-8)
    idx = int(np.argmax(f1s))
    thresh = thresholds[idx] if idx < len(thresholds) else 0.5
    return float(thresh), float(f1s[idx])


def generate_all_figures(
    fold_results: list,
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    our_auc: float,
    our_ap: float,
    prefix: str = "",
) -> dict:
    """Generate 4 publication figures. Returns dict with opt_thresh, f1, mcc."""
    _style()
    opt_thresh, opt_f1 = optimal_threshold(all_labels, all_probs)
    preds = (all_probs >= opt_thresh).astype(int)
    our_mcc = matthews_corrcoef(all_labels.astype(int), preds)
    our_f1 = f1_score(all_labels.astype(int), preds)

    p = prefix  # file prefix

    # Figure 1 — ROC + PR
    fig1, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig1.suptitle("DisorderNet GPU — Performance Curves", fontsize=14, fontweight="bold")

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    ax_roc.plot(fpr, tpr, color=C_OURS, lw=2.5, label=f"DisorderNet GPU (AUC={our_auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], "--", color="#D1D5DB", lw=1.2, label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve (Pooled, 5-Fold CV)")
    ax_roc.legend(loc="lower right")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])

    for r in fold_results:
        fpr_f, tpr_f, _ = roc_curve(r["val_labels"], r["val_probs"])
        ax_roc.plot(fpr_f, tpr_f, color=C_OURS, lw=0.8, alpha=0.3)

    prec, rec, _ = precision_recall_curve(all_labels, all_probs)
    ax_pr.plot(rec, prec, color=C_OURS, lw=2.5, label=f"DisorderNet GPU (AP={our_ap:.4f})")
    baseline = all_labels.mean()
    ax_pr.axhline(baseline, color="#D1D5DB", ls="--", lw=1.2,
                  label=f"Random baseline (AP={baseline:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve (Pooled, 5-Fold CV)")
    ax_pr.legend(loc="upper right")
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])

    fig1.tight_layout()
    fig1.savefig(f"{p}fig1_roc_pr.pdf")
    fig1.savefig(f"{p}fig1_roc_pr.png")
    plt.close(fig1)

    # Figure 2 — Benchmark bar chart
    methods = [
        ("AF3-pLDDT", 0.747, C_AF3),
        ("AF2-pLDDT", 0.770, C_AF3),
        ("IUPred3", 0.789, C_OTHERS),
        ("DisorderNet v4", 0.794, C_OTHERS),
        ("flDPnn", 0.814, C_OTHERS),
        ("SETH", 0.830, C_OTHERS),
        ("DisorderNet v6", 0.831, C_V6),
        ("ESM2_35M-LoRA", 0.868, C_OTHERS),
        ("flDPnn3a", 0.871, C_OTHERS),
        ("ESM2_650M-LoRA", 0.880, C_OTHERS),
        ("DisorderUnetLM", 0.881, C_OTHERS),
        ("ESMDisPred", 0.895, C_SOTA),
        ("DisorderNet GPU", our_auc, C_OURS),
    ]
    methods.sort(key=lambda x: x[1])
    names, aucs, colors = zip(*methods)

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(names))
    bars = ax2.barh(y_pos, aucs, color=colors, edgecolor="white", height=0.7, alpha=0.88)
    for bar, val in zip(bars, aucs):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", ha="left", fontsize=9.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("AUC-ROC")
    ax2.set_title("Intrinsic Disorder Prediction Benchmark", fontweight="bold")
    ax2.set_xlim([0.70, 0.94])
    ax2.axvline(0.88, ls="--", color=C_OURS, lw=1.5, alpha=0.5, label="Target AUC=0.88")
    ax2.legend(handles=[
        mpatches.Patch(color=C_OURS, label="DisorderNet GPU (ours)"),
        mpatches.Patch(color=C_V6, label="DisorderNet v6"),
        mpatches.Patch(color=C_AF3, label="AlphaFold pLDDT"),
        mpatches.Patch(color=C_SOTA, label="ESMDisPred (SOTA)"),
        mpatches.Patch(color=C_OTHERS, label="Other methods"),
    ], loc="lower right", framealpha=0.9)
    fig2.tight_layout()
    fig2.savefig(f"{p}fig2_benchmark.pdf")
    fig2.savefig(f"{p}fig2_benchmark.png")
    plt.close(fig2)

    # Figure 3 — Version progression
    versions = ["v4\n(ESM-8M+\nLightGBM)", "v5\n(ESM-35M+\nXGBoost)",
                "v6\n(ESM-8M+\nGBDT)", "GPU\n(ESM-650M+\nLoRA+CNN)"]
    v_aucs = [0.794, 0.823, 0.831, our_auc]
    v_aps = [0.478, 0.509, 0.537, our_ap]
    v_colors = [C_OTHERS, C_OTHERS, C_V6, C_OURS]

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle("DisorderNet Version Progression", fontsize=14, fontweight="bold")
    for ax, vals, ylabel, title in [
        (ax3a, v_aucs, "AUC-ROC", "AUC-ROC Progression"),
        (ax3b, v_aps, "Average Precision", "Average Precision Progression"),
    ]:
        x = np.arange(len(versions))
        ax.bar(x, vals, color=v_colors, edgecolor="white", alpha=0.88, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(versions, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for xi, val in zip(x, vals):
            ax.text(xi, val + 0.002, f"{val:.3f}", ha="center", fontsize=10,
                    fontweight="bold" if xi == len(versions) - 1 else "normal")
        ax.set_ylim([min(vals) * 0.93, max(vals) * 1.06])
        if ylabel == "AUC-ROC":
            ax.axhline(0.747, ls=":", color=C_AF3, lw=1.5, label="AF3-pLDDT (0.747)")
            ax.legend(loc="upper left")
    fig3.tight_layout()
    fig3.savefig(f"{p}fig3_progression.pdf")
    fig3.savefig(f"{p}fig3_progression.png")
    plt.close(fig3)

    # Figure 4 — Fold stability
    fig4 = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 3, figure=fig4, wspace=0.35)
    ax4a, ax4b, ax4c = fig4.add_subplot(gs[0]), fig4.add_subplot(gs[1]), fig4.add_subplot(gs[2])
    fig4.suptitle("Fold Stability & Score Distributions", fontsize=14, fontweight="bold")

    fold_nums = [r["fold"] for r in fold_results]
    fold_aucs = [r["best_auc"] for r in fold_results]
    fold_aps = [r["best_ap"] for r in fold_results]
    ax4a.plot(fold_nums, fold_aucs, "o-", color=C_OURS, lw=2, ms=8, label="AUC")
    ax4a.plot(fold_nums, fold_aps, "s--", color="#F59E0B", lw=2, ms=8, label="AP")
    ax4a.axhline(np.mean(fold_aucs), ls=":", color=C_OURS, lw=1.5, alpha=0.5)
    ax4a.set_xlabel("Fold")
    ax4a.set_ylabel("Score")
    ax4a.set_title("Fold-by-Fold Metrics")
    ax4a.set_xticks(fold_nums)
    ax4a.legend()
    ax4a.set_ylim([0.75, 1.0])

    pos_probs = all_probs[all_labels == 1]
    neg_probs = all_probs[all_labels == 0]
    bins = np.linspace(0, 1, 40)
    ax4b.hist(neg_probs, bins=bins, alpha=0.6, color=C_V6, density=True,
              label=f"Ordered (n={len(neg_probs):,})")
    ax4b.hist(pos_probs, bins=bins, alpha=0.6, color=C_OURS, density=True,
              label=f"Disordered (n={len(pos_probs):,})")
    ax4b.axvline(opt_thresh, ls="--", color="#1F2937", lw=1.5,
                 label=f"Opt. threshold ({opt_thresh:.2f})")
    ax4b.set_xlabel("Predicted Disorder Probability")
    ax4b.set_ylabel("Density")
    ax4b.set_title("Score Distributions")
    ax4b.legend(loc="upper center")

    last_history = fold_results[-1]["history"]
    epochs_h = [row["epoch"] for row in last_history]
    ax4c.plot(epochs_h, [r["train_loss"] for r in last_history], "o-",
              color="#9CA3AF", lw=2, ms=5, label="Train loss")
    ax4c.plot(epochs_h, [r["val_loss"] for r in last_history], "s-",
              color="#F59E0B", lw=2, ms=5, label="Val loss")
    ax4c_twin = ax4c.twinx()
    ax4c_twin.plot(epochs_h, [r["val_auc"] for r in last_history], "D-",
                   color=C_OURS, lw=2, ms=5, label="Val AUC")
    ax4c.set_xlabel("Epoch")
    ax4c.set_ylabel("BCE Loss")
    ax4c_twin.set_ylabel("AUC-ROC", color=C_OURS)
    ax4c.set_title(f"Training Curves (Fold {fold_results[-1]['fold']})")
    lines1, labels1 = ax4c.get_legend_handles_labels()
    lines2, labels2 = ax4c_twin.get_legend_handles_labels()
    ax4c.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    fig4.savefig(f"{p}fig4_stability.pdf")
    fig4.savefig(f"{p}fig4_stability.png")
    plt.close(fig4)

    print("Saved fig1–fig4 (pdf + png)")
    return {"opt_thresh": opt_thresh, "f1": our_f1, "mcc": our_mcc}


def generate_biological_utility_figure(
    bio_report: dict,
    prefix: str = "",
) -> None:
    """Figure 5 — functional enrichment + segment metrics."""
    _style()
    p = prefix
    func = bio_report.get("functional_enrichment", {})
    seg = bio_report.get("segment_metrics", {})

    fig, (ax_func, ax_seg) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Biological Utility Beyond AUC", fontsize=14, fontweight="bold")

    if func:
        names = list(func.keys())
        recalls = [func[n]["recall_at_function"] for n in names]
        enrichments = [func[n]["enrichment_vs_disorder_rate"] for n in names]
        y = np.arange(len(names))
        ax_func.barh(y, recalls, color=C_OURS, alpha=0.85, height=0.55, label="Recall@function")
        ax_func.set_yticks(y)
        ax_func.set_yticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
        ax_func.set_xlabel("Recall@function")
        ax_func.set_title("Functional Site Recovery")
        ax_func.set_xlim(0, 1.05)
        for yi, (r, e) in enumerate(zip(recalls, enrichments)):
            ax_func.text(r + 0.02, yi, f"{e:.1f}x", va="center", fontsize=8, color="#374151")
    else:
        ax_func.text(0.5, 0.5, "No functional annotations", ha="center", va="center")
        ax_func.set_axis_off()

    seg_names = ["Precision", "Recall", "F1", "MDR recall", "Mean IoU"]
    seg_vals = [
        seg.get("segment_precision", 0),
        seg.get("segment_recall", 0),
        seg.get("segment_f1", 0),
        seg.get("mdr_recall", 0),
        seg.get("mean_segment_iou", 0),
    ]
    colors = [C_OURS, C_V6, "#F59E0B", C_SOTA, C_OTHERS]
    x = np.arange(len(seg_names))
    bars = ax_seg.bar(x, seg_vals, color=colors, edgecolor="white", alpha=0.88)
    ax_seg.set_xticks(x)
    ax_seg.set_xticklabels(seg_names, rotation=20, ha="right")
    ax_seg.set_ylabel("Score")
    ax_seg.set_ylim(0, 1.05)
    ax_seg.set_title("Region-Level Segment Metrics")
    for bar, val in zip(bars, seg_vals):
        ax_seg.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                    ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{p}fig5_biological_utility.pdf")
    fig.savefig(f"{p}fig5_biological_utility.png")
    plt.close(fig)
    print("Saved fig5_biological_utility.{pdf,png}")

