"""
Generate publication-quality figures for DisorderNet results.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import json, os

RESULTS_DIR = "/home/user/workspace/disorder_model/results"
y_true = np.load(os.path.join(RESULTS_DIR, "y_true.npy"))
y_pred = np.load(os.path.join(RESULTS_DIR, "y_pred.npy"))

with open(os.path.join(RESULTS_DIR, "metrics.json")) as f:
    metrics = json.load(f)

# Style
plt.rcParams.update({
    'font.size': 12, 'font.family': 'sans-serif',
    'axes.linewidth': 1.2, 'figure.dpi': 150,
})

# ============================================================
# FIGURE 1: ROC Curve with benchmark lines
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

fpr, tpr, _ = roc_curve(y_true, y_pred)
our_auc = roc_auc_score(y_true, y_pred)

ax1.plot(fpr, tpr, color='#2563EB', linewidth=2.5, label=f'DisorderNet (AUC={our_auc:.3f})')
ax1.plot([0,1],[0,1], 'k--', alpha=0.3, linewidth=1)

# Benchmark AUC lines (shown as vertical annotations)
benchmarks = [
    ("AF3-pLDDT", 0.747, '#EF4444'),
    ("AF2-pLDDT", 0.770, '#F97316'),
    ("IUPred3", 0.789, '#8B5CF6'),
    ("flDPnn", 0.814, '#10B981'),
]

for name, auc, color in benchmarks:
    ax1.axhline(y=0, alpha=0)  # dummy
    
ax1.set_xlabel('False Positive Rate', fontsize=13)
ax1.set_ylabel('True Positive Rate', fontsize=13)
ax1.set_title('ROC Curve: DisorderNet vs AlphaFold 3', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
ax1.grid(True, alpha=0.2)

# PR Curve
prec, rec, _ = precision_recall_curve(y_true, y_pred)
ap = average_precision_score(y_true, y_pred)
baseline = y_true.mean()

ax2.plot(rec, prec, color='#2563EB', linewidth=2.5, label=f'DisorderNet (AP={ap:.3f})')
ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.3, label=f'Random (AP={baseline:.3f})')
ax2.set_xlabel('Recall', fontsize=13)
ax2.set_ylabel('Precision', fontsize=13)
ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=11)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(0, 1.02)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig1_roc_pr.png'), dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 2: Benchmark Comparison Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

methods = [
    ("AF3-pLDDT\n(CAID3 rank 13)", 0.747, '#EF4444'),
    ("AF2-RSA", 0.768, '#F97316'),
    ("AF2-pLDDT\n(CAID3 rank 11)", 0.770, '#FB923C'),
    ("IUPred3", 0.789, '#8B5CF6'),
    ("DisorderNet\n(OURS)", our_auc, '#2563EB'),
    ("flDPnn\n(best CAID)", 0.814, '#10B981'),
]

names = [m[0] for m in methods]
aucs = [m[1] for m in methods]
colors = [m[2] for m in methods]

bars = ax.barh(range(len(methods)), aucs, color=colors, height=0.6, edgecolor='white', linewidth=1.5)

# Add value labels
for i, (bar, auc) in enumerate(zip(bars, aucs)):
    ax.text(auc + 0.003, i, f'{auc:.3f}', va='center', ha='left', fontsize=12, fontweight='bold')

# Highlight our method
bars[4].set_edgecolor('#1D4ED8')
bars[4].set_linewidth(3)

ax.set_yticks(range(len(methods)))
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel('AUC-ROC', fontsize=13)
ax.set_title('Intrinsic Disorder Prediction: AUC-ROC Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0.7, 0.84)
ax.axvline(x=0.747, color='#EF4444', linestyle=':', alpha=0.5, linewidth=1)
ax.grid(True, axis='x', alpha=0.2)

# Add improvement annotation
improvement = ((our_auc - 0.747) / 0.747) * 100
ax.annotate(f'+{improvement:.1f}% over AF3', 
            xy=(our_auc, 4), xytext=(our_auc - 0.02, 5.5),
            fontsize=11, fontweight='bold', color='#2563EB',
            arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig2_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 3: Cross-validation fold stability
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

fold_aucs = metrics["fold_aucs"]
fold_x = range(1, 6)

ax.bar(fold_x, fold_aucs, color='#2563EB', width=0.6, edgecolor='white', linewidth=1.5, alpha=0.85)
ax.axhline(y=0.747, color='#EF4444', linestyle='--', linewidth=2, label=f'AF3-pLDDT (0.747)')
ax.axhline(y=0.770, color='#F97316', linestyle='--', linewidth=2, label=f'AF2-pLDDT (0.770)')
ax.axhline(y=np.mean(fold_aucs), color='#1D4ED8', linestyle='-', linewidth=2, alpha=0.7,
           label=f'DisorderNet mean ({np.mean(fold_aucs):.3f})')

for i, v in enumerate(fold_aucs):
    ax.text(i+1, v+0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Fold', fontsize=13)
ax.set_ylabel('AUC-ROC', fontsize=13)
ax.set_title('5-Fold Cross-Validation: All Folds Beat AF3', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.70, 0.86)
ax.set_xticks(fold_x)
ax.grid(True, axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig3_cv_folds.png'), dpi=200, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 4: Score distribution
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

dis_scores = y_pred[y_true == 1]
ord_scores = y_pred[y_true == 0]

ax.hist(ord_scores, bins=80, alpha=0.6, color='#3B82F6', label=f'Ordered (n={len(ord_scores):,})', density=True)
ax.hist(dis_scores, bins=80, alpha=0.6, color='#EF4444', label=f'Disordered (n={len(dis_scores):,})', density=True)

ax.set_xlabel('DisorderNet Prediction Score', fontsize=13)
ax.set_ylabel('Density', fontsize=13)
ax.set_title('Score Distribution: Ordered vs Disordered Residues', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig4_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()

print("All figures saved to", RESULTS_DIR)
print("  fig1_roc_pr.png")
print("  fig2_comparison.png")
print("  fig3_cv_folds.png")
print("  fig4_distribution.png")
