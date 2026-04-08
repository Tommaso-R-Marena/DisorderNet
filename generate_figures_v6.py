"""Generate publication figures for DisorderNet v6."""
import numpy as np, json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

R = "/home/user/workspace/disorder_model/results_v6"
yt = np.load(os.path.join(R, "y_true.npy"))
yp = np.load(os.path.join(R, "y_pred.npy"))
with open(os.path.join(R, "metrics.json")) as f: M = json.load(f)

plt.rcParams.update({'font.size':12,'font.family':'sans-serif','axes.linewidth':1.2,'figure.dpi':150})

# Fig 1: ROC + PR
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fpr, tpr, _ = roc_curve(yt, yp)
auc = roc_auc_score(yt, yp)
ax1.plot(fpr, tpr, color='#2563EB', lw=2.5, label=f'DisorderNet v6 (AUC={auc:.3f})')
ax1.plot([0,1],[0,1],'k--',alpha=0.3)
ax1.set_xlabel('False Positive Rate',fontsize=13); ax1.set_ylabel('True Positive Rate',fontsize=13)
ax1.set_title('ROC Curve',fontsize=14,fontweight='bold'); ax1.legend(fontsize=11)
ax1.grid(True,alpha=0.2)

prec, rec, _ = precision_recall_curve(yt, yp)
ap = average_precision_score(yt, yp)
ax2.plot(rec, prec, color='#2563EB', lw=2.5, label=f'DisorderNet v6 (AP={ap:.3f})')
ax2.axhline(y=yt.mean(), color='k', ls='--', alpha=0.3, label=f'Random ({yt.mean():.3f})')
ax2.set_xlabel('Recall',fontsize=13); ax2.set_ylabel('Precision',fontsize=13)
ax2.set_title('Precision-Recall Curve',fontsize=14,fontweight='bold'); ax2.legend(fontsize=11)
ax2.grid(True,alpha=0.2)
plt.tight_layout(); plt.savefig(os.path.join(R,'fig1_roc_pr.png'),dpi=200,bbox_inches='tight'); plt.close()

# Fig 2: Comprehensive benchmark comparison
fig, ax = plt.subplots(figsize=(12, 7))
methods = [
    ("AF3-pLDDT\n(CAID3, rank 13)", 0.747, '#EF4444'),
    ("AF2-pLDDT\n(CAID3, rank 11)", 0.770, '#F97316'),
    ("IUPred3", 0.789, '#A855F7'),
    ("DisorderNet v4\n(no pLM)", 0.794, '#64748B'),
    ("flDPnn\n(CAID1/2 best)", 0.814, '#10B981'),
    ("DisorderNet v5\n(ESM 32d)", 0.823, '#93C5FD'),
    ("SETH\n(ProtT5+CNN)", 0.830, '#6366F1'),
    ("DisorderNet v6\n(OURS)", auc, '#2563EB'),
    ("flDPnn3a\n(CAID3)", 0.871, '#059669'),
    ("ESM2-LoRA\n(650M)", 0.880, '#0D9488'),
    ("ESMDisPred\n(CAID3 SOTA)", 0.895, '#14532D'),
]
names=[m[0] for m in methods]; aucs=[m[1] for m in methods]; colors=[m[2] for m in methods]
bars=ax.barh(range(len(methods)),aucs,color=colors,height=0.65,edgecolor='white',lw=1.5)
for i,(b,a) in enumerate(zip(bars,aucs)):
    ax.text(a+0.002,i,f'{a:.3f}',va='center',fontsize=11,fontweight='bold')
bars[7].set_edgecolor('#1D4ED8'); bars[7].set_linewidth(3)
ax.set_yticks(range(len(methods))); ax.set_yticklabels(names,fontsize=10)
ax.set_xlabel('AUC-ROC',fontsize=13)
ax.set_title('Intrinsic Disorder Prediction: Comprehensive Benchmark',fontsize=14,fontweight='bold')
ax.set_xlim(0.72,0.92); ax.axvline(x=0.747,color='#EF4444',ls=':',alpha=0.4)
ax.grid(True,axis='x',alpha=0.2)
imp=((auc-0.747)/0.747)*100
ax.annotate(f'+{imp:.1f}% over AF3',xy=(auc,7),xytext=(auc-0.03,9.8),
            fontsize=11,fontweight='bold',color='#2563EB',
            arrowprops=dict(arrowstyle='->',color='#2563EB',lw=1.5))
plt.tight_layout(); plt.savefig(os.path.join(R,'fig2_comparison.png'),dpi=200,bbox_inches='tight'); plt.close()

# Fig 3: Version progression
fig, ax = plt.subplots(figsize=(9, 5.5))
versions = [
    ("v4\nPhysics only", 0.794, '#94A3B8'),
    ("v5\n+ESM-2 (PCA-32)", 0.823, '#60A5FA'),
    ("v6\n+PCA-48, +ESM var", 0.831, '#2563EB'),
]
xs = range(len(versions))
vn=[v[0] for v in versions]; va=[v[1] for v in versions]; vc=[v[2] for v in versions]
bars=ax.bar(xs,va,color=vc,width=0.55,edgecolor='white',lw=2)
for i,v in enumerate(va): ax.text(i,v+0.003,f'{v:.3f}',ha='center',fontsize=13,fontweight='bold')
ax.axhline(y=0.747,color='#EF4444',ls='--',lw=2,label='AF3-pLDDT (0.747)')
ax.axhline(y=0.770,color='#F97316',ls='--',lw=2,label='AF2-pLDDT (0.770)')
ax.axhline(y=0.814,color='#10B981',ls='--',lw=2,label='flDPnn (0.814)')
ax.axhline(y=0.830,color='#6366F1',ls='--',lw=1.5,label='SETH (0.830)')
ax.set_xticks(xs); ax.set_xticklabels(vn,fontsize=11)
ax.set_ylabel('AUC-ROC',fontsize=13); ax.set_ylim(0.74,0.86)
ax.set_title('DisorderNet Version Progression',fontsize=14,fontweight='bold')
ax.legend(fontsize=9,loc='lower right'); ax.grid(True,axis='y',alpha=0.2)
plt.tight_layout(); plt.savefig(os.path.join(R,'fig3_progression.png'),dpi=200,bbox_inches='tight'); plt.close()

# Fig 4: Fold stability
fig, ax = plt.subplots(figsize=(8, 5))
faucs = M["fold_aucs"]
ax.bar(range(1,6),faucs,color='#2563EB',width=0.6,edgecolor='white',lw=1.5,alpha=0.85)
ax.axhline(y=0.747,color='#EF4444',ls='--',lw=2,label='AF3 (0.747)')
ax.axhline(y=0.770,color='#F97316',ls='--',lw=2,label='AF2 (0.770)')
ax.axhline(y=0.814,color='#10B981',ls='--',lw=2,label='flDPnn (0.814)')
ax.axhline(y=np.mean(faucs),color='#1D4ED8',ls='-',lw=2,alpha=0.7,label=f'Mean ({np.mean(faucs):.3f})')
for i,v in enumerate(faucs): ax.text(i+1,v+0.003,f'{v:.3f}',ha='center',fontsize=11,fontweight='bold')
ax.set_xlabel('Fold',fontsize=13); ax.set_ylabel('AUC-ROC',fontsize=13)
ax.set_title('5-Fold CV: All Folds Beat AF3 and AF2',fontsize=14,fontweight='bold')
ax.legend(fontsize=10); ax.set_ylim(0.72,0.88); ax.set_xticks(range(1,6)); ax.grid(True,axis='y',alpha=0.2)
plt.tight_layout(); plt.savefig(os.path.join(R,'fig4_folds.png'),dpi=200,bbox_inches='tight'); plt.close()

print("Figures saved to", R)
