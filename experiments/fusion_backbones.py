"""Experiment: multi-backbone PLM fusion (ESM-2 35M + 150M).

Does concatenating features from two ESM-2 backbones beat either alone? Uses the
exact v7 protocol (same 1500 proteins, same 5-fold GroupKFold, same balanced
subsample, same LGB+XGB+HistGBM blend + smoothing) with PCA fit on the train fold
only (leakage-free), so the pooled AUC is directly comparable to:
    run_v7 (35M)  = 0.8479
    run_v7 (150M) = 0.8498

Run:  python experiments/fusion_backbones.py
"""
import json, os, sys, time, gc, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, roc_curve
import lightgbm as lgb
import xgboost as xgb

from run_v6_mem import phys, wavg, wvar

warnings.filterwarnings("ignore")

DISPROT = "/home/user/workspace/disorder_model/data/disprot_processed.json"
EMB_35M = "/home/user/workspace/disorder_model/data/embeddings"
EMB_150M = "/home/user/workspace/dn150m/data/embeddings"
SEED = 42; MAX_PROT = 1500; MAX_LEN = 800; PCA_DIM = 96; N_JOBS = 4; SMOOTH = 7


def load_proteins():
    with open(DISPROT) as f:
        data = json.load(f)
    proteins = [p for p in data
                if os.path.exists(os.path.join(EMB_35M, f"{p['disprot_id']}.npy"))
                and os.path.exists(os.path.join(EMB_150M, f"{p['disprot_id']}.npy"))
                and 30 <= p["length"] <= MAX_LEN and sum(p["disorder_labels"]) >= 3
                and p["length"] - sum(p["disorder_labels"]) >= 3]
    rng = np.random.RandomState(SEED)
    if len(proteins) > MAX_PROT:
        idx = rng.choice(len(proteins), MAX_PROT, replace=False)
        proteins = [proteins[i] for i in sorted(idx)]
    return proteins


def rich(ph, ep):
    gmean = np.broadcast_to(ep.mean(0), ep.shape)
    gstd = np.broadcast_to(ep.std(0), ep.shape)
    enorm = np.linalg.norm(ep, axis=1, keepdims=True)
    return np.concatenate([ph, ep, wavg(ep, 4), wavg(ep, 12), wavg(ep, 25), wavg(ep, 50),
                           wvar(ep, 8), wvar(ep, 20), gmean[:, :32], gstd[:, :32], enorm], 1).astype(np.float32)


def train_blend(X_tr, y_tr, X_val):
    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    lm = lgb.train({"objective": "binary", "metric": "auc", "num_leaves": 127, "max_depth": 8,
        "learning_rate": 0.05, "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 5,
        "scale_pos_weight": spw, "min_child_samples": 25, "reg_alpha": 0.05, "reg_lambda": 0.5,
        "verbose": -1, "n_jobs": N_JOBS, "seed": SEED}, lgb.Dataset(X_tr, label=y_tr), 700,
        callbacks=[lgb.log_evaluation(0)])
    xm = xgb.train({"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 7,
        "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "scale_pos_weight": spw,
        "min_child_weight": 25, "reg_alpha": 0.05, "reg_lambda": 0.5, "tree_method": "hist",
        "nthread": N_JOBS, "seed": SEED}, xgb.DMatrix(X_tr, label=y_tr), 700)
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=0.5, class_weight="balanced", random_state=SEED).fit(X_tr, y_tr)
    return 0.45 * lm.predict(X_val) + 0.35 * xm.predict(xgb.DMatrix(X_val)) + 0.20 * hgb.predict_proba(X_val)[:, 1]


def evaluate(yt, yp):
    fpr, tpr, th = roc_curve(yt, yp); opt = th[np.argmax(tpr - fpr)]; yb = (yp >= opt).astype(int)
    return {"auc": roc_auc_score(yt, yp), "ap": average_precision_score(yt, yp),
            "f1": f1_score(yt, yb), "mcc": matthews_corrcoef(yt, yb)}


def main():
    t0 = time.time()
    proteins = load_proteins(); n = len(proteins); ids = [p["disprot_id"] for p in proteins]
    print(f"Proteins: {n}", flush=True)
    ph = [phys(p["sequence"][:p["length"]]).astype(np.float32) for p in proteins]
    e35 = [np.load(os.path.join(EMB_35M, f"{p['disprot_id']}.npy")).astype(np.float16)[:p["length"]] for p in proteins]
    e150 = [np.load(os.path.join(EMB_150M, f"{p['disprot_id']}.npy")).astype(np.float16)[:p["length"]] for p in proteins]
    lab = [np.array(p["disorder_labels"][:p["length"]], dtype=np.float32) for p in proteins]

    gkf = GroupKFold(n_splits=5); rng = np.random.RandomState(SEED)
    oof = {}
    for fold, (tr_i, va_i) in enumerate(gkf.split(range(n), range(n), range(n))):
        ft = time.time()
        p35 = IncrementalPCA(n_components=PCA_DIM, batch_size=10000)
        p35.fit(np.vstack([e35[i].astype(np.float32) for i in tr_i]))
        p150 = IncrementalPCA(n_components=PCA_DIM, batch_size=10000)
        p150.fit(np.vstack([e150[i].astype(np.float32) for i in tr_i]))

        def feat(i):
            f35 = rich(ph[i], p35.transform(e35[i].astype(np.float32)))
            f150 = rich(ph[i], p150.transform(e150[i].astype(np.float32)))
            # fuse: physics once + both ESM feature blocks
            return np.concatenate([f35, f150[:, 118:]], 1).astype(np.float32)  # drop dup physics

        X_tr_f = np.nan_to_num(np.vstack([feat(i) for i in tr_i]))
        y_tr_f = np.concatenate([lab[i] for i in tr_i])
        di = np.where(y_tr_f == 1)[0]; oi = np.where(y_tr_f == 0)[0]
        nk = min(len(oi), len(di) * 3)
        keep = np.sort(np.concatenate([di, rng.choice(oi, nk, replace=False)]))
        X_tr, y_tr = X_tr_f[keep], y_tr_f[keep]; del X_tr_f, y_tr_f; gc.collect()
        X_val = np.vstack([np.nan_to_num(feat(i)) for i in va_i])
        blend = train_blend(X_tr, y_tr, X_val)
        off = 0
        for i in va_i:
            L = proteins[i]["length"]; oof[ids[i]] = blend[off:off + L]; off += L
        del X_tr, y_tr, X_val; gc.collect()
        print(f"  Fold {fold+1}/5 done [{time.time()-ft:.0f}s]  dim={feat(tr_i[0]).shape[1]}", flush=True)

    y_all = np.concatenate([lab[i] for i in range(n)])
    raw = np.concatenate([oof[ids[i]] for i in range(n)])
    sm = np.concatenate([wavg(oof[ids[i]], SMOOTH // 2) for i in range(n)])
    mr, ms = evaluate(y_all, raw), evaluate(y_all, sm)
    print("\n" + "=" * 60)
    print(f"FUSION (35M+150M)  raw AUC={mr['auc']:.4f}  smoothed AUC={ms['auc']:.4f}")
    print(f"  AP={ms['ap']:.4f} F1={ms['f1']:.4f} MCC={ms['mcc']:.4f}")
    print(f"  reference: 35M=0.8479  150M=0.8498")
    print(f"  time: {(time.time()-t0)/60:.1f} min")
    os.makedirs("experiments/out", exist_ok=True)
    json.dump({"raw": mr, "smoothed": ms}, open("experiments/out/fusion_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
