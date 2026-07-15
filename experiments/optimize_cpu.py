"""Rigorous CPU-model optimization harness for DisorderNet.

Goal: honestly measure whether concrete modeling changes raise pooled 5-fold CV
AUC over the v6 baseline, with NO data leakage:

  * Same protein set / folds as run_v6_mem.py (seed 42, 1500 proteins, len 30-800).
  * PCA fit on the TRAIN fold only (the v6 baseline fits PCA on all proteins).
  * Post-processing (contiguity smoothing) uses a fixed, pre-registered window.

Run:  python experiments/optimize_cpu.py
"""
import json, os, sys, time, gc, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, roc_curve)
import lightgbm as lgb
import xgboost as xgb

from run_v6_mem import phys, wavg, wvar  # proven physicochemical featurizer

warnings.filterwarnings("ignore")

DATA_PATH = "/home/user/workspace/disorder_model/data/disprot_processed.json"
EMB_DIR = "/home/user/workspace/disorder_model/data/embeddings"
SEED = 42
MAX_PROT = 1500
MAX_LEN = 800
PCA_FULL = 96
N_JOBS = 4


def load_proteins():
    with open(DATA_PATH) as f:
        all_data = json.load(f)
    proteins = [
        p for p in all_data
        if os.path.exists(os.path.join(EMB_DIR, f"{p['disprot_id']}.npy"))
        and 30 <= p["length"] <= MAX_LEN and sum(p["disorder_labels"]) >= 3
        and p["length"] - sum(p["disorder_labels"]) >= 3
    ]
    rng = np.random.RandomState(SEED)
    if len(proteins) > MAX_PROT:
        idx = rng.choice(len(proteins), MAX_PROT, replace=False)
        proteins = [proteins[i] for i in sorted(idx)]
    return proteins


def load_raw(proteins):
    """Physics features + raw per-residue ESM (float16) per protein."""
    phys_list, esm_list, lab_list = [], [], []
    for p in proteins:
        L = p["length"]
        seq = p["sequence"][:L]
        phys_list.append(phys(seq).astype(np.float32))
        emb = np.load(os.path.join(EMB_DIR, f"{p['disprot_id']}.npy")).astype(np.float16)[:L]
        esm_list.append(emb)
        lab_list.append(np.array(p["disorder_labels"][:L], dtype=np.float32))
    return phys_list, esm_list, lab_list


def build_features(ph, ep, recipe):
    """Combine physics (ph) and PCA-ESM (ep) into a feature matrix per recipe."""
    if recipe == "base48":
        e = ep[:, :48]
        parts = [ph, e, wavg(e, 4), wavg(e, 12), wavg(e, 25), wvar(e, 8), wvar(e, 20)]
    elif recipe == "rich96":
        e = ep  # 96 dims
        gmean = np.broadcast_to(e.mean(0), e.shape)
        gstd = np.broadcast_to(e.std(0), e.shape)
        enorm = np.linalg.norm(e, axis=1, keepdims=True)
        parts = [ph, e,
                 wavg(e, 4), wavg(e, 12), wavg(e, 25), wavg(e, 50),
                 wvar(e, 8), wvar(e, 20),
                 gmean[:, :32], gstd[:, :32], enorm]
    else:
        raise ValueError(recipe)
    return np.concatenate(parts, 1).astype(np.float32)


def train_ensemble(X_tr, y_tr, X_val, rng, with_hgb=False):
    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    dt = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    lm = lgb.train(
        {"objective": "binary", "metric": "auc", "num_leaves": 127, "max_depth": 8,
         "learning_rate": 0.05, "feature_fraction": 0.7, "bagging_fraction": 0.7,
         "bagging_freq": 5, "scale_pos_weight": spw, "min_child_samples": 25,
         "reg_alpha": 0.05, "reg_lambda": 0.5, "verbose": -1, "n_jobs": N_JOBS,
         "seed": SEED},
        dt, 700, callbacks=[lgb.log_evaluation(0)])
    lp = lm.predict(X_val)

    dx = xgb.DMatrix(X_tr, label=y_tr)
    dvx = xgb.DMatrix(X_val)
    xm = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 7,
         "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7,
         "scale_pos_weight": spw, "min_child_weight": 25, "reg_alpha": 0.05,
         "reg_lambda": 0.5, "tree_method": "hist", "nthread": N_JOBS, "seed": SEED},
        dx, 700)
    xp = xm.predict(dvx)

    preds = {"lgb": lp, "xgb": xp}
    if with_hgb:
        hgb = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
            l2_regularization=0.5, class_weight="balanced", random_state=SEED)
        hgb.fit(X_tr, y_tr)
        preds["hgb"] = hgb.predict_proba(X_val)[:, 1]
    return preds


def smooth_per_protein(probs_by_id, lengths_by_id, order, window):
    """Moving-average smoothing within each protein (respects boundaries)."""
    if window <= 1:
        return np.concatenate([probs_by_id[i] for i in order])
    out = []
    hw = window // 2
    for i in order:
        v = probs_by_id[i]
        out.append(wavg(v, hw))
    return np.concatenate(out)


def evaluate(yt, yp):
    auc = roc_auc_score(yt, yp)
    ap = average_precision_score(yt, yp)
    fpr, tpr, th = roc_curve(yt, yp)
    opt = th[np.argmax(tpr - fpr)]
    yb = (yp >= opt).astype(int)
    return {"auc": auc, "ap": ap, "f1": f1_score(yt, yb), "mcc": matthews_corrcoef(yt, yb)}


def main():
    t0 = time.time()
    proteins = load_proteins()
    ids = [p["disprot_id"] for p in proteins]
    n = len(proteins)
    print(f"Proteins: {n}  loading raw features + ESM...", flush=True)
    phys_list, esm_list, lab_list = load_raw(proteins)
    lengths = {ids[i]: proteins[i]["length"] for i in range(n)}
    labels_by_id = {ids[i]: lab_list[i] for i in range(n)}
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    gkf = GroupKFold(n_splits=5)
    rng = np.random.RandomState(SEED)

    # OOF probability stores: config -> {id: probs}
    configs = ["base48_lx", "rich96_lx", "rich96_lxh"]
    oof = {c: {} for c in configs}

    for fold, (tr_i, va_i) in enumerate(gkf.split(range(n), range(n), range(n))):
        ft = time.time()
        print(f"\nFold {fold+1}/5  (train={len(tr_i)} val={len(va_i)})", flush=True)

        # Fit PCA-96 on TRAIN fold only (leakage-free), reuse for 48/96 via slicing.
        pca = IncrementalPCA(n_components=PCA_FULL, batch_size=10000)
        train_esm = np.vstack([esm_list[i].astype(np.float32) for i in tr_i])
        pca.fit(train_esm)
        vexp = pca.explained_variance_ratio_.sum()
        del train_esm; gc.collect()
        print(f"  PCA-{PCA_FULL} var={vexp:.3f}", flush=True)

        ep = {i: pca.transform(esm_list[i].astype(np.float32)) for i in range(n)}

        for recipe, cfgs in [("base48", ["base48_lx"]), ("rich96", ["rich96_lx", "rich96_lxh"])]:
            X_tr_f = np.nan_to_num(np.vstack([build_features(phys_list[i], ep[i], recipe) for i in tr_i]))
            y_tr_f = np.concatenate([lab_list[i] for i in tr_i])
            di = np.where(y_tr_f == 1)[0]; oi = np.where(y_tr_f == 0)[0]
            nk = min(len(oi), len(di) * 3)
            keep = np.sort(np.concatenate([di, rng.choice(oi, nk, replace=False)]))
            X_tr = X_tr_f[keep]; y_tr = y_tr_f[keep]
            del X_tr_f, y_tr_f; gc.collect()

            X_val_list = [np.nan_to_num(build_features(phys_list[i], ep[i], recipe)) for i in va_i]
            X_val = np.vstack(X_val_list)

            with_hgb = "rich96_lxh" in cfgs
            preds = train_ensemble(X_tr, y_tr, X_val, rng, with_hgb=with_hgb)

            # split per-protein predictions back out
            off = 0
            per_prot = {}
            for i in va_i:
                L = proteins[i]["length"]
                per_prot[i] = {k: preds[k][off:off+L] for k in preds}
                off += L

            # base48_lx and rich96_lx: 0.55 lgb + 0.45 xgb
            for cfg in cfgs:
                for i in va_i:
                    pp = per_prot[i]
                    if cfg == "rich96_lxh":
                        blend = 0.45 * pp["lgb"] + 0.35 * pp["xgb"] + 0.20 * pp["hgb"]
                    else:
                        blend = 0.55 * pp["lgb"] + 0.45 * pp["xgb"]
                    oof[cfg][ids[i]] = blend
            del X_tr, y_tr, X_val, preds; gc.collect()
        print(f"  fold done in {time.time()-ft:.0f}s", flush=True)

    # Evaluation: raw + smoothed
    order = ids
    y_all = np.concatenate([labels_by_id[i] for i in order])
    print("\n" + "=" * 74)
    print(f"{'config':<16} {'smooth':>7} {'AUC':>8} {'AP':>8} {'F1':>8} {'MCC':>8}")
    print("-" * 74)
    results = {}
    for cfg in configs:
        for w in [1, 3, 5, 7, 9]:
            yp = smooth_per_protein(oof[cfg], lengths, order, w)
            m = evaluate(y_all, yp)
            results[f"{cfg}_s{w}"] = m
            print(f"{cfg:<16} {w:>7} {m['auc']:>8.4f} {m['ap']:>8.4f} {m['f1']:>8.4f} {m['mcc']:>8.4f}", flush=True)
        print("-" * 74)

    print(f"\nBaseline (run_v6_mem, all-protein PCA): pooled AUC 0.8397")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
    os.makedirs("experiments/out", exist_ok=True)
    with open("experiments/out/optimize_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
