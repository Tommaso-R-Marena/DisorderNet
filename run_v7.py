"""DisorderNet v7 (CPU): optimized ensemble + calibrated conformal confidence.

Improvements over v6 (run_v6_mem.py), all measured with leakage-free 5-fold CV
(PCA fit on the train fold only):

  * Richer ESM features: PCA-96 (vs 48) + global mean/std pooling + per-residue norm
    + an extra long-range window scale.
  * 3-way GBDT blend: LightGBM + XGBoost + HistGradientBoosting.
  * Contiguity-aware smoothing of per-residue probabilities (disorder is contiguous).

New capabilities (see confidence.py) that typical IDR predictors lack:

  * Isotonic probability calibration + Expected Calibration Error, cross-fitted
    across folds (no leakage).
  * Split-conformal per-residue prediction sets with a coverage guarantee, yielding
    "confident disorder / confident order / abstain" decisions at a chosen risk level.

Requires the DisProt cache + ESM embeddings produced by fetch_disprot.py /
extract_esm_embeddings.py. Override the data location with DISORDERNET_HOME.

Run:  python run_v7.py
"""
import json, os, time, gc, warnings
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, roc_curve, balanced_accuracy_score,
                             precision_score, recall_score)
import lightgbm as lgb
import xgboost as xgb

from run_v6_mem import phys, wavg, wvar
from confidence import (expected_calibration_error, fit_calibrator, apply_calibrator,
                        conformal_quantile, conformal_report)

warnings.filterwarnings("ignore")

_HOME = os.environ.get("DISORDERNET_HOME", "/home/user/workspace/disorder_model")
DATA_PATH = os.path.join(_HOME, "data", "disprot_processed.json")
EMB_DIR = os.path.join(_HOME, "data", "embeddings")
RESULTS_DIR = os.path.join(_HOME, "results_v7")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
MAX_PROT = 1500
MAX_LEN = 800
PCA_DIM = 96
SMOOTH_WINDOW = 7
CONFORMAL_ALPHA = 0.10
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


def build_features(ph, ep):
    """Rich feature recipe (phys + PCA-ESM context + global pooling)."""
    gmean = np.broadcast_to(ep.mean(0), ep.shape)
    gstd = np.broadcast_to(ep.std(0), ep.shape)
    enorm = np.linalg.norm(ep, axis=1, keepdims=True)
    parts = [ph, ep,
             wavg(ep, 4), wavg(ep, 12), wavg(ep, 25), wavg(ep, 50),
             wvar(ep, 8), wvar(ep, 20),
             gmean[:, :32], gstd[:, :32], enorm]
    return np.concatenate(parts, 1).astype(np.float32)


def train_blend(X_tr, y_tr, X_val):
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

    xm = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 7,
         "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7,
         "scale_pos_weight": spw, "min_child_weight": 25, "reg_alpha": 0.05,
         "reg_lambda": 0.5, "tree_method": "hist", "nthread": N_JOBS, "seed": SEED},
        xgb.DMatrix(X_tr, label=y_tr), 700)
    xp = xm.predict(xgb.DMatrix(X_val))

    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=0.5, class_weight="balanced", random_state=SEED)
    hgb.fit(X_tr, y_tr)
    hp = hgb.predict_proba(X_val)[:, 1]

    return 0.45 * lp + 0.35 * xp + 0.20 * hp


def smooth(v, window):
    return wavg(v, window // 2) if window > 1 else v


def evaluate(yt, yp):
    fpr, tpr, th = roc_curve(yt, yp)
    opt = th[np.argmax(tpr - fpr)]
    yb = (yp >= opt).astype(int)
    return {
        "auc_roc": float(roc_auc_score(yt, yp)),
        "avg_precision": float(average_precision_score(yt, yp)),
        "f1": float(f1_score(yt, yb)),
        "mcc": float(matthews_corrcoef(yt, yb)),
        "precision": float(precision_score(yt, yb)),
        "recall": float(recall_score(yt, yb)),
        "balanced_acc": float(balanced_accuracy_score(yt, yb)),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("DisorderNet v7: optimized ensemble + conformal confidence")
    print("=" * 70)
    proteins = load_proteins()
    n = len(proteins)
    ids = [p["disprot_id"] for p in proteins]
    tr = sum(p["length"] for p in proteins)
    td = sum(sum(p["disorder_labels"]) for p in proteins)
    print(f"Proteins: {n} | Residues: {tr:,} ({100*td/tr:.1f}% dis)", flush=True)

    phys_list, esm_list, lab_list = [], [], []
    for p in proteins:
        L = p["length"]
        phys_list.append(phys(p["sequence"][:L]).astype(np.float32))
        esm_list.append(np.load(os.path.join(EMB_DIR, f"{p['disprot_id']}.npy")).astype(np.float16)[:L])
        lab_list.append(np.array(p["disorder_labels"][:L], dtype=np.float32))

    gkf = GroupKFold(n_splits=5)
    rng = np.random.RandomState(SEED)
    oof_raw, oof_lab, fold_of = {}, {}, {}

    for fold, (tr_i, va_i) in enumerate(gkf.split(range(n), range(n), range(n))):
        ft = time.time()
        pca = IncrementalPCA(n_components=PCA_DIM, batch_size=10000)
        pca.fit(np.vstack([esm_list[i].astype(np.float32) for i in tr_i]))
        ep = {i: pca.transform(esm_list[i].astype(np.float32)) for i in np.concatenate([tr_i, va_i])}

        X_tr_f = np.nan_to_num(np.vstack([build_features(phys_list[i], ep[i]) for i in tr_i]))
        y_tr_f = np.concatenate([lab_list[i] for i in tr_i])
        di = np.where(y_tr_f == 1)[0]; oi = np.where(y_tr_f == 0)[0]
        nk = min(len(oi), len(di) * 3)
        keep = np.sort(np.concatenate([di, rng.choice(oi, nk, replace=False)]))
        X_tr, y_tr = X_tr_f[keep], y_tr_f[keep]
        del X_tr_f, y_tr_f; gc.collect()

        X_val = np.vstack([np.nan_to_num(build_features(phys_list[i], ep[i])) for i in va_i])
        blend = train_blend(X_tr, y_tr, X_val)

        off = 0
        for i in va_i:
            L = proteins[i]["length"]
            oof_raw[ids[i]] = blend[off:off + L]
            oof_lab[ids[i]] = lab_list[i]
            fold_of[ids[i]] = fold
            off += L
        del X_tr, y_tr, X_val, ep; gc.collect()
        print(f"  Fold {fold+1}/5 done [{time.time()-ft:.0f}s]", flush=True)

    # Pool (raw + smoothed)
    y_all = np.concatenate([oof_lab[i] for i in ids])
    raw = np.concatenate([oof_raw[i] for i in ids])
    smoothed = np.concatenate([smooth(oof_raw[i], SMOOTH_WINDOW) for i in ids])

    m_raw = evaluate(y_all, raw)
    m_smooth = evaluate(y_all, smoothed)

    # Cross-fitted isotonic calibration (fit on other folds, apply to held-out fold)
    smoothed_by_id = {i: smooth(oof_raw[i], SMOOTH_WINDOW) for i in ids}
    calibrated = np.empty_like(smoothed)
    conf_decisions = np.empty(len(smoothed), dtype=np.int64)
    q_by_fold = {}
    pos = {}
    off = 0
    for i in ids:
        pos[i] = (off, off + len(oof_lab[i])); off += len(oof_lab[i])

    for k in range(5):
        cal_p = np.concatenate([smoothed_by_id[i] for i in ids if fold_of[i] != k])
        cal_y = np.concatenate([oof_lab[i] for i in ids if fold_of[i] != k])
        iso = fit_calibrator(cal_p, cal_y)
        q = conformal_quantile(iso.predict(cal_p), cal_y, alpha=CONFORMAL_ALPHA, class_conditional=True)
        q_by_fold[k] = q
        for i in ids:
            if fold_of[i] != k:
                continue
            s, e = pos[i]
            cp = apply_calibrator(iso, smoothed_by_id[i])
            calibrated[s:e] = cp
            from confidence import conformal_sets
            conf_decisions[s:e] = conformal_sets(cp, q)["decision"]

    ece_raw = expected_calibration_error(y_all, smoothed)
    ece_cal = expected_calibration_error(y_all, calibrated)
    covered = np.where(y_all == 1,
                       np.isin(conf_decisions, (1,)) | (conf_decisions == -1),
                       np.isin(conf_decisions, (0,)) | (conf_decisions == -1))
    conf_rep = {
        "alpha": CONFORMAL_ALPHA,
        "target_coverage": 1 - CONFORMAL_ALPHA,
        "empirical_coverage": float(covered.mean()),
        "abstain_rate": float((conf_decisions == -1).mean()),
        "confident_rate": float(np.isin(conf_decisions, (0, 1)).mean()),
    }
    conf_mask = np.isin(conf_decisions, (0, 1))
    if conf_mask.any():
        conf_rep["selective_accuracy"] = float((conf_decisions[conf_mask] == y_all[conf_mask]).mean())

    # Report
    print(f"\n{'='*70}\nRESULTS (leakage-free 5-fold CV, train-only PCA)\n{'='*70}")
    print(f"  {'metric':<16}{'raw':>10}{'+smooth':>10}")
    for k in ["auc_roc", "avg_precision", "f1", "mcc", "balanced_acc"]:
        print(f"  {k:<16}{m_raw[k]:>10.4f}{m_smooth[k]:>10.4f}")
    print(f"\n  v6 baseline pooled AUC (run_v6_mem): 0.8397")
    print(f"  v7 pooled AUC (smoothed):            {m_smooth['auc_roc']:.4f}")
    print(f"\n  Calibration ECE: {ece_raw:.4f} -> {ece_cal:.4f} (isotonic)")
    print(f"  Conformal @ alpha={CONFORMAL_ALPHA} (target cov {1-CONFORMAL_ALPHA:.2f}): "
          f"coverage={conf_rep['empirical_coverage']:.3f} "
          f"confident={conf_rep['confident_rate']:.3f} "
          f"abstain={conf_rep['abstain_rate']:.3f}")
    if "selective_accuracy" in conf_rep:
        print(f"  Selective accuracy on confident residues: {conf_rep['selective_accuracy']:.3f}")

    results = {
        "model": "DisorderNet_v7",
        "pooled_raw": m_raw,
        "pooled_smoothed": m_smooth,
        "smooth_window": SMOOTH_WINDOW,
        "pca_dim": PCA_DIM,
        "n_proteins": n,
        "calibration": {"ece_raw": ece_raw, "ece_calibrated": ece_cal},
        "conformal": conf_rep,
        "v6_baseline_auc": 0.8397,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "y_true.npy"), y_all)
    np.save(os.path.join(RESULTS_DIR, "y_pred.npy"), smoothed)
    np.save(os.path.join(RESULTS_DIR, "y_pred_calibrated.npy"), calibrated)
    np.save(os.path.join(RESULTS_DIR, "conformal_decisions.npy"), conf_decisions)
    print(f"\nSaved to {RESULTS_DIR}/  | total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
