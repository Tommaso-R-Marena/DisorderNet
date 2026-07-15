"""Deployable disorder predictor with calibrated + conformal per-residue confidence.

Turns the DisorderNet v7 CPU model into a reusable predictor. A trained *bundle*
holds everything needed for inference:

    PCA (fit on training ESM) + LightGBM/XGBoost/HistGBM ensemble
    + isotonic calibrator + conformal thresholds.

`fit_bundle` trains it (splitting off a calibration set so the calibrator and the
conformal thresholds are estimated on held-out data — no leakage). `predict_from_embeddings`
runs inference from precomputed ESM embeddings, returning for each residue:

    p_raw        - blended ensemble probability
    p_calibrated - isotonic-calibrated probability (trustworthy)
    decision     - conformal set decision: 1=disorder, 0=order, -1=abstain, 2=empty

Both functions are pure (no ESM download, no file I/O), so they are fast to unit-test.
The CLIs live in train_predictor.py / predict_disorder.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

from run_v6_mem import phys, wavg, wvar
from confidence import fit_calibrator, apply_calibrator, conformal_quantile, conformal_sets

SEED = 42
PCA_DIM = 96
SMOOTH_WINDOW = 7
BLEND = (0.45, 0.35, 0.20)  # lgb, xgb, hgb
ESM_MODEL = "esm2_t12_35M_UR50D"
ESM_REPR_LAYER = 12


def build_features(ph: np.ndarray, ep: np.ndarray) -> np.ndarray:
    """Rich feature recipe (physics + PCA-ESM context + global pooling)."""
    gmean = np.broadcast_to(ep.mean(0), ep.shape)
    gstd = np.broadcast_to(ep.std(0), ep.shape)
    enorm = np.linalg.norm(ep, axis=1, keepdims=True)
    parts = [ph, ep,
             wavg(ep, 4), wavg(ep, 12), wavg(ep, 25), wavg(ep, 50),
             wvar(ep, 8), wvar(ep, 20),
             gmean[:, :32], gstd[:, :32], enorm]
    return np.concatenate(parts, 1).astype(np.float32)


def _smooth(v: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    return wavg(v, window // 2) if window > 1 else v


def phys_features(seq: str) -> np.ndarray:
    return phys(seq).astype(np.float32)


def _train_models(X_tr, y_tr, rng, n_jobs=4):
    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    lm = lgb.train(
        {"objective": "binary", "metric": "auc", "num_leaves": 127, "max_depth": 8,
         "learning_rate": 0.05, "feature_fraction": 0.7, "bagging_fraction": 0.7,
         "bagging_freq": 5, "scale_pos_weight": spw, "min_child_samples": 25,
         "reg_alpha": 0.05, "reg_lambda": 0.5, "verbose": -1, "n_jobs": n_jobs,
         "seed": SEED},
        lgb.Dataset(X_tr, label=y_tr), 700, callbacks=[lgb.log_evaluation(0)])
    xm = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 7,
         "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7,
         "scale_pos_weight": spw, "min_child_weight": 25, "reg_alpha": 0.05,
         "reg_lambda": 0.5, "tree_method": "hist", "nthread": n_jobs, "seed": SEED},
        xgb.DMatrix(X_tr, label=y_tr), 700)
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=0.5, class_weight="balanced", random_state=SEED)
    hgb.fit(X_tr, y_tr)
    return lm, xm, hgb


def _blend(lm, xm, hgb, X):
    lp = lm.predict(X)
    xp = xm.predict(xgb.DMatrix(X))
    hp = hgb.predict_proba(X)[:, 1]
    return BLEND[0] * lp + BLEND[1] * xp + BLEND[2] * hp


def fit_bundle(phys_list, esm_list, lab_list, alpha=0.10, calib_frac=0.2,
               seed=SEED, n_jobs=4, pca_dim=PCA_DIM):
    """Train a deployable predictor bundle.

    A protein-level calibration split (``calib_frac``) is held out to fit the
    isotonic calibrator and conformal thresholds without leakage.
    """
    rng = np.random.RandomState(seed)
    n = len(phys_list)
    perm = rng.permutation(n)
    n_cal = max(1, int(round(n * calib_frac)))
    cal_idx = set(perm[:n_cal].tolist())
    fit_idx = [i for i in range(n) if i not in cal_idx]
    cal_idx = sorted(cal_idx)

    pca = IncrementalPCA(n_components=pca_dim, batch_size=10000)
    pca.fit(np.vstack([np.asarray(esm_list[i], dtype=np.float32) for i in fit_idx]))

    def feats(i):
        ep = pca.transform(np.asarray(esm_list[i], dtype=np.float32))
        return np.nan_to_num(build_features(phys_list[i], ep))

    X_fit = np.vstack([feats(i) for i in fit_idx])
    y_fit = np.concatenate([lab_list[i] for i in fit_idx])
    di = np.where(y_fit == 1)[0]; oi = np.where(y_fit == 0)[0]
    nk = min(len(oi), len(di) * 3)
    keep = np.sort(np.concatenate([di, rng.choice(oi, nk, replace=False)])) if len(di) else np.arange(len(y_fit))
    lm, xm, hgb = _train_models(X_fit[keep], y_fit[keep], rng, n_jobs=n_jobs)

    # Calibration set predictions (per protein, smoothed)
    cal_probs, cal_labels = [], []
    for i in cal_idx:
        p = _smooth(_blend(lm, xm, hgb, feats(i)))
        cal_probs.append(p)
        cal_labels.append(np.asarray(lab_list[i], dtype=np.float32))
    cal_probs = np.concatenate(cal_probs)
    cal_labels = np.concatenate(cal_labels)

    iso = fit_calibrator(cal_probs, cal_labels)
    cal_calibrated = apply_calibrator(iso, cal_probs)
    q = conformal_quantile(cal_calibrated, cal_labels, alpha=alpha, class_conditional=True)

    return {
        "pca": pca,
        "lgb": lm,
        "xgb": xm,
        "hgb": hgb,
        "isotonic": iso,
        "conformal_q": q,
        "alpha": alpha,
        "smooth_window": SMOOTH_WINDOW,
        "pca_dim": pca_dim,
        "blend": BLEND,
        "esm_model": ESM_MODEL,
        "esm_repr_layer": ESM_REPR_LAYER,
    }


def predict_from_embeddings(bundle: dict, phys_feats: np.ndarray, emb: np.ndarray) -> dict:
    """Per-residue prediction for one protein from its physics features + ESM embedding."""
    ep = bundle["pca"].transform(np.asarray(emb, dtype=np.float32))
    X = np.nan_to_num(build_features(phys_feats, ep))
    raw = _blend(bundle["lgb"], bundle["xgb"], bundle["hgb"], X)
    smoothed = _smooth(raw, bundle.get("smooth_window", SMOOTH_WINDOW))
    calibrated = apply_calibrator(bundle["isotonic"], smoothed)
    decision = conformal_sets(calibrated, bundle["conformal_q"])["decision"]
    return {
        "p_raw": raw.astype(np.float32),
        "p_smoothed": smoothed.astype(np.float32),
        "p_calibrated": calibrated.astype(np.float32),
        "decision": decision,
    }


DECISION_LABEL = {1: "disorder", 0: "order", -1: "abstain", 2: "uncertain"}
