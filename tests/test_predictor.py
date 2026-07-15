"""Tests for the deployable disorder predictor bundle (predictor.py).

Uses small synthetic proteins with a label signal injected into the ESM channel,
so no real ESM model or download is needed and the tests run fast.
"""
from __future__ import annotations

import numpy as np
import pytest

from predictor import (build_features, fit_bundle, predict_from_embeddings,
                       phys_features, DECISION_LABEL)

AA = "ACDEFGHIKLMNPQRSTVWY"


def _synthetic(n=40, emb_dim=64, seed=0):
    rng = np.random.RandomState(seed)
    phys_list, esm_list, lab_list = [], [], []
    for _ in range(n):
        L = rng.randint(40, 90)
        seq = "".join(rng.choice(list(AA), L))
        # contiguous disorder block
        labels = np.zeros(L, dtype=np.float32)
        s = rng.randint(0, L - 15)
        labels[s:s + rng.randint(10, 20)] = 1.0
        emb = rng.randn(L, emb_dim).astype(np.float32)
        emb[:, :6] += labels[:, None] * 2.5  # learnable signal
        phys_list.append(phys_features(seq))
        esm_list.append(emb)
        lab_list.append(labels)
    return phys_list, esm_list, lab_list


def test_build_features_shape():
    ph = np.zeros((50, 118), dtype=np.float32)
    ep = np.zeros((50, 96), dtype=np.float32)
    X = build_features(ph, ep)
    assert X.shape[0] == 50 and X.shape[1] > 118


def test_fit_bundle_and_predict_roundtrip(tmp_path):
    import joblib

    phys_list, esm_list, lab_list = _synthetic()
    bundle = fit_bundle(phys_list, esm_list, lab_list, alpha=0.1, pca_dim=32, n_jobs=2)
    for k in ("pca", "lgb", "xgb", "hgb", "isotonic", "conformal_q", "alpha"):
        assert k in bundle

    # joblib round-trip
    p = tmp_path / "bundle.joblib"
    joblib.dump(bundle, p)
    bundle2 = joblib.load(p)

    out = predict_from_embeddings(bundle2, phys_list[0], esm_list[0])
    L = len(lab_list[0])
    for key in ("p_raw", "p_smoothed", "p_calibrated", "decision"):
        assert len(out[key]) == L
    assert out["p_calibrated"].min() >= 0.0 and out["p_calibrated"].max() <= 1.0
    assert set(np.unique(out["decision"])).issubset({-1, 0, 1, 2})
    assert all(int(d) in DECISION_LABEL for d in np.unique(out["decision"]))


def test_bundle_learns_signal():
    """Model should rank held-out disordered residues above ordered ones."""
    from sklearn.metrics import roc_auc_score

    phys_list, esm_list, lab_list = _synthetic(n=48, seed=1)
    bundle = fit_bundle(phys_list[:40], esm_list[:40], lab_list[:40], pca_dim=32, n_jobs=2)

    probs, labs = [], []
    for i in range(40, 48):
        out = predict_from_embeddings(bundle, phys_list[i], esm_list[i])
        probs.append(out["p_calibrated"]); labs.append(lab_list[i])
    auc = roc_auc_score(np.concatenate(labs), np.concatenate(probs))
    assert auc > 0.7  # strong injected signal -> easily separable
