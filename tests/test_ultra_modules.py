"""Tests for meta-ensemble and v6-pro modules."""

from __future__ import annotations

import numpy as np

from colab.meta_ensemble import apply_meta_stacker, fit_meta_stacker
from colab.rich_features import RICH_FEATURE_DIM, RichFeatureEncoder, compute_rich_features
from colab.v6_pro_ensemble import _predict_v6_pro, _train_v6_pro_fold


class TestRichFeatures:
    def test_feature_dim(self):
        feats = compute_rich_features("ACDEFGHIKLMNPQRSTVWY")
        assert feats.shape[1] == RICH_FEATURE_DIM

    def test_encoder_forward(self):
        import torch
        enc = RichFeatureEncoder(out_dim=32)
        x = torch.randn(2, 10, RICH_FEATURE_DIM)
        out = enc(x)
        assert out.shape == (2, 10, 32)


class TestMetaEnsemble:
    def test_fit_stacker(self):
        rng = np.random.default_rng(0)
        n = 400
        y = (rng.random(n) > 0.7).astype(np.float32)
        X = np.column_stack([y * 0.8 + 0.1, y * 0.7 + 0.15, y * 0.6 + 0.2])
        model, scaler = fit_meta_stacker(y, X)
        probs = model.predict_proba(scaler.transform(X))[:, 1]
        assert probs.min() >= 0 and probs.max() <= 1

    def test_apply_meta_stacker(self):
        proteins = [
            {"id": "P1", "length": 4, "labels": [0, 1, 1, 0]},
            {"id": "P2", "length": 4, "labels": [1, 0, 0, 1]},
        ]
        fold_results = [
            {
                "fold": 1,
                "val_probs": np.array([0.2, 0.8, 0.7, 0.3], dtype=np.float32),
                "val_labels": np.array([0, 1, 1, 0], dtype=np.float32),
            },
            {
                "fold": 2,
                "val_probs": np.array([0.9, 0.1, 0.2, 0.8], dtype=np.float32),
                "val_labels": np.array([1, 0, 0, 1], dtype=np.float32),
            },
        ]
        streams = {
            "gpu": {
                "P1": np.array([0.2, 0.8, 0.7, 0.3], dtype=np.float32),
                "P2": np.array([0.9, 0.1, 0.2, 0.8], dtype=np.float32),
            },
            "v6": {
                "P1": np.array([0.3, 0.7, 0.75, 0.25], dtype=np.float32),
                "P2": np.array([0.85, 0.15, 0.25, 0.75], dtype=np.float32),
            },
            "physics": {
                "P1": np.array([0.4, 0.6, 0.65, 0.35], dtype=np.float32),
                "P2": np.array([0.8, 0.2, 0.3, 0.7], dtype=np.float32),
            },
        }
        report, updated = apply_meta_stacker(proteins, fold_results, streams, n_folds=2)
        assert not report.get("skipped")
        assert len(updated) == 2


class TestV6Pro:
    def test_hgb_fallback_train_predict(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 20)).astype(np.float32)
        y = (X[:, 0] + rng.normal(size=200) * 0.1 > 0).astype(np.float32)
        models = _train_v6_pro_fold(X, y, seed=42)
        probs = _predict_v6_pro(models, X[:50])
        assert probs.shape == (50,)
        assert probs.min() >= 0 and probs.max() <= 1
