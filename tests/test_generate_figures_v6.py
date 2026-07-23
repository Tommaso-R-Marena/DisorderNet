"""Test the v6 figure generator writes all figures from synthetic results."""
from __future__ import annotations

import json
import os

import numpy as np

from generate_figures_v6 import generate


def test_generate_writes_all_figures(tmp_path):
    rng = np.random.RandomState(0)
    n = 2000
    y = rng.randint(0, 2, n).astype(np.float32)
    # scores correlated with labels so ROC/PR are well-defined
    p = np.clip(0.5 * y + 0.25 * rng.rand(n), 0, 1).astype(np.float32)
    np.save(tmp_path / "y_true.npy", y)
    np.save(tmp_path / "y_pred.npy", p)
    (tmp_path / "metrics.json").write_text(json.dumps({
        "fold_aucs": [0.83, 0.84, 0.82, 0.85, 0.83],
    }))

    saved = generate(tmp_path)
    assert len(saved) == 4
    for p_ in saved:
        assert os.path.exists(p_) and os.path.getsize(p_) > 0
    for name in ("fig1_roc_pr.png", "fig2_comparison.png",
                 "fig3_progression.png", "fig4_folds.png"):
        assert (tmp_path / name).exists()
