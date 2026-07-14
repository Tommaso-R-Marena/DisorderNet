"""Tests for Disorder → function multi-label prediction."""

from __future__ import annotations

import numpy as np
import torch

from colab.disordernet_gpu import FUNCTIONAL_TERM_GROUPS, TrainConfig
from colab.function_predict import (
    FUNCTION_GROUP_NAMES,
    N_FUNCTION_GROUPS,
    FunctionMultiLabelHead,
    build_function_labels,
    compute_function_metrics,
    function_multilabel_loss,
    function_supervise_mask,
    predict_protein_functions,
    run_function_prediction_report,
    split_function_oof_by_lengths,
    stack_batch_function_labels,
    summarize_function_label_coverage,
)


class TestFunctionLabels:
    def test_group_names_match_disprot_groups(self):
        assert FUNCTION_GROUP_NAMES == tuple(FUNCTIONAL_TERM_GROUPS.keys())
        assert N_FUNCTION_GROUPS == 5

    def test_build_labels_protein_binding(self):
        regions = [
            {"start": 1, "end": 5, "term_norm": "protein binding"},
            {"start": 3, "end": 8, "term_norm": "dna binding"},
        ]
        fl = build_function_labels(10, regions)
        assert fl.shape == (10, 5)
        assert fl[:5, 0].sum() == 5  # protein binding
        assert fl[2:8, 1].sum() == 6  # nucleic acid (dna binding)
        assert fl[8:, :].sum() == 0

    def test_supervise_mask_disordered_only(self):
        dis = np.array([0, 0, 1, 1, 0], dtype=np.float32)
        fl = np.zeros((5, 2), dtype=np.float32)
        fl[0, 0] = 1.0  # function on ordered residue
        mask = function_supervise_mask(dis, fl, disordered_only=True)
        assert mask.tolist() == [True, False, True, True, False]


class TestFunctionHead:
    def test_forward_shape(self):
        head = FunctionMultiLabelHead(in_dim=32, n_groups=5, mid=16)
        x = torch.randn(2, 12, 32)
        mask = torch.ones(2, 12, dtype=torch.bool)
        mask[0, 10:] = False
        out = head(x, pad_mask=mask)
        assert out.shape == (2, 12, 5)

    def test_loss_runs(self):
        logits = torch.randn(2, 8, 5)
        labels = (torch.rand(2, 8, 5) > 0.8).float()
        pad = torch.ones(2, 8, dtype=torch.bool)
        sup = torch.ones(2, 8, dtype=torch.bool)
        loss = function_multilabel_loss(logits, labels, pad, sup)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_batch_stack(self):
        proteins = {
            "a": {
                "id": "a",
                "length": 6,
                "labels": [0, 1, 1, 1, 0, 0],
                "functional_regions": [
                    {"start": 2, "end": 4, "term_norm": "protein binding"},
                ],
            }
        }
        labels, sup = stack_batch_function_labels(
            ["a"], proteins, max_seq=8, device=torch.device("cpu"),
        )
        assert labels.shape == (1, 8, 5)
        assert labels[0, 1:4, 0].sum().item() == 3
        assert bool(sup[0, 1].item()) is True


class TestFunctionMetrics:
    def test_metrics_perfect(self):
        y_true = np.zeros((100, 5), dtype=np.float32)
        y_true[:40, 0] = 1
        y_prob = y_true.copy()
        y_prob[:40, 0] = 0.9
        y_prob[40:, 0] = 0.1
        m = compute_function_metrics(y_true, y_prob)
        assert m["per_group"]["protein binding"]["auc"] is not None
        assert m["per_group"]["protein binding"]["auc"] > 0.95

    def test_report_without_oof(self):
        proteins = [
            {
                "id": "p1",
                "length": 20,
                "labels": [0] * 20,
                "functional_regions": [],
            }
        ]
        report = run_function_prediction_report(proteins, [{}], n_folds=1)
        assert report["enabled"] is False

    def test_split_oof_by_lengths(self):
        flat = np.ones((10, 5), dtype=np.float32)
        flat[5:] = 2.0
        out = split_function_oof_by_lengths(flat, ["a", "b"], {"a": 5, "b": 5})
        assert out["a"].shape == (5, 5)
        assert float(out["b"][0, 0]) == 2.0

    def test_report_with_aligned_oof_and_disordered_metrics(self):
        from colab.cv_splits import get_cv_splits
        from colab.biological_utility import align_fold_predictions

        proteins = [
            {
                "id": "a",
                "sequence": "AAAA",
                "length": 4,
                "labels": [1, 1, 0, 0],
                "functional_regions": [
                    {"start": 1, "end": 2, "term_norm": "protein binding"},
                ],
            },
            {
                "id": "b",
                "sequence": "BBBB",
                "length": 4,
                "labels": [0, 0, 1, 1],
                "functional_regions": [],
            },
        ]
        n_folds = 2
        splits = get_cv_splits(proteins, n_folds)
        fold_results = []
        for _, val_idx in splits:
            # Concatenate val proteins in fold order (same as train/eval)
            probs = []
            labels = []
            fn_probs = []
            fn_labels = []
            for i in val_idx:
                p = proteins[i]
                L = p["length"]
                labs = np.asarray(p["labels"], dtype=np.float32)
                pr = labs * 0.9 + (1 - labs) * 0.1
                probs.append(pr)
                labels.append(labs)
                fp = np.zeros((L, 5), dtype=np.float32)
                ft = np.zeros((L, 5), dtype=np.float32)
                if p["id"] == "a":
                    fp[:2, 0] = 0.9
                    ft[:2, 0] = 1.0
                fn_probs.append(fp)
                fn_labels.append(ft)
            fold_results.append({
                "val_probs": np.concatenate(probs),
                "val_labels": np.concatenate(labels),
                "val_function_probs": np.concatenate(fn_probs),
                "val_function_labels": np.concatenate(fn_labels),
            })

        aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)
        assert len(aligned) == 2
        report = run_function_prediction_report(proteins, fold_results, n_folds=n_folds)
        assert report["enabled"] is True
        assert report["n_oof_residues"] == 8
        assert report["metrics"] is not None
        # 4 disordered residues (≥10 needed for disordered metrics) → may be None
        assert "metrics_on_disordered" in report

    def test_coverage_summary(self):
        proteins = [
            {
                "id": "p1",
                "length": 10,
                "labels": [1] * 10,
                "functional_regions": [
                    {"start": 1, "end": 3, "term_norm": "protein binding"},
                ],
            }
        ]
        cov = summarize_function_label_coverage(proteins)
        assert cov["residues_per_group"]["protein binding"] == 3


class TestIDRAnnotation:
    def test_predict_protein_functions(self):
        seq = "A" * 30
        dis = np.concatenate([np.ones(15), np.zeros(15)]).astype(np.float32)
        fn = np.zeros((30, 5), dtype=np.float32)
        fn[:15, 0] = 0.8
        out = predict_protein_functions(dis, fn, seq, protein_id="x")
        assert out["n_idr_segments"] >= 1
        assert out["idr_function_regions"][0]["predicted_roles"][0]["group"] == "protein binding"


class TestUltraFunProfile:
    def test_ultra_fun_enables_head(self):
        cfg = TrainConfig.from_profile("ultra_fun")
        assert cfg.use_function_head is True
        assert cfg.function_loss_weight == 0.35
        assert cfg.use_rich_features is True
        assert cfg.head_type == "sota"

    def test_cli_override_flag_in_profile_base(self):
        cfg = TrainConfig.from_profile("ultra", use_function_head=True)
        assert cfg.use_function_head is True
