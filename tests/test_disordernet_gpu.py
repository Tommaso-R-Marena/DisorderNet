"""Tests for colab/disordernet_gpu.py (CPU-safe unit tests)."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from colab.disordernet_gpu import (
    TrainConfig,
    LoRALinear,
    DisorderCNNHead,
    DisorderNetGPU,
    DisProtDataset,
    _auto_batch_size,
    _compute_pos_weight,
    _disorder_loss,
    _label_boundary_mask,
    _region_is_disorder,
    _warmup_cosine_scheduler,
    disprot_collate,
    eval_epoch,
    fetch_disprot,
    process_disprot,
)
from tests.conftest import MockESM


class TestRegionFiltering:
    def test_disorder_terms_accepted(self):
        assert _region_is_disorder("disorder")
        assert _region_is_disorder("Disorder")
        assert _region_is_disorder("flexible linker")
        assert _region_is_disorder("molten globule")

    def test_functional_terms_rejected(self):
        assert not _region_is_disorder("protein binding")
        assert not _region_is_disorder("DNA binding")

    def test_order_terms_rejected(self):
        assert not _region_is_disorder("order")
        assert not _region_is_disorder("order to disorder")

    def test_empty_rejected(self):
        assert not _region_is_disorder(None)
        assert not _region_is_disorder("")


class TestProcessDisprot:
    def test_filters_and_labels(self, sample_disprot_entries):
        cfg = TrainConfig(min_seq_len=20, min_disorder=3, min_order=3)
        proteins, skipped = process_disprot(sample_disprot_entries, cfg)

        ids = {p["id"] for p in proteins}
        assert "DP_TEST01" in ids
        assert "DP_TEST06" in ids
        assert "DP_TEST03" not in ids  # no disorder term
        assert "DP_TEST04" not in ids  # order only
        assert skipped["no_sequence"] == 1
        assert skipped["no_disorder_annotation"] >= 1

        p1 = next(p for p in proteins if p["id"] == "DP_TEST01")
        assert p1["n_dis"] == 10
        assert sum(p1["labels"][30:40]) == 10
        assert sum(p1["labels"][:30]) == 0
        assert len(p1["functional_regions"]) >= 2
        assert p1.get("uniprot_acc") == "P12345"
        assert sum(p1["transition_mask"][49:55]) > 0

    def test_too_short_skipped(self):
        cfg = TrainConfig(min_seq_len=50)
        entries = [{
            "disprot_id": "X",
            "sequence": "A" * 30,
            "regions": [{"start": 1, "end": 10, "term_name": "disorder"}],
        }]
        proteins, skipped = process_disprot(entries, cfg)
        assert len(proteins) == 0
        assert skipped["too_short"] == 1


class TestAutoBatchSize:
    @pytest.mark.parametrize(
        "vram,expected",
        [(80, (8, 2)), (40, (6, 3)), (24, (4, 4)), (16, (2, 8)), (8, (1, 16))],
    )
    def test_vram_tiers(self, vram, expected):
        assert _auto_batch_size(vram) == expected


class TestTrainConfig:
    def test_effective_batch(self):
        cfg = TrainConfig(batch_size=4, accum_steps=4)
        assert cfg.effective_batch() == 16

    def test_from_profile_max(self):
        cfg = TrainConfig.from_profile("max")
        assert cfg.lora_rank == 32
        assert cfg.physico_dim == 48


class TestLoRALinear:
    def test_forward_shape_and_frozen_base(self):
        base = nn.Linear(64, 32)
        lora = LoRALinear(base, rank=4, alpha=8)
        x = torch.randn(3, 64)
        out = lora(x)
        assert out.shape == (3, 32)
        assert not any(p.requires_grad for p in base.parameters())
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad


class TestDisorderCNNHead:
    def test_output_shape(self):
        head = DisorderCNNHead(in_dim=1280)
        x = torch.randn(2, 50, 1280)
        assert head(x).shape == (2, 50)


class TestDisorderNetGPU:
    def test_lora_injection_and_forward(self, mock_esm):
        cfg = TrainConfig(lora_layers=2, lora_rank=4, lora_on_k=True, use_physico_features=False)
        model = DisorderNetGPU(mock_esm, cfg, verbose=False)
        assert len(model._lora_modules) == 6  # 2 layers × (q, v, k)

        tokens = torch.randint(0, 20, (2, 32))
        logits = model(tokens)
        assert logits.shape == (2, 30)  # strip BOS/EOS

    def test_lora_injection_without_k(self, mock_esm):
        cfg = TrainConfig(lora_layers=2, lora_rank=4, lora_on_k=False, use_physico_features=False)
        model = DisorderNetGPU(mock_esm, cfg, verbose=False)
        assert len(model._lora_modules) == 4

    def test_physico_forward(self, mock_esm):
        cfg = TrainConfig(lora_layers=2, use_physico_features=True, physico_dim=16)
        model = DisorderNetGPU(mock_esm, cfg, verbose=False)
        tokens = torch.randint(0, 20, (2, 32))
        aa_idx = torch.randint(0, 20, (2, 30))
        logits = model(tokens, aa_idx=aa_idx)
        assert logits.shape == (2, 30)

    def test_train_keeps_esm_in_eval(self, mock_esm):
        cfg = TrainConfig(lora_layers=1)
        model = DisorderNetGPU(mock_esm, cfg, verbose=False)
        model.train()
        assert model.training
        assert not model.esm.training


class TestDataset:
    def test_collate_padding(self):
        batch = [
            (torch.arange(12), torch.zeros(10), torch.ones(10, dtype=torch.bool),
             torch.zeros(10, dtype=torch.long), torch.ones(10), "a"),
            (torch.arange(8), torch.ones(6), torch.ones(6, dtype=torch.bool),
             torch.zeros(6, dtype=torch.long), torch.ones(6), "b"),
        ]
        tok, lab, msk, aa, wt, ids = disprot_collate(batch)
        assert tok.shape == (2, 12)
        assert lab.shape == (2, 10)
        assert msk.shape == (2, 10)
        assert aa.shape == (2, 10)
        assert wt.shape == (2, 10)
        assert ids == ["a", "b"]
        assert msk[1, 6:].sum() == 0

    def test_dataset_cache(self, mock_batch_converter):
        proteins = [
            {"id": "P1", "sequence": "ACDE", "labels": [0, 1, 1, 0]},
            {"id": "P2", "sequence": "FGHI", "labels": [1, 0, 0, 1]},
        ]
        cache: dict = {}
        ds1 = DisProtDataset(proteins[:1], mock_batch_converter, cache)
        assert "P1" in cache
        ds2 = DisProtDataset(proteins, mock_batch_converter, cache)
        assert len(ds2) == 2
        tok, lab, msk, aa, wt, pid = ds2[0]
        assert pid == "P1"
        assert lab.shape[0] == msk.sum()
        assert aa.shape[0] == lab.shape[0]


class TestPosWeight:
    def test_capped_at_twelve(self):
        proteins = [{"n_dis": 1, "length": 1000}]
        pw = _compute_pos_weight(proteins, torch.device("cpu"))
        assert pw.item() == 12.0


class TestBoundaryAndLoss:
    def test_boundary_mask_flags_transitions(self):
        labels = [0, 0, 1, 1, 1, 0, 0]
        m = _label_boundary_mask(labels, radius=1)
        assert m[1] == 1.0
        assert m[2] == 1.0
        assert m[5] == 1.0

    def test_focal_loss_finite(self):
        cfg = TrainConfig(use_focal_loss=True)
        logits = torch.randn(8)
        labels = torch.tensor([0., 1., 0., 1., 1., 0., 1., 0.])
        w = torch.tensor([0., 1., 0., 0., 1., 0., 0., 0.])
        loss = _disorder_loss(logits, labels, None, w, cfg)
        assert loss.item() > 0


class TestScheduler:
    def test_warmup_then_decay(self):
        param = nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=1.0)
        sched = _warmup_cosine_scheduler(opt, total_steps=100, warmup_steps=10)
        lrs = []
        for _ in range(100):
            opt.step()
            sched.step()
            lrs.append(sched.get_last_lr()[0])
        assert lrs[0] < lrs[9]
        assert lrs[-1] < lrs[50]


class TestFetchDisprot:
    def test_loads_from_cache(self, disprot_cache_file):
        data = fetch_disprot(disprot_cache_file)
        assert len(data) == 6
        assert data[0]["disprot_id"] == "DP_TEST01"


class TestEvalEpoch:
    def test_cpu_eval_with_tiny_model(self, mock_esm):
        cfg = TrainConfig(lora_layers=1, num_workers=0, use_physico_features=False)
        model = DisorderNetGPU(mock_esm, cfg, verbose=False)
        proteins = [
            {"id": "P1", "sequence": "ACDE", "labels": [0, 1, 1, 0]},
            {"id": "P2", "sequence": "FGHI", "labels": [1, 0, 0, 1]},
        ]

        def converter(data):
            tokens = torch.zeros(len(data), 8, dtype=torch.long)
            for i in range(len(data)):
                tokens[i, :6] = torch.arange(6)
            return None, None, tokens

        ds = DisProtDataset(proteins, converter)
        dl = DataLoader(ds, batch_size=2, collate_fn=disprot_collate, num_workers=0)
        metrics = eval_epoch(
            model, dl, torch.device("cpu"), torch.float32, cfg=cfg,
        )
        assert 0.0 <= metrics["auc"] <= 1.0
        assert "probs" in metrics
        assert len(metrics["probs"]) > 0


class TestSetupEnvironment:
    def test_raises_without_gpu(self):
        cfg = TrainConfig()
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="No GPU detected"):
                from colab.disordernet_gpu import setup_environment
                setup_environment(cfg)


class TestCvProgress:
    def test_roundtrip_save_load(self, tmp_path):
        from colab.disordernet_gpu import load_cv_progress, save_cv_progress

        cfg = TrainConfig(seed=7, n_folds=2)
        proteins = [{"id": f"P{i}", "length": 10} for i in range(6)]
        fold_results = [
            {
                "fold": 1,
                "best_auc": 0.81,
                "best_ap": 0.42,
                "history": [],
                "val_probs": np.array([0.1, 0.9], dtype=np.float32),
                "val_labels": np.array([0.0, 1.0], dtype=np.float32),
                "ckpt_path": "fold1_best.pt",
                "total_time": 100.0,
            }
        ]
        path = str(tmp_path / "cv_progress.json")
        save_cv_progress(path, fold_results, cfg, proteins)
        loaded = load_cv_progress(path, cfg, proteins)
        assert len(loaded) == 1
        assert loaded[0]["best_auc"] == 0.81
        np.testing.assert_allclose(loaded[0]["val_probs"], [0.1, 0.9])
        np.testing.assert_allclose(loaded[0]["val_labels"], [0.0, 1.0])

    def test_mismatch_returns_empty(self, tmp_path):
        from colab.disordernet_gpu import load_cv_progress, save_cv_progress

        cfg = TrainConfig(seed=7, n_folds=2)
        proteins = [{"id": f"P{i}", "length": 10} for i in range(4)]
        path = str(tmp_path / "cv_progress.json")
        save_cv_progress(path, [], cfg, proteins)
        assert load_cv_progress(path, TrainConfig(seed=99), proteins) == []
        assert load_cv_progress(path, cfg, [{"id": "P99", "length": 10}]) == []

    def test_config_fingerprint_mismatch(self, tmp_path):
        from colab.disordernet_gpu import load_cv_progress, save_cv_progress

        proteins = [{"id": "P1", "length": 10}]
        cfg_a = TrainConfig(seed=7, n_folds=2, lora_rank=16)
        cfg_b = TrainConfig(seed=7, n_folds=2, lora_rank=32)
        proteins = [{"id": f"P{i}", "length": 10} for i in range(6)]
        path = str(tmp_path / "cv_progress.json")
        save_cv_progress(path, [], cfg_a, proteins)
        assert load_cv_progress(path, cfg_b, proteins) == []

    def test_infer_resume_fold(self, tmp_path):
        from colab.disordernet_gpu import infer_resume_fold, save_cv_progress

        cfg = TrainConfig(seed=7, n_folds=2)
        proteins = [{"id": "P1", "length": 10}, {"id": "P2", "length": 20},
                    {"id": "P3", "length": 15}, {"id": "P4", "length": 12}]
        path = str(tmp_path / "cv_progress.json")
        save_cv_progress(
            path,
            [{"fold": 1, "best_auc": 0.8, "best_ap": 0.4, "val_probs": [0.5], "val_labels": [0]}],
            cfg,
            proteins,
        )
        assert infer_resume_fold(path, cfg, proteins) == 1
