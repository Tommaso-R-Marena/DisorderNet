"""Checkpoints and the resume record must never be left half-written.

A lite_frozen replicate was killed during its fold-4 checkpoint write and left a
0-byte fold4_best.pt in the checkpoint directory. Anything that locates folds by
existence check would then treat that fold as done and try to load a truncated
file. cv_progress.json has the same exposure and is worse: it IS the resume
record, so truncated JSON means a resume silently restarts from fold 1 and
discards hours of completed folds.
"""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest
import torch

from colab.compact_checkpoint import atomic_torch_save, load_compact_checkpoint


class TestAtomicSave:
    def test_writes_a_loadable_file(self, tmp_path):
        p = tmp_path / "ckpt.pt"
        atomic_torch_save({"a": torch.ones(3)}, str(p))
        assert torch.equal(torch.load(p, weights_only=False)["a"], torch.ones(3))

    def test_a_failed_write_leaves_the_previous_file_intact(self, tmp_path):
        """The property that matters: readers see the old file or the new one."""
        p = tmp_path / "ckpt.pt"
        atomic_torch_save({"gen": 1}, str(p))

        with mock.patch("torch.save", side_effect=RuntimeError("node died")):
            with pytest.raises(RuntimeError, match="node died"):
                atomic_torch_save({"gen": 2}, str(p))

        assert torch.load(p, weights_only=False)["gen"] == 1

    def test_a_failed_write_leaves_no_temp_litter(self, tmp_path):
        p = tmp_path / "ckpt.pt"
        with mock.patch("torch.save", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                atomic_torch_save({"x": 1}, str(p))
        assert list(tmp_path.iterdir()) == []

    def test_an_interrupt_also_cleans_up(self, tmp_path):
        """SIGINT during a write is the realistic case, and KeyboardInterrupt is
        not an Exception — a bare `except Exception` would miss it."""
        p = tmp_path / "ckpt.pt"
        with mock.patch("torch.save", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                atomic_torch_save({"x": 1}, str(p))
        assert list(tmp_path.iterdir()) == []

    def test_creates_missing_directories(self, tmp_path):
        p = tmp_path / "deep" / "nested" / "ckpt.pt"
        atomic_torch_save({"x": 1}, str(p))
        assert p.is_file()


class TestTruncatedCheckpointIsRejected:
    def test_zero_byte_checkpoint_raises_a_named_error(self, tmp_path):
        """Exactly what the killed job left behind."""
        p = tmp_path / "fold4_best.pt"
        p.write_bytes(b"")
        with pytest.raises(RuntimeError, match="empty"):
            load_compact_checkpoint(str(p), torch.nn.Linear(2, 2))

    def test_the_error_says_the_fold_is_not_complete(self, tmp_path):
        p = tmp_path / "fold4_best.pt"
        p.write_bytes(b"")
        with pytest.raises(RuntimeError) as exc:
            load_compact_checkpoint(str(p), torch.nn.Linear(2, 2))
        assert "killed mid-save" in str(exc.value)


class TestAtomicProgressFile:
    def _save(self, path, folds):
        from types import SimpleNamespace

        from colab.disordernet_gpu import save_cv_progress

        cfg = SimpleNamespace(n_folds=5, seed=42, split_method="protein",
                              homology_min_identity=0.40)
        proteins = [{"id": f"P{i}", "sequence": "ACDEFGHIKL" * 3} for i in range(4)]
        with mock.patch("colab.disordernet_gpu.get_fold_val_protein_ids",
                        return_value=[[] for _ in range(5)]), \
             mock.patch("colab.disordernet_gpu.proteins_fingerprint", return_value="fp"), \
             mock.patch("colab.disordernet_gpu.config_fingerprint", return_value="cf"):
            save_cv_progress(str(path), folds, cfg, proteins)

    def _fold(self, i):
        return {"fold": i, "best_auc": 0.8, "best_ap": 0.5,
                "val_probs": [0.1, 0.9], "val_labels": [0, 1]}

    def test_progress_survives_a_failed_rewrite(self, tmp_path):
        """Three completed folds must not be lost because the fourth write died."""
        p = tmp_path / "cv_progress.json"
        self._save(p, [self._fold(i) for i in (1, 2, 3)])
        assert len(json.loads(p.read_text())["fold_results"]) == 3

        with mock.patch("json.dump", side_effect=OSError("disk gone")):
            with pytest.raises(OSError):
                self._save(p, [self._fold(i) for i in (1, 2, 3, 4)])

        restored = json.loads(p.read_text())
        assert len(restored["fold_results"]) == 3, "resume record was corrupted"

    def test_no_temp_files_remain_in_the_checkpoint_dir(self, tmp_path):
        p = tmp_path / "cv_progress.json"
        self._save(p, [self._fold(1)])
        assert [f.name for f in tmp_path.iterdir()] == ["cv_progress.json"]


class TestBestStateSnapshotIsNotTheWholeModel:
    """`best_state = copy.deepcopy(model.state_dict())` copied the frozen
    backbone too — 653M parameters (~2.6 GB) to preserve the 1.96M that the
    lite profile actually trains, reallocated on every validation improvement.
    Two replicates died at a fold boundary with no traceback, which is what the
    kernel OOM killer looks like: it takes the process before stderr flushes.
    """

    def _model(self):
        m = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.Linear(512, 4))
        for p in m[0].parameters():
            p.requires_grad = False
        return m

    def _bytes(self, state):
        return sum(v.numel() * v.element_size() for v in state.values())

    def test_snapshot_excludes_the_frozen_backbone(self):
        import copy as _copy

        from colab.disordernet_gpu import _snapshot_best_state

        m = self._model()
        full = _copy.deepcopy(m.state_dict())
        snap = _snapshot_best_state(m)
        assert self._bytes(snap) < self._bytes(full) / 10

    def test_omitting_frozen_tensors_is_exact_not_an_approximation(self):
        """A parameter with requires_grad=False cannot be moved by the
        optimizer, so its end-of-fold value is its start-of-fold value."""
        from colab.disordernet_gpu import _snapshot_best_state

        m = self._model()
        frozen_before = m[0].weight.clone()
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1.0)
        for _ in range(5):
            opt.zero_grad()
            m(torch.randn(4, 512)).sum().backward()
            opt.step()
        assert torch.equal(m[0].weight, frozen_before)
        assert not any(k.startswith("0.") for k in _snapshot_best_state(m))

    def test_snapshot_captures_every_trained_tensor(self):
        """The snapshot must not lose weights the fold validated on."""
        from colab.disordernet_gpu import _snapshot_best_state

        m = self._model()
        snap = _snapshot_best_state(m)
        trainable = {n for n, p in m.named_parameters() if p.requires_grad}
        assert trainable <= set(snap)

    def test_snapshot_round_trips_the_trained_weights(self):
        from colab.disordernet_gpu import _snapshot_best_state

        m = self._model()
        x = torch.randn(3, 512)
        m.eval()
        with torch.no_grad():
            best = m(x).clone()
        snap = _snapshot_best_state(m)

        # Train away from the snapshot, then restore it.
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.5)
        for _ in range(5):
            opt.zero_grad(); m(x).sum().backward(); opt.step()
        with torch.no_grad():
            assert not torch.allclose(m(x), best)

        missing, unexpected = m.load_state_dict(snap, strict=False)
        assert not unexpected
        with torch.no_grad():
            assert torch.allclose(m(x), best, atol=1e-6)

    def test_batchnorm_buffers_of_trained_modules_are_kept(self):
        """Running statistics change without gradients and determine eval-mode
        output as surely as the weights do."""
        from colab.disordernet_gpu import _snapshot_best_state

        m = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.BatchNorm1d(8))
        m.train()
        m(torch.randn(16, 8))
        snap = _snapshot_best_state(m)
        assert any("running_mean" in k for k in snap)
