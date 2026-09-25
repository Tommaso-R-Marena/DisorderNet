"""A checkpoint must reproduce the predictions of the model that wrote it.

The compact format selected tensors by name allowlist (lora_A, lora_B, head., …).
The ultra profile sets unfreeze_last_layers=2, fine-tuning
``esm.layers.{N}.self_attn_layer_norm`` and ``final_layer_norm`` — names that
match no marker. Those weights were never saved and silently reverted to
pretrained values on reload. Measured on the 650M run: the restored model scored
AUC 0.5835 with Spearman 0.387 against the predictions it was saved from
(0.7651). Every downstream consumer of a checkpoint was affected — fold soup,
CAID3 evaluation, FASTA deployment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from colab.compact_checkpoint import (
    extract_trainable_state_dict,
    load_compact_checkpoint,
    save_compact_checkpoint,
)


class _ToyBackbone(nn.Module):
    """Frozen trunk with a fine-tuned tail LayerNorm — the ultra arrangement."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(8, 8)
        self.tail_norm = nn.LayerNorm(8)      # matches NO trainable-key marker
        self.head = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.Linear(8, 1))
        for p in self.trunk.parameters():
            p.requires_grad = False           # frozen, must not be saved
        # tail_norm and head are trained
        for p in self.tail_norm.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.head(self.tail_norm(self.trunk(x)))


def _train_a_little(model, steps=40):
    """Move the trainable weights meaningfully away from init, without diverging."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.02)
    x = torch.randn(16, 8)
    y = torch.randn(16, 1)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    model.eval()
    return model


class TestTrainableSelection:
    def test_saves_params_outside_the_name_allowlist(self):
        m = _train_a_little(_ToyBackbone())
        state = extract_trainable_state_dict(m)
        assert "tail_norm.weight" in state, (
            "a fine-tuned LayerNorm outside the marker namespace must be saved — "
            "this is exactly what unfreeze_last_layers trains"
        )
        assert "tail_norm.bias" in state

    def test_excludes_frozen_parameters(self):
        m = _train_a_little(_ToyBackbone())
        state = extract_trainable_state_dict(m)
        assert not any(k.startswith("trunk.") for k in state), (
            "frozen trunk weights are recoverable from the pretrained backbone "
            "and would defeat the point of a compact checkpoint"
        )

    def test_includes_buffers_of_trained_modules(self):
        """BatchNorm running stats determine eval-mode output as surely as weights."""
        m = _train_a_little(_ToyBackbone())
        state = extract_trainable_state_dict(m)
        assert any("running_mean" in k for k in state)
        assert any("running_var" in k for k in state)


class TestRoundTripReproducesPredictions:
    def test_reloaded_model_matches_the_one_that_saved_it(self, tmp_path):
        """The invariant that actually matters, and that nothing asserted before."""
        torch.manual_seed(0)
        trained = _train_a_little(_ToyBackbone())
        x = torch.randn(32, 8)
        with torch.no_grad():
            expected = trained(x)

        path = str(tmp_path / "fold1_best.pt")
        save_compact_checkpoint(path, trained)

        torch.manual_seed(999)          # fresh init: nothing carried over
        restored = _ToyBackbone()
        restored.trunk.load_state_dict(trained.trunk.state_dict())  # frozen trunk
        load_compact_checkpoint(path, restored)
        restored.eval()
        with torch.no_grad():
            got = restored(x)

        assert torch.allclose(expected, got, atol=1e-5), (
            "reloaded model does not reproduce its own predictions; "
            f"max abs diff {float((expected - got).abs().max()):.4g}"
        )

    def test_name_allowlist_alone_would_have_failed(self, tmp_path):
        """Confirms the regression test is not vacuous: saving only the
        allowlisted namespaces loses the tail LayerNorm and changes outputs."""
        from colab.compact_checkpoint import is_trainable_key

        torch.manual_seed(1)
        trained = _train_a_little(_ToyBackbone())
        x = torch.randn(32, 8)
        with torch.no_grad():
            expected = trained(x)

        legacy_state = {
            k: v for k, v in trained.state_dict().items() if is_trainable_key(k)
        }
        assert "tail_norm.weight" not in legacy_state, "fixture must reproduce the bug"

        torch.manual_seed(999)
        restored = _ToyBackbone()
        restored.trunk.load_state_dict(trained.trunk.state_dict())
        restored.load_state_dict(legacy_state, strict=False)
        restored.eval()
        with torch.no_grad():
            got = restored(x)
        assert not torch.allclose(expected, got, atol=1e-5), (
            "the legacy allowlist should NOT reproduce predictions — if it does, "
            "this test no longer guards anything"
        )


class TestLoadRejectsIncompleteCheckpoints:
    def test_missing_trainable_tensor_raises(self, tmp_path):
        torch.manual_seed(2)
        trained = _train_a_little(_ToyBackbone())
        path = str(tmp_path / "partial.pt")
        save_compact_checkpoint(path, trained)

        # Simulate a legacy checkpoint: strip the tail LayerNorm.
        payload = torch.load(path, weights_only=False)
        payload["trainable"] = {
            k: v for k, v in payload["trainable"].items() if not k.startswith("tail_norm")
        }
        torch.save(payload, path)

        with pytest.raises(RuntimeError, match="missing"):
            load_compact_checkpoint(path, _ToyBackbone())
