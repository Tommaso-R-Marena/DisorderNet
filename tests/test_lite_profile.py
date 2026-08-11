"""Integration tests for the `lite` profile: frozen backbone, head-only training.

The claim the profile makes is falsifiable and worth pinning down: the ESM-2
backbone contributes zero trainable parameters, gradients reach only the head,
and the backbone tensors are bit-identical after optimizer steps. If any of that
silently stops holding, the run stops being the experiment it reports being.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from colab.disordernet_gpu import DisorderNetGPU, LoRALinear, TrainConfig

DIM = 64
N_LAYERS = 6


class DifferentiableESM(nn.Module):
    """ESM-2 stand-in whose outputs actually depend on its parameters.

    ``conftest.MockESM`` returns ``torch.randn``, which is fine for shape checks
    but cannot show whether gradients reach the backbone — the exact property
    these tests exist to verify.
    """

    def __init__(self, n_layers: int = N_LAYERS, dim: int = DIM):
        super().__init__()
        self.embed = nn.Embedding(33, dim)
        self.layers = nn.ModuleList(_layer(dim) for _ in range(n_layers))

    def forward(self, tokens, repr_layers=None, return_contacts=False):
        h = self.embed(tokens)
        wanted = set(repr_layers or [len(self.layers)])
        reps = {}
        for i, lyr in enumerate(self.layers, start=1):
            a = lyr.self_attn
            h = lyr.final_layer_norm(
                h + lyr.fc2(lyr.fc1(a.out_proj(a.q_proj(h) + a.k_proj(h) + a.v_proj(h))))
            )
            if i in wanted:
                reps[i] = h
        if 0 in wanted:
            reps[0] = h
        return {"representations": reps}


def _layer(dim: int) -> nn.Module:
    lyr = nn.Module()
    attn = nn.Module()
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        attn.add_module(name, nn.Linear(dim, dim))
    lyr.add_module("self_attn", attn)
    lyr.add_module("fc1", nn.Linear(dim, dim))
    lyr.add_module("fc2", nn.Linear(dim, dim))
    lyr.add_module("self_attn_layer_norm", nn.LayerNorm(dim))
    lyr.add_module("final_layer_norm", nn.LayerNorm(dim))
    return lyr


def build(
    profile: str, backbone: nn.Module | None = None, **overrides
) -> tuple[DisorderNetGPU, TrainConfig]:
    cfg = TrainConfig.from_profile(
        profile,
        esm_embed_dim=DIM,
        esm_fusion_layers=4,
        use_rich_features=False,
        use_physico_features=True,
        physico_dim=32,
        use_plddt_features=False,
        **overrides,
    )
    return DisorderNetGPU(backbone or DifferentiableESM(), cfg, verbose=False), cfg


@pytest.fixture
def batch():
    torch.manual_seed(0)
    b, length = 2, 40
    return {
        "tokens": torch.randint(4, 30, (b, length + 2)),
        "aa_idx": torch.randint(0, 20, (b, length)),
        "pad_mask": torch.ones(b, length, dtype=torch.bool),
        "shape": (b, length),
    }


class TestConfig:
    def test_frozen_backbone_overrides_adapter_settings(self):
        """A run must not report "frozen" while training 70M parameters."""
        cfg = TrainConfig.from_profile("lite", lora_layers=20, unfreeze_last_layers=2)
        assert cfg.lora_layers == 0
        assert cfg.unfreeze_last_layers == 0

    def test_gradient_checkpointing_disabled_when_frozen(self):
        """It only trades compute for activation memory on a backward pass that
        no longer goes through the backbone."""
        assert TrainConfig.from_profile("lite").use_gradient_checkpointing is False

    def test_unknown_head_type_is_rejected(self):
        with pytest.raises(ValueError, match="head_type"):
            TrainConfig(head_type="ilte")

    def test_lite_disables_the_untuned_regularisers(self):
        """Every extra loss term in `ultra` was tuned under a leaking evaluation;
        `lite` starts from the simplest objective that can work."""
        cfg = TrainConfig.from_profile("lite")
        for flag in (
            "use_focal_loss", "use_dice_loss", "use_tversky_loss", "use_rdrop",
            "use_swa", "use_ema", "use_v6_distill", "use_hallucination_weighting",
        ):
            assert getattr(cfg, flag) is False, flag
        assert cfg.label_smoothing == 0.0

    def test_lite_keeps_the_homology_split(self):
        """The whole point is an honest number."""
        assert TrainConfig.from_profile("lite").split_method == "homology"


class TestFrozenBackbone:
    def test_backbone_has_no_trainable_parameters(self):
        model, _ = build("lite")
        assert sum(p.numel() for p in model.esm.parameters() if p.requires_grad) == 0

    def test_head_is_trainable_and_far_smaller_than_ultra(self):
        lite, _ = build("lite")
        ultra, _ = build("ultra")
        assert lite.n_trainable_params() > 0
        assert lite.n_trainable_params() < ultra.n_trainable_params()

    def test_backbone_is_bit_identical_after_training_steps(self, batch):
        """The strongest available statement that the backbone is frozen."""
        model, _ = build("lite")
        before = {k: v.clone() for k, v in model.esm.state_dict().items()}
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        target = (batch["aa_idx"] < 10).float()
        for _ in range(3):
            opt.zero_grad()
            logits = model(batch["tokens"], aa_idx=batch["aa_idx"], pad_mask=batch["pad_mask"])
            nn.functional.binary_cross_entropy_with_logits(logits, target).backward()
            opt.step()
        drifted = [
            k for k, v in model.esm.state_dict().items() if not torch.equal(v, before[k])
        ]
        assert drifted == []

    def test_the_head_still_learns(self, batch):
        """Freezing must not have severed the gradient path."""
        torch.manual_seed(0)
        model, _ = build("lite")
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        target = (batch["aa_idx"] < 10).float()
        losses = []
        for _ in range(15):
            opt.zero_grad()
            logits = model(batch["tokens"], aa_idx=batch["aa_idx"], pad_mask=batch["pad_mask"])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        assert min(losses[-3:]) < losses[0]

    def test_forward_shape(self, batch):
        model, _ = build("lite")
        out = model(batch["tokens"], aa_idx=batch["aa_idx"], pad_mask=batch["pad_mask"])
        assert out.shape == batch["shape"]


class TestLoRALayerClamp:
    def test_requesting_more_lora_layers_than_exist_does_not_double_wrap(self):
        """`start = n_layers - lora_layers` went negative, and range(-8, 12)
        yields -8..-1 before 0..11 — so on a backbone shallower than the request,
        the deeper layers were wrapped twice and carried two stacked adapters."""
        model, _ = build("ultra")  # ultra asks for 20 layers; the stub has 6
        nested = [
            m for m in model.esm.modules()
            if isinstance(m, LoRALinear) and isinstance(m.original, LoRALinear)
        ]
        assert nested == []

    def test_all_layers_are_adapted_when_the_request_exceeds_depth(self):
        model, _ = build("ultra")
        adapted = {
            i for i, lyr in enumerate(model.esm.layers)
            if isinstance(lyr.self_attn.q_proj, LoRALinear)
        }
        assert adapted == set(range(N_LAYERS))


class TestCheckpointRoundTrip:
    def test_compact_checkpoint_captures_the_whole_trainable_state(self, batch):
        """`lite` sets compact_checkpoints=True. Reload must be exact — this
        project previously lost fine-tuned weights to a name allowlist and only
        found out by comparing stored predictions."""
        from colab.compact_checkpoint import extract_trainable_state_dict

        torch.manual_seed(0)
        # One shared backbone: it is frozen, so it is not part of the checkpoint
        # and both models must start from the same weights for the comparison to
        # isolate the head.
        backbone = DifferentiableESM()
        model, _ = build("lite", backbone=backbone)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        target = (batch["aa_idx"] < 10).float()
        for _ in range(3):
            opt.zero_grad()
            logits = model(batch["tokens"], aa_idx=batch["aa_idx"], pad_mask=batch["pad_mask"])
            nn.functional.binary_cross_entropy_with_logits(logits, target).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            expected = model(batch["tokens"], aa_idx=batch["aa_idx"], pad_mask=batch["pad_mask"])

        state = extract_trainable_state_dict(model)
        fresh, _ = build("lite", backbone=backbone)
        fresh.load_state_dict(state, strict=False)
        fresh.eval()
        with torch.no_grad():
            got = fresh(batch["tokens"], aa_idx=batch["aa_idx"], pad_mask=batch["pad_mask"])

        assert torch.allclose(expected, got, atol=1e-5)
