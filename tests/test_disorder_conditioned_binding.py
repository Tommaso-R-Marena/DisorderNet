"""Binding-IDR conditioned on disorder — and the guards that protect the rest.

Binding-IDR is the Binding labels restricted to disordered residues. Trained as
a masked variant of binding, the head scored 0.4945 on the official reference —
below chance — while scoring 0.7924 on Binding itself. Conditioning on disorder
erased the signal, which means what it learned was disorder: across a whole
protein, binding sites sit in IDRs and IDRs are the disordered part, so "is this
disordered" answers Binding well and Binding-IDR not at all.

The fix hands the model the disorder answer instead of making it rediscover one.
The risk is the reverse direction: binding has 891 training proteins, disorder
has 21,386, and disorder currently holds first place on three benchmarks. A
change that lets the weakest task perturb the strongest ones to help itself
would trade three wins for one.

So the conditioning is a detached, additive path, and these tests hold it to
that — not by inspecting intent but by asserting the arithmetic.
"""

from __future__ import annotations

import os

import pytest
import torch

from colab.lite_head import MultiTaskLiteHead

ALL_TASKS = ("disorder_nox", "linker", "binding", "binding_idr", "disorder_pdb")
PROTECTED = ("disorder_pdb", "disorder_nox", "linker")


def head(tasks=ALL_TASKS, in_dim=32, hidden=16, n_blocks=2, **kw):
    torch.manual_seed(0)
    return MultiTaskLiteHead(in_dim=in_dim, tasks=tasks, hidden=hidden,
                             n_blocks=n_blocks, **kw)


def x(batch=2, length=40, in_dim=32):
    torch.manual_seed(1)
    return torch.randn(batch, length, in_dim)


class TestWiring:
    def test_conditioning_is_on_when_both_sides_exist(self):
        m = head()
        assert m.condition_on == ("binding_idr", "binding")
        assert m.condition_source == "disorder_pdb"

    def test_it_prefers_the_task_matched_disorder_source(self):
        """disorder_pdb carries CAID's own label definition and 21,386
        proteins; disorder_nox has 3,333."""
        assert head().condition_source == "disorder_pdb"
        m = head(tasks=("disorder_nox", "binding", "binding_idr"))
        assert m.condition_source == "disorder_nox"

    def test_no_disorder_task_means_no_conditioning(self):
        m = head(tasks=("binding", "binding_idr", "linker"))
        assert m.condition_on == ()
        assert len(m.cond) == 0

    def test_no_binding_task_means_no_conditioning(self):
        m = head(tasks=("disorder_pdb", "linker"))
        assert m.condition_on == ()

    def test_it_starts_as_an_exact_no_op(self):
        """Zero-initialised, so a conditioned run begins precisely where the
        unconditioned one is and any difference is learned rather than given."""
        m = head().eval()
        with torch.no_grad():
            out = m(x())
        # The additive term is exactly zero at initialisation.
        with torch.no_grad():
            h = x().transpose(1, 2)
            hh = m.proj(h)
            for blk in m.blocks:
                hh = blk(hh)
            p = torch.sigmoid(out[m.condition_source]).detach()
            for t in m.condition_on:
                add = m.cond[t](torch.cat([hh, p.unsqueeze(1)], dim=1)).squeeze(1)
                assert torch.equal(add, torch.zeros_like(add))


class TestProtectedTasksAreUntouched:
    """The three first-place tasks must be provably unaffected."""

    def test_protected_logits_are_bit_identical_with_and_without(self):
        m = head().eval()
        inp = x()
        with torch.no_grad():
            conditioned = m(inp)
            m.condition_on = ()          # disable the additive path only
            plain = m(inp)
        for t in PROTECTED:
            assert torch.equal(conditioned[t], plain[t]), t

    def test_binding_loss_never_reaches_the_disorder_readout(self):
        """The detach is the guarantee. Without it, binding's gradient would
        flow back through the disorder read-out."""
        m = head()
        out = m(x())
        out["binding_idr"].sum().backward()
        for name, p in m.out["disorder_pdb"].named_parameters():
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, (
                f"binding_idr gradient reached disorder_pdb read-out ({name})")
        for name, p in m.skip["disorder_pdb"].named_parameters():
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, name

    def test_the_conditioned_readout_does_receive_gradient(self):
        """The guard above must not hold simply because nothing learns."""
        m = head()
        m(x())["binding_idr"].sum().backward()
        grads = [p.grad for p in m.cond["binding_idr"].parameters()]
        assert any(g is not None and torch.count_nonzero(g) > 0 for g in grads)

    def test_a_conditioned_task_still_trains_its_own_trunk_path(self):
        """Conditioning is additive, so the ordinary read-out keeps learning."""
        m = head()
        m(x())["binding_idr"].sum().backward()
        g = m.out["binding_idr"].weight.grad
        assert g is not None and torch.count_nonzero(g) > 0


class TestConditioningActuallyConditions:
    def test_changing_predicted_disorder_changes_the_binding_logit(self):
        """With a trained (non-zero) conditioning weight, the binding logit must
        respond to the disorder probability — otherwise the mechanism is inert
        and the whole change is decoration."""
        m = head().eval()
        with torch.no_grad():
            m.cond["binding_idr"].weight.normal_(0.0, 0.5)
            m.cond["binding_idr"].bias.zero_()
            inp = x()
            before = m(inp)["binding_idr"].clone()
            # Push predicted disorder up by biasing the source read-out.
            m.out[m.condition_source].bias.add_(5.0)
            after = m(inp)["binding_idr"]
        assert not torch.allclose(before, after)

    def test_the_source_task_is_not_itself_conditioned(self):
        m = head()
        assert m.condition_source not in m.condition_on


class TestShapeAndCost:
    def test_output_shape_is_unchanged(self):
        m = head().eval()
        with torch.no_grad():
            out = m(x(batch=3, length=25))
        assert set(out) == set(ALL_TASKS)
        for t in ALL_TASKS:
            assert out[t].shape == (3, 25)

    def test_the_extra_cost_is_small(self):
        """Two 1x1 convolutions over hidden+1 channels, nothing more."""
        plain = head(tasks=("disorder_pdb", "linker"))
        cond = head()
        extra = sum(p.numel() for p in cond.cond.parameters())
        assert extra == 2 * (cond.blocks[0].conv1.in_channels + 1 + 1), extra

    def test_it_works_with_structure_channels(self):
        m = head(structure_dim=24).eval()
        b, ln = 2, 30
        with torch.no_grad():
            out = m(x(b, ln), rsa=torch.rand(b, ln), plddt=torch.rand(b, ln),
                    structure_available=torch.ones(b, ln),
                    contacts=torch.rand(b, ln))
        assert out["binding_idr"].shape == (b, ln)

    def test_it_survives_a_single_residue(self):
        m = head().eval()
        with torch.no_grad():
            out = m(x(batch=1, length=1))
        assert out["binding_idr"].shape == (1, 1)


class TestCheckpointCompatibility:
    """An older checkpoint must still load, or the three wins become unloadable."""

    def test_a_checkpoint_without_conditioning_loads(self):
        old = head(tasks=("disorder_pdb", "linker"))
        state = old.state_dict()
        assert not any(k.startswith("cond.") for k in state)
        fresh = head(tasks=("disorder_pdb", "linker"))
        fresh.load_state_dict(state)

    def test_a_conditioned_checkpoint_round_trips(self):
        m = head()
        with torch.no_grad():
            m.cond["binding"].weight.normal_()
        state = m.state_dict()
        assert any(k.startswith("cond.") for k in state)
        fresh = head()
        fresh.load_state_dict(state)
        fresh.eval()
        m.eval()
        with torch.no_grad():
            a, b = m(x()), fresh(x())
        for t in ALL_TASKS:
            assert torch.equal(a[t], b[t]), t

    def test_loading_an_old_checkpoint_into_a_conditioned_head_is_explicit(self):
        """strict=True must fail rather than silently leaving cond random."""
        old = head(tasks=("disorder_pdb", "binding", "binding_idr"))
        state = {k: v for k, v in old.state_dict().items()
                 if not k.startswith("cond.")}
        fresh = head(tasks=("disorder_pdb", "binding", "binding_idr"))
        with pytest.raises(RuntimeError):
            fresh.load_state_dict(state, strict=True)
        missing, unexpected = fresh.load_state_dict(state, strict=False)
        assert all(k.startswith("cond.") for k in missing), missing
        assert not unexpected


class TestArchitectureIsRestoredFromTheCheckpoint:
    """A checkpoint records more than weights, and loading must honour it.

    Dilation changes the receptive field but not a single weight shape, so a
    head built with the default dilations strict-loads weights trained with the
    wide ones without complaint and then computes a different function. That
    happened: mt_full and mt_windowed were trained at a 213-residue field and
    evaluated at 61, silently, for every number reported from them.
    """

    def test_dilation_does_not_change_weight_shapes(self):
        """The property that makes the mistake silent — pinned, so nobody
        later assumes strict-loading would have caught it."""
        from colab.lite_head import DEFAULT_DILATIONS, WIDE_DILATIONS

        narrow = head(tasks=("disorder_pdb", "linker"), n_blocks=4,
                      dilations=DEFAULT_DILATIONS)
        wide = head(tasks=("disorder_pdb", "linker"), n_blocks=4,
                    dilations=WIDE_DILATIONS)
        a, b = narrow.state_dict(), wide.state_dict()
        assert set(a) == set(b)
        assert all(a[k].shape == b[k].shape for k in a)
        wide.load_state_dict(a)          # strict, and it succeeds

    def test_but_the_computed_function_differs(self):
        from colab.lite_head import DEFAULT_DILATIONS, WIDE_DILATIONS

        narrow = head(tasks=("disorder_pdb", "linker"), n_blocks=4,
                      dilations=DEFAULT_DILATIONS).eval()
        wide = head(tasks=("disorder_pdb", "linker"), n_blocks=4,
                    dilations=WIDE_DILATIONS).eval()
        wide.load_state_dict(narrow.state_dict())
        with torch.no_grad():
            a = narrow(x(length=80))["disorder_pdb"]
            b = wide(x(length=80))["disorder_pdb"]
        assert not torch.allclose(a, b), (
            "if these agreed the receptive field would not matter and the "
            "wide/narrow ablation would be meaningless")
        assert narrow.receptive_field == 61
        assert wide.receptive_field == 213

    def test_the_evaluator_restores_every_architecture_field(self):
        """Each of these changes the function computed and none of them changes
        a weight shape, so the checkpoint is the only source of truth."""
        src = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rockfish", "eval_caid3_official.py")).read()
        for field in ("structure_dim", "condition_binding",
                      "wide_receptive_field"):
            assert f'payload.get("{field}"' in src or \
                   f'payload["{field}"]' in src, field
        assert "WIDE_DILATIONS" in src, (
            "the evaluator must pass dilations, or a wide-trained checkpoint "
            "is silently evaluated at a 61-residue receptive field")

    def test_the_trainer_records_every_architecture_field(self):
        src = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rockfish", "train_multitask.py")).read()
        for field in ("structure_dim", "wide_receptive_field",
                      "condition_binding"):
            assert f'"{field}":' in src, field
