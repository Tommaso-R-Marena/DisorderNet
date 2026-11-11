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


class TestProteinLevelBias:
    """The missing degree of freedom, and the guarantee it cannot cost anything.

    Decomposing CAID's pooled AUC shows 96.7% to 99.7% of the positive-negative
    pairs it counts are between different proteins. A per-residue model with a
    213-residue receptive field has no mechanism for that, and the consequence is
    measurable: on Binding our within-protein AUC is 0.8683 against the leader's
    0.8049 while our pooled score is lower.

    Adding a constant to every residue of one protein moves only that protein's
    between-protein comparisons and leaves its within-protein ranking exactly
    intact. These tests hold the implementation to that property, which is the
    entire reason the term is safe.
    """

    def test_it_is_enabled_only_for_the_binding_tasks(self):
        m = head()
        assert m.protein_bias_on == ("binding_idr", "binding")
        assert all(t not in m.protein_bias_on for t in PROTECTED)

    def test_it_can_be_disabled(self):
        m = head(protein_bias=False)
        assert m.protein_bias_on == ()
        assert len(m.protein_bias) == 0

    def test_it_starts_as_an_exact_no_op(self):
        m_on = head().eval()
        m_off = head(protein_bias=False).eval()
        m_off.load_state_dict(
            {k: v for k, v in m_on.state_dict().items()
             if not k.startswith("protein_bias.")})
        inp = x()
        with torch.no_grad():
            a, b = m_on(inp), m_off(inp)
        for t in ALL_TASKS:
            assert torch.equal(a[t], b[t]), t

    def test_it_is_constant_within_a_protein(self):
        """The whole point: a per-protein term, not a per-residue one. If it
        varied along the sequence it would change within-protein ranking, which
        is the part already working."""
        m = head().eval()
        with torch.no_grad():
            m.protein_bias["binding"].weight.normal_(0.0, 0.2)
            inp = x(batch=3, length=40)
            with_bias = m(inp)["binding"]
            m.protein_bias_on = ()
            without = m(inp)["binding"]
        delta = with_bias - without
        for row in delta:
            assert torch.allclose(row, row[0].expand_as(row), atol=1e-6), (
                "the bias varies within a protein and would disturb "
                "within-protein ranking")

    def test_different_proteins_receive_different_biases(self):
        """A term identical across proteins would be a global constant, which
        changes no AUC at all."""
        m = head().eval()
        with torch.no_grad():
            m.protein_bias["binding"].weight.normal_(0.0, 0.5)
            inp = x(batch=4, length=40)
            with_bias = m(inp)["binding"]
            m.protein_bias_on = ()
            without = m(inp)["binding"]
        per_protein = (with_bias - without)[:, 0]
        assert per_protein.std() > 1e-6, per_protein

    def test_protected_tasks_are_bit_identical(self):
        m = head().eval()
        with torch.no_grad():
            m.protein_bias["binding"].weight.normal_(0.0, 0.5)
            m.protein_bias["binding_idr"].weight.normal_(0.0, 0.5)
            inp = x()
            a = m(inp)
            m.protein_bias_on = ()
            b = m(inp)
        for t in PROTECTED:
            assert torch.equal(a[t], b[t]), t

    def test_no_gradient_reaches_the_trunk_through_it(self):
        """Pooling is detached, so the weakest task cannot reshape the trunk
        that carries three first places."""
        m = head(protein_bias=True)
        # Isolate the bias path: zero the ordinary read-out so only the bias
        # contributes gradient.
        with torch.no_grad():
            m.out["binding"].weight.zero_()
            m.out["binding"].bias.zero_()
            m.skip["binding"].weight.zero_()
            m.skip["binding"].bias.zero_()
            m.cond["binding"].weight.zero_()
            m.cond["binding"].bias.zero_()
        m(x())["binding"].sum().backward()
        assert m.proj.weight.grad is None or \
            torch.count_nonzero(m.proj.weight.grad) == 0, (
                "gradient from the protein bias reached the trunk")

    def test_the_bias_head_itself_does_learn(self):
        """The guard above must not pass because the path is dead."""
        m = head()
        m(x())["binding"].sum().backward()
        g = m.protein_bias["binding"].weight.grad
        assert g is not None and torch.count_nonzero(g) > 0

    def test_an_older_checkpoint_still_loads(self):
        old = head(protein_bias=False, condition_binding=False)
        state = old.state_dict()
        assert not any(k.startswith("protein_bias.") for k in state)
        fresh = head(protein_bias=False, condition_binding=False)
        fresh.load_state_dict(state)

    def test_pooling_uses_both_mean_and_max(self):
        """Mean alone cannot express "this chain contains a strong site
        somewhere", which is exactly the protein-level signal wanted."""
        m = head()
        assert m.protein_bias["binding"].in_features == \
            2 * m.blocks[0].conv1.in_channels


class TestPrivateTrunkIsolatesBinding:
    """The fix for the protein-bias rejection, held to the property it claims.

    That run moved Binding-IDR from 0.5007 to 0.6062 and was rejected because
    Disorder-NOX, Linker and Binding fell below their floors. The bias path was
    detached and the tests proved no gradient reached a disorder read-out — but
    the binding *losses* still shaped the shared trunk, which the disorder tasks
    read. Protecting a read-out is not protecting a representation.

    Reading a detached trunk closes it exactly: binding contributes zero
    gradient to proj and blocks. These tests assert that arithmetic, and assert
    the private stack still learns so the isolation is not simply a dead path.
    """

    def test_binding_contributes_no_gradient_to_the_shared_trunk(self):
        m = head()
        m(x())["binding_idr"].sum().backward()
        for name, p in m.proj.named_parameters():
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, name
        for i, blk in enumerate(m.blocks):
            for name, p in blk.named_parameters():
                assert p.grad is None or torch.count_nonzero(p.grad) == 0, \
                    f"blocks.{i}.{name}"

    def test_both_binding_tasks_are_isolated(self):
        for task in ("binding", "binding_idr"):
            m = head()
            m(x())[task].sum().backward()
            g = m.proj.weight.grad
            assert g is None or torch.count_nonzero(g) == 0, task

    def test_the_private_stack_does_learn(self):
        """Isolation must not be achieved by the path being dead."""
        m = head()
        m(x())["binding"].sum().backward()
        assert torch.count_nonzero(m.private[0].conv1.weight.grad) > 0

    def test_a_disorder_task_still_trains_the_shared_trunk(self):
        """The trunk must remain trainable — by the tasks that own it."""
        m = head()
        m(x())["disorder_pdb"].sum().backward()
        assert torch.count_nonzero(m.proj.weight.grad) > 0

    def test_adding_binding_to_the_loss_changes_no_trunk_gradient(self):
        """The strongest form of the property, tested on one model.

        Comparing two separately-constructed models would compare different
        random initialisations — a head with five tasks consumes more of the RNG
        stream than one with two — so the first version of this test failed for
        that reason rather than for a leak. On a single model the question is
        exact: does adding the binding losses change what the trunk receives?
        """
        # eval(), because dropout is stochastic and two forward passes in
        # train mode differ for that reason alone — which is what the previous
        # two versions of this test were actually detecting.
        m = head().eval()
        inp = x()

        out = m(inp)
        out["disorder_pdb"].sum().backward()
        alone = m.proj.weight.grad.clone()

        m.zero_grad(set_to_none=True)
        out = m(inp)
        (out["disorder_pdb"].sum() + out["binding_idr"].sum()
         + out["binding"].sum()).backward()
        together = m.proj.weight.grad.clone()

        assert torch.allclose(alone, together, atol=0, rtol=0), (
            "the binding losses moved the shared trunk's gradient")
        assert torch.count_nonzero(alone) > 0, "the trunk must still train"

    def test_it_can_be_disabled_for_the_ablation(self):
        m = head(private_trunk=False)
        assert m.private_on == ()
        assert len(m.private) == 0

    def test_an_older_checkpoint_still_loads(self):
        old = head(private_trunk=False, protein_bias=False,
                   condition_binding=False)
        state = old.state_dict()
        assert not any(k.startswith("private.") for k in state)
        fresh = head(private_trunk=False, protein_bias=False,
                     condition_binding=False)
        fresh.load_state_dict(state)

    def test_output_shapes_are_unchanged(self):
        m = head().eval()
        with torch.no_grad():
            out = m(x(batch=3, length=25))
        for t in ALL_TASKS:
            assert out[t].shape == (3, 25), t


class TestNarrowPrivateFieldForBinding:
    """A binding site in an IDR is a short linear motif; the trunk is 213 wide.

    Disorder is regional and the wide field is why windowing worked. A ten-
    residue motif averaged over 213 residues is averaged over twenty times its
    own length, and on CAID3 Binding-IDR the method leading both the pooled and
    the within-protein axis is LIPNet — a linear-interacting-peptide predictor,
    which is what a motif-scale model looks like.
    """

    TASKS = ("disorder_pdb", "disorder_nox", "linker", "binding", "binding_idr")

    def _head(self, **kw):
        from colab.lite_head import WIDE_DILATIONS, MultiTaskLiteHead

        return MultiTaskLiteHead(in_dim=32, tasks=self.TASKS, structure_dim=0,
                                 dilations=WIDE_DILATIONS, private_trunk=True,
                                 **kw)

    def test_narrow_mode_uses_the_narrow_schedule(self):
        from colab.lite_head import NARROW_DILATIONS

        m = self._head(private_narrow=True)
        got = [b.conv1.dilation[0] for b in m.private]
        assert got == list(NARROW_DILATIONS)[:len(got)]

    def test_wide_mode_is_unchanged(self):
        from colab.lite_head import WIDE_DILATIONS

        m = self._head(private_narrow=False)
        got = [b.conv1.dilation[0] for b in m.private]
        assert got == list(WIDE_DILATIONS)[:len(got)]
        assert m.private_narrow is False

    def test_the_narrow_stack_reads_the_projection_not_the_trunk(self):
        """Dilations alone cannot narrow a field the trunk already widened:
        a narrow stack layered on a 213-residue trunk sees 221 residues. The
        rerouting is the part that does the work, so pin it."""
        src = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "colab", "lite_head.py")).read()
        assert "src = h_proj if self.private_narrow else h" in src
        assert "hp = src.detach() if self.private_detach else src" in src

    def test_a_distant_input_cannot_reach_a_narrow_binding_output(self):
        """The behavioural version of the claim: perturb a residue far outside
        the narrow field and the binding logit must not move, while the
        disorder logit — which reads the wide trunk — must.

        The protein-bias term is switched off here because it pools over the
        whole sequence *by design*, so with it on every residue reaches every
        output. That interaction is the subject of the next test rather than
        something to hide behind a tolerance.
        """
        from colab.lite_head import measured_dependency_span

        m = self._head(private_narrow=True, protein_bias=False,
                       condition_binding=False)
        binding = measured_dependency_span(m, task="binding_idr")
        disorder = measured_dependency_span(m, task="disorder_pdb")
        assert binding <= 10, f"binding still reaches {binding} residues"
        assert disorder >= 100, f"disorder collapsed to {disorder} residues"

    def test_the_protein_bias_reaches_everywhere_by_design(self):
        """With the bias on, a distant residue does move the binding logit,
        through the pooled protein-level scalar. That is the bias working, not
        the narrow field leaking, so the locality claim is only made with the
        bias off.

        The bias heads are initialised to exactly zero, so an untrained model
        shows no such effect at all — the measured span is 6, the narrow field.
        They are given weights here because the question is what the *trained*
        architecture does, and testing the initialisation instead would have
        recorded "the bias is local", which is the opposite of true.
        """
        from colab.lite_head import measured_dependency_span

        m = self._head(private_narrow=True, protein_bias=True,
                       condition_binding=False)
        assert m.protein_bias_on, "no task takes a protein-level bias"
        for head in m.protein_bias.values():
            torch.nn.init.normal_(head.weight, std=0.5)
        assert measured_dependency_span(m, task="binding_idr") >= 100

    def test_a_nearby_input_does_reach_it(self):
        """Guard the guard: a binding output that never moves is not narrow,
        it is dead."""
        from colab.lite_head import measured_dependency_span

        m = self._head(private_narrow=True, protein_bias=False,
                       condition_binding=False)
        assert measured_dependency_span(m, task="binding") >= 1

    def test_the_isolation_survives_the_rerouting(self):
        """The private trunk's whole purpose: binding must still contribute
        exactly zero gradient to the shared trunk."""
        m = self._head(private_narrow=True).eval()
        x = torch.randn(2, 60, 32)
        out = m(x)
        out["disorder_pdb"].sum().backward()
        alone = m.blocks[0].conv1.weight.grad.clone()
        m.zero_grad(set_to_none=True)
        out = m(x)
        (out["disorder_pdb"].sum() + out["binding"].sum()
         + out["binding_idr"].sum()).backward()
        together = m.blocks[0].conv1.weight.grad.clone()
        assert torch.allclose(alone, together, atol=0, rtol=0)
        assert torch.count_nonzero(alone) > 0

    def test_the_narrow_stack_still_learns(self):
        m = self._head(private_narrow=True).eval()
        out = m(torch.randn(2, 60, 32))
        out["binding_idr"].sum().backward()
        assert m.private[0].conv1.weight.grad is not None
        assert torch.count_nonzero(m.private[0].conv1.weight.grad) > 0


class TestTheReceptiveFieldFormulaIsNotTheDependencySpan:
    """`receptive_field` describes the convolutions; GroupNorm decides the
    answer.

    GroupNorm normalises over the channel group *and the whole length axis*, so
    every output position depends on every input position however narrow the
    dilations are. The formula returns 213 for the wide schedule; the measured
    span on a 401-residue input is the full half-length. This project has used
    "a per-residue model with a 213-residue field has no channel for
    protein-level information" as an argument. It has one, through the
    normalisation, and these tests keep that from being forgotten again.
    """

    TASKS = ("disorder_pdb", "binding_idr")

    def _head(self, **kw):
        from colab.lite_head import WIDE_DILATIONS, MultiTaskLiteHead

        return MultiTaskLiteHead(in_dim=32, tasks=self.TASKS, structure_dim=0,
                                 dilations=WIDE_DILATIONS, protein_bias=False,
                                 condition_binding=False, **kw)

    def test_groupnorm_makes_the_dependency_global(self):
        from colab.lite_head import (WIDE_DILATIONS, measured_dependency_span,
                                     receptive_field)

        m = self._head(private_trunk=False)
        formula = receptive_field(WIDE_DILATIONS, 4) // 2      # +/- half-width
        measured = measured_dependency_span(m, length=401,
                                            task="disorder_pdb")
        assert measured == 200, measured
        assert measured > formula, (
            f"measured {measured} should exceed the formula's {formula}")

    def test_channel_norm_makes_the_dependency_match_the_formula(self):
        from colab.lite_head import (DilatedResidualBlock,
                                     measured_dependency_span)

        class Stack(torch.nn.Module):
            def __init__(self, local_norm):
                super().__init__()
                self.proj = torch.nn.Conv1d(32, 32, 1)
                self.b1 = DilatedResidualBlock(32, 1, 0.0, local_norm=local_norm)
                self.b2 = DilatedResidualBlock(32, 2, 0.0, local_norm=local_norm)
                self.out = torch.nn.Conv1d(32, 1, 1)

            def forward(self, x):
                h = self.b2(self.b1(self.proj(x.transpose(1, 2))))
                return self.out(h).squeeze(1)

        local = measured_dependency_span(Stack(True), length=401)
        globe = measured_dependency_span(Stack(False), length=401)
        # Two blocks, dilations 1 and 2, two convs each: 2*(1+2) = 6 each side.
        assert local == 6, local
        assert globe == 200, globe

    def test_the_docstring_says_so(self):
        """The claim lived in prose that argued the opposite. Pin the
        correction where it was wrong."""
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        head = open(os.path.join(root, "colab", "lite_head.py")).read()
        assert "normalises over the length axis" in head
        for name in ("auc_decomposition.py", "sequence_biophysics.py"):
            text = open(os.path.join(root, "colab", name)).read()
            assert "213-residue receptive field over a 1,000-residue protein" \
                not in text, f"{name} still asserts the model cannot see the protein"


class TestShapeAndGradientAreSeparable:
    """mt_private detached the binding read-out and paid for it.

    Against its regime-matched control it lost 0.0555 on Binding and 0.0262 on
    Binding-IDR while sparing the disorder tasks about 0.006 — binding has 891
    training proteins against disorder's 21,386, so cutting the gradient in
    both directions loses an order of magnitude more than it protects. The
    narrow read-out is worth keeping; the detach is not, and they are now
    separate settings.
    """

    TASKS = ("disorder_pdb", "disorder_nox", "linker", "binding", "binding_idr")

    def _head(self, **kw):
        from colab.lite_head import WIDE_DILATIONS, MultiTaskLiteHead

        return MultiTaskLiteHead(in_dim=32, tasks=self.TASKS, structure_dim=0,
                                 dilations=WIDE_DILATIONS, private_trunk=True,
                                 protein_bias=False, condition_binding=False,
                                 **kw)

    def test_attached_binding_does_reach_the_shared_trunk(self):
        m = self._head(private_narrow=True, private_detach=False).eval()
        x = torch.randn(2, 60, 32)
        m(x)["binding_idr"].sum().backward()
        g = m.proj.weight.grad
        assert g is not None and torch.count_nonzero(g) > 0

    def test_detached_binding_still_does_not(self):
        """The isolation must remain available and exact where it is wanted."""
        m = self._head(private_narrow=True, private_detach=True).eval()
        x = torch.randn(2, 60, 32)
        out = m(x)
        out["disorder_pdb"].sum().backward()
        alone = m.proj.weight.grad.clone()
        m.zero_grad(set_to_none=True)
        out = m(x)
        (out["disorder_pdb"].sum() + out["binding_idr"].sum()).backward()
        assert torch.allclose(alone, m.proj.weight.grad, atol=0, rtol=0)
        assert torch.count_nonzero(alone) > 0

    def test_attaching_does_not_widen_the_binding_read_out(self):
        """The two settings are orthogonal: gradient flow must not change what
        the binding output can see."""
        from colab.lite_head import measured_dependency_span

        for detach in (True, False):
            m = self._head(private_narrow=True, private_detach=detach)
            span = measured_dependency_span(m, task="binding_idr")
            assert span <= 10, (detach, span)

    def test_the_default_is_still_detached(self):
        """Changing the default would silently alter every checkpoint that
        records private_trunk without recording private_detach."""
        assert self._head(private_narrow=True).private_detach is True
        import inspect

        from colab.lite_head import MultiTaskLiteHead

        sig = inspect.signature(MultiTaskLiteHead.__init__)
        assert sig.parameters["private_detach"].default is True
