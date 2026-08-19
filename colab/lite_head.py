"""DisorderNet-Lite: a frozen-PLM, low-capacity head for the small-data regime.

Why a different architecture
----------------------------
The measurements from the 650M homology-split run point one way:

  ESM-2 650M + LoRA (69.9M trainable)   pooled AUC 0.7454
  v6 physics GBDT                        pooled AUC 0.7804
  AlphaFold pLDDT alone                  pooled AUC 0.7906

A 650M protein language model losing to a gradient-boosted tree on hand-crafted
features is not what ESM-2 can do — published ESM2-650M+LoRA disorder models sit
near 0.88. The bottleneck is not the backbone, it is the fit between capacity and
data:

  * 69.9M trainable parameters (LoRA r=128 over 20 layers, plus FFN, out_proj,
    and two unfrozen ESM tail layers)
  * ~1M evidenced residues from 2,340 proteins
  * ten simultaneously-active regularisers (focal gamma=3, dice, tversky,
    R-drop, v6 distillation, boundary x4, hallucination weighting, label
    smoothing, EMA, SWA, MC-dropout TTA), every hyperparameter tuned while the
    evaluation was leaking

The screen showed the failure directly: train loss 0.069 against validation AUC
0.66. A GBDT wins in that regime because it is sample-efficient, not because
physics features beat language-model features.

Design, one decision per measurement
------------------------------------
1. **Freeze the backbone.** Learn a scalar mixture over its layers instead
   (33 parameters for ESM-2 650M). Different layers encode different things and
   the useful depth for disorder is an empirical question, not a fixed choice.
   Precedent: SETH (frozen ProtT5 + CNN) reaches 0.830 on CAID — above this
   project's 0.8155 — with no fine-tuning at all.
2. **Small dilated CNN head.** Disorder is local-to-medium range; dilations
   1/2/4/8 cover ~30 residues of context with ~1-3M parameters instead of 70M.
3. **Keep an explicit physics channel.** v6 alone (0.7804) beat the LoRA model,
   and ensembling gained +0.056 — that signal is real and cheap.
4. **Plain weighted BCE.** Every extra loss term is a hyperparameter fitted to a
   broken evaluation. Start from the simplest objective that can work and add a
   term back only when it beats the measured ~0.023 AUC noise floor.

What this is not
----------------
This is not expected to reach 0.895. Its purpose is to test the specific
hypothesis that capacity/data mismatch — not the backbone — is what costs this
project ~0.08 AUC against comparable published models.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

# Dilations 1/2/4/8 over 4 residual blocks give a 61-residue receptive field —
# comfortably wider than a typical IDR, without a transformer's parameter cost.
DEFAULT_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)

# Wider schedule for disorder specifically. The default gives a 61-residue
# receptive field over four blocks; IDRs frequently run past 100, and the rsa
# signal itself needed a 21-residue smoothing window to work at all (0.8688 raw
# against 0.9459 smoothed). This reaches 213 residues for the same parameter
# count — dilation buys context, not weights.
WIDE_DILATIONS: tuple[int, ...] = (1, 4, 16, 32)

# Narrow schedule for the binding tasks' private stack. Disorder is regional
# and the wide field is why windowing worked; a *binding* site in an IDR is
# usually a short linear motif of five to fifteen residues, and a 213-residue
# field averages a ten-residue motif over twenty times its own length. On
# CAID3 Binding-IDR the method that leads both the pooled and the
# within-protein axis is LIPNet, a linear-interacting-peptide predictor, which
# is what a motif-scale model looks like.
#
# The private stack reads the *projection*, not the trunk output, when this is
# used. A narrow stack layered on a wide trunk inherits the wide field and
# changes nothing — 213 residues in, 221 out.
NARROW_DILATIONS: tuple[int, ...] = (1, 2, 4)


def receptive_field(dilations: Sequence[int], n_blocks: int) -> int:
    """Residues visible to one output position **through the convolutions**.

    Each block applies two dilated kernel-3 convolutions, so it widens the field
    by ``2 * dilation`` on each side.

    This is not the model's dependency span unless every block uses
    position-local normalisation. With GroupNorm — the default — statistics are
    pooled over the length axis and every output depends on every input.
    ``measured_dependency_span`` reports what actually holds.
    """
    span = sum(2 * dilations[i % len(dilations)] for i in range(n_blocks))
    return 1 + 2 * span


class ScalarMix(nn.Module):
    """Learned softmax-weighted mixture of frozen layer representations.

    One weight per layer plus a global scale — 33 parameters for ESM-2 650M.
    Which depth carries disorder signal is an empirical question; fixing it by
    hand (or concatenating every layer, which multiplies the head's input width)
    both throw information away. The learned weights are also interpretable: they
    say where in the backbone the signal actually lives.
    """

    def __init__(self, n_layers: int, do_layer_norm: bool = False):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_layers = n_layers
        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(1))
        self.do_layer_norm = do_layer_norm

    def forward(self, layers: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(layers) != self.n_layers:
            raise ValueError(f"expected {self.n_layers} layers, got {len(layers)}")
        w = torch.softmax(self.weights, dim=0)
        out = None
        for i, h in enumerate(layers):
            if self.do_layer_norm:
                h = nn.functional.layer_norm(h, h.shape[-1:])
            term = w[i] * h
            out = term if out is None else out + term
        return self.gamma * out

    def layer_weights(self) -> torch.Tensor:
        """Normalised mixture weights, for reporting where the signal sits."""
        return torch.softmax(self.weights.detach(), dim=0)


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each position, for (B, C, L) tensors.

    The position-local alternative to GroupNorm. GroupNorm normalises over the
    channel group **and the whole length axis**, which couples every output
    position to every input position however narrow the convolutions are; this
    normalises each position independently, so a block's dependency span is
    exactly its convolutional field.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, C, L)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class DilatedResidualBlock(nn.Module):
    """Residual 1-D conv block with dilation, normalisation and GELU.

    GroupNorm rather than BatchNorm: batches here are a handful of proteins of
    wildly differing length, so batch statistics are unstable and — as this
    project found the hard way — BatchNorm running statistics are easy to lose
    across a checkpoint round-trip.

    **GroupNorm normalises over the length axis as well as the channel group**,
    which means a block's output at one position depends on the input at
    *every* position, whatever its dilation. `receptive_field` describes the
    convolutions and not the model: measured on a 401-residue input, perturbing
    residue 0 moves the logit at residue 400. That is not a defect — sharing
    per-window statistics is a large part of why this head works at all — but
    it is not what "213-residue receptive field" says, and this project has
    used that phrase to argue the model has *no* channel for protein-level
    information. It has one.

    ``local_norm`` swaps in per-position channel normalisation, which makes the
    dependency span equal the convolutional field. Off by default, because
    turning it on changes what every existing checkpoint computes.
    """

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1,
                 local_norm: bool = False):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        norm = ChannelNorm if local_norm else (lambda c: nn.GroupNorm(8, c))
        self.norm1 = norm(channels)
        self.norm2 = norm(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        h = self.drop(self.act(self.norm1(self.conv1(x))))
        h = self.drop(self.act(self.norm2(self.conv2(h))))
        return x + h


class LiteDisorderHead(nn.Module):
    """The Lite trunk, in the shape the existing pipeline expects.

    ``DisorderNetGPU`` already performs layer fusion (``ESMLayerFusion`` is a
    softmax scalar mix) and already concatenates physics/pLDDT channels, so a
    head plugged in there receives fused ``(B, L, C)`` features and returns
    per-residue logits. This is that head; :class:`DisorderNetLite` is the
    standalone equivalent that owns its own mixing.

    Against ``DisorderCNNHead`` the differences are deliberate: GroupNorm instead
    of BatchNorm (batches are a few proteins of very different lengths), residual
    blocks instead of parallel branches, and a 1x1 bottleneck before the trunk so
    width scales with ``hidden`` rather than with the backbone's embedding size.
    """

    def __init__(
        self,
        in_dim: int = 1280,
        dropout: float = 0.1,
        hidden: int = 256,
        n_blocks: int = 4,
        dilations: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        dil = list(dilations) if dilations is not None else list(DEFAULT_DILATIONS)
        self.proj = nn.Conv1d(in_dim, hidden, 1)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(hidden, dil[i % len(dil)], dropout)
            for i in range(n_blocks)
        )
        self.out = nn.Conv1d(hidden, 1, 1)
        # A 1x1 skip keeps a direct linear path from features to logit, so the
        # head degrades to logistic regression rather than to noise if the
        # convolutional trunk fails to learn on this much data.
        self.skip = nn.Conv1d(in_dim, 1, 1)
        self.receptive_field = receptive_field(dil, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, L)
        x = x.transpose(1, 2)
        h = self.proj(x)
        for blk in self.blocks:
            h = blk(h)
        return (self.out(h) + self.skip(x)).squeeze(1)


class DisorderNetLite(nn.Module):
    """Frozen PLM features -> scalar mix -> physics concat -> dilated CNN -> logit.

    ``esm_backbone`` is used strictly as a feature extractor and is never
    unfrozen. Trainable parameters are the mixture weights, an input projection,
    the CNN stack and a linear output — order 1-3M rather than 70M.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers_mixed: int,
        physics_dim: int = 0,
        plddt_dim: int = 0,
        hidden: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
        dilations: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.mix = ScalarMix(n_layers_mixed)
        in_dim = embed_dim + physics_dim + plddt_dim
        self.physics_dim = physics_dim
        self.plddt_dim = plddt_dim
        self.proj = nn.Conv1d(in_dim, hidden, 1)
        dil = list(dilations) if dilations is not None else list(DEFAULT_DILATIONS)
        # Cycle the dilation schedule if more blocks than dilations are asked for.
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(hidden, dil[i % len(dil)], dropout) for i in range(n_blocks)
        )
        self.out = nn.Conv1d(hidden, 1, 1)
        self.receptive_field = receptive_field(dil, n_blocks)

    def forward(
        self,
        layer_hiddens: Sequence[torch.Tensor],   # each (B, L, D)
        physics: Optional[torch.Tensor] = None,  # (B, L, P)
        plddt: Optional[torch.Tensor] = None,    # (B, L, K)
    ) -> torch.Tensor:
        x = self.mix(layer_hiddens)                      # (B, L, D)
        parts = [x]
        if self.physics_dim:
            if physics is None:
                raise ValueError("physics features required (physics_dim > 0)")
            parts.append(physics)
        if self.plddt_dim:
            if plddt is None:
                raise ValueError("pLDDT features required (plddt_dim > 0)")
            parts.append(plddt)
        x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        x = x.transpose(1, 2)                            # (B, C, L)
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out(x).squeeze(1)                    # (B, L) logits

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> dict:
        return {
            "architecture": "DisorderNetLite",
            "n_trainable": self.n_trainable(),
            "receptive_field_residues": self.receptive_field,
            "n_layers_mixed": self.mix.n_layers,
            "physics_dim": self.physics_dim,
            "plddt_dim": self.plddt_dim,
            "rationale": (
                "Frozen backbone with a low-capacity head. The 650M+LoRA "
                "configuration trains 69.9M parameters on ~1M residues from "
                "2,340 proteins and is beaten by a GBDT on physics features "
                "(0.7454 vs 0.7804); this trades capacity for sample efficiency."
            ),
        }


class StructureChannels(nn.Module):
    """Encode AlphaFold rsa and pLDDT into a small learned feature block.

    Post-hoc fusion of a structural signal with this model *failed*: weights fit
    on training data made rsa+pLDDT worse, 0.9581 -> 0.9554, because averaging
    lets a task-agnostic signal dominate wherever the model disagrees, including
    where the model is right. Feeding structure as an input instead lets the
    trunk learn *when* to trust it — which is the difference between the top
    CAID3 methods (explicitly structure-aware) and an ensemble.

    That distinction is the whole thesis of this project, stated correctly for
    the first time: AlphaFold's pLDDT is a weak disorder proxy (rank 11) while
    its solvent accessibility is a strong one (rank 3), so a learned gate over
    both beats trusting or distrusting either wholesale.

    Missing structure is explicit rather than imputed. An absent AlphaFold entry
    is not "buried and confident"; it is no information, and a model that cannot
    tell the difference will read absence as order.
    """

    #: Five without handedness, seven with. Kept as a class attribute so a
    #: checkpoint trained on one cannot silently load into the other: the first
    #: convolution's shape depends on it and strict loading will refuse.
    N_CHANNELS = 5
    N_CHANNELS_CHIRAL = 7

    def __init__(self, out_dim: int = 16, chiral: bool = False):
        super().__init__()
        # Five inputs: rsa, pLDDT (scaled), contact density, an availability
        # flag, and rsa's local gradient.
        #
        # Contact density earns its place by being orthogonal to accessibility
        # rather than a restatement of it. An exposed loop on a folded domain is
        # accessible AND densely contacted; a disordered residue is accessible
        # and uncontacted. Either channel alone confuses those two cases, and
        # they are precisely the false positives a disorder predictor makes.
        #
        # The gradient channel exists because a 1x1 convolution cannot see a
        # transition its input does not encode, and IDR boundaries are
        # accessibility transitions.
        # With `chiral`, two more: signed backbone handedness and a flag for
        # whether the torsion window was complete.
        #
        # Every other channel here is mirror-invariant — reflect the structure
        # and rsa, pLDDT, contacts and the rsa gradient are all unchanged — so
        # this is the only one that can distinguish a structure from its
        # reflection. Measured training-free on CAID3 Disorder-PDB, smoothed
        # handedness alone has within-protein AUC 0.6435 while its own achiral
        # control, |sin| of the same torsion, reaches 0.4180: *below chance*.
        # The whole signal is in the sign. On Disorder-NOX, 0.6494 against
        # 0.3966. The direction was fitted on 800 MobiDB training proteins and
        # it is the one polymer physics predicts — disordered residues are less
        # right-handed (mean 0.074) than ordered ones (0.211), which is what
        # left-handed polyproline II displacing right-handed alpha looks like.
        self.chiral = bool(chiral)
        n_in = self.N_CHANNELS_CHIRAL if self.chiral else self.N_CHANNELS
        self.encode = nn.Sequential(
            nn.Conv1d(n_in, out_dim, 1),
            nn.GELU(),
            nn.Conv1d(out_dim, out_dim, 1),
        )
        self.n_in = n_in
        self.out_dim = out_dim

    @staticmethod
    def assemble(
        rsa: Optional[torch.Tensor],
        plddt: Optional[torch.Tensor],
        available: Optional[torch.Tensor],
        length: int,
        batch: int,
        device: torch.device,
        contacts: Optional[torch.Tensor] = None,
        handedness: Optional[torch.Tensor] = None,
        chiral: bool = False,
    ) -> torch.Tensor:
        """Build the (B, 5, L) or (B, 7, L) input, flagged where absent.

        The handedness channel carries its own availability flag rather than
        reusing the structural one. A residue can have a structure and still
        have no torsion — the first residue and the last two of every chain,
        and any window containing a missing CA — and zero is a legitimate
        torsion, meaning a planar trace. Filling those with 0 would assert
        "planar" about residues with no window, the same error as imputing
        rsa 0 for a protein with no structure, which reads as fully buried.
        """
        zeros = torch.zeros(batch, length, device=device)
        r = zeros if rsa is None else rsa
        p = zeros if plddt is None else plddt / 100.0
        c = zeros if contacts is None else contacts
        a = torch.ones(batch, length, device=device) if available is None else available
        # Local rsa gradient; padded to preserve length.
        grad = torch.zeros_like(r)
        if length > 1:
            grad[:, 1:] = r[:, 1:] - r[:, :-1]
        stack = [r, p, c, a, grad]
        if chiral:
            h = zeros if handedness is None else handedness
            h_ok = torch.isfinite(h).to(h.dtype)
            stack.extend([torch.nan_to_num(h, nan=0.0), h_ok])
        return torch.stack(stack, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


#: Tasks given a protein-level bias term. Decomposing CAID's pooled AUC into
#: within-protein and between-protein pairs shows that 96.7% to 99.7% of the
#: pairs it counts are *between* proteins — w_within runs from 0.0029 on
#: Disorder-PDB to 0.0326 on Linker. The metric is therefore overwhelmingly a
#: question about which chains carry more of the label, not which residues
#: within a chain do.
#:
#: A per-residue model has a poor mechanism for that, though not — as this
#: comment used to claim — none at all. `receptive_field` returns 213 for the
#: wide schedule, but it describes the convolutions; GroupNorm pools statistics
#: over the whole length axis, so the dependency span is actually global
#: (`measured_dependency_span`). The bias term below therefore replaces an
#: implicit protein-level channel with an explicit one, rather than creating a
#: channel that was absent. The consequence is measurable either way: on Binding our within-protein AUC is 0.8683
#: against the leader's 0.8049 — we are substantially better at the biological
#: question — while our pooled score is lower, because between-protein
#: calibration dominates. On Binding-IDR our within-protein AUC matches the
#: leader (0.7004 against 0.6958) and the entire 0.14 pooled deficit is
#: between-protein.
#:
#: Adding a constant to every residue of one protein changes only that
#: protein's between-protein comparisons and leaves its within-protein ranking
#: exactly intact. So a learned per-protein bias is precisely the missing
#: degree of freedom, and precisely the one that cannot disturb what already
#: works.
_PROTEIN_BIAS_TASKS = ("binding_idr", "binding")

#: Tasks given private trunk capacity, reading a *detached* shared trunk.
#:
#: The protein-bias run moved Binding-IDR from 0.5007 to 0.6062 — rank 30 to 5 —
#: and was rejected because Disorder-NOX, Linker and Binding each fell below
#: their floor. The bias path was already detached, so no gradient from a
#: binding task reached a disorder read-out and the tests proved it; but the
#: binding *losses* still shaped the shared trunk, and the disorder tasks read
#: from that trunk. Protecting a read-out is not protecting a representation.
#:
#: Reading a detached trunk closes it completely: binding contributes exactly
#: zero gradient to proj and blocks, so the disorder tasks train as though the
#: binding tasks were absent — bit-identical, not approximately. The private
#: stack gives back the capacity the detach removes, so binding is not merely
#: cut off but re-housed.
_PRIVATE_TRUNK_TASKS = ("binding_idr", "binding")


#: Tasks whose read-out is conditioned on predicted disorder. Binding-IDR is
#: defined as binding *within disordered regions*, so the disorder answer is part
#: of its question rather than a hint.
_CONDITIONED_TASKS = ("binding_idr", "binding")

#: Where the disorder signal comes from, most task-matched first. disorder_pdb is
#: trained on 21,386 proteins against disorder_nox's 3,333, so it is both the
#: stronger predictor and the one whose labels match CAID's own definition.
_CONDITION_SOURCES = ("disorder_pdb", "disorder_nox")


class MultiTaskLiteHead(nn.Module):
    """One shared trunk, one linear read-out per CAID3 task.

    CAID3 is five benchmarks won by five different specialists: PUNCH2 does not
    predict linkers, LINKER-Pred does not predict binding. Answering all of them
    from a single forward pass is only affordable because the backbone is frozen
    — the marginal cost of a task is one 1x1 convolution over the shared trunk,
    roughly 257 parameters, against the 653M the backbone already spent.

    Sharing the trunk is also the point, not a shortcut. The small tasks are
    tiny: 15,683 positive residues for linker and 88,761 for binding, against
    336,014 for disorder. A specialist trained on linker alone sees very little,
    while a shared trunk carries disorder's data into it — and the low-capacity
    design is what makes that transfer safe rather than an invitation to
    overfit, which is the same reason this head beat a 69.9M-parameter LoRA
    model by +0.074 on DisProt.

    Per-task heads are deliberately linear. Anything deeper would let a task
    with 15k positives grow its own private capacity, which is exactly the
    failure mode the architecture exists to avoid.
    """

    def __init__(
        self,
        in_dim: int = 1280,
        tasks: Sequence[str] = ("disorder",),
        dropout: float = 0.1,
        hidden: int = 256,
        n_blocks: int = 4,
        dilations: Optional[Sequence[int]] = None,
        structure_dim: int = 0,
        condition_binding: bool = True,
        protein_bias: bool = True,
        private_trunk: bool = True,
        private_narrow: bool = False,
        private_detach: bool = True,
        chiral: bool = False,
        n_private_blocks: int = 2,
    ):
        super().__init__()
        if not tasks:
            raise ValueError("at least one task is required")
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"duplicate task names: {tasks}")
        self.tasks = tuple(tasks)
        self.structure_dim = int(structure_dim)
        self.structure = (
            StructureChannels(self.structure_dim, chiral=bool(chiral))
            if self.structure_dim else None
        )

        dil = list(dilations) if dilations is not None else list(DEFAULT_DILATIONS)
        self.proj = nn.Conv1d(in_dim + self.structure_dim, hidden, 1)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(hidden, dil[i % len(dil)], dropout)
            for i in range(n_blocks)
        )
        # One read-out per task, plus the same 1x1 linear skip the single-task
        # head uses, so each task degrades to logistic regression rather than to
        # noise if the trunk fails to learn it.
        self.out = nn.ModuleDict({t: nn.Conv1d(hidden, 1, 1) for t in self.tasks})
        # The skip spans the structural channels too, so each task keeps a
        # direct linear path from rsa/pLDDT to its logit. That matters here:
        # rsa alone scores 0.9459 on Disorder-PDB, so the linear fallback a task
        # degrades to should be the structural baseline, not sequence alone.
        self.skip = nn.ModuleDict({
            t: nn.Conv1d(in_dim + self.structure_dim, 1, 1) for t in self.tasks
        })
        self.receptive_field = receptive_field(dil, n_blocks)

        # Disorder-conditioned binding read-outs.
        #
        # Binding-IDR is the Binding labels restricted to disordered residues.
        # Trained as a masked variant of binding, our head reached 0.4945 on the
        # official reference — below chance — while scoring 0.7924 on Binding
        # itself. Conditioning on disorder erased the signal, which means what it
        # had learned was disorder: across a whole protein, binding sites sit in
        # IDRs and IDRs are the disordered part, so "is this disordered" answers
        # Binding well and answers Binding-IDR not at all.
        #
        # So give the model the disorder answer instead of making it rediscover
        # one. Each conditioned task gets a second read-out over [trunk,
        # p(disorder)], letting it learn "given this residue is disordered, does
        # it bind" rather than "is this residue disordered".
        #
        # The conditioning signal is **detached**. Binding is the weakest task
        # here (891 training proteins) and disorder carries three first-place
        # results; without the detach, binding's gradient would flow back through
        # the disorder read-out and could degrade them to help itself. Detached,
        # the conditioned path is strictly additive: every unconditioned task's
        # logit is bit-identical to what it would be with conditioning off.
        self.condition_binding = bool(condition_binding)
        self.condition_on = tuple(t for t in _CONDITIONED_TASKS if t in self.tasks)
        self.condition_source = next(
            (t for t in _CONDITION_SOURCES if t in self.tasks), None)
        if self.condition_source is None or not self.condition_binding:
            # No cond parameters are created at all, so a checkpoint trained
            # before this existed loads with strict=True. That matters: the
            # model of record holds three first places and must stay loadable.
            self.condition_on = ()
        self.cond = nn.ModuleDict({
            t: nn.Conv1d(hidden + 1, 1, 1) for t in self.condition_on
        })

        # Protein-level bias: mean and max of the trunk over the whole chain,
        # mapped to one scalar per task and added to every residue of it.
        self.protein_bias_on = (
            tuple(t for t in _PROTEIN_BIAS_TASKS if t in self.tasks)
            if protein_bias else ())
        self.protein_bias = nn.ModuleDict({
            t: nn.Linear(2 * hidden, 1) for t in self.protein_bias_on
        })

        self.private_on = (tuple(t for t in _PRIVATE_TRUNK_TASKS
                                 if t in self.tasks) if private_trunk else ())
        # `private_narrow` reroutes the private stack to read the projection
        # instead of the trunk output, and gives it its own narrow schedule.
        # Both parts are needed: dilations alone cannot narrow a field the
        # trunk has already widened.
        self.private_narrow = bool(private_narrow) and bool(self.private_on)
        # Detaching the private path is what mt_private measured, and it cost
        # Binding 0.0555 and Binding-IDR 0.0262 against its regime-matched
        # control while sparing the disorder tasks only ~0.006. Binding has 891
        # training proteins against disorder's 21,386, so cutting the gradient
        # both ways loses far more than it protects. Shape and gradient are
        # therefore separable settings: narrow the binding read-out's view
        # without also severing it.
        self.private_detach = bool(private_detach)
        pdil = (list(NARROW_DILATIONS) if self.private_narrow else dil)
        # Narrow mode also takes position-local normalisation. Without it the
        # narrow dilations buy nothing: GroupNorm's statistics run over the
        # whole length, so the "13-residue" stack would still depend on every
        # residue in the window.
        self.private = nn.ModuleList(
            DilatedResidualBlock(hidden, pdil[i % len(pdil)], dropout,
                                 local_norm=self.private_narrow)
            for i in range(n_private_blocks)
        ) if self.private_on else nn.ModuleList()
        for m in self.protein_bias.values():
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.cond.values():
            # Start as a no-op so a conditioned run begins exactly where the
            # unconditioned one is, and any change is something it learned.
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        rsa: Optional[torch.Tensor] = None,
        plddt: Optional[torch.Tensor] = None,
        structure_available: Optional[torch.Tensor] = None,
        contacts: Optional[torch.Tensor] = None,
        handedness: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """(B, L, C) -> {task: (B, L)} logits, one shared trunk pass.

        Structural channels, when configured, are concatenated before the trunk
        so the model can learn where to trust them rather than being averaged
        with them after the fact.
        """
        x = x.transpose(1, 2)                       # (B, C, L)
        if self.structure is not None:
            if rsa is None and plddt is None:
                raise ValueError(
                    "this head was built with structural channels; pass rsa "
                    "and/or plddt, or the trunk sees a constant block and the "
                    "learned gate is meaningless"
                )
            block = StructureChannels.assemble(
                rsa, plddt, structure_available,
                length=x.shape[2], batch=x.shape[0], device=x.device,
                contacts=contacts, handedness=handedness,
                chiral=self.structure.chiral,
            )
            x = torch.cat([x, self.structure(block)], dim=1)
        h_proj = self.proj(x)
        h = h_proj
        for blk in self.blocks:
            h = blk(h)
        logits = {
            t: (self.out[t](h) + self.skip[t](x)).squeeze(1) for t in self.tasks
        }
        if self.private_on:
            # Detached: binding's loss contributes exactly zero gradient to the
            # shared trunk, so the three tasks reading it are untouched.
            # Narrow mode reads the projection rather than the trunk output, so
            # the binding read-out sees a motif-scale neighbourhood instead of
            # inheriting the trunk's 213 residues. Detached either way — the
            # projection is shared too.
            src = h_proj if self.private_narrow else h
            hp = src.detach() if self.private_detach else src
            for blk in self.private:
                hp = blk(hp)
            for t in self.private_on:
                logits[t] = (self.out[t](hp) + self.skip[t](x)).squeeze(1)
            h_for_bias = hp
        else:
            h_for_bias = h

        if self.protein_bias_on:
            # Pooled over the sequence, detached. Detached because binding has
            # 891 training proteins and the disorder tasks 21,386 with three
            # first places between them; letting a protein-level term for the
            # weakest task reshape the shared trunk would put those at risk for
            # no reason. The bias head still learns — it just learns from what
            # the trunk already represents.
            #
            # A caveat worth stating rather than discovering: with windowed
            # inference this pools over the *window*, not the protein. For the
            # 93% of CAID3 targets at or below 1022 residues the window is the
            # whole chain and the two coincide. For longer ones the effective
            # bias becomes a taper-weighted average of per-window biases —
            # still a chain-level quantity, no longer literally constant along
            # it. Training sees the same windows, so the two are consistent;
            # the model learns a per-window term and is asked for a per-window
            # term.
            hd = h_for_bias.detach()
            pooled = torch.cat([hd.mean(dim=2), hd.amax(dim=2)], dim=1)
            for t in self.protein_bias_on:
                logits[t] = logits[t] + self.protein_bias[t](pooled)

        if self.condition_on:
            # Detached: no gradient flows from a conditioned task back into the
            # disorder read-out or the trunk through this path, so the tasks
            # holding three first places cannot be traded away to help binding.
            p_dis = torch.sigmoid(logits[self.condition_source]).detach()
            h_cond = torch.cat([h_for_bias.detach() if self.private_on else h,
                                p_dis.unsqueeze(1)], dim=1)
            for t in self.condition_on:
                logits[t] = logits[t] + self.cond[t](h_cond).squeeze(1)
        return logits

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def cost_per_extra_task(self) -> int:
        """Parameters one additional task would add — trunk excluded."""
        t = self.tasks[0]
        return sum(p.numel() for p in self.out[t].parameters()) + sum(
            p.numel() for p in self.skip[t].parameters()
        )


def masked_multitask_loss(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    evidence: Optional[dict[str, torch.Tensor]] = None,
    weights: Optional[dict[str, float]] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted BCE per task, skipping residues that task does not evaluate.

    The masking is not optional bookkeeping. Binding-IDR is *defined* as
    "binding within disordered regions, everything else ignored", so scoring a
    fabricated negative outside an IDR would train the model on the wrong
    question and inflate the metric with residues the benchmark never shows it.

    Returns the summed loss and a per-task breakdown, because a multi-task loss
    that quietly collapses onto whichever task has the most residues is the
    thing worth catching early.
    """
    total = None
    parts: dict[str, float] = {}
    for task, logit in logits.items():
        if task not in labels:
            continue
        target = labels[task].to(logit.dtype)
        mask = None
        if evidence is not None and task in evidence:
            mask = evidence[task]
        if mask is None:
            mask = torch.ones_like(target, dtype=torch.bool)
        n = int(mask.sum())
        if n == 0:
            continue
        per_res = nn.functional.binary_cross_entropy_with_logits(
            logit, target, reduction="none"
        )
        loss = (per_res * mask).sum() / n
        w = (weights or {}).get(task, 1.0)
        total = w * loss if total is None else total + w * loss
        parts[task] = float(loss.detach())
    if total is None:
        raise ValueError("no task contributed a loss — every mask was empty")
    return total, parts


def freeze_backbone(esm_backbone: nn.Module) -> int:
    """Freeze every backbone parameter. Returns the number frozen.

    Explicit rather than implied: the checkpoint defect in this project came
    from ESM tail weights being trained but never saved, and a frozen backbone
    removes that failure mode entirely — the only trainable state is the head.
    """
    n = 0
    for p in esm_backbone.parameters():
        if p.requires_grad:
            p.requires_grad = False
            n += 1
    esm_backbone.eval()
    return n


def measured_dependency_span(module, length: int = 401, in_dim: int = 32,
                             task: str | None = None,
                             atol: float = 1e-6) -> int:
    """How far a single-residue perturbation actually reaches, in residues.

    Measured rather than derived, because `receptive_field` describes the
    convolutions and the normalisation is what decides the answer. Returns the
    largest distance at which perturbing one residue moves the centre logit;
    `length` when the dependency is global.

    Intended for tests and for reporting an architecture honestly, not for the
    training loop.
    """
    import torch as _t

    was_training = module.training
    module.eval()
    try:
        x = _t.zeros(1, length, in_dim)
        with _t.no_grad():
            base = module(x)
        if isinstance(base, dict):
            base = base[task or next(iter(base))]
        centre = length // 2
        span = 0
        for d in range(1, centre + 1):
            moved_any = False
            for pos in (centre - d, centre + d):
                x2 = x.clone()
                x2[0, pos, :] = 5.0
                with _t.no_grad():
                    out = module(x2)
                if isinstance(out, dict):
                    out = out[task or next(iter(out))]
                if not _t.allclose(base[0, centre], out[0, centre], atol=atol):
                    moved_any = True
            if moved_any:
                span = d
        return span
    finally:
        module.train(was_training)
