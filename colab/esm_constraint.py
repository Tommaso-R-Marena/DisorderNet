"""Evolutionary constraint inside disordered regions, from the frozen backbone.

The hypothesis, and the biology behind it.

Short linear motifs and molecular recognition features are the parts of an
intrinsically disordered region that do something: they bind. Because they bind,
they are under selection, and they are measurably more conserved than the
disordered sequence around them, which drifts. This is the standard account of
how SLiMs are found — functional constraint against a fast-evolving background.

ESM-2 was trained by masked-token prediction over UniRef. Its probability for
the residue actually present, given the rest of the chain, is therefore a
learned statement about how expected that residue is: a conservation proxy that
needs no alignment, no orthologues and no extra data. We already run this
backbone.

So the prediction is specific and falsifiable:

  **Within a disordered region, binding residues should carry higher
  pseudo-likelihood than non-binding ones.**

That is a claim about Binding-IDR — the benchmark where the whole field is close
to helpless. Its leader reaches 0.641, and the top eight entrants have a mean
pairwise rank correlation of 0.336 with one pair at -0.209, meaning there is no
agreed signal to speak of.

The same quantity aggregated over a chain's disordered regions gives a
protein-level number: how constrained are this protein's IDRs. That is the
between-protein axis where our own deficit lives entirely — our Binding-IDR
within-protein AUC matches the leader (0.7004 against 0.6958) while our
between-protein AUC is 0.4982, chance, against their 0.6400.

If the hypothesis is wrong, the AUC will sit at 0.5 and that is the end of it.
"""

from __future__ import annotations

import numpy as np
import torch


def pseudo_log_likelihood(
    esm,
    batch_converter,
    alphabet,
    sequence: str,
    device,
    n_groups: int = 16,
    batch_size: int = 8,
) -> np.ndarray:
    """Per-residue log p(x_i | x_without_i) from the masked language model.

    Exact pseudo-likelihood masks one position per forward pass, costing L
    passes for a chain of length L. Instead positions are partitioned into
    ``n_groups`` interleaved strides and one stride is masked per pass, so the
    cost is ``n_groups`` passes regardless of length.

    The approximation: masked positions within a pass are ``n_groups`` residues
    apart and each sees the other as a mask rather than as its residue. At the
    default spacing of 16 that perturbs a small fraction of a context window
    which spans the whole chain, and the alternative — one pass per residue — is
    16x to 3000x more expensive for a quantity used as a ranking signal, not a
    calibrated probability.
    """
    n = len(sequence)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    _, _, base = batch_converter([("q", sequence)])
    base = base.to(device)
    # batch_converter prepends BOS, so residue i is at token i + 1.
    offset = 1
    mask_idx = alphabet.mask_idx

    out = np.full(n, np.nan, dtype=np.float64)
    groups = [list(range(g, n, n_groups)) for g in range(min(n_groups, n))]
    groups = [g for g in groups if g]

    with torch.no_grad():
        for s in range(0, len(groups), batch_size):
            chunk = groups[s:s + batch_size]
            batch = base.repeat(len(chunk), 1).clone()
            for bi, positions in enumerate(chunk):
                for p in positions:
                    batch[bi, p + offset] = mask_idx
            logits = esm(batch, repr_layers=[], return_contacts=False)["logits"]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            for bi, positions in enumerate(chunk):
                for p in positions:
                    true_tok = int(base[0, p + offset])
                    out[p] = float(logprobs[bi, p + offset, true_tok])

    if np.isnan(out).any():                       # pragma: no cover - defensive
        raise RuntimeError(
            f"{int(np.isnan(out).sum())} of {n} positions were never masked; "
            f"the stride partition does not cover the sequence")
    return out


def local_constraint(pll: np.ndarray, window: int = 31) -> np.ndarray:
    """Pseudo-likelihood relative to its own neighbourhood.

    The raw value is dominated by composition — a lysine in a lysine-rich
    stretch is predictable whether or not it binds — so what matters is a
    residue being *more* constrained than the sequence around it. Subtracting a
    running mean removes the regional baseline and leaves the local excess,
    which is what a motif embedded in a drifting region should look like.
    """
    n = len(pll)
    if n == 0:
        return pll
    w = min(window, n if n % 2 else n - 1)
    if w < 3:
        return pll - pll.mean()
    kernel = np.ones(w, dtype=np.float64) / w
    padded = np.pad(pll, w // 2, mode="edge")
    return pll - np.convolve(padded, kernel, mode="valid")[:n]


def protein_constraint(pll: np.ndarray, disorder_mask: np.ndarray) -> float:
    """Mean constraint over a chain's disordered residues.

    A protein-level scalar for the between-protein axis: how constrained is
    this chain's disordered sequence, and therefore how much of it is likely to
    be doing something. Returns NaN when the chain has no disordered residues,
    rather than a zero that would read as a real measurement.
    """
    m = np.asarray(disorder_mask, dtype=bool)
    if not m.any():
        return float("nan")
    return float(np.nanmean(pll[m]))
