#!/usr/bin/env python3
"""Does evolutionary constraint mark the binding parts of a disordered region?

Training-free test of a biophysical claim. Short linear motifs bind, therefore
they are under selection, therefore they should be more conserved than the
disordered sequence around them. ESM-2's masked-token probability for the
residue actually present is a conservation proxy that needs no alignment.

Two predictions, both falsifiable here:

**Per-residue.** Within a disordered region, binding residues carry higher local
constraint than non-binding ones — so `local_constraint` should score above 0.5
on Binding-IDR, where the field's leader manages 0.641 and the top eight
entrants agree with each other barely at all.

**Per-protein.** A chain whose disordered regions are more constrained has more
functional IDR, so mean constraint over its disordered residues should rank
proteins by how much IDR binding they carry. That is the between-protein axis
where our whole Binding-IDR deficit lives: our within-protein AUC matches the
leader at 0.7004 against 0.6958, and our between-protein AUC is 0.4982 —
chance — against their 0.6400.

Every benchmark is reported, and the raw quantity is reported beside the
baseline-corrected one. If both sit at 0.5, the hypothesis is wrong and that is
the finding.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.auc_decomposition import decompose_auc  # noqa: E402
from colab.caid3_official import (  # noqa: E402
    LEADERS,
    TASKS,
    evaluated_mask,
    read_reference,
)
from colab.esm_constraint import (  # noqa: E402
    local_constraint,
    protein_constraint,
    pseudo_log_likelihood,
)

REFS = os.environ.get("CAID3_OFFICIAL_DIR",
                      "/scratch4/sfried3/jbeale3_disordernet/caid3_official")
CACHE = os.environ.get("CONSTRAINT_CACHE",
                       "/scratch4/sfried3/jbeale3_disordernet/esm_constraint.json")
WINDOW = int(os.environ.get("CONSTRAINT_WINDOW", "31"))


def load_backbone():
    from colab.disordernet_gpu import TrainConfig, setup_environment
    from colab.esm_backbone import load_esm_backbone
    from colab.lite_head import freeze_backbone

    cfg = setup_environment(TrainConfig.from_profile("lite", esm_backbone="650M"))
    esm, alphabet, batch_converter, _spec = load_esm_backbone(
        cfg.device, backbone="650M", use_gradient_checkpointing=False)
    freeze_backbone(esm)
    esm.eval()
    return esm, alphabet, batch_converter, cfg.device


def main() -> int:
    # One pseudo-likelihood per unique sequence, shared across benchmarks.
    sequences = {}
    for task in TASKS:
        for tid, (seq, _lab) in read_reference(
                os.path.join(REFS, f"{task}.fasta")).items():
            sequences[tid] = seq
    print(f"unique targets across all five references: {len(sequences)}")

    cached = {}
    if os.path.isfile(CACHE):
        cached = {k: np.asarray(v) for k, v in json.load(open(CACHE)).items()}
        print(f"cached pseudo-likelihoods: {len(cached)}")

    todo = [t for t in sequences if t not in cached
            or len(cached[t]) != len(sequences[t])]
    if todo:
        esm, alphabet, batch_converter, device = load_backbone()
        for i, tid in enumerate(todo, 1):
            cached[tid] = pseudo_log_likelihood(
                esm, batch_converter, alphabet, sequences[tid], device)
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} computed", flush=True)
        tmp = CACHE + ".part"
        with open(tmp, "w") as fh:
            json.dump({k: np.asarray(v).tolist() for k, v in cached.items()}, fh)
        os.replace(tmp, CACHE)
        print(f"wrote {CACHE}")

    from sklearn.metrics import roc_auc_score

    print(f"\n{'benchmark':<14}{'raw PLL':>9}{'local':>9}{'-local':>9}"
          f"{'leader':>9}  per-residue, training-free")
    for task in TASKS:
        ref = read_reference(os.path.join(REFS, f"{task}.fasta"))
        ys, raw, loc = [], [], []
        for tid, (_seq, lab) in ref.items():
            p = cached.get(tid)
            if p is None or len(p) != len(lab):
                continue
            m = evaluated_mask(lab)
            y = (np.frombuffer(lab.encode(), dtype=np.uint8)[m]
                 - ord("0")).astype(int)
            ys.append(y)
            raw.append(p[m])
            loc.append(local_constraint(p, window=WINDOW)[m])
        if not ys:
            continue
        y = np.concatenate(ys)
        if len(np.unique(y)) < 2:
            continue
        a_raw = roc_auc_score(y, np.concatenate(raw))
        a_loc = roc_auc_score(y, np.concatenate(loc))
        print(f"{task:<14}{a_raw:>9.4f}{a_loc:>9.4f}{1 - a_loc:>9.4f}"
              f"{LEADERS[task][1]:>9.3f}")

    # ── Per-protein: the between-protein axis ────────────────────────────
    print(f"\n{'benchmark':<14}{'targets':>8}{'spearman':>10}{'AUC_between':>13}"
          f"  protein-level constraint vs IDR binding content")
    from scipy.stats import spearmanr

    for task in TASKS:
        ref = read_reference(os.path.join(REFS, f"{task}.fasta"))
        xs, fracs, per_target = [], [], []
        for tid, (_seq, lab) in ref.items():
            p = cached.get(tid)
            if p is None or len(p) != len(lab):
                continue
            m = evaluated_mask(lab)
            if not m.any():
                continue
            y = (np.frombuffer(lab.encode(), dtype=np.uint8)[m]
                 - ord("0")).astype(int)
            c = protein_constraint(p, m)
            if not np.isfinite(c):
                continue
            xs.append(c)
            fracs.append(float(y.mean()))
            per_target.append((y, np.full(len(y), c)))
        if len(xs) < 8:
            continue
        rho, pv = spearmanr(xs, fracs)
        # A constant per protein: within-protein AUC is 0.5 by construction, so
        # the pooled value isolates the between-protein axis.
        d = decompose_auc([a for a, _ in per_target],
                          [b for _, b in per_target])
        between = d.get("auc_between")
        print(f"{task:<14}{len(xs):>8}{rho:>10.3f}"
              f"{(between if between is not None else float('nan')):>13.4f}"
              f"   p={pv:.4f}")

    print("\nspearman is protein-level constraint against the fraction of that")
    print("protein's evaluated residues that are positive. AUC_between is the")
    print("between-protein AUC of the constant-per-protein score, which is what")
    print("a protein-level feature can contribute and nothing more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
