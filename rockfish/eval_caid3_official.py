#!/usr/bin/env python3
"""Score a checkpoint on the official CAID3 challenge, against all 117 entrants.

This replaces evaluation against reconstructed references. CAID serves the five
challenge references and every entrant's per-residue predictions, so there is
nothing to reconstruct and every comparison is paired: our scores and a
competitor's scores on the same targets, resampled by protein.

Three rules this script will not break, each of which was learned by getting it
wrong first:

**Predict every target.** CAID pools over whatever a method returns, so
declining the hard targets raises the score. On Disorder-PDB the effect is
about +0.004 AUC, which is comparable to the gaps separating the top five. We
predict all targets and record coverage, and a run that fails to cover a target
fails loudly rather than quietly scoring a subset.

**Compare paired, not against a printed number.** "Our interval contains their
point estimate" is far weaker than a paired test, and it was the strongest thing
available while the competitors' predictions were unavailable. They are not.

**Report the rank honestly.** A rank is only meaningful with full coverage, so
the table shows every method's coverage next to its AUC.

Usage:
    python rockfish/eval_caid3_official.py --checkpoint DIR \
        --refs /scratch4/.../caid3_official \
        --predictions /scratch4/.../caid3_predictions \
        --disprot disprot_raw.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colab.caid3_official import (  # noqa: E402
    LEADERS,
    TASKS,
    holm_bonferroni,
    official_leaderboard,
    paired_bootstrap,
    rank_fuse,
    read_caid_predictions,
    read_reference,
    score_method,
    verify_composition,
)

#: Training-free structural baseline. It ranks 3rd on Disorder-PDB and is not
#: separable from the first-placed method, so beating the leader while failing
#: to beat this would not be a meaningful result.
STRUCTURAL_BASELINE = "AlphaFold-rsa"

OURS = "DisorderNet"


def assert_filtered_against(benchmark: str, checkpoint: str) -> None:
    """Refuse to score a checkpoint on a round it was not filtered against.

    The two CAID rounds share exactly one protein, so filtering against CAID3
    leaves 307 of CAID2's 348 Disorder-PDB targets in the training union. A
    CAID3-only checkpoint scored on CAID2 would be measuring memorisation and
    would look excellent doing it — the failure mode that makes a replication
    worthless is the one that produces the most impressive number.

    Checked from the checkpoint's own recorded filter rather than from the
    operator's intent, because the operator's intent is exactly what a
    mis-set environment variable overrides. ``caid3`` is not exempt: a run
    could as easily be launched against CAID2 references alone.
    """
    meta = os.path.join(checkpoint, "multitask_results.json")
    if not os.path.isfile(meta):
        raise SystemExit(
            f"{checkpoint} has no multitask_results.json, so the leak filter "
            f"it was trained under cannot be established. Refusing to score.")
    with open(meta) as fh:
        refs = json.load(fh).get("caid_leak_filter", {}).get("reference")
    refs = refs if isinstance(refs, list) else ([refs] if refs else [])
    if not any(benchmark in str(r) for r in refs):
        raise SystemExit(
            f"{checkpoint} was not filtered against {benchmark} — its "
            f"references are {refs or 'none recorded'}. Its training union "
            f"therefore contains {benchmark} targets, and any score here "
            f"would measure memorisation rather than generalisation.")
    print(f"leak filter covers {benchmark} ({len(refs)} references)")


#: Pre-registered analysis (results/caid3/PREREGISTRATION.md). The primary
#: family is exactly two tests on Disorder-PDB, unfused, all 319 targets. It is
#: fixed here so the confirmatory run cannot have its family redefined after the
#: numbers land — which is precisely what invalidated the previous run's
#: p-values, where 16 exploratory comparisons were made and the two that cleared
#: 0.05 were reported as the result.
PRIMARY_TASK = "disorder_pdb"
PRIMARY_OPPONENTS = (STRUCTURAL_BASELINE, "PUNCH2")

#: Non-inferiority floors — one per benchmark we currently place on, each the
#: corrected mt_windowed figure less a 0.005 margin
#: (results/caid3/PREREGISTRATION_2.md). Four rather than one, because there are
#: four placements to protect and any change to a shared trunk can trade them.
#: A breach rejects the variant outright, whatever it gained elsewhere.
NON_INFERIORITY_FLOORS = {
    "disorder_pdb": 0.9545,
    "disorder_nox": 0.8878,
    "linker": 0.9193,
    "binding": 0.7884,
}

#: Kept for the first pre-registration, whose primary task was Disorder-PDB.
NON_INFERIORITY_FLOOR = NON_INFERIORITY_FLOORS["disorder_pdb"]


def load_accessions(path):
    """DisProt id -> UniProt accession, for locating AlphaFold structures."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as fh:
        data = json.load(fh)
    entries = data if isinstance(data, list) else data.get("data", [])
    return {str(e["disprot_id"]): e["acc"] for e in entries
            if isinstance(e, dict) and e.get("disprot_id") and e.get("acc")}


def write_caid_submission(path, per_target, ref):
    """Emit our predictions in CAID's own format.

    Archiving them means the comparison can be rerun by anyone against the same
    published files, and it is also the format CAID would need for a future
    round.
    """
    with open(path + ".part", "w") as fh:
        for tid, (seq, _lab) in ref.items():
            probs = per_target.get(tid)
            if probs is None:
                continue
            fh.write(f">{tid}\n")
            for i, (aa, p) in enumerate(zip(seq, probs), 1):
                fh.write(f"{i}\t{aa}\t{p:.4f}\t{int(p >= 0.5)}\n")
    os.replace(path + ".part", path)
    return path


#: ESM-2 was trained at 1024 tokens and the head was trained with --max-len
#: 1022, so no protein longer than that was ever seen during training. Scoring a
#: 1500-residue chain in one pass puts both components far outside their regime,
#: and the result is not graceful degradation: on CAID3 Disorder-NOX targets over
#: 1500 residues the model scored 0.5183, chance, while AlphaFold-rsa scored
#: 0.8967 on those same targets. The signal was there and the model was throwing
#: it away.
WINDOW = 1022
STRIDE = 511


def _windows(n: int, window: int = WINDOW, stride: int = STRIDE):
    """Cover [0, n) with overlapping windows, the last one flush to the end."""
    if n <= window:
        return [(0, n)]
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] + window < n:
        starts.append(n - window)
    return [(s, s + window) for s in starts]


def _taper(length: int) -> np.ndarray:
    """Weight a window's contribution, low at its edges.

    A residue at the edge of a window has context on one side only, which is
    exactly the deficit windowing exists to avoid. Overlapping windows are
    averaged with a raised-cosine weight so each position is dominated by the
    window that saw the most of its neighbourhood.
    """
    if length == 1:
        return np.ones(1, dtype=np.float64)
    x = np.linspace(0.0, 1.0, length)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * x) + 1e-3


def predict_task(head, mix, esm, batch_converter, layer_ids, device, ref,
                 task, structures, structure_dim, batch_size=8):
    """Per-residue probabilities for every target, windowed to ESM's regime.

    Chunks from all targets go into one queue, so a few very long proteins do
    not serialise the whole pass, and short proteins are unaffected: a sequence
    at or below the window length yields exactly one chunk covering all of it.
    """
    t0 = time.perf_counter()
    jobs = []
    for tid, (seq, _lab) in ref.items():
        for (a, b) in _windows(len(seq)):
            jobs.append((tid, a, b, seq[a:b]))

    acc = {tid: np.zeros(len(seq), dtype=np.float64)
           for tid, (seq, _l) in ref.items()}
    wsum = {tid: np.zeros(len(seq), dtype=np.float64)
            for tid, (seq, _l) in ref.items()}

    with torch.no_grad():
        for s in range(0, len(jobs), batch_size):
            batch = jobs[s:s + batch_size]
            _, _, tokens = batch_converter([(f"{t}:{a}", sub)
                                            for t, a, _b, sub in batch])
            tokens = tokens.to(device)
            rep = esm(tokens, repr_layers=layer_ids, return_contacts=False)
            feats = mix([rep["representations"][i][:, 1:-1, :] for i in layer_ids])
            kw = {}
            if structure_dim:
                L = feats.shape[1]
                sr = torch.zeros(len(batch), L, device=device)
                sp = torch.zeros(len(batch), L, device=device)
                sa = torch.zeros(len(batch), L, device=device)
                sc = torch.zeros(len(batch), L, device=device)
                # NaN, not zero. Zero is a planar backbone; the head reads
                # finite-ness as this channel's availability flag, and filling
                # absent windows with 0 would assert "planar" about residues
                # with no torsion window at all.
                sh = torch.full((len(batch), L), float("nan"), device=device)
                for bi, (tid, a, b, _sub) in enumerate(batch):
                    f = structures.get(tid) or {}
                    if not f:
                        continue
                    # Structure channels must be sliced to the same window, or
                    # the head sees residue i's embedding beside residue a+i's
                    # accessibility.
                    r = np.asarray(f["rsa"], dtype=np.float32)[a:b]
                    q = np.asarray(f["plddt"], dtype=np.float32)[a:b]
                    k = min(len(r), len(q), L)
                    if k <= 0:
                        continue
                    sr[bi, :k] = torch.from_numpy(r[:k]).to(device)
                    sp[bi, :k] = torch.from_numpy(q[:k]).to(device)
                    sa[bi, :k] = 1.0
                    ct = np.asarray(f.get("contacts", np.zeros(b - a)),
                                    dtype=np.float32)[a:b]
                    kk = min(len(ct), k)
                    sc[bi, :kk] = torch.from_numpy(ct[:kk]).to(device)
                    hd = np.asarray(
                        f.get("handedness", np.full(b - a, np.nan)),
                        dtype=np.float32)[a:b]
                    kh = min(len(hd), k)
                    if kh > 0:
                        sh[bi, :kh] = torch.from_numpy(
                            np.ascontiguousarray(hd[:kh])).to(device)
                kw = {"rsa": sr, "plddt": sp, "structure_available": sa,
                      "contacts": sc, "handedness": sh}
            probs = torch.sigmoid(head(feats, **kw)[task]).float().cpu().numpy()
            for bi, (tid, a, b, sub) in enumerate(batch):
                n = min(len(sub), probs.shape[1])
                w = _taper(n)
                acc[tid][a:a + n] += probs[bi, :n].astype(np.float64) * w
                wsum[tid][a:a + n] += w

    out = {}
    for tid, seq_lab in ref.items():
        w = wsum[tid]
        if not (w > 0).all():
            raise RuntimeError(
                f"{tid}: {int((w <= 0).sum())} of {len(w)} residues were never "
                f"covered by a window. Scoring would silently use zeros.")
        out[tid] = acc[tid] / w
    return out, time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--refs", required=True, help="dir of official CAID3 fastas")
    ap.add_argument("--predictions", required=True,
                    help="dir of official .caid predictions from all entrants")
    ap.add_argument("--disprot", default=None)
    ap.add_argument("--structure-cache",
                    default="/scratch4/sfried3/jbeale3_disordernet/af_structures")
    ap.add_argument("--out", default=None)
    ap.add_argument("--submissions", default=None,
                    help="where to write our predictions in .caid format")
    ap.add_argument("--backbone", default="650M")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--benchmark", default="caid3", choices=("caid3", "caid2"),
                    help="which round the --refs belong to. The composition "
                         "guard is round-specific: it correctly refused to "
                         "score CAID2 references against CAID3's counts.")
    args = ap.parse_args(argv)

    from colab.caid3_official import TASKS_CAID2, verify_composition_for
    round_tasks = TASKS if args.benchmark == "caid3" else TASKS_CAID2
    for task in round_tasks:
        verify_composition_for(args.benchmark, task,
                               os.path.join(args.refs, f"{task}.fasta"))
    print(f"all {len(round_tasks)} {args.benchmark} references match "
          f"their published composition")
    assert_filtered_against(args.benchmark, args.checkpoint)

    ckpt = os.path.join(args.checkpoint, "multitask_head.pt")
    if not os.path.isfile(ckpt):
        print(f"ERROR: no multitask_head.pt under {args.checkpoint}", file=sys.stderr)
        return 2
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    tasks = tuple(payload["tasks"])
    layer_ids = list(payload["layer_ids"])
    structure_dim = int(payload.get("structure_dim", 0))
    # Default False: a checkpoint predating the conditioned read-out has no
    # cond.* tensors, and building a head that expects them would fail
    # strict-loading the model of record.
    condition_binding = bool(payload.get("condition_binding", False))
    protein_bias = bool(payload.get("protein_bias", False))
    private_trunk = bool(payload.get("private_trunk", False))
    private_narrow = bool(payload.get("private_narrow", False))
    chiral = bool(payload.get("chiral", False))

    from colab.disordernet_gpu import TrainConfig, setup_environment
    cfg = setup_environment(TrainConfig.from_profile("lite", esm_backbone=args.backbone))
    device = cfg.device

    from colab.esm_backbone import load_esm_backbone
    from colab.lite_head import (WIDE_DILATIONS, MultiTaskLiteHead, ScalarMix,
                                 freeze_backbone)
    esm, _a, batch_converter, spec = load_esm_backbone(
        device, backbone=args.backbone, use_gradient_checkpointing=False)
    freeze_backbone(esm)
    mix = ScalarMix(len(layer_ids)).to(device)
    mix.load_state_dict(payload["mix"])
    head = MultiTaskLiteHead(in_dim=spec.embed_dim, tasks=tasks,
                             structure_dim=structure_dim,
                             condition_binding=condition_binding,
                             protein_bias=protein_bias,
                             private_trunk=private_trunk,
                             private_narrow=private_narrow,
                             chiral=chiral,
                             dilations=(WIDE_DILATIONS
                                        if payload.get("wide_receptive_field")
                                        else None)).to(device)
    head.load_state_dict(payload["head"])
    head.eval()
    mix.eval()
    print(f"checkpoint: tasks={list(tasks)} structure_dim={structure_dim} "
          f"condition_binding={condition_binding} "
          f"protein_bias={protein_bias} "
          f"trainable={sum(p.numel() for p in head.parameters()):,}")

    subs = args.submissions or os.path.join(args.checkpoint, "caid_submissions")
    os.makedirs(subs, exist_ok=True)

    results = {}
    predictions_by_task: dict[str, dict] = {}
    for task in round_tasks:
        if task not in tasks:
            continue
        ref = read_reference(os.path.join(args.refs, f"{task}.fasta"))

        structures = {}
        if structure_dim:
            from colab.structure_rsa import structure_features
            acc_by_id = load_accessions(args.disprot)
            n_have = 0
            for tid, (seq, _l) in ref.items():
                acc = acc_by_id.get(tid)
                f = (structure_features(acc, seq, args.structure_cache,
                                        allow_fetch=False) if acc else None)
                structures[tid] = f or {}
                n_have += 1 if f else 0
            print(f"  {task}: structure for {n_have}/{len(ref)} targets")
            if n_have == 0:
                print(f"ERROR: structure-aware checkpoint but no structures for "
                      f"{task}; refusing to score it on a constant block",
                      file=sys.stderr)
                return 2

        preds, wall = predict_task(
            head, mix, esm, batch_converter, layer_ids, device, ref, task,
            structures, structure_dim, batch_size=args.batch_size)

        if len(preds) != len(ref):
            print(f"ERROR: {task}: predicted {len(preds)} of {len(ref)} targets. "
                  f"Partial coverage inflates the score and makes the rank "
                  f"meaningless.", file=sys.stderr)
            return 2

        # One file per task. A single {OURS}.caid was overwritten by each task
        # in turn, so the archived submission was whichever task ran last.
        our_path = write_caid_submission(
            os.path.join(subs, f"{OURS}-{task}.caid"), preds, ref)
        # Score the archived file, not the in-memory probabilities. The
        # submission stores four decimals — CAID's own convention — so the two
        # differ slightly, and on Linker they round to 0.8885 against 0.8884.
        # The archived file is what anyone else can rerun, so it is what gets
        # reported; a published number that its own artefact cannot reproduce is
        # not reproducible, however small the discrepancy.
        preds = {k: np.asarray(v) for k, v in
                 read_caid_predictions(our_path).items()}
        predictions_by_task[task] = preds
        ours = score_method(ref, preds)

        board = official_leaderboard(task, args.refs, args.predictions)
        better = [r for r in board if r["auc"] > ours["auc"]]
        rank = len(better) + 1
        # Two fields, always both. Ranking among every entrant is CAID's own
        # accounting, and it rewards declining hard targets — the single method
        # above us on Linker skipped 4 of 31, and 9 of the 12 above us on
        # Disorder-NOX skipped some. Ranking among entrants that answered every
        # target is the equal-footing field. Reporting only the first
        # understates us; reporting only the second is cherry-picking, since a
        # method that skips is not disqualified, only scored on an easier set.
        # The paired tests below settle it properly, on shared targets.
        full_cov = [r for r in board if r["coverage"] >= 1.0]
        rank_full = len([r for r in full_cov if r["auc"] > ours["auc"]]) + 1
        skipped_above = [r for r in better if r["coverage"] < 1.0]

        # Paired tests need our submission alongside theirs.
        staged = os.path.join(subs, "_paired")
        os.makedirs(staged, exist_ok=True)
        for fn in os.listdir(args.predictions):
            dst = os.path.join(staged, fn)
            if not os.path.exists(dst):
                os.symlink(os.path.join(args.predictions, fn), dst)
        shutil.copy(our_path, os.path.join(staged, f"{OURS}.caid"))
        our_name = OURS

        # Fuse with the training-free structural predictor. Our head takes rsa
        # as an input and still scores below rsa alone on Disorder-NOX (0.816
        # against 0.836), so the 24-dimensional gate is not using it. An equal
        # weighted rank average recovers 0.860 — more than the head gains from
        # having the feature at all. Weights are equal and fixed, not fitted:
        # on every CAID3 task where the fusion helps, the AUC-maximising weight
        # sits between 0.45 and 0.50, so equal weighting is both the honest
        # choice and the empirically right one.
        fused_row = None
        baseline_path = os.path.join(args.predictions,
                                     f"{STRUCTURAL_BASELINE}.caid")
        if os.path.isfile(baseline_path):
            base = read_caid_predictions(baseline_path)
            fused = rank_fuse([{k: np.asarray(v) for k, v in preds.items()},
                               base], ref)
            if fused:
                fused_path = write_caid_submission(
                    os.path.join(subs, f"{OURS}-fused-{task}.caid"), fused, ref)
                fused_row = score_method(ref, fused)
                if fused_row:
                    shutil.copy(fused_path,
                                os.path.join(staged, f"{OURS}-fused.caid"))

        # LEADERS is the CAID3 published table, verified against the challenge
        # site. A replication round has no entry there — PUNCH2 did not enter
        # CAID2 — and reaching into it anyway named an opponent whose file does
        # not exist, which is how this crashed. For any other round the leader
        # is taken empirically: the top full-coverage entrant on this very
        # reference, recomputed from the raw prediction files.
        if args.benchmark == "caid3":
            # (method, auc, aps) — three fields, not two. Unpacking it as a
            # pair passed a source-text test and failed at runtime two minutes
            # into a GPU job, which is what source-text tests are worth.
            leader, leader_auc = LEADERS[task][0], LEADERS[task][1]
        else:
            top = (full_cov or board)[0]
            leader, leader_auc = top["method"], top["auc"]
        paired = {}
        for opponent in dict.fromkeys([leader, STRUCTURAL_BASELINE,
                                       board[0]["method"]]):
            if not os.path.exists(os.path.join(staged, f"{opponent}.caid")):
                continue
            paired[opponent] = paired_bootstrap(
                task, args.refs, staged, OURS, opponent, n_boot=args.n_boot)

        if fused_row:
            fused_better = [r for r in board if r["auc"] > fused_row["auc"]]
            # Guarded like the loop above. Without this the fused comparison
            # opened the opponent's file unconditionally and took the whole
            # evaluation down when the named leader was not an entrant.
            if os.path.exists(os.path.join(staged, f"{leader}.caid")):
                paired[f"{STRUCTURAL_BASELINE}(fused-vs-leader)"] = \
                    paired_bootstrap(task, args.refs, staged,
                                     f"{OURS}-fused", leader,
                                     n_boot=args.n_boot)

        results[task] = {
            "ours": ours, "rank": rank,
            "rank_full_coverage": rank_full,
            "n_full_coverage_entrants": len(full_cov) + 1,
            "n_above_us_that_skipped_targets": len(skipped_above),
            "fused": fused_row,
            "fused_rank": (len(fused_better) + 1) if fused_row else None, "n_methods": len(board) + 1,
            "leader": leader, "leader_auc": leader_auc,
            "top_by_auc": [{k: r[k] for k in ("method", "auc", "aps", "coverage")}
                           for r in board[:5]],
            "paired": paired,
            "inference_seconds": round(wall, 2),
            "seconds_per_target": round(wall / max(len(ref), 1), 4),
            "submission": our_path,
        }

    # Binding-IDR is not a separate task. Its labels are the Binding labels,
    # identical on every evaluated residue, restricted to the residues
    # Disorder-NOX evaluates. CAID scores one submission against both
    # references, and its entrants submit one file — bindEmbed21IDR and ESpritz-D
    # each appear in both tables from a single prediction. Training a separate
    # head against a reconstructed "disordered-and-not-binding" convention
    # produced a predictor for a different question, which is how it landed
    # below chance. Score the binding head where the binding head belongs.
    # CAID2 has no Binding-IDR reference — it is a CAID3 addition — so this is
    # conditioned on the file existing rather than on the round, which keeps it
    # true whatever a future round contains.
    idr_path = os.path.join(args.refs, "binding_idr.fasta")
    if "binding" in predictions_by_task and os.path.isfile(idr_path):
        idr_ref = read_reference(idr_path)
        shared = {t: v for t, v in predictions_by_task["binding"].items()
                  if t in idr_ref}
        if shared:
            cross_path = write_caid_submission(
                os.path.join(subs, f"{OURS}-binding-on-idr.caid"), shared, idr_ref)
            scored = score_method(idr_ref,
                                  {k: np.asarray(v) for k, v in shared.items()})
            if scored:
                results["binding_idr_from_binding_head"] = {
                    "ours": scored,
                    "note": ("the binding head scored on the Binding-IDR "
                             "reference, which is how CAID scores its own "
                             "entrants"),
                    "submission": cross_path,
                }
                own = (results.get("binding_idr") or {}).get("ours")
                if own:
                    print(f"\nbinding_idr: dedicated head {own['auc']:.4f} vs "
                          f"binding head scored on the same reference "
                          f"{scored['auc']:.4f}")

    print(f"\n{'=' * 92}\n OFFICIAL {args.benchmark.upper()} — all comparisons "
          f"paired on shared targets\n{'=' * 92}")
    print(f"{'benchmark':<14}{'our AUC':>9}{'our APS':>9}{'cov':>6}"
          f"{'rank/all':>10}{'rank/full':>12}{'leader':>26}{'lead AUC':>10}")
    for task, r in results.items():
        if "rank" not in r:
            # Derived rows (a head scored on another task's reference) have no
            # rank of their own; they are reported next to the task they inform.
            o = r["ours"]
            print(f"{task:<14}{o['auc']:>9.4f}{o['aps']:>9.4f}"
                  f"{o['coverage']:>6.2f}{'—':>10}{'(derived)':>26}")
            continue
        o = r["ours"]
        rank = f"{r['rank']}/{r['n_methods']}"
        rankf = f"{r['rank_full_coverage']}/{r['n_full_coverage_entrants']}"
        print(f"{task:<14}{o['auc']:>9.4f}{o['aps']:>9.4f}"
              f"{o['coverage']:>6.2f}{rank:>10}{rankf:>12}"
              f"{r['leader']:>26}{r['leader_auc']:>10.3f}")
        if r.get("fused"):
            f = r["fused"]
            frank = f"{r['fused_rank']}/{r['n_methods']}"
            print(f"{'  + AF-rsa':<14}{f['auc']:>9.4f}{f['aps']:>9.4f}"
                  f"{f['coverage']:>6.2f}{frank:>10}"
                  f"{'(equal-weight fusion)':>26}")

    print(f"\n{'-' * 92}\n paired differences (ours minus theirs), protein-"
          f"clustered bootstrap\n{'-' * 92}")
    print(f"{'benchmark':<14}{'opponent':<26}{'delta':>9}{'95% CI':>20}"
          f"{'p':>8}{'targets':>9}")
    for task, r in results.items():
        for opp, pr in (r.get("paired") or {}).items():
            if "error" in pr:
                print(f"{task:<14}{opp:<26}  {pr['error']}")
                continue
            ci = f"[{pr['delta_ci'][0]:+.4f},{pr['delta_ci'][1]:+.4f}]"
            print(f"{task:<14}{opp:<26}{pr['delta_auc']:>+9.4f}{ci:>20}"
                  f"{pr['p_two_sided']:>8.3f}{pr['n_common_targets']:>9}")

    # ── Pre-registered analysis ──────────────────────────────────────────
    primary = {}
    prim = results.get(PRIMARY_TASK) or {}
    for opp in PRIMARY_OPPONENTS:
        pr = (prim.get("paired") or {}).get(opp)
        if pr and "p_two_sided" in pr:
            primary[opp] = pr["p_two_sided"]
    primary_holm = holm_bonferroni(primary) if primary else {}

    secondary = {}
    for task, r in results.items():
        if task == PRIMARY_TASK:
            continue
        for opp, pr in (r.get("paired") or {}).items():
            if "p_two_sided" in pr:
                secondary[f"{task}: {opp}"] = pr["p_two_sided"]
    secondary_holm = holm_bonferroni(secondary) if secondary else {}

    ours_primary = (prim.get("ours") or {})
    auc_primary = ours_primary.get("auc")
    cov_primary = ours_primary.get("coverage")

    # The floors are CAID3 numbers. CAID2 shares four task *names* with CAID3
    # and not one of their values — different targets, different label counts,
    # different scale. Applying a 0.9545 Disorder-PDB floor to a 348-target
    # CAID2 reference would print a confident PASS or FAIL about a comparison
    # that was never registered and does not mean anything.
    floors = {}
    active_floors = (NON_INFERIORITY_FLOORS if args.benchmark == "caid3"
                     else {})
    for task, floor in active_floors.items():
        row = (results.get(task) or {}).get("ours")
        if not row:
            continue
        floors[task] = {
            "auc": row["auc"], "floor": floor,
            "pass": bool(row["auc"] >= floor),
            "margin": round(row["auc"] - floor, 4),
        }
    non_inferior = all(v["pass"] for v in floors.values()) if floors else False

    print(f"\n{'=' * 92}\n PRE-REGISTERED ANALYSIS "
          f"(results/caid3/PREREGISTRATION.md)\n{'=' * 92}")
    if args.benchmark != "caid3":
        print(f" benchmark={args.benchmark}: this is an independent "
              f"replication round.\n No non-inferiority floors are registered "
              f"for it — the CAID3 floors are CAID3 numbers and\n do not "
              f"transfer to a different target set. Placements below are "
              f"descriptive.")
    print(f" primary family: {PRIMARY_TASK}, unfused, "
          f"{len(primary)} test(s), Holm across those alone")
    for opp, v in sorted(primary_holm.items(), key=lambda kv: kv[1]["rank"]):
        pr = prim["paired"][opp]
        print(f"   ours - {opp:<20}{pr['delta_auc']:>+9.4f}  "
              f"p={v['p_raw']:.4f}  adj={v['p_adjusted']:.4f}  "
              f"{'CONFIRMED' if v['significant'] else 'not significant'}")
    if floors:
        print(f"\n non-inferiority — every placement must hold, a single "
              f"breach rejects the variant:")
        for task, v in floors.items():
            print(f"   {task:<16}{v['auc']:>8.4f} vs floor {v['floor']:.4f}"
                  f"  {v['margin']:>+8.4f}  "
                  f"{'PASS' if v['pass'] else 'FAIL'}")
        print(f"   overall: "
              f"{'PASS' if non_inferior else 'FAIL — variant rejected'}")
    if cov_primary is not None and cov_primary < 1.0:
        print(f" WARNING: coverage {cov_primary:.3f} on {PRIMARY_TASK}. A "
              f"subset score is void; the pre-registration requires 319/319.")

    if secondary_holm:
        print(f"\n secondary (exploratory), Holm within a family of "
              f"{len(secondary_holm)}:")
        for name, v in sorted(secondary_holm.items(),
                              key=lambda kv: kv[1]["rank"])[:8]:
            print(f"   {name:<52}p={v['p_raw']:.4f}  adj={v['p_adjusted']:.4f}"
                  f"  {'sig' if v['significant'] else '-'}")

    results["_preregistered"] = {
        "primary_task": PRIMARY_TASK,
        "primary_opponents": list(PRIMARY_OPPONENTS),
        "primary_holm": primary_holm,
        "secondary_holm": secondary_holm,
        "non_inferiority_floors": floors,
        "non_inferiority_pass": bool(non_inferior),
        "primary_auc": auc_primary,
        "primary_coverage": cov_primary,
    }

    out = args.out or os.path.join(args.checkpoint, "caid3_official_results.json")
    with open(out + ".part", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    os.replace(out + ".part", out)
    print(f"\nWrote {out}\nSubmissions in {subs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
