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
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colab.caid3_official import (  # noqa: E402
    LEADERS,
    TASKS,
    official_leaderboard,
    paired_bootstrap,
    read_reference,
    score_method,
    verify_composition,
)

#: Training-free structural baseline. It ranks 3rd on Disorder-PDB and is not
#: separable from the first-placed method, so beating the leader while failing
#: to beat this would not be a meaningful result.
STRUCTURAL_BASELINE = "AlphaFold-rsa"

OURS = "DisorderNet"


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


def predict_task(head, mix, esm, batch_converter, layer_ids, device, ref,
                 task, structures, structure_dim, batch_size=8):
    """Per-residue probabilities for every target in a reference."""
    items = list(ref.items())
    out, t0 = {}, time.perf_counter()
    with torch.no_grad():
        for s in range(0, len(items), batch_size):
            batch = items[s:s + batch_size]
            _, _, tokens = batch_converter([(tid, seq) for tid, (seq, _) in batch])
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
                for bi, (tid, _sl) in enumerate(batch):
                    f = structures.get(tid) or {}
                    if not f:
                        continue
                    r = np.asarray(f["rsa"], dtype=np.float32)
                    q = np.asarray(f["plddt"], dtype=np.float32)
                    k = min(len(r), len(q), L)
                    sr[bi, :k] = torch.from_numpy(r[:k]).to(device)
                    sp[bi, :k] = torch.from_numpy(q[:k]).to(device)
                    sa[bi, :k] = 1.0
                    ct = np.asarray(f.get("contacts", np.zeros(k)), dtype=np.float32)
                    kk = min(len(ct), k)
                    sc[bi, :kk] = torch.from_numpy(ct[:kk]).to(device)
                kw = {"rsa": sr, "plddt": sp, "structure_available": sa,
                      "contacts": sc}
            probs = torch.sigmoid(head(feats, **kw)[task]).float().cpu().numpy()
            for bi, (tid, (seq, _lab)) in enumerate(batch):
                out[tid] = probs[bi, :len(seq)].astype(np.float64)
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
    args = ap.parse_args(argv)

    for task in TASKS:
        verify_composition(task, os.path.join(args.refs, f"{task}.fasta"))
    print(f"all {len(TASKS)} official references match CAID's composition")

    ckpt = os.path.join(args.checkpoint, "multitask_head.pt")
    if not os.path.isfile(ckpt):
        print(f"ERROR: no multitask_head.pt under {args.checkpoint}", file=sys.stderr)
        return 2
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    tasks = tuple(payload["tasks"])
    layer_ids = list(payload["layer_ids"])
    structure_dim = int(payload.get("structure_dim", 0))

    from colab.disordernet_gpu import TrainConfig, setup_environment
    cfg = setup_environment(TrainConfig.from_profile("lite", esm_backbone=args.backbone))
    device = cfg.device

    from colab.esm_backbone import load_esm_backbone
    from colab.lite_head import MultiTaskLiteHead, ScalarMix, freeze_backbone
    esm, _a, batch_converter, spec = load_esm_backbone(
        device, backbone=args.backbone, use_gradient_checkpointing=False)
    freeze_backbone(esm)
    mix = ScalarMix(len(layer_ids)).to(device)
    mix.load_state_dict(payload["mix"])
    head = MultiTaskLiteHead(in_dim=spec.embed_dim, tasks=tasks,
                             structure_dim=structure_dim).to(device)
    head.load_state_dict(payload["head"])
    head.eval()
    mix.eval()
    print(f"checkpoint: tasks={list(tasks)} structure_dim={structure_dim} "
          f"trainable={sum(p.numel() for p in head.parameters()):,}")

    subs = args.submissions or os.path.join(args.checkpoint, "caid_submissions")
    os.makedirs(subs, exist_ok=True)

    results = {}
    predictions_by_task: dict[str, dict] = {}
    for task in TASKS:
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
        predictions_by_task[task] = preds
        ours = score_method(ref, {k: np.asarray(v) for k, v in preds.items()})

        board = official_leaderboard(task, args.refs, args.predictions)
        better = [r for r in board if r["auc"] > ours["auc"]]
        rank = len(better) + 1

        # Paired tests need our submission alongside theirs.
        import shutil
        staged = os.path.join(subs, "_paired")
        os.makedirs(staged, exist_ok=True)
        for fn in os.listdir(args.predictions):
            dst = os.path.join(staged, fn)
            if not os.path.exists(dst):
                os.symlink(os.path.join(args.predictions, fn), dst)
        shutil.copy(our_path, os.path.join(staged, f"{OURS}.caid"))
        our_name = OURS

        leader = LEADERS[task][0]
        paired = {}
        for opponent in dict.fromkeys([leader, STRUCTURAL_BASELINE,
                                       board[0]["method"]]):
            if not os.path.exists(os.path.join(staged, f"{opponent}.caid")):
                continue
            paired[opponent] = paired_bootstrap(
                task, args.refs, staged, OURS, opponent, n_boot=args.n_boot)

        results[task] = {
            "ours": ours, "rank": rank, "n_methods": len(board) + 1,
            "leader": leader, "leader_auc": LEADERS[task][1],
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
    if "binding" in predictions_by_task:
        idr_ref = read_reference(os.path.join(args.refs, "binding_idr.fasta"))
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

    print(f"\n{'=' * 92}\n OFFICIAL CAID3 — all comparisons paired on shared "
          f"targets\n{'=' * 92}")
    print(f"{'benchmark':<14}{'our AUC':>9}{'our APS':>9}{'cov':>6}"
          f"{'rank':>10}{'leader':>26}{'lead AUC':>10}")
    for task, r in results.items():
        o = r["ours"]
        rank = f"{r['rank']}/{r['n_methods']}"
        print(f"{task:<14}{o['auc']:>9.4f}{o['aps']:>9.4f}"
              f"{o['coverage']:>6.2f}{rank:>10}"
              f"{r['leader']:>26}{r['leader_auc']:>10.3f}")

    print(f"\n{'-' * 92}\n paired differences (ours minus theirs), protein-"
          f"clustered bootstrap\n{'-' * 92}")
    print(f"{'benchmark':<14}{'opponent':<26}{'delta':>9}{'95% CI':>20}"
          f"{'p':>8}{'targets':>9}")
    for task, r in results.items():
        for opp, pr in r["paired"].items():
            if "error" in pr:
                print(f"{task:<14}{opp:<26}  {pr['error']}")
                continue
            ci = f"[{pr['delta_ci'][0]:+.4f},{pr['delta_ci'][1]:+.4f}]"
            print(f"{task:<14}{opp:<26}{pr['delta_auc']:>+9.4f}{ci:>20}"
                  f"{pr['p_two_sided']:>8.3f}{pr['n_common_targets']:>9}")

    out = args.out or os.path.join(args.checkpoint, "caid3_official_results.json")
    with open(out + ".part", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    os.replace(out + ".part", out)
    print(f"\nWrote {out}\nSubmissions in {subs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
