#!/usr/bin/env python3
"""Score a multi-task model on the reconstructed CAID3 benchmarks.

Only Linker reconstructs exactly (31 targets, 1,379 positives, matching the
published composition to the residue), so only Linker is reported as a
head-to-head comparison against its published leader. The rest are scored and
labelled approximate, because a benchmark whose composition differs is a
different benchmark and its number cannot sit beside a published one however
close it looks.

Every score carries a protein-clustered bootstrap interval. The published
leaders have no intervals, so "our CI contains their point estimate" is the
strongest honest statement available — never "we beat them".

Usage:
    python rockfish/eval_caid3_tasks.py --checkpoint DIR --refs DIR --disprot FILE
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

from colab.caid3_eval import parse_caid_reference_fasta  # noqa: E402
from colab.caid3_references import PUBLISHED  # noqa: E402


def bootstrap_ci(labels_by_target, probs_by_target, n_boot=1000, seed=0):
    """Protein-clustered bootstrap: residues within a target are correlated."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.concatenate(labels_by_target)
    s = np.concatenate(probs_by_target)
    if len(np.unique(y)) < 2:
        return None
    point_auc = roc_auc_score(y, s)
    point_aps = average_precision_score(y, s)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels_by_target))
    aucs, apss = [], []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        yy = np.concatenate([labels_by_target[i] for i in pick])
        ss = np.concatenate([probs_by_target[i] for i in pick])
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, ss))
        apss.append(average_precision_score(yy, ss))
    return {
        "auc": float(point_auc), "aps": float(point_aps),
        "auc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "aps_ci": [float(np.percentile(apss, 2.5)), float(np.percentile(apss, 97.5))],
        "n_targets": len(labels_by_target), "n_residues": int(len(y)),
        "prevalence": float(y.mean()),
    }


def verdict(task: str, scored: dict, head_to_head: bool) -> dict:
    """State plainly where a score sits against the published leader."""
    pub = PUBLISHED.get(task, {})
    lead_auc, lead_aps = pub.get("auc"), pub.get("aps")
    out = {
        "leader": pub.get("leader"), "leader_auc": lead_auc, "leader_aps": lead_aps,
        "head_to_head": head_to_head,
    }
    if lead_auc is None or scored is None:
        return out
    lo, hi = scored["auc_ci"]
    out.update({
        "delta_auc": round(scored["auc"] - lead_auc, 4),
        "delta_aps": round(scored["aps"] - lead_aps, 4)
        if lead_aps is not None else None,
        "ci_contains_leader": bool(lo <= lead_auc <= hi),
        "ci_entirely_above_leader": bool(lo > lead_auc),
    })
    if not head_to_head:
        out["caveat"] = (
            "Composition differs from the published benchmark, so this delta is "
            "NOT a comparison with the leader. Reported for direction only."
        )
    elif out["ci_entirely_above_leader"]:
        out["caveat"] = (
            "Same composition and the whole interval sits above the published "
            "point estimate. The leader has no published interval, so this is "
            "evidence, not a significance test against them."
        )
    elif out["ci_contains_leader"]:
        out["caveat"] = "Same composition; consistent with the leader, not above it."
    else:
        out["caveat"] = "Same composition; below the leader."
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="dir with multitask_head.pt")
    ap.add_argument("--refs", required=True, help="dir of reconstructed caid3_*.fasta")
    ap.add_argument("--out", default=None)
    ap.add_argument("--backbone", default="650M")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    ckpt_path = os.path.join(args.checkpoint, "multitask_head.pt")
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: no multitask_head.pt under {args.checkpoint}", file=sys.stderr)
        return 2
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tasks = tuple(payload["tasks"])
    layer_ids = list(payload["layer_ids"])

    from colab.disordernet_gpu import TrainConfig, setup_environment
    cfg = setup_environment(TrainConfig.from_profile("lite", esm_backbone=args.backbone))
    device = cfg.device

    from colab.esm_backbone import load_esm_backbone
    from colab.lite_head import MultiTaskLiteHead, ScalarMix, freeze_backbone
    esm, _alpha, batch_converter, spec = load_esm_backbone(
        device, backbone=args.backbone, use_gradient_checkpointing=False)
    freeze_backbone(esm)
    mix = ScalarMix(len(layer_ids)).to(device)
    mix.load_state_dict(payload["mix"])
    head = MultiTaskLiteHead(in_dim=spec.embed_dim, tasks=tasks).to(device)
    head.load_state_dict(payload["head"])
    head.eval(); mix.eval()
    print(f"loaded multi-task head: tasks={list(tasks)}  "
          f"trainable={sum(p.numel() for p in head.parameters()):,}")

    validation = {}
    vpath = os.path.join(args.refs, "reference_validation.json")
    if os.path.isfile(vpath):
        validation = json.load(open(vpath))

    results = {}
    for task in tasks:
        ref_path = os.path.join(args.refs, f"caid3_{task}.fasta")
        if not os.path.isfile(ref_path):
            continue
        ref = parse_caid_reference_fasta(ref_path)
        labs, prbs = [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for s in range(0, len(ref), args.batch_size):
                batch = ref[s:s + args.batch_size]
                _, _, tokens = batch_converter([(p["id"], p["sequence"]) for p in batch])
                tokens = tokens.to(device)
                out = esm(tokens, repr_layers=layer_ids, return_contacts=False)
                feats = mix([out["representations"][i][:, 1:-1, :] for i in layer_ids])
                logits = head(feats)[task]
                probs = torch.sigmoid(logits).float().cpu().numpy()
                for bi, p in enumerate(batch):
                    lab = np.asarray(p["labels"], dtype=np.int8)
                    m = np.asarray(p["eval_mask"], dtype=bool)
                    n = min(len(lab), len(m), probs.shape[1])
                    sel = m[:n]
                    if sel.sum() == 0:
                        continue
                    labs.append(lab[:n][sel])
                    prbs.append(probs[bi, :n][sel])
        wall = time.perf_counter() - t0
        if not labs:
            continue
        scored = bootstrap_ci(labs, prbs, n_boot=args.n_boot)
        h2h = bool((validation.get(task) or {}).get("head_to_head"))
        results[task] = {
            "scored": scored,
            "verdict": verdict(task, scored, h2h),
            "inference_seconds": round(wall, 2),
            "seconds_per_target": round(wall / max(len(ref), 1), 4),
        }

    print(f"\n{'='*78}\n CAID3 TASK EVALUATION\n{'='*78}")
    print(f"{'benchmark':<15}{'AUC':>8}{'95% CI':>18}{'APS':>8}"
          f"{'leader':>9}{'Δ':>8}  status")
    for task, r in results.items():
        s, v = r["scored"], r["verdict"]
        if s is None:
            continue
        ci = f"[{s['auc_ci'][0]:.3f},{s['auc_ci'][1]:.3f}]"
        tag = "head-to-head" if v["head_to_head"] else "approximate"
        print(f"{task:<15}{s['auc']:>8.4f}{ci:>18}{s['aps']:>8.4f}"
              f"{v['leader_auc']:>9.3f}{v['delta_auc']:>+8.4f}  {tag}")
    for task, r in results.items():
        v = r["verdict"]
        if v.get("caveat"):
            print(f"\n  {task}: {v['caveat']}")

    out = args.out or os.path.join(args.checkpoint, "caid3_task_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
