#!/usr/bin/env python3
"""Measure inference cost the way CAID3 does: wall time per target.

CAID3 records timings for every method, and a frozen backbone changes that
arithmetic in two ways worth measuring rather than asserting:

  * no backward pass and no optimizer state, so batch size is bounded by
    activations alone;
  * all five task predictions come from ONE backbone pass, because the tasks
    share a trunk and differ only in a 1x1 read-out. A five-specialist pipeline
    pays the backbone cost five times.

What this measures, and what it does not
----------------------------------------
Reported numbers are end-to-end per target: tokenisation, backbone, head. They
exclude one-off model loading, which is amortised over any real workload, and
they are measured with CUDA synchronisation so the timings are not just the
latency of queueing async kernels — a mistake that makes a GPU look ~10x faster
than it is.

They are NOT comparable to the CAID3 timing table. Those were collected on the
organisers' hardware with each method's own I/O; this runs on one A100 with a
warm cache. A speed claim across different machines is not a measurement, so
the output states the hardware and refuses to phrase itself as a comparison.

Usage:
    python rockfish/benchmark_speed.py --fasta REF.fasta [--checkpoint DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colab.caid3_eval import parse_caid_reference_fasta  # noqa: E402


def sync(device: torch.device) -> None:
    """CUDA kernels are async; without this we would time the queueing."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_batches(fn, batches, device, warmup: int = 2, repeats: int = 3) -> dict:
    """Median-of-repeats wall time, after warmup.

    Warmup matters on GPU: the first pass pays cuDNN autotuning and allocator
    growth, and including it inflates the mean by more than the effect most
    speed claims rest on.
    """
    for b in batches[:warmup]:
        fn(b)
    sync(device)

    per_run = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        for b in batches:
            fn(b)
        sync(device)
        per_run.append(time.perf_counter() - t0)
    return {
        "median_seconds": float(np.median(per_run)),
        "min_seconds": float(np.min(per_run)),
        "runs": [round(x, 4) for x in per_run],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True, help="reference FASTA of targets")
    ap.add_argument("--checkpoint", default=None,
                    help="dir with multitask_head.pt (else an untrained head)")
    ap.add_argument("--backbone", default="650M")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    targets = [p for p in parse_caid_reference_fasta(args.fasta)
               if len(p["sequence"]) <= args.max_len]
    lengths = np.array([len(p["sequence"]) for p in targets])
    print(f"targets: {len(targets)}  residues: {lengths.sum():,}  "
          f"median length {int(np.median(lengths))}")

    from colab.disordernet_gpu import TrainConfig, setup_environment
    cfg = setup_environment(TrainConfig.from_profile("lite", esm_backbone=args.backbone))
    device = cfg.device

    from colab.esm_backbone import load_esm_backbone
    from colab.lite_head import MultiTaskLiteHead, ScalarMix, freeze_backbone

    t_load = time.perf_counter()
    esm, _alpha, batch_converter, spec = load_esm_backbone(
        device, backbone=args.backbone, use_gradient_checkpointing=False)
    freeze_backbone(esm)
    load_seconds = time.perf_counter() - t_load

    tasks = ("disorder_nox", "linker", "binding", "binding_idr")
    layer_ids = list(range(len(esm.layers) - 12, len(esm.layers)))
    if args.checkpoint:
        ck = os.path.join(args.checkpoint, "multitask_head.pt")
        if os.path.isfile(ck):
            payload = torch.load(ck, map_location="cpu", weights_only=False)
            tasks = tuple(payload["tasks"])
            layer_ids = list(payload["layer_ids"])
    mix = ScalarMix(len(layer_ids)).to(device).eval()
    head = MultiTaskLiteHead(in_dim=spec.embed_dim, tasks=tasks).to(device).eval()

    batches = []
    for s in range(0, len(targets), args.batch_size):
        chunk = targets[s:s + args.batch_size]
        _, _, tokens = batch_converter([(p["id"], p["sequence"]) for p in chunk])
        batches.append(tokens.to(device))

    @torch.no_grad()
    def full(tokens):
        out = esm(tokens, repr_layers=layer_ids, return_contacts=False)
        feats = mix([out["representations"][i][:, 1:-1, :] for i in layer_ids])
        return head(feats)

    @torch.no_grad()
    def backbone_only(tokens):
        return esm(tokens, repr_layers=layer_ids, return_contacts=False)

    print("\ntiming (median of "
          f"{args.repeats} passes over all targets, CUDA-synchronised)…")
    all_tasks = time_batches(full, batches, device, repeats=args.repeats)
    backbone = time_batches(backbone_only, batches, device, repeats=args.repeats)

    n = len(targets)
    per_target = all_tasks["median_seconds"] / n
    head_share = all_tasks["median_seconds"] - backbone["median_seconds"]

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor()
    report = {
        "hardware": gpu,
        "backbone": args.backbone,
        "n_targets": n,
        "n_residues": int(lengths.sum()),
        "tasks": list(tasks),
        "batch_size": args.batch_size,
        "model_load_seconds": round(load_seconds, 2),
        "all_tasks": all_tasks,
        "backbone_only": backbone,
        "seconds_per_target": round(per_target, 5),
        "targets_per_second": round(1.0 / per_target, 2),
        "residues_per_second": round(lengths.sum() / all_tasks["median_seconds"], 1),
        "head_fraction_of_runtime": round(
            head_share / all_tasks["median_seconds"], 4),
        "cost_of_five_specialists": {
            "note": (
                "A separate model per task repeats the backbone pass. Estimated "
                "from the measured backbone time, which dominates."
            ),
            "estimated_seconds": round(
                backbone["median_seconds"] * len(tasks) + head_share, 3),
            "speedup_from_sharing": round(
                (backbone["median_seconds"] * len(tasks) + head_share)
                / all_tasks["median_seconds"], 2),
        },
        "excluded_from_timing": "model load, disk I/O, one-off warmup",
        "comparability": (
            "NOT comparable to the CAID3 timing table: those were collected on "
            "the organisers' hardware with each method's own I/O. Reported to "
            "characterise this model, not to rank it against others."
        ),
    }

    print(f"\n{'='*66}\n INFERENCE COST — {gpu}\n{'='*66}")
    print(f"  all {len(tasks)} tasks   : {all_tasks['median_seconds']:.2f} s "
          f"for {n} targets")
    print(f"  per target      : {per_target*1000:.1f} ms")
    print(f"  throughput      : {report['targets_per_second']:.1f} targets/s, "
          f"{report['residues_per_second']:,.0f} residues/s")
    print(f"  head share      : {report['head_fraction_of_runtime']:.1%} of runtime "
          "(the backbone dominates, which is why sharing it pays)")
    print(f"  five specialists: ~{report['cost_of_five_specialists']['estimated_seconds']:.1f} s "
          f"({report['cost_of_five_specialists']['speedup_from_sharing']:.1f}x slower)")
    print(f"  model load      : {load_seconds:.1f} s (excluded above)")
    print(f"\n  {report['comparability']}")

    out = args.out or "speed_benchmark.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
