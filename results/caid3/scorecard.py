#!/usr/bin/env python3
"""The definitive table: where DisorderNet places, against whom, and whether it counts.

Assembled from the result JSONs rather than transcribed, so the numbers cannot
drift from what the jobs produced.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
RUNS = ["multitask_pbias", "multitask_windowed", "multitask_publication",
        "multitask_chiral", "multitask_control", "multitask_motif",
        "multitask_wass", "multitask_rank", "multitask_private"]
TASKS = ("disorder_pdb", "disorder_nox", "binding", "binding_idr", "linker")
NICE = {"disorder_pdb": "Disorder-PDB", "disorder_nox": "Disorder-NOX",
        "binding": "Binding", "binding_idr": "Binding-IDR", "linker": "Linker"}


def load(run, name):
    p = os.path.join(ROOT, run, name)
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def best_per_task(fname):
    """Best of our checkpoints per task, with its run and paired stats."""
    out = {}
    for run in RUNS:
        j = load(run, fname)
        if not j:
            continue
        for task in TASKS:
            r = j.get(task)
            if not isinstance(r, dict) or not r.get("ours"):
                continue
            auc = r["ours"]["auc"]
            if task not in out or auc > out[task]["auc"]:
                out[task] = {
                    "run": run.replace("multitask_", ""), "auc": auc,
                    "cov": r["ours"]["coverage"], "rank": r.get("rank"),
                    "rank_full": r.get("rank_full_coverage"),
                    "n_methods": r.get("n_methods"),
                    "n_full": r.get("n_full_coverage_entrants"),
                    "leader": r.get("leader"), "leader_auc": r.get("leader_auc"),
                    "paired": r.get("paired") or {},
                }
    return out


def main() -> int:
    print("=" * 108)
    print(" CAID3 — official references, coverage 1.00, paired protein-clustered "
          "bootstrap (10,000 resamples)")
    print("=" * 108)
    print(f" {'benchmark':<14}{'leader':<26}{'theirs':>8}{'ours':>8}"
          f"{'rank':>9}{'full-cov':>10}{'delta':>9}{'p':>8}")
    c3 = best_per_task("caid3_official_results.json")
    for t in TASKS:
        r = c3.get(t)
        if not r:
            continue
        pr = r["paired"].get(r["leader"]) or {}
        d = pr.get("delta_auc")
        p = pr.get("p_two_sided")
        print(f" {NICE[t]:<14}{str(r['leader'])[:25]:<26}"
              f"{(r['leader_auc'] or 0):>8.4f}{r['auc']:>8.4f}"
              f"{'%s/%s' % (r['rank'], r['n_methods']):>9}"
              f"{'%s/%s' % (r['rank_full'], r['n_full']):>10}"
              f"{(f'{d:+.4f}' if d is not None else '—'):>9}"
              f"{(f'{p:.3f}' if p is not None else '—'):>8}")

    print()
    print("=" * 108)
    print(" CAID2 — independent round, never seen in training, coverage 1.00")
    print("=" * 108)
    print(f" {'benchmark':<14}{'leader':<26}{'theirs':>8}{'ours':>8}"
          f"{'rank':>9}{'full-cov':>10}{'delta':>9}{'p':>8}")
    c2 = best_per_task("caid2_official_results.json")
    for t in TASKS:
        r = c2.get(t)
        if not r:
            continue
        pr = r["paired"].get(r["leader"]) or {}
        d, p = pr.get("delta_auc"), pr.get("p_two_sided")
        print(f" {NICE[t]:<14}{str(r['leader'])[:25]:<26}"
              f"{(r['leader_auc'] or 0):>8.4f}{r['auc']:>8.4f}"
              f"{'%s/%s' % (r['rank'], r['n_methods']):>9}"
              f"{'%s/%s' % (r['rank_full'], r['n_full']):>10}"
              f"{(f'{d:+.4f}' if d is not None else '—'):>9}"
              f"{(f'{p:.3f}' if p is not None else '—'):>8}")

    wt = os.path.join(ROOT, "within_protein_test_all.json")
    if os.path.isfile(wt):
        print()
        print("=" * 108)
        print(" WITHIN-PROTEIN — per-target paired comparison, the "
              "calibration-invariant axis")
        print("=" * 108)
        print(f" {'comparison':<48}{'targets':>9}{'wins':>10}"
              f"{'delta':>9}{'Wilcoxon p':>12}")
        j = json.load(open(wt))
        for k, v in j.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            if "PUNCH2" not in k:
                continue
            print(f" {k:<48}{v['n_targets']:>9}"
                  f"{'%d (%.0f%%)' % (v['wins'], 100*v['wins']/v['n_targets']):>11}"
                  f"{v['mean']:>+9.4f}{v['p_wilcoxon']:>12.5f}")

    cap = os.path.join(ROOT, "benchmark_capacity.json")
    rel = os.path.join(ROOT, "relative_noise.json")
    if os.path.isfile(cap):
        j = json.load(open(cap))
        print()
        print("=" * 108)
        print(f" CAPACITY — at the measured annotation error rate "
              f"eps = {j['epsilon']:.4f}")
        print("=" * 108)
        for rnd, rows in j["rounds"].items():
            for t, r in rows.items():
                print(f" {rnd} {NICE.get(t, t):<16}"
                      f"capacity {r['capacity_targets']:>3} of "
                      f"{r['n_entrants'] or '—'} entrants")
    if os.path.isfile(rel):
        j = json.load(open(rel))
        print(f"\n label-flip noise    {j['eps_label_pooled']:.4f} -> capacity "
              f"{j['capacity_label']}")
        print(f" pairwise noise      {j['eps_pairwise_pooled']:.4f} -> capacity "
              f"{j['capacity_pairwise']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
