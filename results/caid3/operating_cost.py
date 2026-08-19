#!/usr/bin/env python3
"""What each CAID3 entrant costs at a guaranteed miss rate.

CAID ranks 115 methods by AUC. AUC is threshold-free, which is its virtue and
its evasion: it never says what a user should do with a score, and two methods
an AUC apart of 0.07 can be entirely different instruments in practice.

This asks the operational question instead, for every entrant, with a
distribution-free finite-sample guarantee and no assumption that anyone's
scores are calibrated:

    Fix a tolerated miss rate. How much of a protein must this method flag as
    disordered to achieve it?

The guarantee is conformal risk control (Angelopoulos, Bates, Fisch, Lei,
Schuster 2023). For a bounded loss monotone in a threshold,

    lambda = inf { t : (n/(n+1)) * mean_i L_i(t) + 1/(n+1) <= alpha }

gives `E[L(lambda)] <= alpha` over a fresh **protein**. The loss is the fraction
of a protein's disordered residues the call misses. Proteins are the
exchangeable unit — residues within a chain share a fold, a construct and an
experiment — so the split is by protein and the promise is made at that level.
A per-residue promise from a protein-level split would be false, and is measured
rather than claimed (see GUARANTEES.md, where the shortfall is 0.876 against a
0.90 target).

Scores are rank-transformed within the pooled set before calibration. Conformal
needs a consistent score, not a probability, so this makes 115 methods with
wildly different output conventions comparable without asserting that any of
them is calibrated — which `Calibration.risk_decomposition` says would not be
enough even if it were true.

**Validity is free and identical for everyone** (`PredictionSets.validity_is_free`):
every method achieves the guarantee, because the threshold is chosen to make it
so. The entire content of the ranking below is the *price*.

    export ANALYSIS_SCRIPT=results/caid3/operating_cost.py
    sbatch rockfish/slurm/analysis_cpu.sbatch
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.conformal import (  # noqa: E402
    control_chain_risk,
    control_chain_risk_quantile,
    evaluate_chain_risk,
    evaluate_chain_risk_quantile,
    split_by_chain,
)
from colab.caid3_official import (  # noqa: E402
    TASKS,
    evaluated_mask,
    read_caid_predictions,
    read_reference,
    verify_composition_for,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("COST_REFS", f"{ROOT}/caid3_official")
PREDS = os.environ.get("COST_PREDS", f"{ROOT}/caid3_predictions")
EXTRA = os.environ.get("COST_EXTRA", "")
OUT = os.environ.get("COST_OUT", "")
ALPHAS = tuple(float(a) for a in
               os.environ.get("COST_ALPHAS", "0.20,0.10,0.05").split(","))
N_SPLITS = int(os.environ.get("COST_SPLITS", "20"))


def per_target(ref, pred, targets):
    """(labels, scores) per target over exactly ``targets``, or None."""
    ys, ss = [], []
    for tid in targets:
        _seq, lab = ref[tid]
        p = pred.get(tid)
        if p is None or len(p) != len(lab):
            return None
        m = evaluated_mask(lab)
        y = np.frombuffer(lab.encode(), np.uint8)[m].astype(np.int8) - ord("0")
        s = np.asarray(p)[m]
        if not np.isfinite(s).all():
            return None
        ys.append(y)
        ss.append(s.astype(np.float64))
    return ys, ss


def two_class_targets(ref):
    keep = []
    for tid, (_seq, lab) in ref.items():
        m = evaluated_mask(lab)
        if not m.any():
            continue
        y = np.frombuffer(lab.encode(), np.uint8)[m].astype(np.int8) - ord("0")
        if 0 < int(y.sum()) < len(y):
            keep.append(tid)
    return keep


def rank_transform(ss):
    """Pooled rank to [0, 1], preserving each method's own ordering.

    Within-method only. Ranks are not comparable across methods as values, but
    the conformal threshold is chosen per method, so only the ordering matters.
    """
    flat = np.concatenate(ss)
    r = (np.argsort(np.argsort(flat)) + 0.5) / len(flat)
    out, off = [], 0
    for y in ss:
        out.append(r[off:off + len(y)])
        off += len(y)
    return out


def price(ys, ps, alpha, n_splits, seed):
    """Median flagged fraction over repeated protein splits.

    One split of 233 proteins into 116/117 is a noisy estimate of both the
    threshold and its cost, and the split is arbitrary. Repeating it and taking
    the median makes the number a property of the method rather than of one
    draw; the spread is reported alongside.
    """
    rng = np.random.default_rng(seed)
    flagged, realised, quantile = [], [], []
    idx = np.arange(len(ys))
    for _ in range(n_splits):
        cal_m, _ = split_by_chain(idx, rng, 0.5)
        cal = [i for i in idx if cal_m[i]]
        test = [i for i in idx if not cal_m[i]]
        got = control_chain_risk([ps[i] for i in cal], [ys[i] for i in cal],
                                 alpha)
        if not got.get("achievable", True):
            return {"unachievable": got.get("reason"),
                    "min_achievable_alpha": got.get("min_achievable_alpha")}
        ev = evaluate_chain_risk([ps[i] for i in test], [ys[i] for i in test],
                                 got["threshold"])
        if ev["mean_fraction_called_disordered"] is None:
            continue
        flagged.append(ev["mean_fraction_called_disordered"])
        realised.append(ev["mean_chain_miss_rate"])
        # The calibration-invariant twin: flag a fixed quantile of each chain's
        # own scores. Depends only on the within-chain ordering, so no
        # per-protein recalibration can change it — the operational counterpart
        # of auc_within_strictMono_invariant. The gap between the two costs is
        # what calibration is worth operationally.
        gq = control_chain_risk_quantile([ps[i] for i in cal],
                                         [ys[i] for i in cal], alpha)
        eq = evaluate_chain_risk_quantile([ps[i] for i in test],
                                          [ys[i] for i in test], gq["q"])
        quantile.append(eq["mean_fraction_called_disordered"])
    if not flagged:
        return None
    return {
        "flagged_median": float(np.median(flagged)),
        "flagged_iqr": [float(np.percentile(flagged, 25)),
                        float(np.percentile(flagged, 75))],
        "realised_risk_median": float(np.median(realised)),
        "flagged_quantile_median": (float(np.median(quantile))
                                    if quantile else None),
        "calibration_credit": (float(np.median(quantile))
                               - float(np.median(flagged))
                               if quantile else None),
        "n_splits": len(flagged),
    }


def main() -> int:
    report = {"alphas": list(ALPHAS), "n_splits": N_SPLITS, "tasks": {}}
    for task in TASKS:
        path = os.path.join(REFS, f"{task}.fasta")
        if not os.path.isfile(path):
            continue
        verify_composition_for("caid3", task, path)
        ref = read_reference(path)
        targets = two_class_targets(ref)

        methods = {}
        for fn in sorted(os.listdir(PREDS)):
            if not fn.endswith(".caid"):
                continue
            got = per_target(ref, read_caid_predictions(
                os.path.join(PREDS, fn)), targets)
            if got:
                methods[fn[:-5]] = got
        for item in filter(None, (s.strip() for s in EXTRA.split(","))):
            parts = item.split(":")
            fp = parts[1] if len(parts) > 1 and parts[1] else "DisorderNet"
            label = parts[2] if len(parts) > 2 and parts[2] else fp
            p = os.path.join(parts[0], f"{fp}-{task}.caid")
            if os.path.isfile(p):
                got = per_target(ref, read_caid_predictions(p), targets)
                if got:
                    methods[label] = got
        if len(methods) < 5:
            continue

        print(f"\n{'=' * 96}")
        print(f" {task}: {len(targets)} two-class targets, "
              f"{len(methods)} full-coverage methods")
        print(f" fraction of each protein that must be flagged to guarantee "
              f"the stated miss rate")
        print("=" * 96)

        rows = {}
        for name, (ys, ss) in methods.items():
            ps = rank_transform(ss)
            row = {}
            for a in ALPHAS:
                got = price(ys, ps, a, N_SPLITS, seed=20260818)
                if got:
                    row[f"alpha_{a}"] = got
            if row:
                rows[name] = row

        key = f"alpha_{ALPHAS[1] if len(ALPHAS) > 1 else ALPHAS[0]}"
        # Rank on the tightest level that is actually achievable at this
        # sample size, not on one the finite-sample correction forbids.
        for k in [f"alpha_{a}" for a in sorted(ALPHAS)]:
            if any("flagged_median" in (rows[n].get(k) or {}) for n in rows):
                key = k
                break
        unach = {a for a in ALPHAS
                 if all((rows[n].get(f"alpha_{a}") or {}).get("unachievable")
                        for n in rows)}
        for a in sorted(unach):
            floor = next((rows[n][f"alpha_{a}"].get("min_achievable_alpha")
                          for n in rows
                          if (rows[n].get(f"alpha_{a}") or {}).get(
                              "min_achievable_alpha")), None)
            print(f" risk <= {a:.2f} is unachievable here: with this many "
                  f"calibration proteins no method can certify a risk below "
                  f"{floor:.4f}" if floor else
                  f" risk <= {a:.2f} is unachievable at this sample size")
        order = sorted((n for n in rows if "flagged_median" in
                        (rows[n].get(key) or {})),
                       key=lambda n: rows[n][key]["flagged_median"])
        if not order:
            print(" no achievable risk level at this sample size")
            continue
        head = "".join(f"{'risk<=' + format(a, '.2f'):>14}" for a in ALPHAS)
        print(f" {'#':>3} {'method':<28}{head}"
              f"{'per-prot':>10}{'credit':>9}")
        mid = f"alpha_{ALPHAS[1] if len(ALPHAS) > 1 else ALPHAS[0]}"
        for i, n in enumerate(order[:20], 1):
            cells = ""
            for a in ALPHAS:
                v = rows[n].get(f"alpha_{a}")
                if v and "flagged_median" in v:
                    cells += f"{v['flagged_median']:>13.1%} "
                elif v and v.get("unachievable"):
                    cells += f"{'n too small':>13} "
                else:
                    cells += f"{'—':>14}"
            m = rows[n].get(mid) or {}
            pq = m.get("flagged_quantile_median")
            cr = m.get("calibration_credit")
            cells += (f"{pq:>9.1%}" if pq is not None else f"{'—':>10}")
            cells += (f"{cr:>+9.1%}" if cr is not None else f"{'—':>9}")
            print(f" {i:>3} {n:<28}{cells}")
        print("\n per-prot: the same guarantee bought with a per-protein "
              "quantile instead of a global\n threshold — calibration-"
              "invariant, so it prices discrimination alone. credit is what\n "
              "protein-level calibration saves: large means the method's "
              "advantage is calibration.")
        if len(order) > 20:
            print(f"     … {len(order) - 20} more; worst: "
                  f"{order[-1]} at {rows[order[-1]][key]['flagged_median']:.1%}")

        report["tasks"][task] = {"n_targets": len(targets),
                                 "n_methods": len(rows), "methods": rows,
                                 "order_at_" + key: order}

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
