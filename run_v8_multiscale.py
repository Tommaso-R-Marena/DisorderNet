"""DisorderNet v8: multi-scale protein-LM ensemble.

Averaging the out-of-fold predictions of the v7 model run on several ESM-2
backbones (35M / 150M / 650M) beats any single backbone, because different PLM
scales capture complementary disorder signal. Weights are fixed (equal) and every
input is an out-of-fold prediction, so the ensemble is leakage-free.

Leakage-free 5-fold CV (identical proteins/folds), pooled AUC:
    35M                    0.8479
    150M                   0.8498
    650M                   0.8505
    35M + 150M + 650M      0.8568   <-- this script

It reuses the per-backbone OOF arrays saved by run_v7.py (y_true.npy / y_pred.npy)
and adds cross-fitted isotonic calibration + conformal confidence on the ensemble.

Prereq: run run_v7.py once per backbone, e.g.
    python run_v7.py                                            # 35M (repo-local ./data)
    DISORDERNET_HOME=.../dn150m python run_v7.py                # 150M
    DISORDERNET_HOME=.../dn650m python run_v7.py                # 650M

Run:  python run_v8_multiscale.py --backbones <dir1> <dir2> ...
"""
import argparse, json, os
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, roc_curve, balanced_accuracy_score)

from disordernet_paths import DISPROT_JSON, EMB_DIR, results_dir
from confidence import (expected_calibration_error, fit_calibrator, apply_calibrator,
                        conformal_quantile, conformal_sets)

SEED = 42
MAX_PROT = 1500
MAX_LEN = 800

DEFAULT_BACKBONES = [
    "/home/user/workspace/disorder_model/results_v7",  # 35M
    "/home/user/workspace/dn150m/results_v7",          # 150M
    "/home/user/workspace/dn650m/results_v7",          # 650M
]


def load_proteins():
    with open(DISPROT_JSON) as f:
        data = json.load(f)
    proteins = [
        p for p in data
        if os.path.exists(os.path.join(str(EMB_DIR), f"{p['disprot_id']}.npy"))
        and 30 <= p["length"] <= MAX_LEN and sum(p["disorder_labels"]) >= 3
        and p["length"] - sum(p["disorder_labels"]) >= 3
    ]
    rng = np.random.RandomState(SEED)
    if len(proteins) > MAX_PROT:
        idx = rng.choice(len(proteins), MAX_PROT, replace=False)
        proteins = [proteins[i] for i in sorted(idx)]
    return proteins


def residue_fold_labels(proteins):
    """Per-residue fold id, matching run_v7's ids-order concatenation."""
    n = len(proteins)
    gkf = GroupKFold(n_splits=5)
    fold_of_protein = np.empty(n, dtype=np.int64)
    for fold, (_, va) in enumerate(gkf.split(range(n), range(n), range(n))):
        for i in va:
            fold_of_protein[i] = fold
    return np.concatenate([
        np.full(proteins[i]["length"], fold_of_protein[i], dtype=np.int64)
        for i in range(n)
    ])


def evaluate(yt, yp):
    fpr, tpr, th = roc_curve(yt, yp)
    opt = th[np.argmax(tpr - fpr)]
    yb = (yp >= opt).astype(int)
    return {
        "auc_roc": float(roc_auc_score(yt, yp)),
        "avg_precision": float(average_precision_score(yt, yp)),
        "f1": float(f1_score(yt, yb)),
        "mcc": float(matthews_corrcoef(yt, yb)),
        "balanced_acc": float(balanced_accuracy_score(yt, yb)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES,
                    help="results_v7 dirs (each with y_true.npy / y_pred.npy)")
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    y_true = None
    preds = []
    names = []
    for d in args.backbones:
        yt = np.load(os.path.join(d, "y_true.npy"))
        yp = np.load(os.path.join(d, "y_pred.npy"))
        if y_true is None:
            y_true = yt
        elif not np.array_equal(y_true, yt):
            raise SystemExit(f"y_true mismatch in {d}: backbones must share proteins/folds/seed")
        preds.append(yp)
        names.append(os.path.basename(os.path.dirname(d)) or d)
        print(f"  {d}: AUC={roc_auc_score(yt, yp):.4f}", flush=True)

    ensemble = np.mean(preds, axis=0)  # equal weights -> leakage-free
    m = evaluate(y_true, ensemble)

    # Cross-fitted calibration + conformal on the ensemble. Prefer the per-residue
    # fold ids saved by run_v7 (correct for any split method); else reconstruct the
    # default protein GroupKFold.
    fold_path = os.path.join(args.backbones[0], "fold_ids.npy")
    if os.path.exists(fold_path):
        fold = np.load(fold_path)
    else:
        fold = residue_fold_labels(load_proteins())
    if len(fold) != len(y_true):
        raise SystemExit(f"residue count mismatch ({len(fold)} vs {len(y_true)}) — "
                         "backbone OOF must come from the same run_v7 protein selection")
    calibrated = np.empty_like(ensemble)
    decision = np.empty(len(ensemble), dtype=np.int64)
    for k in range(5):
        tr = fold != k
        iso = fit_calibrator(ensemble[tr], y_true[tr])
        q = conformal_quantile(apply_calibrator(iso, ensemble[tr]), y_true[tr],
                               alpha=args.alpha, class_conditional=True)
        te = fold == k
        cp = apply_calibrator(iso, ensemble[te])
        calibrated[te] = cp
        decision[te] = conformal_sets(cp, q)["decision"]

    ece_before = expected_calibration_error(y_true, ensemble)
    ece_after = expected_calibration_error(y_true, calibrated)
    confident = np.isin(decision, (0, 1))
    covered = np.where(y_true == 1, np.isin(decision, (1,)) | (decision == -1),
                       np.isin(decision, (0,)) | (decision == -1))

    print("\n" + "=" * 66)
    print("DisorderNet v8 — multi-scale PLM ensemble")
    print("=" * 66)
    print(f"  backbones: {', '.join(names)}  (equal weights)")
    for k in ["auc_roc", "avg_precision", "f1", "mcc", "balanced_acc"]:
        print(f"  {k:16s}: {m[k]:.4f}")
    print(f"\n  best single backbone AUC: {max(roc_auc_score(y_true, p) for p in preds):.4f}")
    print(f"  ensemble AUC            : {m['auc_roc']:.4f}")
    print(f"  calibration ECE        : {ece_before:.4f} -> {ece_after:.4f}")
    print(f"  conformal @ alpha={args.alpha}: coverage={covered.mean():.3f} "
          f"confident={confident.mean():.3f}")
    if confident.any():
        print(f"  selective accuracy      : {(decision[confident] == y_true[confident]).mean():.3f}")

    out = results_dir("results_v8", create=True)
    json.dump({
        "backbones": names,
        "single_backbone_aucs": [float(roc_auc_score(y_true, p)) for p in preds],
        "ensemble": m,
        "calibration": {"ece_before": ece_before, "ece_after": ece_after},
        "conformal": {"alpha": args.alpha, "coverage": float(covered.mean()),
                      "confident_rate": float(confident.mean())},
    }, open(os.path.join(str(out), "metrics.json"), "w"), indent=2)
    np.save(os.path.join(str(out), "y_true.npy"), y_true)
    np.save(os.path.join(str(out), "y_pred.npy"), ensemble)
    np.save(os.path.join(str(out), "y_pred_calibrated.npy"), calibrated)
    print(f"\n  saved -> {out}/")


if __name__ == "__main__":
    main()
