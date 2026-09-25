"""
Learned meta-ensemble for OOF predictions (replaces coarse grid search).

Fits a regularized logistic stacker on pooled OOF residues.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from colab.biological_utility import align_fold_predictions
from colab.cv_splits import resolve_cv_splits
from colab.inference_fusion import compute_pooled_metrics, write_fused_probs_to_fold_results


def _assign_protein_folds(
    proteins: list,
    fold_results: list,
    used_ids: list[str],
    n_folds: int,
) -> np.ndarray:
    """Fold index per entry of ``used_ids``, reusing the run's own CV partition.

    Keeping the meta-learner's folds identical to the base CV folds means a
    residue is never scored by a stacker fitted on predictions from a model that
    trained on that same protein.
    """
    fold_of_id: dict[str, int] = {}
    for fold_idx, (_, val_idx) in enumerate(
        resolve_cv_splits(proteins, n_folds, fold_results=fold_results)
    ):
        for i in val_idx:
            fold_of_id[proteins[i]["id"]] = fold_idx
    # Any id missing from the partition round-robins into a fold so it is still
    # scored out-of-sample rather than silently dropped.
    return np.array(
        [fold_of_id.get(pid, idx % max(n_folds, 1)) for idx, pid in enumerate(used_ids)],
        dtype=np.int64,
    )


def _build_stacker_matrix(
    aligned: list[dict],
    streams: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Stack aligned prediction streams into (N, n_streams) feature matrix.

    Also returns the usable protein ids and a per-row protein index, so the
    meta-learner can be fitted out-of-fold rather than on its own scores.
    """
    names = list(streams.keys())
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    used_ids: list[str] = []
    row_protein: list[np.ndarray] = []

    for item in aligned:
        pid = item["id"]
        cols = []
        ok = True
        for name in names:
            arr = streams[name].get(pid)
            if arr is None:
                ok = False
                break
            gpu_p = np.asarray(item["probs"], dtype=np.float32)
            if len(arr) != len(gpu_p):
                ok = False
                break
            cols.append(arr)
        if not ok:
            continue
        # Train the stacker only on residues that carry a real label; the
        # aligned arrays are full-length with sentinels elsewhere.
        from colab.biological_utility import evidenced

        m = evidenced(item)
        if not m.any():
            continue
        mat = np.stack(cols, axis=1)[m]
        chunks_x.append(mat)
        chunks_y.append(np.asarray(item["labels"], dtype=np.float32)[m])
        row_protein.append(np.full(mat.shape[0], len(used_ids), dtype=np.int64))
        used_ids.append(pid)

    if not chunks_x:
        raise ValueError("No aligned residues for meta-ensemble stacking")
    return (
        np.vstack(chunks_x),
        np.concatenate(chunks_y),
        used_ids,
        np.concatenate(row_protein),
    )


def fit_meta_stacker(
    labels: np.ndarray,
    features: np.ndarray,
    C: float = 0.5,
) -> tuple[LogisticRegression, StandardScaler]:
    """Fit L2-regularized logistic meta-learner."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    model = LogisticRegression(
        C=C,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X, labels.astype(np.int8))
    return model, scaler


def apply_meta_stacker(
    proteins: list,
    fold_results: list,
    streams: dict[str, dict[str, np.ndarray]],
    n_folds: int = 5,
    C: float = 0.5,
) -> tuple[dict, list]:
    """
    Learned blend of named prediction streams on OOF residues.

    streams: {"gpu": probs_by_id, "v6": ..., "physics": ...}
    """
    before = compute_pooled_metrics(fold_results)
    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)

    try:
        X, y, used_ids, row_protein = _build_stacker_matrix(aligned, streams)
    except ValueError as exc:
        return {"skipped": True, "reason": str(exc)}, fold_results

    if len(np.unique(y)) < 2:
        return {"skipped": True, "reason": "insufficient label diversity"}, fold_results

    # Out-of-fold stacking. Fitting the meta-learner on every OOF residue and
    # then scoring those same residues makes `after` a resubstitution score, so
    # delta_auc_pooled would flatter the model against a
    # genuinely held-out reference. Instead each residue is scored by a stacker
    # that never saw its protein, grouping by protein so no protein straddles
    # the fit/score boundary.
    protein_fold = _assign_protein_folds(proteins, fold_results, used_ids, n_folds)
    row_fold = protein_fold[row_protein]

    stacked_probs = np.empty(len(y), dtype=np.float32)
    n_meta_folds = 0
    for f in np.unique(row_fold):
        score_mask = row_fold == f
        fit_mask = ~score_mask
        if not fit_mask.any() or len(np.unique(y[fit_mask])) < 2:
            # Degenerate meta-fold: fall back to the full fit for these rows.
            continue
        m_f, s_f = fit_meta_stacker(y[fit_mask], X[fit_mask], C=C)
        stacked_probs[score_mask] = (
            m_f.predict_proba(s_f.transform(X[score_mask]))[:, 1].astype(np.float32)
        )
        n_meta_folds += 1

    # Final reported coefficients come from a full fit (for interpretability and
    # for deployment), but they are NOT what produced `stacked_probs`.
    model, scaler = fit_meta_stacker(y, X, C=C)
    if n_meta_folds == 0:
        stacked_probs = model.predict_proba(scaler.transform(X))[:, 1].astype(np.float32)

    stream_names = list(streams.keys())
    coefs = dict(zip(stream_names, model.coef_[0].tolist()))
    in_sample = model.predict_proba(scaler.transform(X))[:, 1].astype(np.float32)
    report_fit = {
        "stream_names": stream_names,
        "coefficients": coefs,
        "intercept": float(model.intercept_[0]),
        "train_auc": float(roc_auc_score(y, in_sample)),
        "train_ap": float(average_precision_score(y, in_sample)),
        "oof_auc": float(roc_auc_score(y, stacked_probs)),
        "oof_ap": float(average_precision_score(y, stacked_probs)),
        "n_meta_folds": n_meta_folds,
        "stacking": "out_of_fold" if n_meta_folds else "in_sample_fallback",
    }

    # stacked_probs has one row per *evidenced* residue, in the same order the
    # training matrix was built. Scatter it back to sequence positions so the
    # item keeps its full-length contract, leaving unlabelled positions at their
    # original sentinel rather than shifting every downstream residue.
    from colab.biological_utility import evidenced

    offset = 0
    aligned_stacked = []
    for item in aligned:
        pid = item["id"]
        m = evidenced(item)
        n = int(m.sum())
        if n and all(pid in streams[k] for k in stream_names):
            new_item = dict(item)
            probs_full = np.asarray(item["probs"], dtype=np.float32).copy()
            probs_full[m] = stacked_probs[offset:offset + n]
            new_item["probs"] = probs_full
            new_item["meta_ensemble"] = True
            aligned_stacked.append(new_item)
            offset += n
        else:
            aligned_stacked.append(item)
    if offset != len(stacked_probs):
        raise RuntimeError(
            f"meta-ensemble scatter consumed {offset} of {len(stacked_probs)} "
            "stacked rows — the write-back order does not match the fit order."
        )

    fold_results_stacked = write_fused_probs_to_fold_results(
        proteins, fold_results, aligned_stacked, n_folds=n_folds,
    )
    for fr in fold_results_stacked:
        fr["meta_ensemble"] = True
        fr["meta_coefficients"] = coefs

    after = compute_pooled_metrics(fold_results_stacked)
    report = {
        "fit": report_fit,
        "before": {"pooled": {k: before[k] for k in ("auc", "ap", "n_residues")}},
        "after": {"pooled": {k: after[k] for k in ("auc", "ap", "n_residues")}},
        "delta_auc_pooled": after["auc"] - before["auc"],
        "delta_ap_pooled": after["ap"] - before["ap"],
        "benchmark_comparability": (
            "DisProt homology-CV pooled AUC. NOT comparable to CAID3 figures such as ESMDisPred 0.895: different label definition (curator-annotated functional disorder vs missing residues in crystal structures), different proteins, different protocol. The comparable measurement is caid3_eval_report.json."
        ),
        "method": "logistic_meta_stacker",
    }
    return report, fold_results_stacked


def print_meta_ensemble_report(report: dict) -> None:
    print(f"\n{'═' * 64}")
    print(" LEARNED META-ENSEMBLE (logistic stacker)")
    print(f"{'═' * 64}")
    if report.get("skipped"):
        print(f"  Skipped: {report.get('reason')}")
        return
    fit = report["fit"]
    print(f"  Streams: {', '.join(fit['stream_names'])}")
    for name, coef in fit["coefficients"].items():
        print(f"    {name:10s}: {coef:+.4f}")
    b, a = report["before"]["pooled"], report["after"]["pooled"]
    print(f"  Before : AUC={b['auc']:.4f}  AP={b['ap']:.4f}")
    print(f"  After  : AUC={a['auc']:.4f}  AP={a['ap']:.4f}")
    print(f"  Δ AUC  : {report['delta_auc_pooled']:+.4f}")
    print(f"{'═' * 64}")


def save_meta_ensemble_report(report: dict, path: str = "meta_ensemble_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
