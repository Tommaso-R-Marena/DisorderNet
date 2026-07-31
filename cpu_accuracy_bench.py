"""Reproducible accuracy harness for the CPU disorder pipeline.

Answers one question: *does a change to the CPU model help, hurt, or do
nothing?* It runs the same 5-fold protein-grouped CV the v6 pipeline runs, so
two variants can be compared under identical splits, seeds and folds.

Two data sources:

* **Real DisProt** (``--data real``, the default when
  ``data/disprot_processed.json`` plus ESM embeddings are present). This is the
  number that matters — ``run_v6_mem.py`` reports ~0.831 pooled AUC.
* **Synthetic** (``--data synthetic``) for environments without the DisProt
  download. Sequences, disorder segments and a correlated pseudo-ESM channel
  are generated with the structure the real task has: contiguous IDRs, biased
  composition inside them, spatial autocorrelation, and an embedding that
  carries a noisy smoothed copy of the latent disorder signal. Absolute AUC on
  synthetic data is **not** comparable to the DisProt number; it is only
  meaningful as an A/B between variants run on the same corpus.

Variants (``--variant``) exist so an improvement can be measured rather than
asserted:

    baseline   fixed 0.55/0.45 LGB/XGB blend, no post-hoc smoothing
    smoothed   baseline + windowed smoothing of per-residue probabilities

Usage::

    python cpu_accuracy_bench.py --data synthetic --variant baseline
    python cpu_accuracy_bench.py --data synthetic --variant smoothed
    python cpu_accuracy_bench.py --data real --variant smoothed --out ab.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from run_v6_mem import (ESM_PCA, MAX_LEN, MAX_PROT, SEED, evaluate, phys, wavg, wvar,
                        youden_threshold)

AA = "ACDEFGHIKLMNPQRSTVWY"
DISORDER_RICH = "PEKSQGATDNR"
ORDER_RICH = "WCFIYVLMHA"

VARIANTS = ("baseline", "smoothed")
SMOOTH_HALF_WIDTH = 3  # window 7, matching predictor.py / run_v7.py


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------
def _smooth_field(values, half_width, passes=2):
    """Approximate Gaussian smoothing by repeated box filtering."""
    out = np.asarray(values, dtype=np.float32)
    for _ in range(passes):
        out = wavg(out, half_width)
    return out


def make_synthetic_corpus(
    n_proteins=300,
    emb_dim=64,
    seed=SEED,
    composition_strength=0.85,
    embedding_noise=2.6,
    label_noise=0.03,
    hidden_strength=0.65,
):
    """Proteins with contiguous IDRs, weakly biased composition and a noisy
    pseudo-ESM channel.

    The generative process is deliberately *hard*, because a saturated
    benchmark cannot rank variants:

    * a smooth latent field ``z`` along the sequence produces contiguous IDRs
      with fuzzy boundaries, rather than rectangular blocks;
    * residues are drawn from a softmax whose log-odds are shifted by
      ``composition_strength * z * top_idp_propensity`` — so composition is
      informative but far from deterministic, as in real IDRs;
    * the pseudo-ESM channel is a low-rank, heavily-noised projection of ``z``;
    * labels depend on ``z`` **plus a second smooth field that is never
      observed** (``hidden_strength``). This is what stops the benchmark from
      saturating: no featurizer can recover the hidden component, so there is a
      genuine Bayes ceiling below 1.0, exactly as on real data where disorder
      is not a deterministic function of local sequence;
    * a few percent of labels are flipped near segment boundaries, mimicking
      annotation ambiguity in DisProt.

    Defaults are tuned so the v6 pipeline lands in the 0.80-0.87 AUC band —
    the same regime as the real DisProt run (~0.831 pooled), leaving headroom
    to detect both improvements and regressions. Raising ``hidden_strength``
    lowers the ceiling; the calibration sweep behind the default is:

        hidden_strength  0.60  1.00  1.35  1.80  2.40
        single-LGB AUC   0.845 0.754 0.710 0.669 0.598
    """
    rng = np.random.RandomState(seed)
    proteins = []
    projection = rng.randn(3, emb_dim).astype(np.float32)
    # Top-IDP disorder propensity, indexed like AA; drives composition drift.
    propensity = np.array(
        [0.06, -0.02, 0.192, 0.736, -0.697, 0.166, 0.303, -0.486, 0.586, -0.326,
         -0.397, 0.007, 0.987, 0.318, 0.18, 0.341, 0.059, -0.121, -0.884, -0.510],
        dtype=np.float32,
    )
    base_log_freq = np.log(rng.dirichlet(np.full(20, 12.0)).astype(np.float32))

    for i in range(n_proteins):
        length = int(rng.randint(60, 420))

        # Smooth latent disorder field, standardised per protein.
        z = _smooth_field(rng.randn(length).astype(np.float32), half_width=9, passes=2)
        z = (z - z.mean()) / (z.std() + 1e-6)

        # Second smooth field driving the label but never exposed to the model.
        hidden = _smooth_field(rng.randn(length).astype(np.float32), half_width=7, passes=2)
        hidden = (hidden - hidden.mean()) / (hidden.std() + 1e-6)
        drive = z + hidden_strength * hidden

        # Threshold set per protein so the disorder fraction varies around ~30%.
        threshold = float(rng.normal(0.45, 0.35)) * np.sqrt(1.0 + hidden_strength ** 2)
        labels = (drive > threshold).astype(np.int8)
        if labels.sum() < 3 or (length - labels.sum()) < 3:
            labels = (drive > np.quantile(drive, 0.7)).astype(np.int8)

        # Annotation ambiguity is a *boundary* phenomenon in DisProt: regions are
        # curated as intervals, so labels stay segmental and only the edges move.
        # (No i.i.d. per-residue flips — those would be unrealistic and would
        # also rig any comparison involving prediction smoothing.)
        if label_noise > 0:
            jitter_scale = max(1, int(round(label_noise * 100)))
            for start, end in _segments(labels):
                jitter = int(rng.randint(-jitter_scale, jitter_scale + 1))
                if jitter > 0 and end + jitter <= length:
                    labels[end:end + jitter] = 1
                elif jitter < 0 and end + jitter > start:
                    labels[end + jitter:end] = 0

        # Composition: softmax over the 20 residues, log-odds shifted by z.
        logits = base_log_freq[None, :] + composition_strength * z[:, None] * propensity[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        cumulative = np.cumsum(probs, axis=1)
        draws = rng.rand(length, 1)
        picks = (draws > cumulative).sum(axis=1).clip(0, 19)
        sequence = "".join(AA[p] for p in picks)

        # Pseudo-ESM: low-rank projection of the latent field, heavily noised.
        latent_stack = np.stack([z, _smooth_field(z, 20, passes=1),
                                 np.linspace(0.0, 1.0, length, dtype=np.float32)], axis=1)
        emb = latent_stack @ projection
        emb += rng.randn(length, emb_dim).astype(np.float32) * embedding_noise

        proteins.append({
            "disprot_id": f"SYN_{i:05d}",
            "sequence": sequence,
            "length": length,
            "disorder_labels": labels.astype(int).tolist(),
            "_embedding": emb.astype(np.float32),
        })
    return proteins


def _segments(labels):
    """Contiguous runs of 1s as half-open intervals."""
    flags = np.asarray(labels) != 0
    edges = np.diff(np.concatenate(([False], flags, [False])).view(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


# ---------------------------------------------------------------------------
# Real corpus
# ---------------------------------------------------------------------------
def load_real_corpus(max_proteins=MAX_PROT, seed=SEED):
    """The exact protein selection run_v6_mem.main() uses."""
    from disordernet_paths import DISPROT_JSON, EMB_DIR

    if not os.path.exists(DISPROT_JSON):
        raise FileNotFoundError(
            f"{DISPROT_JSON} not found — run `python fetch_disprot.py` first "
            "(requires network access to disprot.org)"
        )
    with open(DISPROT_JSON) as handle:
        all_data = json.load(handle)

    proteins = [
        p for p in all_data
        if os.path.exists(os.path.join(EMB_DIR, f"{p['disprot_id']}.npy"))
        and 30 <= p["length"] <= MAX_LEN
        and sum(p["disorder_labels"]) >= 3
        and p["length"] - sum(p["disorder_labels"]) >= 3
    ]
    if not proteins:
        raise FileNotFoundError(
            f"no proteins with embeddings under {EMB_DIR} — run "
            "`python extract_esm_embeddings.py` first"
        )

    rng = np.random.RandomState(seed)
    if len(proteins) > max_proteins:
        idx = rng.choice(len(proteins), max_proteins, replace=False)
        proteins = [proteins[i] for i in sorted(idx)]
    for p in proteins:
        emb = np.load(os.path.join(EMB_DIR, f"{p['disprot_id']}.npy"))
        p["_embedding"] = emb.astype(np.float32)[: p["length"]]
    return proteins


# ---------------------------------------------------------------------------
# Feature + CV pipeline (mirrors run_v6_mem.main)
# ---------------------------------------------------------------------------
def build_feature_matrices(proteins, esm_pca=ESM_PCA, seed=SEED, verbose=False):
    """Physics features + PCA-reduced ESM context, one matrix per protein."""
    rng = np.random.RandomState(seed)
    pca = IncrementalPCA(n_components=min(esm_pca, proteins[0]["_embedding"].shape[1]),
                         batch_size=10000)
    sample_idx = rng.choice(len(proteins), min(800, len(proteins)), replace=False)
    pca.fit(np.vstack([proteins[i]["_embedding"] for i in sample_idx]))

    feats, labels = [], []
    for p in proteins:
        length = p["length"]
        ph = phys(p["sequence"][:length])
        ep = pca.transform(p["_embedding"][:length])
        block = np.concatenate(
            [ph, ep, wavg(ep, 4), wavg(ep, 12), wavg(ep, 25), wvar(ep, 8), wvar(ep, 20)], axis=1,
        )
        feats.append(np.nan_to_num(block).astype(np.float32))
        labels.append(np.asarray(p["disorder_labels"][:length], dtype=np.float32))
    if verbose:
        print(f"  feature dim: {feats[0].shape[1]}  explained var: "
              f"{pca.explained_variance_ratio_.sum():.3f}")
    return feats, labels


def smooth_per_protein(probs, lengths, half_width=SMOOTH_HALF_WIDTH):
    """Windowed mean of per-residue probabilities, applied within each protein.

    Disorder is a *segment* property: neighbouring residues share a label far
    more often than not. Averaging inside a protein removes isolated
    single-residue spikes without crossing protein boundaries.
    """
    out = np.empty_like(probs)
    offset = 0
    for length in lengths:
        out[offset:offset + length] = wavg(probs[offset:offset + length], half_width)
        offset += length
    return out


def run_cv(proteins, variant="baseline", n_splits=5, seed=SEED, n_jobs=2, verbose=True,
           model_seed=None, n_rounds=700):
    """5-fold protein-grouped CV. Returns pooled + per-fold metrics.

    ``model_seed`` is separate from ``seed`` so the booster's stochasticity can
    be varied while holding the corpus, splits and features fixed. Re-running
    with only ``model_seed`` changed measures the noise floor of this benchmark,
    which is the yardstick any claimed improvement has to beat.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    if model_seed is None:
        model_seed = seed

    import lightgbm as lgb
    import xgboost as xgb

    feats, labels = build_feature_matrices(proteins, seed=seed, verbose=verbose)
    n = len(proteins)
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics, pooled_true, pooled_pred = [], [], []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(range(n), range(n), range(n))):
        # Per-fold RNG: undersampling must not depend on how many folds ran before.
        fold_rng = np.random.RandomState(seed + fold)

        x_val = np.vstack([feats[i] for i in val_idx])
        y_val = np.concatenate([labels[i] for i in val_idx])
        val_lengths = [proteins[i]["length"] for i in val_idx]

        x_train_full = np.vstack([feats[i] for i in train_idx])
        y_train_full = np.concatenate([labels[i] for i in train_idx])
        pos = np.where(y_train_full == 1)[0]
        neg = np.where(y_train_full == 0)[0]
        keep_neg = min(len(neg), len(pos) * 3)
        keep = np.sort(np.concatenate([pos, fold_rng.choice(neg, keep_neg, replace=False)]))
        x_train, y_train = x_train_full[keep], y_train_full[keep]
        del x_train_full, y_train_full

        spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        booster = lgb.train(
            {"objective": "binary", "metric": "auc", "num_leaves": 127, "max_depth": 8,
             "learning_rate": 0.05, "feature_fraction": 0.7, "bagging_fraction": 0.7,
             "bagging_freq": 5, "scale_pos_weight": spw, "min_child_samples": 25,
             "reg_alpha": 0.05, "reg_lambda": 0.5, "verbose": -1, "n_jobs": n_jobs,
             "seed": model_seed},
            lgb.Dataset(x_train, label=y_train), n_rounds,
            valid_sets=[lgb.Dataset(x_val, label=y_val)],
            callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)])
        lgb_pred = booster.predict(x_val)

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        xgb_model = xgb.train(
            {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 7,
             "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7,
             "scale_pos_weight": spw, "min_child_weight": 25, "reg_alpha": 0.05,
             "reg_lambda": 0.5, "tree_method": "hist", "nthread": n_jobs,
             "seed": model_seed},
            dtrain, n_rounds, evals=[(dval, "v")], early_stopping_rounds=25,
            verbose_eval=False)
        xgb_pred = xgb_model.predict(dval)

        blended = 0.55 * lgb_pred + 0.45 * xgb_pred
        if variant == "smoothed":
            blended = smooth_per_protein(blended, val_lengths)

        pooled_true.append(y_val)
        pooled_pred.append(blended)
        if verbose:
            fold_auc = roc_auc_score(y_val, blended)
            print(f"  fold {fold + 1}/{n_splits}  AUC={fold_auc:.4f} "
                  f"AP={average_precision_score(y_val, blended):.4f}")

    y_all = np.concatenate(pooled_true)
    p_all = np.concatenate(pooled_pred)

    # Second pass for the thresholded metrics. A fold's decision threshold comes
    # from the *other* folds' out-of-fold predictions, never from the fold being
    # scored — picking it in-fold (run_v6_mem.evaluate's default) inflates
    # f1/mcc/precision/recall because the threshold has seen those labels.
    fold_metrics = []
    for fold in range(len(pooled_true)):
        others_y = np.concatenate([pooled_true[j] for j in range(len(pooled_true)) if j != fold])
        others_p = np.concatenate([pooled_pred[j] for j in range(len(pooled_pred)) if j != fold])
        threshold = youden_threshold(others_y, others_p)
        fold_metrics.append(evaluate(pooled_true[fold], pooled_pred[fold], threshold=threshold))

    held_out_threshold = float(np.mean([m["threshold"] for m in fold_metrics]))
    pooled = evaluate(y_all, p_all, threshold=held_out_threshold)
    pooled_in_fold = evaluate(y_all, p_all)  # legacy, optimistic — for comparison
    fold_aucs = [m["auc_roc"] for m in fold_metrics]
    return {
        "variant": variant,
        "model_seed": model_seed,
        "n_proteins": len(proteins),
        "n_residues": int(len(y_all)),
        "pooled": {k: float(v) for k, v in pooled.items()},
        "pooled_in_fold_threshold": {k: float(v) for k, v in pooled_in_fold.items()},
        "fold_aucs": [float(v) for v in fold_aucs],
        "cv_mean_auc": float(np.mean(fold_aucs)),
        "cv_std_auc": float(np.std(fold_aucs)),
        "fold_metrics": [{k: float(v) for k, v in m.items()} for m in fold_metrics],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", choices=("auto", "real", "synthetic"), default="auto")
    parser.add_argument("--variant", choices=VARIANTS + ("all",), default="all")
    parser.add_argument("--n-proteins", type=int, default=300,
                        help="synthetic corpus size (real corpus uses MAX_PROT)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args(argv)

    source = args.data
    if source == "auto":
        from disordernet_paths import DISPROT_JSON
        source = "real" if os.path.exists(DISPROT_JSON) else "synthetic"

    if source == "real":
        proteins = load_real_corpus(seed=args.seed)
    else:
        proteins = make_synthetic_corpus(n_proteins=args.n_proteins, seed=args.seed)

    residues = sum(p["length"] for p in proteins)
    disordered = sum(int(np.sum(p["disorder_labels"])) for p in proteins)
    print(f"Source: {source} | {len(proteins)} proteins | {residues:,} residues "
          f"({100 * disordered / residues:.1f}% disordered)")
    if source == "synthetic":
        print("NOTE: synthetic AUC is only meaningful as an A/B between variants,\n"
              "      not as a substitute for the DisProt number.")

    variants = VARIANTS if args.variant == "all" else (args.variant,)
    results = {}
    for variant in variants:
        print(f"\n=== variant: {variant} ===")
        started = time.time()
        results[variant] = run_cv(
            proteins, variant=variant, seed=args.seed, n_jobs=args.n_jobs,
        )
        results[variant]["seconds"] = round(time.time() - started, 1)
        pooled = results[variant]["pooled"]
        print(f"  pooled AUC={pooled['auc_roc']:.4f} AP={pooled['avg_precision']:.4f} "
              f"F1={pooled['f1']:.4f} MCC={pooled['mcc']:.4f} "
              f"({results[variant]['seconds']}s)")

    if len(results) > 1:
        print("\n=== summary ===")
        for name, res in results.items():
            print(f"  {name:10s} pooled AUC {res['pooled']['auc_roc']:.4f}  "
                  f"CV {res['cv_mean_auc']:.4f} ± {res['cv_std_auc']:.4f}")
        base, best = results.get("baseline"), results.get("smoothed")
        if base and best:
            delta = best["pooled"]["auc_roc"] - base["pooled"]["auc_roc"]
            wins = sum(b > a for a, b in zip(base["fold_aucs"], best["fold_aucs"]))
            print(f"  smoothed - baseline: {delta:+.4f} pooled AUC, "
                  f"better on {wins}/{len(base['fold_aucs'])} folds")

    payload = {"source": source, "seed": args.seed, "results": results}
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote {args.out}")
    return payload


if __name__ == "__main__":
    main()
