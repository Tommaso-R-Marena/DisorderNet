"""Accuracy regression tests for the CPU disorder pipeline.

These guard the properties that determine whether the CPU model still predicts
as well as it used to. They deliberately avoid asserting an absolute AUC on
real DisProt data (which is not vendored); instead they pin the things that
would silently move that number:

* the featurizer is deterministic, finite, and numerically correct against a
  brute-force float64 reference (a drift here changes every downstream model);
* the CV protocol keeps proteins whole and folds disjoint (a leak here inflates
  the number rather than lowering it);
* per-protein smoothing does not bleed across protein boundaries;
* an end-to-end CV run on a learnable synthetic corpus clears a floor, so a
  change that breaks the pipeline outright cannot pass silently.

``cpu_accuracy_bench.py --data real`` is the harness for the real DisProt
number; these tests are the fast guard rail that runs in CI.
"""
from __future__ import annotations

import numpy as np
import pytest

from cpu_accuracy_bench import (
    _segments as _bench_segments,
    build_feature_matrices,
    make_synthetic_corpus,
    run_cv,
    smooth_per_protein,
)
from run_v6_mem import (PHYS_DIM, evaluate, phys, smooth_by_protein, wavg, wvar,
                        youden_threshold)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------------------
# Featurizer correctness — the input every CPU model depends on
# ---------------------------------------------------------------------------
def _brute_force_window_mean(values, half):
    length = len(values)
    return np.array([
        values[max(0, i - half):min(length, i + half + 1)].astype(np.float64).mean(axis=0)
        for i in range(length)
    ])


def _brute_force_window_var(values, half):
    length = len(values)
    return np.array([
        values[max(0, i - half):min(length, i + half + 1)].astype(np.float64).var(axis=0)
        for i in range(length)
    ])


@pytest.mark.parametrize("half", [3, 7, 30, 50])
def test_window_mean_matches_brute_force_on_large_scales(half):
    """The molecular-weight/bulkiness scales are where float32 cumsum used to fail."""
    rng = np.random.RandomState(0)
    values = (rng.rand(3000, 4).astype(np.float32) * 40.0 + 180.0)
    got = wavg(values, half)
    want = _brute_force_window_mean(values, half)
    assert np.abs(got - want).max() < 1e-3


@pytest.mark.parametrize("half", [5, 15, 30])
def test_window_variance_matches_brute_force_and_stays_non_negative(half):
    rng = np.random.RandomState(1)
    values = (rng.rand(3000, 3).astype(np.float32) * 2.0 + 200.0)
    got = wvar(values, half)
    want = _brute_force_window_var(values, half)
    assert np.abs(got - want).max() < 1e-2
    assert (got >= 0).all()


def test_phys_is_deterministic():
    seq = "".join(np.random.RandomState(2).choice(list("ACDEFGHIKLMNPQRSTVWY"), 500))
    first = phys(seq)
    for _ in range(3):
        assert np.array_equal(phys(seq), first)


def test_phys_output_is_finite_and_correctly_shaped():
    rng = np.random.RandomState(3)
    for length in (30, 137, 800):
        seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWYX"), length))
        out = phys(seq)
        assert out.shape == (length, PHYS_DIM)
        assert np.isfinite(out).all()


def test_phys_separates_disordered_from_ordered_composition():
    """Sanity check that the features carry the biological signal at all."""
    disordered = phys("PEKSQGPEKSQG" * 20)
    ordered = phys("WCFIYVWCFIYV" * 20)
    # column 2 of the property block is the Top-IDP disorder propensity scale
    assert disordered[:, 2].mean() > ordered[:, 2].mean()


def test_phys_is_translation_invariant_in_the_interior():
    """Identical local context must give identical windowed features.

    Only the position/global columns may differ, so a regression that leaks
    absolute position into a windowed feature is caught here.
    """
    motif = "PEKSQGARND"
    left = phys("A" * 60 + motif + "A" * 60)
    right = phys("A" * 90 + motif + "A" * 90)
    # windowed block spans columns 11..109 (see phys() layout)
    assert np.allclose(left[60:70, 11:110], right[90:100, 11:110], atol=1e-5)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def test_smoothing_does_not_cross_protein_boundaries():
    probs = np.concatenate([np.zeros(20, np.float64), np.ones(20, np.float64)])
    smoothed = smooth_per_protein(probs, [20, 20], half_width=3)
    assert smoothed[:20].max() == pytest.approx(0.0)
    assert smoothed[20:].min() == pytest.approx(1.0)


def test_smoothing_preserves_length_and_range():
    rng = np.random.RandomState(4)
    lengths = [17, 40, 3]
    probs = rng.rand(sum(lengths))
    smoothed = smooth_per_protein(probs, lengths, half_width=3)
    assert smoothed.shape == probs.shape
    assert smoothed.min() >= probs.min() - 1e-9
    assert smoothed.max() <= probs.max() + 1e-9


def test_smoothing_suppresses_isolated_spikes():
    probs = np.zeros(41)
    probs[20] = 1.0
    smoothed = smooth_per_protein(probs, [41], half_width=3)
    assert smoothed[20] < probs[20]
    assert smoothed[20] == pytest.approx(1.0 / 7.0)


def test_smoothing_is_identity_on_a_constant_protein():
    probs = np.full(30, 0.42)
    assert np.allclose(smooth_per_protein(probs, [30], half_width=3), 0.42)


def test_run_v6_smoothing_helper_matches_the_bench():
    """run_v6_mem.smooth_by_protein and the bench's helper must not diverge."""
    rng = np.random.RandomState(14)
    lengths = [23, 41, 7]
    probs = rng.rand(sum(lengths))
    assert np.allclose(
        smooth_by_protein(probs, lengths, 3),
        smooth_per_protein(probs, lengths, half_width=3),
    )


def test_run_v6_smoothing_is_off_by_default_and_a_no_op_at_zero():
    """The historical results_v6 numbers must stay reproducible out of the box."""
    import run_v6_mem

    assert run_v6_mem.SMOOTH_HW == 0
    probs = np.linspace(0.0, 1.0, 25)
    assert np.array_equal(smooth_by_protein(probs, [25], 0), probs)


def test_run_v6_smoothing_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        smooth_by_protein(np.zeros(20), [10, 5], 3)


# ---------------------------------------------------------------------------
# Corpus + CV protocol
# ---------------------------------------------------------------------------
def test_synthetic_corpus_is_reproducible_and_well_formed():
    a = make_synthetic_corpus(n_proteins=6, seed=7)
    b = make_synthetic_corpus(n_proteins=6, seed=7)
    for pa, pb in zip(a, b):
        assert pa["sequence"] == pb["sequence"]
        assert pa["disorder_labels"] == pb["disorder_labels"]
        assert np.array_equal(pa["_embedding"], pb["_embedding"])

    for p in a:
        assert len(p["sequence"]) == p["length"]
        assert len(p["disorder_labels"]) == p["length"]
        assert p["_embedding"].shape[0] == p["length"]
        # both classes present, otherwise the task is degenerate
        assert 0 < sum(p["disorder_labels"]) < p["length"]


def test_synthetic_corpus_has_contiguous_disorder_segments():
    """Disorder must be segmental, not i.i.d. per residue.

    Segmental labels are what make sequence-context features (and prediction
    smoothing) meaningful, so this compares the observed transition rate
    against the rate i.i.d. labels with the same marginal would produce.
    """
    proteins = make_synthetic_corpus(n_proteins=60, seed=8)
    labels = [np.asarray(p["disorder_labels"]) for p in proteins]

    transitions = sum(int(np.sum(a[1:] != a[:-1])) for a in labels)
    residues = sum(len(a) for a in labels)
    disorder_rate = sum(int(a.sum()) for a in labels) / residues
    iid_transitions = 2 * disorder_rate * (1 - disorder_rate) * residues

    # An order of magnitude below the i.i.d. rate means genuinely blocky labels.
    assert transitions < iid_transitions / 10

    segment_lengths = [
        end - start for a in labels for start, end in _bench_segments(a)
    ]
    assert np.median(segment_lengths) >= 10
    assert 0.20 < disorder_rate < 0.45, "disorder fraction should resemble DisProt"


def test_feature_matrices_align_with_labels_and_are_finite():
    proteins = make_synthetic_corpus(n_proteins=12, seed=9)
    feats, labels = build_feature_matrices(proteins, esm_pca=8, seed=9, verbose=False)
    assert len(feats) == len(labels) == len(proteins)
    for block, label, protein in zip(feats, labels, proteins):
        assert block.shape[0] == label.shape[0] == protein["length"]
        assert np.isfinite(block).all()
    widths = {block.shape[1] for block in feats}
    assert len(widths) == 1, "every protein must produce the same feature width"


def test_cv_folds_are_disjoint_and_keep_proteins_whole():
    from sklearn.model_selection import GroupKFold

    n = 37
    seen = []
    for _, val_idx in GroupKFold(n_splits=5).split(range(n), range(n), range(n)):
        seen.append(set(val_idx.tolist()))
    union = set().union(*seen)
    assert union == set(range(n)), "every protein must be validated exactly once"
    for i, a in enumerate(seen):
        for b in seen[i + 1:]:
            assert not (a & b), "validation folds must not overlap"


def test_evaluate_accepts_an_external_threshold():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.9, 0.45, 0.55])
    at_half = evaluate(y_true, y_prob, threshold=0.5)
    assert at_half["threshold"] == 0.5
    # a threshold of 0 labels everything positive -> perfect recall
    assert evaluate(y_true, y_prob, threshold=0.0)["recall"] == pytest.approx(1.0)
    # ranking metrics must not depend on the threshold at all
    assert at_half["auc_roc"] == pytest.approx(evaluate(y_true, y_prob)["auc_roc"])


def test_in_fold_threshold_is_optimistic_relative_to_held_out():
    """The default Youden threshold is fitted on the data it is graded against.

    On pure noise an in-fold threshold still manufactures above-chance MCC,
    while a threshold taken from independent data does not. This is why the
    bench derives fold thresholds from the other folds.
    """
    rng = np.random.RandomState(12)
    n = 4000
    y_true = (rng.rand(n) < 0.3).astype(float)
    y_prob = rng.rand(n)  # no signal whatsoever

    in_fold = evaluate(y_true, y_prob)["mcc"]

    independent_y = (rng.rand(n) < 0.3).astype(float)
    independent_p = rng.rand(n)
    external = youden_threshold(independent_y, independent_p)
    held_out = evaluate(y_true, y_prob, threshold=external)["mcc"]

    assert in_fold > held_out
    assert abs(held_out) < 0.05, "an independent threshold should score near chance"


def test_run_cv_reports_both_threshold_conventions():
    proteins = make_synthetic_corpus(n_proteins=40, seed=13)
    result = run_cv(proteins, variant="baseline", n_splits=3, seed=13,
                    n_jobs=2, verbose=False, n_rounds=40)
    assert "pooled" in result and "pooled_in_fold_threshold" in result
    # AUC is threshold-free, so both conventions must agree on it exactly
    assert result["pooled"]["auc_roc"] == pytest.approx(
        result["pooled_in_fold_threshold"]["auc_roc"]
    )
    # the optimistic convention cannot score worse on the metric it optimises
    assert result["pooled_in_fold_threshold"]["balanced_acc"] >= (
        result["pooled"]["balanced_acc"] - 1e-9
    )
    for fold_metric in result["fold_metrics"]:
        assert np.isfinite(fold_metric["threshold"])


def test_evaluate_returns_finite_metrics_in_range():
    rng = np.random.RandomState(10)
    y_true = (rng.rand(2000) < 0.3).astype(float)
    y_prob = np.clip(0.25 * rng.randn(2000) + y_true * 0.5 + 0.25, 0, 1)
    metrics = evaluate(y_true, y_prob)
    for key in ("auc_roc", "avg_precision", "f1", "precision", "recall", "balanced_acc"):
        assert 0.0 <= metrics[key] <= 1.0, key
    assert -1.0 <= metrics["mcc"] <= 1.0
    assert metrics["auc_roc"] > 0.6, "signal should be recoverable"


# ---------------------------------------------------------------------------
# End-to-end floor
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_end_to_end_cv_clears_accuracy_floor():
    """A structural break in the pipeline must not pass silently.

    The floor is deliberately loose — this is a smoke floor on a synthetic
    corpus, not a claim about DisProt accuracy.
    """
    proteins = make_synthetic_corpus(n_proteins=60, seed=11)
    result = run_cv(proteins, variant="smoothed", n_splits=3, seed=11,
                    n_jobs=2, verbose=False)
    assert result["pooled"]["auc_roc"] > 0.70
    assert len(result["fold_aucs"]) == 3
    assert all(np.isfinite(result["fold_aucs"]))
    assert result["n_residues"] == sum(p["length"] for p in proteins)
