"""Two defects that made a headline number describe something it wasn't.

1. `stage_stack` reassigned `cv_summary["pooled_auc"]` to the stacked result, so
   the field kept its name while changing meaning — from the neural model's
   cross-validated AUC to the score of an ensemble containing a gradient-boosted
   tree on physics features. In the 650M run the model's own pooled OOF AUC was
   0.7203 and the field reported 0.7876.

2. Pooled AUC ranks residues across folds scored by separately-calibrated
   models. That run's per-fold median probability ranged from 0.0003 to 0.6523
   at near-identical prevalence, and pooled AUC (0.7203) sat 0.029 *below*
   mean-of-folds (0.7491) as a result.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score


def _rank_normalize(folds):
    out = []
    for p in folds:
        p = np.asarray(p, dtype=float)
        out.append((p.argsort().argsort() + 0.5) / len(p))
    return np.concatenate(out)


class TestCrossFoldCalibrationDrift:
    """Reproduces the mechanism on synthetic folds with a known ground truth."""

    def _two_folds(self, offset):
        rng = np.random.default_rng(0)
        n = 4000
        folds, labels = [], []
        for i in range(2):
            lab = rng.binomial(1, 0.18, n)
            # Same discriminative signal in both folds...
            score = rng.normal(lab * 1.2, 1.0)
            # ...but fold 1's model outputs a shifted score scale.
            probs = 1 / (1 + np.exp(-(score + (offset if i else 0))))
            folds.append(probs)
            labels.append(lab)
        return folds, labels

    def test_an_offset_between_folds_damages_pooled_auc(self):
        folds, labels = self._two_folds(offset=4.0)
        L = np.concatenate(labels)
        per_fold = [roc_auc_score(labels[i], folds[i]) for i in range(2)]
        pooled = roc_auc_score(L, np.concatenate(folds))
        assert pooled < np.mean(per_fold) - 0.02, (
            "an offset between equally-good fold models must cost pooled AUC"
        )

    def test_rank_normalising_recovers_mean_of_folds(self):
        """The diagnostic: if this closes the gap, the gap was calibration."""
        folds, labels = self._two_folds(offset=4.0)
        L = np.concatenate(labels)
        per_fold = [roc_auc_score(labels[i], folds[i]) for i in range(2)]
        recovered = roc_auc_score(L, _rank_normalize(folds))
        assert recovered == pytest.approx(np.mean(per_fold), abs=0.01)

    def test_rank_normalisation_cannot_change_a_within_fold_auc(self):
        """It is monotone inside each fold, so it cannot manufacture skill."""
        folds, labels = self._two_folds(offset=4.0)
        for i in range(2):
            p = np.asarray(folds[i])
            ranked = (p.argsort().argsort() + 0.5) / len(p)
            assert roc_auc_score(labels[i], ranked) == pytest.approx(
                roc_auc_score(labels[i], p), abs=1e-9
            )

    def test_well_matched_folds_show_no_drift(self):
        """No false alarm when the fold models agree on scale."""
        folds, labels = self._two_folds(offset=0.0)
        L = np.concatenate(labels)
        pooled = roc_auc_score(L, np.concatenate(folds))
        recovered = roc_auc_score(L, _rank_normalize(folds))
        assert abs(recovered - pooled) < 0.01


class TestStackedNumberIsNotTheModelNumber:
    def test_stage_stack_no_longer_overwrites_pooled_auc(self):
        """The one-line defect: the stacked score took over the model's field."""
        import inspect

        from rockfish import run_disordernet

        src = inspect.getsource(run_disordernet.stage_stack)
        assert 'cv_summary["stacked_pooled_auc"]' in src
        assert 'cv_summary["pooled_auc"] =' not in src

    def test_eval_summary_names_both_figures(self):
        import inspect

        from rockfish import run_disordernet

        src = inspect.getsource(run_disordernet.stage_eval)
        for key in ("final_pooled_auc", "model_pooled_auc", "final_components"):
            assert key in src, key

    def test_headline_does_not_attribute_the_stack_to_the_gpu_model(self):
        from colab.phase3_synthesis import _build_headline

        headline = _build_headline(
            {"our_auc": 0.788, "rank_among_published": 8, "n_methods": 10,
             "beats_af3_plddt": False},
            {}, {},
        )
        assert "GPU AUC" not in headline
        assert "GBDT" in headline or "stacked" in headline.lower()
