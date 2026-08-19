"""The guarantee has to hold, and the set semantics have to be stated right.

Split conformal's promise is finite-sample and distribution-free: with a
calibration sample exchangeable with the test sample, coverage is at least
`1-alpha` for every `n`, whatever the model does. Both halves of that are
tested — the guarantee under a *deliberately terrible* model, where validity is
the only thing left, and the finite-sample correction that makes it exact
rather than asymptotic.

The set semantics are tested because they are easy to state backwards. With
score `1-p_y` and threshold `q < 0.5`, the uninformative outcome is the **empty**
set, not `{both}`; `{both}` needs `q >= 0.5`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from colab.conformal import (  # noqa: E402
    calibrate,
    conformal_quantile,
    evaluate_sets,
    prediction_sets,
    split_by_chain,
)


def synthetic(n=40000, prevalence=0.27, skill=0.32, noise=0.18, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(np.int8)
    p = np.clip(0.5 + skill * (2 * y - 1) + rng.normal(0, noise, n),
                1e-6, 1 - 1e-6)
    return p, y


class TestTheGuaranteeHolds:
    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
    def test_marginal_coverage_reaches_the_level(self, alpha):
        p, y = synthetic(seed=1)
        h = p.size // 2
        cal = calibrate(p[:h], y[:h], alpha, class_conditional=False)
        e = evaluate_sets(prediction_sets(p[h:], cal), y[h:])
        assert e["coverage"] >= 1.0 - alpha - 0.02, (alpha, e["coverage"])

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
    def test_class_conditional_coverage_holds_within_each_class(self, alpha):
        """The one that matters at 27% prevalence: a marginal guarantee can be
        satisfied by covering the majority and abandoning the minority."""
        p, y = synthetic(seed=2)
        h = p.size // 2
        cal = calibrate(p[:h], y[:h], alpha, class_conditional=True)
        e = evaluate_sets(prediction_sets(p[h:], cal), y[h:])
        for c in (0, 1):
            assert e[f"coverage_class{c}"] >= 1.0 - alpha - 0.03, (alpha, c, e)

    def test_it_holds_for_a_useless_model(self):
        """Validity is free — `PredictionSets.validity_is_free`. A model with
        no signal must still cover; what it loses is informativeness."""
        rng = np.random.default_rng(3)
        n = 20000
        y = (rng.random(n) < 0.27).astype(np.int8)
        p = rng.random(n)                     # pure noise
        h = n // 2
        cal = calibrate(p[:h], y[:h], 0.10, class_conditional=True)
        e = evaluate_sets(prediction_sets(p[h:], cal), y[h:])
        assert e["coverage"] >= 0.87
        assert e["singleton_rate"] < 0.75, (
            "a signal-free model should not be able to call most residues")

    def test_the_singleton_rate_is_not_monotone_in_model_quality(self):
        """A trap worth pinning rather than discovering in a comparison.

        A sharper model concentrates its scores, so the calibrated threshold
        `q` shrinks; but the empty-set band is exactly `q < p < 1-q`, so a
        smaller `q` *widens* it. Two effects pull opposite ways and the
        callable fraction is not monotone in skill: measured 0.542, 0.966,
        0.902 at skill 0.10, 0.25, 0.40.

        So the singleton rate says how much of *this* data is decidable by
        *this* model at *this* level. It does not rank models, and reporting it
        as though it did would be wrong in a way no coverage check catches.
        """
        h = 20000
        rates = []
        for skill in (0.10, 0.25, 0.40):
            p, y = synthetic(n=2 * h, skill=skill, seed=4)
            cal = calibrate(p[:h], y[:h], 0.10, class_conditional=True)
            e = evaluate_sets(prediction_sets(p[h:], cal), y[h:])
            rates.append(e["singleton_rate"])
        assert rates[0] < rates[1], "a signal-free model should call less"
        assert rates != sorted(rates), (
            "if this ever becomes monotone the caveat above can be dropped")

    def test_singleton_accuracy_does_rank_models(self):
        """What the singleton rate cannot do, this can: among residues that
        received a call, how often the call was right."""
        h = 20000
        acc = []
        for skill in (0.10, 0.25, 0.40):
            p, y = synthetic(n=2 * h, skill=skill, seed=4)
            cal = calibrate(p[:h], y[:h], 0.10, class_conditional=True)
            e = evaluate_sets(prediction_sets(p[h:], cal), y[h:])
            acc.append(e["singleton_accuracy"])
        assert acc == sorted(acc), acc


class TestTheFiniteSampleCorrection:
    def test_the_quantile_uses_n_plus_one(self):
        scores = np.linspace(0.0, 1.0, 100)
        # ceil(101 * 0.9) = 91 -> the 91st smallest, index 90
        assert conformal_quantile(scores, 0.10) == pytest.approx(scores[90])

    def test_an_unachievable_level_returns_infinity(self):
        """With n points no level above 1 - 1/(n+1) is achievable. Returning
        infinity makes every set the full label space — vacuous and valid —
        rather than silently returning the largest score."""
        assert conformal_quantile(np.array([0.1, 0.2, 0.3]), 0.01) == \
            float("inf")
        assert conformal_quantile(np.array([]), 0.10) == float("inf")

    def test_an_infinite_threshold_admits_every_class(self):
        cal = {"q": {0: float("inf"), 1: float("inf")}}
        s = prediction_sets(np.array([0.01, 0.5, 0.99]), cal)
        assert s.all()

    def test_small_calibration_samples_still_cover(self):
        """The correction earns its keep here: at n=40 an uncorrected quantile
        undercovers."""
        got = []
        for seed in range(40):
            p, y = synthetic(n=40 + 4000, seed=100 + seed)
            cal = calibrate(p[:40], y[:40], 0.10, class_conditional=False)
            got.append(evaluate_sets(prediction_sets(p[40:], cal),
                                     y[40:])["coverage"])
        assert float(np.mean(got)) >= 0.90, float(np.mean(got))


class TestTheSetSemanticsAreStatedRight:
    def test_below_half_the_uninformative_outcome_is_the_empty_set(self):
        cal = {"q": {0: 0.3, 1: 0.3}}
        s = prediction_sets(np.array([0.5]), cal)     # p_1 = 0.5, both scores 0.5 > 0.3
        assert s.sum() == 0
        assert not s.any()

    def test_above_half_both_classes_can_appear(self):
        cal = {"q": {0: 0.6, 1: 0.6}}
        s = prediction_sets(np.array([0.5]), cal)
        assert s.sum() == 2

    def test_a_confident_correct_call_is_a_singleton(self):
        cal = {"q": {0: 0.3, 1: 0.3}}
        s = prediction_sets(np.array([0.95, 0.05]), cal)
        assert s[0].tolist() == [False, True]
        assert s[1].tolist() == [True, False]

    def test_empty_sets_count_as_misses(self):
        sets = np.zeros((4, 2), dtype=bool)
        labels = np.array([0, 1, 0, 1], np.int8)
        e = evaluate_sets(sets, labels)
        assert e["coverage"] == 0.0
        assert e["empty_rate"] == 1.0
        assert e["singleton_rate"] == 0.0

    def test_the_docstring_states_the_empty_case(self):
        src = open(os.path.join(REPO, "colab", "conformal.py")).read()
        assert "the uninformative outcome is then the **empty" in src
        assert "Only at `q >= 0.5`" in src


class TestExchangeabilityIsRespected:
    def test_the_split_is_by_chain_not_by_residue(self):
        """Residues within a chain share a protein, a fold, a construct and an
        experiment. A residue-level split would make calibration and test agree
        for reasons that have nothing to do with the model."""
        chains = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50 + ["D"] * 50)
        cal, test = split_by_chain(chains, np.random.default_rng(0))
        for c in ("A", "B", "C", "D"):
            m = chains == c
            assert cal[m].all() or test[m].all(), f"chain {c} was split"

    def test_both_halves_are_non_empty(self):
        chains = np.array([f"C{i//10}" for i in range(200)])
        cal, test = split_by_chain(chains, np.random.default_rng(1))
        assert cal.sum() > 0 and test.sum() > 0
        assert cal.sum() + test.sum() == len(chains)

    def test_a_single_chain_still_yields_a_calibration_half(self):
        chains = np.array(["only"] * 20)
        cal, test = split_by_chain(chains, np.random.default_rng(2))
        assert cal.all() and not test.any()


class TestConformalRiskControlAtTheChainLevel:
    """The valid guarantee, at the level where exchangeability holds.

    Split conformal's promise is per residue and residues are not exchangeable.
    Conformal risk control moves the promise to the chain, which is what the
    chain-level split makes exchangeable, and controls the expected per-chain
    miss rate instead of per-residue coverage.
    """

    @staticmethod
    def _chains(n_chains=200, seed=0, skill=0.32):
        rng = np.random.default_rng(seed)
        P, Y = [], []
        for _ in range(n_chains):
            n = int(rng.integers(60, 400))
            y = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(np.int8)
            p = np.clip(0.5 + skill * (2 * y - 1) + rng.normal(0, 0.18, n),
                        1e-6, 1 - 1e-6)
            P.append(p)
            Y.append(y)
        return P, Y

    def test_the_loss_is_monotone_in_the_threshold(self):
        """Conformal risk control requires it; without monotonicity the search
        is not a search."""
        from colab.conformal import chain_miss_rate

        P, Y = self._chains(n_chains=5, seed=1)
        for p, y in zip(P, Y):
            losses = [chain_miss_rate(p, y, t)
                      for t in np.linspace(1.0, 0.0, 60)]
            assert losses == sorted(losses, reverse=True), losses[:8]

    def test_the_loss_is_bounded_in_the_unit_interval(self):
        from colab.conformal import chain_miss_rate

        P, Y = self._chains(n_chains=20, seed=2)
        for p, y in zip(P, Y):
            for t in (0.0, 0.3, 0.7, 1.0):
                assert 0.0 <= chain_miss_rate(p, y, t) <= 1.0

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
    def test_the_realised_chain_risk_is_controlled(self, alpha):
        from colab.conformal import control_chain_risk, evaluate_chain_risk

        P, Y = self._chains(n_chains=400, seed=3)
        h = len(P) // 2
        got = control_chain_risk(P[:h], Y[:h], alpha)
        e = evaluate_chain_risk(P[h:], Y[h:], got["threshold"])
        assert e["mean_chain_miss_rate"] <= alpha + 0.03, (alpha, e)

    def test_it_holds_for_a_signal_free_model(self):
        """Validity must not depend on the model, only informativeness."""
        from colab.conformal import control_chain_risk, evaluate_chain_risk

        P, Y = self._chains(n_chains=400, seed=4, skill=0.0)
        h = len(P) // 2
        got = control_chain_risk(P[:h], Y[:h], 0.10)
        e = evaluate_chain_risk(P[h:], Y[h:], got["threshold"])
        assert e["mean_chain_miss_rate"] <= 0.13
        # And it pays for it by calling almost everything disordered.
        assert e["mean_fraction_called_disordered"] > 0.6

    def test_a_better_model_calls_less_at_the_same_risk(self):
        """This is the comparison the singleton rate could not support: at a
        fixed guaranteed miss rate, a better model flags fewer residues."""
        from colab.conformal import control_chain_risk, evaluate_chain_risk

        called = []
        for skill in (0.05, 0.20, 0.40):
            P, Y = self._chains(n_chains=400, seed=5, skill=skill)
            h = len(P) // 2
            got = control_chain_risk(P[:h], Y[:h], 0.10)
            e = evaluate_chain_risk(P[h:], Y[h:], got["threshold"])
            called.append(e["mean_fraction_called_disordered"])
        assert called == sorted(called, reverse=True), called

    def test_a_chain_with_no_disordered_residue_contributes_no_loss(self):
        from colab.conformal import chain_miss_rate

        assert chain_miss_rate(np.array([0.9, 0.9]), np.array([0, 0]), 0.5) \
            == 0.0

    def test_an_empty_calibration_set_refuses(self):
        from colab.conformal import control_chain_risk

        got = control_chain_risk([], [], 0.10)
        assert "reason" in got


class TestTheCalibrationInvariantOperatingCost:
    """Two ways to spend the same guarantee, separating the same two abilities
    the AUC decomposition does.

    A global threshold uses discrimination *and* calibration. A per-protein
    quantile uses discrimination alone, because it depends only on the ordering
    inside each chain — the same invariance `auc_within_strictMono_invariant`
    states for AUC_within. The difference between the two costs is what
    calibration is worth operationally.
    """

    @staticmethod
    def _chains(n_chains=200, seed=0, skill=0.32, offset=0.0):
        rng = np.random.default_rng(seed)
        P, Y = [], []
        for _ in range(n_chains):
            n = int(rng.integers(60, 400))
            y = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(np.int8)
            p = 0.5 + skill * (2 * y - 1) + rng.normal(0, 0.18, n)
            if offset:
                p = p + rng.normal(0, offset)      # per-chain miscalibration
            P.append(p)
            Y.append(y)
        return P, Y

    def test_the_quantile_rule_is_invariant_under_per_chain_recalibration(self):
        from colab.conformal import chain_quantile_miss_rate

        rng = np.random.default_rng(0)
        P, Y = self._chains(n_chains=30, seed=1)
        maps = [lambda v, k: v * (0.2 + 3 * k),
                lambda v, k: np.exp(v / (1 + k)),
                lambda v, k: v ** 3 + 5 * k,
                lambda v, k: np.arcsinh(v) + k]
        for i, (p, y) in enumerate(zip(P, Y)):
            k = float(rng.integers(0, 4))
            recal = maps[i % len(maps)](p, k)
            for q in (0.1, 0.3, 0.5, 0.8):
                assert chain_quantile_miss_rate(p, y, q) == pytest.approx(
                    chain_quantile_miss_rate(recal, y, q), abs=1e-12)

    def test_the_global_rule_is_not_invariant(self):
        """Guard the guard: if both were invariant the pair would measure
        nothing and the difference between them would be identically zero."""
        from colab.conformal import chain_miss_rate

        p = np.array([0.2, 0.4, 0.6, 0.8])
        y = np.array([0, 0, 1, 1], np.int8)
        assert chain_miss_rate(p, y, 0.5) != chain_miss_rate(p - 0.3, y, 0.5)

    def test_the_quantile_loss_is_monotone(self):
        from colab.conformal import chain_quantile_miss_rate

        P, Y = self._chains(n_chains=5, seed=2)
        for p, y in zip(P, Y):
            losses = [chain_quantile_miss_rate(p, y, q)
                      for q in np.linspace(0.0, 1.0, 50)]
            assert losses == sorted(losses, reverse=True)

    @pytest.mark.parametrize("alpha", [0.10, 0.05])
    def test_the_quantile_guarantee_holds(self, alpha):
        from colab.conformal import (control_chain_risk_quantile,
                                     evaluate_chain_risk_quantile)

        P, Y = self._chains(n_chains=400, seed=3)
        h = len(P) // 2
        got = control_chain_risk_quantile(P[:h], Y[:h], alpha)
        e = evaluate_chain_risk_quantile(P[h:], Y[h:], got["q"])
        assert e["mean_chain_miss_rate"] <= alpha + 0.03, (alpha, e)

    def test_miscalibration_costs_the_global_rule_and_not_the_quantile_rule(self):
        """The prediction the pair exists to test. Adding per-chain offsets
        leaves within-chain ordering untouched, so the quantile cost must not
        move while the global cost must get worse."""
        from colab.conformal import (control_chain_risk,
                                     control_chain_risk_quantile,
                                     evaluate_chain_risk,
                                     evaluate_chain_risk_quantile)

        costs_global, costs_quantile = [], []
        for offset in (0.0, 0.5):
            P, Y = self._chains(n_chains=400, seed=4, offset=offset)
            h = len(P) // 2
            g = control_chain_risk(P[:h], Y[:h], 0.10)
            costs_global.append(evaluate_chain_risk(
                P[h:], Y[h:], g["threshold"])["mean_fraction_called_disordered"])
            q = control_chain_risk_quantile(P[:h], Y[:h], 0.10)
            costs_quantile.append(evaluate_chain_risk_quantile(
                P[h:], Y[h:], q["q"])["mean_fraction_called_disordered"])
        assert costs_global[1] > costs_global[0] + 0.02, costs_global
        assert abs(costs_quantile[1] - costs_quantile[0]) < 0.05, costs_quantile
