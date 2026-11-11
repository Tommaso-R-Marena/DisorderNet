"""The theorems this analysis cites, checked numerically against the code.

Every claim in `certified_leaderboard.py` names a machine-checked Lean theorem.
That is only worth something if the Python agrees with the Lean, so these tests
check the statements directly on data rather than checking that the docstrings
mention them.

The load-bearing one is `auc_within_strictMono_invariant`: AUC_within is
invariant under *any* per-protein strictly monotone recalibration, not merely
an additive shift. That is what makes "within-protein AUC" the calibration-free
part of CAID's statistic rather than a metaphor, and it is the premise of the
whole within-protein leaderboard.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from colab.auc_decomposition import decompose_auc  # noqa: E402
from colab.bias_ceiling import bias_bound, ceiling_attainable  # noqa: E402


def _load():
    path = os.path.join(REPO, "results", "caid3", "certified_leaderboard.py")
    spec = importlib.util.spec_from_file_location("cert", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cert = _load()


def synthetic(n_proteins=14, seed=0, skill=1.0):
    rng = np.random.default_rng(seed)
    ys, ss = [], []
    for _ in range(n_proteins):
        n = int(rng.integers(30, 90))
        y = (rng.random(n) < 0.35).astype(np.int8)
        y[0], y[1] = 1, 0
        s = rng.normal(size=n) + skill * y + rng.normal(scale=2.0)   # offset
        ys.append(y)
        ss.append(s)
    return ys, ss


class TestWithinIsTheCalibrationFreePart:
    def test_an_additive_per_protein_shift_leaves_within_exactly_alone(self):
        ys, ss = synthetic(seed=1)
        rng = np.random.default_rng(2)
        shifted = [s + rng.normal(scale=5.0) for s in ss]
        a = decompose_auc(ys, ss)["auc_within"]
        b = decompose_auc(ys, shifted)["auc_within"]
        assert a == pytest.approx(b, abs=1e-12)

    def test_any_strictly_monotone_per_protein_map_leaves_it_alone(self):
        """The strong form. Not just shifts: exponentials, cubes, logistic
        squashes, each different per protein."""
        ys, ss = synthetic(seed=3)
        rng = np.random.default_rng(4)
        maps = [
            lambda v, k: v * (0.2 + 3.0 * k),
            lambda v, k: np.exp(v / (1.0 + k)),
            lambda v, k: v ** 3 + 7.0 * k,
            lambda v, k: 1.0 / (1.0 + np.exp(-(v - k))),
            lambda v, k: np.arcsinh(v) + k,
        ]
        recal = [maps[i % len(maps)](s, float(rng.integers(0, 5)))
                 for i, s in enumerate(ss)]
        a = decompose_auc(ys, ss)["auc_within"]
        b = decompose_auc(ys, recal)["auc_within"]
        assert a == pytest.approx(b, abs=1e-12), (a, b)

    def test_a_decreasing_map_does_not_count(self):
        """Guard the guard: the invariance is for *strictly increasing* maps.
        If a decreasing one also left it alone, the test above would be
        checking nothing."""
        ys, ss = synthetic(seed=5)
        flipped = [-s for s in ss]
        a = decompose_auc(ys, ss)["auc_within"]
        b = decompose_auc(ys, flipped)["auc_within"]
        assert abs(a - b) > 0.1

    def test_pooled_is_not_invariant(self):
        """The complement, and the reason the decomposition is interesting:
        everything a per-protein recalibration can change lives in the between
        term."""
        ys, ss = synthetic(seed=6)
        rng = np.random.default_rng(7)
        shifted = [s + rng.normal(scale=5.0) for s in ss]
        a = decompose_auc(ys, ss)
        b = decompose_auc(ys, shifted)
        assert abs(a["pooled"] - b["pooled"]) > 1e-6
        # auc_pooled_shift_diff: the whole difference is w_between times the
        # between-protein difference.
        assert (b["pooled"] - a["pooled"]) == pytest.approx(
            a["w_between"] * (b["auc_between"] - a["auc_between"]), abs=1e-10)


class TestTheCeilingIsAnUpperBound:
    def test_no_random_bias_ever_exceeds_the_ceiling(self):
        ys, ss = synthetic(seed=8)
        d = decompose_auc(ys, ss)
        ceiling = d["w_within"] * d["auc_within"] + d["w_between"]
        rng = np.random.default_rng(9)
        for _ in range(300):
            b = rng.normal(scale=3.0, size=len(ss))
            got = decompose_auc(ys, [s + bi for s, bi in zip(ss, b)])["pooled"]
            assert got <= ceiling + 1e-12, (got, ceiling)

    def test_the_headroom_is_exactly_w_between_times_one_minus_between(self):
        ys, ss = synthetic(seed=10)
        d = decompose_auc(ys, ss)
        ceiling = d["w_within"] * d["auc_within"] + d["w_between"]
        assert ceiling - d["pooled"] == pytest.approx(
            d["w_between"] * (1.0 - d["auc_between"]), abs=1e-12)

    def test_the_upper_bound_holds_under_search_when_unattainable(self):
        """When the ceiling is unreachable the crossed-matching bound is what
        binds, and no bias may cross it."""
        ys, ss = synthetic(seed=11, skill=0.3)
        att = ceiling_attainable(ys, ss)
        if att["attainable"]:
            pytest.skip("this instance's ceiling is attainable, so the "
                        "crossed-matching bound is not the binding one")
        bound = bias_bound(ys, ss)["upper_bound"]
        rng = np.random.default_rng(12)
        best = max(
            decompose_auc(ys, [s + b for s, b in
                               zip(ss, rng.normal(scale=3.0, size=len(ss)))]
                          )["pooled"]
            for _ in range(400))
        assert best <= bound + 1e-12, (best, bound)

    def test_attainability_agrees_with_the_two_largest_overlaps(self):
        ys, ss = synthetic(seed=13)
        att = ceiling_attainable(ys, ss)
        ov = sorted(
            float(np.asarray(s)[np.asarray(y) == 0].max()
                  - np.asarray(s)[np.asarray(y) == 1].min())
            for y, s in zip(ys, ss))[::-1]
        assert att["attainable"] == bool(ov[0] + ov[1] < 0)


class TestTheInversionCertificate:
    @staticmethod
    def _row(ys, ss):
        return cert.certify(ys, ss)

    def test_the_lean_worked_example_reproduces(self):
        """`Example.inversion`: five residues, two proteins, two predictors
        whose pooled and within-protein orderings disagree. The Lean file gives
        exact values; reproducing them is the strongest check that this code
        computes the same statistic the theorems are about."""
        ys = [np.array([1, 1, 0], np.int8), np.array([1, 0], np.int8)]
        a = [np.array([1.0, 2.0, 0.0]), np.array([4.0, 3.0])]
        b = [np.array([1.0, -1.0, 0.0]), np.array([3.0, -2.0])]
        da, db = decompose_auc(ys, a), decompose_auc(ys, b)
        assert da["auc_within"] == pytest.approx(1.0)
        assert da["pooled"] == pytest.approx(2.0 / 3.0)
        assert db["auc_within"] == pytest.approx(2.0 / 3.0)
        assert db["pooled"] == pytest.approx(5.0 / 6.0)
        assert da["auc_within"] > db["auc_within"]
        assert da["pooled"] < db["pooled"]

    def test_the_certificate_fires_exactly_on_an_inversion(self):
        ys = [np.array([1, 1, 0], np.int8), np.array([1, 0], np.int8)]
        ra = cert.certify(ys, [np.array([1.0, 2.0, 0.0]),
                               np.array([4.0, 3.0])])
        rb = cert.certify(ys, [np.array([1.0, -1.0, 0.0]),
                               np.array([3.0, -2.0])])
        c = cert.inversion_certificate(ra, rb)
        assert c is not None
        assert c["iff_holds"]
        assert c["between_gap_measured"] > c["between_gap_required"]
        assert c["slack"] > 0
        # And it does not fire the other way round.
        assert cert.inversion_certificate(rb, ra) is None

    def test_the_required_gap_matches_the_theorem(self):
        ys, ss = synthetic(seed=14)
        # A second scorer over the *same* labels — different lengths would make
        # these two decompositions of different quantities.
        rng = np.random.default_rng(15)
        tt = [rng.normal(size=len(y)) + 0.4 * y + rng.normal(scale=2.0)
              for y in ys]
        ra, rb = cert.certify(ys, ss), cert.certify(ys, tt)
        for x, y in ((ra, rb), (rb, ra)):
            c = cert.inversion_certificate(x, y)
            if c is None:
                continue
            assert c["between_gap_required"] == pytest.approx(
                (x["w_within"] / x["w_between"]) * c["within_gap"], abs=1e-12)
            assert c["iff_holds"] == bool(
                x["w_within"] * c["within_gap"]
                < x["w_between"] * c["between_gap_measured"])


class TestEligibility:
    def test_a_single_class_target_is_excluded(self):
        ref = {"a": ("AAAA", "0011"), "b": ("AAAA", "0000")}
        assert cert.two_class_targets(ref) == ["a"]

    def test_a_method_missing_a_target_yields_nothing(self):
        ref = {"a": ("AAAA", "0011"), "b": ("AAAAA", "00111")}
        assert cert.target_arrays(ref, {"a": np.zeros(4)}, ["a", "b"]) is None
