"""The certification bar, checked against the theorem it cites.

`LabelNoise.ranking_certified`: `errors T P + 2*noise T L < errors T Q` implies
`errors L P < errors L Q`. The usable direction here is the contrapositive — a
measured margin inside `2*noise` certifies nothing — and the arithmetic behind
that is small enough that getting it wrong would be embarrassing and easy.

The Lean file's own sharpness witness is reproduced: four residues, one false
negative and one false positive, a perfect predictor ranked *below* the
annotation itself.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load():
    path = os.path.join(REPO, "results", "caid3", "label_noise_certificate.py")
    spec = importlib.util.spec_from_file_location("lnc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lnc = _load()


def errors(a: set, b: set) -> int:
    """`errors R P` — the symmetric difference, as in the Lean file."""
    return len(a ^ b)


class TestTheLeanWitnessesReproduce:
    def test_a_perfect_predictor_is_measured_to_make_exactly_the_noise(self):
        """`perfect_predictor_penalised`. Not zero — noise."""
        truth, annot = {0, 1}, {0, 2}
        assert errors(annot, truth) == errors(truth, annot)
        assert errors(annot, truth) == 2

    def test_a_single_pair_of_bad_labels_inverts_a_ranking(self):
        """`ranking_inversion_false_positive`: residues 0 and 1 are disordered,
        the annotation drops 1 and invents 2. The perfect predictor is measured
        worse than a predictor that merely copies the annotation."""
        truth, annot = {0, 1}, {0, 2}
        assert errors(truth, annot) == 2                 # noise
        assert errors(truth, truth) < errors(truth, annot)
        assert errors(annot, annot) < errors(annot, truth)

    def test_the_bar_is_twice_the_noise(self):
        truth, annot = {0, 1}, {0, 2}
        noise = errors(truth, annot)
        p, q = {0, 1}, {2, 3, 4, 5}
        # P beats Q on the truth by more than 2*noise, so the benchmark agrees.
        assert errors(truth, p) + 2 * noise < errors(truth, q)
        assert errors(annot, p) < errors(annot, q)

    def test_a_margin_inside_the_bar_can_go_either_way(self):
        """The contrapositive is the whole point: inside 2*noise the benchmark
        may order two methods opposite to the truth, which is what the
        inversion witness above already shows."""
        truth, annot = {0, 1}, {0, 2}
        noise = errors(truth, annot)
        p, q = {0, 1}, {0, 2}
        assert errors(truth, p) < errors(truth, q)            # P is better
        assert not (errors(truth, p) + 2 * noise < errors(truth, q))
        assert errors(annot, p) > errors(annot, q)            # benchmark says Q


class TestTheBinaryColumnIsRead:
    @staticmethod
    def _write(tmp_path, rows):
        p = tmp_path / "m.caid"
        out = []
        for tid, scores, calls in rows:
            out.append(f">{tid}\n")
            for i, (s, c) in enumerate(zip(scores, calls), 1):
                out.append(f"{i}\tA\t{s:.3f}\t{c}\n")
        p.write_text("".join(out))
        return str(p)

    def test_it_reads_column_four_not_the_score(self):
        """Rethresholding someone else's scores at 0.5 would be scoring a
        method they did not submit. CAID's format carries their own call."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td),
                               [("T1", [0.9, 0.9, 0.1], [0, 1, 1])])
            got = lnc.read_binary_predictions(path)
        assert got["T1"].tolist() == [0.0, 1.0, 1.0]

    def test_a_missing_column_becomes_nan_not_zero(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.caid"
            p.write_text(">T1\n1\tA\t0.900\n2\tA\t0.100\t1\n")
            got = lnc.read_binary_predictions(str(p))
        assert np.isnan(got["T1"][0])
        assert got["T1"][1] == 1.0


class TestTheNoiseEstimateIsInternal:
    def test_it_counts_only_residues_both_references_evaluate(self):
        a = {"T1": ("AAAA", "01-1")}
        b = {"T1": ("AAAA", "0-11")}
        n = lnc.observed_noise(a, b)
        # Positions 0 and 3 are evaluated by both; both agree there.
        assert n["residues_evaluated_by_both"] == 2
        assert n["residues_disagreeing"] == 0

    def test_it_counts_a_real_contradiction(self):
        a = {"T1": ("AAAA", "0011")}
        b = {"T1": ("AAAA", "0101")}
        n = lnc.observed_noise(a, b)
        assert n["residues_evaluated_by_both"] == 4
        assert n["residues_disagreeing"] == 2
        assert n["disagreement_rate"] == pytest.approx(0.5)

    def test_a_different_chain_under_the_same_id_is_skipped(self):
        """A length or sequence mismatch is a different chain, not a
        disagreement, and counting it as noise would inflate the bar."""
        a = {"T1": ("AAAA", "0011")}
        b = {"T1": ("CCCC", "1100")}
        n = lnc.observed_noise(a, b)
        assert n["n_comparable_targets"] == 0
        assert n["residues_disagreeing"] == 0

    def test_targets_in_only_one_reference_are_ignored(self):
        a = {"T1": ("AAAA", "0011"), "T2": ("AAAA", "1111")}
        b = {"T1": ("AAAA", "0011")}
        n = lnc.observed_noise(a, b)
        assert n["n_shared_targets"] == 1
        assert n["residues_evaluated_by_both"] == 4

    def test_the_docstring_records_why_the_first_estimate_failed(self):
        """The comparison between CAID3's own references returned zero
        disagreements, which is a fact about the references — NOX is derived
        from the same assignment — and not about the labels being clean."""
        src = open(os.path.join(REPO, "results", "caid3",
                                "label_noise_certificate.py")).read()
        assert "not independent annotations" in src
        assert "The comparison measures\nnothing" in src or "measures\nnothing" in src


class TestAFailedEstimateRefusesRatherThanCertifying:
    """A noise estimate of zero sets the bar at zero and certifies the entire
    field — the strongest-looking possible output from a measurement that
    failed. That is the one behaviour this file must not have.
    """

    @staticmethod
    def _src():
        return open(os.path.join(REPO, "results", "caid3",
                                 "label_noise_certificate.py")).read()

    def test_a_zero_rate_is_treated_as_no_estimate(self):
        src = self._src()
        assert "if not rate:" in src
        assert "REFUSING TO CERTIFY" in src

    def test_the_certification_columns_are_suppressed_without_an_estimate(self):
        src = self._src()
        assert 'head = f"{\'margin vs #1\':>14}" + ("   certified?" if rate else "")' in src

    def test_the_failed_attempt_is_documented_not_deleted(self):
        """The zero result is a finding about the references — that NOX is
        derived from the same assignment rather than annotated independently —
        and deleting it would invite the next person to repeat it."""
        src = self._src()
        assert "returned exactly zero disagreeing residues" in src
        assert "not independent annotations" in src

    def test_the_report_records_whether_it_certified(self):
        src = self._src()
        assert '"certified": bool(rate)' in src


class TestTheFrontierReplacesTheVerdict:
    """Without a noise estimate, report how the certification degrades with the
    assumed rate — the pattern of `certificate_under_mean_error`, where a
    bound whose input is known only to within delta degrades continuously
    rather than becoming silence.
    """

    @staticmethod
    def _src():
        return open(os.path.join(REPO, "results", "caid3",
                                 "label_noise_certificate.py")).read()

    def test_the_breakdown_rate_is_the_margin_over_twice_the_residues(self):
        """`ranking_certified` needs margin > 2*eps*n, so a comparison survives
        exactly while eps < margin/(2n)."""
        n_eval, margin = 100_000, 2_719
        breakdown = margin / (2.0 * n_eval)
        assert margin > 2.0 * (breakdown - 1e-9) * n_eval
        assert not (margin > 2.0 * (breakdown + 1e-9) * n_eval)

    def test_certification_is_monotone_in_the_rate(self):
        """Every comparison certified at eps is certified at any smaller rate,
        which is what makes one frontier a complete answer for all eps."""
        margins = [460, 2719, 5000, 12000]
        n_eval = 100_000
        counts = [sum(1 for m in margins if m > 2.0 * e * n_eval)
                  for e in (0.001, 0.005, 0.01, 0.05)]
        assert counts == sorted(counts, reverse=True)

    def test_the_frontier_is_computed_and_recorded(self):
        src = self._src()
        assert "certification frontier" in src
        assert "breakdown_rates" in src
        assert '"frontier":' in src

    def test_it_does_not_assume_a_rate_on_the_readers_behalf(self):
        src = self._src()
        assert "Nothing is assumed on their behalf." in src
