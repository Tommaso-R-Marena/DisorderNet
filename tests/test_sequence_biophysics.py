"""Charge patterning and complexity descriptors — properties, not fits.

Every function is a deterministic map from sequence, so each has properties that
must hold exactly. SCD in particular is the only descriptor here sensitive to
the *arrangement* of charge rather than its amount, which is the whole reason it
might carry between-protein signal the composition measures do not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from colab.sequence_biophysics import (charges, descriptors, fcr_ncpr,
                                       kappa_like, local_descriptors,
                                       mean_hydropathy, scd, shannon_entropy)


class TestCharges:
    def test_signs_follow_the_residue(self):
        assert charges("KRDEA").tolist() == [1.0, 1.0, -1.0, -1.0, 0.0]

    def test_histidine_is_neutral_at_physiological_ph(self):
        assert charges("H").tolist() == [0.0]


class TestComposition:
    def test_fcr_counts_all_charges_and_ncpr_their_sum(self):
        fcr, ncpr = fcr_ncpr("KKDD")
        assert fcr == pytest.approx(1.0)
        assert ncpr == pytest.approx(0.0)

    def test_a_neutral_sequence_has_zero_of_both(self):
        assert fcr_ncpr("AAAA") == (pytest.approx(0.0), pytest.approx(0.0))


class TestSCDSeesArrangement:
    def test_identical_composition_different_order_differ(self):
        """The property that matters: FCR and NCPR cannot tell these apart."""
        mixed, blocky = "KDKDKDKD", "KKKKDDDD"
        assert fcr_ncpr(mixed) == fcr_ncpr(blocky)
        assert scd(mixed) != pytest.approx(scd(blocky))

    def test_segregated_charge_is_more_negative_than_mixed(self):
        """The opposite-charge pairs carry the negative sign and sit far apart
        in a blocky sequence, so they are weighted by the larger sqrt. Blocky
        polyampholytes are the compact ones, so SCD runs negatively with chain
        dimension — the Sawle-Ghosh direction. The first version of this test
        asserted the opposite and caught a sign error in the docstring."""
        assert scd("KKKKDDDD") < scd("KDKDKDKD")
        assert scd("KKKKDDDD") == pytest.approx(-2.0196, abs=1e-3)
        assert scd("KDKDKDKD") == pytest.approx(-0.4537, abs=1e-3)

    def test_too_few_charges_is_zero_not_an_error(self):
        assert scd("AAAAK") == 0.0
        assert scd("AAAAA") == 0.0

    def test_it_is_invariant_to_global_charge_inversion(self):
        """Swapping every K/R for D/E mirrors the sequence's electrostatics and
        SCD, being quadratic in charge, must be unchanged."""
        assert scd("KKDDKD") == pytest.approx(scd("DDKKDK"))


class TestKappaLike:
    def test_blocky_exceeds_mixed(self):
        assert kappa_like("KKKKKKDDDDDD") > kappa_like("KDKDKDKDKDKD")

    def test_uncharged_sequence_is_nan_not_zero(self):
        """Zero would read as 'perfectly mixed'; there is nothing to mix."""
        assert math.isnan(kappa_like("AAAAAAAAAA"))

    def test_a_short_sequence_is_nan(self):
        assert math.isnan(kappa_like("KD"))


class TestEntropy:
    def test_a_homopolymer_has_zero_entropy(self):
        assert shannon_entropy("AAAAAA") == pytest.approx(0.0)

    def test_it_is_bounded_by_log2_of_the_alphabet(self):
        assert shannon_entropy("ACDEFGHIKLMNPQRSTVWY") == pytest.approx(
            math.log2(20))

    def test_low_complexity_scores_below_diverse(self):
        assert shannon_entropy("QQQQQQQQPPPP") < shannon_entropy("ACDEFGHIKLMN")


class TestHydropathy:
    def test_isoleucine_is_the_most_hydrophobic(self):
        assert mean_hydropathy("I") > mean_hydropathy("A")
        assert mean_hydropathy("I") > mean_hydropathy("R")

    def test_unknown_residues_are_skipped(self):
        assert mean_hydropathy("IXI") == pytest.approx(mean_hydropathy("II"))


class TestDescriptorBundle:
    def test_every_descriptor_is_finite_on_a_real_sequence(self):
        d = descriptors("MEKSDLKRQLSDEEKKAPGGSTQPRTPKSKAEDLLQRRK")
        for k, v in d.items():
            assert not math.isnan(v), k

    def test_the_bundle_is_deterministic(self):
        s = "KRDESTAQPGGLLKRDE"
        assert descriptors(s) == descriptors(s)

    def test_local_descriptors_are_full_length(self):
        seq = "MEKSDLKRQLSDEEKKAPGGSTQPRTPKSKAEDLLQRRK"
        loc = local_descriptors(seq, window=11)
        for k, v in loc.items():
            assert len(v) == len(seq), k
            assert np.isfinite(v).all(), k

    def test_local_entropy_dips_inside_a_low_complexity_stretch(self):
        seq = "ACDEFGHIKL" + "Q" * 25 + "ACDEFGHIKL"
        ent = local_descriptors(seq, window=15)["entropy"]
        assert ent[22] < ent[2]
