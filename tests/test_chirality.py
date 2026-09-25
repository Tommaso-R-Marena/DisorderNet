"""Handedness is the one thing the existing structural channels cannot see.

The claim this module rests on is narrow and checkable: reflect a protein and
its solvent accessibility, its contact counts and AlphaFold's confidence in it
are all unchanged, while the backbone virtual torsion changes sign. If that
were false the channel would be redundant with what the model already reads.

Two sign traps were hit writing the dihedral, and only one of them was caught
by the mirror test — taking the first bond backwards rotates every angle by
180 degrees while leaving it perfectly odd under reflection. So the sign
convention is pinned against an ideal right-handed alpha-helix built from its
own helical parameters, and the formula is checked against an independent
implementation on random points. Neither check alone was sufficient.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from colab.chirality import (  # noqa: E402
    ca_virtual_torsion,
    dihedral,
    handedness,
    ideal_alpha_helix,
    mirror,
)
from colab.structure_rsa import contact_density  # noqa: E402


def _independent_dihedral(p0, p1, p2, p3):
    """The 'praxeolitic' formulation, written from a different decomposition.

    Deliberately not a copy of the implementation under test: it projects out
    the axis instead of taking cross products of bond pairs, so a sign error in
    one is very unlikely to be mirrored in the other.
    """
    b0 = -1.0 * (np.asarray(p1) - np.asarray(p0))
    b1 = np.asarray(p2) - np.asarray(p1)
    b2 = np.asarray(p3) - np.asarray(p2)
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


class TestTheDihedralIsRight:
    def test_it_agrees_with_an_independent_implementation(self):
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(2000):
            pts = rng.normal(size=(4, 3))
            a = float(dihedral(*pts))
            b = _independent_dihedral(*pts)
            worst = max(worst, abs(((a - b + 180.0) % 360.0) - 180.0))
        assert worst < 1e-8, f"{worst} degrees apart"

    def test_an_ideal_right_handed_alpha_helix_reads_plus_fifty(self):
        """The textbook CA virtual torsion of an alpha-helix. Pinned because a
        sign convention that is merely *odd* under reflection can still report
        every helix in the dataset backwards."""
        tau = ca_virtual_torsion(ideal_alpha_helix(12))
        assert np.nanmin(tau[1:9]) > 45.0
        assert np.nanmax(tau[1:9]) < 55.0

    def test_the_mirror_helix_reads_minus_fifty(self):
        tau = ca_virtual_torsion(ideal_alpha_helix(12, right_handed=False))
        assert -55.0 < float(np.nanmean(tau[1:9])) < -45.0

    def test_a_straight_chain_has_no_handedness(self):
        line = np.stack([np.arange(12) * 3.8, np.zeros(12), np.zeros(12)],
                        axis=1)
        tau = ca_virtual_torsion(line)
        assert np.allclose(tau[1:9], 0.0, atol=1e-6)

    def test_the_angle_stays_in_the_principal_branch(self):
        rng = np.random.default_rng(3)
        for _ in range(500):
            v = float(dihedral(*rng.normal(size=(4, 3))))
            assert -180.0 <= v <= 180.0


class TestOddUnderReflectionAndNothingElseIs:
    @staticmethod
    def _structure(seed=1, n=60):
        """A compact chain with real handedness: a helix with noise."""
        rng = np.random.default_rng(seed)
        base = ideal_alpha_helix(n)
        return base + rng.normal(scale=0.15, size=base.shape)

    def test_the_torsion_changes_sign_under_reflection(self):
        c = self._structure()
        a = ca_virtual_torsion(c)
        b = ca_virtual_torsion(mirror(c))
        ok = np.isfinite(a) & np.isfinite(b)
        assert ok.sum() > 10
        np.testing.assert_allclose(b[ok], -a[ok], atol=1e-6)

    def test_contact_density_does_not(self):
        """The channel is only worth adding if it is not already there."""
        c = self._structure()
        np.testing.assert_array_equal(contact_density(c),
                                      contact_density(mirror(c)))

    def test_pairwise_distances_do_not(self):
        """rsa and pLDDT are functions of the structure through quantities that
        are invariant under any isometry, proper or not. Distances stand in for
        them here: SASA needs Biopython, which is not installed off-cluster,
        and every mirror-invariance argument for it runs through distances."""
        c = self._structure()
        d1 = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        m = mirror(c)
        d2 = np.linalg.norm(m[:, None, :] - m[None, :, :], axis=-1)
        np.testing.assert_allclose(d1, d2, atol=1e-6)

    def test_reflection_is_improper_not_a_rotation(self):
        """Guard the guard: if `mirror` happened to be a rotation, every test
        above would pass while establishing nothing."""
        c = self._structure()
        m = mirror(c)
        # A rotation preserves the signed volume of any tetrahedron; a
        # reflection negates it.
        v1 = np.dot(np.cross(c[1] - c[0], c[2] - c[0]), c[3] - c[0])
        v2 = np.dot(np.cross(m[1] - m[0], m[2] - m[0]), m[3] - m[0])
        assert v1 * v2 < 0

    def test_a_proper_rotation_leaves_the_torsion_alone(self):
        """The complement: the torsion must be invariant under rotation, or it
        would be reporting the coordinate frame rather than the molecule."""
        from scipy.spatial.transform import Rotation

        c = self._structure()
        r = Rotation.from_rotvec([0.3, -1.1, 0.7]).as_matrix()
        rotated = c @ r.T + np.array([12.0, -4.0, 3.0])
        a, b = ca_virtual_torsion(c), ca_virtual_torsion(rotated)
        ok = np.isfinite(a) & np.isfinite(b)
        np.testing.assert_allclose(b[ok], a[ok], atol=1e-4)


class TestTheProfileIsHonestAboutWhatItDoesNotKnow:
    def test_the_ends_are_nan_not_zero(self):
        """Zero is a legitimate torsion — a planar trace. Filling the ends with
        it would assert 'planar' about residues that have no window, the same
        mistake as imputing rsa 0 for a protein with no structure, which reads
        as fully buried."""
        tau = ca_virtual_torsion(ideal_alpha_helix(10))
        assert np.isnan(tau[0])
        assert np.isnan(tau[-1]) and np.isnan(tau[-2])
        assert np.isfinite(tau[1:-2]).all()

    def test_a_chain_too_short_for_a_window_is_all_nan(self):
        assert np.isnan(ca_virtual_torsion(np.zeros((3, 3)))).all()
        assert len(ca_virtual_torsion(np.zeros((3, 3)))) == 3

    def test_a_missing_atom_poisons_only_its_own_windows(self):
        c = ideal_alpha_helix(20)
        c[10] = np.nan
        tau = ca_virtual_torsion(c)
        assert np.isnan(tau[8:11]).all()          # windows containing residue 10
        assert np.isfinite(tau[1:7]).all()
        assert np.isfinite(tau[12:18]).all()

    def test_the_length_always_matches_the_input(self):
        for n in (0, 1, 3, 4, 5, 50):
            assert len(ca_virtual_torsion(np.zeros((n, 3)))) == n


class TestHandednessIsContinuousAcrossTheWrap:
    def test_neighbouring_conformations_are_neighbouring_numbers(self):
        """+179 and -179 are adjacent conformations and 358 apart as angles.
        A model fed the raw angle would see a discontinuity in the middle of
        the extended region, which is where most disordered residues sit."""
        assert abs(float(handedness(179.0)) - float(handedness(-179.0))) < 0.05
        assert abs(179.0 - (-179.0)) > 350.0

    def test_it_is_still_odd(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(-180, 180, size=100)
        np.testing.assert_allclose(handedness(-x), -handedness(x), atol=1e-12)

    def test_it_is_largest_where_the_helices_are(self):
        """|sin| peaks at +/-90 and vanishes at 0 and +/-180, so it separates
        the right- and left-handed helical regions and stays near zero through
        the extended region where the two are least distinguishable."""
        assert abs(float(handedness(50.0))) > 0.7
        assert abs(float(handedness(0.0))) < 0.01
        assert abs(float(handedness(180.0))) < 0.01

    def test_it_stays_in_range(self):
        rng = np.random.default_rng(6)
        v = handedness(rng.uniform(-720, 720, size=1000))
        assert v.min() >= -1.0 and v.max() <= 1.0


class TestTheChannelReachesTheModel:
    """A channel that is computed and then dropped is worse than none.

    Every step between the mmCIF and the logit is a place where handedness can
    be silently lost — the parse, the cache, the window slice, the batch, the
    head. Each is checked, because "we added a chirality channel" is exactly
    the kind of claim that survives a broken pipeline.
    """

    TASKS = ("disorder_pdb", "disorder_nox", "linker", "binding", "binding_idr")

    def _head(self, chiral):
        from colab.lite_head import WIDE_DILATIONS, MultiTaskLiteHead

        return MultiTaskLiteHead(in_dim=32, tasks=self.TASKS, structure_dim=24,
                                 dilations=WIDE_DILATIONS, chiral=chiral).eval()

    @staticmethod
    def _inputs(n=40, b=2):
        import torch

        return dict(rsa=torch.rand(b, n), plddt=torch.rand(b, n) * 100,
                    structure_available=torch.ones(b, n),
                    contacts=torch.rand(b, n))

    def test_the_chiral_head_takes_two_more_channels(self):
        from colab.lite_head import StructureChannels

        assert self._head(True).structure.n_in == \
            StructureChannels.N_CHANNELS_CHIRAL
        assert self._head(False).structure.n_in == StructureChannels.N_CHANNELS

    def test_flipping_handedness_changes_the_prediction(self):
        import torch

        m = self._head(True)
        x = torch.randn(2, 40, 32)
        kw = self._inputs()
        with torch.no_grad():
            a = m(x, handedness=torch.full((2, 40), 0.9), **kw)["disorder_pdb"]
            b = m(x, handedness=torch.full((2, 40), -0.9), **kw)["disorder_pdb"]
        assert not torch.allclose(a, b, atol=1e-6)

    def test_a_non_chiral_head_ignores_it(self):
        """Otherwise an old checkpoint would silently change behaviour the
        moment the caller started passing the argument."""
        import torch

        m = self._head(False)
        x = torch.randn(2, 40, 32)
        kw = self._inputs()
        with torch.no_grad():
            a = m(x, handedness=torch.full((2, 40), 0.9), **kw)["disorder_pdb"]
            b = m(x, handedness=None, **kw)["disorder_pdb"]
        assert torch.allclose(a, b, atol=0, rtol=0)

    def test_absent_handedness_is_flagged_not_imputed(self):
        import torch

        from colab.lite_head import StructureChannels

        h = torch.full((1, 6), float("nan"))
        h[0, 2:4] = 0.0                      # a genuine planar backbone
        block = StructureChannels.assemble(
            None, None, None, length=6, batch=1,
            device=torch.device("cpu"), handedness=h, chiral=True)
        assert block[0, 5].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert block[0, 6].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], (
            "a real 0.0 torsion must be flagged present, and NaN absent")

    def test_the_batcher_pads_with_nan(self):
        import numpy as np

        from rockfish.train_multitask import structure_batch

        row = {"length": 3, "rsa": np.ones(3, np.float32),
               "plddt": np.ones(3, np.float32),
               "structure_available": np.ones(3, np.float32),
               "contacts": np.ones(3, np.float32),
               "handedness": np.array([0.0, 0.5, np.nan], np.float32)}
        _r, _p, _a, _c, hd = structure_batch([row], L=5,
                                             device=__import__("torch").device("cpu"))
        got = hd[0].tolist()
        assert got[0] == 0.0 and got[1] == 0.5
        assert all(g != g for g in got[2:]), f"padding was not NaN: {got}"

    def test_windowing_slices_handedness_with_everything_else(self):
        import numpy as np

        from rockfish.train_multitask import chunk_long_rows

        n = 2500
        row = {"id": "P", "sequence": "A" * n, "length": n,
               "task_labels": {"disorder_pdb": np.zeros(n, np.int8)},
               "task_evidence": {"disorder_pdb": np.ones(n, bool)},
               "rsa": np.arange(n, dtype=np.float32),
               "plddt": np.arange(n, dtype=np.float32),
               "contacts": np.arange(n, dtype=np.float32),
               "structure_available": np.ones(n, np.float32),
               "handedness": np.arange(n, dtype=np.float32)}
        out, _ = chunk_long_rows([row], max_len=1022, stride=511)
        assert len(out) > 1
        for w in out:
            a = w["window_offset"]
            assert w["handedness"][0] == float(a), (
                "handedness was not sliced to the same window as rsa")
            np.testing.assert_array_equal(w["handedness"], w["rsa"])

    def test_the_evaluator_passes_it(self):
        import os

        src = open(os.path.join(REPO, "rockfish",
                                "eval_caid3_official.py")).read()
        assert '"handedness": sh' in src
        assert 'float("nan")' in src.split("sh = torch.full")[1][:60]
