"""Backbone handedness — the one thing every structural channel we feed is blind to.

Why this exists
---------------
DisorderNet reads three structural channels: relative solvent accessibility,
contact density and pLDDT. **All three are mirror-invariant.** Reflect a
protein through any plane and its accessibility is unchanged, its contact
counts are unchanged, and AlphaFold's confidence in it is unchanged. The model
cannot tell a structure from its mirror image, and neither can any feature it
currently receives.

The physics is not mirror-invariant. L-amino acids build **right-handed**
alpha-helices, and the conformation that dominates disordered and denatured
chains is polyproline II, which is **left-handed**. Handedness is not a
decoration on the order/disorder distinction; it is one of the few local
geometric quantities that separates the two.

The standard experimental assay for disorder is itself a chirality
measurement — far-UV circular dichroism, the differential absorption of left
and right circularly polarised light. And the accompanying Lean development
proves that this assay cannot deliver the distinction: the PPII and
statistical-coil reference spectra are nearly identical, so in a CD
deconvolution the PPII/coil split is free up to the noise
(`Dichroism.swapDir_isNull`, `equal_basis_split_free`,
`near_degenerate_tolerance`). If the handedness of a disordered region is
wanted, it has to come from geometry, not from the spectrum.

The quantity
------------
The signed dihedral over four consecutive alpha-carbons — the CA-trace virtual
torsion. It is the cheapest quantity that is *odd* under reflection: mirror the
coordinates and it changes sign, while rsa, contacts and pLDDT do not move.
That property is asserted in the tests rather than assumed, because it is the
entire justification for the channel.

Nothing here is fitted. The sign convention is verified against an ideal
right-handed alpha-helix built from its own helical parameters, so it does not
depend on anyone's remembered table of torsion values, and what the bands
*mean* for disorder is measured on data rather than asserted here.
"""

from __future__ import annotations

import numpy as np


def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
             p3: np.ndarray) -> np.ndarray:
    """Signed dihedral about the p1-p2 axis, in degrees, in (-180, 180].

    Vectorised over a leading axis. The sign is the whole point: an unsigned
    angle is mirror-invariant and would carry no handedness at all.
    """
    # Two sign traps, both hit while writing this, and only the second was
    # caught by the mirror test. Taking the first bond backwards rotates every
    # angle by 180 degrees (an ideal right-handed alpha-helix reads +130
    # instead of +50). Omitting the minus below negates it (+50 becomes -50) —
    # still exactly odd under reflection, so a mirror test passes and the
    # handedness of every helix in the dataset is reported backwards.
    # What caught both was agreement with an independent implementation on
    # random points, which the tests keep.
    b1 = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    b2 = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    b3 = np.asarray(p3, dtype=np.float64) - np.asarray(p2, dtype=np.float64)

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_hat = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    m1 = np.cross(n1, b2_hat)
    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(m1 * n2, axis=-1)
    return -np.degrees(np.arctan2(y, x))


def ca_virtual_torsion(ca: np.ndarray) -> np.ndarray:
    """Per-residue CA-trace virtual torsion, NaN where the window is incomplete.

    Residue i takes the dihedral over CA(i-1), CA(i), CA(i+1), CA(i+2), so the
    first residue and the last two have none. NaN rather than zero: zero is a
    legitimate torsion, and filling it in would tell the model "planar" about
    a residue we know nothing about — the same mistake as imputing rsa 0 for a
    protein with no structure, which reads as "fully buried".
    """
    ca = np.asarray(ca, dtype=np.float64)
    n = len(ca)
    out = np.full(n, np.nan, dtype=np.float32)
    if n < 4:
        return out
    valid = np.isfinite(ca).all(axis=1)
    tors = dihedral(ca[0:n - 3], ca[1:n - 2], ca[2:n - 1], ca[3:n])
    window_ok = valid[0:n - 3] & valid[1:n - 2] & valid[2:n - 1] & valid[3:n]
    tors = np.where(window_ok, tors, np.nan)
    out[1:n - 2] = tors.astype(np.float32)
    return out


def handedness(torsion: np.ndarray) -> np.ndarray:
    """Torsion mapped to [-1, 1] by its sine — a smooth, signed handedness.

    The sine rather than the raw angle because the angle wraps: +179 and -179
    are neighbouring conformations and 358 degrees apart as numbers. sin is
    continuous across the wrap, is odd (so it still flips under reflection),
    and is largest exactly where the right- and left-handed helical regions
    sit rather than in the extended region where the two are hardest to tell
    apart.
    """
    return np.sin(np.radians(np.asarray(torsion, dtype=np.float64)))


def ideal_alpha_helix(n: int, radius: float = 2.3, rise: float = 1.5,
                      turn_deg: float = 100.0,
                      right_handed: bool = True) -> np.ndarray:
    """CA coordinates of an ideal alpha-helix, from its helical parameters.

    Built rather than tabulated so the sign convention is checkable from first
    principles: a right-handed helix advances along +z while rotating
    anticlockwise seen from +z. Setting ``right_handed=False`` reverses the
    rotation and gives the exact mirror image, which is how the tests confirm
    that the torsion is odd under reflection.
    """
    k = 1.0 if right_handed else -1.0
    t = np.radians(turn_deg) * np.arange(n) * k
    return np.stack([radius * np.cos(t), radius * np.sin(t),
                     rise * np.arange(n)], axis=1)


def mirror(coords: np.ndarray, axis: int = 0) -> np.ndarray:
    """Reflect through a coordinate plane — a genuine improper transformation.

    Used by the tests to establish the claim this module rests on: rsa,
    contacts and pLDDT are unchanged by this and the torsion is negated.
    """
    out = np.array(coords, dtype=np.float64, copy=True)
    out[:, axis] *= -1.0
    return out
