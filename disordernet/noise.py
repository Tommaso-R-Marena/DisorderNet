"""Measuring a benchmark's annotation error rate from repeat determinations.

Most benchmarks cannot do this: they have one annotation per item. Where the
same item has been annotated twice -- two deposited structures of one protein,
two assessors on one query, an original label and an adjudicated correction --
both rates the capacity results need can be measured directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseRates:
    """Both rates, and the counts they came from."""

    eps_label: float
    eps_pair: float
    n_items: int
    n_disagreements: int
    comparable_pairs: int
    discordant_pairs: int
    agree_positive: int
    agree_negative: int

    @property
    def ratio(self) -> float:
        return self.eps_label / self.eps_pair if self.eps_pair else float("inf")

    @property
    def kappa(self) -> float:
        from .capacity import imbalance_factor
        return imbalance_factor(self.agree_positive, self.agree_negative)

    def __str__(self) -> str:
        return (f"eps_label {self.eps_label:.4f}   "
                f"eps_pair {self.eps_pair:.2e}   "
                f"ratio {self.ratio:,.0f}x   kappa {self.kappa:.3f}")


def rates(truth: np.ndarray, annotation: np.ndarray) -> NoiseRates:
    """Both noise rates from one pair of binary annotations of the same items.

    The counts are exact, not estimates:

        discordant = 2*d*u                         `discordant_eq_flip_product`
        comparable = 2*(a*e + d*u)                 `card_comparablePairs`

    with `a = |T & L|`, `e = |~T & ~L|`, `d = |T & ~L|`, `u = |~T & L|`. A pair
    is reversed only when both its members flip in opposite directions, which is
    the whole reason the pairwise rate is second-order.

    >>> import numpy as np
    >>> t = np.array([1,1,1,0,0,0,0,0])
    >>> l = np.array([1,1,0,1,0,0,0,0])
    >>> r = rates(t, l); r.n_disagreements, r.discordant_pairs
    (2, 2)
    """
    T = np.asarray(truth).astype(bool)
    L = np.asarray(annotation).astype(bool)
    if T.shape != L.shape:
        raise ValueError("truth and annotation must have the same shape")
    n = int(T.size)
    a = int((T & L).sum())
    e = int((~T & ~L).sum())
    d = int((T & ~L).sum())
    u = int((~T & L).sum())
    discordant = 2 * d * u
    comparable = 2 * (a * e + d * u)
    return NoiseRates(
        eps_label=(d + u) / n if n else 0.0,
        eps_pair=discordant / comparable if comparable else 0.0,
        n_items=n, n_disagreements=d + u,
        comparable_pairs=comparable, discordant_pairs=discordant,
        agree_positive=a, agree_negative=e,
    )


def rates_by_group(truth: dict[str, np.ndarray],
                   annotation: dict[str, np.ndarray]) -> NoiseRates:
    """Pool both rates over groups, scoring pairs *within* a group only.

    This is the quantity a grouped ranking statistic actually suffers: pairs
    that cross a group are never compared, so they cannot be reversed. Pooling
    is a ratio of sums, so the result is not covered by the per-instance bound
    `nuPair_le_imbalanced` -- that bound holds group by group.
    """
    tot = dict(n=0, dis=0, disc=0, comp=0, a=0, e=0)
    for k, t in truth.items():
        if k not in annotation:
            continue
        r = rates(t, annotation[k])
        tot["n"] += r.n_items
        tot["dis"] += r.n_disagreements
        tot["disc"] += r.discordant_pairs
        tot["comp"] += r.comparable_pairs
        tot["a"] += r.agree_positive
        tot["e"] += r.agree_negative
    return NoiseRates(
        eps_label=tot["dis"] / tot["n"] if tot["n"] else 0.0,
        eps_pair=tot["disc"] / tot["comp"] if tot["comp"] else 0.0,
        n_items=tot["n"], n_disagreements=tot["dis"],
        comparable_pairs=tot["comp"], discordant_pairs=tot["disc"],
        agree_positive=tot["a"], agree_negative=tot["e"],
    )
