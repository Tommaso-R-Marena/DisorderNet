"""How many methods a benchmark can place in a certified order.

Every function here implements a statement that is machine-checked in Lean 4
(`lean/RequestProject/`), and each names the theorem it implements. The point of
the package is that a benchmark can compute its own resolution before it
publishes a ranking; the point of naming the theorems is that the arithmetic can
be checked against something stronger than this docstring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def capacity(eps: float, delta: float = 0.0, n: int | None = None) -> int:
    """Methods a benchmark can place in a certified total order.

    `BenchmarkCapacity.card_le_benchCapacity`, with the `n`-free ceiling
    `benchCapacity_noise_only` when `n` is not given:

        k(n, eps, delta) = n / (floor(c*n) + 1) + 1,   c = max(delta, 2*eps)
        k(eps)           = ceil(1 / (2*eps))           for every n

    `eps` is a worst-case budget on the fraction of wrong annotations, not a
    probability model — which is why the converse (`unresolvable_pair`) is an
    impossibility rather than a power statement. `delta` is the smallest score
    difference worth calling a difference; any positive `delta` only lowers `k`.

    >>> capacity(0.0583)          # ImageNet, validated label-error rate
    9
    >>> capacity(0.0801)          # CAID3 Disorder-PDB
    7
    """
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if delta < 0.0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    c = max(delta, 2.0 * eps)
    if n is None:
        return max(1, math.ceil(1.0 / (2.0 * eps)) if delta == 0.0
                   else math.ceil(1.0 / c))
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return max(1, n // (math.floor(c * n) + 1) + 1)


def capacity_over_range(eps: float, lo: float, hi: float) -> int:
    """Capacity restricted to the score range methods actually occupy.

    The theorem allows the whole of `[0, 1]`. Published methods occupy a far
    narrower band, and a certified family of `k` members needs `k-1` gaps wider
    than `2*eps`, so on a band of width `w` at most `floor(w / 2*eps) + 1` fit.
    Reported separately because the band is an empirical input, not a theorem.

    >>> capacity_over_range(0.0583, 0.55, 0.92)   # ImageNet, published top-1
    4
    """
    if hi < lo:
        raise ValueError("hi must be at least lo")
    return max(1, int((hi - lo) // (2.0 * eps)) + 1)


def imbalance_factor(agree_positive: int, agree_negative: int) -> float:
    """`kappa = (a+e)^2 / (4*a*e)`, the price of lopsided agreement classes.

    `DiscordantImbalance.kappa`, with `one_le_kappa` and
    `kappa_eq_one_iff_balanced`: it is 1 exactly when the two classes the
    annotations agree on are the same size, and larger otherwise.

    >>> round(imbalance_factor(240506, 555402), 4)   # CAID3 Disorder-PDB
    1.1856
    """
    a, e = float(agree_positive), float(agree_negative)
    if a <= 0 or e <= 0:
        raise ValueError("both agreement classes must be non-empty")
    return (a + e) ** 2 / (4.0 * a * e)


def pairwise_noise_bound(eps: float, kappa: float = 1.0) -> float:
    """Upper bound on the pairwise discordance rate.

    `DiscordantImbalance.nuPair_le_imbalanced`:

        nu_pair <= kappa * eps^2 / (1 - eps)^2

    assuming only that both agreement classes are non-empty — no balance
    condition and no bound on the noise rate. At `kappa = 1` this is the
    balanced case, and `nuPair_le_two_eps_sq_of_kappa` recovers the published
    `nu_pair <= 2*eps^2` whenever `kappa <= 2*(1-eps)^2`.

    A pair is reversed only when *both* its members are mis-annotated, in
    opposite directions (`discordant_eq_flip_product`), which is why the rate is
    second-order rather than first-order.
    """
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must be in (0, 1), got {eps}")
    if kappa < 1.0:
        raise ValueError("kappa is at least 1 (one_le_kappa)")
    return kappa * eps ** 2 / (1.0 - eps) ** 2


def pairwise_capacity(eps: float, kappa: float = 1.0,
                      eps_pair: float | None = None) -> int:
    """Capacity of a benchmark scored on within-group ordered pairs.

    `pairwise_capacity_bound`. Pass a measured `eps_pair` when repeat
    annotations are available; otherwise the bound above is used, which is what
    lets a benchmark with no repeat annotations still compute its own gain.

    >>> pairwise_capacity(0.0651, eps_pair=0.00996)   # CAID3, measured
    51
    """
    rate = eps_pair if eps_pair is not None else pairwise_noise_bound(eps, kappa)
    if rate <= 0.0:
        raise ValueError("the pairwise rate must be positive")
    return max(1, math.ceil(1.0 / (2.0 * rate)))


def unresolvable_comparisons(n_methods: int, cap: int) -> int:
    """Comparisons the benchmark cannot decide, however they are analysed.

    `UnresolvableCount.card_closePairs_lower` is a pigeonhole on score blocks
    with a Cauchy-Schwarz step; `unresolvable_count_117` is the instance this
    returns in general form: with `k` methods and capacity `c`,

        k^2 <= c * (2u + k)   =>   u >= (k^2 - c*k) / (2c)

    Each counted comparison is undecidable rather than merely undecided: for
    such a pair `unresolvable_pair` constructs two ground truths consistent with
    every observation, one favouring each method.

    >>> unresolvable_comparisons(117, 8)
    798
    """
    if n_methods < 2 or cap < 1:
        raise ValueError("need at least two methods and a positive capacity")
    return max(0, math.ceil((n_methods ** 2 - cap * n_methods) / (2.0 * cap)))


@dataclass(frozen=True)
class Verdict:
    """What a benchmark can and cannot support, in one object."""

    n_methods: int
    eps: float
    capacity: int
    pairwise_capacity: int
    unresolvable: int
    total_comparisons: int
    over_capacity_by: float

    def __str__(self) -> str:
        verb = "is over" if self.n_methods > self.capacity else "is within"
        return (
            f"{self.n_methods} methods, annotation error rate {self.eps:.4f}\n"
            f"  capacity, as scored          {self.capacity:>10,}\n"
            f"  capacity, scored on pairs    {self.pairwise_capacity:>10,}\n"
            f"  the benchmark {verb} capacity by "
            f"{self.over_capacity_by:.1f}x\n"
            f"  comparisons it cannot decide {self.unresolvable:>10,} "
            f"of {self.total_comparisons:,}"
        )


def assess(n_methods: int, eps: float, *, delta: float = 0.0,
           n_items: int | None = None, kappa: float = 1.0,
           eps_pair: float | None = None) -> Verdict:
    """The whole verdict for a benchmark, from its size and its error rate.

    >>> print(assess(117, 0.0801))                       # doctest: +ELLIPSIS
    117 methods, annotation error rate 0.0801...
    """
    cap = capacity(eps, delta, n_items)
    return Verdict(
        n_methods=n_methods,
        eps=eps,
        capacity=cap,
        pairwise_capacity=pairwise_capacity(eps, kappa, eps_pair),
        unresolvable=unresolvable_comparisons(n_methods, cap),
        total_comparisons=n_methods * (n_methods - 1) // 2,
        over_capacity_by=n_methods / cap,
    )
