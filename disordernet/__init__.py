"""DisorderNet: what a benchmark can resolve, and the protocol that resolves more.

    >>> from disordernet import assess
    >>> print(assess(117, 0.0801).capacity)
    7

The theory is machine-checked in Lean 4 (`lean/RequestProject/`); every function
here names the theorem it implements.
"""

from .capacity import (
    Verdict,
    assess,
    capacity,
    capacity_over_range,
    imbalance_factor,
    pairwise_capacity,
    pairwise_noise_bound,
    unresolvable_comparisons,
)
from .noise import NoiseRates, rates, rates_by_group
from .protocol import Leaderboard, MethodScore, rank, score_method, target_auc

__version__ = "1.0.0"

__all__ = [
    "assess", "capacity", "capacity_over_range", "imbalance_factor",
    "pairwise_capacity", "pairwise_noise_bound", "unresolvable_comparisons",
    "Verdict", "rates", "rates_by_group", "NoiseRates",
    "rank", "score_method", "target_auc", "Leaderboard", "MethodScore",
    "__version__",
]
