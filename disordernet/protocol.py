"""The pairwise scoring protocol, as specified in the paper's Methods.

Six steps, each implemented here and each carrying the reason it exists. The
protocol scores a method by the unweighted mean of its per-target AUCs, reports
two methods as ordered only when a paired test rejects equality, and refuses to
print a rank the labels cannot support.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .capacity import pairwise_capacity


def target_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Mann-Whitney AUC over one target's residues alone.

    Step 2. A target carrying one class contributes no ordered pair and is
    skipped rather than scored as 0.5, because a benchmark that averages in a
    number no method could have influenced is measuring its own composition.

    Invariant under any strictly monotone recalibration of *this* target's
    scores (`AUCTargetMean.auc_target_strictMono_invariant`), which is what
    makes the protocol calibration-invariant once step 3 averages them.
    """
    y = np.asarray(labels).astype(np.int8)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0 or not np.isfinite(s).all():
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1, dtype=float)
    # average ranks over ties, so a constant predictor scores exactly 0.5
    _, first, counts = np.unique(s[order], return_index=True, return_counts=True)
    for f, c in zip(first, counts):
        if c > 1:
            ranks[order[f:f + c]] = ranks[order[f:f + c]].mean()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


@dataclass
class MethodScore:
    name: str
    score: float
    per_target: np.ndarray
    n_targets: int
    eligible: bool = True
    reason: str = ""


def score_method(name: str, reference: dict[str, np.ndarray],
                 predictions: dict[str, np.ndarray]) -> MethodScore:
    """Steps 1-3: eligibility gate, per-target AUC, unweighted mean.

    Step 1 is a gate and not a covariate. A method that declines targets is
    scored on an easier benchmark, not a harder one -- on CAID3 the declined
    targets are longer and harder by +0.035 to +0.125 AUC -- so declining is
    excluded from the ranking rather than penalised inside it.

    Step 3 is the unweighted mean: one target, one vote. The pair-weighted mean
    is the quantity the decomposition identity needs, but it lets a few long
    chains carry the number, and the two genuinely differ
    (`AUCTargetMean.weighted_ne_unweighted_instance`).
    """
    missing = [t for t in reference if t not in predictions]
    if missing:
        return MethodScore(name, float("nan"), np.array([]), 0, False,
                           f"declined {len(missing)} of {len(reference)} targets")
    wrong = [t for t in reference
             if len(predictions[t]) != len(reference[t])]
    if wrong:
        return MethodScore(name, float("nan"), np.array([]), 0, False,
                           f"length mismatch on {len(wrong)} targets")
    aucs = []
    for t, y in reference.items():
        a = target_auc(y, predictions[t])
        if a is not None:
            aucs.append(a)
    if not aucs:
        return MethodScore(name, float("nan"), np.array([]), 0, False,
                           "no target carries both classes")
    arr = np.asarray(aucs, dtype=float)
    return MethodScore(name, float(arr.mean()), arr, arr.size)


@dataclass
class Leaderboard:
    rows: list[dict] = field(default_factory=list)
    capacity: int | None = None
    eps_pair: float | None = None
    n_entered: int = 0
    n_eligible: int = 0

    def __str__(self) -> str:
        out = [f"{len(self.rows)} eligible of {self.n_entered} entered"]
        if self.capacity is not None:
            out.append(f"capacity under this protocol: {self.capacity}")
        out.append(f"{'#':>4}  {'method':<28}{'score':>9}  separated")
        for r in self.rows:
            sep = "-" if r["rank"] == 1 else ("yes" if r["separated"] else "tied")
            out.append(f"{r['rank']:>4}  {r['method']:<28}{r['score']:>9.4f}  {sep}")
        if self.capacity is not None and len(self.rows) > self.capacity:
            out.append(f"\nRanks beyond {self.capacity} are not supported by the "
                       f"labels and are reported as an unresolved group.")
        return "\n".join(out)


def rank(reference: dict[str, np.ndarray],
         predictions: dict[str, dict[str, np.ndarray]],
         *, eps_pair: float | None = None,
         alpha: float = 0.05) -> Leaderboard:
    """Steps 4-6: rank, separate with a Holm-corrected paired test, and stop.

    Step 5 uses Wilcoxon signed-rank on the per-target differences against the
    leader, Holm-corrected within the benchmark. Two methods that do not
    separate are reported at the same rank rather than in an arbitrary order.

    Step 6 is what keeps the protocol honest: with `eps_pair` supplied, ranks
    beyond `ceil(1 / 2*eps_pair)` are reported as an unresolved group. A rank
    the labels cannot support is not printed.
    """
    from scipy.stats import wilcoxon

    scored = [score_method(k, reference, v) for k, v in predictions.items()]
    ok = sorted((m for m in scored if m.eligible), key=lambda m: -m.score)
    if not ok:
        return Leaderboard([], None, eps_pair, len(scored), 0)

    leader = ok[0]
    raw = {}
    for m in ok[1:]:
        n = min(leader.per_target.size, m.per_target.size)
        d = leader.per_target[:n] - m.per_target[:n]
        raw[m.name] = (1.0 if not np.any(d)
                       else float(wilcoxon(d, alternative="two-sided",
                                           zero_method="wilcox").pvalue))
    # Holm-Bonferroni within the benchmark
    adj, running = {}, 0.0
    for i, (name, p) in enumerate(sorted(raw.items(), key=lambda kv: kv[1])):
        running = max(running, (len(raw) - i) * p)
        adj[name] = min(1.0, running)

    cap = pairwise_capacity(0.5, eps_pair=eps_pair) if eps_pair else None
    rows = [{"rank": 1, "method": leader.name, "score": leader.score,
             "separated": True, "p_holm": None, "n_targets": leader.n_targets}]
    for i, m in enumerate(ok[1:], start=2):
        rows.append({"rank": i, "method": m.name, "score": m.score,
                     "separated": adj[m.name] < alpha,
                     "p_holm": adj[m.name], "n_targets": m.n_targets})
    return Leaderboard(rows, cap, eps_pair, len(scored), len(ok))
