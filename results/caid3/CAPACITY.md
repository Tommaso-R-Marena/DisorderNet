# CAID3 admits 117 methods and can order at most 7

## The theorem

`BenchmarkCapacity.card_le_benchCapacity` (Lean 4, sorry-free, only `propext`,
`Classical.choice`, `Quot.sound`). A benchmark with `n` targets, scores averaged
into [0,1] and therefore on the grid of denominator `n`, can place in a
certified order at most

    k(n, ε, δ) = n / (⌊c·n⌋ + 1) + 1        c = max(δ, 2ε)

methods, **however many are entered**. The factor two on the annotation error
rate is the two-sided price of label noise; `δ` is the smallest difference worth
calling a difference.

`benchCapacity_attained` exhibits a family of exactly `k` methods the benchmark
*does* resolve, so `k` is the capacity and not an estimate. Proved in three
forms — grid-valued, integer scores in `{0..N}`, real scores in `[0,R]` — each
with its own attainment instance.

## The measurement

ε is not assumed. MobiDB publishes the missing-residue call of each individual
deposited structure, up to 806 for one protein. Across 160 CAID3 Disorder-PDB
targets, **4,886 of 61,013 evidenced residues are context-dependent** — missing
in some structures, observed in others:

    ε = 0.0801

Not measurement error: the label is not a function of the sequence. A region
ordered in one crystal form and not another, or ordered on binding a partner,
is scored as one number per residue against a reference that depends on which
structure was consulted.

## The instantiation

| | targets | residues | **k** (targets) | **k** (residues) | entrants |
|---|---:|---:|---:|---:|---:|
| CAID3 Disorder-PDB | 233 | 99,239 | **7** | **7** | 117 |
| CAID3 Disorder-NOX | 178 | 99,977 | **7** | **7** | 117 |
| CAID3 Binding | 49 | 28,263 | **7** | **7** | 117 |
| CAID3 Binding-IDR | 42 | 7,741 | **7** | **7** | 117 |
| CAID3 Linker | 31 | 20,498 | **7** | **7** | 117 |
| CAID2 Disorder-PDB | 256 | 130,877 | **7** | **7** | 71 |

Three independent routes — the target-grid bound, the residue-level form
`card_le_capacity_labelNoise`, and the n-free ceiling
`benchCapacity_noise_only` (`k ≤ ⌈1/2ε⌉ = 7`) — agree on every benchmark of two
rounds.

**CAID3 admits 117 entrants and can place at most 7 of them in a certified
order.**

## Why more targets do not help

`benchCapacity_noise_only`: with δ = 0, `k ≤ ⌈1/(2ε)⌉` **whatever n is**.
Disorder-PDB has 233 targets and Disorder-NOX 178; both give 7. CAID2's
Disorder-PDB has 256 and also gives 7. Collecting targets does not buy
resolution the labels lack.

To place the full field of 117 in order, the annotation error rate would have
to fall to **0.427%** — a nineteen-fold improvement in label quality, not a
larger benchmark.

## Why this is impossibility, not low power

`unresolvable_pair` is the converse and it is what makes this different from a
power calculation. What a benchmark observes is the **annotation**, not the
truth, and every set within the noise budget of the annotation is a consistent
truth. For two methods whose measured scores are close, the proof *constructs
two such truths* — one making each method strictly better. Both are compatible
with everything the benchmark recorded.

So the ordering is **not a function of the data**. No analysis of those data
recovers it: not a larger bootstrap, not permutation testing, not LOOCV, not
Monte Carlo. The only remedy is better labels.

`over_capacity_has_close_pair` closes the loop: once the entry list exceeds the
capacity, such a pair **necessarily exists**. With 117 entrants and a capacity
of 7, CAID3 is over capacity by a factor of seventeen.

## What this project's own results look like under it

Consistent, and that is the point. Five architectures and an ensemble, each
pre-registered, failed to separate from PUNCH2 on pooled AUC: +0.0043, +0.0019,
+0.0035, +0.0063, +0.0062, none significant. **We were trying to resolve a pair
the benchmark cannot resolve.**

The separation that does exist is on the calibration-invariant axis, which is a
different statistic and is not bounded by this theorem:

| | vs PUNCH2, per-target within-protein AUC | Wilcoxon p |
|---|---:|---:|
| DisorderNet-pbias | +0.0190 | 0.00011 |
| DisorderNet-rank | +0.0185 | 0.00018 |
| **Ensemble** | **+0.0185** (138/233 targets) | **0.00001** |
| on Disorder-NOX (windowed) | +0.0522 | 0.00028 |

## Scope, as documented in `CAPACITY_CONTEXT.md`

- The noise model is a **worst-case budget**, not a probability model. That is
  exactly why the converse is a hard impossibility rather than a power
  statement.
- "At capacity" is the maximum size of a **certifiable family**. It is never
  promoted to a claim that such a family has been certified.
- ε = 0.0801 is measured on 160 of 319 Disorder-PDB targets — those with a
  usable MobiDB record. The rate is pooled over 61,013 residues.
- δ = 0 throughout. Any non-zero effect-size threshold only lowers the capacity.
