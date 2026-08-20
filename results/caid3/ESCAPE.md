# The way around the capacity bound, measured

## The bound, and where it comes from

`card_le_benchCapacity` gives `k ≤ ⌈1/(2ε)⌉`. At the measured label-noise rate
CAID3 can place **at most 7** of its 117 entrants in a certified order.

But `ε` there is the rate at which the annotation gets a **label** wrong, and
AUC is not a function of labels one at a time. It is a function of **ordered
pairs**. So the noise a ranking statistic actually suffers is the rate at which
the annotation **reverses a pair's order** — a different quantity, and the whole
question is whether it is smaller.

It is, because label noise in protein structure is not independent across
residues. A region orders in one crystal form and not another, and it flips
**together**. Every pair *inside* a flipping region keeps its relation; only
pairs *crossing* the boundary reverse. Spatial correlation, which makes the
label problem worse, makes the pairwise problem better.

## The measurement

MobiDB's per-structure missing-residue calls, all pairs of structures of the
same protein, restricted to residues both structures cover. 159 CAID3
Disorder-PDB targets with at least two structures.

| | pooled | median protein |
|---|---:|---:|
| label-flip rate `ε_label` | **0.0651** | 0.0342 |
| pairwise discordance `ε_pair` | **0.0070** | 0.0006 |

**Ratio 9.32.** (`ε_label` here is computed directly from structure pairs;
MobiDB's own context-dependent annotation gives 0.0801 by a different route, so
the two estimators agree to within 20%.)

## The consequence

| scoring protocol | ε | **capacity `⌈1/2ε⌉`** |
|---|---:|---:|
| residue labels, as CAID scores now | 0.0651 | **8** |
| **within-protein pairwise** | **0.0070** | **72** |

**A benchmark scored on within-protein pairwise comparisons can order 72
methods where the residue-level protocol orders 8.** CAID3 has 117 entrants.
The resolution is not absent from the data — the current protocol discards it.

## This is not a coincidence with the empirical result

It is the mechanism behind it. Five architectures and an ensemble, every one
pre-registered, failed to separate from PUNCH2 on pooled AUC: +0.0043, +0.0019,
+0.0035, +0.0063, +0.0062, none significant. On the per-target within-protein
AUC the same comparison separates:

| | vs PUNCH2 | wins | Wilcoxon p |
|---|---:|---:|---:|
| Ensemble | +0.0185 | 138 / 233 | **0.00001** |
| DisorderNet-pbias | +0.0190 | 131 / 233 | 0.00011 |
| Disorder-NOX (windowed) | +0.0522 | 106 / 178 | 0.00028 |

**The within-protein statistic sees a tenth of the noise.** That is why it
resolves a pair the pooled statistic provably cannot, and it was predicted by
the decomposition before it was measured.

## What to prove next — superseded, and what replaced it

An earlier version of this document asked for a **block-correlation lemma**:
that when label noise flips whole regions, pairwise discordance is bounded by
the boundary-to-area ratio times the label rate, `O(1/L)` for compact blocks in
a chain. That was the wrong mechanism, and measuring it is what showed so. The
block statistics are real — mass-weighted mean block length 73.8 residues,
67.4% of context-dependent residues in blocks of ten or more — but `1/L`
predicts `ε_pair/ε_label = 0.0136` against a measured **0.1075**, over-predicting
the reduction eightfold.

The correct mechanism needs no correlation assumption at all. A pair is
discordant only when **both** its residues flip, in **opposite** directions:

    discordant(T, L) = 2·d·u        exactly,  d = |T \ L|, u = |L \ T|
    noise(T, L)      = d + u

so by AM–GM `discordant ≤ noise²/2`, and as rates over `n` residues with
balanced classes `ε_pair ≤ 2·ε_label²`. That predicts **0.00848** against a
measured **0.0070** — the bound holds, and is tight to 20%. Pairwise scoring
does not need the noise to be correlated; it squares it.

The open Lean items are therefore:

```lean
theorem discordant_eq_flip_product (T L : Finset ι) :
    discordant T L = 2 * (T \ L).card * (L \ T).card

theorem discordant_le_noise_sq (T L : Finset ι) :
    (discordant T L : ℝ) ≤ (noise T L : ℝ) ^ 2 / 2

theorem pairwise_capacity_quadratic (eps : ℝ) (h : 0 < eps) :
    benchCapacity_pairwise eps = ⌈1 / (4 * eps ^ 2)⌉
```

The first is a counting identity, the second is AM–GM on it, and the third
substitutes into `benchCapacity_noise_only`. Together they turn the escape route
from a measurement into a theorem, and they upgrade the capacity from linear in
`1/ε` to **quadratic**.

## Scope

- `ε_pair` is measured on the 159 targets with ≥2 deposited structures. Proteins
  with one structure contribute no disagreement to either rate and are excluded
  from both.
- Coverage per structure is approximated by the span of its annotated region,
  since MobiDB publishes per-structure *missing* regions and not per-structure
  coverage. Residues outside a structure's span are treated as undetermined by
  it rather than observed, which is the conservative direction: counting them as
  observed would manufacture disagreement from absence.
- Both rates are worst-case budgets over structure pairs, not probability
  models, matching the noise model the capacity theorem uses.
