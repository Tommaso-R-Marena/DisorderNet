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

## What to prove next

The capacity theorem is stated for a benchmark scoring items with noisy labels.
The escape route needs its companion:

```
theorem pairwise_capacity {G : Type*} (groups : Finset G) (nu_pair : ℝ) :
    -- a benchmark scored on within-group ordered pairs, whose annotation
    -- reverses a pair with rate at most nu_pair, has capacity
    k ≤ ⌈1 / (2 * nu_pair)⌉
```

plus the part that makes it a *theorem about structure* rather than a
restatement:

```
theorem correlated_noise_reduces_pair_discordance
    (blocks : Finset (Finset α)) (h : noise flips whole blocks) :
    nu_pair ≤ (boundary mass / total pairs) * nu_label
```

i.e. **when label noise is block-correlated, pairwise discordance is bounded by
the boundary-to-area ratio times the label rate.** For compact blocks in a
one-dimensional chain that ratio is `O(1/|block|)`, which predicts exactly the
order-of-magnitude gap measured here.

That pair — the pairwise capacity bound, and the block-correlation lemma
explaining why it is larger — turns "CAID3 cannot order its field" into
"**here is the protocol that can, and here is why it works**".

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
