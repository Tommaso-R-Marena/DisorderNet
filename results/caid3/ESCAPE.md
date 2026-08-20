# The way around the capacity bound, measured

## The bound, and where it comes from

`card_le_benchCapacity` gives `k ≤ ⌈1/(2ε)⌉`. At the measured label-noise rate
CAID3 can place **at most 7** of its 117 entrants in a certified order.

But `ε` there is the rate at which the annotation gets a **label** wrong, and
AUC is not a function of labels one at a time. It is a function of **ordered
pairs**. So the noise a ranking statistic actually suffers is the rate at which
the annotation **reverses a pair's order** — a different quantity, and the whole
question is whether it is smaller.

It is, and the reason is a counting identity rather than any property of
proteins. A labelling orders a pair only by calling one residue disordered and
the other ordered, so the annotation reverses a pair only when **both** its
residues are wrong, **in opposite directions**:

    discordant(T, L) = 2·|T \ L|·|L \ T|        exactly

(`discordant_eq_flip_product`). A pairwise protocol therefore suffers a
second-order event where a residue-level protocol suffers a first-order one.

## The measurement

MobiDB's per-structure missing-residue calls, all pairs of structures of the
same protein, restricted to residues both structures cover. 159 CAID3
Disorder-PDB targets with at least two structures, 2,746 structure pairs.

The denominator is the one `card_comparablePairs` defines — the pairs **both**
labellings order, `2·(|T ∩ L|·|(T ∪ L)ᶜ| + d·u)` — and not any larger set.

| | pooled | median protein |
|---|---:|---:|
| label-flip rate `ε_label` | **0.0651** | 0.0342 |
| pairwise discordance `ε_pair` | **0.0100** | 0.0006 |

**Ratio 6.54.** (`ε_label` here is computed directly from structure pairs;
MobiDB's own context-dependent annotation gives 0.0801 by a different route, so
the two estimators agree to within 20%.)

### A correction

An earlier version of this document reported `ε_pair = 0.0070` and a capacity of
72. That denominator counted the pairs ordered by **either** labelling, with the
discordant ones counted twice — `2ae + 4du + (a+e)(d+u)` — which is larger than
the comparable set and mixes ordered with unordered counts. The formalisation is
what caught it: `card_comparablePairs` counts the denominator exactly, and the
two expressions are not equal. On the corrected denominator the rate is
**1.42× larger** and the capacity is **51**, not 72. Both numbers are kept in
`relative_noise.json` so the change is auditable rather than silent.

## The consequence

| scoring protocol | ε | **capacity `⌈1/2ε⌉`** |
|---|---:|---:|
| residue labels, as CAID scores now | 0.0651 | **8** |
| **within-protein pairwise** | **0.0100** | **51** |

**A benchmark scored on within-protein pairwise comparisons can order 51
methods where the residue-level protocol orders 8.** CAID3 has 117 entrants.
The resolution is not absent from the data — the current protocol discards it.

## The `2ε²` bound does not hold on this data, and the theorem says why

`nuPair_le_two_eps_sq` carries two hypotheses, and one of them is
`BalancedClasses T L`: among the residues the two labellings agree on, as many
are disordered as ordered, `|T ∩ L| = |(T ∪ L)ᶜ|`. It is load-bearing — the
denominator `|T ∩ L|·|(T ∪ L)ᶜ|` collapses when the classes are lopsided — and
on CAID3 it fails comprehensively. Pooled, the agreement classes stand at
240,506 disordered against 555,402 ordered, a ratio of **0.433**. Per structure
pair:

| | pairs | |
|---|---:|---:|
| checked | 2,746 | |
| balanced agreement classes within 10% | **54** | 2.0% |
| noise rate `ε ≤ 1/4` | 2,606 | 94.9% |
| **both hypotheses** | **32** | **1.2%** |
| bound `ν_pair ≤ 2ε²` holds | 2,248 | 81.9% |
| **… among the 32 satisfying both** | **32** | **100%** |

So the bound holds wherever the theorem says it holds, and the pooled rate
exceeds it — 0.00996 against 0.00848 — because the hypothesis the theorem needs
is false on a reference that is 31.6% disordered.

**A candidate that would restore the closed form, checked before asking for
it.** Dropping `d·u` from the denominator and substituting `a + e = n(1 − ε)`
gives, with `κ = (a+e)²/(4ae)` the imbalance factor and no balance hypothesis at
all:

    ν_pair ≤ κ · ε² / (1 − ε)²

`κ = 1` recovers the balanced case, and `κ ≤ 2(1−ε)²` recovers `2ε²`, so the
published bound would become a corollary. It holds on **2,266 of the 2,266**
CAID3 structure pairs where both agreement classes are nonempty — every one
(`relative_noise.py`, job 30101834). It is stated as request 1 in
`LEAN_REQUESTS.md` and is **not** used anywhere in the paper until it is proved.

**This does not weaken the result; it relocates it.** The identity
`discordant = 2·d·u` is unconditional, `discordant ≤ ν²/2` is unconditional, and
`ν_pair = d·u / (|T ∩ L|·|(T ∪ L)ᶜ| + d·u)` is exact. The capacity of 51 is
measured from that exact expression and needs no hypothesis at all. What is not
available on CAID3 is the closed form `2ε²`, and the honest consequence is that
the gain must be **measured on each benchmark** rather than predicted from its
label-noise rate.

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

## The Lean, as it now stands — all three proved

`RequestProject/DiscordantPairs.lean`, sorry-free, axioms `propext`,
`Classical.choice`, `Quot.sound`:

| statement | hypotheses | status |
|---|---|---|
| `discordant_eq_flip_product` : `discordant T L = 2·d·u` | none | **proved, exact** |
| `discordant_le_noise_sq` : `discordant ≤ ν²/2` | none | **proved** |
| `discordant_eq_of_balanced` : `2·discordant = ν²` | `d = u` | **proved** (equality case) |
| `card_comparablePairs` : `= 2(ae + du)` | none | **proved, exact** |
| `noise_orderKey` : pair-level noise **is** the discordant count | none | **proved** |
| `pairwise_capacity_bound` : `k ≤ max 1 ⌈1/(2ν_pair)⌉` | none | **proved** |
| `nuPair_le_two_eps_sq` : `ν_pair ≤ 2ε²` | balance, `ε ≤ 1/4` | **proved** — hypotheses fail on CAID3 |
| `pairwise_capacity_quadratic` : `⌈1/(4ε²)⌉ ≤ k` | balance, `ε ≤ 1/4`, `d,u > 0` | **proved** — hypotheses fail on CAID3 |

`sharp_instance` exhibits the equality case on four residues; `capacity_example`
exhibits sixteen residues where every hypothesis holds and the capacity is 25
against a residue capacity of 4, so none of it is vacuous.

The two results this document actually rests on — the exact identity and the
exact denominator — carry **no** hypotheses. The two that carry hypotheses are
the ones CAID3 cannot use.

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
measured **0.0100** on the corrected denominator — larger than the bound,
because the bound's balance hypothesis fails here. The mechanism does not need
the noise to be correlated — it is second-order in the label noise whatever the
noise looks like — but how much that is worth on a given benchmark has to be
measured, not derived.

All three asks are now proved, and the table above records what each of them
needs. The one thing the formalisation did **not** do is validate the number
this document used to report: it refuted it, by counting the denominator
exactly.

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
