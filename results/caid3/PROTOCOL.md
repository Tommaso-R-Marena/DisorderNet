# The CAID pairwise protocol — specification, and the field re-scored

## Why

At the annotation error rate measured from MobiDB's per-structure disagreements,
CAID3's residue-level protocol can place **8** of its 117 entrants in a certified
order (`card_le_benchCapacity`). The pairwise protocol can place **51**, because
a pair is reversed only when **both** its residues flip, in opposite
directions — a second-order event.

Measured: `ε_label = 0.0651`, `ε_pair = 0.0100`, on the denominator
`card_comparablePairs` defines. The closed-form bound `ε_pair ≤ 2·ε_label²`
requires balanced agreement classes, which CAID3 does not have (240,506
disordered against 555,402 ordered); the pooled rate exceeds it, and the gain is
therefore **measured rather than predicted**. See `ESCAPE.md`.

## The protocol

For a reference with targets `t` and per-residue labels in `{0, 1, -}`:

**1. Eligibility.** A method is scored iff it supplies a finite prediction for
every evaluated residue of every target, at the reference's length. Declining
targets raises a within-protein score, so coverage is a gate, not a covariate.
Ineligible methods are listed as *not scored*, never as last.

**2. Per-target statistic.** For each target carrying both classes among
evaluated residues, compute the Mann–Whitney AUC over that target's residues
alone. Targets with one class contribute no ordered pair and are skipped, with
the count reported.

**3. Method score.** The **unweighted mean** of the per-target AUCs. One protein,
one vote. Pair-weighting would let a few long chains carry the number.

**4. Ranking.** By that mean, descending.

**5. Separation.** Two methods are reported as *ordered* iff a paired test over
targets rejects equality — Wilcoxon signed-rank on the per-target differences,
Holm-corrected within the benchmark. Otherwise they are reported as **tied**, at
the same rank.

**6. Capacity.** The number of methods the reference can order is
`⌈1/(2·ε_pair)⌉` (`pairwise_capacity_bound`), with `ε_pair` the pairwise
discordance rate of the annotation — `discordant / comparable`, both counted as
`DiscordantPairs.lean` defines them — estimated from repeat determinations of
the same protein. It is **estimated, not predicted from the label-noise rate**:
the closed form needs a balance hypothesis this data does not satisfy. Methods beyond that
are reported as an unresolved group, never as a rank.

Step 3 is what makes the protocol **calibration-invariant**: each per-target AUC
is unchanged by any strictly monotone recalibration of that target's scores, so
any function of them is too.

**A citation correction.** `auc_within_strictMono_invariant` states this for
`AUC_within`, which is the **pair-weighted** mean of per-target AUCs — the form
the decomposition identity requires. The protocol uses the **unweighted** mean
(step 3), and the theorem as stated does not cover it. The invariance is
immediate for both, since it holds target by target before any averaging, but
the statement to cite should be the per-target one:

```lean
theorem auc_target_strictMono_invariant (f : ℝ → ℝ) (hf : StrictMono f) :
    aucOn t (f ∘ s) = aucOn t s
```

from which both the weighted and unweighted means follow. Worth adding, because
the protocol's central property should rest on a theorem about the statistic it
actually uses.

Step 6 is what keeps it honest: a rank the labels cannot support is not printed.

## CAID3 Disorder-PDB, re-scored — 120 entered, 60 eligible, 233 targets

| # | method | pairwise | pooled | pooled # | move | sep. from #1 |
|---:|---|---:|---:|---:|---:|---|
| **1** | **DisorderNet-pbias** | **0.9512** | 0.9635 | 1 | — | — |
| 2 | DisorderNet-Ensemble | 0.9508 | 0.9613 | 2 | — | tied |
| 3 | DisorderNet-windowed | 0.9481 | 0.9590 | 3 | — | tied |
| 4 | PredIDR2-Prof-Art | 0.9371 | 0.9360 | 10 | **+6** | tied |
| 5 | PredIDR2-Seq-Art | 0.9345 | 0.9362 | 9 | +4 | yes |
| 6 | AlphaFold-pLDDT | 0.9345 | 0.9342 | 13 | **+7** | yes |
| **7** | **PUNCH2** | 0.9323 | 0.9552 | **4** | **−3** | yes |
| 10 | PUNCH2-Light | 0.9278 | 0.9525 | 5 | −5 | yes |
| 13 | AlphaFold-rsa | 0.9176 | 0.9498 | 6 | **−7** | yes |
| 21 | Metapredict-v3 | 0.8809 | 0.9289 | 17 | −4 | yes |
| 23 | AlphaFold-binding | 0.8738 | 0.9342 | 12 | **−11** | yes |

**DisorderNet-pbias is separated from 55 of 59 eligible methods** — Wilcoxon,
Holm-corrected within the benchmark, and inside the capacity of 51. The
residue-level protocol could certify seven orderings in total.

Largest movers: flDPlr2 **+14**, flDPnn3a **+11**, AlphaFold-binding **−11**,
AIUPred-2-disorder −10, ESpritz-D −10.

## CAID3 Disorder-NOX, re-scored — 178 targets

| # | method | pairwise | pooled | pooled # | move |
|---:|---|---:|---:|---:|---:|
| **1** | **DisorderNet-pbias** | **0.8581** | 0.8760 | 2 | +1 |
| 2 | DisorderNet-windowed | 0.8579 | 0.8900 | 1 | −1 |
| 3 | DisorderNet-Ensemble | 0.8553 | 0.8640 | 3 | — |
| 4 | flDPnn3a | 0.8425 | 0.8602 | 4 | — |
| **6** | **AlphaFold-pLDDT** | 0.8094 | 0.7776 | **39** | **+33** |
| 8 | PUNCH2 | 0.8057 | 0.8422 | 6 | −2 |

**AlphaFold-pLDDT moves from 39th to 6th.** A training-free confidence score,
never trained on disorder, is sixth in the field at the residue-level question
and thirty-ninth on the number the challenge reports.

## The identity the protocol rests on

Let `T` be the truth and `L` the annotation on one protein. A pair `(p, q)` is
**discordant** when `T` orders it one way and `L` the other. That requires
`T(p)=1, L(p)=0` and `T(q)=0, L(q)=1` — **both residues flip, in opposite
directions.** So with `d = |T \ L|` (down-flips) and `u = |L \ T|` (up-flips):

    discordant(T, L) = 2·d·u        exactly     `discordant_eq_flip_product`
    noise(T, L)      = d + u

and by AM–GM, `d·u ≤ ((d+u)/2)²`, so

    discordant ≤ noise² / 2                     `discordant_le_noise_sq`

Both are unconditional. The pairs a pairwise protocol can score are the ones
**both** labellings order, and they too are counted exactly:

    |comparable(T, L)| = 2·(|T ∩ L|·|(T ∪ L)ᶜ| + d·u)   `card_comparablePairs`

so the pair-level noise rate is exactly `d·u / (|T ∩ L|·|(T ∪ L)ᶜ| + d·u)`, and
`noise_orderKey` shows it is literally the label noise of the answer key the
annotation induces on those pairs — which is why the existing capacity machinery
applies at the pair level verbatim (`pairwise_capacity_bound`).

The closed form `ε_pair ≤ 2·ε_label²`, and with it the quadratic capacity
`⌈1/(4ε²)⌉`, hold **when the agreement classes balance** and `ε ≤ 1/4`
(`nuPair_le_two_eps_sq`, `pairwise_capacity_quadratic`). On CAID3 they do not:
balance holds on 32 of 2,746 structure pairs, and on all 32 of those the bound
holds. Step 6 therefore says *estimate* `ε_pair`, not *predict it from* `ε`.

## Timing

CAID4's references are not yet published — rounds 4 and 5 return the site's
index page rather than a FASTA, and the dataset repository lists only CAID2 and
CAID3. **This protocol can be adopted rather than applied retrospectively**,
which is the difference between a critique and a contribution.
