# A benchmark cannot order more methods than its labels can distinguish: capacity limits in CAID, and the protocol that escapes them

**Tommaso R. Marena**

*Correspondence: marenatommaso@gmail.com*

---

## Abstract

Community benchmarks rank methods on noisy labels, and the field treats the
resulting order as data. We show it is largely not. We prove, in machine-checked
Lean 4, that a benchmark with annotation error rate ε can place at most
⌈1/(2ε)⌉ methods in a certifiable order **however many are entered and however
the data are analysed**, and that beyond that number the ordering is not a
function of the observations: for two close methods the proof constructs two
ground truths, each consistent with everything the benchmark recorded, one
making each method better. Measuring ε directly from MobiDB's per-structure
missing-residue calls — 4,886 of 61,013 evidenced residues of CAID3
Disorder-PDB are ordered in one deposited structure and disordered in another,
ε = 0.080 — gives CAID3 a capacity of **seven**. It admits 117.

The resolution is not absent from the data; the protocol discards it. AUC is a
function of ordered pairs, not of labels, and a pair is reversed only when both
its residues flip in opposite directions. That is an exact identity —
discordant = 2·d·u, machine-checked and unconditional — and the pairs a pairwise
protocol can score are counted exactly too. Measured on 2,746 pairs of deposited
structures of the same protein, the pairwise rate is 0.0100 against a label rate
of 0.0651, and the capacity rises from 8 to **51 of 117**. We specify a six-step
pairwise protocol that realises it, re-score all of CAID3 under it, and show the
resulting ranking reproduces itself on a held-out round (CAID2, 57 shared
entrants) as well as the reported one does — six times the resolution at no cost
in reproducibility. Under it, AlphaFold's pLDDT, a
confidence score never trained on disorder, moves from 39th to 6th on
Disorder-NOX, and the declared winners of three of five CAID3 benchmarks are
21st to 25th at the residue-level question the metric is read as asking.

We also give disorder prediction its first distribution-free operating
guarantee: fix a tolerated miss rate, and conformal risk control returns a
threshold that achieves it on a protein nobody has seen. Validity is free — all
70 full-coverage CAID3 methods achieve it — and the entire content is the
price, which runs from 32% to 97% of the protein flagged.

The accompanying predictor, DisorderNet, ranks first of 115 entrants on three
of five CAID3 references and first among full-coverage entrants on all four
CAID2 references, and beats the CAID3 Disorder-PDB winner on the
calibration-invariant axis on all five benchmarks after Holm correction over
thirty comparisons — a separation the benchmark's own metric provably cannot
make.

---

## Introduction

Intrinsically disordered regions are not folded, and the experiment that
reports them — a deposited structure with residues absent from the coordinates —
reports a property of one crystal, one construct, one condition. The Critical
Assessment of Intrinsic Disorder (CAID) turns those annotations into a
leaderboard: every residue of every target is pooled into one area under the ROC
curve, and 117 submissions are placed in order [1,2].

Two things about that number are worth stating precisely before anything else.

**It is mostly not the question people read it as.** A pooled AUC is the
probability that a random disordered residue outranks a random ordered one, and
every such pair is either within one protein or between two. The statistic is
therefore exactly

    AUC_pooled = w_within · AUC_within + w_between · AUC_between,

and on CAID3 the within-protein weight is 0.51% to 2.9%. This much is
arithmetic — for n proteins of comparable size about 1/n of pairs fall inside
one — and by itself it proves nothing. What it licenses is a question: does the
metric's 1:200 weighting change who wins? On three of five CAID3 benchmarks it
does, and by a lot.

**It is being asked to resolve more than it can.** Every benchmark orders
methods on annotations, and annotations are wrong at some rate. The question
nobody asks is how many methods that rate permits to be ordered *at all*. We
prove the answer is ⌈1/(2ε)⌉, prove it is attained, and prove the converse:
past that count, two close methods admit two consistent truths, one favouring
each. No amount of resampling, cross-validation or Monte Carlo recovers an
order that is not a function of the data. The only remedy is better labels — or
a different statistic.

This paper supplies the second. The capacity bound is a statement about
labels; AUC is not a function of labels one at a time but of ordered pairs, and
the two noise rates are not the same quantity. A pair is reversed only when
both of its residues are mis-annotated, in opposite directions — a second-order
event. We give the exact identity, measure both rates on repeat determinations
of the same protein, specify a protocol that scores the pairs rather than the
labels, and re-score the entire CAID3 field under it.

---

## Results

### CAID's headline statistic is 97–99.5% a between-protein question, and the split is not a metaphor

The decomposition above is exact, and its first term has a property that makes
it more than a bookkeeping device. `auc_within_strictMono_invariant` (Lean 4,
sorry-free): **AUC_within is invariant under any per-protein strictly monotone
recalibration.** Rescore every chain by its own exponential, cube or logistic
squash and the quantity does not move. `auc_pooled_shift_diff` completes it: the
entire effect of such a recalibration on the reported number falls in the
between-protein term.

So the split partitions CAID's statistic into the part no chain-by-chain
rescoring can touch and the part that is nothing else (**Fig. 1a**;
Supplementary Table S1).

| benchmark | targets | w_within | share a recalibration can move |
|---|---:|---:|---:|
| Disorder-PDB | 233 | 0.5091% | **99.49%** |
| Disorder-NOX | 178 | 0.6747% | **99.33%** |
| Binding | 49 | 2.5518% | 97.45% |
| Binding-IDR | 42 | 2.8913% | 97.11% |
| Linker | 31 | 2.8826% | 97.12% |

### The declared winner is 21st–25th at the residue-level question, and each inversion is certified

On three of five CAID3 benchmarks the method the challenge declares best is
21st to 25th at telling which residues of a chain are disordered (**Fig. 1b**):

| benchmark | CAID3 winner | its within-protein rank | best within-protein |
|---|---|---:|---|
| Disorder-PDB | PUNCH2 | **3** / 57 | AlphaFold-pLDDT (0.9392) |
| Disorder-NOX | flDPnn3a | **25** / 57 | Metapredict-v3 (0.8564) |
| Binding | UdonPred-combined | **21** / 69 | AlphaFold3-binding (0.7998) |
| Binding-IDR | LIPNet | **1** / 69 | LIPNet (0.7620) |
| Linker | LINKER-Pred2 | **25** / 90 | APOD (0.8344) |

The three Linker specialists sweep the pooled table at #1, #2 and #3 and sit at
#25, #26 and #29 within protein: extremely good at ranking whole chains by
linker content, unremarkable at locating linkers inside one.

Each disagreement is a certificate, not an observation.
`inversion_requires_between_gap` states that the pooled order reverses the
within-protein order only when w_within·(within gap) < w_between·(between gap),
so an inversion *forces* a between-protein gap of at least (w_within/w_between)
times the within-protein gap. On Disorder-NOX, against the pooled winner
(**Fig. 1c**):

| method | within-protein lead | between-protein deficit | required | over-determined |
|---|---:|---:|---:|---:|
| Metapredict-v3 | +0.0218 | 0.0833 | 0.000148 | **562×** |
| AIUPred-2-disorder | +0.0093 | 0.0695 | 0.000063 | 1,102× |
| AlphaFold3-pLDDT | +0.0077 | 0.0833 | 0.000053 | 1,584× |
| AlphaFold3-rsa | +0.0033 | 0.0523 | 0.000023 | **2,305×** |

Every certificate holds. A within-protein deficit of 0.0218 needs a
between-protein advantage of 0.000148 to be overturned, and the one actually
present is 562 times larger. Calibration differences of ordinary size do not
merely outweigh discrimination differences of ordinary size; they swamp them by
three orders of magnitude.

**The honest counterweight belongs here and not in a footnote.**
Spearman(pooled, within) is +0.964 on Disorder-PDB and never below +0.849
across the five references. A method that predicts residues well is usually
calibrated across chains too, and the decomposition does not overturn CAID3
(**Fig. 3a**, ρ = 0.844 across 357 method–benchmark pairs). What it does is
locate where the two orderings disagree, and show that where they disagree the
movement is large.

### A benchmark can order at most ⌈1/2ε⌉ methods, and CAID3 admits seventeen times that

`BenchmarkCapacity.card_le_benchCapacity` (Lean 4, sorry-free; axioms `propext`,
`Classical.choice`, `Quot.sound`). A benchmark with n targets, scores averaged
into [0,1] and therefore on the grid of denominator n, can place in a certified
order at most

    k(n, ε, δ) = n / (⌊c·n⌋ + 1) + 1,    c = max(δ, 2ε)

methods, **however many are entered**. The factor two on ε is the two-sided
price of label noise; δ is the smallest difference worth calling a difference
(δ = 0 throughout here; any positive δ only lowers k). `benchCapacity_attained`
exhibits a family of exactly k methods the benchmark does resolve, so k is the
capacity and not an estimate; the theorem is proved in three forms —
grid-valued, integer-valued and real-valued scores — each with its own
attainment instance. `benchCapacity_noise_only` gives the n-free ceiling
k ≤ ⌈1/(2ε)⌉: **collecting more targets does not buy resolution the labels
lack.**

ε is measured, not assumed. MobiDB publishes the missing-residue call of each
individual deposited structure, up to 806 for one protein. Across 160 CAID3
Disorder-PDB targets, **4,886 of 61,013 evidenced residues are
context-dependent** — missing in some structures and observed in others:

    ε = 0.0801.

That is not measurement error. It is that the label is not a function of the
sequence: a region ordered in one crystal form and not another, or ordered on
binding a partner, is scored as one number per residue against a reference that
depends on which structure was consulted. An independent estimator, over all
pairs of structures of the same protein restricted to residues both cover,
gives ε = 0.0651 — the two agree to within 20%.

Instantiated (**Fig. 2a**; Supplementary Table S3):

| | targets | residues | k (targets) | k (residues) | entrants |
|---|---:|---:|---:|---:|---:|
| CAID3 Disorder-PDB | 233 | 99,239 | **7** | **7** | 117 |
| CAID3 Disorder-NOX | 178 | 99,977 | **7** | **7** | 117 |
| CAID3 Binding | 49 | 28,263 | **7** | **7** | 117 |
| CAID3 Binding-IDR | 42 | 7,741 | **7** | **7** | 117 |
| CAID3 Linker | 31 | 20,498 | **7** | **7** | 117 |
| CAID2 Disorder-PDB | 256 | 130,877 | **7** | **7** | 71 |

Three independent routes — the target grid, the residue-level form
`card_le_capacity_labelNoise`, and the n-free ceiling — agree on every benchmark
of two rounds. **CAID3 admits 117 entrants and can place at most seven of them
in a certified order.** To order the full field, the annotation error rate would
have to fall to 0.427%: a nineteen-fold improvement in label quality, not a
larger benchmark.

**This is impossibility, not low power.** `unresolvable_pair` is the converse
and it is what separates this from a power calculation. A benchmark observes the
annotation, not the truth, and every labelling within the noise budget of the
annotation is a consistent truth. For two methods whose measured scores are
close, the proof *constructs two such truths* — one making each strictly better
— both compatible with everything the benchmark recorded. The ordering is
therefore not a function of the data, and no analysis of those data recovers it:
not a larger bootstrap, not permutation testing, not leave-one-out, not Monte
Carlo. `over_capacity_has_close_pair` closes the loop: once the entry list
exceeds capacity, such a pair necessarily exists.

Our own results are consistent with it, which is the point rather than an
excuse. Five pre-registered architectures and an ensemble each failed to
separate from PUNCH2 on pooled AUC — +0.0043, +0.0019, +0.0035, +0.0063,
+0.0062 — none significant, and each of the five also failed the endpoint its
own registration named, against a regime-matched control, with not one of
fifteen paired comparisons surviving Holm (Supplementary Note S4). We were
trying to resolve a pair the benchmark cannot resolve.

### Pairwise scoring makes the noise second-order, and the capacity rises sixfold

The bound is a statement about **labels**. AUC is not a function of labels one
at a time; it is a function of **ordered pairs**. The noise a ranking statistic
actually suffers is the rate at which the annotation *reverses a pair*, which is
a different quantity.

Let T be the truth on one protein and L the annotation. A pair (p, q) is
discordant when T orders it one way and L the other. That requires T(p)=1,
L(p)=0 **and** T(q)=0, L(q)=1: both residues flip, in opposite directions. With
d = |T \ L| and u = |L \ T|,

    discordant(T, L) = 2·d·u        exactly,
    noise(T, L)      = d + u,

and by AM–GM, d·u ≤ ((d+u)/2)², so discordant ≤ noise²/2. Both are identities
about counting and hold with no hypotheses at all
(`discordant_eq_flip_product`, `discordant_le_noise_sq`).

The pairs a pairwise protocol can actually score are the ones **both** labellings
order, and `card_comparablePairs` counts those exactly as well:

    |comparable(T, L)| = 2·(|T ∩ L|·|(T ∪ L)ᶜ| + d·u),

so the noise rate a pairwise protocol suffers is exactly

    ε_pair = d·u / (|T ∩ L|·|(T ∪ L)ᶜ| + d·u).

`noise_orderKey` closes the loop: on comparable pairs, that quantity *is* the
label noise of the answer key the annotation induces, so the capacity theorem
applies at the pair level verbatim (`pairwise_capacity_bound`).

Measured, from MobiDB's per-structure calls over all pairs of structures of the
same protein, on the 159 CAID3 Disorder-PDB targets with at least two deposited
structures (**Fig. 2b,c**):

| | pooled | median protein |
|---|---:|---:|
| label-flip rate ε_label | **0.0651** | 0.0342 |
| pairwise discordance ε_pair | **0.0100** | 0.0006 |

Ratio 6.54. The consequence is the central result of this paper:

| scoring protocol | ε | capacity ⌈1/2ε⌉ |
|---|---:|---:|
| residue labels, as CAID scores now | 0.0651 | **8** |
| **within-protein pairwise** | **0.0100** | **51** |

**A benchmark scored on within-protein pairwise comparisons can order 51 methods
where the residue-level protocol orders 8.** CAID3 has 117 entrants.

**The closed form does not apply here, and the theorem is what says so.**
`nuPair_le_two_eps_sq` gives ε_pair ≤ 2·ε_label², and with it a capacity
quadratic in label quality rather than linear — but it carries two hypotheses,
and one is that the *agreement* classes balance, |T ∩ L| = |(T ∪ L)ᶜ|. It is
load-bearing: the denominator |T ∩ L|·|(T ∪ L)ᶜ| collapses when the classes are
lopsided, and the gain collapses with it. On a reference that is 31.6%
disordered it fails comprehensively — pooled, the agreement classes stand at
240,506 disordered against 555,402 ordered — and it holds on **32 of 2,746**
structure pairs. On all 32 of those, the bound holds. Pooled, the measured rate
exceeds it, 0.00996 against 0.00848.

So the sixfold gain is a **measurement**, not a prediction from the label-noise
rate, and step 6 of the protocol below says *estimate* ε_pair rather than derive
it. The two results the argument rests on — the identity and the exact
denominator — carry no hypotheses; the closed form that would have let a
benchmark skip the measurement is exactly the part CAID3 cannot use.

**Two things this paper got wrong here, both caught by formalising them.** The
first was the mechanism: an earlier version attributed the gap to spatial
correlation — disorder flips in blocks, so only pairs crossing a block boundary
reverse, and the reduction should scale as 1/L. The block statistics are real
(mass-weighted mean block length 73.8 residues; 67.4% of context-dependent
residues in blocks of ten or more), but 1/L predicts a ratio of 0.0136 against a
measured 0.1075, over-predicting the reduction eightfold. The identity above
needs no correlation assumption and is exact.

The second was the number. An earlier measurement put ε_pair at 0.0070 and the
capacity at 72, on a denominator that counted the pairs ordered by *either*
labelling with the discordant ones counted twice — 2ae + 4du + (a+e)(d+u) —
which is larger than the comparable set and mixes ordered with unordered counts.
`card_comparablePairs` counts it exactly, the two expressions are not equal, and
on the correct denominator the rate is 1.42× larger and the capacity is 51. We
record both because a corrected headline number should be visible as a
correction, not as a different number.

### A protocol, and the field re-scored under it

**The CAID pairwise protocol.** For a reference with targets t and per-residue
labels in {0, 1, –}:

1. **Eligibility.** A method is scored iff it supplies a finite prediction for
   every evaluated residue of every target, at the reference's length. Declining
   targets raises a within-protein score, so coverage is a gate, not a
   covariate. Ineligible methods are listed as *not scored*, never as last.
2. **Per-target statistic.** For each target carrying both classes among
   evaluated residues, the Mann–Whitney AUC over that target's residues alone.
   Single-class targets contribute no ordered pair and are skipped, with the
   count reported.
3. **Method score.** The **unweighted mean** of the per-target AUCs. One
   protein, one vote; pair-weighting would let a few long chains carry the
   number.
4. **Ranking.** By that mean, descending.
5. **Separation.** Two methods are reported as *ordered* iff a paired test over
   targets rejects equality — Wilcoxon signed-rank on the per-target
   differences, Holm-corrected within the benchmark. Otherwise they are reported
   as **tied**, at the same rank.
6. **Capacity.** The number of methods the reference can order is
   ⌈1/(2·ε_pair)⌉, with ε_pair estimated from repeat determinations of the same
   protein. Methods beyond that are reported as an unresolved group, never as a
   rank.

Step 3 is what makes the protocol calibration-invariant: each per-target AUC is
unchanged by any strictly monotone recalibration of that target's scores, so any
function of them is too. Step 6 is what keeps it honest: a rank the labels
cannot support is not printed.

Re-scoring CAID3 (**Fig. 3**; full tables in Supplementary Tables S5–S9), the
two orderings agree overall (ρ = 0.844 over 357 method–benchmark pairs,
**Fig. 3a**) and disagree sharply in places (**Fig. 3b**). On Disorder-PDB, 120
entered and 60 are eligible:

| # | method | pairwise | pooled | pooled # | move |
|---:|---|---:|---:|---:|---:|
| **1** | **DisorderNet-pbias** | **0.9512** | 0.9635 | 1 | — |
| 2 | DisorderNet-Ensemble | 0.9508 | 0.9613 | 2 | — |
| 3 | DisorderNet-windowed | 0.9481 | 0.9590 | 3 | — |
| 4 | PredIDR2-Prof-Art | 0.9371 | 0.9360 | 10 | **+6** |
| 6 | AlphaFold-pLDDT | 0.9345 | 0.9342 | 13 | **+7** |
| **7** | **PUNCH2** | 0.9323 | 0.9552 | **4** | **−3** |
| 13 | AlphaFold-rsa | 0.9176 | 0.9498 | 6 | **−7** |
| 23 | AlphaFold-binding | 0.8738 | 0.9342 | 12 | **−11** |

On Disorder-NOX, **AlphaFold-pLDDT moves from 39th to 6th**: a training-free
confidence score, never trained on disorder, is sixth in the field at the
residue-level question and thirty-ninth on the number the challenge reports.

The protocol also declares what it cannot order. The leader is separated at the
certified margin from 55 of 59 eligible methods on Disorder-PDB, 51 of 59 on
Disorder-NOX, 56 of 71 on Binding, 50 of 71 on Binding-IDR and 87 of 92 on
Linker (**Fig. 3c**) — against seven orderings in total for the residue-level
protocol.

Those counts exceed the capacity of 51, and they are allowed to. The capacity
bounds a certified **totally ordered family**; 87 separations against one leader
are 87 statements about the pairs {leader, X}, which need not compose into a
chain of 88. Step 6 governs the ranking the protocol prints, not the number of
pairs a test rejects. We flag the distinction because the two numbers appear side
by side and invite the wrong reading, and because the theorem that would make it
precise — the longest certified chain is at most the capacity, while the number
of certified pairs may exceed it — is not yet stated.

A single-class target sharpens the point. On Binding-IDR, ten of 52 targets
carry one class among evaluated residues, so every pair they enter is
between-protein and they ask no residue-level question at all. Restricting to
the 42 that do **reverses the sign of all three of our registered variants'
comparisons against their control** — `mt_motif` −0.0141 → +0.0480, `mt_wass`
−0.0308 → +0.0535, `mt_rank` −0.0050 → +0.0117. Their whole apparent regression
sits in targets that carry no within-protein information. The registered
endpoints still fail, because they were registered on the pooled statistic; but
the reason they fail is not that the variants are worse at the question the
benchmark is read as asking.

CAID4's references are not yet published: rounds 4 and 5 return the site's index
page rather than a FASTA, and the dataset repository lists only CAID2 and CAID3.
**This protocol can be adopted rather than applied retrospectively**, which is
the difference between a critique and a contribution.

### The extra resolution is not noise

A protocol could resolve more pairs and resolve them wrongly. That is testable:
**57 methods entered both CAID2 and CAID3.** Rank them on CAID3 under each
protocol and ask which CAID3 ranking reproduces their CAID2 ranking — a
different round, different targets, one protein shared (**Fig. 6a**).

| benchmark | methods | pairwise → pairwise | pooled → pooled |
|---|---:|---:|---:|
| Disorder-PDB | 24 | **0.997** | 0.976 |
| Disorder-NOX | 24 | 0.927 | 0.950 |
| Binding | 29 | 0.828 | 0.906 |
| Linker | 47 | 0.843 | 0.822 |

Self-consistency is 0.83–0.997 for pairwise against 0.82–0.95 for pooled —
higher on two benchmarks, lower on two, and the highest single figure in the
table. On Disorder-PDB the pairwise ranking predicts the held-out pairwise
ordering better than the pooled ranking does (+0.036, 95% CI [+0.010, +0.122],
p = 0.0062); on the other three the difference is not significant.

**This answers the objection and does not overreach.** The extra resolution is
reproducible. It is *not* shown to be more valid, and this paper does not claim
it is: three of four benchmarks show no significant difference and one favours
the pooled protocol. The case for the protocol rests on capacity — what the
labels can support — not on a validity advantage the data do not show. A
protocol that resolves six times as much of the field with the same cross-round
stability, from the same data, is worth adopting whether or not it is also more
accurate.

### An operating guarantee, and the price of it

Every disorder predictor in CAID reports a ranking statistic. None answers the
question a biologist has: *can I act on this call, and how often will it be
wrong?* `Calibration.risk_decomposition` is exact about what calibration buys —
risk = calibration error + resolution — so a perfectly calibrated model has paid
the first term and nothing else. Calibration is necessary and provably
insufficient.

Conformal risk control [3] supplies the missing guarantee, assuming nothing
about the model, the distribution, or the calibration of the scores. For a
bounded loss monotone in a threshold,

    λ = inf { t : (n/(n+1))·mean_i L_i(t) + 1/(n+1) ≤ α }   ⟹   E[L_test(λ)] ≤ α

over a fresh **protein**. The loss is the fraction of a chain's disordered
residues the call misses. Proteins are the exchangeable unit and the split is by
protein, so the promise is made where the assumption holds. (A per-residue
promise is *not* valid here, and the shortfall is measured rather than hidden:
at a 90% target the realised per-residue coverage on the temporal holdout is
0.876. It is quoted as a measurement of how far the assumption is strained, not
as a guarantee.)

Applied to all 70 full-coverage CAID3 Disorder-PDB methods (**Fig. 5a**):
**validity is free and identical for everyone** — the threshold is chosen to
make it so — and the entire content of the table is the price.

| # | method | flagged at risk ≤ 0.10 | at risk ≤ 0.05 | per-protein rule | credit |
|---:|---|---:|---:|---:|---:|
| **1** | **DisorderNet-windowed** | **32.0%** | **46.9%** | 55.1% | +8.2% |
| 2 | PUNCH2 | 36.2% | 47.3% | 60.4% | +13.1% |
| 3 | PUNCH2-Light | 36.8% | 48.0% | 61.3% | +13.3% |
| 6 | DisorderNet-pbias | 34.6% | 49.1% | **54.9%** | **+5.8%** |
| 70 | bindEmbed21IDR-rawGeneral | 97.3% | — | — | — |

The `per-protein` column buys the identical guarantee with a different knob:
flag a fixed **quantile of each protein's own scores** rather than apply one
global threshold. That rule depends only on the ordering inside a chain, so it
is invariant under any per-protein strictly monotone recalibration — the same
invariance `auc_within_strictMono_invariant` states for AUC_within. The two
costs therefore price the two abilities the decomposition separates: a global
threshold prices discrimination **and** calibration, a per-protein quantile
prices discrimination **alone**, and `credit` is what protein-level calibration
is worth operationally (**Fig. 5b**). The leaders lean on it about twice as hard
as we do.

This yields the sharpest single comparison in the paper (**Fig. 5c**). On
Disorder-PDB, pooled AUC separates DisorderNet from PUNCH2 by +0.0043 at
p = 0.20 — a difference this project has never been able to call real. The
calibration-invariant operating cost separates them by **5.3 points** (55.1%
against 60.4% of the protein flagged), and by 10.2 points on Disorder-NOX. *The
two methods are statistically inseparable on the benchmark's own metric and are
not close as instruments.*

To our knowledge no disorder predictor has been published with a
distribution-free operating guarantee.

### Reporting a screen: the unit of testing is a design choice too

The capacity result says the unit of *scoring* is a choice, and that scoring
pairs rather than labels changes what a benchmark can resolve. The same
structure appears one step downstream, when a predictor is run over a proteome
and asked which regions are disordered.

A screen that reports a list must control its false discovery rate, and the
candidates in a disorder screen are not independent: they share a calibration
run, a model, a training set. Under arbitrary dependence the Benjamini–Hochberg
procedure needs the harmonic correction. We prove it here from scratch, in a
finite, measure-theory-free setting, assuming only that the p-value of a true
null is valid:

- `selfConsistent_fdr_le_harmonic` — **any** self-consistent step-up rule at
  level q has E[FDP] ≤ q·H_m·|H₀|/m under an **arbitrary** joint law. No
  independence, no positive-dependence condition, nothing assumed about the
  non-nulls.
- `benjamini_yekutieli_level` — BH run at the deflated level α/H_m controls the
  rate at α, whatever the dependence.

The mechanism is isolated as `wgt_decomp`, and it is why the joint law never
enters: the weight 1/|R| of a discovery telescopes into a combination of the
**nested** events {i ∈ R, |R| ≤ j}, and self-consistency contains each of those
in the single-candidate event {p_i ≤ j·q/m}, whose probability validity alone
bounds. The price is logarithmic — log(m+1) ≤ H_m ≤ 1 + log m — against
Bonferroni's factor of m (**Fig. 7a**), and the corrected list still contains
the Bonferroni list (`by_dominates_bonferroni`).

**The correction is necessary, not conservative.** For every screen size and
level we construct an explicit joint law — outcome (j, s) gives the cyclic
window of length j+1 starting at s the p-value (j+1)q/m and everyone else 1,
with probability q/((j+1)m) — in which all m hypotheses are null, every p-value
is valid, BH rejects exactly the window, and therefore E[FDP] = q·H_m **exactly**
(`bh_fdr_eq_harmonic`). So the harmonic factor is attained and cannot be lowered
(`harmonic_factor_sharp`); uncorrected BH strictly exceeds its nominal level from
two candidates on (`uncorrected_bh_exceeds_level`, **Fig. 7c**); and the
corrected procedure sits exactly at its own bound (`by_level_is_attained`).

That makes the factor a **design parameter**, because it depends on one thing
only: how many hypotheses the screen states. And disorder is a property of a
region, not of a residue. `fdp_lift` shows that a region-level report and its
residue-level reading have the **same** false discovery proportion — both
numerator and denominator scale by the region length, and it cancels — so
`region_screen_fdr_le` gives residue-level control at α from a region-level
procedure run at α/H_M, and `harmonic_gain_log` bounds the saving below by
log b − 1.

Instantiated on this paper's own references, taking the candidate regions to be
the maximal same-label runs — the unit the annotation is actually constant on,
with a run broken by any unevaluated residue, which states more hypotheses
rather than fewer (**Fig. 7b**):

| reference | residues n | regions M | H_n | H_M | saving | threshold |
|---|---:|---:|---:|---:|---:|---:|
| CAID3 Disorder-PDB | 99,239 | 1,015 | 12.08 | 7.50 | 4.58 | **×1.61** |
| CAID3 Disorder-NOX | 99,977 | 576 | 12.09 | 6.93 | 5.16 | ×1.74 |
| CAID3 Binding | 28,263 | 165 | 10.83 | 5.69 | 5.14 | ×1.90 |
| CAID3 Linker | 20,498 | 105 | 10.51 | 5.24 | 5.27 | **×2.01** |
| CAID2 Linker | 37,150 | 124 | 11.10 | 5.40 | 5.70 | **×2.05** |

A region-level screen may test at a threshold 1.6× to 2.05× larger than a
residue-level one **for the same residue-level false discovery rate**, and by
`fdp_lift` it gives up nothing to do so. No screen is run here; what is measured
is the price of the design decision on real references, so the theorem is
instantiated rather than only cited.

### DisorderNet

DisorderNet is a frozen-backbone, low-capacity multitask head: ESM-2 650M held
fixed, a learned 33-parameter scalar mixture over its layers, a four-block
dilated CNN trunk at dilations (1, 4, 16, 32), an explicit AlphaFold structure
channel block, and one linear read-out per CAID3 task (~2M trainable
parameters). Design and rationale are in Methods.

**CAID3, `mt_windowed`** — the pre-registered model of record. Coverage 1.00 on
every reference, zero exact ties, ranks recomputed independently (**Fig. 4b**):

| benchmark | ours | rank / all | rank / full-coverage | leader (coverage) |
|---|---:|---:|---:|---|
| **Disorder-PDB** | **0.9595** | **1 / 115** | **1 / 58** | PUNCH2 0.9552 (1.00) |
| **Disorder-NOX** | **0.8928** | **1 / 115** | **1 / 58** | ESMDisPred-2PDB 0.8855 (0.89) |
| **Linker** | **0.9243** | **1 / 115** | **1 / 91** | IPA-AF2-Linker 0.8985 (0.87) |
| Binding | 0.7934 | 2 / 115 | 2 / 70 | DisoFLAG-PB 0.7760 (0.98) |
| Binding-IDR | 0.5007 | 30 / 115 | 17 / 70 | bindEmbed21IDR 0.6407 (1.00) |

The pre-registered primary family, fixed before the run and encoded in the
evaluator: **P1, beat AlphaFold-rsa — confirmed** (+0.0097, p = 0.0126, Holm
adj. 0.0252). **P2, beat PUNCH2 — not significant** (+0.0043, p = 0.2022). A
rank is a point estimate; superiority over PUNCH2 on pooled AUC is not claimed.

**CAID2** — an independent round, held out by construction, sharing exactly one
protein with CAID3. Only the two checkpoints filtered against all five
references may be scored here, and the evaluator reads the filter recorded in
the checkpoint rather than trusting the launch script. Coverage 1.00 on all
four:

| benchmark | ours (`mt_pbias`) | rank / all | rank / full-coverage | round leader |
|---|---:|---:|---:|---|
| Disorder-PDB | 0.9470 | 2 / 71 | **1 / 54** | PredIDR-long 0.9335 |
| Disorder-NOX | 0.8515 | **1 / 71** | **1 / 54** | Dispredict3 0.8378 |
| Binding | 0.8445 | **1 / 71** | **1 / 55** | Dispredict3 0.8244 |
| Linker | 0.7805 | 3 / 71 | **1 / 64** | SETH-0 0.7695 |

**First among full-coverage entrants on all four.** Every method placed above us
in the all-entrants ordering declined targets — not most of them, all of them.

**On the calibration-invariant axis, DisorderNet beats the CAID3 Disorder-PDB
winner on all five benchmarks** (**Fig. 4a**). Per-target paired Wilcoxon,
Holm-corrected across the full family of thirty comparisons (three variants ×
five benchmarks × two opponents):

| benchmark | targets | wins | mean Δ | 95% CI | Holm-adj. p |
|---|---:|---:|---:|---|---:|
| Disorder-PDB | 233 | 138 | +0.0185 | [+0.0084, +0.0293] | **0.00014** |
| Disorder-NOX | 178 | 96 | +0.0496 | [+0.0254, +0.0750] | **0.041** |
| Binding | 49 | 33 | +0.1435 | [+0.0728, +0.2215] | **0.019** |
| Binding-IDR | 42 | 27 | +0.2253 | [+0.0839, +0.3622] | **0.050** |
| **Linker** | 31 | **29** | +0.1364 | [+0.0789, +0.2060] | **0.000031** |

Against AlphaFold-rsa the same system separates on four of five (Disorder-NOX
adj. p = 0.072). On Linker we win 29 of 31 targets.

**A temporal holdout** (**Fig. 6b,c**). The training caches are dated
2026-08-08 and 2026-08-10; a structure first released on 2026-08-11 or later was
not in them, whatever any filter does. Of 1,916 protein polymer entities
released after the cutoff, 645 are usable and **186 survive after removing exact
training sequences and BLAST homologues at ≥40% identity — 71% of the "new"
chains were already represented.** A temporal cutoff alone is not a leak
control; the calendar and the homology filter are both necessary.

| model | pooled | within-protein |
|---|---:|---:|
| **`mt_windowed`** | **0.8933** | **0.8502** |
| `mt_pbias` | 0.8859 | 0.8171 |
| AlphaFold-pLDDT | 0.8737 | 0.7816 |
| AlphaFold-rsa | 0.8229 | 0.7383 |

On the 61 chains with a matching AlphaFold model, `mt_windowed` beats
AlphaFold-pLDDT by +0.0358 (95% CI [+0.0020, +0.0738], p = 0.037), which does
not survive Holm across the three comparisons (adj. 0.074). Sixty-one chains is
a small sample and the analysis was not pre-registered.

**The unflattering result the temporal set exists to produce:** `mt_pbias` is
our best CAID3 model on Disorder-PDB (0.9635) and the worst of the three here,
and the only one that fails to separate from a training-free baseline. Its
within-protein AUC on the full 186 is 0.8171 against `mt_windowed`'s 0.8502 — a
gap larger than the one between them on CAID3. **The protein-level bias term
buys CAID3 points and generalises worse**, which is exactly what a temporal
holdout is for and what a homology filter cannot produce. `mt_windowed` is the
model of record for generalisation.

### The recalibration decomposition is NP-hard to compute

If the reported number is 97–99.5% the recalibration-sensitive part, the natural
next question is how much of a given method's deficit is fixable by
recalibration alone. `Complexity/biasThreshold_hard` answers it: **that quantity
is NP-hard to compute**, proved end to end from first principles rather than by
citing Karp — a verifier definition of NP → circuit satisfiability → a verified
Tseitin transformation to CNF → independent set → weighted linear ordering → the
bias-threshold problem. Every link is an explicit total reduction with a
machine-checked correctness proof and a machine-checked polynomial bound on
output size. (Lean has no cost model; polynomiality is carried by those size
bounds on structurally simple, explicitly given functions rather than proved as
running time. This is documented in the development.)

The companion negative was proved too, and it cost us a result:
`crossed_bound_loose_unbounded` shows the greedy crossed-matching certificate we
had been using to bound irreducible error is loose by an **unbounded** factor —
one edge in the crossed graph while the true deficit exceeds any C — because the
obstruction is cyclic across three proteins and the certificate is pairwise.
That invalidated a ranking we were making, and it has been withdrawn.

No evaluation metric in structural biology has previously carried a hardness
result.

---

## Discussion

The result we would most like to see used is the capacity bound, because it is
not about disorder. Any benchmark that ranks methods on annotations of finite
accuracy has a capacity, the capacity is ⌈1/(2ε)⌉ regardless of how many items
it scores, and past that number the ordering is not a function of the
observations. The converse is what makes this different from a power
calculation: `unresolvable_pair` constructs, for any two close methods, two
ground truths consistent with everything the benchmark recorded, one favouring
each. That is not a limitation of a particular analysis. It is a statement that
no analysis exists.

The obvious objection — *then nothing can be ranked* — is answered by the second
result. The bound is stated for label noise, and a ranking statistic does not
consume labels one at a time. Because a pair reverses only when both its
residues flip in opposite directions, the noise a pairwise protocol suffers is
second-order where a label protocol's is first-order. This is a general
mechanism, not a fact about proteins: it holds for any benchmark whose statistic
is a function of ordered pairs within groups. How much it buys is a different
question, and one that has to be measured — the closed form ε_pair ≤ 2ε², which
would let a benchmark predict its own gain, needs the agreement classes to
balance, and a disorder reference at 31.6% prevalence is not close. Where the labels are noisy and the statistic is a ranking, **scoring
pairs rather than items is not a refinement, it is a different order of
resolution.**

The same shape appears once more, one step downstream. A screen's deflation
factor under arbitrary dependence is the harmonic number of the number of
hypotheses it states, that factor is attained and so cannot be argued away, and
a region-level report has the *same* false discovery proportion as its
residue-level reading. So the unit of testing is a design choice exactly as the
unit of scoring is, and choosing it well is worth a factor of two in threshold
on these references at no cost in what is reported. Two independent instances of
one principle: **where the labels are noisy or the tests are many, what a study
can resolve depends on the unit it chose, and the unit is usually chosen by
convention rather than by argument.**

For CAID specifically we recommend two changes. Report the within-protein
component alongside the pooled AUC — not because the decomposition is elegant,
but because on three of five benchmarks it changes who wins, and because we
tested and rejected the argument that AUC_within is a better predictor of
operating cost. And print a capacity: the number of entrants the reference can
order, with everything below it reported as an unresolved group rather than as a
rank. CAID4's references are unpublished, so both are adoptable now.

The operating guarantee is the part a practitioner can use tomorrow. AUC is
threshold-free, which is its virtue and its evasion: it never says what to do
with a score. Conformal risk control does, with no assumption about the model or
the distribution, and it exposes a difference between methods that AUC hides
entirely — on the temporal holdout, AlphaFold-rsa achieves a certified 10% miss
rate by flagging 96% of the protein. AUC scores it 0.82 against our 0.89 and
reads that as a modest gap. The two are not comparable instruments.

**What we do not claim.** The closed form ε_pair ≤ 2ε² is proved but does not
apply to CAID3: its balance hypothesis holds on 32 of 2,746 structure pairs, and
the pooled rate exceeds it. The sixfold gain is measured, not predicted, and a
benchmark adopting the protocol must measure its own. No proteome screen is run
here; the screening section prices a design decision and proves the procedure,
it does not report discoveries. DisorderNet is not state of the art on the
metric CAID reports: +0.0043 over PUNCH2 at p = 0.20, and no framing makes that a win. The
decomposition does not overturn the leaderboard (ρ ≥ +0.849 on every reference).
The pairwise ordering is reproducible but not shown to be *more* valid than the
pooled one. The temporal and conformal analyses are exploratory, built after the
CAID3 numbers existed; only the CAID3 primary families were pre-registered. The
operating guarantee is an expectation over proteins, not a per-protein promise.
And the capacity result is a statement about the maximum size of a certifiable
family — it is never promoted to a claim that such a family has been certified.

**What the field should take from the errors.** Several of this project's own
results were wrong and were caught by measurement rather than review: ten CAID3
numbers computed at a third of the trained receptive field, because dilation
changes no weight shape and `load_state_dict(strict=True)` accepts the mismatch
silently; a homology filter that removed nothing for weeks because a membership
test compared a string against a set of tuples while the log printed a plausible
count; a block-correlation mechanism that predicted the right sign and the wrong
magnitude by a factor of eight; a certified error ranking invalidated by our own
Lean development; and — while this manuscript was being written — a headline
capacity of 72 that formalising the denominator turned into 51, together with a
bound we had called tight whose hypothesis turns out to fail on the very data we
had checked it against. Each is documented with the measurement that caught it. A
benchmark culture that reports only the number cannot catch any of them.

---

## Methods

### Data and provenance

All CAID3 numbers derive from the official reference FASTAs
(`https://caid.idpcentral.org/assets/.../references/3/`) and the official
prediction archive (`predictions.zip`, 76,480,806 bytes, 117 `.caid` files,
aggregate md5 `cdef7ae6dc3dd6ec8055aacdb14a54a7`). Of the 117, 115 produce a
score on any given reference, which is the denominator of every rank reported
here; the capacity tables use 117, since a method that entered is a method the
benchmark was asked to place. Per-file md5s and per-target
composition counts are in Supplementary Table S12. Composition is cross-checked
against CAID's dataset API, which serves those counts separately from the files;
the evaluator refuses any reference whose composition does not match. The
dataset used is **"CAID3 v3"**, the challenge as scored; the API also serves an
entry named plainly "CAID3" (185 proteins), which is a different, smaller set
that does not reproduce the published leaderboard.

Three prediction files — `FoldUnfold`, `NeProc-binding`, `NeProc-disorder` —
leave the score field blank on declined residues. These parse to NaN, are
counted, and are never silently dropped.

**The check that licenses everything else:** each challenge's published leader is
recomputed through our own pipeline. All five reproduce to the published
precision (PUNCH2 0.9552 vs 0.955; ESMDisPred-2PDB 0.8855 vs 0.885; DisoFLAG-PB
0.7760 vs 0.776; bindEmbed21IDR-rawGeneral 0.6407 vs 0.641; IPA-AF2-Linker
0.8985 vs 0.897). If a leader stops reproducing, nothing computed afterwards is
trusted until it is explained.

### Annotation error rate

ε is estimated two independent ways from MobiDB, neither of which uses CAID's
own references (which are not independent annotations of each other and return
exactly zero disagreements).

**Context-dependent residues.** MobiDB's
`derived-missing_residues_context_dependent-th_90` annotation marks residues
missing in some deposited structures of a protein and observed in others.
Pooled over 160 CAID3 Disorder-PDB targets with a usable record: 4,886 of 61,013
evidenced residues, ε = 0.0801.

**Structure pairs.** For each protein, every pair of its deposited structures
(`derived-missing_residues-mobi-{PDBID}_{CHAIN}`), restricted to residues both
structures cover, gives a direct disagreement rate. Over 159 targets with ≥2
structures: ε_label = 0.0651 pooled, 0.0342 median protein. The same
enumeration gives the pairwise discordance ε_pair = 0.0100 pooled, 0.0006
median, on the denominator `card_comparablePairs` defines — the pairs both
structures order, 2·(|T ∩ L|·|(T ∪ L)ᶜ| + d·u), and no larger set. The same run
records the balance hypothesis rather than assuming it: 32 of 2,746 structure
pairs satisfy both hypotheses of `nuPair_le_two_eps_sq`, and the bound holds on
all 32. Per-structure *coverage* is approximated by the span of its annotated
region, because MobiDB publishes per-structure missing regions and not
per-structure coverage; residues outside a structure's span are treated as
undetermined rather than observed, which is the conservative direction.

Both rates are worst-case budgets over structure pairs, not probability models,
matching the noise model the capacity theorem uses.

### Formal development

All theorems cited are Lean 4, sorry-free, and depend only on `propext`,
`Classical.choice` and `Quot.sound`. The development is organised as: the AUC
decomposition and its invariance (`auc_pooled_decomp`,
`auc_within_strictMono_invariant`, `auc_pooled_shift_diff`,
`inversion_requires_between_gap`, `auc_shift_le_ceiling`); label noise and
certification (`LabelNoise.ranking_certified`, `RobustCertificate`); benchmark
capacity (`card_le_benchCapacity` in three score models, `benchCapacity_attained`,
`benchCapacity_noise_only`, `card_le_capacity_labelNoise`, `unresolvable_pair`,
`over_capacity_has_close_pair`); calibration (`Calibration.risk_decomposition`,
`risk_eq_resolution_of_calibrated`, `PredictionSets.validity_is_free`); and
complexity (`Complexity/biasThreshold_hard`,
`AUCCrossedMatchingLooseness.crossed_bound_loose_unbounded`). Where the
development supplies a worked example the analysis code reproduces it exactly —
`Example.inversion` returns within 1 / pooled 2/3 against within 2/3 / pooled
5/6 — which is the strongest available check that the Python computes the
statistic the theorems describe.

The pairwise development (`DiscordantPairs.lean`) is the part that most changed
this paper, so its status is given in full:

| statement | hypotheses | used here |
|---|---|---|
| `discordant_eq_flip_product` : `= 2·d·u` | none | yes, exact |
| `discordant_le_noise_sq` : `≤ ν²/2` | none | yes |
| `discordant_eq_of_balanced` : `2·discordant = ν²` | `d = u` | equality case |
| `card_comparablePairs` : `= 2(ae + du)` | none | yes, exact — and it corrected our denominator |
| `noise_orderKey` : pair noise **is** the discordant count | none | yes |
| `pairwise_capacity_bound` : `k ≤ max 1 ⌈1/(2ν_pair)⌉` | none | yes |
| `nuPair_le_two_eps_sq` : `ν_pair ≤ 2ε²` | balance, `ε ≤ 1/4` | **not applicable to CAID3** |
| `pairwise_capacity_quadratic` : `⌈1/(4ε²)⌉ ≤ k` | balance, `ε ≤ 1/4`, `d,u>0` | **not applicable to CAID3** |

`sharp_instance` exhibits the equality case on four residues and
`capacity_example` a sixteen-residue instance where every hypothesis holds, so
neither the bound nor its hypotheses are vacuous.

One statement is still used ahead of its formalisation and is flagged as such:
`auc_target_strictMono_invariant`, the per-target form of the invariance, which
is what the protocol's step 3 needs — the published theorem is stated for the
pair-weighted mean. The distinction is recorded rather than elided.

### Model

**Backbone.** ESM-2 650M, frozen. A learned scalar mixture over its 33 hidden
layers (33 parameters) replaces fine-tuning. The design follows a measurement:
in the small-data regime here (~1M evidenced residues from 2,340 proteins) a
69.9M-parameter LoRA configuration reached pooled AUC 0.7454 while a
gradient-boosted tree on hand-crafted features reached 0.7804 and AlphaFold
pLDDT alone reached 0.7906. The bottleneck was capacity/data fit, not the
backbone.

**Head.** Four residual dilated-convolution blocks, hidden width 256, dilations
(1, 4, 16, 32) — a 213-residue convolutional receptive field for the same
parameter count as the default (1, 2, 4, 8). Note that the *dependency span* is
global, not 213: the head's GroupNorm pools statistics over the whole length
axis, so every output position already depends on every input position. This is
verified directly (`lite_head.measured_dependency_span`: on a 401-residue input,
perturbing residue 0 moves the logit at residue 400). An earlier version of this
work described the field as though it bounded dependency; it does not, and the
correction changes how the protein-bias result should be read — that term did
not supply a missing channel, it supplied a better-shaped one.

**Structure channels.** rsa, pLDDT (scaled), contact density, an availability
flag, and rsa's local gradient, encoded into a small learned block. Missing
structure is explicit rather than imputed: an absent AlphaFold entry is not
"buried and confident", it is no information. Post-hoc fusion of the structural
signal *failed* (0.9581 → 0.9554); feeding structure as an input lets the trunk
learn when to trust it.

**Read-outs.** One 1×1 convolution per task (~257 parameters each) over the
shared trunk. Deliberately linear: anything deeper would let a task with 15,683
positive residues (Linker) grow private capacity, which is the failure mode the
architecture exists to avoid. The shared trunk carries disorder's 336,014
positive residues into the small tasks.

**Windowing.** Training and inference use a 1,022-residue window at stride 511
with a raised-cosine taper on the overlap. Every window of a chain is assigned
to the same side of every split, keyed on the parent sequence.

**Variants.** `mt_windowed` (no protein bias, no private trunk; the
pre-registered model of record), `mt_pbias` (adds a per-protein bias term),
and `DisorderNet-Ensemble` (equal-weight rank fusion of the checkpoints that
share the fixed validation holdout, with membership decided on the holdout and
never on CAID3, both fixed in advance in `PREREGISTRATION_10`).

### Leak control

Splits are homology-clustered with BLASTp, clustered once over the union of
proteins so a protein appearing in two tasks cannot land in different folds for
each. Benchmark targets and their ≥40%-identity homologues are removed from
training. A fixed 5% validation holdout is selected by a salted hash of the
parent sequence — membership is a property of the sequence, not of the run — so
that two runs with different leak filters remain comparable; the salt is a
module constant with no command-line override, and the holdout membership is
written into every checkpoint.

The homology filter refuses to run if BLAST is unavailable rather than holding
out nothing, because a validation number computed with a paralogue in training
is worse than none: it looks like a measurement.

### Statistics

Every p-value belongs to a family fixed in advance and Holm–Bonferroni is
applied within it (Holm rather than Bonferroni: uniformly more powerful, and it
assumes no independence, which these comparisons badly lack — they share our
predictions and share targets). The primary CAID3 family is fixed in
`PREREGISTRATION.md` and encoded in the evaluator so it cannot be redefined
after the numbers land; everything outside it is secondary, corrected within its
own family, and labelled exploratory.

Paired comparisons resample **proteins**, not residues, since residues within a
chain are heavily correlated and a residue-level interval is far too narrow.
Two-sided, 10,000 resamples for confirmatory runs; p-values use both tail
boundaries and the Davison–Hinkley +1, so a perfect null returns 1.0 rather than
0 and p is never reported below 1/(B+1).

**Coverage is a gate.** A method that declines targets is scored on an easier
benchmark: measured against a fully-covering yardstick, declined targets are
harder by +0.035 to +0.125 AUC and have median length 1,214–1,684 against 344
across all CAID3 targets. We predict every target on every benchmark; the
evaluator aborts rather than scoring a subset. Both rank fields are always
reported — rank among all scored entrants (CAID's own accounting, which embeds
the advantage) and rank among full-coverage entrants (the equal-footing field).
Reporting only the first understates us; reporting only the second is
cherry-picking. Neither alone is the truth.

### Conformal analysis

Split conformal and conformal risk control [3] as implemented in
`colab/conformal.py` (34 tests). Scores are rank-transformed within each method,
because conformal needs a consistent score and not a probability. Calibration
and evaluation are on disjoint halves of the targets, median over 12 protein
splits; the interquartile range is recorded. The implementation refuses by name
to report a guarantee at α < 1/(n+1), which is arithmetic about n rather than a
fact about any method — an earlier run at an unachievable α produced "100% for
all Linker methods" and it is now an explicit refusal.

### Screening analysis

`results/caid3/region_screen.py` (job 30101613) counts, for each CAID3 and CAID2
reference, the evaluated residues and the maximal same-label runs, and reports
both harmonic numbers, their difference, the proved floor log b − 1, and the
ratio of the two testing thresholds. A run is broken by any unevaluated residue,
so a region interrupted by missing evidence counts as two candidates — the
conservative direction, since it states more hypotheses rather than fewer.
Harmonic numbers are summed from the small end. No screen is run and no
discoveries are reported.

### Reproduction

Figures are regenerated from the analysis outputs by `paper/make_figures.py`;
each panel reads the JSON its job wrote, and the two panels that quote a table
parse it from the committed result document rather than restating it, so a
figure cannot drift from the number it illustrates. Job identifiers, sbatch
scripts and raw outputs for every analysis are listed in Supplementary Table
S13.

---

## Data availability

CAID3 and CAID2 references and prediction archives are public at
`caid.idpcentral.org`; md5s and composition counts of every file used are in
Supplementary Table S12. MobiDB per-structure annotations are public at
`mobidb.org`. The temporal holdout is constructed from RCSB
`UNOBSERVED_RESIDUE_XYZ` annotations by
`rockfish/build_temporal_holdout.py --cutoff 2026-08-11`, which is
deterministic given the cutoff.

## Code availability

Model, evaluation pipeline, analysis scripts, the Lean 4 development and the
figure generator are in the project repository. The pre-registrations
(`PREREGISTRATION.md` and `PREREGISTRATION_2` through `_10`) are committed with
timestamps predating the runs they govern.

## References

1. Necci, M., Piovesan, D., CAID Predictors, DisProt Curators & Tosatto, S.C.E.
   Critical assessment of protein intrinsic disorder prediction. *Nat. Methods*
   **18**, 472–481 (2021).
2. Conte, A.D. *et al.* Critical assessment of protein intrinsic disorder
   prediction (CAID) — Results of round 2. *Proteins* **91**, 1925–1934 (2023).
3. Angelopoulos, A.N., Bates, S., Fisch, A., Lei, L. & Schuster, T. Conformal
   risk control. *ICLR* (2024).
4. Vovk, V., Gammerman, A. & Shafer, G. *Algorithmic Learning in a Random
   World* (Springer, 2005).
5. Lin, Z. *et al.* Evolutionary-scale prediction of atomic-level protein
   structure with a language model. *Science* **379**, 1123–1130 (2023).
6. Jumper, J. *et al.* Highly accurate protein structure prediction with
   AlphaFold. *Nature* **596**, 583–589 (2021).
7. Piovesan, D. *et al.* MobiDB: 10 years of intrinsically disordered proteins.
   *Nucleic Acids Res.* **51**, D438–D444 (2023).
8. Garey, M.R. & Johnson, D.S. *Computers and Intractability* (Freeman, 1979),
   problem GT44.
9. Holm, S. A simple sequentially rejective multiple test procedure.
   *Scand. J. Stat.* **6**, 65–70 (1979).
10. Davison, A.C. & Hinkley, D.V. *Bootstrap Methods and their Application*
    (Cambridge Univ. Press, 1997).
11. de Moura, L. & Ullrich, S. The Lean 4 theorem prover and programming
    language. *CADE* (2021).
12. Benjamini, Y. & Hochberg, Y. Controlling the false discovery rate: a
    practical and powerful approach to multiple testing. *J. R. Stat. Soc. B*
    **57**, 289–300 (1995).
13. Benjamini, Y. & Yekutieli, D. The control of the false discovery rate in
    multiple testing under dependency. *Ann. Stat.* **29**, 1165–1188 (2001).

---

## Figures

**Figure 1 — CAID's statistic is a calibration contest.**
(**a**) Share of the metric's comparison pairs that are within-protein versus
between-protein, for each CAID3 reference. (**b**) The declared winner's rank on
the within-protein axis. (**c**) Inversion certificates on Disorder-NOX: the
measured between-protein gap divided by the gap `inversion_requires_between_gap`
requires, log scale.

**Figure 2 — Capacity, and the way past it.**
(**a**) Methods orderable, ⌈1/2ε⌉, against annotation error rate, with the
measured ε marked and the 117-entrant line for scale. (**b**) Capacity under the
residue-level and pairwise protocols. (**c**) The balance hypothesis
`nuPair_le_two_eps_sq` requires, measured on 2,746 structure pairs.

**Figure 3 — CAID3 re-scored.**
(**a**) Pooled rank against pairwise rank, all 357 method–benchmark pairs, with
the identity line. (**b**) The 14 largest rank changes; dot = pooled,
arrowhead = pairwise. (**c**) Entrants the leader is separated from at the
certified margin, per benchmark.

**Figure 4 — DisorderNet.**
(**a**) Per-target within-protein margins against PUNCH2 and AlphaFold-rsa,
95% CI, p Holm-adjusted over all 30 comparisons. (**b**) Placement on both
rounds under each benchmark's own reported metric.

**Figure 5 — The price of a guarantee.**
(**a**) Realised miss rate against fraction of protein flagged, all 70
full-coverage Disorder-PDB methods, at α = 0.10. (**b**) The same guarantee
bought two ways: one global threshold versus a per-protein quantile; the number
is the calibration credit. (**c**) The comparison AUC cannot make.

**Figure 7 — The price of a screen, and where to pay it.**
(**a**) The deflation factor against the number of hypotheses stated: harmonic
under arbitrary dependence, against Bonferroni's linear factor, with the
residue-level and region-level counts for CAID3 Disorder-PDB marked.
(**b**) The saving from stating one hypothesis per region rather than per
residue, per reference, against the log b − 1 floor `harmonic_gain_log` proves.
(**c**) Sharpness: on the constructed joint law, uncorrected BH realises
E[FDP] = q·H_m exactly, so it exceeds its nominal level from two candidates on.

**Figure 6 — Reproducibility and generalisation.**
(**a**) Cross-round rank correlation, CAID3 → held-out CAID2, 57 shared
entrants. (**b**) Construction of the temporal holdout. (**c**) Performance on
186 chains released after the training caches were built.
