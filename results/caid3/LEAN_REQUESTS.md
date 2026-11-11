# What would help most, in order

Concrete asks for the Lean development, ranked by how much each would change
what the paper can claim. Each is stated as a target theorem with the pieces
that already exist, so none of them starts from nothing.

**Closed since the last revision.** `biasThreshold_hard` (NP-hardness of the
recalibration optimum, from first principles) and its companion negative
`crossed_bound_loose_unbounded`; the capacity theorem in three score models with
attainment, `unresolvable_pair` and `over_capacity_has_close_pair`; and the
whole of `DiscordantPairs.lean`, `DependentBH.lean`, `DependentBHSharp.lean`,
`RegionScreen.lean`. Formalising the third of those **refuted two numbers this
paper was publishing** — the pairwise capacity (72 → 51) and the claim that
`ε_pair ≤ 2ε²` held on our data. That is the standard the list below is written
to.

---

## 1. The imbalance-corrected pair bound — **verified empirically, not proved**

This is the highest value per unit of effort on the list, and the only item that
recovers something the last round took away.

`nuPair_le_two_eps_sq` needs `BalancedClasses T L`, and CAID3 fails it: balance
holds on **54 of 2,746** structure pairs, both hypotheses on **32**, and the
pooled rate exceeds the bound. So the paper currently has **no closed form at
all** on its own references and must measure `ε_pair` per benchmark. The
following restores one, with no balance hypothesis:

```lean
/-- The imbalance factor: `1` exactly when the agreement classes balance. -/
noncomputable def kappa (T L : Finset α) : ℝ :=
  ((T ∩ L).card + ((T ∪ L)ᶜ).card) ^ 2 / (4 * (T ∩ L).card * ((T ∪ L)ᶜ).card)

theorem nuPair_le_imbalanced {T L : Finset α}
    (ha : 0 < (T ∩ L).card) (he : 0 < ((T ∪ L)ᶜ).card) (hv : 0 < noise T L) :
    nuPair T L ≤ kappa T L * eps ^ 2 / (1 - eps) ^ 2
  where eps := (noise T L : ℝ) / Fintype.card α
```

**The proof is three lines of the one already written.** Drop `d·u` from the
denominator of `nuPair` (`nuPair ≤ du / ae`); apply the AM–GM step already
proved as `h4du` (`4du ≤ ν²`); and substitute `a + e = n − ν = n(1 − ε)` from
`card_split`, which gives `ae = n²(1−ε)²/(4·kappa)` by definition of `kappa`.
No new machinery. `kappa = 1` recovers the balanced case, and
`kappa ≤ 2(1−ε)²` recovers `2ε²` exactly, which is worth stating as a corollary
so the published bound becomes a special case rather than a separate result.

**Checked before asking.** On all 2,746 CAID3 structure pairs, restricted to the
2,266 where both agreement classes are nonempty, the inequality holds on
**2,266 — every one.** (`relative_noise.py`, job 30101834; the field is
`structure_pairs.imbalanced_holds`.) The published bound holds on 2,248 of the
full 2,746, and on only 32 does it hold *under its own hypotheses*.

**What it buys.** A closed form on real references — a benchmark could predict
its own pairwise gain from its label-noise rate and its class prevalence, which
is exactly what the balance failure cost us. **What it does not buy**, and the
paper should not claim it does: a *pooled* prediction. The pooled rate is a ratio
of sums over heterogeneous structure pairs, and a per-instance bound gives only
the weighted average of the per-instance bounds, not a formula in the pooled
`ε` and prevalence. Worth stating that weighted-average corollary too, since it
is what a benchmark would actually compute.

---

## 2. Conformal p-values are superuniform — the two halves become one instrument

The paper currently has two operational results that do not touch:
the conformal operating guarantee (§ "An operating guarantee, and the price of
it") and the dependent-screen result (§ "Reporting a screen"). `DependentBH`
assumes `Superuniform P X` and says nothing about where such a p-value comes
from. Split conformal produces one.

```lean
/-- The split-conformal p-value of a fresh point against `n` calibration
scores, under exchangeability, is superuniform. -/
theorem conformal_pvalue_superuniform {n : ℕ} (P : Law Ω)
    (s : Fin (n+1) → Ω → ℝ) (hexch : Exchangeable P s) :
    Superuniform P (fun ω =>
      (1 + (Finset.univ.filter (fun i : Fin n => s i.castSucc ω ≥ s (Fin.last n) ω)).card)
        / (n + 1 : ℝ))
```

The proof is the standard rank argument: under exchangeability the rank of the
test score among the `n+1` scores is uniform on `Fin (n+1)`, so the p-value
exceeds `k/(n+1)` with the right probability. The development already has
`Exchangeable` (it appears in the `crc_valid` statement below) and the finite
`Law Ω` setting, so this is stated in machinery that exists.

**Then the composition, which is the actual prize:**

```lean
theorem conformal_region_screen_fdr (P : Law Ω) {α : ℝ} (hα : 0 ≤ α)
    (B : Blocks n M b) (hexch : ∀ i, Exchangeable P (calibration scores for region i)) :
    residue-level E[FDP] of (BY at level α / harm M on conformal p-values) ≤ α
```

**Why this matters more than either half.** Conformal p-values for different
candidate regions are *not* independent — they share one calibration set, by
construction. That is precisely the dependence structure
`selfConsistent_fdr_le_harmonic` was proved to survive. So the composition is
not two results bolted together: the arbitrary-dependence theorem is the
*right* tool rather than a conservative one, **because** the calibration step
that supplies valid p-values is what couples them. Proving it turns two
disconnected sections into one end-to-end pipeline with a guarantee at both
ends, and it is the strongest structural claim available to this paper.

---

## 3. `crc_valid` — the one guaranteed claim still rests on a citation

Unchanged from the last round and still the claim a referee probes first,
because everything else in the paper is a measurement.

```lean
theorem crc_valid {n : ℕ} (L : Fin (n+1) → ℝ → ℝ)
    (hmono : ∀ i, Antitone (L i)) (hbdd : ∀ i t, L i t ∈ Set.Icc 0 1)
    (hexch : Exchangeable L) :
    let lam := sInf {t | (n / (n+1) : ℝ) * (∑ i in range n, L i t) / n
                         + 1/(n+1) ≤ α}
    𝔼 [L n lam] ≤ α
```

Short proof — monotonicity plus exchangeability of the `n+1` losses. Note that
item 2 above needs the same `Exchangeable` infrastructure, so doing 3 first
makes 2 cheaper.

---

## 4. A certified chain is not a separation count

**This one exists because the paper got it wrong.** `PROTOCOL.md` claimed
DisorderNet was "separated from 55 of 59 eligible methods, and inside the
capacity of 51". Fifty-five is not inside fifty-one, and on Linker the same
table reports 87 separations against the same capacity of 51.

The numbers are both correct and they answer different questions:
`pairwise_capacity_bound` limits a certified **totally ordered family**, while a
separation count is `k` statements about the pairs `{leader, X}`, which need not
compose into a chain of `k+1`. The paper now says so in prose. A theorem would
make it precise:

```lean
/-- The longest certified chain is at most the capacity … -/
theorem chain_le_capacity {T L : Finset α} (C : Finset M) (hC : CertifiedChain C) :
    C.card ≤ pairwiseCapacity T L

/-- … while the number of certified *pairs* can exceed it without contradiction. -/
theorem separations_exceed_capacity :
    ∃ (instance) (leader : M), pairwiseCapacity T L < (certifiedAgainst leader).card
```

The first is presumably how `pairwise_capacity_bound` is already stated, in
which case it is a naming and exposition fix rather than new work. The second —
an instance with many pairwise separations and a short maximum chain — is the
part that actually settles the reader's objection, and it is small: a star
configuration where one method beats everything and the rest are mutually
inseparable should do it.

---

## 5. Capacity when `ε` is estimated rather than known

The capacity theorem takes `ε` as a **worst-case budget**. The paper feeds it a
**rate measured on 159 proteins and 2,746 structure pairs**, and then applies the
result to 233 targets. That step is currently prose. It is the most attackable
join in the argument and nothing in the development covers it.

```lean
theorem capacity_under_estimated_noise {m : ℕ} (γ : ℝ)
    (hat_eps : ℝ) (h : estimated from m exchangeable repeat determinations) :
    P(certifiable family size ≤ benchCapacity (hat_eps + dev m γ)) ≥ 1 - γ
```

with `dev m γ` an explicit deviation term (Hoeffding is enough; the quantities
are bounded in `[0,1]`). `RobustCertificate.certificate_under_mean_error`
already degrades a bound continuously in an input known only to within `δ`, so
the shape exists — what is missing is the statistical half that turns `m` repeat
determinations into a `δ`.

**What it would let the paper say:** *at 95% confidence, CAID3 can order at most
K of its 117 entrants* — a defensible sentence, where the current one silently
treats a sample statistic as a hard budget.

---

## 6. A counting converse

`over_capacity_has_close_pair` gives the existence of **one** unresolvable pair
once the field exceeds capacity. The paper wants to say how many.

```lean
theorem count_unresolvable {ε : ℝ} (methods : Finset M) (h : capacity < methods.card) :
    (unresolvablePairs ε methods).card ≥ f methods.card (capacity)
```

with `f` explicit — a pigeonhole on the score grid should give something like
`⌊k/c⌋` pairs at spacing below the budget, where `k` is the field size and `c`
the capacity. **Why it is worth having:** "at least one pair is unresolvable" is
a footnote; "at least N of the 6,786 pairwise comparisons in CAID3's top field
are unresolvable, and no analysis of these data can settle any of them" is the
sentence people will quote. The empirical instance already exists
(`ranking_certified` leaves 30 of 45 and 41 of 45 unresolvable on the two
disorder benchmarks); this makes it a theorem rather than a tally.

---

## 7. `auc_target_strictMono_invariant` — small, and still open

The protocol's step 3 scores the **unweighted** mean of per-target AUCs.
`auc_within_strictMono_invariant` is stated for the **pair-weighted** mean, which
the decomposition identity requires. The invariance holds target by target
before any averaging, so the per-target statement implies both:

```lean
theorem auc_target_strictMono_invariant (f : ℝ → ℝ) (hf : StrictMono f) :
    aucOn t (f ∘ s) = aucOn t s
```

The protocol's central property should rest on a theorem about the statistic it
actually uses. Likely a short refactor of the existing proof.

**A companion worth having while in there:** the two means are different
statistics, and the paper reports both without relating them. A bound on
`|unweighted mean − pair-weighted AUC_within|` in terms of the spread of
per-target pair counts would let it say by how much they can disagree, instead of
reporting the pair-weighted one and hoping the unweighted one agrees.

---

## 8. The conformal negative companion

A group-level split does **not** give a per-element guarantee. We measure a
realised per-residue coverage of 0.876 against a 0.90 target and currently
explain it in prose. A theorem exhibiting two groupings with identical
group-level validity and arbitrarily bad element-level coverage would turn that
paragraph into a result — and it is the honest half of the operating guarantee,
which is the section most likely to be over-read.

---

## Not needed

The single-molecule certificate machinery (FRET, SAXS, transport, entropy
floors) and the ensemble realisability results (`DistanceRealizability`,
`DistanceCutEnsembles`) are excellent and none of it applies to a benchmark
scored on per-residue labels. It belongs to the ensemble paper. No further work
there helps DisorderNet.
