# What would help most, in order

Concrete asks for the Lean development, ranked by how much each would change
what this paper can claim. Each is stated as a target theorem with the pieces
that already exist, so none of them starts from nothing.

---

## 1. NP-hardness of the recalibration optimum — **DONE**

`Complexity/biasThreshold_hard`, proved end to end from first principles rather
than by citing Karp: a verifier definition of NP → circuit satisfiability → a
verified Tseitin transformation to CNF → independent set → weighted linear
ordering → the bias-threshold problem. Every link is an explicit total reduction
with a machine-checked correctness proof and a machine-checked polynomial bound
on output size, and the caveat is documented in `Problem.lean`: Lean has no cost
model, so polynomiality is carried by those size bounds on structurally simple,
explicitly given functions rather than proved as running time.

**And the companion negative was proved too**, which this project did not ask
for and which matters more to its own numbers:
`AUCCrossedMatchingLooseness.crossed_bound_loose_unbounded` shows the greedy
crossed-matching certificate is loose by an unbounded factor — one edge in the
crossed graph while the true deficit exceeds any `C` — because the obstruction is
cyclic across three proteins and the certificate is pairwise. That invalidated a
ranking `CERTIFIED.md` was making and it has been corrected.

The original ask is preserved below for the record.

### (original) NP-hardness of the recalibration optimum

`AUC_THEOREMS.md` already says the exact optimum over per-protein biases is a
weighted linear ordering problem (`exists_order_ge`, `exists_bias_of_order`),
and then explicitly declines the hardness claim:

> A full NP-hardness statement would require reducing an arbitrary weighted
> linear ordering instance to a *score table*; that reduction is **not**
> formalised here and no NP-hardness claim is made.

**That reduction is the missing theorem, and it is the one worth having.**

What it would establish, in plain terms: *given a method's scores on a
benchmark, computing how much of its deficit is fixable by per-protein
recalibration is NP-hard.* Since the certified analysis shows CAID's metric is
97–99.5% the recalibration-sensitive part, that is a statement that **the
benchmark's own headline number cannot in general be decomposed into "skill"
and "calibration" in polynomial time.**

No evaluation metric in structural biology has a hardness result attached to
it. This one would.

### The target

```
theorem bias_optimum_NP_hard :
    ∀ (W : Fin K → Fin K → ℕ),           -- an arbitrary LOP instance
    ∃ (lab : ι → Bool) (grp : ι → Fin K) (s : ι → ℝ),
      ∀ π : Equiv.Perm (Fin K),
        (max over b of U_pooled (s + b∘grp))
          = U_within s + (LOP objective of π on W) + const
```

i.e. **every** weighted linear ordering instance is realised by some score
table, so maximising pooled AUC over per-protein biases is at least as hard as
LOP, which is NP-hard (Karp; Garey–Johnson GT44).

### What already exists to build on

- `exists_order_ge`, `exists_bias_of_order` — the equivalence in the
  well-separated regime, which is the hard direction of the *correspondence*.
- `exists_gapMatrix`, `feasible_iff_gap` — the reduction of the residue-level
  system to a `K × K` matrix, which is where an arbitrary `W` would be
  installed.
- `AUCCeilingExamples.lean` — two worked score tables, so the encoding is
  already known to be constructible by hand.

### What is actually needed

A **gadget**: given target weights `w k l`, a construction placing positives and
negatives of proteins `k` and `l` at scores that make the number of
cross-protein comparisons decided by `b k − b l` equal to `w k l`. The
separable gap matrix `c k l = maxneg l − minpos k` suggests placing one positive
and `w k l` negatives per ordered pair at arranged offsets. If the gadget needs
weights bounded by a polynomial in `K`, that is fine — LOP is NP-hard for
0/1 weights.

**A weaker but still valuable version**, if the full reduction is hard: hardness
of the *decision* problem "is the ceiling attainable minus `m` pairs?" for
arbitrary `m`, or APX-hardness, or even just a family of instances where the
greedy crossed-matching bound is provably loose by an unbounded factor. Any of
those would let the paper say the quantity is not merely uncomputed but
uncomputable in practice.

---

## 1b. The general capacity bound — **now the most valuable open item**

The empirical instance is done: at the annotation error rate measured from
MobiDB's per-structure disagreements (`ε = 0.0801`), `ranking_certified` leaves
**30 of 45** comparisons unresolvable on CAID3 Disorder-PDB and **41 of 45** on
Disorder-NOX. What is missing is the general statement:

```
theorem benchmark_capacity {n : ℕ} (eps : ℝ) (methods : Finset M)
    (err : M → ℕ) (h : ∀ m, err m ≤ n) :
    (certifiable eps n err).card ≤ f n eps   -- an explicit bound
```

i.e. *a benchmark with `n` scored items and annotation error rate `eps` can
certify at most `f n eps` pairwise orderings*, plus **sharpness**: an instance
attaining it.

Why this is worth more than the hardness result. NP-hardness says the
recalibration gap is expensive to compute. The capacity bound says what a
benchmark can **ever** establish, whatever anyone computes and however many
resamples they take — and it applies to every benchmark with noisy labels, which
is all of them. That is a statement that leaves this field.

The pieces exist: `ranking_certified` is the core, and the counting argument is
elementary once the margins are ordered. Sharpness needs an instance where the
margins are spaced exactly at the bar.

## 2. Conformal risk control for a grouped loss

`GUARANTEES.md` reports the first distribution-free operating guarantee for
disorder prediction, and it rests on conformal risk control, which is cited
rather than formalised. What would strengthen it:

```
theorem crc_valid {n : ℕ} (L : Fin (n+1) → ℝ → ℝ)
    (hmono : ∀ i, Antitone (L i)) (hbdd : ∀ i t, L i t ∈ Set.Icc 0 1)
    (hexch : Exchangeable L) :
    let lam := sInf {t | (n / (n+1) : ℝ) * (∑ i in range n, L i t) / n
                         + 1/(n+1) ≤ α}
    𝔼 [L n lam] ≤ α
```

The proof is short — monotonicity plus the exchangeability of the `n+1` losses
— and having it machine-checked would make the paper's one *guaranteed* claim
rest on a proof rather than a citation. This is the claim a referee is most
likely to probe, since everything else in the paper is a measurement.

**Also worth having:** the *negative* companion, that a group-level split does
**not** give a per-element guarantee. We measure a shortfall of 0.876 against a
0.90 target and currently explain it informally; a theorem exhibiting two
groupings with identical group-level validity and arbitrarily bad element-level
coverage would turn that paragraph into a result.

---

## 3. Realisability hardness — beautiful, and for the other paper

`DistanceRealizability` / `DistanceCutEnsembles` prove ensembles realise the
whole cut cone truncated at the contour scale. The cut cone is exactly the
ℓ¹-embeddable metrics, and membership in it is NP-hard (Avis–Deza; equivalent to
max-cut). So the natural statement is:

> **Deciding whether a measured mean-distance panel is realisable by any
> ensemble is NP-hard.**

The forward inclusion is done. The obstacle is that ensemble panels are convex
combinations of *Euclidean* metrics, a cone strictly between the cut cone and
the metric cone, so containment alone does not transfer hardness — a reduction
landing inside the sub-family where the two agree is needed. If it works it is a
striking result about experimental IDR data, and it belongs to the ensemble
paper rather than to DisorderNet.

---

## 4. Not needed

The single-molecule certificate machinery (Parts CXLVI–CLII: FRET, SAXS,
transport, entropy floors) is excellent and none of it applies to a benchmark
scored on per-residue labels. It belongs entirely to the ensemble paper. No
further work there helps DisorderNet.
