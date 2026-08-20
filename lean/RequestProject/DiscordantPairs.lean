/-
# Discordant pairs: an exact second-order identity, and the quadratic capacity it buys

`RequestProject.LabelNoise` and `RequestProject.BenchmarkCapacityLabels` price a benchmark that
scores a method residue by residue against an annotation `L` of a truth `T`: a comparison is
certified only when the true gap exceeds twice the annotation error, so a benchmark whose
labels are a fraction `ε` wrong orders at most about `1 / (2 ε)` methods.

`RequestProject.PairwiseProtocol` changes the protocol to score *pairs* of residues and proves
the same capacity law with the pair-level rate substituted, together with a block-length bound
on that rate.  This file replaces the block-length mechanism by an **exact second-order
identity**, which is what actually drives the gain.

Write `d = |T \ L|` for the residues the annotation wrongly calls ordered and `u = |L \ T|` for
those it wrongly calls disordered -- the two *flip classes* -- so that the ordinary label noise
is `ν = |T Δ L| = d + u`.

* `discordant_eq_flip_product` -- **the identity.**  An ordered pair `(p, q)` is *discordant*
  when the truth and the annotation order it oppositely.  Since each labelling only orders a
  pair by calling one residue disordered and the other ordered, a reversal needs *both* residues
  to flip, and in opposite directions: `p ∈ T \ L` and `q ∈ L \ T`, or the other way round.
  Hence

      discordantPairs T L = 2 · d · u.

  Not a bound -- an identity.

* `discordant_le_noise_sq` -- **the consequence.**  `2 d u ≤ (d + u)² / 2 = ν² / 2`, by AM-GM,
  with equality exactly when the two flip classes balance (`discordant_eq_of_balanced`).  The
  pair statistic is *second order* in the noise.

* `pairwise_capacity_quadratic` -- **the capacity corollary.**  Scoring the pairs that both
  labellings order (`comparablePairs`), the pair-level noise is exactly the discordant count
  (`noise_orderKey`), so the capacity theorem of `RequestProject.BenchmarkCapacity` applies with
  the pair rate `ν_pair = 2 d u / |comparablePairs|` (`pairwise_capacity_bound`).  With balanced
  classes -- as many residues both labellings call disordered as both call ordered -- and a
  noise rate below `1/4`,

      ν_pair ≤ 2 ε²      and therefore      pairwiseCapacity ≥ ⌈1 / (4 ε²)⌉.

  The residue protocol pays `1 / (2 ε)`; the pairwise protocol pays `1 / (4 ε²)`.  The capacity
  is quadratic in `1 / ε` instead of linear.  `capacity_at_eps_0651` evaluates both at
  `ε = 0.0651`: `8` methods residue-wise against `59` pairwise.

On the block-length bound of `RequestProject.PairwiseProtocol`
(`correlated_noise_reduces_pair_discordance`): it is left in place -- it is a correct and
independent statement, about a different mechanism (noise that is constant on blocks only
exposes block boundaries).  It is not used here, and nothing below depends on it; the gain
proved here comes from the order of the statistic in the noise, not from block structure.
-/
import Mathlib
import RequestProject.LabelNoise
import RequestProject.BenchmarkCapacity
import RequestProject.BenchmarkCapacityLabels

set_option autoImplicit false

namespace IDR
namespace Discordant

open Finset
open IDR.LabelNoise
open IDR.BenchCapacity

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## 1.  The identity -/

/-- The **discordant pairs** of two labellings: the ordered pairs `(p, q)` that the truth `T`
and the annotation `L` order oppositely.  A labelling orders a pair by calling the first residue
disordered and the second ordered, so a reversal forces both residues to flip, in opposite
directions. -/
def discordantSet (T L : Finset α) : Finset (α × α) :=
  Finset.univ.filter
    (fun p => (p.1 ∈ T \ L ∧ p.2 ∈ L \ T) ∨ (p.1 ∈ L \ T ∧ p.2 ∈ T \ L))

/-- The number of discordant ordered pairs. -/
def discordantPairs (T L : Finset α) : ℕ := (discordantSet T L).card

lemma mem_discordantSet {T L : Finset α} {p : α × α} :
    p ∈ discordantSet T L ↔
      ((p.1 ∈ T ∧ p.1 ∉ L ∧ p.2 ∉ T ∧ p.2 ∈ L) ∨ (p.1 ∈ L ∧ p.1 ∉ T ∧ p.2 ∈ T ∧ p.2 ∉ L)) := by
  simp only [discordantSet, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_sdiff]
  tauto

/-- **The identity.**  Discordance is exactly twice the product of the two flip classes. -/
theorem discordant_eq_flip_product (T L : Finset α) :
    discordantPairs T L = 2 * (T \ L).card * (L \ T).card := by
  classical
  have hsplit : discordantSet T L = ((T \ L) ×ˢ (L \ T)) ∪ ((L \ T) ×ˢ (T \ L)) := by
    ext p
    simp only [discordantSet, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union,
      Finset.mem_product]
  have hdisj : Disjoint ((T \ L) ×ˢ (L \ T)) ((L \ T) ×ˢ (T \ L)) := by
    rw [Finset.disjoint_left]
    rintro p hp hq
    rw [Finset.mem_product] at hp hq
    exact (Finset.mem_sdiff.mp hq.1).2 (Finset.mem_sdiff.mp hp.1).1
  rw [discordantPairs, hsplit, Finset.card_union_of_disjoint hdisj, Finset.card_product,
    Finset.card_product]
  ring

/-! ## 2.  The second-order bound -/

omit [Fintype α] in
/-- The label noise splits into the two flip classes. -/
lemma noise_eq_flip_add (T L : Finset α) :
    noise T L = (T \ L).card + (L \ T).card := by
  rw [noise, show symmDiff T L = (T \ L) ∪ (L \ T) from rfl,
    Finset.card_union_of_disjoint (Finset.disjoint_left.mpr (by
      intro a ha hb
      exact (Finset.mem_sdiff.mp hb).2 (Finset.mem_sdiff.mp ha).1))]

/-- **The consequence.**  Discordance is at most half the square of the noise: `2 d u ≤ ν²/2`,
which is AM-GM on the two flip classes. -/
theorem discordant_le_noise_sq (T L : Finset α) :
    discordantPairs T L ≤ (noise T L) ^ 2 / 2 := by
  rw [discordant_eq_flip_product, noise_eq_flip_add]
  rw [Nat.le_div_iff_mul_le (by norm_num)]
  nlinarith [Nat.sub_add_cancel (le_max_left (T \ L).card (L \ T).card),
    sq_nonneg ((T \ L).card - (L \ T).card : ℤ),
    Nat.zero_le ((T \ L).card), Nat.zero_le ((L \ T).card)]

/-- The same bound over the reals, where the division is exact. -/
theorem discordant_le_noise_sq_real (T L : Finset α) :
    (discordantPairs T L : ℝ) ≤ (noise T L : ℝ) ^ 2 / 2 := by
  have h := discordant_eq_flip_product T L
  have hn := noise_eq_flip_add T L
  have : ((discordantPairs T L : ℕ) : ℝ) = 2 * ((T \ L).card : ℝ) * ((L \ T).card : ℝ) := by
    rw [h]; push_cast; ring
  rw [this, hn]
  push_cast
  nlinarith [sq_nonneg (((T \ L).card : ℝ) - ((L \ T).card : ℝ))]

/-- **Sharpness of the bound.**  When the two flip classes balance the inequality is an
equality: `2 · discordance = ν²`. -/
theorem discordant_eq_of_balanced {T L : Finset α} (h : (T \ L).card = (L \ T).card) :
    2 * discordantPairs T L = (noise T L) ^ 2 := by
  rw [discordant_eq_flip_product, noise_eq_flip_add, ← h]
  ring

/-! ## 3.  The pairwise protocol scored on comparable pairs -/

/-- A labelling **separates** an ordered pair when it calls exactly one of the two residues
disordered -- that is, when it orders the pair. -/
def Separates (S : Finset α) (p : α × α) : Prop :=
  (p.1 ∈ S ∧ p.2 ∉ S) ∨ (p.2 ∈ S ∧ p.1 ∉ S)

instance decidableSeparates (S : Finset α) (p : α × α) : Decidable (Separates S p) := by
  unfold Separates; infer_instance

/-- The **comparable pairs** of the protocol: the ordered pairs that both the truth and the
annotation order.  These are the pairs the benchmark can score. -/
def comparablePairs (T L : Finset α) : Finset (α × α) :=
  Finset.univ.filter (fun p => Separates T p ∧ Separates L p)

/-- The type of comparable pairs, the targets of the pairwise benchmark. -/
abbrev ComparablePair (T L : Finset α) := {p : α × α // p ∈ comparablePairs T L}

/-- The answer key of the pairwise protocol induced by a labelling `S`: the comparable pairs it
orders "first disordered, second ordered". -/
def orderKey (T L : Finset α) (S : Finset α) : Finset (ComparablePair T L) :=
  Finset.univ.filter (fun p => (p : α × α).1 ∈ S ∧ (p : α × α).2 ∉ S)

lemma mem_comparablePairs {T L : Finset α} {p : α × α} :
    p ∈ comparablePairs T L ↔ Separates T p ∧ Separates L p := by
  simp [comparablePairs]

lemma discordantSet_subset_comparablePairs (T L : Finset α) :
    discordantSet T L ⊆ comparablePairs T L := by
  intro p hp
  rw [mem_discordantSet] at hp
  rw [mem_comparablePairs]
  rcases hp with ⟨h1, h2, h3, h4⟩ | ⟨h1, h2, h3, h4⟩
  · exact ⟨Or.inl ⟨h1, h3⟩, Or.inr ⟨h4, h2⟩⟩
  · exact ⟨Or.inr ⟨h3, h2⟩, Or.inl ⟨h1, h4⟩⟩

/-- **The noise of the pairwise answer key is exactly the discordant count.**  On a pair both
labellings order, they differ precisely when they order it oppositely. -/
lemma noise_orderKey (T L : Finset α) :
    noise (orderKey T L T) (orderKey T L L) = discordantPairs T L := by
  classical
  have hset : symmDiff (orderKey T L T) (orderKey T L L)
      = Finset.univ.filter (fun p : ComparablePair T L => (p : α × α) ∈ discordantSet T L) := by
    ext p
    obtain ⟨q, hq⟩ := p
    rw [mem_comparablePairs] at hq
    simp only [Finset.mem_symmDiff, orderKey, Finset.mem_filter, Finset.mem_univ, true_and,
      mem_discordantSet]
    unfold Separates at hq
    tauto
  rw [noise, hset, discordantPairs]
  refine Finset.card_bij (fun p _ => (p : α × α)) ?_ ?_ ?_
  · intro p hp
    simpa using hp
  · intro p _ q _ h
    exact Subtype.ext h
  · intro p hp
    exact ⟨⟨p, discordantSet_subset_comparablePairs T L hp⟩, by simpa using hp, rfl⟩

/-- The **pair error rate** `ν_pair`: the fraction of comparable pairs the annotation orders
backwards. -/
noncomputable def nuPair (T L : Finset α) : ℝ :=
  (discordantPairs T L : ℝ) / (comparablePairs T L).card

/-- The **capacity of the pairwise benchmark**: the number of methods it can place in a
certified order, `⌈1 / (2 ν_pair)⌉` (with the degenerate one-method case, as elsewhere). -/
noncomputable def pairwiseCapacity (T L : Finset α) : ℕ := max 1 ⌈1 / (2 * nuPair T L)⌉₊

/-- The **capacity of the residue benchmark** at noise rate `eps`, for comparison:
`⌈1 / (2 ε)⌉`. -/
noncomputable def residueCapacity (eps : ℝ) : ℕ := max 1 ⌈1 / (2 * eps)⌉₊

lemma card_comparablePair (T L : Finset α) :
    Fintype.card (ComparablePair T L) = (comparablePairs T L).card :=
  Fintype.card_coe (comparablePairs T L)

/-- **The capacity theorem for the pairwise protocol.**  A family of methods whose true
pair-scores are separated by more than twice the discordant count -- the certification
condition -- has at most `pairwiseCapacity T L` members. -/
theorem pairwise_capacity_bound {iota : Type*} {T L : Finset α}
    {pred : iota → Finset (ComparablePair T L)} {M : Finset iota}
    (hP : 0 < (comparablePairs T L).card) (hnu : 0 < discordantPairs T L)
    (hcert : Certified (orderKey T L T) (discordantPairs T L) pred M) :
    M.card ≤ pairwiseCapacity T L := by
  classical
  set P : ℕ := (comparablePairs T L).card with hPdef
  set d : ℕ := discordantPairs T L with hddef
  have hcard : Fintype.card (ComparablePair T L) = P := card_comparablePair T L
  have hPR : (0:ℝ) < (P : ℝ) := by exact_mod_cast hP
  have heps : (0:ℝ) < (d : ℝ) / P := by
    have : (0:ℝ) < (d : ℝ) := by exact_mod_cast hnu
    positivity
  have hbudget : ((d : ℝ) / P) * (Fintype.card (ComparablePair T L) : ℝ) ≤ (d : ℕ) :=
    le_of_eq (by rw [hcard, div_mul_cancel₀ _ (ne_of_gt hPR)])
  have h1 : M.card ≤ benchCapacity (Fintype.card (ComparablePair T L)) ((d : ℝ) / P) 0 :=
    card_le_benchCapacity_of_rate (by rw [hcard]; exact hP) heps.le hbudget hcert
  have h2 : benchCapacity (Fintype.card (ComparablePair T L)) ((d : ℝ) / P) 0
      ≤ max 1 ⌈1 / (2 * ((d : ℝ) / P))⌉₊ :=
    benchCapacity_noise_only _ (by rw [hcard]; exact hP) heps
  exact h1.trans h2

/-! ## 4.  Counting the comparable pairs, and the quadratic capacity -/

/-- The comparable pairs split into the pairs on which the two labellings agree in opposite
directions (`T ∩ L` against the residues neither calls disordered) and the discordant pairs. -/
theorem card_comparablePairs (T L : Finset α) :
    (comparablePairs T L).card
      = 2 * ((T ∩ L).card * ((T ∪ L)ᶜ).card + (T \ L).card * (L \ T).card) := by
  classical
  have hsplit : comparablePairs T L =
      (((T ∩ L) ×ˢ ((T ∪ L)ᶜ)) ∪ (((T ∪ L)ᶜ) ×ˢ (T ∩ L))) ∪ discordantSet T L := by
    ext p
    simp only [mem_comparablePairs, Separates, Finset.mem_union, Finset.mem_product,
      Finset.mem_inter, Finset.mem_compl, Finset.mem_union, mem_discordantSet]
    tauto
  have hd1 : Disjoint ((T ∩ L) ×ˢ ((T ∪ L)ᶜ)) (((T ∪ L)ᶜ) ×ˢ (T ∩ L)) := by
    rw [Finset.disjoint_left]
    rintro p hp hq
    rw [Finset.mem_product] at hp hq
    have h1 := Finset.mem_inter.mp hp.1
    have h2 := Finset.mem_compl.mp hq.1
    exact h2 (Finset.mem_union_left _ h1.1)
  have hd2 : Disjoint (((T ∩ L) ×ˢ ((T ∪ L)ᶜ)) ∪ (((T ∪ L)ᶜ) ×ˢ (T ∩ L))) (discordantSet T L) := by
    rw [Finset.disjoint_left]
    intro p hp hq
    rw [mem_discordantSet] at hq
    have h1 : p.1 ∈ T ∩ L ∨ p.1 ∈ (T ∪ L)ᶜ := by
      rcases Finset.mem_union.mp hp with h | h
      · exact Or.inl (Finset.mem_product.mp h).1
      · exact Or.inr (Finset.mem_product.mp h).1
    simp only [Finset.mem_inter, Finset.mem_compl, Finset.mem_union, not_or] at h1
    tauto
  have hdc : (discordantSet T L).card = 2 * (T \ L).card * (L \ T).card :=
    discordant_eq_flip_product T L
  rw [hsplit, Finset.card_union_of_disjoint hd2, Finset.card_union_of_disjoint hd1,
    Finset.card_product, Finset.card_product, hdc]
  ring

/-- **Balanced classes**: the annotation and the truth agree on as many disordered residues as
ordered ones. -/
def BalancedClasses (T L : Finset α) : Prop := (T ∩ L).card = ((T ∪ L)ᶜ).card

/-- The residues split into the agreement classes and the two flip classes. -/
lemma card_split (T L : Finset α) :
    Fintype.card α = (T ∩ L).card + ((T ∪ L)ᶜ).card + noise T L := by
  classical
  have h1 : (T ∪ L).card = (T ∩ L).card + (T \ L).card + (L \ T).card := by
    have : T ∪ L = ((T ∩ L) ∪ (T \ L)) ∪ (L \ T) := by
      ext x; simp only [Finset.mem_union, Finset.mem_inter, Finset.mem_sdiff]; tauto
    have hdisj1 : Disjoint (T ∩ L) (T \ L) := Finset.disjoint_left.mpr (by
      intro a ha hb
      exact (Finset.mem_sdiff.mp hb).2 (Finset.mem_inter.mp ha).2)
    have hdisj2 : Disjoint ((T ∩ L) ∪ (T \ L)) (L \ T) := Finset.disjoint_left.mpr (by
      intro a ha hb
      rcases Finset.mem_union.mp ha with h | h
      · exact (Finset.mem_sdiff.mp hb).2 (Finset.mem_inter.mp h).1
      · exact (Finset.mem_sdiff.mp h).2 (Finset.mem_sdiff.mp hb).1)
    rw [this, Finset.card_union_of_disjoint hdisj2, Finset.card_union_of_disjoint hdisj1]
  have h2 : ((T ∪ L)ᶜ).card = Fintype.card α - (T ∪ L).card := Finset.card_compl _
  have h3 : (T ∪ L).card ≤ Fintype.card α := Finset.card_le_univ _
  rw [h2, noise_eq_flip_add]
  omega

/-- The pair rate is small: with balanced classes and a noise rate below `1/4`, the fraction of
comparable pairs the annotation orders backwards is at most twice the square of the residue
noise rate. -/
theorem nuPair_le_two_eps_sq {T L : Finset α} (hbal : BalancedClasses T L)
    (h4 : 4 * noise T L ≤ Fintype.card α) (hpos : 0 < noise T L) :
    nuPair T L ≤ 2 * ((noise T L : ℝ) / Fintype.card α) ^ 2 := by
  classical
  set n : ℕ := Fintype.card α with hn
  set a : ℕ := (T ∩ L).card with ha
  set e : ℕ := ((T ∪ L)ᶜ).card with he
  set d : ℕ := (T \ L).card with hd
  set u : ℕ := (L \ T).card with hu
  set v : ℕ := noise T L with hv
  have hsplit : n = a + e + v := card_split T L
  have hvdu : v = d + u := noise_eq_flip_add T L
  have hae : a = e := hbal
  -- positivity
  have hnpos : 0 < n := by omega
  have haepos : 0 < a := by omega
  have hnR : (0:ℝ) < (n : ℝ) := by exact_mod_cast hnpos
  have haR : (0:ℝ) < (a : ℝ) := by exact_mod_cast haepos
  have hvR : (0:ℝ) < (v : ℝ) := by exact_mod_cast hpos
  have hduR : (0:ℝ) ≤ (d : ℝ) * u := by positivity
  -- the two ingredients
  have h4du : 4 * ((d : ℝ) * u) ≤ ((v : ℝ)) ^ 2 := by
    have : ((v : ℝ)) = (d : ℝ) + (u : ℝ) := by exact_mod_cast hvdu
    rw [this]
    nlinarith [sq_nonneg ((d : ℝ) - (u : ℝ))]
  have hnv : (n : ℝ) ^ 2 ≤ 8 * (a : ℝ) ^ 2 := by
    have h1 : (n : ℝ) = 2 * (a : ℝ) + (v : ℝ) := by
      have : n = 2 * a + v := by omega
      exact_mod_cast this
    have h2 : 4 * (v : ℝ) ≤ (n : ℝ) := by exact_mod_cast h4
    nlinarith
  -- the count
  have hcard : ((comparablePairs T L).card : ℝ) = 2 * ((a : ℝ) * e + (d : ℝ) * u) := by
    rw [card_comparablePairs]; push_cast; ring
  have hdisc : ((discordantPairs T L : ℕ) : ℝ) = 2 * ((d : ℝ) * u) := by
    rw [discordant_eq_flip_product]; push_cast; ring
  have haeR : (a : ℝ) = (e : ℝ) := by exact_mod_cast hae
  have key : ((d : ℝ) * u) * (n : ℝ) ^ 2 ≤ 2 * (v : ℝ) ^ 2 * (a : ℝ) ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left hnv hduR,
      mul_le_mul_of_nonneg_right h4du (sq_nonneg (a : ℝ))]
  have hR : 2 * (((v : ℝ) / n) ^ 2) = (2 * (v : ℝ) ^ 2) / (n : ℝ) ^ 2 := by
    field_simp
  rw [nuPair, hcard, hdisc, hR, ← haeR,
    div_le_div_iff₀ (by nlinarith) (by positivity)]
  nlinarith [mul_nonneg (sq_nonneg (v : ℝ)) hduR]

/-- **The capacity corollary.**  Pairwise scoring squares the noise: with balanced classes and a
residue noise rate `ε` below `1/4`, the pair rate is at most `2 ε²`, and the number of methods
the pairwise benchmark can order is at least `⌈1 / (4 ε²)⌉` -- quadratic in `1 / ε`, where the
residue protocol is linear. -/
theorem pairwise_capacity_quadratic {T L : Finset α} (hbal : BalancedClasses T L)
    (h4 : 4 * noise T L ≤ Fintype.card α) (hd : 0 < (T \ L).card) (hu : 0 < (L \ T).card) :
    nuPair T L ≤ 2 * ((noise T L : ℝ) / Fintype.card α) ^ 2 ∧
      ⌈1 / (4 * ((noise T L : ℝ) / Fintype.card α) ^ 2)⌉₊ ≤ pairwiseCapacity T L := by
  classical
  have hpos : 0 < noise T L := by rw [noise_eq_flip_add]; omega
  have hbound := nuPair_le_two_eps_sq hbal h4 hpos
  refine ⟨hbound, ?_⟩
  set eps : ℝ := (noise T L : ℝ) / Fintype.card α with heps
  have hnpos : 0 < Fintype.card α := by omega
  have hnR : (0:ℝ) < (Fintype.card α : ℝ) := by exact_mod_cast hnpos
  have hvR : (0:ℝ) < (noise T L : ℝ) := by exact_mod_cast hpos
  have hepspos : 0 < eps := by rw [heps]; positivity
  have h1 : 0 < (discordantSet T L).card := by
    have h := discordant_eq_flip_product T L
    unfold discordantPairs at h
    rw [h]
    exact Nat.mul_pos (Nat.mul_pos two_pos hd) hu
  obtain ⟨p, hp⟩ := Finset.card_pos.mp h1
  have h2 : 0 < (comparablePairs T L).card :=
    Finset.card_pos.mpr ⟨p, discordantSet_subset_comparablePairs T L hp⟩
  have hnu : 0 < nuPair T L := by
    have h1R : (0:ℝ) < (discordantPairs T L : ℝ) := by exact_mod_cast h1
    have h2R : (0:ℝ) < ((comparablePairs T L).card : ℝ) := by exact_mod_cast h2
    rw [nuPair]
    positivity
  have hle : 1 / (4 * eps ^ 2) ≤ 1 / (2 * nuPair T L) := by
    apply one_div_le_one_div_of_le (by positivity)
    linarith
  calc ⌈1 / (4 * eps ^ 2)⌉₊ ≤ ⌈1 / (2 * nuPair T L)⌉₊ := Nat.ceil_le_ceil hle
    _ ≤ pairwiseCapacity T L := le_max_right _ _

/-! ## 5.  The numbers -/

/-- At a residue noise rate of `ε = 0.0651` the residue protocol orders `8` methods and the
pairwise protocol `59`. -/
theorem capacity_at_eps_0651 :
    residueCapacity (651 / 10000) = 8 ∧ ⌈1 / (4 * (651 / 10000 : ℝ) ^ 2)⌉₊ = 59 := by
  constructor
  · rw [residueCapacity]
    have : ⌈1 / (2 * (651 / 10000 : ℝ))⌉₊ = 8 := by
      rw [Nat.ceil_eq_iff (by norm_num)]
      norm_num
    rw [this]
    norm_num
  · rw [Nat.ceil_eq_iff (by norm_num)]
    refine ⟨?_, ?_⟩
    · rw [lt_div_iff₀ (by norm_num)]; norm_num
    · rw [div_le_iff₀ (by norm_num)]; norm_num

/-! ## 6.  Sharpness -/

/-- **The identity and the bound are sharp.**  On four residues with `T = {0, 1}` and
`L = {2, 3}` the two flip classes balance (`d = u = 2`), every cross pair is realised, and
`discordantPairs = 8 = ν² / 2`. -/
theorem sharp_instance :
    discordantPairs ({0, 1} : Finset (Fin 4)) ({2, 3} : Finset (Fin 4)) = 8 ∧
      noise ({0, 1} : Finset (Fin 4)) ({2, 3} : Finset (Fin 4)) = 4 ∧
      discordantPairs ({0, 1} : Finset (Fin 4)) ({2, 3} : Finset (Fin 4))
        = (noise ({0, 1} : Finset (Fin 4)) ({2, 3} : Finset (Fin 4))) ^ 2 / 2 := by
  refine ⟨by decide, by decide, by decide⟩

/-! ## 7.  The hypotheses of the capacity corollary are satisfiable -/

/-- A truth on sixteen residues: one residue the annotation misses, seven both agree are
disordered. -/
def exampleTruth : Finset (Fin 16) := {0, 2, 3, 4, 5, 6, 7, 8}

/-- The matching annotation: it misses residue `0` and wrongly adds residue `1`, so the two
flip classes are `{0}` and `{1}`. -/
def exampleAnnot : Finset (Fin 16) := {1, 2, 3, 4, 5, 6, 7, 8}

/-- **The corollary is not vacuous.**  On sixteen residues, with the annotation missing one
disordered residue and adding one, all its hypotheses hold: the classes are balanced, the noise
rate is `2/16 = 1/8 ≤ 1/4`, and both flip classes are nonempty. -/
theorem exampleHypotheses :
    BalancedClasses exampleTruth exampleAnnot ∧
      4 * noise exampleTruth exampleAnnot ≤ Fintype.card (Fin 16) ∧
      0 < (exampleTruth \ exampleAnnot).card ∧ 0 < (exampleAnnot \ exampleTruth).card := by
  refine ⟨by unfold BalancedClasses; decide, by decide, by decide, by decide⟩

/-- On that instance the counts are `ν = 2` residues wrong, `2` discordant pairs out of `100`
comparable ones -- a pair rate of `1/50` against a residue rate of `1/8`. -/
theorem exampleCounts :
    noise exampleTruth exampleAnnot = 2 ∧
      discordantPairs exampleTruth exampleAnnot = 2 ∧
      (comparablePairs exampleTruth exampleAnnot).card = 100 := by
  refine ⟨by decide, by decide, by decide⟩

/-- The pair rate on the instance. -/
theorem nuPair_example : nuPair exampleTruth exampleAnnot = 1 / 50 := by
  rw [nuPair, show discordantPairs exampleTruth exampleAnnot = 2 from exampleCounts.2.1,
    show (comparablePairs exampleTruth exampleAnnot).card = 100 from exampleCounts.2.2]
  norm_num

/-- **The instance in numbers.**  The residue protocol at noise rate `1/8` orders `4` methods;
the pairwise protocol on the same annotation orders `25`. -/
theorem capacity_example :
    residueCapacity (1 / 8) = 4 ∧ pairwiseCapacity exampleTruth exampleAnnot = 25 := by
  constructor
  · rw [residueCapacity, show ((1:ℝ) / (2 * (1 / 8))) = 4 by norm_num]
    norm_num
  · rw [pairwiseCapacity, nuPair_example, show ((1:ℝ) / (2 * (1 / 50))) = 25 by norm_num]
    norm_num

end Discordant
end IDR
