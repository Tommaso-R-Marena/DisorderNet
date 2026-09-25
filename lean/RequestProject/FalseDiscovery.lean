/-
# Part XCI.1  A discovery list, and what it is allowed to mean

A screen over thousands of candidate disordered regions produces a *list*, and the error rate of
a list is not the error rate of any of its entries.  The conservative answer — split the level by
the number of candidates, as in `RequestProject.PanelPower` — controls the probability of *any*
false entry, and at proteome scale it demands per-candidate levels no experiment can reach.  The
quantity a screen can afford to control is the false discovery rate: the expected fraction of the
reported list that is wrong.

This file develops that theory from scratch, in a finite product experiment, with no measure
theory and no asymptotics.

* `bhR`, `bhRej` — the Benjamini–Hochberg step-up index and the discovery list it defines,
  as deterministic functions of the observed p-values; `below_bhR` proves the list has exactly
  `bhR` entries, and `bh_dominates_bonferroni` that it always contains the Bonferroni list.
* `bhR_eq_iff_pZero` — **the leave-one-out lemma**, the heart of the argument: on the event that
  candidate `i`'s p-value lies below the `k`-th threshold, the BH index is unchanged by replacing
  that p-value by zero.  The replaced vector ignores coordinate `i` entirely, so `{BH stops at k}`
  becomes an event about the other candidates — which is what makes independence usable.
* `fdp`, `fdp_eq_sum` — the false discovery proportion, and its expansion as a double sum over
  candidates and list lengths, which linearises the ratio `V/R`.
* `EE`, `EE_factor` — the product experiment (one independent coordinate per candidate) and the
  factorisation of an expectation of (function of coordinate `i`) × (function of the rest).
* `bh_fdr_control`, `bh_fdr_le` — **the theorem**: with independent candidates and superuniform
  nulls, the expected false discovery proportion of the BH list at level `q` is at most
  `|H₀|·q/m`, hence at most `q`.  Nothing is assumed about the non-null candidates.

The p-values that feed this machinery are built, from the read-out law of Part XC and nothing
else, in `RequestProject.ScreenBudget` and `RequestProject.ChernoffScreen`.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset
open scoped Classical

namespace IDR
namespace FDR

/-! ## 1. The Benjamini–Hochberg procedure, as a deterministic function of the p-values -/

/-- The number of p-values at or below the `k`-th step-up threshold `k·q/m`. -/
noncomputable def below (m : ℕ) (q : ℝ) (p : Fin m → ℝ) (k : ℕ) : ℕ :=
  (Finset.univ.filter (fun i => p i ≤ k * q / m)).card

lemma below_le (m : ℕ) (q : ℝ) (p : Fin m → ℝ) (k : ℕ) : below m q p k ≤ m := by
  simpa [below] using (Finset.card_filter_le (Finset.univ : Finset (Fin m))
    (fun i => p i ≤ k * q / m))

lemma below_mono (m : ℕ) {q : ℝ} (hq : 0 ≤ q) (p : Fin m → ℝ) {k k' : ℕ} (h : k ≤ k') :
    below m q p k ≤ below m q p k' := by
  apply Finset.card_le_card
  intro i hi
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
  refine hi.trans ?_
  have hkk : (k : ℝ) ≤ (k' : ℝ) := by exact_mod_cast h
  have hm : (0:ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
  gcongr

/-- The BH step-up index: the largest `k ≤ m` with at least `k` p-values below `k·q/m`. -/
noncomputable def bhR (m : ℕ) (q : ℝ) (p : Fin m → ℝ) : ℕ :=
  ((Finset.range (m + 1)).filter (fun k => k ≤ below m q p k)).sup id

/-- The set of hypotheses BH rejects. -/
noncomputable def bhRej (m : ℕ) (q : ℝ) (p : Fin m → ℝ) : Finset (Fin m) :=
  Finset.univ.filter (fun i => p i ≤ bhR m q p * q / m)

lemma bhR_mem (m : ℕ) (q : ℝ) (p : Fin m → ℝ) :
    bhR m q p ∈ (Finset.range (m + 1)).filter (fun k => k ≤ below m q p k) := by
  have hne : ((Finset.range (m + 1)).filter (fun k => k ≤ below m q p k)).Nonempty := by
    refine ⟨0, ?_⟩
    simp
  obtain ⟨b, hb, hsup⟩ := Finset.exists_mem_eq_sup _ hne (id : ℕ → ℕ)
  simpa [bhR, hsup] using hb

lemma bhR_le (m : ℕ) (q : ℝ) (p : Fin m → ℝ) : bhR m q p ≤ m := by
  have := bhR_mem m q p
  simp only [Finset.mem_filter, Finset.mem_range] at this
  omega

lemma bhR_le_below (m : ℕ) (q : ℝ) (p : Fin m → ℝ) : bhR m q p ≤ below m q p (bhR m q p) := by
  have := bhR_mem m q p
  simp only [Finset.mem_filter, Finset.mem_range] at this
  exact this.2

/-- Maximality: any admissible step-up index is at most the BH index. -/
lemma le_bhR (m : ℕ) (q : ℝ) (p : Fin m → ℝ) {k : ℕ} (hk : k ≤ m) (h : k ≤ below m q p k) :
    k ≤ bhR m q p := by
  refine Finset.le_sup (f := (id : ℕ → ℕ)) ?_
  simp only [Finset.mem_filter, Finset.mem_range]
  exact ⟨by omega, h⟩

/-- **The BH index counts its own rejections.**  Exactly `bhR` p-values lie below the threshold
`bhR·q/m`, so the reported discovery list has size `bhR`. -/
lemma below_bhR (m : ℕ) {q : ℝ} (hq : 0 ≤ q) (p : Fin m → ℝ) :
    below m q p (bhR m q p) = bhR m q p := by
  refine le_antisymm ?_ (bhR_le_below m q p)
  set r := bhR m q p with hr
  set k := below m q p r with hk
  have hkm : k ≤ m := below_le m q p r
  have hmono : below m q p r ≤ below m q p k := below_mono m hq p (bhR_le_below m q p)
  exact le_bhR m q p hkm hmono

lemma card_bhRej (m : ℕ) {q : ℝ} (hq : 0 ≤ q) (p : Fin m → ℝ) :
    (bhRej m q p).card = bhR m q p := by
  simpa [bhRej, below] using below_bhR m hq p

/-- **BH is never more conservative than Bonferroni.**  Anything rejected at the Bonferroni
level `q/m` is on the BH discovery list. -/
theorem bh_dominates_bonferroni (m : ℕ) {q : ℝ} (hq : 0 ≤ q) (p : Fin m → ℝ) {i : Fin m}
    (hi : p i ≤ q / m) : i ∈ bhRej m q p := by
  have h1 : 1 ≤ below m q p 1 := by
    have : i ∈ Finset.univ.filter (fun j => p j ≤ (1 : ℕ) * q / m) := by
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      simpa using hi
    exact Finset.card_pos.mpr ⟨i, this⟩
  have hm : 1 ≤ m := by
    by_contra h
    have : m = 0 := by omega
    subst this
    exact absurd i.2 (by omega)
  have hR : 1 ≤ bhR m q p := le_bhR m q p hm h1
  simp only [bhRej, Finset.mem_filter, Finset.mem_univ, true_and]
  refine hi.trans ?_
  have hR' : (1 : ℝ) ≤ (bhR m q p : ℝ) := by exact_mod_cast hR
  have hmnn : (0:ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
  have : q / m = 1 * q / m := by ring
  rw [this]
  gcongr


/-! ## 2. The leave-one-out identity

The crux of the Benjamini–Hochberg analysis: on the event that `p i` is below the `k`-th
threshold, the BH index is unchanged by *replacing* `p i` with zero.  Since the replaced vector
does not depend on the `i`-th p-value at all, the event `{BH index = k}` becomes, on that event,
an event about the *other* hypotheses only — which is what makes independence usable.
-/

/-- The p-value vector with coordinate `i` forced to zero. -/
noncomputable def pZero (m : ℕ) (p : Fin m → ℝ) (i : Fin m) : Fin m → ℝ :=
  fun j => if j = i then 0 else p j

lemma pZero_le (m : ℕ) {p : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (i : Fin m) (j : Fin m) :
    pZero m p i j ≤ p j := by
  unfold pZero
  by_cases h : j = i
  · simp [h, hp]
  · simp [h]

lemma below_antitone_fun (m : ℕ) (q : ℝ) {p p' : Fin m → ℝ} (h : ∀ j, p' j ≤ p j) (k : ℕ) :
    below m q p k ≤ below m q p' k := by
  apply Finset.card_le_card
  intro i hi
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
  exact (h i).trans hi

lemma below_pZero_eq (m : ℕ) {q : ℝ} (p : Fin m → ℝ) (i : Fin m) {k : ℕ} {t : ℝ}
    (ht : (0:ℝ) ≤ t) (hi : p i ≤ t) (hk : (k : ℝ) * q / m = t) :
    below m q (pZero m p i) k = below m q p k := by
  unfold below
  congr 1
  apply Finset.filter_congr
  intro j _
  rw [hk]
  by_cases h : j = i
  · subst h; simp [pZero, ht, hi]
  · simp [pZero, h]

/-- **The leave-one-out lemma.**  If the `i`-th p-value is below the `k`-th threshold, then the
BH index equals `k` exactly when it equals `k` for the vector with `p i` replaced by zero. -/
theorem bhR_eq_iff_pZero (m : ℕ) {q : ℝ} (hq : 0 ≤ q) {p : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j)
    (i : Fin m) {k : ℕ} (hi : p i ≤ (k : ℝ) * q / m) :
    bhR m q p = k ↔ bhR m q (pZero m p i) = k := by
  have hmnn : (0:ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
  have hthr : ∀ k' : ℕ, (0:ℝ) ≤ (k' : ℝ) * q / m := by
    intro k'
    positivity
  have hagree : ∀ k' : ℕ, k ≤ k' → below m q (pZero m p i) k' = below m q p k' := by
    intro k' hk'
    refine below_pZero_eq m p i (hthr k') ?_ rfl
    refine hi.trans ?_
    have : (k : ℝ) ≤ (k' : ℝ) := by exact_mod_cast hk'
    gcongr
  have hmono : ∀ k' : ℕ, below m q p k' ≤ below m q (pZero m p i) k' := fun k' =>
    below_antitone_fun m q (pZero_le m hp i) k'
  constructor
  · intro h
    have hk : k ≤ below m q p k := h ▸ bhR_le_below m q p
    have hkm : k ≤ m := h ▸ bhR_le m q p
    have hge : k ≤ bhR m q (pZero m p i) := le_bhR m q _ hkm (hk.trans (hmono k))
    refine le_antisymm ?_ hge
    by_contra hcon
    push_neg at hcon
    set r := bhR m q (pZero m p i) with hr
    have hrm : r ≤ m := bhR_le m q _
    have hrb : r ≤ below m q (pZero m p i) r := bhR_le_below m q _
    rw [hagree r (by omega)] at hrb
    have : r ≤ bhR m q p := le_bhR m q p hrm hrb
    omega
  · intro h
    have hk : k ≤ below m q (pZero m p i) k := h ▸ bhR_le_below m q _
    have hkm : k ≤ m := h ▸ bhR_le m q _
    rw [hagree k le_rfl] at hk
    have hge : k ≤ bhR m q p := le_bhR m q p hkm hk
    refine le_antisymm ?_ hge
    set r := bhR m q p with hr
    have hrm : r ≤ m := bhR_le m q p
    have hrb : r ≤ below m q p r := bhR_le_below m q p
    have : r ≤ bhR m q (pZero m p i) := le_bhR m q _ hrm (hrb.trans (hmono r))
    omega

/-! ## 3. The false discovery proportion -/

/-- The false discoveries: null hypotheses on the BH list. -/
noncomputable def falseRej (m : ℕ) (q : ℝ) (p : Fin m → ℝ) (H₀ : Finset (Fin m)) :
    Finset (Fin m) :=
  H₀.filter (fun i => p i ≤ bhR m q p * q / m)

lemma falseRej_eq_inter (m : ℕ) (q : ℝ) (p : Fin m → ℝ) (H₀ : Finset (Fin m)) :
    falseRej m q p H₀ = H₀ ∩ bhRej m q p := by
  ext i
  simp [falseRej, bhRej, Finset.mem_inter]

/-- The false discovery proportion: the fraction of the discovery list that is null, and `0`
when the list is empty (`x / 0 = 0` in Lean, which is exactly the convention BH uses). -/
noncomputable def fdp (m : ℕ) (q : ℝ) (p : Fin m → ℝ) (H₀ : Finset (Fin m)) : ℝ :=
  ((falseRej m q p H₀).card : ℝ) / (bhR m q p : ℝ)

lemma fdp_nonneg (m : ℕ) (q : ℝ) (p : Fin m → ℝ) (H₀ : Finset (Fin m)) :
    0 ≤ fdp m q p H₀ :=
  div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)

lemma fdp_le_one (m : ℕ) {q : ℝ} (hq : 0 ≤ q) (p : Fin m → ℝ) (H₀ : Finset (Fin m)) :
    fdp m q p H₀ ≤ 1 := by
  rcases Nat.eq_zero_or_pos (bhR m q p) with h | h
  · simp [fdp, h]
  · rw [fdp, div_le_one (by exact_mod_cast h)]
    have : (falseRej m q p H₀).card ≤ (bhRej m q p).card := by
      rw [falseRej_eq_inter]
      exact Finset.card_le_card (Finset.inter_subset_right)
    rw [card_bhRej m hq p] at this
    exact_mod_cast this

/-- **The FDP as a sum over hypotheses and list lengths.**  The identity that turns the ratio
`V/R` into something linear enough to take expectations of. -/
lemma fdp_eq_sum (m : ℕ) {q : ℝ} (p : Fin m → ℝ) (H₀ : Finset (Fin m)) :
    fdp m q p H₀ =
      ∑ i ∈ H₀, ∑ k ∈ Finset.Icc 1 m,
        (if p i ≤ (k : ℝ) * q / m ∧ bhR m q p = k then (1 : ℝ) / k else 0) := by
  rcases Nat.eq_zero_or_pos (bhR m q p) with h | h
  · rw [fdp, h]
    simp only [Nat.cast_zero, div_zero]
    refine (Finset.sum_eq_zero ?_).symm
    intro i _
    refine Finset.sum_eq_zero ?_
    intro k hk
    simp only [Finset.mem_Icc] at hk
    have hne : (0:ℕ) ≠ k := by omega
    rw [if_neg (by rintro ⟨-, h2⟩; exact hne h2)]
  · have hmem : bhR m q p ∈ Finset.Icc 1 m := by
      simp only [Finset.mem_Icc]
      exact ⟨h, bhR_le m q p⟩
    have hinner : ∀ i : Fin m,
        (∑ k ∈ Finset.Icc 1 m,
          (if p i ≤ (k : ℝ) * q / m ∧ bhR m q p = k then (1 : ℝ) / k else 0))
        = (if p i ≤ (bhR m q p : ℝ) * q / m then (1 : ℝ) / (bhR m q p : ℝ) else 0) := by
      intro i
      rw [Finset.sum_eq_single (bhR m q p)]
      · by_cases hc : p i ≤ (bhR m q p : ℝ) * q / m <;> simp [hc]
      · intro k _ hk
        have : ¬ (bhR m q p = k) := fun hh => hk hh.symm
        simp [this]
      · intro hc; exact absurd hmem hc
    rw [Finset.sum_congr rfl (fun i _ => hinner i)]
    rw [fdp, falseRej, Finset.sum_ite, Finset.sum_const, Finset.sum_const]
    simp [div_eq_mul_inv]

/-! ## 4. Independent p-values: the product model -/

section Product

variable {m : ℕ} {V : Fin m → Type*} [∀ i, Fintype (V i)]

/-- The weight of a configuration under independent coordinates. -/
noncomputable def wt (nu : ∀ i, V i → ℝ) (ω : ∀ i, V i) : ℝ := ∏ i, nu i (ω i)

/-- Expectation under the product law. -/
noncomputable def EE (nu : ∀ i, V i → ℝ) (f : (∀ i, V i) → ℝ) : ℝ := ∑ ω, wt nu ω * f ω

/-- The weight of a configuration of all coordinates *other than* `i`. -/
noncomputable def wtOff (nu : ∀ i, V i → ℝ) (i : Fin m) (w : ∀ j : {j // j ≠ i}, V j) : ℝ :=
  ∏ j : {j // j ≠ i}, nu j.1 (w j)

/-- Expectation over all coordinates other than `i`. -/
noncomputable def EEOff (nu : ∀ i, V i → ℝ) (i : Fin m)
    (f : (∀ j : {j // j ≠ i}, V j) → ℝ) : ℝ := ∑ w, wtOff nu i w * f w

omit [∀ i, Fintype (V i)] in
lemma wtOff_nonneg {nu : ∀ i, V i → ℝ} (hnu : ∀ i v, 0 ≤ nu i v) (i : Fin m)
    (w : ∀ j : {j // j ≠ i}, V j) : 0 ≤ wtOff nu i w :=
  Finset.prod_nonneg (fun _ _ => hnu _ _)

lemma sum_wtOff {nu : ∀ i, V i → ℝ} (hnu1 : ∀ i, ∑ v, nu i v = 1) (i : Fin m) :
    ∑ w : (∀ j : {j // j ≠ i}, V j), wtOff nu i w = 1 := by
  have h := Finset.prod_univ_sum (fun j : {j // j ≠ i} => (Finset.univ : Finset (V j.1)))
      (fun (j : {j // j ≠ i}) (v : V j.1) => nu j.1 v)
  simp only [hnu1, Finset.prod_const_one] at h
  simp only [wtOff]
  rw [h, Fintype.piFinset_univ]

lemma EEOff_nonneg {nu : ∀ i, V i → ℝ} (hnu : ∀ i v, 0 ≤ nu i v) (i : Fin m)
    {f : (∀ j : {j // j ≠ i}, V j) → ℝ} (hf : ∀ w, 0 ≤ f w) : 0 ≤ EEOff nu i f :=
  Finset.sum_nonneg (fun w _ => mul_nonneg (wtOff_nonneg hnu i w) (hf w))

lemma EEOff_mono {nu : ∀ i, V i → ℝ} (hnu : ∀ i v, 0 ≤ nu i v) (i : Fin m)
    {f g : (∀ j : {j // j ≠ i}, V j) → ℝ} (h : ∀ w, f w ≤ g w) :
    EEOff nu i f ≤ EEOff nu i g :=
  Finset.sum_le_sum (fun w _ => by
    exact mul_le_mul_of_nonneg_left (h w) (wtOff_nonneg hnu i w))

lemma EEOff_smul (nu : ∀ i, V i → ℝ) (i : Fin m) (c : ℝ)
    (f : (∀ j : {j // j ≠ i}, V j) → ℝ) : EEOff nu i (fun w => c * f w) = c * EEOff nu i f := by
  rw [EEOff, EEOff, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun w _ => by ring)

lemma EEOff_sum (nu : ∀ i, V i → ℝ) (i : Fin m) {ι : Type*} (s : Finset ι)
    (F : ι → (∀ j : {j // j ≠ i}, V j) → ℝ) :
    EEOff nu i (fun w => ∑ a ∈ s, F a w) = ∑ a ∈ s, EEOff nu i (F a) := by
  simp only [EEOff, Finset.mul_sum]
  rw [Finset.sum_comm]

lemma EE_sum (nu : ∀ i, V i → ℝ) {ι : Type*} (s : Finset ι) (F : ι → (∀ i, V i) → ℝ) :
    EE nu (fun ω => ∑ a ∈ s, F a ω) = ∑ a ∈ s, EE nu (F a) := by
  simp only [EE, Finset.mul_sum]
  rw [Finset.sum_comm]

/-- **Independence, in the only form the argument needs.**  A function of the `i`-th coordinate
times a function of all the others factorises. -/
lemma EE_factor (nu : ∀ i, V i → ℝ) (i : Fin m) (g : V i → ℝ)
    (f : (∀ j : {j // j ≠ i}, V j) → ℝ) :
    EE nu (fun ω => g (ω i) * f (fun j => ω j.1))
      = (∑ v, nu i v * g v) * EEOff nu i f := by
  rw [EE]
  rw [← Equiv.sum_comp (Equiv.piSplitAt i V).symm (fun ω => wt nu ω * (g (ω i) * f (fun j => ω j.1))),
    Fintype.sum_prod_type]
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl (fun v _ => ?_)
  have hwt : ∀ w : (∀ j : {j // j ≠ i}, V j),
      wt nu ((Equiv.piSplitAt i V).symm (v, w)) = nu i v * wtOff nu i w := by
    intro w
    rw [wt, wtOff, Fintype.prod_eq_mul_prod_compl i]
    congr 1
    · simp [Equiv.piSplitAt]
    · rw [Finset.prod_subtype (s := ({i}ᶜ : Finset (Fin m))) (p := fun j => j ≠ i)
        (by intro x; simp)]
      refine Finset.prod_congr rfl (fun j _ => ?_)
      simp [Equiv.piSplitAt, j.2]
  have hcoord : ∀ w : (∀ j : {j // j ≠ i}, V j),
      ((Equiv.piSplitAt i V).symm (v, w)) i = v := by
    intro w; simp [Equiv.piSplitAt]
  have hrest : ∀ w : (∀ j : {j // j ≠ i}, V j),
      (fun j : {j // j ≠ i} => ((Equiv.piSplitAt i V).symm (v, w)) j.1) = w := by
    intro w
    funext j
    simp [Equiv.piSplitAt, j.2]
  rw [EEOff, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun w _ => ?_)
  rw [hwt w, hcoord w, hrest w]
  ring

end Product

/-! ## 5. Benjamini–Hochberg controls the false discovery rate -/

section Main

variable {m : ℕ} {V : Fin m → Type*} [∀ i, Fintype (V i)]

/-- The observed p-value vector. -/
noncomputable def pvec (pv : ∀ i, V i → ℝ) (ω : ∀ i, V i) : Fin m → ℝ := fun i => pv i (ω i)

/-- The p-value vector with coordinate `i` zeroed: a function of the *other* coordinates only. -/
noncomputable def pOff (pv : ∀ i, V i → ℝ) (i : Fin m) (w : ∀ j : {j // j ≠ i}, V j) :
    Fin m → ℝ := fun j => if h : j = i then 0 else pv j (w ⟨j, h⟩)

omit [∀ i, Fintype (V i)] in
lemma pOff_eq (pv : ∀ i, V i → ℝ) (i : Fin m) (ω : ∀ i, V i) :
    pOff pv i (fun j => ω j.1) = pZero m (pvec pv ω) i := by
  funext j
  unfold pOff pZero pvec
  by_cases h : j = i <;> simp [h]

omit [∀ i, Fintype (V i)] in
/-- The indicator of `{p i below the k-th threshold and BH stops at k}` factorises into a
function of the `i`-th p-value times a function of all the others. -/
lemma indicator_factor {q : ℝ} (hq : 0 ≤ q) {pv : ∀ i, V i → ℝ} (hpv : ∀ i v, 0 ≤ pv i v)
    (i : Fin m) (k : ℕ) (ω : ∀ i, V i) :
    (if pvec pv ω i ≤ (k : ℝ) * q / m ∧ bhR m q (pvec pv ω) = k then (1 : ℝ) / k else 0)
      = (if pv i (ω i) ≤ (k : ℝ) * q / m then (1 : ℝ) else 0) *
        (if bhR m q (pOff pv i (fun j => ω j.1)) = k then (1 : ℝ) / k else 0) := by
  by_cases hA : pv i (ω i) ≤ (k : ℝ) * q / m
  · have hA' : pvec pv ω i ≤ (k : ℝ) * q / m := hA
    have hiff := bhR_eq_iff_pZero m hq (p := pvec pv ω) (fun j => hpv j (ω j)) i hA'
    rw [pOff_eq]
    rw [if_pos hA, one_mul]
    by_cases hB : bhR m q (pvec pv ω) = k
    · rw [if_pos ⟨hA', hB⟩, if_pos (hiff.mp hB)]
    · rw [if_neg (fun h => hB h.2), if_neg (fun h => hB (hiff.mpr h))]
  · have hA' : ¬ (pvec pv ω i ≤ (k : ℝ) * q / m) := hA
    rw [if_neg (fun h => hA' h.1), if_neg hA, zero_mul]

/-- One term of the double sum, bounded by the null's superuniformity. -/
lemma term_bound {q : ℝ} (hq : 0 ≤ q) {nu : ∀ i, V i → ℝ} (hnu0 : ∀ i v, 0 ≤ nu i v)
    {pv : ∀ i, V i → ℝ} (hpv : ∀ i v, 0 ≤ pv i v) (i : Fin m) {k : ℕ} (hk : 1 ≤ k)
    (hsuper : ∀ t : ℝ, 0 ≤ t →
      ∑ v ∈ Finset.univ.filter (fun v => pv i v ≤ t), nu i v ≤ t) :
    EE nu (fun ω =>
        (if pvec pv ω i ≤ (k : ℝ) * q / m ∧ bhR m q (pvec pv ω) = k then (1 : ℝ) / k else 0))
      ≤ (q / m) * EEOff nu i (fun w => if bhR m q (pOff pv i w) = k then (1 : ℝ) else 0) := by
  have hmnn : (0:ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
  have ht : (0:ℝ) ≤ (k : ℝ) * q / m := by positivity
  have hkpos : (0:ℝ) < (k : ℝ) := by exact_mod_cast hk
  set A : ℝ := EEOff nu i (fun w => if bhR m q (pOff pv i w) = k then (1 : ℝ) else 0) with hA
  have hAnn : 0 ≤ A := EEOff_nonneg hnu0 i (fun w => by positivity)
  have hfun : (fun ω => (if pvec pv ω i ≤ (k : ℝ) * q / m ∧ bhR m q (pvec pv ω) = k
      then (1 : ℝ) / k else 0))
      = fun ω => (if pv i (ω i) ≤ (k : ℝ) * q / m then (1 : ℝ) else 0) *
        (if bhR m q (pOff pv i (fun j => ω j.1)) = k then (1 : ℝ) / k else 0) := by
    funext ω
    exact indicator_factor hq hpv i k ω
  rw [hfun, EE_factor nu i (fun v => if pv i v ≤ (k : ℝ) * q / m then (1 : ℝ) else 0)
    (fun w => if bhR m q (pOff pv i w) = k then (1 : ℝ) / k else 0)]
  have hsecond : EEOff nu i (fun w => if bhR m q (pOff pv i w) = k then (1 : ℝ) / k else 0)
      = (1 / k) * A := by
    rw [hA, ← EEOff_smul]
    congr 1
    funext w
    by_cases h : bhR m q (pOff pv i w) = k <;> simp [h]
  have hfirst : (∑ v, nu i v * (if pv i v ≤ (k : ℝ) * q / m then (1 : ℝ) else 0))
      ≤ (k : ℝ) * q / m := by
    have : (∑ v, nu i v * (if pv i v ≤ (k : ℝ) * q / m then (1 : ℝ) else 0))
        = ∑ v ∈ Finset.univ.filter (fun v => pv i v ≤ (k : ℝ) * q / m), nu i v := by
      rw [Finset.sum_filter]
      exact Finset.sum_congr rfl (fun v _ => by
        by_cases h : pv i v ≤ (k : ℝ) * q / m <;> simp [h])
    rw [this]
    exact hsuper _ ht
  rw [hsecond]
  have hstep : (∑ v, nu i v * (if pv i v ≤ (k : ℝ) * q / m then (1 : ℝ) else 0)) * ((1 / k) * A)
      ≤ ((k : ℝ) * q / m) * ((1 / k) * A) := by
    apply mul_le_mul_of_nonneg_right hfirst
    positivity
  refine hstep.trans (le_of_eq ?_)
  field_simp

/-- The BH stopping events are disjoint, so their probabilities sum to at most one. -/
lemma sum_stop_le_one {q : ℝ} {nu : ∀ i, V i → ℝ} (hnu0 : ∀ i v, 0 ≤ nu i v)
    (hnu1 : ∀ i, ∑ v, nu i v = 1) (pv : ∀ i, V i → ℝ) (i : Fin m) :
    ∑ k ∈ Finset.Icc 1 m,
        EEOff nu i (fun w => if bhR m q (pOff pv i w) = k then (1 : ℝ) else 0) ≤ 1 := by
  rw [← EEOff_sum]
  have hle : ∀ w : (∀ j : {j // j ≠ i}, V j),
      (∑ k ∈ Finset.Icc 1 m, if bhR m q (pOff pv i w) = k then (1 : ℝ) else 0) ≤ 1 := by
    intro w
    rw [Finset.sum_ite_eq (Finset.Icc 1 m) (bhR m q (pOff pv i w)) (fun _ => (1:ℝ))]
    by_cases h : bhR m q (pOff pv i w) ∈ Finset.Icc 1 m <;> simp [h]
  refine (EEOff_mono hnu0 i hle).trans ?_
  rw [EEOff]
  simp only [mul_one]
  exact le_of_eq (sum_wtOff hnu1 i)

/-- **Benjamini–Hochberg controls the false discovery rate.**

`m` hypotheses are tested; the p-values are independent across hypotheses (each is a function of
its own coordinate of a product experiment); every p-value in the null set `H₀` is *superuniform*,
`P(pᵢ ≤ t) ≤ t`.  Then the expected proportion of false discoveries in the BH discovery list at
level `q` is at most `|H₀|·q/m` — and hence at most `q`, whatever the truth is at the non-null
hypotheses and however many of them there are. -/
theorem bh_fdr_control {q : ℝ} (hq : 0 ≤ q) {nu : ∀ i, V i → ℝ} (hnu0 : ∀ i v, 0 ≤ nu i v)
    (hnu1 : ∀ i, ∑ v, nu i v = 1) {pv : ∀ i, V i → ℝ} (hpv : ∀ i v, 0 ≤ pv i v)
    (H₀ : Finset (Fin m))
    (hsuper : ∀ i ∈ H₀, ∀ t : ℝ, 0 ≤ t →
      ∑ v ∈ Finset.univ.filter (fun v => pv i v ≤ t), nu i v ≤ t) :
    EE nu (fun ω => fdp m q (pvec pv ω) H₀) ≤ (H₀.card : ℝ) * q / m := by
  have hmnn : (0:ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
  have hqm : 0 ≤ q / m := by positivity
  have hrw : EE nu (fun ω => fdp m q (pvec pv ω) H₀)
      = ∑ i ∈ H₀, ∑ k ∈ Finset.Icc 1 m,
          EE nu (fun ω => (if pvec pv ω i ≤ (k : ℝ) * q / m ∧ bhR m q (pvec pv ω) = k
            then (1 : ℝ) / k else 0)) := by
    have h1 : (fun ω => fdp m q (pvec pv ω) H₀)
        = fun ω => ∑ i ∈ H₀, ∑ k ∈ Finset.Icc 1 m,
            (if pvec pv ω i ≤ (k : ℝ) * q / m ∧ bhR m q (pvec pv ω) = k
              then (1 : ℝ) / k else 0) := by
      funext ω
      exact fdp_eq_sum m (pvec pv ω) H₀
    rw [h1, EE_sum]
    exact Finset.sum_congr rfl (fun i _ => EE_sum nu _ _)
  rw [hrw]
  have hbound : ∀ i ∈ H₀,
      (∑ k ∈ Finset.Icc 1 m,
        EE nu (fun ω => (if pvec pv ω i ≤ (k : ℝ) * q / m ∧ bhR m q (pvec pv ω) = k
          then (1 : ℝ) / k else 0))) ≤ q / m := by
    intro i hi
    have hterm : ∀ k ∈ Finset.Icc 1 m,
        EE nu (fun ω => (if pvec pv ω i ≤ (k : ℝ) * q / m ∧ bhR m q (pvec pv ω) = k
          then (1 : ℝ) / k else 0))
        ≤ (q / m) * EEOff nu i (fun w => if bhR m q (pOff pv i w) = k then (1 : ℝ) else 0) := by
      intro k hk
      simp only [Finset.mem_Icc] at hk
      exact term_bound hq hnu0 hpv i hk.1 (hsuper i hi)
    refine (Finset.sum_le_sum hterm).trans ?_
    rw [← Finset.mul_sum]
    calc (q / m) * ∑ k ∈ Finset.Icc 1 m,
          EEOff nu i (fun w => if bhR m q (pOff pv i w) = k then (1 : ℝ) else 0)
        ≤ (q / m) * 1 := by
          exact mul_le_mul_of_nonneg_left (sum_stop_le_one hnu0 hnu1 pv i) hqm
      _ = q / m := mul_one _
  refine (Finset.sum_le_sum hbound).trans ?_
  rw [Finset.sum_const, nsmul_eq_mul]
  exact le_of_eq (by ring)

/-- The headline form: at level `q`, the expected false discovery proportion is at most `q`. -/
theorem bh_fdr_le {q : ℝ} (hq : 0 ≤ q) {nu : ∀ i, V i → ℝ} (hnu0 : ∀ i v, 0 ≤ nu i v)
    (hnu1 : ∀ i, ∑ v, nu i v = 1) {pv : ∀ i, V i → ℝ} (hpv : ∀ i v, 0 ≤ pv i v)
    (H₀ : Finset (Fin m))
    (hsuper : ∀ i ∈ H₀, ∀ t : ℝ, 0 ≤ t →
      ∑ v ∈ Finset.univ.filter (fun v => pv i v ≤ t), nu i v ≤ t) :
    EE nu (fun ω => fdp m q (pvec pv ω) H₀) ≤ q := by
  refine (bh_fdr_control hq hnu0 hnu1 hpv H₀ hsuper).trans ?_
  rcases Nat.eq_zero_or_pos m with hm | hm
  · subst hm
    simpa using hq
  · rw [div_le_iff₀ (by exact_mod_cast hm)]
    have : (H₀.card : ℝ) ≤ (m : ℝ) := by
      exact_mod_cast (Finset.card_le_card (Finset.subset_univ H₀)).trans
        (le_of_eq (by simp))
    nlinarith

end Main

end FDR
end IDR
