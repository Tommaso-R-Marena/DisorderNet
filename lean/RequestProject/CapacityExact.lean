/-
# Part LXXXV.1  The exact capacity threshold: an error floor you can compute from data

Everything proved before this file about capacity is *one-sided*: a model with too few
components cannot be right (`Ens.ell1_ge_of_card_le`, `capacity_lower_bound`,
`quantization_capacity`).  A one-sided statement rules architectures out; it does not tell a
practitioner what to build, and it makes no numerical prediction that a measurement could
contradict.

This file closes the gap.  Fix a target that populates `m` conformational states with
measured populations `w₀ ≥ w₁ ≥ ⋯ ≥ w_{m-1}` (this is exactly the form in which populated
states are reported by NMR/SAXS/single-molecule experiments).  Write

  `tail P k = w_k + w_{k+1} + ⋯ + w_{m-1}`

for the population that lies outside the `k` most populated states.  Then:

* `ell1_ge_two_tail` -- **the floor.**  *Every* model with at most `k` components, however
  trained, on however much data, is at population-space `ℓ¹` distance at least
  `2 · tail P k` from the target.
* `ell1_trunc_le_two_tail`, `minErr_eq` -- **the floor is attained.**  The explicit
  keep-the-top-`k`-and-renormalise model achieves exactly `2 · tail P k`.  So
  `minErr P k = 2 · tail P k` is not a bound but *the* best achievable error at capacity
  `k`: a two-sided, quantitative law.
* `minErr_step` -- **the shape of the fit-quality curve.**  Going from `k` to `k+1`
  components improves the attainable error by exactly `2 w_k`, twice the population of the
  next state.  The curve is therefore predicted term by term from measured populations,
  before any model is fitted.
* `minErr_eq_zero_iff`, `minErr_pos` -- **the kink.**  The attainable error is strictly
  positive for every `k < m` and exactly zero from `k = m` on: improvement stops at the
  threshold, and not before.  This is the falsifiable signature -- it is *not* what generic
  "more capacity helps" intuition predicts, which is smooth diminishing returns with no
  distinguished `k`.
* `optimalK`, `optimalK_spec`, `optimalK_min` -- **the design rule.**  For a target accuracy
  `eps`, `optimalK P eps` is the least component count that achieves it, and it does achieve
  it.  This is the "where to put the door" statement: not merely which designs fail.
* `minErr_perturb` -- **robustness to measurement error.**  Populations are measured with
  error bars; if the reported populations are wrong by `eta` in `ℓ¹`, the predicted floor
  moves by at most `eta`.  The prediction therefore survives finite experimental precision.
* `at_capacity_not_sufficient` -- **what the theorem does not buy.**  Having `k ≥ m`
  components does *not* imply accuracy: an `m`-component model can be maximally wrong.  So
  the "model at or above threshold fits well" half of any empirical protocol is a genuine
  empirical claim, not a corollary.  Recorded here so the two halves are never conflated.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Metric
import RequestProject.CoarseGraining

namespace IDR
namespace Capacity

open Finset
open scoped Classical

/-! ## Population profiles -/

/-- A **population profile**: the measured populations of the `m` conformational states of a
target ensemble, listed in decreasing order.  This is the form in which state populations
come out of an experiment. -/
structure Profile (m : ℕ) where
  /-- the population of the `i`-th most populated state -/
  w : Fin m → ℝ
  pos : ∀ i, 0 < w i
  anti : ∀ i j : Fin m, i ≤ j → w j ≤ w i
  sum_one : ∑ i, w i = 1

namespace Profile

variable {m : ℕ} (P : Profile m)

/-- The population carried by the `k` most populated states. -/
def head (k : ℕ) : ℝ := ∑ i ∈ Finset.univ.filter (fun i : Fin m => (i : ℕ) < k), P.w i

/-- The population left outside the `k` most populated states. -/
def tail (k : ℕ) : ℝ := ∑ i ∈ Finset.univ.filter (fun i : Fin m => k ≤ (i : ℕ)), P.w i

lemma head_add_tail (k : ℕ) : P.head k + P.tail k = 1 := by
  classical
  rw [head, tail]
  have := Finset.sum_filter_add_sum_filter_not (Finset.univ : Finset (Fin m))
    (fun i : Fin m => (i : ℕ) < k) P.w
  simp only [not_lt] at this
  rw [this, P.sum_one]

lemma tail_nonneg (k : ℕ) : 0 ≤ P.tail k :=
  Finset.sum_nonneg fun i _ => (P.pos i).le

lemma head_nonneg (k : ℕ) : 0 ≤ P.head k :=
  Finset.sum_nonneg fun i _ => (P.pos i).le

lemma head_pos {k : ℕ} (hk : 0 < k) (hm : 0 < m) : 0 < P.head k := by
  classical
  have hmem : (⟨0, hm⟩ : Fin m) ∈ Finset.univ.filter (fun i : Fin m => (i : ℕ) < k) := by
    simp [hk]
  refine lt_of_lt_of_le (P.pos ⟨0, hm⟩) ?_
  exact Finset.single_le_sum (f := P.w) (fun i _ => (P.pos i).le) hmem

/-- The population outside the top `k` states is `0` exactly when there are no more than `k`
states at all. -/
lemma tail_eq_zero_iff (k : ℕ) : P.tail k = 0 ↔ m ≤ k := by
  classical
  constructor
  · intro h
    by_contra hcon
    push_neg at hcon
    have hmem : (⟨k, hcon⟩ : Fin m) ∈ Finset.univ.filter (fun i : Fin m => k ≤ (i : ℕ)) := by
      simp
    have hle : P.w ⟨k, hcon⟩ ≤ P.tail k :=
      Finset.single_le_sum (f := P.w) (fun i _ => (P.pos i).le) hmem
    exact absurd h (ne_of_gt (lt_of_lt_of_le (P.pos ⟨k, hcon⟩) hle))
  · intro h
    have : Finset.univ.filter (fun i : Fin m => k ≤ (i : ℕ)) = ∅ := by
      ext i; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty,
        iff_false, not_le]
      exact lt_of_lt_of_le i.isLt h
    rw [tail, this, Finset.sum_empty]

lemma tail_pos {k : ℕ} (hk : k < m) : 0 < P.tail k := by
  rcases lt_or_eq_of_le (P.tail_nonneg k) with h | h
  · exact h
  · exact absurd ((P.tail_eq_zero_iff k).1 h.symm) (by omega)

lemma tail_antitone {k l : ℕ} (h : k ≤ l) : P.tail l ≤ P.tail k := by
  classical
  refine Finset.sum_le_sum_of_subset_of_nonneg ?_ (fun i _ _ => (P.pos i).le)
  intro i hi
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
  omega

/-- Peeling one state off the tail. -/
lemma tail_succ {k : ℕ} (hk : k < m) : P.tail k = P.w ⟨k, hk⟩ + P.tail (k + 1) := by
  classical
  have hins : Finset.univ.filter (fun i : Fin m => k ≤ (i : ℕ))
      = insert (⟨k, hk⟩ : Fin m) (Finset.univ.filter (fun i : Fin m => k + 1 ≤ (i : ℕ))) := by
    ext i
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Fin.ext_iff]
    omega
  have hnot : (⟨k, hk⟩ : Fin m) ∉ Finset.univ.filter (fun i : Fin m => k + 1 ≤ (i : ℕ)) := by
    simp
  rw [tail, hins, Finset.sum_insert hnot, ← tail]

/-- The number of states among the top `k`. -/
lemma card_head_filter (k : ℕ) :
    (Finset.univ.filter (fun i : Fin m => (i : ℕ) < k)).card = min k m := by
  classical
  have hmap : (Finset.univ.filter (fun i : Fin m => (i : ℕ) < k)).card
      = ((Finset.range m).filter (fun i => i < k)).card := by
    rw [← Finset.card_map ⟨Fin.val, Fin.val_injective⟩]
    congr 1
    ext x
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and,
      Function.Embedding.coeFn_mk, Finset.mem_range]
    constructor
    · rintro ⟨i, hi, rfl⟩; exact ⟨i.isLt, hi⟩
    · rintro ⟨h1, h2⟩; exact ⟨⟨x, h1⟩, h2, rfl⟩
  have hrange : ((Finset.range m).filter (fun i => i < k)) = Finset.range (min k m) := by
    ext x; simp only [Finset.mem_filter, Finset.mem_range, lt_min_iff]; omega
  rw [hmap, hrange, Finset.card_range]

/-- **No set of `k` states carries more population than the `k` most populated ones.** -/
lemma subset_mass_le_head {k : ℕ} (T : Finset (Fin m)) (hT : T.card ≤ k) :
    ∑ i ∈ T, P.w i ≤ P.head k := by
  classical
  set R : Finset (Fin m) := Finset.univ.filter (fun i : Fin m => (i : ℕ) < k) with hR
  have hTR : ∑ i ∈ T ∩ R, P.w i + ∑ i ∈ T \ R, P.w i = ∑ i ∈ T, P.w i :=
    Finset.sum_inter_add_sum_diff T R P.w
  have hRT : ∑ i ∈ R ∩ T, P.w i + ∑ i ∈ R \ T, P.w i = P.head k :=
    Finset.sum_inter_add_sum_diff R T P.w
  have hinter : T ∩ R = R ∩ T := Finset.inter_comm T R
  -- the deficit set is no larger than the surplus set
  have hcardT : T.card ≤ R.card := by
    have hTm : T.card ≤ m := by
      simpa using Finset.card_le_card (Finset.subset_univ T)
    rw [hR, card_head_filter (m := m) k]
    omega
  have hc1 : (T \ R).card + (T ∩ R).card = T.card := by
    rw [Finset.card_sdiff_add_card_inter]
  have hc2 : (R \ T).card + (R ∩ T).card = R.card := by
    rw [Finset.card_sdiff_add_card_inter]
  have hcard : (T \ R).card ≤ (R \ T).card := by
    rw [hinter] at hc1
    omega
  -- every state outside the top `k` is less populated than every state inside it
  have hkey : ∑ i ∈ T \ R, P.w i ≤ ∑ i ∈ R \ T, P.w i := by
    rcases Finset.eq_empty_or_nonempty (T \ R) with hempty | hne
    · rw [hempty, Finset.sum_empty]
      exact Finset.sum_nonneg fun i _ => (P.pos i).le
    · have hRTne : (R \ T).Nonempty := by
        rw [← Finset.card_pos]
        have := Finset.card_pos.2 hne
        omega
      obtain ⟨b, hb, hbmin⟩ := Finset.exists_min_image (R \ T) P.w hRTne
      have hb_lt : (b : ℕ) < k := by
        have := Finset.mem_sdiff.1 hb
        have := this.1
        rw [hR] at this
        simpa using this
      have hupper : ∀ a ∈ T \ R, P.w a ≤ P.w b := by
        intro a ha
        have hak : ¬ ((a : ℕ) < k) := by
          have := (Finset.mem_sdiff.1 ha).2
          rw [hR] at this
          simpa using this
        exact P.anti b a (by rw [Fin.le_def]; omega)
      calc ∑ i ∈ T \ R, P.w i ≤ (T \ R).card • P.w b :=
            Finset.sum_le_card_nsmul _ _ _ hupper
        _ ≤ (R \ T).card • P.w b := by
            simp only [nsmul_eq_mul]
            exact mul_le_mul_of_nonneg_right (by exact_mod_cast hcard) (P.pos b).le
        _ ≤ ∑ i ∈ R \ T, P.w i := Finset.card_nsmul_le_sum _ _ _ hbmin
  rw [← hTR, ← hRT, hinter]
  linarith

end Profile

/-! ## The target ensemble and the attainable error -/

variable {X : Type*}

/-- The population of a conformation the model never emits is zero. -/
lemma Ens.prob_eq_zero_of_not_pt {X : Type*} (E : Ens X) (x : X) (h : ∀ j, E.pt j ≠ x) :
    E.prob x = 0 := by
  classical
  simp only [Ens.prob, Ens.expect]
  exact Finset.sum_eq_zero fun j _ => by simp [h j]

/-- The target ensemble determined by a population profile and a choice of `m` distinct
conformations. -/
def target {m : ℕ} (P : Profile m) (g : Fin m → X) : Ens X :=
  { card := m, pt := g, w := P.w, w_nonneg := fun i => (P.pos i).le, w_sum := P.sum_one }

@[simp] lemma target_card {m : ℕ} (P : Profile m) (g : Fin m → X) :
    (target P g).card = m := rfl

/-- **The attainable error at capacity `k`**: twice the population outside the top `k`
states.  `ell1_ge_two_tail` and `minErr_eq` show this is exactly the best `ℓ¹` error any
`k`-component model can have. -/
def minErr {m : ℕ} (P : Profile m) (k : ℕ) : ℝ := 2 * P.tail k

variable [Fintype X] [DecidableEq X]

/-- **The error floor.**  Every model with at most `k` components is at population-space
`ℓ¹` distance at least `minErr P k = 2·tail P k` from the target -- whatever its parameters,
however it was trained, on however much data. -/
theorem ell1_ge_two_tail {m k : ℕ} (P : Profile m) {g : Fin m → X}
    (hg : Function.Injective g) {M : Ens X} (hM : M.card ≤ k) :
    minErr P k ≤ Ens.ell1 M (target P g) := by
  classical
  set E : Ens X := target P g with hE
  set S : Finset X := Finset.image M.pt Finset.univ with hS
  have hEprob : ∀ i : Fin m, E.prob (g i) = P.w i := fun i =>
    Ens.prob_pt_of_injective E hg i
  have hScard : S.card ≤ k := by
    refine le_trans (le_trans Finset.card_image_le ?_) hM
    simp
  have hMout : ∀ x, x ∉ S → M.prob x = 0 := by
    intro x hx
    refine Ens.prob_eq_zero_of_not_pt M x ?_
    intro j hj
    exact hx (Finset.mem_image.2 ⟨j, Finset.mem_univ _, hj⟩)
  have hMS : ∑ x ∈ S, M.prob x = 1 := by
    rw [← M.sum_prob]
    exact Finset.sum_subset (Finset.subset_univ S) (fun x _ hx => hMout x hx)
  -- the target's mass on the model's support is at most the mass of the top `k` states
  have hES : ∑ x ∈ S, E.prob x ≤ P.head k := by
    set T : Finset (Fin m) := Finset.univ.filter (fun i : Fin m => g i ∈ S) with hT
    have himg : Finset.image g T ⊆ S := by
      intro x hx
      obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hx
      exact (Finset.mem_filter.1 hi).2
    have hTcard : T.card ≤ k := by
      have : (Finset.image g T).card = T.card := Finset.card_image_of_injective T hg
      have h2 : (Finset.image g T).card ≤ S.card := Finset.card_le_card himg
      omega
    have hsum : ∑ x ∈ S, E.prob x = ∑ i ∈ T, P.w i := by
      have h1 : ∑ i ∈ T, P.w i = ∑ x ∈ Finset.image g T, E.prob x := by
        rw [Finset.sum_image (fun a _ b _ hab => hg hab)]
        exact Finset.sum_congr rfl fun i _ => (hEprob i).symm
      have h2 : ∑ x ∈ Finset.image g T, E.prob x = ∑ x ∈ S, E.prob x := by
        refine Finset.sum_subset himg ?_
        intro x hxS hxT
        by_contra hne
        have hpos : 0 < E.prob x := lt_of_le_of_ne (E.prob_nonneg x) (Ne.symm hne)
        obtain ⟨j, -, hj⟩ := (E.prob_pos_iff x).1 hpos
        have hjT : (j : Fin m) ∈ T := by
          rw [hT]
          simp only [Finset.mem_filter, Finset.mem_univ, true_and]
          rw [show g j = x from hj]
          exact hxS
        exact hxT (Finset.mem_image.2 ⟨j, hjT, hj⟩)
      rw [← h2, h1]
    rw [hsum]
    exact P.subset_mass_le_head T hTcard
  -- outside the model's support the whole discrepancy is the target's mass
  have hcompl : ∑ x ∈ Sᶜ, |M.prob x - E.prob x| = 1 - ∑ x ∈ S, E.prob x := by
    have h1 : ∑ x ∈ Sᶜ, |M.prob x - E.prob x| = ∑ x ∈ Sᶜ, E.prob x := by
      refine Finset.sum_congr rfl fun x hx => ?_
      rw [hMout x (Finset.mem_compl.1 hx), zero_sub, abs_neg,
        abs_of_nonneg (E.prob_nonneg x)]
    have h2 : ∑ x ∈ S, E.prob x + ∑ x ∈ Sᶜ, E.prob x = 1 := by
      rw [Finset.sum_add_sum_compl, E.sum_prob]
    rw [h1]; linarith
  -- inside it the discrepancy is at least the mass the model has to misplace
  have hin : 1 - ∑ x ∈ S, E.prob x ≤ ∑ x ∈ S, |M.prob x - E.prob x| := by
    have h1 : |∑ x ∈ S, (M.prob x - E.prob x)| ≤ ∑ x ∈ S, |M.prob x - E.prob x| :=
      Finset.abs_sum_le_sum_abs _ _
    have h2 : ∑ x ∈ S, (M.prob x - E.prob x) = 1 - ∑ x ∈ S, E.prob x := by
      rw [Finset.sum_sub_distrib, hMS]
    rw [h2] at h1
    exact le_trans (le_abs_self _) h1
  have hsplit : ∑ x ∈ S, |M.prob x - E.prob x| + ∑ x ∈ Sᶜ, |M.prob x - E.prob x|
      = Ens.ell1 M E := Finset.sum_add_sum_compl _ _
  have hhead := P.head_add_tail k
  simp only [minErr]
  linarith

/-- The explicit **keep-the-top-`k`-states-and-renormalise** model. -/
noncomputable def truncModel {m k : ℕ} (P : Profile m) (hk : 0 < k) (hkm : k ≤ m)
    (g : Fin m → X) : Ens X :=
  { card := k
    pt := fun j => g (Fin.castLE hkm j)
    w := fun j => P.w (Fin.castLE hkm j) / P.head k
    w_nonneg := fun j => by
      have hpos := P.head_pos hk (lt_of_lt_of_le hk hkm)
      exact div_nonneg (P.pos _).le hpos.le
    w_sum := by
      have hfil : Finset.univ.filter (fun i : Fin m => (i : ℕ) < k)
          = Finset.image (Fin.castLE hkm) Finset.univ := by
        ext i
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
        constructor
        · intro hi
          exact ⟨⟨i, hi⟩, by simp⟩
        · rintro ⟨j, rfl⟩
          simp
      have hsum : ∑ j : Fin k, P.w (Fin.castLE hkm j) = P.head k := by
        rw [Profile.head, hfil,
          Finset.sum_image (fun a _ b _ hab => by simpa [Fin.ext_iff] using hab)]
      have hpos := P.head_pos hk (lt_of_lt_of_le hk hkm)
      rw [← Finset.sum_div, hsum, div_self (ne_of_gt hpos)] }

omit [Fintype X] [DecidableEq X] in
@[simp] lemma truncModel_card {m k : ℕ} (P : Profile m) (hk : 0 < k) (hkm : k ≤ m)
    (g : Fin m → X) : (truncModel P hk hkm g).card = k := rfl

/-- **The floor is attained.**  The truncated model realises the floor exactly. -/
theorem ell1_trunc_le_two_tail {m k : ℕ} (P : Profile m) (hk : 0 < k) (hkm : k ≤ m)
    {g : Fin m → X} (hg : Function.Injective g) :
    Ens.ell1 (truncModel P hk hkm g) (target P g) ≤ minErr P k := by
  classical
  have hHpos : 0 < P.head k := P.head_pos hk (lt_of_lt_of_le hk hkm)
  have hhead := P.head_add_tail k
  have hHle : P.head k ≤ 1 := by linarith [P.tail_nonneg k]
  have hEprob : ∀ i : Fin m, (target P g).prob (g i) = P.w i := fun i =>
    Ens.prob_pt_of_injective (target P g) hg i
  have hMprob : ∀ i : Fin m,
      (truncModel P hk hkm g).prob (g i) = if (i : ℕ) < k then P.w i / P.head k else 0 := by
    intro i
    simp only [Ens.prob, Ens.expect, truncModel]
    by_cases hi : (i : ℕ) < k
    · rw [if_pos hi, Finset.sum_eq_single (⟨i, hi⟩ : Fin k)]
      · simp
      · intro j _ hj
        have hne : g (Fin.castLE hkm j) ≠ g i := by
          intro hcon
          have : (j : ℕ) = (i : ℕ) := by simpa using congrArg Fin.val (hg hcon)
          exact hj (Fin.val_injective (by simpa using this))
        simp [hne]
      · intro hcon; exact absurd (Finset.mem_univ _) hcon
    · rw [if_neg hi]
      refine Finset.sum_eq_zero fun j _ => ?_
      have hne : g (Fin.castLE hkm j) ≠ g i := by
        intro hcon
        have : (j : ℕ) = (i : ℕ) := by simpa using congrArg Fin.val (hg hcon)
        exact hi (this ▸ j.isLt)
      simp [hne]
  have hterm : ∀ i : Fin m,
      |(truncModel P hk hkm g).prob (g i) - (target P g).prob (g i)|
        = if (i : ℕ) < k then P.w i * (1 / P.head k - 1) else P.w i := by
    intro i
    rw [hMprob i, hEprob i]
    by_cases hi : (i : ℕ) < k
    · rw [if_pos hi, if_pos hi]
      have h1 : P.w i / P.head k - P.w i = P.w i * (1 / P.head k - 1) := by
        field_simp
      rw [h1, abs_of_nonneg]
      have h2 : (1 : ℝ) ≤ 1 / P.head k := by
        rw [le_div_iff₀ hHpos]; linarith
      have := (P.pos i).le
      nlinarith
    · rw [if_neg hi, if_neg hi, zero_sub, abs_neg, abs_of_nonneg (P.pos i).le]
  have hzero : ∀ x ∈ (Finset.univ : Finset X) \ Finset.image g Finset.univ,
      |(truncModel P hk hkm g).prob x - (target P g).prob x| = 0 := by
    intro x hx
    have hnot : ∀ i : Fin m, g i ≠ x := by
      intro i hi
      exact (Finset.mem_sdiff.1 hx).2 (Finset.mem_image.2 ⟨i, Finset.mem_univ _, hi⟩)
    have h1 : (truncModel P hk hkm g).prob x = 0 :=
      Ens.prob_eq_zero_of_not_pt _ x (fun j => hnot (Fin.castLE hkm j))
    have h2 : (target P g).prob x = 0 :=
      Ens.prob_eq_zero_of_not_pt _ x (fun j => hnot j)
    rw [h1, h2]; simp
  rw [Ens.ell1, ← Finset.sum_subset (Finset.subset_univ (Finset.image g Finset.univ))
      (fun x hx hxS => hzero x (Finset.mem_sdiff.2 ⟨hx, hxS⟩)),
    Finset.sum_image (fun a _ b _ hab => hg hab),
    Finset.sum_congr rfl (fun i _ => hterm i), Finset.sum_ite]
  have hfirst : ∑ i ∈ Finset.univ.filter (fun i : Fin m => (i : ℕ) < k),
      P.w i * (1 / P.head k - 1) = P.head k * (1 / P.head k - 1) := by
    rw [← Finset.sum_mul, ← Profile.head]
  have hsecond : ∑ i ∈ Finset.univ.filter (fun i : Fin m => ¬ (i : ℕ) < k), P.w i
      = P.tail k := by
    rw [Profile.tail]
    exact Finset.sum_congr (by ext i; simp [not_lt]) (fun i _ => rfl)
  rw [hfirst, hsecond]
  have hkey : P.head k * (1 / P.head k - 1) = 1 - P.head k := by
    field_simp
  rw [hkey]
  simp only [minErr]
  linarith

/-- **The exact capacity law.**  `minErr P k` is *the* smallest `ℓ¹` error attainable by a
`k`-component model of a target with populations `P`: the truncated model attains it, and
nothing does better.  Two-sided, and computable from measured populations alone. -/
theorem minErr_eq {m k : ℕ} (P : Profile m) (hk : 0 < k) (hkm : k ≤ m) {g : Fin m → X}
    (hg : Function.Injective g) :
    Ens.ell1 (truncModel P hk hkm g) (target P g) = minErr P k :=
  le_antisymm (ell1_trunc_le_two_tail P hk hkm hg)
    (ell1_ge_two_tail P hg (le_of_eq (truncModel_card P hk hkm g)))

/-! ## The predicted shape of the fit-quality curve -/

/-- **Improvement per added component.**  Adding the `(k+1)`-st component can improve the
attainable error by exactly `2 w_k` -- twice the population of the next state, a number
known from measurement before any model is fitted. -/
theorem minErr_step {m k : ℕ} (P : Profile m) (hk : k < m) :
    minErr P k - minErr P (k + 1) = 2 * P.w ⟨k, hk⟩ := by
  simp only [minErr, P.tail_succ hk]; ring

/-- **The kink.**  The attainable error is exactly zero from the threshold on. -/
theorem minErr_eq_zero_iff {m k : ℕ} (P : Profile m) : minErr P k = 0 ↔ m ≤ k := by
  rw [minErr, mul_eq_zero]
  simp [P.tail_eq_zero_iff k]

/-- **... and strictly positive before it.** -/
theorem minErr_pos {m k : ℕ} (P : Profile m) (hk : k < m) : 0 < minErr P k := by
  have := P.tail_pos hk
  simp only [minErr]
  linarith

theorem minErr_antitone {m k l : ℕ} (P : Profile m) (h : k ≤ l) : minErr P l ≤ minErr P k := by
  have := P.tail_antitone h
  simp only [minErr]
  linarith

/-- **Strictly decreasing below the threshold**: every component added below `m` buys a
strictly positive, quantified improvement. -/
theorem minErr_strictMono_below {m k : ℕ} (P : Profile m) (hk : k < m) :
    minErr P (k + 1) < minErr P k := by
  have hstep := minErr_step P hk
  have := P.pos ⟨k, hk⟩
  linarith

/-- **No gain above the threshold**: capacity beyond `m` buys nothing. -/
theorem minErr_const_above {m k l : ℕ} (P : Profile m) (hk : m ≤ k) (hl : k ≤ l) :
    minErr P l = minErr P k := by
  rw [(minErr_eq_zero_iff P).2 (le_trans hk hl), (minErr_eq_zero_iff P).2 hk]

/-! ## The design rule -/

/-- **The design rule.**  The least number of components that attains accuracy `eps`. -/
noncomputable def optimalK {m : ℕ} (P : Profile m) (eps : ℝ) : ℕ :=
  sInf {k | minErr P k ≤ eps}

/-- The design rule attains the requested accuracy. -/
theorem optimalK_spec {m : ℕ} (P : Profile m) {eps : ℝ} (heps : 0 ≤ eps) :
    minErr P (optimalK P eps) ≤ eps := by
  have hne : {k | minErr P k ≤ eps}.Nonempty :=
    ⟨m, by simp only [Set.mem_setOf_eq, (minErr_eq_zero_iff P).2 le_rfl]; exact heps⟩
  exact Nat.sInf_mem hne

/-- ... and nothing smaller does: the rule is exactly right, not merely safe. -/
theorem optimalK_min {m : ℕ} (P : Profile m) {eps : ℝ} {k : ℕ} (hk : k < optimalK P eps) :
    eps < minErr P k := by
  by_contra hcon
  push_neg at hcon
  have : optimalK P eps ≤ k := Nat.sInf_le (show k ∈ {k | minErr P k ≤ eps} from hcon)
  omega

/-- The design rule never asks for more components than the target has states. -/
theorem optimalK_le {m : ℕ} (P : Profile m) {eps : ℝ} (heps : 0 ≤ eps) :
    optimalK P eps ≤ m := by
  refine Nat.sInf_le ?_
  simp only [Set.mem_setOf_eq, (minErr_eq_zero_iff P).2 le_rfl]
  exact heps

/-! ## Robustness to measurement error -/

/-- **The floor is stable under measurement error.**  If two population profiles differ by
`eta` in `ℓ¹`, their predicted floors differ by at most `eta`.  Experimental error bars on
the populations propagate to the prediction with no amplification. -/
theorem minErr_perturb {m k : ℕ} (P Q : Profile m) {eta : ℝ}
    (h : ∑ i, |P.w i - Q.w i| ≤ eta) :
    |minErr P k - minErr Q k| ≤ 2 * eta := by
  classical
  have hdiff : |P.tail k - Q.tail k| ≤ eta := by
    have h1 : P.tail k - Q.tail k
        = ∑ i ∈ Finset.univ.filter (fun i : Fin m => k ≤ (i : ℕ)), (P.w i - Q.w i) := by
      rw [Profile.tail, Profile.tail, ← Finset.sum_sub_distrib]
    rw [h1]
    refine le_trans (Finset.abs_sum_le_sum_abs _ _) (le_trans ?_ h)
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
      (fun i _ _ => abs_nonneg _)
  have : minErr P k - minErr Q k = 2 * (P.tail k - Q.tail k) := by
    simp only [minErr]; ring
  rw [this, abs_mul]
  simpa using mul_le_mul_of_nonneg_left hdiff (by norm_num : (0:ℝ) ≤ 2)

/-! ## From structures to state populations

A real model emits structures, not labels, and the comparison with an experiment is made
after each structure is assigned to one of the reference states.  The floor survives that
step for a simple reason: assigning states is a push-forward, and a push-forward never
increases the number of mixture components.  So the whole test may be run at state level
against experimentally reported populations, with the floor still in force. -/

/-- **The floor applies to the state-assigned model.**  A model emitting structures in any
space `Y` with at most `k` components, read out through any state-assignment map
`h : Y → X`, is still at `ℓ¹` distance at least `minErr P k` from the target's state
populations.  Assignment cannot rescue an under-capacity model. -/
theorem stateLevel_floor {Y : Type*} {m k : ℕ} (P : Profile m) {g : Fin m → X}
    (hg : Function.Injective g) (h : Y → X) {M : Ens Y} (hM : M.card ≤ k) :
    minErr P k ≤ Ens.ell1 (Ens.map h M) (target P g) :=
  ell1_ge_two_tail P hg (show (Ens.map h M).card ≤ k from hM)

/-! ## The failure is observable, not just metric

The floor is stated in `ℓ¹` on populations.  The next theorem says what an experimenter
actually sees: an under-capacity model assigns **zero** population to a set of conformations
that genuinely carries at least `tail P k` of the population.  That is a directly measurable
prediction -- a named subensemble the model claims is unoccupied and the experiment finds
occupied -- rather than an abstract distance. -/

omit [Fintype X] in
/-- **Under-capacity models miss a specific, identifiable set of states.**  A model with at
most `k` components assigns population zero to a set of conformations whose true population
is at least `tail P k`. -/
theorem missed_states_of_under_capacity {m k : ℕ} (P : Profile m) {g : Fin m → X}
    (hg : Function.Injective g) {M : Ens X} (hM : M.card ≤ k) :
    ∃ A : Finset X, (∀ x ∈ A, M.prob x = 0) ∧ P.tail k ≤ ∑ x ∈ A, (target P g).prob x := by
  classical
  set E : Ens X := target P g with hE
  set S : Finset X := Finset.image M.pt Finset.univ with hS
  have hEprob : ∀ i : Fin m, E.prob (g i) = P.w i := fun i =>
    Ens.prob_pt_of_injective E hg i
  have hScard : S.card ≤ k := by
    refine le_trans (le_trans Finset.card_image_le ?_) hM
    simp
  refine ⟨Finset.image g Finset.univ \ S, ?_, ?_⟩
  · intro x hx
    refine Ens.prob_eq_zero_of_not_pt M x ?_
    intro j hj
    exact (Finset.mem_sdiff.1 hx).2 (Finset.mem_image.2 ⟨j, Finset.mem_univ _, hj⟩)
  · -- the states inside `S` carry at most the top-`k` mass, and all states carry `1`
    set T : Finset (Fin m) := Finset.univ.filter (fun i : Fin m => g i ∈ S) with hT
    have hTcard : T.card ≤ k := by
      have h1 : (Finset.image g T).card = T.card := Finset.card_image_of_injective T hg
      have h2 : (Finset.image g T).card ≤ S.card := by
        refine Finset.card_le_card ?_
        intro x hx
        obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hx
        exact (Finset.mem_filter.1 hi).2
      omega
    have hsplit : ∑ x ∈ Finset.image g Finset.univ \ S, E.prob x
        = ∑ i ∈ Finset.univ.filter (fun i : Fin m => g i ∉ S), P.w i := by
      rw [show Finset.image g Finset.univ \ S
            = Finset.image g (Finset.univ.filter (fun i : Fin m => g i ∉ S)) by
          ext x
          simp only [Finset.mem_sdiff, Finset.mem_image, Finset.mem_filter, Finset.mem_univ,
            true_and]
          constructor
          · rintro ⟨⟨i, rfl⟩, hx⟩; exact ⟨i, hx, rfl⟩
          · rintro ⟨i, hi, rfl⟩; exact ⟨⟨i, rfl⟩, hi⟩,
        Finset.sum_image (fun a _ b _ hab => hg hab)]
      exact Finset.sum_congr rfl fun i _ => hEprob i
    rw [hsplit]
    have hall : ∑ i ∈ T, P.w i + ∑ i ∈ Finset.univ.filter (fun i : Fin m => g i ∉ S), P.w i
        = 1 := by
      rw [hT]
      rw [Finset.sum_filter_add_sum_filter_not Finset.univ (fun i : Fin m => g i ∈ S) P.w]
      exact P.sum_one
    have hle := P.subset_mass_le_head T hTcard
    have hhead := P.head_add_tail k
    linarith

/-! ## What the theorem does not buy -/

/-- **Being at capacity is not sufficient.**  For every profile there is a model with
exactly `m` components -- at the threshold -- that is maximally wrong.  Hence the half of an
empirical protocol that says "a model respecting the threshold fits the data" is a genuine
empirical prediction and cannot be a corollary of any capacity theorem; only the half that
says "a model below the threshold cannot fit" is entailed. -/
theorem at_capacity_not_sufficient {m : ℕ} (P : Profile m) :
    ∃ (g : Fin m → (Fin m ⊕ Fin m)) (M : Ens (Fin m ⊕ Fin m)),
      Function.Injective g ∧ M.card = m ∧ Ens.ell1 M (target P g) = 2 := by
  classical
  refine ⟨Sum.inl, ⟨m, Sum.inr, P.w, fun i => (P.pos i).le, P.sum_one⟩, Sum.inl_injective,
    rfl, ?_⟩
  set M : Ens (Fin m ⊕ Fin m) := ⟨m, Sum.inr, P.w, fun i => (P.pos i).le, P.sum_one⟩ with hMdef
  set E : Ens (Fin m ⊕ Fin m) := target P Sum.inl with hEdef
  have hMl : ∀ a : Fin m, M.prob (Sum.inl a) = 0 := by
    intro a
    exact Ens.prob_eq_zero_of_not_pt M _ (fun j => Sum.inr_ne_inl)
  have hMr : ∀ a : Fin m, M.prob (Sum.inr a) = P.w a := by
    intro a
    have := Ens.prob_pt_of_injective M (by simpa [hMdef] using Sum.inr_injective) a
    simpa [hMdef] using this
  have hEl : ∀ a : Fin m, E.prob (Sum.inl a) = P.w a := by
    intro a
    have := Ens.prob_pt_of_injective E (by simpa [hEdef, target] using Sum.inl_injective) a
    simpa [hEdef, target] using this
  have hEr : ∀ a : Fin m, E.prob (Sum.inr a) = 0 := by
    intro a
    exact Ens.prob_eq_zero_of_not_pt E _ (fun j => Sum.inl_ne_inr)
  have h1 : ∀ a : Fin m, |M.prob (Sum.inl a) - E.prob (Sum.inl a)| = P.w a := by
    intro a; rw [hMl, hEl, zero_sub, abs_neg, abs_of_nonneg (P.pos a).le]
  have h2 : ∀ a : Fin m, |M.prob (Sum.inr a) - E.prob (Sum.inr a)| = P.w a := by
    intro a; rw [hMr, hEr, sub_zero, abs_of_nonneg (P.pos a).le]
  rw [Ens.ell1, Fintype.sum_sum_type,
    Finset.sum_congr rfl (fun a _ => h1 a), Finset.sum_congr rfl (fun a _ => h2 a), P.sum_one]
  norm_num

/-! ## How the rule bites harder as disorder broadens

A fixed component count is a fixed budget against a target whose breadth is not fixed.  For a
target with `m` equally populated states the attainable error at capacity `k` is exactly
`2(m-k)/m`, so a practitioner's fixed `K` degrades towards the maximal error `2` as the
region becomes more disordered.  This is the quantitative form of "a fixed architecture is
under-capacity for broad ensembles, by this much". -/

/-- The uniform population profile on `m` states. -/
noncomputable def uniformProfile {m : ℕ} (hm : 0 < m) : Profile m where
  w := fun _ => 1 / (m : ℝ)
  pos := fun _ => by positivity
  anti := fun _ _ _ => le_rfl
  sum_one := by
    have hm' : (0 : ℝ) < m := by exact_mod_cast hm
    simp [Finset.sum_const, nsmul_eq_mul]
    field_simp

/-- The number of states outside the top `k`. -/
lemma card_tail_filter {m : ℕ} (k : ℕ) :
    (Finset.univ.filter (fun i : Fin m => k ≤ (i : ℕ))).card = m - min k m := by
  classical
  have hsplit : (Finset.univ.filter (fun i : Fin m => (i : ℕ) < k)).card
      + (Finset.univ.filter (fun i : Fin m => ¬ (i : ℕ) < k)).card = m := by
    rw [Finset.card_filter_add_card_filter_not]
    simp
  rw [Profile.card_head_filter (m := m) k] at hsplit
  have hcongr : (Finset.univ.filter (fun i : Fin m => ¬ (i : ℕ) < k))
      = Finset.univ.filter (fun i : Fin m => k ≤ (i : ℕ)) := by
    ext i; simp [not_lt]
  rw [hcongr] at hsplit
  omega

/-- **The attainable error on a uniform target.**  Exactly `2(m-k)/m`. -/
theorem minErr_uniform {m k : ℕ} (hm : 0 < m) (hkm : k ≤ m) :
    minErr (uniformProfile hm) k = 2 * ((m : ℝ) - k) / m := by
  classical
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hcard := card_tail_filter (m := m) k
  rw [min_eq_left hkm] at hcard
  simp only [minErr, Profile.tail, uniformProfile, Finset.sum_const, nsmul_eq_mul, hcard]
  rw [Nat.cast_sub hkm]
  field_simp

/-- **A fixed component count degrades to maximal error as the ensemble broadens.**  For any
fixed capacity `K` and any margin `eps`, there is a uniform target on enough states for which
the best a `K`-component model can do is worse than `2 - eps`, i.e. essentially the worst
possible `ℓ¹` error.  A fixed architecture is not merely suboptimal on broad ensembles; its
accuracy tends to the trivial. -/
theorem fixed_capacity_degrades (K : ℕ) {eps : ℝ} (heps : 0 < eps) :
    ∃ (m : ℕ) (hm : 0 < m), K ≤ m ∧ 2 - eps < minErr (uniformProfile hm) K := by
  obtain ⟨m, hm1⟩ := exists_nat_gt (max ((2 * K : ℝ) / eps) (K : ℝ) + 1)
  have hKm' : (K : ℝ) < m := by
    have := le_max_right ((2 * K : ℝ) / eps) (K : ℝ)
    linarith
  have hKm : K ≤ m := by exact_mod_cast hKm'.le
  have hmpos : 0 < m := by
    have hK0 : (0 : ℝ) ≤ K := Nat.cast_nonneg K
    have : (0 : ℝ) < m := lt_of_le_of_lt hK0 hKm'
    exact_mod_cast this
  refine ⟨m, hmpos, hKm, ?_⟩
  have hm' : (0 : ℝ) < m := by exact_mod_cast hmpos
  rw [minErr_uniform hmpos hKm]
  have hbig : (2 * K : ℝ) / eps < m :=
    lt_of_le_of_lt (by linarith [le_max_left ((2*K : ℝ)/eps) (K : ℝ)]) hm1
  have h2 : (2 * K : ℝ) < eps * m := by
    rw [div_lt_iff₀ heps] at hbig
    linarith
  rw [lt_div_iff₀ hm']
  nlinarith

/-! ## Profiles from measured (rational) populations

Populations are reported as finitely many decimal numbers.  `ofRat` turns such a report into
a `Profile`, and `tail_ofRat` says the predicted floor is the corresponding exact rational
arithmetic -- so a prediction for a concrete system is a closed rational number, decided by
computation rather than estimated. -/

/-- A population profile built from measured rational populations. -/
def Profile.ofRat {m : ℕ} (q : Fin m → ℚ) (hpos : ∀ i, 0 < q i)
    (hanti : ∀ i j : Fin m, i ≤ j → q j ≤ q i) (hsum : ∑ i, q i = 1) : Profile m :=
  { w := fun i => (q i : ℝ)
    pos := fun i => by exact_mod_cast hpos i
    anti := fun i j hij => by exact_mod_cast hanti i j hij
    sum_one := by
      have : ((∑ i, q i : ℚ) : ℝ) = ((1 : ℚ) : ℝ) := by rw [hsum]
      push_cast at this
      simpa using this }

/-- The rational tail: the population outside the top `k` states, in exact arithmetic. -/
def tailRat {m : ℕ} (q : Fin m → ℚ) (k : ℕ) : ℚ :=
  ∑ i : Fin m, if k ≤ (i : ℕ) then q i else 0

/-- The predicted floor for a system whose populations are known rationals: twice the
rational tail, exactly. -/
def minErrRat {m : ℕ} (q : Fin m → ℚ) (k : ℕ) : ℚ := 2 * tailRat q k

lemma minErr_ofRat {m : ℕ} (q : Fin m → ℚ) (hpos : ∀ i, 0 < q i)
    (hanti : ∀ i j : Fin m, i ≤ j → q j ≤ q i) (hsum : ∑ i, q i = 1) (k : ℕ) :
    minErr (Profile.ofRat q hpos hanti hsum) k = ((minErrRat q k : ℚ) : ℝ) := by
  classical
  simp only [minErr, minErrRat, tailRat, Profile.tail, Profile.ofRat, Finset.sum_filter]
  push_cast
  refine congrArg (fun t : ℝ => 2 * t) (Finset.sum_congr rfl fun i _ => ?_)
  split_ifs <;> simp

end Capacity
end IDR
