/-
# How wrong, and how big?  A metric geometry of ensemble models

The earlier files answer *whether* a family of models can be right.  This file answers the
two quantitative questions that a designer actually faces:

* **How is error measured?**  On a finite conformation library, the uniform observational
  error `ApproxSame` used in `RequestProject.ModelNature` is *exactly* the `ℓ¹` (twice
  total-variation) distance between the predicted and the true population vectors
  (`approxSame_iff_ell1_le`, `ell1_eq_sSup_expect_diff`).  So there is no choice to make:
  the natural loss on a disorder model is the population-space `ℓ¹` distance, and it is a
  genuine metric which vanishes exactly on observationally equal ensembles
  (`ell1_eq_zero_iff`).
* **How big must the model be?**  Two matching bounds.  A model with at most `k`
  components misses at least `(m - k)·δ` of `ℓ¹` error on a target populating `m`
  conformations each with weight at least `δ` (`ell1_ge_of_card_le`); equivalently,
  reaching uniform accuracy `eps` *forces* `k ≥ m - eps/δ` (`capacity_lower_bound`) --
  capacity must grow **linearly** in the number of populated conformations, not merely be
  "large".  The bound is attained up to a factor two by truncating a uniform ensemble
  (`ell1_truncated_unif`), so the linear law is the truth, not an artefact.
* **The intrinsic scale is conformational entropy.**  Gibbs' inequality on an arbitrary
  finite support (`gibbs_finset`) gives `entropy_le_log_card`, and hence
  `card_ge_exp_entropy`: any model that reproduces a target ensemble must carry at least
  `exp (H)` components, where `H` is the conformational entropy of the target.  For a
  disordered region -- whose defining feature is a large `H` -- this is the sharpest
  architecture-free statement of the cost of correctness.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.ModelNature

namespace IDR

open Finset
open scoped Classical

variable {X : Type*}

namespace Ens

/-! ## Populations on a finite conformation space -/

section Fin

variable [Fintype X]

/-- On a finite conformation space an expectation is the population-weighted sum. -/
lemma expect_eq_sum_prob (E : Ens X) (f : X → ℝ) :
    E.expect f = ∑ x, E.prob x * f x := by
  classical
  simp only [prob, expect, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Finset.sum_eq_single (E.pt j)] <;> simp +contextual [eq_comm]

/-- The populations of a finite conformation space sum to one. -/
lemma sum_prob (E : Ens X) : ∑ x, E.prob x = 1 := by
  have := E.expect_eq_sum_prob (fun _ => (1 : ℝ))
  simpa using this.symm

lemma prob_le_one (E : Ens X) (x : X) : E.prob x ≤ 1 := by
  have h := E.sum_prob
  have : E.prob x ≤ ∑ y, E.prob y :=
    Finset.single_le_sum (fun y _ => E.prob_nonneg y) (Finset.mem_univ x)
  simpa [h] using this

/-- Observational equality is exactly equality of the population vectors. -/
lemma same_iff_prob_eq (E F : Ens X) : E.Same F ↔ ∀ x, E.prob x = F.prob x := by
  refine ⟨fun h x => h _, fun h f => ?_⟩
  rw [expect_eq_sum_prob, expect_eq_sum_prob]
  exact Finset.sum_congr rfl fun x _ => by rw [h x]

/-! ## The `ℓ¹` distance between ensembles -/

/-- The `ℓ¹` distance between the population vectors of two ensembles: twice the
total-variation distance. -/
noncomputable def ell1 (E F : Ens X) : ℝ := ∑ x, |E.prob x - F.prob x|

lemma ell1_nonneg (E F : Ens X) : 0 ≤ ell1 E F :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

lemma ell1_comm (E F : Ens X) : ell1 E F = ell1 F E :=
  Finset.sum_congr rfl fun _ _ => abs_sub_comm _ _

lemma ell1_triangle (E F G : Ens X) : ell1 E G ≤ ell1 E F + ell1 F G := by
  rw [ell1, ell1, ell1, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun x _ => ?_
  calc |E.prob x - G.prob x| = |(E.prob x - F.prob x) + (F.prob x - G.prob x)| := by ring_nf
    _ ≤ |E.prob x - F.prob x| + |F.prob x - G.prob x| := abs_add_le _ _

/-- The `ℓ¹` distance is a genuine metric on ensembles *modulo observational equality*: it
vanishes exactly when no experiment can tell the model from the target. -/
theorem ell1_eq_zero_iff (E F : Ens X) : ell1 E F = 0 ↔ E.Same F := by
  rw [same_iff_prob_eq]
  constructor
  · intro h x
    have := (Finset.sum_eq_zero_iff_of_nonneg (fun y _ => abs_nonneg (E.prob y - F.prob y))).1 h
      x (Finset.mem_univ x)
    have := abs_eq_zero.1 this
    linarith
  · intro h
    exact Finset.sum_eq_zero fun x _ => by rw [h x]; simp

/-- **Duality.**  Uniform agreement on all observables bounded by one *is* closeness in
`ℓ¹`.  The abstract notion of approximate correctness used throughout therefore has a
completely concrete meaning: it is the population-space `ℓ¹` error. -/
theorem approxSame_iff_ell1_le (eps : ℝ) (E F : Ens X) :
    ApproxSame eps E F ↔ ell1 E F ≤ eps := by
  constructor
  · intro h
    have hf : ∀ x : X, |(if 0 ≤ E.prob x - F.prob x then (1 : ℝ) else -1)| ≤ 1 := by
      intro x; by_cases hx : 0 ≤ E.prob x - F.prob x <;> simp [hx]
    have hexp := h _ hf
    have hval : E.expect (fun x => if 0 ≤ E.prob x - F.prob x then (1 : ℝ) else -1)
        - F.expect (fun x => if 0 ≤ E.prob x - F.prob x then (1 : ℝ) else -1) = ell1 E F := by
      rw [expect_eq_sum_prob, expect_eq_sum_prob, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun x _ => ?_
      by_cases hx : 0 ≤ E.prob x - F.prob x
      · rw [if_pos hx]
        rw [abs_of_nonneg hx]; ring
      · rw [if_neg hx]
        rw [abs_of_neg (by linarith [not_le.1 hx])]; ring
    rw [hval] at hexp
    exact le_trans (le_abs_self _) hexp
  · intro h f hf
    have : |E.expect f - F.expect f| ≤ ell1 E F := by
      rw [expect_eq_sum_prob, expect_eq_sum_prob, ← Finset.sum_sub_distrib]
      calc |∑ x, (E.prob x * f x - F.prob x * f x)|
          ≤ ∑ x, |E.prob x * f x - F.prob x * f x| := Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ x, |E.prob x - F.prob x| := by
            refine Finset.sum_le_sum fun x _ => ?_
            have : E.prob x * f x - F.prob x * f x = (E.prob x - F.prob x) * f x := by ring
            rw [this, abs_mul]
            exact mul_le_of_le_one_right (abs_nonneg _) (hf x)
    linarith

/-- The `ℓ¹` distance is the supremum of the observational discrepancies. -/
theorem ell1_eq_sSup_expect_diff (E F : Ens X) :
    IsLUB {r : ℝ | ∃ f : X → ℝ, (∀ x, |f x| ≤ 1) ∧ r = |E.expect f - F.expect f|}
      (ell1 E F) := by
  constructor
  · rintro r ⟨f, hf, rfl⟩
    exact ((approxSame_iff_ell1_le (ell1 E F) E F).2 le_rfl) f hf
  · intro b hb
    have := hb (a := |E.expect (fun x => if 0 ≤ E.prob x - F.prob x then (1 : ℝ) else -1)
      - F.expect (fun x => if 0 ≤ E.prob x - F.prob x then (1 : ℝ) else -1)|)
      ⟨_, by intro x; by_cases hx : 0 ≤ E.prob x - F.prob x <;> simp [hx], rfl⟩
    have hval : E.expect (fun x => if 0 ≤ E.prob x - F.prob x then (1 : ℝ) else -1)
        - F.expect (fun x => if 0 ≤ E.prob x - F.prob x then (1 : ℝ) else -1) = ell1 E F := by
      rw [expect_eq_sum_prob, expect_eq_sum_prob, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun x _ => ?_
      by_cases hx : 0 ≤ E.prob x - F.prob x
      · rw [if_pos hx, abs_of_nonneg hx]; ring
      · rw [if_neg hx, abs_of_neg (by linarith [not_le.1 hx])]; ring
    rw [hval, abs_of_nonneg (ell1_nonneg E F)] at this
    exact this

/-- Two probability vectors are never more than `2` apart. -/
lemma ell1_le_two (E F : Ens X) : ell1 E F ≤ 2 := by
  calc ell1 E F ≤ ∑ x, (E.prob x + F.prob x) := by
        refine Finset.sum_le_sum fun x _ => ?_
        rw [abs_sub_le_iff]
        constructor <;> linarith [E.prob_nonneg x, F.prob_nonneg x]
    _ = 2 := by rw [Finset.sum_add_distrib, E.sum_prob, F.sum_prob]; norm_num

end Fin

/-! ## Capacity: a linear lower bound -/

/-- Any set of conformations that a model populates injects into its list of components. -/
lemma card_support_le (M : Ens X) (s : Finset X) (hs : ∀ x ∈ s, 0 < M.prob x) :
    s.card ≤ M.card := by
  classical
  rcases s.eq_empty_or_nonempty with rfl | ⟨x₀, hx₀⟩
  · simp
  obtain ⟨j₀, -, -⟩ := (M.prob_pos_iff x₀).1 (hs x₀ hx₀)
  haveI : Nonempty (Fin M.card) := ⟨j₀⟩
  have hchoice : ∀ x ∈ s, ∃ j : Fin M.card, M.pt j = x := by
    intro x hx
    obtain ⟨j, -, hj⟩ := (M.prob_pos_iff x).1 (hs x hx)
    exact ⟨j, hj⟩
  choose! phi hphi using hchoice
  have : s.card ≤ (Finset.univ : Finset (Fin M.card)).card := by
    refine Finset.card_le_card_of_injOn phi (fun x _ => by simp) ?_
    intro a ha b hb hab
    rw [← hphi a ha, ← hphi b hb, hab]
  simpa using this

variable [Fintype X]

/-- **Quantitative capacity bound.**  If the target populates `m` distinct conformations,
each with weight at least `δ`, then a model built from at most `k` components is at `ℓ¹`
distance at least `(m - k)·δ` from it.  Error decreases only as capacity is spent, one
component per populated conformation. -/
theorem ell1_ge_of_card_le {k m : ℕ} {M E : Ens X} (hM : M.card ≤ k) {g : Fin m → X}
    (hg : Function.Injective g) {delta : ℝ} (hdelta : 0 ≤ delta)
    (hd : ∀ l, delta ≤ E.prob (g l)) :
    ((m : ℝ) - k) * delta ≤ ell1 M E := by
  classical
  set S : Finset X := Finset.image g Finset.univ with hS
  have hScard : S.card = m := by
    rw [hS, Finset.card_image_of_injective _ hg]; simp
  set A : Finset X := S.filter (fun x => M.prob x = 0) with hA
  set B : Finset X := S.filter (fun x => ¬ M.prob x = 0) with hB
  have hBcard : B.card ≤ k := by
    refine le_trans (card_support_le M B ?_) hM
    intro x hx
    have := (Finset.mem_filter.1 hx).2
    exact lt_of_le_of_ne (M.prob_nonneg x) (Ne.symm this)
  have hAB : A.card + B.card = m := by
    rw [hA, hB, Finset.card_filter_add_card_filter_not, hScard]
  have hAcard : (m : ℝ) - k ≤ (A.card : ℝ) := by
    have : (m : ℕ) ≤ A.card + k := by omega
    have := (Nat.cast_le (α := ℝ)).2 this
    push_cast at this
    linarith
  have hsum : ∑ x ∈ A, delta ≤ ∑ x ∈ A, |M.prob x - E.prob x| := by
    refine Finset.sum_le_sum fun x hx => ?_
    have hx0 : M.prob x = 0 := (Finset.mem_filter.1 hx).2
    obtain ⟨l, -, hl⟩ := Finset.mem_image.1 (Finset.mem_filter.1 hx).1
    have : delta ≤ E.prob x := by rw [← hl]; exact hd l
    rw [hx0, zero_sub, abs_neg, abs_of_nonneg (E.prob_nonneg x)]
    exact this
  have hle : ∑ x ∈ A, |M.prob x - E.prob x| ≤ ell1 M E :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ A)
      (fun x _ _ => abs_nonneg _)
  have hconst : ∑ _x ∈ A, delta = (A.card : ℝ) * delta := by
    simp [Finset.sum_const, nsmul_eq_mul]
  have h1 : ((m : ℝ) - k) * delta ≤ (A.card : ℝ) * delta :=
    mul_le_mul_of_nonneg_right hAcard hdelta
  rw [hconst] at hsum
  linarith

/-- **Capacity must grow linearly with the breadth of the ensemble.**  To be uniformly
accurate to `eps` on a target that populates `m` conformations with weight at least `δ`
each, a model needs at least `m - eps/δ` components.  No fixed architecture achieves a
fixed accuracy across increasingly disordered regions. -/
theorem capacity_lower_bound {k m : ℕ} {M E : Ens X} (hM : M.card ≤ k) {g : Fin m → X}
    (hg : Function.Injective g) {delta eps : ℝ} (hdpos : 0 < delta)
    (hd : ∀ l, delta ≤ E.prob (g l)) (h : ApproxSame eps M E) :
    (m : ℝ) - eps / delta ≤ k := by
  have h1 := ell1_ge_of_card_le hM hg (le_of_lt hdpos) hd
  have h2 := (approxSame_iff_ell1_le eps M E).1 h
  have h3 : ((m : ℝ) - k) * delta ≤ eps := le_trans h1 h2
  have h4 : (m : ℝ) - k ≤ eps / delta := (le_div_iff₀ hdpos).2 h3
  linarith

end Ens

/-! ## Sharpness: truncating a uniform ensemble -/

section Sharp

variable {X : Type*} [Fintype X] [DecidableEq X]

omit [Fintype X] [DecidableEq X] in
/-- Populations of a uniform ensemble outside its library vanish. -/
lemma unif_prob_not_mem {m : ℕ} (hm : 0 < m) {g : Fin m → X} {x : X}
    (hx : ∀ l, g l ≠ x) : (unif hm g).prob x = 0 := by
  classical
  simp only [Ens.prob, Ens.expect, unif]
  exact Finset.sum_eq_zero fun l _ => by simp [hx l]

omit [Fintype X] [DecidableEq X] in
/-- The population of a library conformation under the truncated uniform model. -/
lemma unif_trunc_prob {k m : ℕ} (hk : 0 < k) (hkm : k ≤ m) {g : Fin m → X}
    (hg : Function.Injective g) (l : Fin m) :
    (unif hk (fun j : Fin k => g (Fin.castLE hkm j))).prob (g l)
      = if (l : ℕ) < k then 1 / (k : ℝ) else 0 := by
  classical
  simp only [Ens.prob, Ens.expect, unif]
  by_cases hl : (l : ℕ) < k
  · rw [if_pos hl, Finset.sum_eq_single (⟨l, hl⟩ : Fin k)]
    · simp
    · intro j _ hj
      have hne : g (Fin.castLE hkm j) ≠ g l := by
        intro hcon
        have hv : (j : ℕ) = (l : ℕ) := by simpa using congrArg Fin.val (hg hcon)
        exact hj (Fin.val_injective (by simpa using hv))
      simp [hne]
    · intro hcon; exact absurd (Finset.mem_univ _) hcon
  · rw [if_neg hl]
    refine Finset.sum_eq_zero fun j _ => ?_
    have hne : g (Fin.castLE hkm j) ≠ g l := by
      intro hcon
      have hv : (j : ℕ) = (l : ℕ) := by simpa using congrArg Fin.val (hg hcon)
      exact hl (hv ▸ j.isLt)
    simp [hne]

/-- **The linear capacity law is sharp.**  Keeping `k` of `m` equally populated
conformations gives `ℓ¹` error exactly `2(m-k)/m`, matching the lower bound `(m-k)/m` up to
a factor two.  Capacity buys accuracy at exactly a linear rate: the model must grow with
the ensemble. -/
theorem ell1_truncated_unif {k m : ℕ} (hk : 0 < k) (hkm : k ≤ m) {g : Fin m → X}
    (hg : Function.Injective g) :
    Ens.ell1 (unif hk (fun l : Fin k => g (Fin.castLE hkm l))) (unif (lt_of_lt_of_le hk hkm) g)
      = 2 * ((m : ℝ) - k) / m := by
  classical
  have hm : 0 < m := lt_of_lt_of_le hk hkm
  have hk' : (0 : ℝ) < k := by exact_mod_cast hk
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hzero : ∀ x ∈ (Finset.univ : Finset X) \ Finset.image g Finset.univ,
      |(unif hk (fun l : Fin k => g (Fin.castLE hkm l))).prob x - (unif hm g).prob x| = 0 := by
    intro x hx
    have hnot : ∀ l : Fin m, g l ≠ x := by
      intro l hl
      exact (Finset.mem_sdiff.1 hx).2 (Finset.mem_image.2 ⟨l, Finset.mem_univ _, hl⟩)
    rw [unif_prob_not_mem hk (fun j => hnot (Fin.castLE hkm j)), unif_prob_not_mem hm hnot]
    simp
  rw [Ens.ell1, ← Finset.sum_subset (Finset.subset_univ (Finset.image g Finset.univ))
      (fun x hx hxS => hzero x (Finset.mem_sdiff.2 ⟨hx, hxS⟩)),
    Finset.sum_image (fun a _ b _ hab => hg hab)]
  have hterm : ∀ l : Fin m,
      |(unif hk (fun j : Fin k => g (Fin.castLE hkm j))).prob (g l) - (unif hm g).prob (g l)|
      = if (l : ℕ) < k then 1 / (k : ℝ) - 1 / (m : ℝ) else 1 / (m : ℝ) := by
    intro l
    rw [unif_trunc_prob hk hkm hg l, unif_prob_eq hm hg l]
    by_cases hl : (l : ℕ) < k
    · have hkm' : (1 : ℝ) / m ≤ 1 / k := by
        apply one_div_le_one_div_of_le hk'
        exact_mod_cast hkm
      rw [if_pos hl, if_pos hl, abs_of_nonneg (by linarith)]
    · rw [if_neg hl, if_neg hl, zero_sub, abs_neg, abs_of_nonneg (by positivity)]
  rw [Finset.sum_congr rfl (fun l _ => hterm l), Finset.sum_ite, Finset.sum_const,
    Finset.sum_const]
  have hfil : (Finset.univ.filter (fun l : Fin m => (l : ℕ) < k))
      = Finset.image (Fin.castLE hkm) Finset.univ := by
    ext l
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
    constructor
    · intro hl
      exact ⟨⟨l, hl⟩, Fin.val_injective (by simp)⟩
    · rintro ⟨j, rfl⟩; simp
  have hcard : (Finset.univ.filter (fun l : Fin m => (l : ℕ) < k)).card = k := by
    rw [hfil, Finset.card_image_of_injective _ (Fin.castLE_injective hkm)]
    simp
  have hcard2 : (Finset.univ.filter (fun l : Fin m => ¬ (l : ℕ) < k)).card = m - k := by
    have hsplit := Finset.card_filter_add_card_filter_not
      (s := (Finset.univ : Finset (Fin m))) (p := fun l : Fin m => (l : ℕ) < k)
    rw [hcard] at hsplit
    simp only [Finset.card_univ, Fintype.card_fin] at hsplit
    omega
  rw [hcard, hcard2, nsmul_eq_mul, nsmul_eq_mul, Nat.cast_sub hkm]
  field_simp
  ring

end Sharp

/-! ## Conformational entropy sets the capacity -/

section Entropy

variable {X : Type*} [Fintype X]

/-- Gibbs' inequality on an arbitrary finite index set. -/
lemma gibbs_finset {I : Type*} (s : Finset I) (p q : I → ℝ) (hp : ∀ i ∈ s, 0 ≤ p i)
    (hq : ∀ i ∈ s, 0 < q i) (hps : ∑ i ∈ s, p i = 1) (hqs : ∑ i ∈ s, q i = 1) :
    0 ≤ ∑ i ∈ s, p i * Real.log (p i / q i) := by
  have key : ∀ i ∈ s, p i - q i ≤ p i * Real.log (p i / q i) := by
    intro i hi
    rcases eq_or_lt_of_le (hp i hi) with h0 | hpos
    · have : p i = 0 := h0.symm
      simp [this]
      linarith [hq i hi]
    · have hq' := hq i hi
      have hlog : 1 - q i / p i ≤ Real.log (p i / q i) := by
        have := Real.add_one_le_exp (Real.log (q i / p i))
        rw [Real.exp_log (by positivity)] at this
        have hrw : Real.log (q i / p i) = - Real.log (p i / q i) := by
          rw [← Real.log_inv]
          congr 1
          field_simp
        rw [hrw] at this
        linarith
      have := mul_le_mul_of_nonneg_left hlog (le_of_lt hpos)
      calc p i - q i = p i * (1 - q i / p i) := by field_simp
        _ ≤ p i * Real.log (p i / q i) := this
  have hsum : ∑ i ∈ s, (p i - q i) ≤ ∑ i ∈ s, p i * Real.log (p i / q i) :=
    Finset.sum_le_sum key
  rw [Finset.sum_sub_distrib, hps, hqs] at hsum
  simpa using hsum

/-- The conformational (Gibbs--Shannon) entropy of an ensemble. -/
noncomputable def entropy (E : Ens X) : ℝ := ∑ x, -(E.prob x * Real.log (E.prob x))

/-- Observationally equal ensembles have equal entropy: entropy is a property of the
ensemble, not of the way a model lists its components. -/
lemma entropy_eq_of_same {E F : Ens X} (h : E.Same F) : entropy E = entropy F := by
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [(Ens.same_iff_prob_eq E F).1 h x]

/-- **Entropy bounds capacity.**  An ensemble's entropy never exceeds the logarithm of its
number of components. -/
theorem entropy_le_log_card (E : Ens X) :
    entropy E ≤ Real.log E.card := by
  classical
  set S : Finset X := Finset.univ.filter (fun x => E.prob x ≠ 0) with hS
  have hsupp : ∀ x ∈ (Finset.univ : Finset X) \ S, E.prob x = 0 := by
    intro x hx
    have := (Finset.mem_sdiff.1 hx).2
    simpa [hS] using this
  have hsum1 : ∑ x ∈ S, E.prob x = 1 := by
    have hss : ∑ x ∈ S, E.prob x = ∑ x : X, E.prob x :=
      Finset.sum_subset (Finset.subset_univ S)
        (fun x hx hxS => hsupp x (Finset.mem_sdiff.2 ⟨hx, hxS⟩))
    rw [hss, E.sum_prob]
  have hScard : 0 < S.card := by
    rcases Nat.eq_zero_or_pos S.card with h0 | hpos
    · rw [Finset.card_eq_zero] at h0
      rw [h0] at hsum1; simp at hsum1
    · exact hpos
  have hScard' : (0 : ℝ) < (S.card : ℝ) := by exact_mod_cast hScard
  have hqsum : ∑ _x ∈ S, (1 / (S.card : ℝ)) = 1 := by
    rw [Finset.sum_const, nsmul_eq_mul]
    field_simp
  have hgibbs := gibbs_finset S E.prob (fun _ => 1 / (S.card : ℝ))
    (fun x _ => E.prob_nonneg x) (fun _ _ => by positivity) hsum1 hqsum
  have hrw : ∀ x ∈ S, E.prob x * Real.log (E.prob x / (1 / (S.card : ℝ)))
      = E.prob x * Real.log (E.prob x) + E.prob x * Real.log (S.card : ℝ) := by
    intro x hx
    have hx0 : E.prob x ≠ 0 := by simpa [hS] using (Finset.mem_filter.1 hx).2
    have : E.prob x / (1 / (S.card : ℝ)) = E.prob x * (S.card : ℝ) := by field_simp
    rw [this, Real.log_mul hx0 (ne_of_gt hScard')]
    ring
  rw [Finset.sum_congr rfl hrw, Finset.sum_add_distrib, ← Finset.sum_mul, hsum1,
    one_mul] at hgibbs
  have hsplit : ∑ x, E.prob x * Real.log (E.prob x)
      = ∑ x ∈ S, E.prob x * Real.log (E.prob x) :=
    (Finset.sum_subset (Finset.subset_univ S) (fun x hx hxS => by
      rw [hsupp x (Finset.mem_sdiff.2 ⟨hx, hxS⟩)]; simp)).symm
  have hent : entropy E = -∑ x ∈ S, E.prob x * Real.log (E.prob x) := by
    rw [entropy, Finset.sum_neg_distrib, hsplit]
  have hcards : (S.card : ℝ) ≤ (E.card : ℝ) := by
    have := Ens.card_support_le E S (fun x hx => by
      have hx0 : E.prob x ≠ 0 := by simpa [hS] using (Finset.mem_filter.1 hx).2
      exact lt_of_le_of_ne (E.prob_nonneg x) (Ne.symm hx0))
    exact_mod_cast this
  have hlog : Real.log (S.card : ℝ) ≤ Real.log (E.card : ℝ) :=
    Real.log_le_log hScard' hcards
  rw [hent]
  linarith

/-- **The cost of correctness is the exponential of the conformational entropy.**  Any
model that reproduces a target ensemble -- on every observable -- must carry at least
`exp (H)` components, where `H` is the target's conformational entropy.  Disorder is
precisely large `H`, so the capacity requirement is exponential in the entropy of the
region being modelled. -/
theorem card_ge_exp_entropy {M E : Ens X} (h : M.Same E) (hM : 0 < M.card) :
    Real.exp (entropy E) ≤ M.card := by
  have h1 : entropy E = entropy M := (entropy_eq_of_same h).symm
  have h2 := entropy_le_log_card M
  rw [h1]
  calc Real.exp (entropy M) ≤ Real.exp (Real.log M.card) := Real.exp_le_exp.2 h2
    _ = M.card := Real.exp_log (by exact_mod_cast hM)

end Entropy

end IDR
