/-
# Part XXII.2  Prediction sets: validity is free, informativeness is the disorder

A model of an intrinsically disordered region is usually asked, in practice, for something
weaker than a full ensemble: a *set* of conformations that is claimed to contain the truth
with high probability -- the top-`N` decoys, the states above a population cut, the
conformers deposited as "the ensemble".  This file asks what such a set can and cannot
promise.

Throughout, `X` is the finite conformation library, `p q : X → ℝ` are population vectors
(`Scoring.IsProbVec`), and `mass p S` is the population a prediction assigns to the set
`S`.  A set `Covers` the truth at level `alpha` when its true mass is at least
`1 - alpha`.

* `univ_covers`, `validity_is_free` -- the whole library covers at every level, so
  *validity by itself ranks no model*: the only content of a prediction set is its size.
* `mass_diff_le_half_ell1`, `covers_of_close` -- coverage transfers with accuracy: a model
  within `eps` of the truth in population-space `l^1` loses at most `eps/2` of coverage.
  This is the precise sense in which a good ensemble model yields honest prediction sets.
* `card_ge_of_covers` -- the size law.  Any set covering at level `alpha` has cardinality
  at least `(1 - alpha) / pmax`, where `pmax` is the largest population of the target.  The
  size of an honest answer is fixed by the *truth's* flatness, not by the model.
* `card_ge_of_covers_unif` -- for `m` equally populated conformations this forces at least
  `(1 - alpha) * m` of them, and `unif_covers_iff` shows the bound is attained: the law is
  sharp.
* `no_singleton_covers` -- a single-structure answer is not a valid prediction set for any
  target whose largest population is below `1 - alpha`: the Part I verdict, in the language
  of coverage.
* `marginal_not_conditional` -- and the coverage that a benchmark reports is a *marginal*:
  an explicit pair of contexts on which the reported coverage is 90% while the coverage in
  one of the two contexts is exactly zero.  Coverage must therefore be reported per
  context, which is the same context-conditionality the earlier parts force on the model.
-/
import Mathlib
import RequestProject.Scoring

namespace IDR

namespace PredSet

open Finset
open scoped BigOperators Classical

open Scoring

variable {X : Type*}

/-- The population a prediction assigns to a set of conformations. -/
noncomputable def mass (p : X → ℝ) (S : Finset X) : ℝ := ∑ x ∈ S, p x

/-- `S` covers the truth `q` at level `alpha`. -/
def Covers (q : X → ℝ) (S : Finset X) (alpha : ℝ) : Prop := 1 - alpha ≤ mass q S

lemma mass_univ [Fintype X] {p : X → ℝ} (hp : IsProbVec p) : mass p Finset.univ = 1 := hp.2

lemma mass_nonneg [Fintype X] {p : X → ℝ} (hp : IsProbVec p) (S : Finset X) : 0 ≤ mass p S :=
  Finset.sum_nonneg (fun x _ => hp.1 x)

lemma mass_le_one [Fintype X] {p : X → ℝ} (hp : IsProbVec p) (S : Finset X) : mass p S ≤ 1 := by
  rw [← mass_univ hp]
  exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ S)
    (fun x _ _ => hp.1 x)

/-! ## Validity alone is empty -/

/-- The whole conformation library covers at every level. -/
theorem univ_covers [Fintype X] {q : X → ℝ} (hq : IsProbVec q) {alpha : ℝ} (halpha : 0 ≤ alpha) :
    Covers q Finset.univ alpha := by
  unfold Covers
  rw [mass_univ hq]
  linarith

/-- **Validity is free.**  For every target there is a valid prediction set, so a
coverage-only evaluation cannot distinguish a model that knows the ensemble from one that
knows nothing.  What a prediction set says is said by its *size*. -/
theorem validity_is_free [Fintype X] {alpha : ℝ} (halpha : 0 ≤ alpha) :
    ∀ q : X → ℝ, IsProbVec q → ∃ S : Finset X, Covers q S alpha :=
  fun _ hq => ⟨Finset.univ, univ_covers hq halpha⟩

/-! ## Coverage transfers with accuracy -/

/-- The mass of a set changes by at most half the `l^1` population error. -/
theorem mass_diff_le_half_ell1 [Fintype X] [DecidableEq X] {p q : X → ℝ} (hp : IsProbVec p) (hq : IsProbVec q)
    (S : Finset X) :
    |mass p S - mass q S| ≤ (∑ x, |p x - q x|) / 2 := by
  set f : X → ℝ := fun x => p x - q x with hf
  have htot : ∑ x, f x = 0 := by
    simp only [hf, Finset.sum_sub_distrib, hp.2, hq.2, sub_self]
  have hsplitf : ∑ x ∈ S, f x + ∑ x ∈ Sᶜ, f x = ∑ x, f x :=
    Finset.sum_add_sum_compl S f
  have hsplita : ∑ x ∈ S, |f x| + ∑ x ∈ Sᶜ, |f x| = ∑ x, |f x| :=
    Finset.sum_add_sum_compl S (fun x => |f x|)
  have hS : |∑ x ∈ S, f x| ≤ ∑ x ∈ S, |f x| := Finset.abs_sum_le_sum_abs _ _
  have hSc : |∑ x ∈ Sᶜ, f x| ≤ ∑ x ∈ Sᶜ, |f x| := Finset.abs_sum_le_sum_abs _ _
  have hcomp : ∑ x ∈ Sᶜ, f x = -∑ x ∈ S, f x := by
    have := hsplitf
    rw [htot] at this
    linarith
  have hmass : mass p S - mass q S = ∑ x ∈ S, f x := by
    simp only [mass, hf, Finset.sum_sub_distrib]
  rw [hmass]
  have h1 : |∑ x ∈ S, f x| ≤ ∑ x ∈ S, |f x| := hS
  have h2 : |∑ x ∈ S, f x| ≤ ∑ x ∈ Sᶜ, |f x| := by
    have : |∑ x ∈ Sᶜ, f x| = |∑ x ∈ S, f x| := by rw [hcomp, abs_neg]
    linarith [this ▸ hSc]
  have : 2 * |∑ x ∈ S, f x| ≤ ∑ x, |f x| := by linarith [hsplita]
  linarith

/-- **Coverage transfers with accuracy.**  A set that the model believes covers at level
`alpha` really covers at level `alpha + eps/2` once the model's populations are within
`eps` of the truth in `l^1`. -/
theorem covers_of_close [Fintype X] [DecidableEq X] {p q : X → ℝ} (hp : IsProbVec p) (hq : IsProbVec q) {S : Finset X}
    {alpha eps : ℝ} (hcov : Covers p S alpha) (hclose : ∑ x, |p x - q x| ≤ eps) :
    Covers q S (alpha + eps / 2) := by
  have hdiff := mass_diff_le_half_ell1 hp hq S
  have h1 : mass p S - mass q S ≤ (∑ x, |p x - q x|) / 2 :=
    (le_abs_self _).trans hdiff
  unfold Covers at hcov ⊢
  linarith

/-! ## The size law -/

/-- **The size of an honest prediction set is set by the target.**  If no conformation
carries more population than `pmax`, a set covering at level `alpha` needs at least
`(1 - alpha)/pmax` conformations. -/
theorem card_ge_of_covers {q : X → ℝ} {S : Finset X} {alpha pmax : ℝ}
    (hcov : Covers q S alpha) (hmax : ∀ x, q x ≤ pmax) :
    1 - alpha ≤ (S.card : ℝ) * pmax := by
  have hle : mass q S ≤ (S.card : ℝ) * pmax := by
    unfold mass
    calc ∑ x ∈ S, q x ≤ ∑ _x ∈ S, pmax := Finset.sum_le_sum (fun x _ => hmax x)
      _ = (S.card : ℝ) * pmax := by rw [Finset.sum_const, nsmul_eq_mul]
  exact hcov.trans hle

/-- The uniform target: a set covers exactly when it is large enough, so the size law
below is sharp. -/
theorem unif_covers_iff {m : ℕ} (hm : 0 < m) (S : Finset (Fin m)) (alpha : ℝ) :
    Covers (fun _ : Fin m => (m : ℝ)⁻¹) S alpha ↔ (1 - alpha) * m ≤ (S.card : ℝ) := by
  have hmpos : (0 : ℝ) < m := by exact_mod_cast hm
  unfold Covers mass
  rw [Finset.sum_const, nsmul_eq_mul]
  constructor
  · intro h
    have h2 := mul_le_mul_of_nonneg_right h hmpos.le
    rwa [mul_assoc, inv_mul_cancel₀ (ne_of_gt hmpos), mul_one] at h2
  · intro h
    have h2 := mul_le_mul_of_nonneg_right h (by positivity : (0 : ℝ) ≤ (m : ℝ)⁻¹)
    rwa [mul_assoc, mul_inv_cancel₀ (ne_of_gt hmpos), mul_one] at h2

/-- For `m` equally populated conformations, coverage at level `alpha` costs at least
`(1 - alpha) * m` of them. -/
theorem card_ge_of_covers_unif {m : ℕ} (hm : 0 < m) {S : Finset (Fin m)} {alpha : ℝ}
    (hcov : Covers (fun _ : Fin m => (m : ℝ)⁻¹) S alpha) :
    (1 - alpha) * m ≤ (S.card : ℝ) := (unif_covers_iff hm S alpha).1 hcov

/-- **A single structure is not a valid answer.**  If the most populated conformation of
the target carries less than `1 - alpha`, no one-element prediction set covers. -/
theorem no_singleton_covers {q : X → ℝ} {alpha : ℝ} {pmax : ℝ}
    (hmax : ∀ x, q x ≤ pmax) (hlt : pmax < 1 - alpha) (x₀ : X) :
    ¬ Covers q {x₀} alpha := by
  intro hcov
  have h := card_ge_of_covers hcov hmax
  rw [Finset.card_singleton] at h
  simp only [Nat.cast_one, one_mul] at h
  linarith

/-! ## Marginal coverage is not conditional coverage -/

/-- **The reported coverage is a marginal.**  Two contexts, an ordered target in each, and
a prediction set that ignores the context: the benchmark reports 90% coverage at level
`alpha = 1/10`, and the coverage in the second context is exactly zero.  A coverage number
is a statement about the *benchmark's mixture of contexts*, not about the region in
front of you. -/
theorem marginal_not_conditional :
    ∃ (T : Fin 2 → Fin 2 → ℝ) (mu : Fin 2 → ℝ) (S : Fin 2 → Finset (Fin 2)),
      (∀ i, IsProbVec (T i)) ∧ (∀ i, 0 ≤ mu i) ∧ (∑ i, mu i = 1) ∧
      (9 / 10 : ℝ) ≤ ∑ i, mu i * mass (T i) (S i) ∧
      mass (T 1) (S 1) = 0 := by
  refine ⟨fun i => pointVec i, ![9 / 10, 1 / 10], fun _ => {0}, ?_, ?_, ?_, ?_, ?_⟩
  · intro i; exact isProbVec_pointVec i
  · intro i; fin_cases i <;> norm_num
  · simp [Fin.sum_univ_two]; norm_num
  · have h0 : mass (pointVec (0 : Fin 2)) {0} = 1 := by
      simp [mass, pointVec]
    have h1 : mass (pointVec (1 : Fin 2)) {0} = 0 := by
      simp [mass, pointVec]
    rw [Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [h0, h1]
    norm_num
  · simp [mass, pointVec]

end PredSet

end IDR
