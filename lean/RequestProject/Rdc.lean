/-
# Part LVIII.1  Orientational NMR: residual dipolar couplings and what they determine

The observables treated so far are scalar couplings (Part XXVIII), relaxation (Part XXXII),
transfer efficiencies (Parts XXXIII, XLVII, LIII) and scattering.  None of them is orientational.
The experiment that *is* -- and that is used, routinely, on disordered regions, precisely because
it survives conformational averaging -- is the residual dipolar coupling of a weakly aligned
sample.  This file brings it inside the development.

A bond direction `u` in a medium with alignment tensor `A` gives

  `rdc A u = (3·uᵀAu − tr A)/2`,

the standard form; `A` is symmetric and traceless, which is the whole content of `Alignment`.

**What one measurement cannot do.**

* `rdc_neg` -- the coupling is invariant under `u ↦ −u`: a bond vector and its reverse are not
  distinguishable, ever, by any alignment medium.
* `rdc_level_set_circle` -- for an explicit axially symmetric tensor the coupling is *constant*
  on a whole circle of directions.  One medium therefore never determines an orientation: the
  level set is a curve, not a point, and the ambiguity is continuous rather than discrete.
* `rdc_axes_mean_zero` -- averaged over the three coordinate axes the coupling vanishes for every
  alignment tensor.  A vanishing RDC is not evidence of a rigid orientation perpendicular to
  anything; it is what an orientationally symmetric ensemble gives.
* `order_population_degenerate` -- **population and order parameter enter only through their
  product.**  An explicit fully ordered ensemble at an intermediate angle and an explicit
  half-ordered ensemble at the pole give exactly the same coupling.

**What any number of measurements can do, and no more.**

* `meanRdc_eq_quad_secondMoment` -- the ensemble-averaged coupling is a linear functional of the
  second-moment matrix `⟨u_i u_j⟩` of the orientational distribution, for every `A`.
* `meanRdc_eq_of_secondMoment_eq` -- hence two ensembles with the same second moment are
  indistinguishable by RDCs *in every medium simultaneously*.
* `secondMoment_not_injective` -- and the second moment does not determine the ensemble: two
  explicit two-conformer ensembles with **no conformer in common** have the same second moment,
  hence identical residual dipolar couplings in every alignment medium.

**What is genuinely determined.**

* `alignment_decomposition` -- an alignment tensor is exactly five numbers: the explicit
  decomposition of a symmetric traceless `3×3` matrix on five basis tensors.
* `rdc_five_directions_determine` -- and five suitably chosen bond directions determine it.  The
  familiar "five independent alignment media" is a statement of linear algebra, and it is proved
  here rather than asserted.

The reading for a model of a disordered region: RDCs are a genuine orientational constraint, they
are not a distance, and the object they constrain is the second moment of the orientational
distribution of each bond -- one symmetric traceless tensor per bond, not a conformation.  A model
is compared with RDC data correctly only by predicting that tensor from the ensemble.
-/
import Mathlib

set_option autoImplicit false

namespace IDR
namespace Rdc

open Matrix Finset

/-- The quadratic form `uᵀAu`. -/
def quad (A : Matrix (Fin 3) (Fin 3) ℝ) (u : Fin 3 → ℝ) : ℝ := ∑ i, ∑ j, A i j * u i * u j

/-- The residual dipolar coupling of a bond direction `u` in a medium with alignment tensor `A`,
in units of the maximal dipolar coupling: `(3uᵀAu − tr A)/2`. -/
noncomputable def rdc (A : Matrix (Fin 3) (Fin 3) ℝ) (u : Fin 3 → ℝ) : ℝ :=
  (3 * quad A u - Matrix.trace A) / 2

/-- An alignment tensor: symmetric and traceless. -/
structure Alignment (A : Matrix (Fin 3) (Fin 3) ℝ) : Prop where
  symm : A.IsSymm
  traceless : Matrix.trace A = 0

lemma quad_neg (A : Matrix (Fin 3) (Fin 3) ℝ) (u : Fin 3 → ℝ) : quad A (-u) = quad A u := by
  unfold quad
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  simp only [Pi.neg_apply]
  ring

/-- **A bond vector and its reverse are indistinguishable.** -/
theorem rdc_neg (A : Matrix (Fin 3) (Fin 3) ℝ) (u : Fin 3 → ℝ) : rdc A (-u) = rdc A u := by
  unfold rdc
  rw [quad_neg]

lemma quad_diagonal (d : Fin 3 → ℝ) (u : Fin 3 → ℝ) :
    quad (Matrix.diagonal d) u = ∑ i, d i * (u i) ^ 2 := by
  unfold quad
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.sum_eq_single i]
  · simp [Matrix.diagonal_apply_eq]
    ring
  · intro j _ hj
    simp [Matrix.diagonal_apply_ne' _ hj]
  · intro hmem
    exact absurd (Finset.mem_univ i) hmem

/-- An explicit axially symmetric alignment tensor. -/
def axialTensor : Matrix (Fin 3) (Fin 3) ℝ := Matrix.diagonal ![1, 1, -2]

lemma axialTensor_alignment : Alignment axialTensor where
  symm := by
    unfold axialTensor
    exact Matrix.isSymm_diagonal _
  traceless := by
    unfold axialTensor
    simp [Matrix.trace_diagonal, Fin.sum_univ_three]
    norm_num

/-- **One medium never determines an orientation.**  For an axially symmetric alignment tensor
the coupling is constant on a whole circle of unit directions. -/
theorem rdc_level_set_circle (theta : ℝ) :
    rdc axialTensor ![Real.cos theta, Real.sin theta, 0] = 3 / 2 := by
  have htr : Matrix.trace axialTensor = 0 := axialTensor_alignment.traceless
  have hax : axialTensor = Matrix.diagonal ![1, 1, -2] := rfl
  unfold rdc
  rw [htr, hax, quad_diagonal]
  simp only [Fin.sum_univ_three, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
  nlinarith [Real.sin_sq_add_cos_sq theta]

/-- Averaged over the three coordinate axes, the coupling vanishes for every alignment tensor:
an orientationally symmetric ensemble gives zero. -/
theorem rdc_axes_mean_zero {A : Matrix (Fin 3) (Fin 3) ℝ} (hA : Alignment A) :
    (rdc A ![1, 0, 0] + rdc A ![0, 1, 0] + rdc A ![0, 0, 1]) / 3 = 0 := by
  have htr : Matrix.trace A = 0 := hA.traceless
  have htr3 : A 0 0 + A 1 1 + A 2 2 = 0 := by
    have := htr
    simpa [Matrix.trace, Matrix.diag, Fin.sum_univ_three] using this
  unfold rdc quad
  rw [htr]
  simp only [Fin.sum_univ_three, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
  ring_nf
  linarith

/-! ## Ensembles: what any number of media determine -/

/-- The mean residual dipolar coupling of a finite weighted ensemble of bond directions. -/
noncomputable def meanRdc {n : ℕ} (w : Fin n → ℝ) (u : Fin n → Fin 3 → ℝ)
    (A : Matrix (Fin 3) (Fin 3) ℝ) : ℝ := ∑ k, w k * rdc A (u k)

/-- The second moment `⟨u_i u_j⟩` of the orientational distribution. -/
def secondMoment {n : ℕ} (w : Fin n → ℝ) (u : Fin n → Fin 3 → ℝ) :
    Matrix (Fin 3) (Fin 3) ℝ := Matrix.of fun i j => ∑ k, w k * u k i * u k j

/-- **The ensemble average is a linear functional of the second moment.** -/
theorem meanRdc_eq_quad_secondMoment {n : ℕ} (w : Fin n → ℝ) (u : Fin n → Fin 3 → ℝ)
    (A : Matrix (Fin 3) (Fin 3) ℝ) :
    meanRdc w u A =
      (3 * (∑ i, ∑ j, A i j * secondMoment w u i j) - (∑ k, w k) * Matrix.trace A) / 2 := by
  have key : (∑ k, w k * quad A (u k)) = ∑ i, ∑ j, A i j * secondMoment w u i j := by
    unfold quad secondMoment
    simp only [Matrix.of_apply]
    calc (∑ k, w k * ∑ i, ∑ j, A i j * u k i * u k j)
        = ∑ k, ∑ i, ∑ j, w k * (A i j * u k i * u k j) := by
          refine Finset.sum_congr rfl fun k _ => ?_
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun i _ => Finset.mul_sum _ _ _
      _ = ∑ i, ∑ k, ∑ j, w k * (A i j * u k i * u k j) := Finset.sum_comm
      _ = ∑ i, ∑ j, ∑ k, w k * (A i j * u k i * u k j) := by
          exact Finset.sum_congr rfl fun i _ => Finset.sum_comm
      _ = ∑ i, ∑ j, A i j * ∑ k, w k * u k i * u k j := by
          refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun k _ => by ring
  rw [← key]
  unfold meanRdc rdc
  rw [Finset.mul_sum, Finset.sum_mul, ← Finset.sum_sub_distrib, Finset.sum_div]
  exact Finset.sum_congr rfl fun k _ => by ring

/-- **RDCs in every medium determine at most the second moment.**  Two ensembles with the same
total weight and the same second-moment matrix give identical couplings for every alignment
tensor. -/
theorem meanRdc_eq_of_secondMoment_eq {n m : ℕ} (w : Fin n → ℝ) (u : Fin n → Fin 3 → ℝ)
    (w' : Fin m → ℝ) (v : Fin m → Fin 3 → ℝ)
    (hwt : (∑ k, w k) = ∑ k, w' k) (hS : secondMoment w u = secondMoment w' v)
    (A : Matrix (Fin 3) (Fin 3) ℝ) : meanRdc w u A = meanRdc w' v A := by
  rw [meanRdc_eq_quad_secondMoment, meanRdc_eq_quad_secondMoment, hS, hwt]

/-- The first witness ensemble: the two coordinate directions `e₁`, `e₂` with equal weight. -/
def ensA : Fin 2 → Fin 3 → ℝ := ![![1, 0, 0], ![0, 1, 0]]

/-- The second witness ensemble: the two diagonal directions `(e₁±e₂)/√2` with equal weight. -/
noncomputable def ensB : Fin 2 → Fin 3 → ℝ :=
  ![![Real.sqrt 2 / 2, Real.sqrt 2 / 2, 0], ![Real.sqrt 2 / 2, -(Real.sqrt 2 / 2), 0]]

lemma sqrt_two_half_sq : (Real.sqrt 2 / 2) ^ 2 = 1 / 2 := by
  rw [div_pow, Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2)]
  norm_num

/-- **The second moment does not determine the ensemble.**  Two equally weighted two-conformer
ensembles with no conformer in common have the same second moment, hence identical residual
dipolar couplings in *every* alignment medium. -/
theorem secondMoment_not_injective :
    (∀ k l : Fin 2, ensA k ≠ ensB l) ∧
    secondMoment ![1/2, 1/2] ensA = secondMoment ![1/2, 1/2] ensB ∧
    ∀ A : Matrix (Fin 3) (Fin 3) ℝ,
      meanRdc ![1/2, 1/2] ensA A = meanRdc ![1/2, 1/2] ensB A := by
  have hne : ∀ k l : Fin 2, ensA k ≠ ensB l := by
    intro k l hkl
    have h0 := congrFun hkl 0
    have h1 := congrFun hkl 1
    fin_cases k <;> fin_cases l <;>
      simp [ensA, ensB] at h0 h1 <;> linarith
  have hS : secondMoment ![1/2, 1/2] ensA = secondMoment ![1/2, 1/2] ensB := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [secondMoment, ensA, ensB, Fin.sum_univ_two] <;>
      nlinarith [sqrt_two_half_sq]
  refine ⟨hne, hS, fun A => ?_⟩
  exact meanRdc_eq_of_secondMoment_eq _ _ _ _ rfl hS A

/-- The fully ordered witness: a single direction at the angle where the axial coupling is
`-3/2`. -/
noncomputable def orderedDir : Fin 3 → ℝ := ![Real.sqrt (1/3), 0, Real.sqrt (2/3)]

/-- The "magic angle" witness: the direction at which the axial coupling vanishes. -/
noncomputable def nullDir : Fin 3 → ℝ := ![Real.sqrt (2/3), 0, Real.sqrt (1/3)]

/-- **Population and order enter only through their product.**  A fully ordered ensemble at an
intermediate angle and a half-ordered ensemble at the pole give exactly the same coupling. -/
theorem order_population_degenerate :
    meanRdc ![1/2, 1/2] ![orderedDir, orderedDir] axialTensor
      = meanRdc ![1/2, 1/2] ![![0, 0, 1], nullDir] axialTensor := by
  have h13 : Real.sqrt (1/3) ^ 2 = 1/3 := Real.sq_sqrt (by norm_num)
  have h23 : Real.sqrt (2/3) ^ 2 = 2/3 := Real.sq_sqrt (by norm_num)
  have htr : Matrix.trace axialTensor = 0 := axialTensor_alignment.traceless
  have hq : ∀ u : Fin 3 → ℝ, rdc axialTensor u
      = (3 * ((u 0) ^ 2 + (u 1) ^ 2 - 2 * (u 2) ^ 2)) / 2 := by
    intro u
    have hax : axialTensor = Matrix.diagonal ![1, 1, -2] := rfl
    unfold rdc
    rw [htr, hax, quad_diagonal]
    simp only [Fin.sum_univ_three, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
    ring
  unfold meanRdc
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val,
    hq, orderedDir, nullDir]
  rw [h13, h23]
  norm_num

/-! ## What is determined: the five parameters of an alignment tensor -/

/-- **An alignment tensor is exactly five numbers.**  Explicit decomposition of a symmetric
traceless `3×3` matrix on five basis tensors. -/
theorem alignment_decomposition {A : Matrix (Fin 3) (Fin 3) ℝ} (hA : Alignment A) :
    A = A 0 0 • Matrix.diagonal ![1, 0, -1] + A 1 1 • Matrix.diagonal ![0, 1, -1]
      + A 0 1 • (Matrix.of ![![0, 1, 0], ![1, 0, 0], ![0, 0, 0]])
      + A 0 2 • (Matrix.of ![![0, 0, 1], ![0, 0, 0], ![1, 0, 0]])
      + A 1 2 • (Matrix.of ![![0, 0, 0], ![0, 0, 1], ![0, 1, 0]]) := by
  have hsymm : ∀ i j, A i j = A j i := fun i j => by
    simpa [Matrix.transpose] using congrFun (congrFun hA.symm j) i
  have htr3 : A 0 0 + A 1 1 + A 2 2 = 0 := by
    simpa [Matrix.trace, Matrix.diag, Fin.sum_univ_three] using hA.traceless
  have h10 := hsymm 1 0
  have h20 := hsymm 2 0
  have h21 := hsymm 2 1
  ext i j
  fin_cases i <;> fin_cases j <;> simp <;> linarith

/-- The normalisation constant `1/√2` of a diagonal unit direction. -/
noncomputable def rt : ℝ := Real.sqrt 2 / 2

lemma rt_mul_rt : rt * rt = 1 / 2 := by
  rw [rt, ← sq]
  exact sqrt_two_half_sq

lemma quad_expand (M : Matrix (Fin 3) (Fin 3) ℝ) (u : Fin 3 → ℝ) :
    quad M u = M 0 0 * u 0 * u 0 + M 0 1 * u 0 * u 1 + M 0 2 * u 0 * u 2
      + M 1 0 * u 1 * u 0 + M 1 1 * u 1 * u 1 + M 1 2 * u 1 * u 2
      + M 2 0 * u 2 * u 0 + M 2 1 * u 2 * u 1 + M 2 2 * u 2 * u 2 := by
  unfold quad
  simp [Fin.sum_univ_three]
  ring

/-- **Five suitably chosen bond directions determine the alignment tensor.**  This is the linear
algebra behind "five independent alignment media": the two coordinate directions `e₁`, `e₂` and
the three diagonal directions `(e_i + e_j)/√2` suffice. -/
theorem rdc_five_directions_determine {A B : Matrix (Fin 3) (Fin 3) ℝ}
    (hA : Alignment A) (hB : Alignment B)
    (h0 : rdc A ![1, 0, 0] = rdc B ![1, 0, 0])
    (h1 : rdc A ![0, 1, 0] = rdc B ![0, 1, 0])
    (h2 : rdc A ![rt, rt, 0] = rdc B ![rt, rt, 0])
    (h3 : rdc A ![rt, 0, rt] = rdc B ![rt, 0, rt])
    (h4 : rdc A ![0, rt, rt] = rdc B ![0, rt, rt]) : A = B := by
  have hquad : ∀ (M : Matrix (Fin 3) (Fin 3) ℝ), Matrix.trace M = 0 →
      ∀ u : Fin 3 → ℝ, rdc M u = 3 * quad M u / 2 := by
    intro M hM u
    unfold rdc
    rw [hM]
    ring
  have qe1 : ∀ M : Matrix (Fin 3) (Fin 3) ℝ, quad M ![1, 0, 0] = M 0 0 := by
    intro M
    rw [quad_expand]
    simp
  have qe2 : ∀ M : Matrix (Fin 3) (Fin 3) ℝ, quad M ![0, 1, 0] = M 1 1 := by
    intro M
    rw [quad_expand]
    simp
  have qd01 : ∀ M : Matrix (Fin 3) (Fin 3) ℝ,
      quad M ![rt, rt, 0] = (M 0 0 + M 0 1 + M 1 0 + M 1 1) / 2 := by
    intro M
    rw [quad_expand]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
    linear_combination (M 0 0 + M 0 1 + M 1 0 + M 1 1) * rt_mul_rt
  have qd02 : ∀ M : Matrix (Fin 3) (Fin 3) ℝ,
      quad M ![rt, 0, rt] = (M 0 0 + M 0 2 + M 2 0 + M 2 2) / 2 := by
    intro M
    rw [quad_expand]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
    linear_combination (M 0 0 + M 0 2 + M 2 0 + M 2 2) * rt_mul_rt
  have qd12 : ∀ M : Matrix (Fin 3) (Fin 3) ℝ,
      quad M ![0, rt, rt] = (M 1 1 + M 1 2 + M 2 1 + M 2 2) / 2 := by
    intro M
    rw [quad_expand]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
    linear_combination (M 1 1 + M 1 2 + M 2 1 + M 2 2) * rt_mul_rt
  rw [hquad A hA.traceless, hquad B hB.traceless, qe1, qe1] at h0
  rw [hquad A hA.traceless, hquad B hB.traceless, qe2, qe2] at h1
  rw [hquad A hA.traceless, hquad B hB.traceless, qd01, qd01] at h2
  rw [hquad A hA.traceless, hquad B hB.traceless, qd02, qd02] at h3
  rw [hquad A hA.traceless, hquad B hB.traceless, qd12, qd12] at h4
  have hsA : ∀ i j, A i j = A j i := fun i j => by
    simpa [Matrix.transpose] using congrFun (congrFun hA.symm j) i
  have hsB : ∀ i j, B i j = B j i := fun i j => by
    simpa [Matrix.transpose] using congrFun (congrFun hB.symm j) i
  have htA : A 0 0 + A 1 1 + A 2 2 = 0 := by
    simpa [Matrix.trace, Matrix.diag, Fin.sum_univ_three] using hA.traceless
  have htB : B 0 0 + B 1 1 + B 2 2 = 0 := by
    simpa [Matrix.trace, Matrix.diag, Fin.sum_univ_three] using hB.traceless
  have a10 := hsA 1 0
  have a20 := hsA 2 0
  have a21 := hsA 2 1
  have b10 := hsB 1 0
  have b20 := hsB 2 0
  have b21 := hsB 2 1
  have h00 : A 0 0 = B 0 0 := by linarith
  have h11 : A 1 1 = B 1 1 := by linarith
  have h01 : A 0 1 = B 0 1 := by linarith
  have h02 : A 0 2 = B 0 2 := by linarith
  have h12 : A 1 2 = B 1 2 := by linarith
  rw [alignment_decomposition hA, alignment_decomposition hB, h00, h11, h01, h02, h12]

end Rdc
end IDR
