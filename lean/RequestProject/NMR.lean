/-
# Part X.1  Orientational order: what NMR relaxation and RDCs measure

Distances (Part IX.2) are only half of the experimental record.  The other half is
*orientational*: NMR spin relaxation reports the generalised order parameter `S²` of a bond
vector, and residual dipolar couplings report the second Legendre polynomial of the angle
between the bond vector and the alignment frame.  Both are quadratic in the bond direction,
so both are genuinely ensemble observables: they cannot be reproduced by any single
structure unless the region is rigid.

This file fixes the exact forward models and proves their sharp properties.

* `orderTensor` -- the second-rank orientational order tensor `M_{ab} = Σ_k w_k u_{ka}u_{kb}`
  of a weighted ensemble of unit bond vectors, and `orderParam` the Lipari--Szabo
  generalised order parameter `S² = (3/2)Σ_{ab}M_{ab}² - 1/2`.
* `orderParam_eq_pair` -- the exact identity `S² = (3/2)Σ_{kl} w_k w_l ⟨u_k,u_l⟩² - 1/2`,
  the form in which `S²` is computed from a conformational ensemble.
* `orderParam_nonneg` and `orderParam_le_one` -- `0 ≤ S² ≤ 1`, with the lower bound coming
  from the three-dimensionality of space (it is `(tr M)² ≤ 3 tr M²`) and the upper bound
  from Cauchy--Schwarz.
* `orderParam_eq_one_of_aligned` -- a rigid bond vector has `S² = 1`;
  `orientational_disorder_of_orderParam_lt_one` -- the converse in the usable direction: a
  measured `S² < 1` *proves* the existence of two populated, non-parallel orientations.  This
  is the observable that certifies disorder, and no single structure can carry it.
* `rdc` -- the residual dipolar coupling, `D = D_max Σ_k w_k P₂(cos θ_k)`; `rdc_abs_le` its
  sharp range, and `rdc_cancellation` an explicit three-orientation ensemble with `D = 0` in
  which *no* member has `D = 0`.  A vanishing RDC is not evidence of a vanishing bond
  anisotropy; it is evidence of averaging.
-/
import Mathlib

namespace IDR

open Finset

namespace NMR

variable {m : ℕ}

/-! ## The order tensor -/

/-- The Euclidean inner product on three-space, written out in coordinates. -/
def dot (u v : Fin 3 → ℝ) : ℝ := ∑ a, u a * v a

lemma dot_comm (u v : Fin 3 → ℝ) : dot u v = dot v u := by
  simp only [dot]; exact Finset.sum_congr rfl fun a _ => mul_comm _ _

/-- A bond direction: a unit vector of three-space. -/
def IsBondVector (u : Fin 3 → ℝ) : Prop := dot u u = 1

/-- Cauchy--Schwarz for bond directions: `⟨u,v⟩² ≤ 1`. -/
lemma dot_sq_le_one {u v : Fin 3 → ℝ} (hu : IsBondVector u) (hv : IsBondVector v) :
    dot u v ^ 2 ≤ 1 := by
  have h := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ u v
  have hu' : ∑ a, u a ^ 2 = 1 := by simpa [dot, pow_two] using hu
  have hv' : ∑ a, v a ^ 2 = 1 := by simpa [dot, pow_two] using hv
  calc dot u v ^ 2 = (∑ a, u a * v a) ^ 2 := rfl
    _ ≤ (∑ a, u a ^ 2) * ∑ a, v a ^ 2 := h
    _ = 1 := by rw [hu', hv']; ring

/-- The second-rank orientational order tensor of a weighted ensemble of bond vectors. -/
def orderTensor (w : Fin m → ℝ) (u : Fin m → Fin 3 → ℝ) (a b : Fin 3) : ℝ :=
  ∑ k, w k * (u k a * u k b)

/-- The Lipari--Szabo generalised order parameter `S² = (3/2)Σ_{ab}M_{ab}² - 1/2`. -/
noncomputable def orderParam (w : Fin m → ℝ) (u : Fin m → Fin 3 → ℝ) : ℝ :=
  3 / 2 * ∑ a, ∑ b, orderTensor w u a b ^ 2 - 1 / 2

/-- A square of a sum, resummed over the pair of summation indices. -/
private lemma sum_sq_sum {ι κ : Type*} [Fintype ι] [Fintype κ] (f : ι → κ → ℝ) :
    ∑ a, (∑ k, f k a) ^ 2 = ∑ k, ∑ l, ∑ a, f k a * f l a := by
  simp_rw [sq, Finset.sum_mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun _ _ => Finset.sum_comm

/-- **The order parameter as a pair average.**  Exact identity, no approximation:
`S² = (3/2) Σ_{kl} w_k w_l ⟨u_k,u_l⟩² - 1/2`.  This is the formula by which `S²` is
predicted from a conformational ensemble. -/
theorem orderParam_eq_pair (w : Fin m → ℝ) (u : Fin m → Fin 3 → ℝ) :
    orderParam w u = 3 / 2 * ∑ k, ∑ l, w k * w l * dot (u k) (u l) ^ 2 - 1 / 2 := by
  have key : ∑ a, ∑ b, orderTensor w u a b ^ 2
      = ∑ k, ∑ l, w k * w l * dot (u k) (u l) ^ 2 := by
    have h1 : ∑ a, ∑ b, orderTensor w u a b ^ 2
        = ∑ p : Fin 3 × Fin 3, (∑ k, w k * (u k p.1 * u k p.2)) ^ 2 := by
      rw [Fintype.sum_prod_type]; rfl
    rw [h1, sum_sq_sum (fun (k : Fin m) (p : Fin 3 × Fin 3) => w k * (u k p.1 * u k p.2))]
    refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => ?_
    rw [Fintype.sum_prod_type]
    have h1 : ∀ a : Fin 3, ∑ b : Fin 3, w k * (u k a * u k b) * (w l * (u l a * u l b))
        = w k * w l * (u k a * u l a) * dot (u k) (u l) := by
      intro a
      rw [dot, Finset.mul_sum]
      exact Finset.sum_congr rfl fun b _ => by ring
    simp only [h1, ← Finset.sum_mul]
    have h2 : ∑ a : Fin 3, w k * w l * (u k a * u l a) = w k * w l * dot (u k) (u l) := by
      rw [dot, Finset.mul_sum]
    rw [h2]; ring
  rw [orderParam, key]

/-- The order tensor of a normalised ensemble of unit vectors has unit trace. -/
theorem orderTensor_trace {w : Fin m → ℝ} {u : Fin m → Fin 3 → ℝ} (hsum : ∑ k, w k = 1)
    (hu : ∀ k, IsBondVector (u k)) : ∑ a, orderTensor w u a a = 1 := by
  have : ∑ a, orderTensor w u a a = ∑ k, w k * dot (u k) (u k) := by
    simp only [orderTensor, dot, Finset.mul_sum]
    exact Finset.sum_comm
  rw [this, ← hsum]
  refine Finset.sum_congr rfl fun k _ => ?_
  have hk : dot (u k) (u k) = 1 := hu k
  rw [hk, mul_one]

/-- **The order parameter never exceeds one.**  Cauchy--Schwarz on each pair term. -/
theorem orderParam_le_one {w : Fin m → ℝ} {u : Fin m → Fin 3 → ℝ} (hw : ∀ k, 0 ≤ w k)
    (hsum : ∑ k, w k = 1) (hu : ∀ k, IsBondVector (u k)) : orderParam w u ≤ 1 := by
  rw [orderParam_eq_pair]
  have hle : ∑ k, ∑ l, w k * w l * dot (u k) (u l) ^ 2 ≤ ∑ k, ∑ l, w k * w l := by
    refine Finset.sum_le_sum fun k _ => Finset.sum_le_sum fun l _ => ?_
    have := dot_sq_le_one (hu k) (hu l)
    nlinarith [mul_nonneg (hw k) (hw l)]
  have hone : ∑ k, ∑ l, w k * w l = 1 := by
    rw [← Finset.sum_mul_sum, hsum]; ring
  rw [hone] at hle
  linarith

/-- **The order parameter is nonnegative.**  This is the three-dimensionality of space:
`(tr M)² ≤ 3 tr M²`, with `tr M = 1`. -/
theorem orderParam_nonneg {w : Fin m → ℝ} {u : Fin m → Fin 3 → ℝ} (hsum : ∑ k, w k = 1)
    (hu : ∀ k, IsBondVector (u k)) : 0 ≤ orderParam w u := by
  have htr : ∑ a, orderTensor w u a a = 1 := orderTensor_trace hsum hu
  set M := orderTensor w u with hM
  have hsym : ∀ a b, M a b = M b a := by
    intro a b; simp only [hM, orderTensor]
    exact Finset.sum_congr rfl fun k _ => by ring
  have hdiag : ∑ a, ∑ b, M a b ^ 2 ≥ ∑ a, M a a ^ 2 := by
    refine Finset.sum_le_sum fun a _ => ?_
    exact Finset.single_le_sum (f := fun b => M a b ^ 2) (fun b _ => sq_nonneg _)
      (Finset.mem_univ a)
  have hcs : (∑ a, M a a) ^ 2 ≤ 3 * ∑ a, M a a ^ 2 := by
    simp only [Fin.sum_univ_three]
    nlinarith [sq_nonneg (M 0 0 - M 1 1), sq_nonneg (M 1 1 - M 2 2), sq_nonneg (M 0 0 - M 2 2)]
  rw [htr] at hcs
  have : (1 : ℝ) / 3 ≤ ∑ a, ∑ b, M a b ^ 2 := by linarith
  simp only [orderParam, ← hM]
  linarith

/-- **A rigid bond vector has `S² = 1`.**  If every populated orientation is parallel or
antiparallel to every other, the order parameter saturates. -/
theorem orderParam_eq_one_of_aligned {w : Fin m → ℝ} {u : Fin m → Fin 3 → ℝ}
    (hsum : ∑ k, w k = 1) (halign : ∀ k l, dot (u k) (u l) ^ 2 = 1) : orderParam w u = 1 := by
  rw [orderParam_eq_pair]
  have : ∑ k, ∑ l, w k * w l * dot (u k) (u l) ^ 2 = 1 := by
    simp only [halign, mul_one]
    rw [← Finset.sum_mul_sum, hsum]; ring
  rw [this]; norm_num

/-- **A measured `S² < 1` proves orientational disorder.**  There are two populated
conformers whose bond vectors are not parallel.  No single structure -- indeed no ensemble of
mutually parallel structures -- can reproduce a sub-unit order parameter. -/
theorem orientational_disorder_of_orderParam_lt_one {w : Fin m → ℝ} {u : Fin m → Fin 3 → ℝ}
    (hw : ∀ k, 0 ≤ w k) (hsum : ∑ k, w k = 1) (hu : ∀ k, IsBondVector (u k))
    (hlt : orderParam w u < 1) :
    ∃ k l, 0 < w k ∧ 0 < w l ∧ dot (u k) (u l) ^ 2 < 1 := by
  by_contra hcon
  push_neg at hcon
  have hall : ∀ k l, w k * w l * dot (u k) (u l) ^ 2 = w k * w l := by
    intro k l
    rcases (hw k).lt_or_eq with hk | hk
    · rcases (hw l).lt_or_eq with hl | hl
      · have heq : dot (u k) (u l) ^ 2 = 1 :=
          le_antisymm (dot_sq_le_one (hu k) (hu l)) (hcon k l hk hl)
        rw [heq, mul_one]
      · rw [← hl]; ring
    · rw [← hk]; ring
  have : orderParam w u = 1 := by
    rw [orderParam_eq_pair]
    have : ∑ k, ∑ l, w k * w l * dot (u k) (u l) ^ 2 = 1 := by
      simp only [hall]
      rw [← Finset.sum_mul_sum, hsum]; ring
    rw [this]; norm_num
  linarith

/-! ## Residual dipolar couplings -/

/-- The residual dipolar coupling of a bond vector ensemble in an alignment frame with
principal axis `e`: `D = D_max Σ_k w_k (3cos²θ_k - 1)/2`. -/
noncomputable def rdc (Dmax : ℝ) (w : Fin m → ℝ) (u : Fin m → Fin 3 → ℝ) (e : Fin 3 → ℝ) : ℝ :=
  Dmax * ∑ k, w k * ((3 * dot (u k) e ^ 2 - 1) / 2)

/-- The RDC of a single orientation. -/
lemma rdc_of_one (Dmax : ℝ) (u e : Fin 3 → ℝ) :
    rdc Dmax (fun _ : Fin 1 => 1) (fun _ => u) e = Dmax * ((3 * dot u e ^ 2 - 1) / 2) := by
  simp [rdc]

/-- **The sharp range of an RDC.**  `-D_max/2 ≤ D ≤ D_max` for `D_max ≥ 0`. -/
theorem rdc_bounds {Dmax : ℝ} (hD : 0 ≤ Dmax) {w : Fin m → ℝ} {u : Fin m → Fin 3 → ℝ}
    {e : Fin 3 → ℝ} (hw : ∀ k, 0 ≤ w k) (hsum : ∑ k, w k = 1) (hu : ∀ k, IsBondVector (u k))
    (he : IsBondVector e) : -(Dmax / 2) ≤ rdc Dmax w u e ∧ rdc Dmax w u e ≤ Dmax := by
  have hlow : ∀ k, w k * (-(1 / 2)) ≤ w k * ((3 * dot (u k) e ^ 2 - 1) / 2) := by
    intro k
    have h0 : 0 ≤ dot (u k) e ^ 2 := sq_nonneg _
    nlinarith [hw k]
  have hhigh : ∀ k, w k * ((3 * dot (u k) e ^ 2 - 1) / 2) ≤ w k * 1 := by
    intro k
    have := dot_sq_le_one (hu k) he
    nlinarith [hw k]
  constructor
  · have := Finset.sum_le_sum (fun k (_ : k ∈ Finset.univ) => hlow k)
    rw [← Finset.sum_mul, hsum, one_mul] at this
    have := mul_le_mul_of_nonneg_left this hD
    simpa [rdc] using this
  · have := Finset.sum_le_sum (fun k (_ : k ∈ Finset.univ) => hhigh k)
    rw [← Finset.sum_mul, hsum, one_mul] at this
    have := mul_le_mul_of_nonneg_left this hD
    simpa [rdc] using this

/-- Three coordinate directions, used as an explicit bond-vector ensemble. -/
def axis (a : Fin 3) : Fin 3 → ℝ := fun b => if b = a then 1 else 0

lemma axis_isBondVector (a : Fin 3) : IsBondVector (axis a) := by
  fin_cases a <;> simp [IsBondVector, dot, axis]

/-- Weights `(1/3, 2/3, 0)` on the three coordinate axes. -/
noncomputable def cancelWeights : Fin 3 → ℝ :=
  fun k => if k = 0 then 1 / 3 else if k = 1 then 2 / 3 else 0

lemma cancelWeights_sum : ∑ k, cancelWeights k = 1 := by
  simp [cancelWeights, Fin.sum_univ_three]
  norm_num

lemma cancelWeights_nonneg (k : Fin 3) : 0 ≤ cancelWeights k := by
  fin_cases k <;> norm_num [cancelWeights]

/-- **A vanishing RDC does not mean a vanishing bond anisotropy.**  The ensemble that puts
weight `1/3` on a bond parallel to the alignment axis and `2/3` on a bond perpendicular to it
has exactly zero residual dipolar coupling, while every populated member has a coupling of
full magnitude (`+D_max` and `-D_max/2` respectively).  Fitting a single orientation to a
measured `D = 0` therefore returns a structure that is nowhere in the ensemble. -/
theorem rdc_cancellation (Dmax : ℝ) :
    rdc Dmax cancelWeights axis (axis 0) = 0 ∧
      Dmax * ((3 * dot (axis 0) (axis 0) ^ 2 - 1) / 2) = Dmax ∧
      Dmax * ((3 * dot (axis 1) (axis 0) ^ 2 - 1) / 2) = -(Dmax / 2) := by
  have h00 : dot (axis 0) (axis (0 : Fin 3)) = 1 := by
    simp [dot, axis]
  have h10 : dot (axis 1) (axis (0 : Fin 3)) = 0 := by
    simp [dot, axis]
  have h20 : dot (axis 2) (axis (0 : Fin 3)) = 0 := by
    simp [dot, axis]
  refine ⟨?_, by rw [h00]; ring, by rw [h10]; ring⟩
  simp only [rdc, Fin.sum_univ_three, cancelWeights, h00, h10, h20,
    show (2 : Fin 3) ≠ 0 by decide, show (2 : Fin 3) ≠ 1 by decide, if_false]
  norm_num

end NMR

end IDR
