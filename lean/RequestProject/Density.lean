/-
# Part XXXVIII  Density maps: occupancy and disorder are the same parameter

The operational definition of a disordered region, in the structural databases, is negative: it
is the part of the chain for which there is no interpretable density in the crystallographic or
cryo-EM map.  This file asks what that absence is a statement about, in the standard harmonic
(Debye–Waller) treatment of an atom smeared by disorder: occupancy `q`, isotropic displacement
parameter `B`, structure-factor contribution `formFactor q B s = q·exp(−B s²/4)` at resolution
shell `s`, and real-space peak height `peak q B = q·(4π/B)^{3/2}` for the corresponding Gaussian.

Reciprocal space, first.

* `single_shell_degenerate` -- one resolution shell determines nothing: for any occupancy
  whatsoever there is a `B` reproducing the measured amplitude exactly.
* `two_shells_identify` -- two shells determine both, so the pair is identifiable in principle.
* `formFactor_close` -- but not in practice, and the failure is quantitative: two models agreeing
  at a shell `s₀` differ at a shell `s > s₀` by at most `F₀·|B − B'|·(s² − s₀²)/4`.  Resolving a
  `B`-factor difference therefore requires data quality proportional to the *span* of resolution
  actually measured, which is exactly what a disordered region, whose scattering has decayed
  away, does not provide.

Real space, second.

* `peak_scale_invariant` -- the exact statement of the classical occupancy–`B` correlation:
  multiplying the occupancy by `c³` and the displacement parameter by `c²` leaves the peak height
  unchanged, for every `c > 0`.  A missing side chain and a fully occupied but mobile one are the
  same map.
* `peak_antitone`, `peak_hundred` -- disorder costs peak height at a fixed rate: raising `B` from
  25 Å² to 100 Å² divides the peak by eight at unchanged occupancy.
* `visibility_bound`, `invisible_of_large_B` -- and therefore the operational definition is a
  threshold statement: a peak at or above a contour level `τ` forces `B³τ² ≤ (4π)³q²`, and
  conversely any `B` with `(4π)³q² < B³τ²` is invisible at that contour.  "No density" is a bound
  on `q²/B³`, not the absence of a residue.

Design consequence, and it is the one that motivates the whole development: a model of a
disordered region cannot be validated against a deposited structure, because the deposited
structure does not contain the region.  What a map can be compared with is a forward-modelled
density from the *ensemble*, at the occupancy and displacement parameters the ensemble itself
implies — and the comparison determines only the combination `q²/B³` unless the resolution span
is wide enough to separate them.
-/
import Mathlib

set_option autoImplicit false

namespace Dens

/-! ### Reciprocal space: the structure-factor contribution -/

/-- The Debye–Waller contribution of an atom of occupancy `q` and isotropic displacement
parameter `B` to the structure factor at resolution shell `s = 1/d`. -/
noncomputable def formFactor (q B s : ℝ) : ℝ := q * Real.exp (-B * s ^ 2 / 4)

lemma formFactor_pos {q B s : ℝ} (hq : 0 < q) : 0 < formFactor q B s :=
  mul_pos hq (Real.exp_pos _)

/-- **One resolution shell determines nothing.**  Whatever occupancy is assumed, a displacement
parameter reproduces the amplitude measured in a single shell exactly. -/
theorem single_shell_degenerate {q q' s₀ : ℝ} (hq : 0 < q) (hq' : 0 < q') (hs : s₀ ≠ 0) (B : ℝ) :
    ∃ B' : ℝ, B' = B + 4 * Real.log (q' / q) / s₀ ^ 2 ∧
      formFactor q' B' s₀ = formFactor q B s₀ := by
  refine ⟨B + 4 * Real.log (q' / q) / s₀ ^ 2, rfl, ?_⟩
  have hs2 : s₀ ^ 2 ≠ 0 := pow_ne_zero 2 hs
  have hratio : (0:ℝ) < q' / q := div_pos hq' hq
  have harg : -(B + 4 * Real.log (q' / q) / s₀ ^ 2) * s₀ ^ 2 / 4
      = -B * s₀ ^ 2 / 4 + -Real.log (q' / q) := by
    field_simp
    ring
  simp only [formFactor, harg, Real.exp_add, Real.exp_neg, Real.exp_log hratio]
  field_simp

/-- **Two resolution shells determine both.**  Occupancy and displacement parameter are
identifiable from amplitudes in two distinct shells. -/
theorem two_shells_identify {q q' B B' s₁ s₂ : ℝ} (hq : 0 < q) (hq' : 0 < q')
    (hs : s₁ ^ 2 ≠ s₂ ^ 2)
    (h1 : formFactor q B s₁ = formFactor q' B' s₁)
    (h2 : formFactor q B s₂ = formFactor q' B' s₂) : q = q' ∧ B = B' := by
  have key : ∀ s : ℝ, formFactor q B s = formFactor q' B' s →
      Real.log q - B * s ^ 2 / 4 = Real.log q' - B' * s ^ 2 / 4 := by
    intro s hs'
    have := congrArg Real.log hs'
    simp only [formFactor, Real.log_mul (ne_of_gt hq) (ne_of_gt (Real.exp_pos _)),
      Real.log_mul (ne_of_gt hq') (ne_of_gt (Real.exp_pos _)), Real.log_exp] at this
    linarith [this]
  have e1 := key s₁ h1
  have e2 := key s₂ h2
  have hB : B = B' := by
    have hsub : (B' - B) * (s₁ ^ 2 - s₂ ^ 2) / 4 = 0 := by linarith
    have hne : s₁ ^ 2 - s₂ ^ 2 ≠ 0 := sub_ne_zero.mpr hs
    have : B' - B = 0 := by
      rcases mul_eq_zero.mp (by linarith [hsub] : (B' - B) * (s₁ ^ 2 - s₂ ^ 2) = 0) with h | h
      · exact h
      · exact absurd h hne
    linarith
  refine ⟨?_, hB⟩
  have : Real.log q = Real.log q' := by rw [hB] at e1; linarith
  exact Real.log_injOn_pos (Set.mem_Ioi.mpr hq) (Set.mem_Ioi.mpr hq') this

/-- `|e^x − e^y| ≤ |x − y|` for nonpositive exponents. -/
lemma abs_exp_sub_le_of_nonpos {x y : ℝ} (hx : x ≤ 0) (hy : y ≤ 0) :
    |Real.exp x - Real.exp y| ≤ |x - y| := by
  have main : ∀ a b : ℝ, b ≤ a → a ≤ 0 → Real.exp a - Real.exp b ≤ a - b := by
    intro a b hba ha
    have ht : 0 ≤ a - b := by linarith
    have hexp : Real.exp (a - b) - 1 ≤ (a - b) * Real.exp (a - b) := by
      have h := Real.add_one_le_exp (-(a - b))
      have hpos : 0 < Real.exp (a - b) := Real.exp_pos _
      have hinv : Real.exp (-(a - b)) = (Real.exp (a - b))⁻¹ := Real.exp_neg _
      rw [hinv] at h
      have := mul_le_mul_of_nonneg_right h hpos.le
      rw [add_mul, inv_mul_cancel₀ (ne_of_gt hpos)] at this
      nlinarith [this]
    have hfac : Real.exp a - Real.exp b = Real.exp b * (Real.exp (a - b) - 1) := by
      rw [mul_sub, ← Real.exp_add]
      ring_nf
    rw [hfac]
    have hb1 : Real.exp b ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    have hea : Real.exp b * Real.exp (a - b) = Real.exp a := by
      rw [← Real.exp_add]; ring_nf
    have ha1 : Real.exp a ≤ 1 := Real.exp_le_one_iff.mpr ha
    nlinarith [Real.exp_pos b, Real.exp_pos (a - b), Real.exp_pos a]
  rcases le_total y x with h | h
  · rw [abs_of_nonneg (by nlinarith [Real.exp_le_exp.mpr h] : 0 ≤ Real.exp x - Real.exp y),
      abs_of_nonneg (by linarith : (0:ℝ) ≤ x - y)]
    exact main x y h hx
  · rw [abs_sub_comm, abs_sub_comm x y,
      abs_of_nonneg (by nlinarith [Real.exp_le_exp.mpr h] : 0 ≤ Real.exp y - Real.exp x),
      abs_of_nonneg (by linarith : (0:ℝ) ≤ y - x)]
    exact main y x h hy

/-- **Separating occupancy from disorder needs a span of resolution.**  Two models that agree at
a shell `s₀` differ at any higher shell `s` by at most `F₀·|B − B'|·(s² − s₀²)/4`, where `F₀` is
their common amplitude at `s₀`. -/
theorem formFactor_close {q q' B B' s₀ s : ℝ} (hB : 0 ≤ B) (hB' : 0 ≤ B')
    (hs : s₀ ^ 2 ≤ s ^ 2) (hagree : formFactor q B s₀ = formFactor q' B' s₀) (hq : 0 < q) :
    |formFactor q B s - formFactor q' B' s|
      ≤ formFactor q B s₀ * (|B - B'| * (s ^ 2 - s₀ ^ 2) / 4) := by
  set w : ℝ := s ^ 2 - s₀ ^ 2 with hw
  have hw0 : 0 ≤ w := by simp [hw]; linarith
  have hF : ∀ (Q C : ℝ), formFactor Q C s = formFactor Q C s₀ * Real.exp (-C * w / 4) := by
    intro Q C
    simp only [formFactor, hw, mul_assoc, ← Real.exp_add]
    congr 2
    ring
  rw [hF q B, hF q' B', hagree, ← mul_sub, abs_mul]
  have hFpos : 0 < formFactor q' B' s₀ := by
    rw [← hagree]; exact formFactor_pos hq
  rw [abs_of_pos hFpos]
  refine mul_le_mul_of_nonneg_left ?_ hFpos.le
  have hx : -B * w / 4 ≤ 0 := by
    have : 0 ≤ B * w := mul_nonneg hB hw0
    linarith
  have hy : -B' * w / 4 ≤ 0 := by
    have : 0 ≤ B' * w := mul_nonneg hB' hw0
    linarith
  have hbound := abs_exp_sub_le_of_nonpos hx hy
  have harg : -B * w / 4 - -B' * w / 4 = (B' - B) * w / 4 := by ring
  rw [harg] at hbound
  calc |Real.exp (-B * w / 4) - Real.exp (-B' * w / 4)| ≤ |(B' - B) * w / 4| := hbound
    _ = |B - B'| * w / 4 := by
        rw [abs_div, abs_mul, abs_of_nonneg hw0, abs_sub_comm]
        simp

/-! ### Real space: the peak height -/

/-- The peak height of the Gaussian density of an atom of occupancy `q` and isotropic
displacement parameter `B`, namely `q·(4π/B)^{3/2}`. -/
noncomputable def peak (q B : ℝ) : ℝ := q * (4 * Real.pi / B) * Real.sqrt (4 * Real.pi / B)

lemma peak_pos {q B : ℝ} (hq : 0 < q) (hB : 0 < B) : 0 < peak q B := by
  have h : 0 < 4 * Real.pi / B := by positivity
  have : 0 < Real.sqrt (4 * Real.pi / B) := Real.sqrt_pos.mpr h
  simp only [peak]
  positivity

/-- **The occupancy–`B` degeneracy, exactly.**  Multiplying the occupancy by `c³` and the
displacement parameter by `c²` leaves the peak height unchanged. -/
theorem peak_scale_invariant (q : ℝ) {B c : ℝ} (hB : 0 < B) (hc : 0 < c) :
    peak (c ^ 3 * q) (c ^ 2 * B) = peak q B := by
  have hnn : (0:ℝ) ≤ 4 * Real.pi / B := by positivity
  have hrw : 4 * Real.pi / (c ^ 2 * B) = (4 * Real.pi / B) * (c⁻¹) ^ 2 := by
    field_simp
  have hsqrt : Real.sqrt (4 * Real.pi / (c ^ 2 * B))
      = Real.sqrt (4 * Real.pi / B) * c⁻¹ := by
    rw [hrw, Real.sqrt_mul hnn, Real.sqrt_sq (by positivity : (0:ℝ) ≤ c⁻¹)]
  have key : peak (c ^ 3 * q) (c ^ 2 * B)
      = (c ^ 3 * q) * ((4 * Real.pi / B) * (c⁻¹) ^ 2)
          * (Real.sqrt (4 * Real.pi / B) * c⁻¹) := by
    unfold peak
    rw [hsqrt, hrw]
  rw [key]
  unfold peak
  generalize Real.sqrt (4 * Real.pi / B) = S
  field_simp

/-- Disorder costs peak height: at fixed occupancy the peak is decreasing in `B`. -/
theorem peak_antitone {q B B' : ℝ} (hq : 0 ≤ q) (hB : 0 < B) (hBB : B ≤ B') :
    peak q B' ≤ peak q B := by
  have hB' : 0 < B' := lt_of_lt_of_le hB hBB
  have h1 : 4 * Real.pi / B' ≤ 4 * Real.pi / B := by
    apply div_le_div_of_nonneg_left (by positivity) hB hBB
  have h2 : Real.sqrt (4 * Real.pi / B') ≤ Real.sqrt (4 * Real.pi / B) :=
    Real.sqrt_le_sqrt h1
  have hnn : 0 ≤ 4 * Real.pi / B' := by positivity
  have hs : 0 ≤ Real.sqrt (4 * Real.pi / B') := Real.sqrt_nonneg _
  simp only [peak]
  calc q * (4 * Real.pi / B') * Real.sqrt (4 * Real.pi / B')
      ≤ q * (4 * Real.pi / B) * Real.sqrt (4 * Real.pi / B') := by
        exact mul_le_mul_of_nonneg_right (by nlinarith) hs
    _ ≤ q * (4 * Real.pi / B) * Real.sqrt (4 * Real.pi / B) := by
        have : 0 ≤ q * (4 * Real.pi / B) := by positivity
        exact mul_le_mul_of_nonneg_left h2 this

/-- Concretely: raising `B` from 25 Å² to 100 Å² divides the peak height by eight. -/
theorem peak_hundred (q : ℝ) : peak q 100 = peak q 25 / 8 := by
  have h := peak_scale_invariant q (B := 25) (c := 2) (by norm_num) (by norm_num)
  have hlin : peak ((2:ℝ) ^ 3 * q) ((2:ℝ) ^ 2 * 25) = 8 * peak q ((2:ℝ) ^ 2 * 25) := by
    simp only [peak]
    ring
  rw [hlin] at h
  norm_num at h
  linarith

lemma peak_sq {q B : ℝ} (hB : 0 < B) : (peak q B) ^ 2 = q ^ 2 * (4 * Real.pi / B) ^ 3 := by
  have hnn : (0:ℝ) ≤ 4 * Real.pi / B := by positivity
  simp only [peak, mul_pow]
  rw [Real.sq_sqrt hnn]
  ring

/-- **What visible density bounds.**  If the peak reaches a contour level `τ`, then
`B³τ² ≤ (4π)³q²`. -/
theorem visibility_bound {q B tau : ℝ} (hq : 0 < q) (hB : 0 < B) (htau : 0 < tau)
    (hvis : tau ≤ peak q B) : B ^ 3 * tau ^ 2 ≤ (4 * Real.pi) ^ 3 * q ^ 2 := by
  have hsq : tau ^ 2 ≤ (peak q B) ^ 2 := by
    have hp : 0 < peak q B := lt_of_lt_of_le htau hvis
    nlinarith
  rw [peak_sq hB] at hsq
  have hexp : q ^ 2 * (4 * Real.pi / B) ^ 3 = q ^ 2 * (4 * Real.pi) ^ 3 / B ^ 3 := by
    field_simp
  rw [hexp] at hsq
  rw [← sub_nonneg]
  have hB3 : 0 < B ^ 3 := by positivity
  have := mul_le_mul_of_nonneg_right hsq hB3.le
  rw [div_mul_cancel₀ _ (ne_of_gt hB3)] at this
  nlinarith

/-- **And its contrapositive: absence of density is a bound, not an absence of atoms.**  Any
displacement parameter with `(4π)³q² < B³τ²` produces no peak at contour level `τ`, whatever the
occupancy. -/
theorem invisible_of_large_B {q B tau : ℝ} (hq : 0 < q) (hB : 0 < B) (htau : 0 < tau)
    (hbig : (4 * Real.pi) ^ 3 * q ^ 2 < B ^ 3 * tau ^ 2) : peak q B < tau := by
  by_contra hc
  push_neg at hc
  exact absurd (visibility_bound hq hB htau hc) (not_le.mpr hbig)

end Dens
