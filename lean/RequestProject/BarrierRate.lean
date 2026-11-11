/-
# Part L.1  A barrier is neither necessary nor sufficient for slow kinetics

`RequestProject.FirstPassage` proves that the equilibrium profile leaves the crossing time free
by an arbitrary factor, and `RequestProject.Committor` that it leaves the transition state free.
This file states the consequence in the form in which it is usually needed: **given only the
landscape, no bound on the rate holds in either direction**, and the exact conditions under which
the familiar Arrhenius reasoning becomes a theorem.

* `flatLandscape_arbitrarily_slow` -- for every target `M` there is a hopping model on a
  perfectly *flat* three-state landscape -- no barrier anywhere, all three states equally
  populated -- whose crossing time exceeds `M`.  Slowness is not evidence of a barrier.
* `barrier_arbitrarily_fast` -- conversely, for every barrier height `B` and every `eps > 0`
  there is a hopping model whose profile has a barrier of exactly height `B`
  (`p 0 / p 1 = exp B`) and whose crossing time is exactly `eps`.  A barrier is not evidence of
  slowness.
* `barrier_brackets_rate` -- and the positive statement that survives: **once the kinetic
  prefactor is bounded**, `kmin ≤ kp i ≤ kmax`, the crossing time is bracketed between
  `exp (β (F b - F 0)) / kmax` and `(Σ_{i<n} 1/p i) / kmin`.  Arrhenius reasoning is a theorem
  about a landscape *plus* a bounded diffusion profile, and about nothing less.

The two counterexamples use the same machinery as the general theorems: an explicit rate profile,
its detailed-balanced backward rates, and the exact first-passage formula.
-/
import Mathlib
import RequestProject.FirstPassage

set_option autoImplicit false

namespace IDR

namespace BarrierRate

open Finset
open IDR.FirstPassage

/-! ## A flat landscape can be arbitrarily slow -/

/-- Forward rates on the flat three-state landscape whose second step has rate `d`. -/
noncomputable def slowRate (d : ℝ) : ℕ → ℝ := fun i => if i = 1 then d else 1

/-- The backward rates detailed balance forces on `slowRate d`. -/
noncomputable def slowRateBack (d : ℝ) : ℕ → ℝ :=
  fun i => if i = 0 then 0 else if i = 2 then d else 1

lemma slowRate_detailedBalance (d : ℝ) : DetailedBalance 2 flatP (slowRate d) (slowRateBack d) := by
  intro i hi
  interval_cases i <;> norm_num [flatP, slowRate, slowRateBack]

lemma mfpt_slowRate (d : ℝ) :
    mfptFormula 2 flatP (slowRate d) = 1 + 2 / d := by
  rw [mfptFormula_eq]
  norm_num [Finset.sum_range_succ, cum, flatP, slowRate]

/-- **A flat landscape can be arbitrarily slow.**  For every `M` there is a hopping model whose
equilibrium profile is exactly flat -- every state equally populated, no barrier anywhere -- and
whose mean first-passage time exceeds `M`.  A slow conformational transition in a disordered
region is therefore not, by itself, evidence of a free-energy barrier. -/
theorem flatLandscape_arbitrarily_slow (M : ℝ) :
    ∃ d : ℝ, 0 < d ∧ DetailedBalance 2 flatP (slowRate d) (slowRateBack d) ∧
      (∀ i, i ≤ 2 → flatP i = 1) ∧ (∀ i, i < 2 → 0 < slowRate d i) ∧ slowRateBack d 0 = 0 ∧
      M ≤ mfptFormula 2 flatP (slowRate d) := by
  refine ⟨2 / (|M| + 1), by positivity, slowRate_detailedBalance _, fun i _ => rfl, ?_, by
    norm_num [slowRateBack], ?_⟩
  · intro i _
    simp only [slowRate]
    split
    · positivity
    · norm_num
  · rw [mfpt_slowRate]
    have h1 : (0:ℝ) < |M| + 1 := by positivity
    have : 2 / (2 / (|M| + 1)) = |M| + 1 := by field_simp
    rw [this]
    have := le_abs_self M
    linarith

/-! ## A barrier can be arbitrarily fast -/

/-- The Boltzmann profile of a three-state landscape with a barrier of height `B` in the middle
(energies `0, B, 0`). -/
noncomputable def barrierP (B : ℝ) : ℕ → ℝ := fun i => if i = 1 then Real.exp (-B) else 1

/-- Uniform forward rates of size `R`. -/
noncomputable def fastRate (R : ℝ) : ℕ → ℝ := fun _ => R

/-- The backward rates detailed balance forces on `barrierP B` and `fastRate R`. -/
noncomputable def fastRateBack (B R : ℝ) : ℕ → ℝ :=
  fun i => if i = 1 then R * Real.exp B else if i = 2 then R * Real.exp (-B) else 0

lemma barrier_detailedBalance (B R : ℝ) :
    DetailedBalance 2 (barrierP B) (fastRate R) (fastRateBack B R) := by
  intro i hi
  interval_cases i
  · simp only [barrierP, fastRate, fastRateBack]
    norm_num
    have h : Real.exp (-B) * Real.exp B = 1 := by
      rw [← Real.exp_add]
      simp
    linear_combination (-R) * h
  · simp only [barrierP, fastRate, fastRateBack]
    norm_num
    ring

lemma barrier_height (B : ℝ) : barrierP B 0 / barrierP B 1 = Real.exp B := by
  simp only [barrierP]
  norm_num
  rw [← Real.exp_neg]
  norm_num

lemma mfpt_barrier {B R : ℝ} (hR : 0 < R) :
    mfptFormula 2 (barrierP B) (fastRate R) = (2 + Real.exp B) / R := by
  rw [mfptFormula_eq]
  have hexp : Real.exp (-B) ≠ 0 := ne_of_gt (Real.exp_pos _)
  have hinv : (Real.exp (-B))⁻¹ = Real.exp B := by
    rw [← Real.exp_neg]
    norm_num
  norm_num [Finset.sum_range_succ, cum, barrierP, fastRate]
  field_simp
  have h : Real.exp (-B) * Real.exp B = 1 := by
    rw [← Real.exp_add]
    simp
  linear_combination -h

/-- **A barrier can be arbitrarily fast.**  For every barrier height `B` and every target time
`eps > 0` there is a hopping model whose equilibrium profile has a barrier of exactly height `B`
(`p 0 / p 1 = exp B`) and whose mean first-passage time is exactly `eps`.  A barrier in a
reported free-energy profile is therefore not, by itself, evidence of slow kinetics. -/
theorem barrier_arbitrarily_fast (B : ℝ) {eps : ℝ} (heps : 0 < eps) :
    ∃ R : ℝ, 0 < R ∧ DetailedBalance 2 (barrierP B) (fastRate R) (fastRateBack B R) ∧
      barrierP B 0 / barrierP B 1 = Real.exp B ∧
      mfptFormula 2 (barrierP B) (fastRate R) = eps := by
  refine ⟨(2 + Real.exp B) / eps, by positivity, barrier_detailedBalance _ _, barrier_height B, ?_⟩
  rw [mfpt_barrier (by positivity)]
  field_simp

/-! ## What survives: the Arrhenius bracket -/

/-- **The Arrhenius bracket.**  With the kinetic prefactor bounded between `kmin` and `kmax`, the
crossing time of a Boltzmann profile `p i = exp (-β F i)` is bracketed: below by the Arrhenius
factor of any intermediate state divided by `kmax`, above by the total resistance of the profile
divided by `kmin`.  This is the exact sense in which a free-energy landscape predicts a rate, and
it is a statement about the landscape *together with* a bounded diffusion profile. -/
theorem barrier_brackets_rate {n : ℕ} {p kp : ℕ → ℝ} {kmin kmax : ℝ} {b : ℕ} (hb : b < n)
    (hp : ∀ i, i ≤ n → 0 < p i) (hkmin : 0 < kmin)
    (hlo : ∀ i, i < n → kmin ≤ kp i) (hhi : ∀ i, i < n → kp i ≤ kmax)
    (hnorm : ∀ i, i < n → cum p i ≤ 1) :
    p 0 / (p b * kmax) ≤ mfptFormula n p kp ∧
      mfptFormula n p kp ≤ (∑ i ∈ range n, 1 / p i) / kmin := by
  have hkp : ∀ i, i < n → 0 < kp i := fun i hi => lt_of_lt_of_le hkmin (hlo i hi)
  constructor
  · refine le_trans ?_ (mfptFormula_ge_barrier hb hp hkp)
    have hpb : 0 < p b := hp b hb.le
    have hkpb : 0 < kp b := hkp b hb
    have hmax : kp b ≤ kmax := hhi b hb
    have h0 : 0 < p 0 := hp 0 (Nat.zero_le n)
    gcongr
  · refine le_trans (mfptFormula_le hp hkp hnorm) ?_
    rw [Finset.sum_div]
    refine Finset.sum_le_sum (fun i hi => ?_)
    have hi' : i < n := Finset.mem_range.mp hi
    have hpi : 0 < p i := hp i hi'.le
    have hlo' : kmin ≤ kp i := hlo i hi'
    rw [div_div]
    gcongr

end BarrierRate

end IDR
