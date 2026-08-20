/-
# Part XXVIII.1  Scalar couplings: what a Karplus restraint can and cannot pin down

Three-bond scalar couplings (`³J(H^N,H^α)` and its relatives) are, together with chemical
shifts, the cheapest and most widely used local restraints on a disordered region.  They are
read through the Karplus relation

  `J(θ) = A cos²θ + B cos θ + C`,

a *quadratic polynomial in `cos θ`*.  In an interconverting region the measured number is the
population average `⟨J⟩ = Σ_k w_k J(θ_k)` (fast exchange on the coupling timescale).  This file
determines exactly which functional of the torsion distribution that number is.

* `avgJ_eq` -- the measurement is `A⟨cos²θ⟩ + B⟨cos θ⟩ + C`: it sees the torsion distribution
  only through the first two moments of `cos θ`.
* `avgJ_congr_of_moments` -- hence *any* two ensembles matching those two moments give the same
  reading, for *every* Karplus parametrisation simultaneously.  Measuring more couplings of the
  same torsion (different `A,B,C`, different nuclei) adds no information whatsoever.
* `avgJ_eq_karplusOfCos_add_var` -- the exact bias of the single-angle interpretation:
  `⟨J⟩ - J(⟨cos θ⟩) = A · Var(cos θ)`.  Fitting one torsion angle to a measured coupling is
  wrong by `A` times the variance of `cos θ`, i.e. by exactly the quantity that makes the
  region disordered; `cosVar_nonneg` and `cosVar_pos_of_two` make the sign and the
  non-vacuity precise.
* `avgJ_reflect` -- a coupling cannot see the sign of a torsion: reflecting the whole ensemble
  `θ ↦ -θ` (α_R ↦ α_L) leaves every scalar coupling unchanged.
* `karplus_two_ensembles_agree`, `karplus_underdetermined_family`,
  `karplus_any_population_consistent` -- an explicit pair of five-basin ensembles with the same
  first two `cos` moments, hence the same coupling for every `A,B,C`, whose population of the
  `θ = π/2` basin differs by `1/2`; and the whole segment between them is consistent, so the
  measurement leaves that population entirely free in `[0, 1/2]`.
* `twoBasin_identifiable` -- the positive counterpart, and the exact reason the classical
  two-state analysis is legitimate: with only *two* basins and a coupling that distinguishes
  them, the populations are uniquely determined by one measured `J`.

Design conclusion: scalar couplings are two numbers per torsion.  They are a valid constraint
on a forward-modelled ensemble and a valid population read-out for a two-state torsion, but
they cannot be inverted into a torsion distribution, and a model must not be scored as if they
could.
-/
import Mathlib

set_option autoImplicit false

namespace Karplus

open Finset

/-- The Karplus polynomial as a function of `c = cos θ`: `A c² + B c + C`. -/
noncomputable def karplusOfCos (A B C c : ℝ) : ℝ := A * c ^ 2 + B * c + C

/-- The Karplus relation `J(θ) = A cos²θ + B cos θ + C`. -/
noncomputable def karplus (A B C θ : ℝ) : ℝ := karplusOfCos A B C (Real.cos θ)

variable {m : ℕ}

/-- The population-averaged coupling of an ensemble of torsion angles `θ` with weights `w`. -/
noncomputable def avgJ (A B C : ℝ) (w θ : Fin m → ℝ) : ℝ := ∑ k, w k * karplus A B C (θ k)

/-- First moment of `cos θ` over the ensemble. -/
noncomputable def cosMean (w θ : Fin m → ℝ) : ℝ := ∑ k, w k * Real.cos (θ k)

/-- Second moment of `cos θ` over the ensemble. -/
noncomputable def cosSq (w θ : Fin m → ℝ) : ℝ := ∑ k, w k * Real.cos (θ k) ^ 2

/-- Variance of `cos θ` over the ensemble. -/
noncomputable def cosVar (w θ : Fin m → ℝ) : ℝ := cosSq w θ - cosMean w θ ^ 2

/-- **A scalar coupling is two moments.**  The population-averaged Karplus coupling depends on
the torsion distribution only through `⟨cos θ⟩` and `⟨cos²θ⟩`. -/
theorem avgJ_eq (A B C : ℝ) {w θ : Fin m → ℝ} (hsum : ∑ k, w k = 1) :
    avgJ A B C w θ = A * cosSq w θ + B * cosMean w θ + C := by
  have hpt : ∀ k, w k * karplus A B C (θ k)
      = A * (w k * Real.cos (θ k) ^ 2) + B * (w k * Real.cos (θ k)) + C * w k := by
    intro k; simp only [karplus, karplusOfCos]; ring
  rw [avgJ, Finset.sum_congr rfl fun k _ => hpt k, Finset.sum_add_distrib,
    Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, hsum]
  simp [cosSq, cosMean]

/-- **No family of couplings on one torsion adds information.**  Two ensembles with the same
first two moments of `cos θ` produce the same measured coupling for every Karplus
parametrisation `A, B, C` at once. -/
theorem avgJ_congr_of_moments {m' : ℕ} {w θ : Fin m → ℝ} {v phi : Fin m' → ℝ}
    (hw : ∑ k, w k = 1) (hv : ∑ k, v k = 1)
    (h1 : cosMean w θ = cosMean v phi) (h2 : cosSq w θ = cosSq v phi) :
    ∀ A B C : ℝ, avgJ A B C w θ = avgJ A B C v phi := by
  intro A B C
  rw [avgJ_eq A B C hw, avgJ_eq A B C hv, h1, h2]

/-- The variance is the mean square deviation. -/
theorem cosVar_eq_sum {w θ : Fin m → ℝ} (hsum : ∑ k, w k = 1) :
    cosVar w θ = ∑ k, w k * (Real.cos (θ k) - cosMean w θ) ^ 2 := by
  have hpt : ∀ k, w k * (Real.cos (θ k) - cosMean w θ) ^ 2
      = w k * Real.cos (θ k) ^ 2 - 2 * cosMean w θ * (w k * Real.cos (θ k))
        + cosMean w θ ^ 2 * w k := by
    intro k; ring
  rw [Finset.sum_congr rfl fun k _ => hpt k, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, hsum]
  simp only [cosVar, cosSq, cosMean]
  ring

theorem cosVar_nonneg {w θ : Fin m → ℝ} (hw : ∀ k, 0 ≤ w k) (hsum : ∑ k, w k = 1) :
    0 ≤ cosVar w θ := by
  rw [cosVar_eq_sum hsum]
  exact Finset.sum_nonneg fun k _ => mul_nonneg (hw k) (sq_nonneg _)

/-- **Non-vacuity.**  Two conformers with positive weight and different `cos θ` give a strictly
positive variance, hence a strictly biased single-angle interpretation. -/
theorem cosVar_pos_of_two {w θ : Fin m → ℝ} (hw : ∀ k, 0 ≤ w k) (hsum : ∑ k, w k = 1)
    {i j : Fin m} (hi : 0 < w i) (hj : 0 < w j)
    (hcos : Real.cos (θ i) ≠ Real.cos (θ j)) : 0 < cosVar w θ := by
  rw [cosVar_eq_sum hsum]
  set mu := cosMean w θ with hmu
  have hne : Real.cos (θ i) ≠ mu ∨ Real.cos (θ j) ≠ mu := by
    by_contra hcon
    push_neg at hcon
    exact hcos (hcon.1.trans hcon.2.symm)
  have hnonneg : ∀ k ∈ Finset.univ, 0 ≤ w k * (Real.cos (θ k) - mu) ^ 2 :=
    fun k _ => mul_nonneg (hw k) (sq_nonneg _)
  rcases hne with h | h
  · refine lt_of_lt_of_le ?_ (Finset.single_le_sum hnonneg (Finset.mem_univ i))
    exact mul_pos hi (pow_two_pos_of_ne_zero (sub_ne_zero.mpr h))
  · refine lt_of_lt_of_le ?_ (Finset.single_le_sum hnonneg (Finset.mem_univ j))
    exact mul_pos hj (pow_two_pos_of_ne_zero (sub_ne_zero.mpr h))

/-- **The exact bias of the single-angle interpretation.**  The measured coupling exceeds the
coupling of the torsion whose cosine is the ensemble mean cosine by exactly `A · Var(cos θ)`. -/
theorem avgJ_eq_karplusOfCos_add_var (A B C : ℝ) {w θ : Fin m → ℝ} (hsum : ∑ k, w k = 1) :
    avgJ A B C w θ = karplusOfCos A B C (cosMean w θ) + A * cosVar w θ := by
  rw [avgJ_eq A B C hsum]
  simp only [karplusOfCos, cosVar]
  ring

/-- A coupling is blind to the sign of the torsion. -/
theorem karplus_neg_angle (A B C θ : ℝ) : karplus A B C (-θ) = karplus A B C θ := by
  simp [karplus, Real.cos_neg]

/-- **Couplings cannot distinguish α_R from α_L.**  Reflecting every torsion of the ensemble
leaves every scalar coupling unchanged. -/
theorem avgJ_reflect (A B C : ℝ) (w θ : Fin m → ℝ) :
    avgJ A B C w (fun k => -(θ k)) = avgJ A B C w θ := by
  simp [avgJ, karplus_neg_angle]

/-! ### An explicit pair of indistinguishable torsion ensembles -/

/-- Five torsion basins with cosines `1, √2/2, 0, -√2/2, -1`. -/
noncomputable def fiveAngles : Fin 5 → ℝ :=
  ![0, Real.pi / 4, Real.pi / 2, Real.pi - Real.pi / 4, Real.pi]

/-- A three-basin population: `1/4` at `cos = 1`, `1/2` at `cos = 0`, `1/4` at `cos = -1`. -/
noncomputable def popA : Fin 5 → ℝ := ![1 / 4, 0, 1 / 2, 0, 1 / 4]

/-- A two-basin population: `1/2` at `cos = ±√2/2`, and *nothing* in the `θ = π/2` basin. -/
noncomputable def popB : Fin 5 → ℝ := ![0, 1 / 2, 0, 1 / 2, 0]

lemma cos_fiveAngles :
    Real.cos (fiveAngles 0) = 1 ∧ Real.cos (fiveAngles 1) = Real.sqrt 2 / 2 ∧
      Real.cos (fiveAngles 2) = 0 ∧ Real.cos (fiveAngles 3) = -(Real.sqrt 2 / 2) ∧
      Real.cos (fiveAngles 4) = -1 := by
  refine ⟨by simp [fiveAngles], by simp [fiveAngles, Real.cos_pi_div_four], ?_, ?_, ?_⟩
  · simp [fiveAngles]
  · simp [fiveAngles, Real.cos_pi_sub, Real.cos_pi_div_four]
  · simp [fiveAngles]

lemma popA_nonneg (k : Fin 5) : 0 ≤ popA k := by
  fin_cases k <;> norm_num [popA]

lemma popB_nonneg (k : Fin 5) : 0 ≤ popB k := by
  fin_cases k <;> norm_num [popB]

lemma popA_sum : ∑ k, popA k = 1 := by
  simp [popA, Fin.sum_univ_five]; norm_num

lemma popB_sum : ∑ k, popB k = 1 := by
  simp [popB, Fin.sum_univ_five]; norm_num

lemma cosMean_popA : cosMean popA fiveAngles = 0 := by
  obtain ⟨h0, h1, h2, h3, h4⟩ := cos_fiveAngles
  simp [cosMean, Fin.sum_univ_five, popA, h0, h1, h2, h3, h4]

lemma cosMean_popB : cosMean popB fiveAngles = 0 := by
  obtain ⟨h0, h1, h2, h3, h4⟩ := cos_fiveAngles
  simp [cosMean, Fin.sum_univ_five, popB, h0, h1, h2, h3, h4]

lemma cosSq_popA : cosSq popA fiveAngles = 1 / 2 := by
  obtain ⟨h0, h1, h2, h3, h4⟩ := cos_fiveAngles
  simp [cosSq, Fin.sum_univ_five, popA, h0, h1, h2, h3, h4]
  norm_num

lemma cosSq_popB : cosSq popB fiveAngles = 1 / 2 := by
  obtain ⟨h0, h1, h2, h3, h4⟩ := cos_fiveAngles
  have hs : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  simp [cosSq, Fin.sum_univ_five, popB, h0, h1, h2, h3, h4]
  nlinarith [hs]

/-- **Two ensembles, every coupling identical.**  `popA` and `popB` are distinct torsion
distributions -- one of them has half its population in the `θ = π/2` basin, the other none --
yet they predict exactly the same value for *every* Karplus coupling. -/
theorem karplus_two_ensembles_agree (A B C : ℝ) :
    avgJ A B C popA fiveAngles = avgJ A B C popB fiveAngles :=
  avgJ_congr_of_moments popA_sum popB_sum
    (by rw [cosMean_popA, cosMean_popB]) (by rw [cosSq_popA, cosSq_popB]) A B C

/-- The mixture of the two indistinguishable ensembles. -/
noncomputable def popMix (t : ℝ) : Fin 5 → ℝ := fun k => (1 - t) * popA k + t * popB k

lemma popMix_nonneg {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (k : Fin 5) : 0 ≤ popMix t k :=
  add_nonneg (mul_nonneg (by linarith) (popA_nonneg k)) (mul_nonneg ht0 (popB_nonneg k))

lemma popMix_sum (t : ℝ) : ∑ k, popMix t k = 1 := by
  simp only [popMix, Finset.sum_add_distrib, ← Finset.mul_sum, popA_sum, popB_sum]
  ring

lemma cosMean_popMix (t : ℝ) : cosMean (popMix t) fiveAngles = 0 := by
  have h : ∀ k, popMix t k * Real.cos (fiveAngles k)
      = (1 - t) * (popA k * Real.cos (fiveAngles k)) + t * (popB k * Real.cos (fiveAngles k)) := by
    intro k; simp only [popMix]; ring
  simp only [cosMean, Finset.sum_congr rfl fun k _ => h k, Finset.sum_add_distrib,
    ← Finset.mul_sum]
  have := cosMean_popA
  have := cosMean_popB
  simp only [cosMean] at *
  simp [*]

lemma cosSq_popMix (t : ℝ) : cosSq (popMix t) fiveAngles = 1 / 2 := by
  have h : ∀ k, popMix t k * Real.cos (fiveAngles k) ^ 2
      = (1 - t) * (popA k * Real.cos (fiveAngles k) ^ 2)
        + t * (popB k * Real.cos (fiveAngles k) ^ 2) := by
    intro k; simp only [popMix]; ring
  simp only [cosSq, Finset.sum_congr rfl fun k _ => h k, Finset.sum_add_distrib,
    ← Finset.mul_sum]
  have hA := cosSq_popA
  have hB := cosSq_popB
  simp only [cosSq] at hA hB
  rw [hA, hB]
  ring

/-- **A one-parameter family of ensembles fits the data exactly.**  Every mixture of the two
ensembles reproduces every Karplus coupling of `popA`, while the population of the `θ = π/2`
basin sweeps the whole interval `[0, 1/2]`. -/
theorem karplus_underdetermined_family {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    (∀ k, 0 ≤ popMix t k) ∧ (∑ k, popMix t k = 1) ∧
      (∀ A B C : ℝ, avgJ A B C (popMix t) fiveAngles = avgJ A B C popA fiveAngles) ∧
      popMix t 2 = (1 - t) / 2 := by
  refine ⟨popMix_nonneg ht0 ht1, popMix_sum t, ?_, ?_⟩
  · exact avgJ_congr_of_moments (popMix_sum t) popA_sum
      (by rw [cosMean_popMix, cosMean_popA]) (by rw [cosSq_popMix, cosSq_popA])
  · simp [popMix, popA, popB]
    norm_num
    ring

/-- **The basin population is not identifiable.**  For every target population `u ∈ [0, 1/2]`
of the `θ = π/2` basin there is an admissible ensemble with that population which reproduces
every measured Karplus coupling. -/
theorem karplus_any_population_consistent {u : ℝ} (hu0 : 0 ≤ u) (hu1 : u ≤ 1 / 2) :
    ∃ p : Fin 5 → ℝ, (∀ k, 0 ≤ p k) ∧ (∑ k, p k = 1) ∧
      (∀ A B C : ℝ, avgJ A B C p fiveAngles = avgJ A B C popA fiveAngles) ∧ p 2 = u := by
  refine ⟨popMix (1 - 2 * u), ?_, ?_, ?_, ?_⟩
  · exact popMix_nonneg (by linarith) (by linarith)
  · exact popMix_sum _
  · exact avgJ_congr_of_moments (popMix_sum _) popA_sum
      (by rw [cosMean_popMix, cosMean_popA]) (by rw [cosSq_popMix, cosSq_popA])
  · simp [popMix, popA, popB]
    norm_num
    linarith

/-! ### The positive counterpart: two basins are identifiable -/

/-- **Where the classical analysis is valid.**  If the torsion has only two basins and the
coupling tells them apart, then a single measured `J` determines their populations uniquely. -/
theorem twoBasin_identifiable {A B C c1 c2 p p' : ℝ}
    (hne : karplusOfCos A B C c1 ≠ karplusOfCos A B C c2)
    (h : p * karplusOfCos A B C c1 + (1 - p) * karplusOfCos A B C c2
      = p' * karplusOfCos A B C c1 + (1 - p') * karplusOfCos A B C c2) : p = p' := by
  have hd : (p - p') * (karplusOfCos A B C c1 - karplusOfCos A B C c2) = 0 := by linarith
  rcases mul_eq_zero.mp hd with h1 | h2
  · linarith
  · exact absurd (sub_eq_zero.mp h2) hne

end Karplus
