/-
# Part XVI.2  The tether: why a disordered linker is part of the binding site

Most disordered regions act as *linkers*: they hold a motif near its partner, or two motifs
near each other.  The affinity that results is not a property of the motif; it is the
motif's intrinsic constant multiplied by the *effective concentration* the linker imposes,
and that concentration is set by the linker's conformational statistics.

The model here is the standard ideal chain, kept exactly solvable: `2m` steps, each step
independently `±b` along each of the three Cartesian axes (three independent
one-dimensional walks).  Contact between the two ends is the simultaneous return of all
three coordinates, so the contact probability is the cube of the one-dimensional return
probability `ret1 m = C(2m, m) / 4^m`.

* `ret1_succ` -- the exact recursion `a_{m+1} = a_m·(2m+1)/(2m+2)`, from the central
  binomial identity.
* `ret1_strictAnti`, `contactProb_strictAnti` -- longer linkers give strictly lower contact
  probability, at every length: there is no plateau.
* `ret1_sq_mul_le_one`, `ret1_le_inv_sqrt`, `contactProb_le` -- the quantitative law
  `a_m ≤ (3m+1)^{-1/2}`, hence `c_contact ≤ (3m+1)^{-3/2}`: the familiar `N^{-3/2}` decay of
  the effective concentration of an ideal tether, proved rather than assumed.
* `effConc_strictAnti` and `avidity_not_a_property_of_the_motif` -- the design consequence:
  two constructs with the *same* motif and the same intrinsic binding constant have
  strictly different effective affinities when their linkers differ in length, so a model
  that predicts binding must model the disordered linker, and a measured affinity may not
  be transferred between constructs.
-/
import Mathlib
import RequestProject.EnsembleCore

namespace IDR

open Finset
open scoped Classical

namespace Tether

/-- The return probability of a one-dimensional `2m`-step walk: `C(2m, m)/4^m`. -/
noncomputable def ret1 (m : ℕ) : ℝ := (Nat.centralBinom m : ℝ) / 4 ^ m

/-- The end-to-end contact probability of the three-dimensional ideal linker of `2m`
steps: all three coordinates must return. -/
noncomputable def contactProb (m : ℕ) : ℝ := ret1 m ^ 3

lemma ret1_pos (m : ℕ) : 0 < ret1 m := by
  refine div_pos ?_ (by positivity)
  exact_mod_cast Nat.centralBinom_pos m

@[simp] lemma ret1_zero : ret1 0 = 1 := by
  simp [ret1, Nat.centralBinom]

/-- The exact recursion for the one-dimensional return probability. -/
theorem ret1_succ (m : ℕ) : ret1 (m + 1) = ret1 m * ((2 * m + 1) / (2 * m + 2)) := by
  have hkey : ((m : ℝ) + 1) * (Nat.centralBinom (m + 1) : ℝ)
      = 2 * (2 * m + 1) * (Nat.centralBinom m : ℝ) := by
    exact_mod_cast congrArg (fun k : ℕ => (k : ℝ)) (Nat.succ_mul_centralBinom_succ m)
  have hm1 : ((m : ℝ) + 1) ≠ 0 := by positivity
  have h4 : (4 : ℝ) ^ m ≠ 0 := by positivity
  simp only [ret1, pow_succ]
  field_simp
  nlinarith [hkey]

lemma ret1_lt (m : ℕ) : ret1 (m + 1) < ret1 m := by
  have hpos := ret1_pos m
  have hfrac : ((2 * (m : ℝ) + 1) / (2 * m + 2)) < 1 := by
    rw [div_lt_one (by positivity)]
    linarith
  rw [ret1_succ]
  nlinarith

/-- Longer linkers contact strictly less often -- at every length. -/
theorem ret1_strictAnti : StrictAnti ret1 :=
  strictAnti_nat_of_succ_lt ret1_lt

theorem contactProb_pos (m : ℕ) : 0 < contactProb m := pow_pos (ret1_pos m) 3

theorem contactProb_strictAnti : StrictAnti contactProb := by
  intro a b hab
  have h := ret1_strictAnti hab
  have hpos := ret1_pos b
  simp only [contactProb]
  gcongr

/-! ## The quantitative law -/

/-- The sharp elementary bound `a_m² · (3m+1) ≤ 1`, proved by induction on the recursion. -/
theorem ret1_sq_mul_le_one (m : ℕ) : ret1 m ^ 2 * (3 * (m : ℝ) + 1) ≤ 1 := by
  induction m with
  | zero => simp
  | succ m ih =>
      have hpos := ret1_pos m
      have hrec := ret1_succ m
      have hden : (0:ℝ) < 2 * (m : ℝ) + 2 := by positivity
      have hstep : ((2 * (m : ℝ) + 1)) ^ 2 * (3 * ((m : ℝ) + 1) + 1)
          ≤ (3 * (m : ℝ) + 1) * (2 * (m : ℝ) + 2) ^ 2 := by
        nlinarith [sq_nonneg ((m : ℝ)), Nat.cast_nonneg (α := ℝ) m]
      have hcast : ((m : ℝ) + 1) = ((m + 1 : ℕ) : ℝ) := by push_cast; ring
      rw [← hcast, hrec]
      have hexp : (ret1 m * ((2 * (m : ℝ) + 1) / (2 * m + 2))) ^ 2 * (3 * ((m : ℝ) + 1) + 1)
          = (ret1 m ^ 2 * ((2 * (m : ℝ) + 1) ^ 2 * (3 * ((m : ℝ) + 1) + 1)))
            / (2 * (m : ℝ) + 2) ^ 2 := by
        field_simp
      rw [hexp, div_le_one (by positivity)]
      nlinarith [sq_nonneg (ret1 m), mul_le_mul_of_nonneg_left hstep (sq_nonneg (ret1 m))]

/-- Hence the `N^{-1/2}` law for one coordinate. -/
theorem ret1_le_inv_sqrt (m : ℕ) : ret1 m ≤ 1 / Real.sqrt (3 * (m : ℝ) + 1) := by
  have hpos : (0:ℝ) < 3 * (m : ℝ) + 1 := by positivity
  have hs : 0 < Real.sqrt (3 * (m : ℝ) + 1) := Real.sqrt_pos.mpr hpos
  rw [le_div_iff₀ hs]
  have hsq : (ret1 m * Real.sqrt (3 * (m : ℝ) + 1)) ^ 2 ≤ 1 := by
    rw [mul_pow, Real.sq_sqrt hpos.le]
    exact ret1_sq_mul_le_one m
  nlinarith [mul_pos (ret1_pos m) hs]

/-- And the `N^{-3/2}` law for the three-dimensional contact probability: the effective
concentration of an ideal tether falls off as the `-3/2` power of its length. -/
theorem contactProb_le (m : ℕ) :
    contactProb m ≤ 1 / Real.sqrt (3 * (m : ℝ) + 1) ^ 3 := by
  have h := ret1_le_inv_sqrt m
  have hpos : (0:ℝ) < 3 * (m : ℝ) + 1 := by positivity
  have hs : 0 < Real.sqrt (3 * (m : ℝ) + 1) := Real.sqrt_pos.mpr hpos
  have h1 : (0:ℝ) ≤ ret1 m := (ret1_pos m).le
  calc contactProb m = ret1 m ^ 3 := rfl
    _ ≤ (1 / Real.sqrt (3 * (m : ℝ) + 1)) ^ 3 := by
        exact pow_le_pow_left₀ h1 h 3
    _ = 1 / Real.sqrt (3 * (m : ℝ) + 1) ^ 3 := by
        rw [div_pow, one_pow]

/-- The matching lower bound `a_m ² · (4m+1) ≥ 1`, again by induction on the recursion. -/
theorem one_le_ret1_sq_mul (m : ℕ) : 1 ≤ ret1 m ^ 2 * (4 * (m : ℝ) + 1) := by
  induction m with
  | zero => simp
  | succ m ih =>
      have hpos := ret1_pos m
      have hrec := ret1_succ m
      have hstep : (4 * (m : ℝ) + 1) * (2 * (m : ℝ) + 2) ^ 2
          ≤ ((2 * (m : ℝ) + 1)) ^ 2 * (4 * ((m : ℝ) + 1) + 1) := by
        nlinarith [Nat.cast_nonneg (α := ℝ) m]
      have hcast : ((m : ℝ) + 1) = ((m + 1 : ℕ) : ℝ) := by push_cast; ring
      rw [← hcast, hrec]
      have hexp : (ret1 m * ((2 * (m : ℝ) + 1) / (2 * m + 2))) ^ 2 * (4 * ((m : ℝ) + 1) + 1)
          = (ret1 m ^ 2 * ((2 * (m : ℝ) + 1) ^ 2 * (4 * ((m : ℝ) + 1) + 1)))
            / (2 * (m : ℝ) + 2) ^ 2 := by
        field_simp
      rw [hexp, le_div_iff₀ (by positivity)]
      nlinarith [mul_le_mul_of_nonneg_left hstep (sq_nonneg (ret1 m))]

/-- Hence the two-sided law: the one-dimensional return probability is trapped between
`(4m+1)^{-1/2}` and `(3m+1)^{-1/2}`. -/
theorem inv_sqrt_le_ret1 (m : ℕ) : 1 / Real.sqrt (4 * (m : ℝ) + 1) ≤ ret1 m := by
  have hpos : (0:ℝ) < 4 * (m : ℝ) + 1 := by positivity
  have hs : 0 < Real.sqrt (4 * (m : ℝ) + 1) := Real.sqrt_pos.mpr hpos
  rw [div_le_iff₀ hs]
  have hsq : 1 ≤ (ret1 m * Real.sqrt (4 * (m : ℝ) + 1)) ^ 2 := by
    rw [mul_pow, Real.sq_sqrt hpos.le]
    exact one_le_ret1_sq_mul m
  nlinarith [mul_pos (ret1_pos m) hs]

/-- And the two-sided `N^{-3/2}` law for the contact probability. -/
theorem le_contactProb (m : ℕ) :
    1 / Real.sqrt (4 * (m : ℝ) + 1) ^ 3 ≤ contactProb m := by
  have hpos : (0:ℝ) < 4 * (m : ℝ) + 1 := by positivity
  have hs : 0 < Real.sqrt (4 * (m : ℝ) + 1) := Real.sqrt_pos.mpr hpos
  have h := inv_sqrt_le_ret1 m
  calc 1 / Real.sqrt (4 * (m : ℝ) + 1) ^ 3
      = (1 / Real.sqrt (4 * (m : ℝ) + 1)) ^ 3 := by rw [div_pow, one_pow]
    _ ≤ ret1 m ^ 3 := by
        exact pow_le_pow_left₀ (by positivity) h 3
    _ = contactProb m := rfl

/-! ## Effective concentration and avidity -/

/-- The effective concentration imposed by a linker of `2m` steps, given a reaction volume
`vol` for the contact: the contact probability per unit volume. -/
noncomputable def effConc (vol : ℝ) (m : ℕ) : ℝ := contactProb m / vol

/-- The apparent (intramolecular) binding constant: the intrinsic constant of the motif
multiplied by the effective concentration the linker provides. -/
noncomputable def apparentK (Kintr vol : ℝ) (m : ℕ) : ℝ := Kintr * effConc vol m

theorem effConc_strictAnti {vol : ℝ} (hvol : 0 < vol) : StrictAnti (effConc vol) := by
  intro a b hab
  exact (div_lt_div_iff_of_pos_right hvol).mpr (contactProb_strictAnti hab)

/-- **Avidity is not a property of the motif.**  Two constructs with the same motif -- the
same intrinsic binding constant -- and linkers of different lengths have strictly different
apparent affinities, the longer linker always the weaker.  A model that predicts binding by
a disordered region must therefore model the linker's conformational statistics; an
affinity measured in one construct does not transfer to another. -/
theorem avidity_not_a_property_of_the_motif {Kintr vol : ℝ} (hK : 0 < Kintr) (hvol : 0 < vol)
    {m₁ m₂ : ℕ} (h : m₁ < m₂) :
    apparentK Kintr vol m₂ < apparentK Kintr vol m₁ := by
  have hc := effConc_strictAnti hvol h
  simpa [apparentK] using (mul_lt_mul_of_pos_left hc hK)

end Tether

end IDR
