/-
# Part CXLVII  What a single mean transfer efficiency proves about a disordered ensemble

Part XLVII.1 showed that a mean FRET efficiency is not any mean of the distances present, and Part
CXLVI showed that a bounded observable with a known average certifies populations in both
directions.  Combining the two gives the statement a single-molecule FRET experiment on a
disordered region actually supports — with no polymer model, no Gaussian-chain assumption and no
fitted distribution whatsoever.

The trick is that the transfer efficiency needs no external bound: `E(r) = R₀⁶/(R₀⁶ + r⁶)` always
lies in `[0, 1]`, so the reverse-Markov certificate of Part CXLVI applies to it directly, and `E`
is strictly decreasing in the distance, so a threshold in efficiency *is* a threshold in distance.

* `eff_lt_eff_iff` — `E` is strictly decreasing on the non-negative distances: `E(s) < E(r) ↔ r < s`.
  This is what converts a population of efficiencies into a population of conformations.

* `compact_population_certificate` — **a measured mean efficiency certifies compact conformers.**
  For any distance `s > 0`, the fraction of the ensemble with `r < s` is at least

  `(Ē − E(s)) / (1 − E(s))`.

* `expanded_population_certificate` — **and simultaneously expanded ones.**  The fraction with
  `r > s` is at least

  `(E(s) − Ē) / E(s)`.

  Whichever side of `E(s)` the measured mean falls, one of the two is a positive number: a mean
  efficiency is never compatible with an ensemble concentrated on the other side of the
  corresponding distance.

* `compact_certificate_sharp` — the compact certificate is attained, by the two-state ensemble on
  distances `{0, s}`: no larger compact population follows from the mean efficiency alone.

* `fret_single_number_law` — the two certificates together, which is the honest content of a mean
  efficiency: an interval of admissible compact populations whose lower edge is exactly the number
  above.

* `worked_certificate` — a worked number for a standard dye pair: with `R₀ = 54 Å` a measured mean
  efficiency of `0.5` proves that at least 45% of the ensemble is more compact than 80 Å.

Design consequence: an ensemble model of a disordered region fitted to a mean FRET efficiency can
be *falsified* by these inequalities without any reference model — if its compact population falls
below the certified value it is inconsistent with the datum itself — and it can never be *confirmed*
past them, since the bound is attained.
-/
import Mathlib
import RequestProject.PopulationCertificate
import RequestProject.SaxsFret

set_option autoImplicit false

namespace IDR
namespace FretCertificate

open Finset IDR.PopulationCertificate IDR.SaxsFret

variable {N : ℕ}

/-- Population of conformations strictly more compact than the distance `s`. -/
noncomputable def popLt (w r : Fin N → ℝ) (s : ℝ) : ℝ := ∑ j, if r j < s then w j else 0

/-- Population of conformations strictly more expanded than the distance `s`. -/
noncomputable def popGtDist (w r : Fin N → ℝ) (s : ℝ) : ℝ := ∑ j, if s < r j then w j else 0

theorem eff_lt_one {R0 s : ℝ} (hR : 0 < R0) (hs : 0 < s) : eff R0 s < 1 := by
  rw [eff, div_lt_one (by positivity)]
  nlinarith [pow_pos hs 6]

/-- The transfer efficiency is strictly decreasing in the distance: a threshold in efficiency is a
threshold in distance. -/
theorem eff_lt_eff_iff {R0 r s : ℝ} (hR : 0 < R0) (hr : 0 ≤ r) (hs : 0 ≤ s) :
    eff R0 s < eff R0 r ↔ r < s := by
  have hden : ∀ t : ℝ, 0 < R0 ^ 6 + t ^ 6 := fun t => by positivity
  rw [eff, eff, div_lt_div_iff₀ (hden s) (hden r)]
  constructor
  · intro h
    have h6 : r ^ 6 < s ^ 6 := by nlinarith [pow_pos hR 6]
    by_contra hc
    push_neg at hc
    exact absurd (pow_le_pow_left₀ hs hc 6) (not_le.mpr h6)
  · intro h
    have h6 : r ^ 6 < s ^ 6 := by
      have := pow_lt_pow_left₀ h hr (n := 6) (by norm_num)
      simpa using this
    nlinarith [pow_pos hR 6]

/-- Reading the efficiency population as a distance population. -/
theorem popGt_eff_eq_popLt {w r : Fin N → ℝ} {R0 s : ℝ} (hR : 0 < R0) (hr : ∀ j, 0 ≤ r j)
    (hs : 0 ≤ s) :
    popGt w (fun j => eff R0 (r j)) (eff R0 s) = popLt w r s := by
  rw [popGt, popLt]
  refine Finset.sum_congr rfl fun j _ => ?_
  by_cases h : r j < s
  · rw [if_pos ((eff_lt_eff_iff hR (hr j) hs).mpr h), if_pos h]
  · rw [if_neg (fun hc => h ((eff_lt_eff_iff hR (hr j) hs).mp hc)), if_neg h]

/-- Reading the complementary efficiency population as a distance population. -/
theorem popGt_coeff_eq_popGtDist {w r : Fin N → ℝ} {R0 s : ℝ} (hR : 0 < R0) (hr : ∀ j, 0 ≤ r j)
    (hs : 0 ≤ s) :
    popGt w (fun j => 1 - eff R0 (r j)) (1 - eff R0 s) = popGtDist w r s := by
  rw [popGt, popGtDist]
  refine Finset.sum_congr rfl fun j _ => ?_
  by_cases h : s < r j
  · have : eff R0 (r j) < eff R0 s := (eff_lt_eff_iff hR hs (hr j)).mpr h
    rw [if_pos (by linarith), if_pos h]
  · have : eff R0 s ≤ eff R0 (r j) := by
      push_neg at h
      exact eff_antitone hR (hr j) h
    rw [if_neg (by linarith), if_neg h]

/-- **The compact-population certificate.**  A measured mean transfer efficiency `Ē` proves that at
least `(Ē − E(s))/(1 − E(s))` of the ensemble is more compact than any chosen distance `s > 0`. -/
theorem compact_population_certificate {w r : Fin N → ℝ} {R0 s : ℝ} (hR : 0 < R0) (hs : 0 < s)
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) (hr : ∀ j, 0 ≤ r j) :
    (wmean w (fun j => eff R0 (r j)) - eff R0 s) / (1 - eff R0 s) ≤ popLt w r s := by
  have hle : ∀ j, eff R0 (r j) ≤ 1 := fun j => eff_le_one hR
  have hbound := reverse_markov_lower (x := fun j => eff R0 (r j)) (B := 1) (a := eff R0 s)
    hw hsum hle (eff_lt_one hR hs)
  rwa [popGt_eff_eq_popLt hR hr hs.le] at hbound

/-- **The expanded-population certificate.**  The same measurement proves that at least
`(E(s) − Ē)/E(s)` of the ensemble is more expanded than `s`. -/
theorem expanded_population_certificate {w r : Fin N → ℝ} {R0 s : ℝ} (hR : 0 < R0) (hs : 0 < s)
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) (hr : ∀ j, 0 ≤ r j) :
    (eff R0 s - wmean w (fun j => eff R0 (r j))) / eff R0 s ≤ popGtDist w r s := by
  have hle : ∀ j, 1 - eff R0 (r j) ≤ 1 := fun j => by
    have := eff_pos (R0 := R0) (r := r j) hR
    linarith
  have hlt : 1 - eff R0 s < 1 := by
    have := eff_pos (R0 := R0) (r := s) hR
    linarith
  have hbound := reverse_markov_lower (x := fun j => 1 - eff R0 (r j)) (B := 1)
    (a := 1 - eff R0 s) hw hsum hle hlt
  rw [popGt_coeff_eq_popGtDist hR hr hs.le] at hbound
  have hmean : wmean w (fun j => 1 - eff R0 (r j)) = 1 - wmean w (fun j => eff R0 (r j)) := by
    rw [wmean, wmean]
    have h : ∀ j ∈ (Finset.univ : Finset (Fin N)), w j * (1 - eff R0 (r j))
        = w j - w j * eff R0 (r j) := fun j _ => by ring
    rw [Finset.sum_congr rfl h, Finset.sum_sub_distrib, hsum]
  rw [hmean] at hbound
  have heq : (1 - wmean w (fun j => eff R0 (r j)) - (1 - eff R0 s)) / (1 - (1 - eff R0 s))
      = (eff R0 s - wmean w (fun j => eff R0 (r j))) / eff R0 s := by
    ring_nf
  rwa [heq] at hbound

/-- The compact certificate is attained: for every `q ∈ [0, 1]` the two-state ensemble putting
weight `q` on a fully collapsed conformation and `1 − q` on one at exactly the distance `s` has
compact population `q`, and its mean efficiency makes the certificate an equality.  No larger
compact population follows from the mean efficiency alone. -/
theorem compact_certificate_sharp {R0 s q : ℝ} (hR : 0 < R0) (hs : 0 < s) (hq0 : 0 ≤ q)
    (hq1 : q ≤ 1) :
    ∃ w r : Fin 2 → ℝ, (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ (∀ j, 0 ≤ r j) ∧
      popLt w r s = q ∧
      (wmean w (fun j => eff R0 (r j)) - eff R0 s) / (1 - eff R0 s) = popLt w r s := by
  have he1 : eff R0 s < 1 := eff_lt_one hR hs
  have hR6 : R0 ^ 6 ≠ 0 := by positivity
  have he0 : eff R0 0 = 1 := by
    rw [eff]
    norm_num [hR6]
  refine ⟨![1 - q, q], ![s, 0], ?_, ?_, ?_, ?_, ?_⟩
  · intro j
    fin_cases j
    · simpa using hq1
    · simpa using hq0
  · rw [Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    ring
  · intro j
    fin_cases j <;> simp [hs.le]
  · rw [popLt, Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [if_neg (lt_irrefl s), if_pos hs]
    ring
  · have hpop : popLt ![1 - q, q] ![s, (0:ℝ)] s = q := by
      rw [popLt, Fin.sum_univ_two]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
      rw [if_neg (lt_irrefl s), if_pos hs]
      ring
    have hne : (1 : ℝ) - eff R0 s ≠ 0 := by linarith
    rw [hpop, wmean, Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [he0, show (1 - q) * eff R0 s + q * 1 - eff R0 s = q * (1 - eff R0 s) from by ring,
      mul_div_assoc, div_self hne, mul_one]

/-- **The single-number FRET law.**  One mean efficiency, two certificates: a guaranteed compact
population below any distance and a guaranteed expanded population above it, the first of which is
exactly attained.  Everything a mean transfer efficiency proves about a disordered ensemble, and
nothing it does not. -/
theorem fret_single_number_law {w r : Fin N → ℝ} {R0 s : ℝ} (hR : 0 < R0) (hs : 0 < s)
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) (hr : ∀ j, 0 ≤ r j) :
    (wmean w (fun j => eff R0 (r j)) - eff R0 s) / (1 - eff R0 s) ≤ popLt w r s ∧
      (eff R0 s - wmean w (fun j => eff R0 (r j))) / eff R0 s ≤ popGtDist w r s ∧
      ∀ q : ℝ, 0 ≤ q → q ≤ 1 → ∃ w' r' : Fin 2 → ℝ, (∀ j, 0 ≤ w' j) ∧ (∑ j, w' j = 1) ∧
        (∀ j, 0 ≤ r' j) ∧ popLt w' r' s = q ∧
        (wmean w' (fun j => eff R0 (r' j)) - eff R0 s) / (1 - eff R0 s) = popLt w' r' s :=
  ⟨compact_population_certificate hR hs hw hsum hr,
   expanded_population_certificate hR hs hw hsum hr,
   fun _ hq0 hq1 => compact_certificate_sharp hR hs hq0 hq1⟩

/-- A worked number for a standard dye pair.  With `R₀ = 54 Å`, a measured mean transfer efficiency
of `0.5` certifies that more than 45% of the ensemble is more compact than 80 Å. -/
theorem worked_certificate :
    (0.45 : ℝ) ≤ ((0.5 : ℝ) - eff 54 80) / (1 - eff 54 80) := by
  rw [eff]
  norm_num

end FretCertificate
end IDR
