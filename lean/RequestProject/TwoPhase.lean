/-
# Part VII.3  What a bulk experiment sees when the sample has two phases

`RequestProject.Condensate` and `RequestProject.Valence` are about the thermodynamics of
condensation.  This file is about its consequence for *modelling*, and it is the place where
Part VII rejoins the ensemble theory of Parts I--VI.

A condensing sample is not one ensemble.  It is a mass-weighted mixture of the dilute-phase
ensemble and the dense-phase ensemble, and every bulk observable -- an SAXS profile, a
chemical shift, an ensemble-averaged FRET efficiency -- reports the mixture and nothing
else.

* `Ens.prob_mix` and `Ens.expect_mix` (the latter from `RequestProject.ModelNature`) --
  **the bulk-measurement law**: every observable of a two-phase sample is the mass-weighted
  average of its values in the two phases.
* `Ens.ell1_mix_left`, `Ens.ell1_mix_right` -- the mixture sits on the segment between the
  phases, at `ℓ¹` distance `(1-t)·d` from the dilute phase and `t·d` from the dense one,
  where `d` is the distance between the phase ensembles.  So a single ensemble fitted to
  bulk data of a demixed sample is wrong about **both** phases unless one of them carries
  essentially all the material.
* `Ens.no_single_ensemble_fits_both_phases` -- and that is unavoidable for any single
  ensemble whatsoever, fitted by any means: it is off by at least `d/2` on one of the two
  phases.
* `bulk_cannot_detect_demixing` -- worse, the bulk data do not even reveal that there are
  two phases: a demixed sample and a homogeneous sample with the averaged populations are
  observationally identical.  Deciding between them requires spatially resolved or
  single-molecule data, exactly the latent-variable structure that
  `RequestProject.LatentRank` prices.
-/
import Mathlib
import RequestProject.Metric
import RequestProject.Condensate

namespace IDR

open Finset
open scoped Classical

namespace Ens

variable {X : Type*} [Fintype X]

omit [Fintype X] in
/-- **The bulk-measurement law, in populations.**  The population of a conformation in a
two-phase sample is the mass-weighted average of its populations in the two phases. -/
theorem prob_mix (E F : Ens X) {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (x : X) :
    (Ens.mix E F t ht0 ht1).prob x = t * E.prob x + (1 - t) * F.prob x := by
  simp [prob]

/-- The `ℓ¹` distance depends on an ensemble only through its populations. -/
theorem ell1_congr_left {M M' E : Ens X} (h : M.Same M') : ell1 M E = ell1 M' E := by
  have hp := (same_iff_prob_eq M M').1 h
  simp only [ell1, hp]

/-- **A two-phase sample is at distance `(1-t)·d` from its dilute phase.** -/
theorem ell1_mix_left (E F : Ens X) {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    ell1 (Ens.mix E F t ht0 ht1) E = (1 - t) * ell1 E F := by
  have h1t : (0:ℝ) ≤ 1 - t := by linarith
  simp only [ell1, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [prob_mix E F ht0 ht1 x, show t * E.prob x + (1 - t) * F.prob x - E.prob x
      = (1 - t) * (F.prob x - E.prob x) by ring, abs_mul, abs_of_nonneg h1t,
    abs_sub_comm]

/-- **A two-phase sample is at distance `t·d` from its dense phase.** -/
theorem ell1_mix_right (E F : Ens X) {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    ell1 (Ens.mix E F t ht0 ht1) F = t * ell1 E F := by
  simp only [ell1, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [prob_mix E F ht0 ht1 x, show t * E.prob x + (1 - t) * F.prob x - F.prob x
      = t * (E.prob x - F.prob x) by ring, abs_mul, abs_of_nonneg ht0]

/-- **No single ensemble describes both phases.**  Whatever ensemble is fitted -- to bulk
data or otherwise -- it is off by at least half the distance between the phases on one of
them.  A model of a condensing disordered protein must therefore be conditional on the
phase, i.e. on the local concentration. -/
theorem no_single_ensemble_fits_both_phases (M E F : Ens X) :
    ell1 E F / 2 ≤ max (ell1 M E) (ell1 M F) := by
  have htri : ell1 E F ≤ ell1 E M + ell1 M F := ell1_triangle E M F
  rw [ell1_comm E M] at htri
  rcases le_total (ell1 M E) (ell1 M F) with h | h
  · rw [max_eq_right h]; linarith
  · rw [max_eq_left h]; linarith

/-- **The error of a bulk fit, exactly.**  An ensemble matching the bulk data of a sample
that is a fraction `t` dilute phase and `1-t` dense phase is at distance `(1-t)·d` from the
first and `t·d` from the second: the two errors add up to the full separation `d` between
the phases, so accuracy on one is bought at the price of the other. -/
theorem bulk_fit_error {M E F : Ens X} {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1)
    (hM : M.Same (Ens.mix E F t ht0 ht1)) :
    ell1 M E = (1 - t) * ell1 E F ∧ ell1 M F = t * ell1 E F ∧
      ell1 M E + ell1 M F = ell1 E F := by
  have h1 : ell1 M E = (1 - t) * ell1 E F := by
    rw [ell1_congr_left hM, ell1_mix_left]
  have h2 : ell1 M F = t * ell1 E F := by
    rw [ell1_congr_left hM, ell1_mix_right]
  exact ⟨h1, h2, by rw [h1, h2]; ring⟩

end Ens

/-- **A bulk experiment cannot even detect that the sample has demixed.**  An equal mixture
of two distinct phase ensembles is observationally identical to a homogeneous sample whose
ensemble is their average, while neither phase is observationally equal to that homogeneous
ensemble.  Phase structure is a latent variable: it is invisible to any averaged
measurement, however precise. -/
theorem bulk_cannot_detect_demixing :
    ∃ E F G : Ens Bool,
      (Ens.mix E F (1/2) (by norm_num) (by norm_num)).Same
        (Ens.mix G G (1/2) (by norm_num) (by norm_num)) ∧ ¬ E.Same G ∧ ¬ F.Same G := by
  classical
  refine ⟨Ens.dirac true, Ens.dirac false,
    Ens.mix (Ens.dirac true) (Ens.dirac false) (1/2) (by norm_num) (by norm_num), ?_, ?_, ?_⟩
  · intro f
    rw [Ens.expect_mix, Ens.expect_mix, Ens.expect_mix]
    ring
  · intro h
    have := h (fun y => if y = true then (1:ℝ) else 0)
    rw [Ens.expect_mix] at this
    simp [Ens.dirac, Ens.expect] at this
  · intro h
    have := h (fun y => if y = true then (1:ℝ) else 0)
    rw [Ens.expect_mix] at this
    simp [Ens.dirac, Ens.expect] at this

end IDR
