/-
# Part XXXIV  Diffusion-limited association: what "fly-casting" would have to mean

A disordered region is often said to bind *faster* than a folded domain because its extended,
fluctuating conformations present a larger capture radius — the fly-casting hypothesis.  The
claim is kinetic, and in the diffusion-limited regime it is decidable by two textbook laws
already used elsewhere in this development:

  Smoluchowski:      `k = 4π D R_c`     (`smoluchowski`, `R_c` the capture radius),
  Stokes--Einstein:  `D = kT/(6πη R_h)` (`IDR.Hydro.stokesEinstein`, `R_h` the hydrodynamic
                                         radius).

* `smoluchowski_stokes` -- composing them gives the exact rate `k = (2kT/3η)·(R_c/R_h)`.  The
  two radii enter only through their *ratio*, and the prefactor contains no property of the
  chain at all.
* `rate_scale_invariant` -- consequently a chain that swells without changing the ratio binds at
  exactly the same rate, however large it gets.
* `flycasting_iff` -- a genuine speed-up happens *if and only if* the capture radius grows
  faster than the hydrodynamic radius; nothing else about the ensemble can produce one.
* `no_flycasting_from_scaling` -- and a polymer does not do that by being a polymer: if both
  radii follow the same scaling law `∝ N^ν` — as they do for ideal and for self-avoiding chains
  alike — the diffusion-limited rate is independent of chain length.  Fly-casting must come from
  a capture radius set by something other than the chain's own size (a long-range electrostatic
  steering, a large target, a reaction that can be initiated anywhere along the contour), and a
  model that predicts association rates must say which.
* `ensembleRate_eq`, `rate_not_determined_by_apparent_size` -- and the measured rate is an
  average of conformer rates, not the rate of the average conformer: for an explicit
  two-conformer ensemble, the single structure carrying its *measured* (Kirkwood /
  Stokes--Einstein) hydrodynamic radius and its mean capture radius binds strictly faster than
  the ensemble does, by `4/3`.  The experiment averages the two radii differently — harmonically
  and arithmetically — so a surrogate structure fitted to both has a rate error with a
  determined sign.
* `selection_scales_rate`, `selection_slows` -- conformational selection multiplies the rate by
  the competent population, the kinetic counterpart of the entropy penalty of Part XXX.
-/
import Mathlib
import RequestProject.Hydrodynamics

set_option autoImplicit false

namespace Assoc

open Finset

/-- The Smoluchowski rate constant for diffusion-limited capture at radius `Rc` with relative
diffusion coefficient `D`. -/
noncomputable def smoluchowski (D Rc : ℝ) : ℝ := 4 * Real.pi * D * Rc

/-- The diffusion-limited association rate of a chain of hydrodynamic radius `Rh` and capture
radius `Rc`. -/
noncomputable def rate (kT eta Rc Rh : ℝ) : ℝ :=
  smoluchowski (IDR.Hydro.stokesEinstein kT eta Rh) Rc

/-- **The diffusion-limited rate depends on the two radii only through their ratio.** -/
theorem smoluchowski_stokes {kT eta Rc Rh : ℝ} (heta : 0 < eta) (hRh : 0 < Rh) :
    rate kT eta Rc Rh = (2 * kT / (3 * eta)) * (Rc / Rh) := by
  unfold rate smoluchowski IDR.Hydro.stokesEinstein
  have hpi : Real.pi ≠ 0 := Real.pi_ne_zero
  field_simp
  ring

/-- **Swelling at fixed shape changes nothing.**  Scaling both radii by the same factor leaves
the diffusion-limited rate exactly unchanged. -/
theorem rate_scale_invariant {kT eta Rc Rh lam : ℝ} (heta : 0 < eta) (hRh : 0 < Rh)
    (hlam : 0 < lam) : rate kT eta (lam * Rc) (lam * Rh) = rate kT eta Rc Rh := by
  rw [smoluchowski_stokes heta (by positivity), smoluchowski_stokes heta hRh]
  rw [mul_div_mul_left _ _ (ne_of_gt hlam)]

/-- **A speed-up requires the capture radius to outgrow the hydrodynamic radius.** -/
theorem flycasting_iff {kT eta Rc Rh Rc' Rh' : ℝ} (hkT : 0 < kT) (heta : 0 < eta)
    (hRh : 0 < Rh) (hRh' : 0 < Rh') :
    rate kT eta Rc Rh < rate kT eta Rc' Rh' ↔ Rc / Rh < Rc' / Rh' := by
  rw [smoluchowski_stokes heta hRh, smoluchowski_stokes heta hRh']
  have hpre : 0 < 2 * kT / (3 * eta) := by positivity
  exact mul_lt_mul_iff_of_pos_left hpre

/-- **A polymer does not fly-cast by being a polymer.**  If the capture radius and the
hydrodynamic radius follow the same scaling law, the diffusion-limited rate does not depend on
chain length. -/
theorem no_flycasting_from_scaling {kT eta c h nu : ℝ} (heta : 0 < eta) (hc : 0 < c)
    (hh : 0 < h) (N1 N2 : ℝ) (hN1 : 0 < N1) (hN2 : 0 < N2) :
    rate kT eta (c * N1 ^ nu) (h * N1 ^ nu) = rate kT eta (c * N2 ^ nu) (h * N2 ^ nu) := by
  have hp1 : (0 : ℝ) < N1 ^ nu := Real.rpow_pos_of_pos hN1 nu
  have hp2 : (0 : ℝ) < N2 ^ nu := Real.rpow_pos_of_pos hN2 nu
  rw [smoluchowski_stokes heta (by positivity), smoluchowski_stokes heta (by positivity)]
  rw [show c * N1 ^ nu / (h * N1 ^ nu) = c / h by
        field_simp,
      show c * N2 ^ nu / (h * N2 ^ nu) = c / h by
        field_simp]

/-! ### What an ensemble binds like -/

variable {m : ℕ}

/-- The measured association rate of an ensemble in fast exchange: rates average. -/
noncomputable def ensembleRate (kT eta : ℝ) (w Rc Rh : Fin m → ℝ) : ℝ :=
  ∑ k, w k * rate kT eta (Rc k) (Rh k)

/-- The measured rate is the population-weighted mean of the conformer ratios. -/
theorem ensembleRate_eq {kT eta : ℝ} (heta : 0 < eta) {w Rc Rh : Fin m → ℝ}
    (hRh : ∀ k, 0 < Rh k) :
    ensembleRate kT eta w Rc Rh = (2 * kT / (3 * eta)) * ∑ k, w k * (Rc k / Rh k) := by
  unfold ensembleRate
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun k _ => by
    rw [smoluchowski_stokes heta (hRh k)]; ring)

/-- **The rate is not a function of the measured size.**  For an equally weighted two-conformer
ensemble with capture radii and hydrodynamic radii `1, 3`, the single structure carrying the
ensemble's *measured* (harmonic-mean, Stokes--Einstein) hydrodynamic radius `3/2` and its mean
capture radius `2` binds strictly *faster* than the ensemble itself — by a factor `4/3`.  The
two radii are averaged differently by the experiment, so a surrogate structure fitted to both
overestimates the diffusion-limited rate. -/
theorem rate_not_determined_by_apparent_size {kT eta : ℝ} (hkT : 0 < kT) (heta : 0 < eta) :
    IDR.Hydro.appRadius ![1/2, 1/2] ![(1 : ℝ), 3] = 3/2 ∧
      ensembleRate kT eta ![1/2, 1/2] ![(1 : ℝ), 3] ![(1 : ℝ), 3] < rate kT eta 2 (3/2) := by
  constructor
  · norm_num [IDR.Hydro.appRadius, Fin.sum_univ_two]
  · have hRh : ∀ k, 0 < (![(1 : ℝ), 3] : Fin 2 → ℝ) k := by
      intro k; fin_cases k <;> norm_num
    rw [ensembleRate_eq heta hRh, smoluchowski_stokes heta (by norm_num)]
    have hpre : 0 < 2 * kT / (3 * eta) := by positivity
    have hsum : ∑ k, (![1/2, 1/2] : Fin 2 → ℝ) k *
        ((![(1 : ℝ), 3] : Fin 2 → ℝ) k / (![(1 : ℝ), 3] : Fin 2 → ℝ) k) = 1 := by
      norm_num [Fin.sum_univ_two]
    rw [hsum]
    have hratio : (1 : ℝ) < 2 / (3/2) := by norm_num
    nlinarith [hpre]

/-- Conformational selection multiplies the diffusion-limited rate by the competent
population. -/
theorem selection_scales_rate (kT eta Rc Rh pS : ℝ) :
    pS * rate kT eta Rc Rh = rate kT eta (pS * Rc) Rh := by
  unfold rate smoluchowski
  ring

/-- And so it strictly slows association whenever some conformers are incompetent. -/
theorem selection_slows {kT eta Rc Rh pS : ℝ} (hkT : 0 < kT) (heta : 0 < eta) (hRc : 0 < Rc)
    (hRh : 0 < Rh) (h1 : pS < 1) :
    pS * rate kT eta Rc Rh < rate kT eta Rc Rh := by
  rw [smoluchowski_stokes heta hRh]
  have hpos : 0 < (2 * kT / (3 * eta)) * (Rc / Rh) := by positivity
  nlinarith

end Assoc
