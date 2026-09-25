/-
# Part XXXIV  Association kinetics: what "fly-casting" would have to mean

`RequestProject.Association` composes the two textbook laws that govern diffusion-limited
binding — Smoluchowski capture and the Stokes--Einstein relation already used in Part X.2 — and
reads off what a disordered region's extended conformations can and cannot buy it kinetically.

The rate is exactly `(2kT/3η)·(R_c/R_h)`: the capture radius and the hydrodynamic radius enter
only through their ratio, and the prefactor is a property of the solvent alone.  So swelling at
fixed shape changes nothing, a speed-up requires the capture radius to grow strictly faster than
the hydrodynamic radius, and a chain whose two radii obey the same scaling law binds at a rate
independent of its length.  The measured rate is a population average of conformer rates, and a
single structure fitted to the ensemble's measured hydrodynamic radius and mean capture radius
gets it wrong with a determined sign.

`IDR.association_laws` bundles the six statements.
-/
import Mathlib
import RequestProject.Association

set_option autoImplicit false

namespace IDR

/-- **The design laws of diffusion-limited association for a disordered region.**

1. *The rate is a ratio*: `k = (2kT/3η)·(R_c/R_h)` exactly.
2. *Swelling at fixed shape is free*: scaling both radii leaves the rate unchanged.
3. *Fly-casting has a criterion*: the rate increases iff the capture radius outgrows the
   hydrodynamic radius.
4. *Polymer scaling gives none of it*: with both radii `∝ N^ν` the rate is length-independent.
5. *Ensembles average rates*: the measured rate is the mean of the conformer ratios.
6. *And the measured size does not determine it*: a surrogate structure carrying the ensemble's
   apparent hydrodynamic radius and mean capture radius binds strictly faster than the
   ensemble. -/
theorem association_laws :
    -- 1  the diffusion-limited rate depends only on the ratio of the two radii
    (∀ kT eta Rc Rh : ℝ, 0 < eta → 0 < Rh →
        Assoc.rate kT eta Rc Rh = (2 * kT / (3 * eta)) * (Rc / Rh)) ∧
    -- 2  scale invariance
    (∀ kT eta Rc Rh lam : ℝ, 0 < eta → 0 < Rh → 0 < lam →
        Assoc.rate kT eta (lam * Rc) (lam * Rh) = Assoc.rate kT eta Rc Rh) ∧
    -- 3  the fly-casting criterion
    (∀ kT eta Rc Rh Rc' Rh' : ℝ, 0 < kT → 0 < eta → 0 < Rh → 0 < Rh' →
        (Assoc.rate kT eta Rc Rh < Assoc.rate kT eta Rc' Rh' ↔ Rc / Rh < Rc' / Rh')) ∧
    -- 4  common scaling gives no length dependence
    (∀ kT eta c h nu : ℝ, 0 < eta → 0 < c → 0 < h → ∀ N1 N2 : ℝ, 0 < N1 → 0 < N2 →
        Assoc.rate kT eta (c * N1 ^ nu) (h * N1 ^ nu)
          = Assoc.rate kT eta (c * N2 ^ nu) (h * N2 ^ nu)) ∧
    -- 5  an ensemble binds at the mean of its conformer rates
    (∀ (m : ℕ) (kT eta : ℝ), 0 < eta → ∀ w Rc Rh : Fin m → ℝ, (∀ k, 0 < Rh k) →
        Assoc.ensembleRate kT eta w Rc Rh
          = (2 * kT / (3 * eta)) * ∑ k, w k * (Rc k / Rh k)) ∧
    -- 6  and the measured size does not determine the rate
    (∀ kT eta : ℝ, 0 < kT → 0 < eta →
        IDR.Hydro.appRadius ![1/2, 1/2] ![(1 : ℝ), 3] = 3/2 ∧
          Assoc.ensembleRate kT eta ![1/2, 1/2] ![(1 : ℝ), 3] ![(1 : ℝ), 3]
            < Assoc.rate kT eta 2 (3/2)) := by
  refine ⟨fun kT eta Rc Rh heta hRh => Assoc.smoluchowski_stokes heta hRh,
    fun kT eta Rc Rh lam heta hRh hlam => Assoc.rate_scale_invariant heta hRh hlam,
    fun kT eta Rc Rh Rc' Rh' hkT heta hRh hRh' => Assoc.flycasting_iff hkT heta hRh hRh',
    fun kT eta c h nu heta hc hh N1 N2 hN1 hN2 =>
      Assoc.no_flycasting_from_scaling heta hc hh N1 N2 hN1 hN2,
    fun m kT eta heta w Rc Rh hRh => Assoc.ensembleRate_eq heta hRh,
    fun kT eta hkT heta => Assoc.rate_not_determined_by_apparent_size hkT heta⟩

end IDR
