/-
# Part LVII  Density maps without the harmonic, isotropic, single-conformer idealisation

Part XXXVIII models a smeared atom by one occupancy and one isotropic displacement parameter,
and states the restriction: anisotropic displacement and multiple discrete conformers are outside
it.  For a disordered region those are the whole phenomenon.  `RequestProject.Anisotropy` removes
both restrictions and prices them.

`IDR.anisotropy_laws` bundles five statements:

1. *No inflated `B` reproduces a split site.*  A two-conformer atom is not in the one-parameter
   single-site family at all -- for every width whatsoever the densities differ somewhere.
2. *The sign of the error at the place one looks.*  Under the standard "inflate `B` by the
   spread" prescription `u² = s² + d²` the fitted density is strictly too high at the midpoint.
3. *Resolved conformers show as a shape, not a width.*  Once `2s² ≤ d²` the true density dips at
   the midpoint, which no Gaussian does anywhere.
4. *The map determines the displacement tensor, the isotropic equivalent does not.*  Anisotropic
   densities that agree everywhere have equal principal widths; two tensors with the same
   isotropic `B` have densities that differ; and at fixed isotropic `B` the anisotropy ratio is
   unbounded.
5. *The occupancy trade-off survives in tensor form.*  Any peak height is reproduced at any
   occupancy by rescaling the tensor, with a density that is a different function.

The reading for a model of a disordered region: a deposited `(occupancy, B_iso)` pair is a
projection of the ensemble onto two numbers, many-to-one in two independent directions at once --
along conformer multiplicity and along the shape of the displacement.  A model that predicts an
ensemble must be compared with the map.
-/
import Mathlib
import RequestProject.Anisotropy

set_option autoImplicit false

namespace IDR

open IDR.Anisotropy

/-- **The anisotropy and multi-conformer laws for a density map.**

1. no single isotropic width reproduces a two-conformer density;
2. at the second-moment-matched width the fit is too high at the midpoint;
3. resolved conformers produce a dip that no Gaussian has;
4. the anisotropic map determines its tensor, the isotropic equivalent does not, and the
   anisotropy it discards is unbounded;
5. the occupancy/displacement trade-off in tensor form. -/
theorem anisotropy_laws :
    (∀ s d u : ℝ, 0 < s → d ≠ 0 → 0 < u → ∃ x : ℝ, twoSite s d x ≠ gauss u x) ∧
    (∀ s d : ℝ, 0 < s → d ≠ 0 →
        twoSite s d 0 < gauss (Real.sqrt (s ^ 2 + d ^ 2)) 0) ∧
    (∀ s d : ℝ, 0 < s → 2 * s ^ 2 ≤ d ^ 2 → twoSite s d 0 < twoSite s d d) ∧
    ((∀ u v : Fin 3 → ℝ, (∀ i, 0 < u i) → (∀ i, 0 < v i) →
        (∀ x : Fin 3 → ℝ, anisoDensity u x = anisoDensity v x) → u = v) ∧
      (∃ u v : Fin 3 → ℝ, (∀ i, 0 < u i) ∧ (∀ i, 0 < v i) ∧ equivB u = equivB v ∧
        ∃ x : Fin 3 → ℝ, anisoDensity u x ≠ anisoDensity v x) ∧
      (∀ B : ℝ, 0 < B → ∀ R : ℝ,
        ∃ u : Fin 3 → ℝ, (∀ i, 0 < u i) ∧ equivB u = B ∧ R < u 2 / u 0)) ∧
    (∀ q q' : ℝ, 0 < q → 0 < q' → q ≠ q' → ∀ u : Fin 3 → ℝ, (∀ i, 0 < u i) →
        ∃ v : Fin 3 → ℝ, (∀ i, 0 < v i) ∧
          q' * anisoDensity v (fun _ => 0) = q * anisoDensity u (fun _ => 0) ∧
          ∃ x, anisoDensity v x ≠ anisoDensity u x) := by
  refine ⟨fun s d u hs hd hu => twoSite_ne_gauss hs hd hu,
    fun s d hs hd => twoSite_center_lt_matched hs hd,
    fun s d hs hd => twoSite_bimodal hs hd,
    ⟨fun u v hu hv h => anisoDensity_inj hu hv h, equivB_not_determining,
      fun B hB R => aniso_ratio_unbounded hB R⟩,
    fun q q' hq hq' hne u hu => occupancy_anisotropy_degenerate hq hq' hne hu⟩

end IDR
