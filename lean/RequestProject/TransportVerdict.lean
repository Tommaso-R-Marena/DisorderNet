/-
# The design law for structural error in a disorder model, in one statement

This file bundles the transport development into a single theorem about the object a
modeller actually hands over: a finitely supported ensemble of Cartesian conformations of
an `m`-residue chain, to be compared with the truth in ångström.

`transport_design_law` states, for the RMSD geometry on `Struct m`:

1. the structural error `transportCost rmsd` is a metric on ensembles modulo experiment --
   nonnegative, symmetric, subadditive along chains, attained by an explicit optimal
   matching of predicted structures to true ones, and zero exactly when no experiment can
   tell the model from the truth;
2. it is jointly convex, so a per-state error budget is a budget for the assembled model;
3. it controls the radius of gyration one-for-one, and a measured `Rg` discrepancy is a
   lower bound on it -- the certificate an experiment returns;
4. it controls a single labelled-pair (FRET) distance only up to `√(2m)`, and that penalty
   is attained, so local observables are intrinsically weaker probes than global ones;
5. and the population-space `ℓ¹` error used by the capacity theory is the same quantity
   computed in the discrete geometry, i.e. the special case in which no two distinct
   structures are held to resemble each other.

Nothing here is new mathematics beyond the four preceding files; the point is that the five
statements are about one and the same quantity.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Transport
import RequestProject.TransportGeometry
import RequestProject.TransportDuality
import RequestProject.TransportRadiusGyration
import RequestProject.TransportProcessing
import RequestProject.TransportTotalVariation

namespace IDR

open Finset
open scoped Classical

/-- **The design law for structural error.**  One quantity -- the transport distance
against RMSD -- is simultaneously a metric on ensembles modulo experiment, convex under
assembly of sub-ensembles, an exact one-for-one budget for the radius of gyration with a
matching experimental certificate, a `√(2m)`-weakened budget for a single FRET pair with
that weakening attained, and (in the crudest geometry) the population-space `ℓ¹` error of
the capacity theory. -/
theorem transport_design_law (m : ℕ) (hm : 0 < m) :
    -- 1. a metric on ensembles modulo experiment, with an optimal matching
    (∀ E F : Ens (Struct m), 0 ≤ transportCost rmsd E F) ∧
    (∀ E F : Ens (Struct m), ∃ g, IsCoupling E F g ∧
      transportCost rmsd E F = planCost E F rmsd g) ∧
    (∀ E F : Ens (Struct m), transportCost rmsd E F = transportCost rmsd F E) ∧
    (∀ E F G : Ens (Struct m),
      transportCost rmsd E G ≤ transportCost rmsd E F + transportCost rmsd F G) ∧
    (∀ E F : Ens (Struct m), transportCost rmsd E F = 0 ↔ E.Same F) ∧
    -- 2. convex under assembly of sub-ensembles
    (∀ (E₁ E₂ F₁ F₂ : Ens (Struct m)) (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1),
      transportCost rmsd (Ens.mix E₁ E₂ t ht0 ht1) (Ens.mix F₁ F₂ t ht0 ht1)
        ≤ t * transportCost rmsd E₁ F₁ + (1 - t) * transportCost rmsd E₂ F₂) ∧
    -- 3. a one-for-one budget for the radius of gyration, and the experimental certificate
    (∀ E F : Ens (Struct m),
      |E.expect gyr - F.expect gyr| ≤ transportCost rmsd E F) ∧
    -- 4. only a `√(2m)` budget for one labelled pair, and that is sharp
    (∀ (k l : Fin m), k ≠ l → ∀ E F : Ens (Struct m),
      |E.expect (fun x => resDist x k l) - F.expect (fun x => resDist x k l)|
        ≤ Real.sqrt (2 * m) * transportCost rmsd E F) ∧
    (∀ (k l : Fin m), k ≠ l → ∀ t : ℝ, 0 < t → ∃ x y : Struct m,
      0 < rmsd x y ∧ transportCost rmsd (Ens.dirac x) (Ens.dirac y) = rmsd x y ∧
      |resDist x k l - resDist y k l| = Real.sqrt (2 * m) * rmsd x y) ∧
    -- 5. the `ℓ¹` capacity theory is the discrete-geometry case of the same quantity
    (∀ (Y : Type) (_ : Fintype Y) (E F : Ens Y),
      transportCost unitDist E F = Ens.ell1 E F / 2) := by
  refine ⟨fun E F => transportCost_nonneg rmsd_nonneg E F,
    fun E F => exists_optimal_coupling rmsd_nonneg E F,
    fun E F => transportCost_comm rmsd_nonneg rmsd_comm E F,
    fun E F G => transportCost_triangle rmsd_nonneg rmsd_triangle E F G,
    fun E F => transportCost_eq_zero_iff_same rmsd_nonneg rmsd_self
      (fun _ _ h => rmsd_eq_zero hm h) E F,
    fun E₁ E₂ F₁ F₂ t ht0 ht1 => transportCost_mix_le rmsd_nonneg ht0 ht1,
    fun E F => rg_gap_le_transportCost E F,
    fun k l hkl E F => fret_bound_from_transport hkl E F,
    fun k l hkl t ht => resDist_lipschitz_sharp hm hkl ht,
    fun Y _ E F => transportCost_unitDist_eq_half_ell1 E F⟩

end IDR
