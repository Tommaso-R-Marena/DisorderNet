/-
# Fit distributions, not averages: the distributional design law in one statement

`RequestProject.TransportVerdict` bundles the structural transport geometry.  This file
bundles what the three following files say about the *data* a disorder model must be fitted
and falsified against.

`distributional_design_law` states, for an arbitrary structural dissimilarity `c` on a
conformation space `X`:

1. **averages never suffice** -- for any finite panel of observables whatsoever and any error
   budget, two ensembles exist that reproduce every average in the panel exactly and are
   still further apart than the budget;
2. **one measured distribution suffices, exactly** -- along a measured coordinate the
   transport distance *equals* the `L¹` distance between the cumulative distributions, so a
   histogram determines the error rather than merely bounding it;
3. **and it dominates the average** -- the distributional discrepancy is never smaller than
   the discrepancy of the means, so nothing is lost by using it;
4. **finite instrument resolution costs one bin width, and no more** -- reading both
   ensembles through any map that displaces conformations by at most `d` perturbs the
   distance by at most `2d`;
5. **and the whole thing converts to ångströms** -- through an `L`-Lipschitz descriptor, a
   binned measured histogram certifies a structural transport error of at least
   `(cdfL1 - w) / L`.

Together: a model of an intrinsically disordered region must be scored against measured
*distributions* of structural descriptors, at a resolution fine compared with the error one
intends to detect; a panel of averages, however large, is provably incapable of bounding the
structural error at all.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Transport
import RequestProject.TransportGeometry
import RequestProject.TransportOneDim
import RequestProject.TransportResolution
import RequestProject.TransportPanelLimits

namespace IDR

open Finset
open scoped Classical

/-- **The distributional design law.**  Averages are provably insufficient at any panel
size; a single measured distribution along a coordinate gives the transport distance
exactly; that distributional certificate dominates the mean-based one; it degrades by at
most one bin width under finite resolution; and it converts into ångströms of structural
error through the Lipschitz constant of the descriptor. -/
theorem distributional_design_law {X : Type*} (c : X → X → ℝ) (hc : ∀ x y, 0 ≤ c x y) :
    -- 1. no finite panel of averages can bound the structural error
    (∀ (m : ℕ) (f : Fin m → ℝ → ℝ) (eps : ℝ), 0 < eps →
      ∃ E F : Ens ℝ, (∀ j, E.expect (f j) = F.expect (f j)) ∧
        eps ≤ transportCost lineCost E F) ∧
    -- 2. one measured distribution determines the distance exactly
    (∀ (n : ℕ) (t : ℕ → ℝ), Monotone t → ∀ (p q : ℕ → ℝ) (hp : ∀ i, 0 ≤ p i)
      (hq : ∀ i, 0 ≤ q i) (hps : ∑ i ∈ Finset.range n, p i = 1)
      (hqs : ∑ i ∈ Finset.range n, q i = 1),
      transportCost lineCost (gridEns n t p hp hps) (gridEns n t q hq hqs)
        = cdfL1 n t p q) ∧
    -- 3. and it is never weaker than the average
    (∀ (n : ℕ) (t : ℕ → ℝ), Monotone t → ∀ (p q : ℕ → ℝ), (∀ i, 0 ≤ p i) → (∀ i, 0 ≤ q i) →
      ∑ i ∈ Finset.range n, p i = 1 → ∑ i ∈ Finset.range n, q i = 1 →
      |∑ i ∈ Finset.range n, p i * t i - ∑ i ∈ Finset.range n, q i * t i| ≤ cdfL1 n t p q) ∧
    -- 4. finite resolution perturbs the distance by at most twice the displacement
    ((∀ x y, c x y = c y x) → (∀ x y z, c x z ≤ c x y + c y z) →
      ∀ (E F : Ens X) (r : X → X) (d : ℝ), (∀ x, c x (r x) ≤ d) →
        |transportCost c E F - transportCost c (E.map r) (F.map r)| ≤ 2 * d) ∧
    -- 5. a binned histogram of a Lipschitz descriptor certifies structural error
    (∀ (L : ℝ), 0 < L → ∀ (h : X → ℝ), (∀ x y, |h x - h y| ≤ L * c x y) →
      ∀ (E F : Ens X) (w : ℝ), 0 < w →
      ∀ (n : ℕ) (t : ℕ → ℝ), Monotone t → ∀ (p q : ℕ → ℝ) (hp : ∀ i, 0 ≤ p i)
        (hq : ∀ i, 0 ≤ q i) (hps : ∑ i ∈ Finset.range n, p i = 1)
        (hqs : ∑ i ∈ Finset.range n, q i = 1),
        ((E.map h).map (binR w)).Same (gridEns n t p hp hps) →
        ((F.map h).map (binR w)).Same (gridEns n t q hq hqs) →
        (cdfL1 n t p q - w) / L ≤ transportCost c E F) := by
  refine ⟨fun m f eps heps => no_finite_panel_of_means_certifies f heps,
    fun n t ht p q hp hq hps hqs => transportCost_line_eq_cdfL1 n ht hp hq hps hqs,
    fun n t ht p q hp hq hps hqs => mean_gap_le_cdfL1 n ht hp hq hps hqs,
    fun hsymm htri E F r d hr =>
      transportCost_binning_stability hc hsymm htri E F r hr,
    fun L hL h hlip E F w hw n t ht p q hp hq hps hqs hE hF =>
      resolution_certificate hc hL hlip E F hw n ht hp hq hps hqs hE hF⟩

end IDR
