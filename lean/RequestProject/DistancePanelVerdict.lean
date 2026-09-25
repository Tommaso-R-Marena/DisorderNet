/-
# Capstone: what a distance panel can and cannot say about a disordered region

One statement, `distance_panel_design_law`, collecting the results of
`RequestProject.DistanceRealizability`, `RequestProject.DistanceCutEnsembles`,
`RequestProject.ContactBudget` and `RequestProject.ContactPacking`.  Read as a specification for
a model of an intrinsically disordered region and for the data it is fitted to:

1. **Consistency is testable without a model.**  Any panel of ensemble-mean distances obeys the
   triangle inequality and the contour bound; with error bars, a triangle defect above `3 e`
   or a contour excess above `e` refutes every ensemble at once, so the panel is checked before
   any model is proposed.
2. **Consistency is not enough, and single structures are the wrong currency.**  Ensembles
   realise the whole cut cone up to the contour scale, and there is an explicit two-state
   exchange whose mean panel is a perfectly good metric, passes every triangle test, and is the
   distance panel of no conformation whatsoever -- every single structure misses an entry by at
   least `1/5` of the panel scale.  A model must therefore denote an ensemble and predict
   averages, not fit one structure to averaged restraints.
3. **Populations are budgeted by excluded volume.**  Short measured mean distances force
   population (Markov), and no conformation can host more than `(2⌈2D/σ⌉+1)³` partners inside a
   contact radius `D` at hard-core separation `σ`.  A contact panel whose population demand
   `∑ₖ (1 - Mₖ/D)` exceeds that cap is unrealisable, and mutually exclusive populated contacts
   force a matching number of distinct conformations: the data, not convenience, fix the size of
   the ensemble.
-/
import Mathlib
import RequestProject.DistanceRealizability
import RequestProject.DistanceCutEnsembles
import RequestProject.ContactBudget
import RequestProject.ContactPacking
import RequestProject.DistanceThreePoint

namespace RequestProject.DistancePanelVerdict

open Finset RequestProject.DistanceRealizability RequestProject.DistanceCutEnsembles
open RequestProject.ContactBudget RequestProject.ContactPacking
open RequestProject.DistanceThreePoint

/-- **The distance-panel design law.**  Five clauses: the two model-free consistency tests, the
error-bar threshold that turns a triangle defect into a refutation, the demonstration that
consistency does not buy a structure (an explicit ensemble panel realised by no conformation,
with a quantitative gap), and the excluded-volume budget that caps how much contact population a
panel may demand. -/
theorem distance_panel_design_law :
    -- (i) the triangle inequality survives ensemble averaging
    (∀ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)),
        (∀ a, 0 ≤ w a) → ∀ i j k,
          meanDist w X i k ≤ meanDist w X i j + meanDist w X j k)
    ∧ -- (ii) so does the contour bound
      (∀ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)) (b : ℝ),
        (∀ a, 0 ≤ w a) → (∑ a, w a = 1) → (∀ a k, dist (X a k) (X a (k + 1)) ≤ b) →
          ∀ i j, i ≤ j → meanDist w X i j ≤ b * (j - i : ℕ))
    ∧ -- (iii) a triangle defect above three error bars refutes every ensemble
      (∀ (M : ℕ → ℕ → ℝ) (e : ℝ) (i j k : ℕ), M i j + M j k + 3 * e < M i k →
        ¬ ∃ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)),
            (∀ a, 0 ≤ w a) ∧ |M i k - meanDist w X i k| ≤ e ∧
            |M i j - meanDist w X i j| ≤ e ∧ |M j k - meanDist w X j k| ≤ e)
    ∧ -- (iv) consistency does not buy a structure, and the gap is quantitative
      ((∃ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)),
          (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧
          (∀ a k, dist (X a k) (X a (k + 1)) ≤ 2) ∧
          (∀ i < 4, ∀ j < 4, meanDist w X i j = fourCyclePanel i j) ∧
          (∀ i j k, meanDist w X i k ≤ meanDist w X i j + meanDist w X j k) ∧
          ¬ ∃ y : ℕ → EuclideanSpace ℝ (Fin 3),
              ∀ i < 4, ∀ j < 4, fourCyclePanel i j = dist (y i) (y j))
        ∧ ∀ y : ℕ → EuclideanSpace ℝ (Fin 3),
            ∃ i < 4, ∃ j < 4, (1 : ℝ) / 5 ≤ |fourCyclePanel i j - dist (y i) (y j)|)
    ∧ -- (v) excluded volume caps the contact population a panel may demand
      (∀ (r : ℕ) (i : ℕ) (q : Fin r → ℕ), Function.Injective q →
        ∀ (D sigma : ℝ) (M : Fin r → ℝ), 0 < D → 0 < sigma →
          ((2 * ⌈2 * D / sigma⌉₊ + 1) ^ 3 : ℕ) < ∑ k, (1 - M k / D) →
          ¬ ∃ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)),
              (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧
              (∀ (a : Fin m) (p p' : ℕ), p ≠ p' → sigma ≤ dist (X a p) (X a p')) ∧
              (∀ k, meanDist w X i (q k) ≤ M k))
    ∧ -- (vi) at three labelled sites the triangle test is exactly the feasibility condition
      (∀ p q r : ℝ,
        (∃ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)),
            (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧
            meanDist w X 0 1 = p ∧ meanDist w X 0 2 = q ∧ meanDist w X 1 2 = r)
          ↔ (r ≤ p + q ∧ q ≤ p + r ∧ p ≤ q + r)) := by
  refine ⟨fun m w X hw i j k => meanDist_triangle hw i j k,
    fun m w X b hw hsum hb i j hij => meanDist_le_chain hw hsum hb hij,
    fun M e i j k hdef => no_ensemble_of_triangle_defect hdef,
    ⟨ensemble_mean_beyond_single_structures, single_structure_fit_error_lower_bound⟩,
    fun r i q hq D sigma M hD hsig hover => hard_core_panel_falsified i q hq hD hsig hover,
    fun _ _ _ => three_point_feasible_iff⟩

end RequestProject.DistancePanelVerdict
