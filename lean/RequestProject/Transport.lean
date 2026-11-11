/-
# The loss must see the geometry

`RequestProject.Metric` shows that the uniform observational error is exactly the
population-space `ℓ¹` distance, and uses it to derive capacity laws.  That is the right
metric for *counting* -- but it is blind to the geometry of conformation space: it charges
the same amount for putting the population on a conformation `0.1 Å` away as for putting it
on an unrelated fold.

This file adds the transport (earth-mover) cost of a structural dissimilarity `c` --
RMSD, a contact-map distance, whatever the modeller measures structures with -- and proves
the two statements a designer needs:

* `transportCost_le_of_coupling`, `transportCost_le_indep`, `transportCost_nonneg` -- basic
  calculus of the cost: any explicit coupling gives an upper bound, and the independent
  coupling is always available.
* `transportCost_dirac` -- between two single structures the cost is exactly their
  structural dissimilarity, so the transport cost *does* see the geometry.
* `ell1_geometry_blind` -- the punchline: two ensembles can be at the **maximal** `ℓ¹`
  distance `2` while their transport cost is as small as one likes.  A model that places
  the population on structures that are all but identical to the true ones scores zero
  credit under `ℓ¹`.

The design consequence is not that the `ℓ¹` results are wrong -- they are lower bounds on
error and remain valid -- but that the *training* objective of an ensemble model must be a
transport cost against a structural dissimilarity, while the *capacity* budget must be set
by the `ℓ¹`/entropy laws.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.ModelNature
import RequestProject.Metric
import RequestProject.Invariance

namespace IDR

open Finset
open scoped Classical

variable {X : Type*}

/-- A coupling (transport plan) between the components of two ensembles. -/
def IsCoupling (E F : Ens X) (gam : Fin E.card → Fin F.card → ℝ) : Prop :=
  (∀ i j, 0 ≤ gam i j) ∧ (∀ i, ∑ j, gam i j = E.w i) ∧ (∀ j, ∑ i, gam i j = F.w j)

/-- The cost of a transport plan against a structural dissimilarity `c`. -/
noncomputable def planCost (E F : Ens X) (c : X → X → ℝ)
    (gam : Fin E.card → Fin F.card → ℝ) : ℝ :=
  ∑ i, ∑ j, gam i j * c (E.pt i) (F.pt j)

/-- The transport (earth-mover) cost between two ensembles: the cheapest way of moving the
predicted population onto the true one, charged by the structural dissimilarity `c`. -/
noncomputable def transportCost (c : X → X → ℝ) (E F : Ens X) : ℝ :=
  sInf {r : ℝ | ∃ gam : Fin E.card → Fin F.card → ℝ, IsCoupling E F gam ∧ r = planCost E F c gam}

/-- The independent (product) plan is always a coupling. -/
lemma isCoupling_indep (E F : Ens X) :
    IsCoupling E F (fun i j => E.w i * F.w j) := by
  refine ⟨fun i j => mul_nonneg (E.w_nonneg i) (F.w_nonneg j), fun i => ?_, fun j => ?_⟩
  · rw [← Finset.mul_sum, F.w_sum, mul_one]
  · rw [← Finset.sum_mul, E.w_sum, one_mul]

lemma transportCost_set_nonempty (E F : Ens X) (c : X → X → ℝ) :
    {r : ℝ | ∃ gam : Fin E.card → Fin F.card → ℝ,
      IsCoupling E F gam ∧ r = planCost E F c gam}.Nonempty :=
  ⟨planCost E F c (fun i j => E.w i * F.w j), _, isCoupling_indep E F, rfl⟩

lemma transportCost_set_bddBelow {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) (E F : Ens X) :
    BddBelow {r : ℝ | ∃ gam : Fin E.card → Fin F.card → ℝ,
      IsCoupling E F gam ∧ r = planCost E F c gam} := by
  refine ⟨0, ?_⟩
  rintro r ⟨gam, ⟨hnn, -, -⟩, rfl⟩
  exact Finset.sum_nonneg fun i _ =>
    Finset.sum_nonneg fun j _ => mul_nonneg (hnn i j) (hc _ _)

/-- The transport cost of a nonnegative dissimilarity is nonnegative. -/
theorem transportCost_nonneg {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) (E F : Ens X) :
    0 ≤ transportCost c E F :=
  le_csInf (transportCost_set_nonempty E F c) (by
    rintro r ⟨gam, ⟨hnn, -, -⟩, rfl⟩
    exact Finset.sum_nonneg fun i _ =>
      Finset.sum_nonneg fun j _ => mul_nonneg (hnn i j) (hc _ _))

/-- Every explicit transport plan bounds the transport cost from above: exhibiting a way of
matching predicted structures to true ones certifies the model. -/
theorem transportCost_le_of_coupling {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) {E F : Ens X}
    {gam : Fin E.card → Fin F.card → ℝ} (hgam : IsCoupling E F gam) :
    transportCost c E F ≤ planCost E F c gam :=
  csInf_le (transportCost_set_bddBelow hc E F) ⟨gam, hgam, rfl⟩

/-- In particular the independent plan gives a computable upper bound. -/
theorem transportCost_le_indep {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) (E F : Ens X) :
    transportCost c E F ≤ ∑ i, ∑ j, E.w i * F.w j * c (E.pt i) (F.pt j) :=
  transportCost_le_of_coupling hc (isCoupling_indep E F)

/-- **The transport cost sees the geometry.**  Between two single structures it is exactly
their structural dissimilarity -- unlike the `ℓ¹` distance, which is `2` whenever the two
structures differ at all. -/
theorem transportCost_dirac (c : X → X → ℝ) (a b : X) :
    transportCost c (Ens.dirac a) (Ens.dirac b) = c a b := by
  have hset : {r : ℝ | ∃ gam : Fin (Ens.dirac a).card → Fin (Ens.dirac b).card → ℝ,
      IsCoupling (Ens.dirac a) (Ens.dirac b) gam ∧ r = planCost (Ens.dirac a) (Ens.dirac b) c gam}
      = {c a b} := by
    ext r
    simp only [Set.mem_setOf_eq, Set.mem_singleton_iff]
    constructor
    · rintro ⟨gam, ⟨-, hrow, -⟩, rfl⟩
      have htot : ∑ i, ∑ j, gam i j = 1 := by
        rw [Finset.sum_congr rfl (fun i _ => hrow i)]
        exact (Ens.dirac a).w_sum
      have hpt : ∀ i j, gam i j * c ((Ens.dirac a).pt i) ((Ens.dirac b).pt j)
          = gam i j * c a b := fun i j => rfl
      calc planCost (Ens.dirac a) (Ens.dirac b) c gam
          = ∑ i, ∑ j, gam i j * c a b := by
            refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => hpt i j
        _ = (∑ i, ∑ j, gam i j) * c a b := by
            rw [Finset.sum_mul]
            exact Finset.sum_congr rfl fun i _ => (Finset.sum_mul _ _ _).symm
        _ = c a b := by rw [htot, one_mul]
    · rintro rfl
      refine ⟨fun _ _ => 1, ⟨fun _ _ => zero_le_one, fun i => ?_, fun j => ?_⟩, ?_⟩
      · simp [Ens.dirac]
      · simp [Ens.dirac]
      · simp [planCost, Ens.dirac]
  rw [transportCost, hset, csInf_singleton]

/-! ## `ℓ¹` is blind to structural similarity -/

/-- A two-point conformation space with a tunable structural dissimilarity: the two
conformations differ, but by an amount `eps` that we may take as small as we like. -/
noncomputable def tinyDist (eps : ℝ) : Fin 2 → Fin 2 → ℝ :=
  fun x y => if x = y then 0 else eps

/-- **The `ℓ¹` loss is geometry-blind.**  Two ensembles can sit at the maximal `ℓ¹`
distance `2` -- the worst score attainable -- while the transport cost of turning one into
the other is an arbitrarily small `eps`.  A model whose predicted structures are all
essentially correct but never exactly the true ones is maximally wrong in `ℓ¹`.  The
capacity theorems of `RequestProject.Metric` are therefore conservative lower bounds on
error, and the objective a disorder model is *trained* on must be a transport cost against
a structural dissimilarity, not a population-space `ℓ¹`. -/
theorem ell1_geometry_blind (eps : ℝ) :
    ∃ E F : Ens (Fin 2),
      Ens.ell1 E F = 2 ∧ transportCost (tinyDist eps) E F = eps := by
  refine ⟨Ens.dirac 0, Ens.dirac 1, ?_, ?_⟩
  · exact ell1_dirac_dirac (by decide)
  · rw [transportCost_dirac]
    simp [tinyDist]

end IDR
