/-
# The transport distance between conformational ensembles is a genuine metric

`RequestProject.Transport` introduced the transport (earth-mover) cost of a structural
dissimilarity `c` between two ensembles and showed that, unlike the `ℓ¹` distance, it sees
the geometry of conformation space.  What it did *not* establish is that this cost behaves
like a distance -- and without that, "the model is within `1.5 Å` of the truth" is not a
statement one may chain, compare, or use to combine errors.

This file supplies the missing geometry, for an arbitrary conformation space `X` and an
arbitrary structural metric `c` (RMSD after alignment, a contact-map distance, ...):

* `exists_optimal_coupling` -- the infimum defining the cost is **attained**: there is a
  best way of matching the model's structures to the true ones.  (The set of transport
  plans is a compact polytope and the cost is continuous on it.)
* `transportCost_self`, `transportCost_comm` -- vanishing on the diagonal and symmetry.
* `glue` and `transportCost_triangle` -- the **gluing lemma**: two transport plans compose
  into a plan for the composite move, so structural errors add along a chain of ensembles.
  This is what makes "distance" language legitimate.
* `transportCost_eq_zero_iff_same` -- the cost vanishes exactly on observationally
  identical ensembles, so it is a metric on ensembles-modulo-experiment, not merely a
  pseudometric.
* `transportCost_pseudoMetric` packages the three metric axioms.

Everything is proved for finitely supported ensembles with no assumption on `X`.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Transport

namespace IDR

open Finset
open scoped Classical

variable {X : Type*}

/-! ## Elementary structure of transport plans -/

lemma IsCoupling.nonneg {E F : Ens X} {g : Fin E.card → Fin F.card → ℝ}
    (h : IsCoupling E F g) (i : Fin E.card) (j : Fin F.card) : 0 ≤ g i j := h.1 i j

lemma IsCoupling.row {E F : Ens X} {g : Fin E.card → Fin F.card → ℝ}
    (h : IsCoupling E F g) (i : Fin E.card) : ∑ j, g i j = E.w i := h.2.1 i

lemma IsCoupling.col {E F : Ens X} {g : Fin E.card → Fin F.card → ℝ}
    (h : IsCoupling E F g) (j : Fin F.card) : ∑ i, g i j = F.w j := h.2.2 j

lemma IsCoupling.le_row {E F : Ens X} {g : Fin E.card → Fin F.card → ℝ}
    (h : IsCoupling E F g) (i : Fin E.card) (j : Fin F.card) : g i j ≤ E.w i := by
  have := Finset.single_le_sum (f := fun j => g i j) (fun j _ => h.nonneg i j) (mem_univ j)
  rwa [h.row i] at this

lemma IsCoupling.le_col {E F : Ens X} {g : Fin E.card → Fin F.card → ℝ}
    (h : IsCoupling E F g) (i : Fin E.card) (j : Fin F.card) : g i j ≤ F.w j := by
  have := Finset.single_le_sum (f := fun i => g i j) (fun i _ => h.nonneg i j) (mem_univ i)
  rwa [h.col j] at this

/-- A transport plan cannot move mass out of an unpopulated conformation. -/
lemma IsCoupling.eq_zero_of_row_zero {E F : Ens X} {g : Fin E.card → Fin F.card → ℝ}
    (h : IsCoupling E F g) {i : Fin E.card} (hw : E.w i = 0) (j : Fin F.card) : g i j = 0 :=
  le_antisymm (hw ▸ h.le_row i j) (h.nonneg i j)

/-- A transport plan cannot move mass into an unpopulated conformation. -/
lemma IsCoupling.eq_zero_of_col_zero {E F : Ens X} {g : Fin E.card → Fin F.card → ℝ}
    (h : IsCoupling E F g) (i : Fin E.card) {j : Fin F.card} (hw : F.w j = 0) : g i j = 0 :=
  le_antisymm (hw ▸ h.le_col i j) (h.nonneg i j)

lemma planCost_nonneg {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) {E F : Ens X}
    {g : Fin E.card → Fin F.card → ℝ} (hg : IsCoupling E F g) : 0 ≤ planCost E F c g :=
  Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => mul_nonneg (hg.nonneg i j) (hc _ _)

/-! ## The optimum is attained -/

lemma isClosed_couplingSet (E F : Ens X) :
    IsClosed {g : Fin E.card → Fin F.card → ℝ | IsCoupling E F g} := by
  have hcoord : ∀ (i : Fin E.card) (j : Fin F.card),
      Continuous fun g : Fin E.card → Fin F.card → ℝ => g i j := fun i j => by fun_prop
  have h1 : IsClosed {g : Fin E.card → Fin F.card → ℝ | ∀ i j, 0 ≤ g i j} := by
    simp only [Set.setOf_forall]
    exact isClosed_iInter fun i => isClosed_iInter fun j => isClosed_le continuous_const (hcoord i j)
  have h2 : IsClosed {g : Fin E.card → Fin F.card → ℝ | ∀ i, ∑ j, g i j = E.w i} := by
    simp only [Set.setOf_forall]
    exact isClosed_iInter fun i =>
      isClosed_eq (continuous_finset_sum _ fun j _ => hcoord i j) continuous_const
  have h3 : IsClosed {g : Fin E.card → Fin F.card → ℝ | ∀ j, ∑ i, g i j = F.w j} := by
    simp only [Set.setOf_forall]
    exact isClosed_iInter fun j =>
      isClosed_eq (continuous_finset_sum _ fun i _ => hcoord i j) continuous_const
  have : {g : Fin E.card → Fin F.card → ℝ | IsCoupling E F g}
      = {g | ∀ i j, 0 ≤ g i j} ∩ ({g | ∀ i, ∑ j, g i j = E.w i} ∩ {g | ∀ j, ∑ i, g i j = F.w j}) := by
    ext g; simp [IsCoupling]
  rw [this]
  exact h1.inter (h2.inter h3)

lemma isCompact_couplingSet (E F : Ens X) :
    IsCompact {g : Fin E.card → Fin F.card → ℝ | IsCoupling E F g} := by
  refine IsCompact.of_isClosed_subset (isCompact_Icc (a := fun _ _ => (0 : ℝ))
    (b := fun _ _ => (1 : ℝ))) (isClosed_couplingSet E F) ?_
  rintro g hg
  have hg' : IsCoupling E F g := hg
  refine ⟨fun i => ?_, fun i => ?_⟩ <;> intro j
  · exact hg'.nonneg i j
  · have hle : E.w i ≤ 1 := by
      have := Finset.single_le_sum (f := fun i' => E.w i') (fun i' _ => E.w_nonneg i') (mem_univ i)
      rwa [E.w_sum] at this
    exact (hg'.le_row i j).trans hle

lemma continuous_planCost (E F : Ens X) (c : X → X → ℝ) :
    Continuous fun g : Fin E.card → Fin F.card → ℝ => planCost E F c g := by
  refine continuous_finset_sum _ fun i _ => continuous_finset_sum _ fun j _ => ?_
  have hij : Continuous fun g : Fin E.card → Fin F.card → ℝ => g i j := by fun_prop
  exact hij.mul continuous_const

/-- **An optimal transport plan exists.**  The infimum defining the transport cost is
attained: there really is a cheapest way of matching the model's structures to the true
ones, so the cost is a minimum, not just a greatest lower bound. -/
theorem exists_optimal_coupling {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) (E F : Ens X) :
    ∃ g : Fin E.card → Fin F.card → ℝ,
      IsCoupling E F g ∧ transportCost c E F = planCost E F c g := by
  obtain ⟨g, hgmem, hmin⟩ := (isCompact_couplingSet E F).exists_isMinOn
    ⟨_, isCoupling_indep E F⟩ (continuous_planCost E F c).continuousOn
  refine ⟨g, hgmem, le_antisymm (transportCost_le_of_coupling hc hgmem) ?_⟩
  refine le_csInf (transportCost_set_nonempty E F c) ?_
  rintro r ⟨g', hg', rfl⟩
  exact hmin (Set.mem_setOf.2 hg')

/-! ## The metric axioms -/

/-- The identity plan: leave every conformation where it is. -/
lemma isCoupling_id (E : Ens X) : IsCoupling E E (fun i j => if i = j then E.w i else 0) := by
  refine ⟨fun i j => ?_, fun i => ?_, fun j => ?_⟩
  · by_cases h : i = j <;> simp [h, E.w_nonneg]
  · simp
  · simp

/-- A structural dissimilarity that vanishes on identical structures gives an ensemble zero
transport cost to itself. -/
theorem transportCost_self {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) (hcd : ∀ x, c x x = 0)
    (E : Ens X) : transportCost c E E = 0 := by
  refine le_antisymm ?_ (transportCost_nonneg hc E E)
  have h := transportCost_le_of_coupling hc (isCoupling_id E)
  refine h.trans_eq ?_
  simp only [planCost]
  refine Finset.sum_eq_zero fun i _ => Finset.sum_eq_zero fun j _ => ?_
  by_cases hij : i = j
  · subst hij; simp [hcd]
  · simp [hij]

/-- Transport cost is symmetric whenever the structural dissimilarity is. -/
theorem transportCost_comm {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) (hsymm : ∀ x y, c x y = c y x)
    (E F : Ens X) : transportCost c E F = transportCost c F E := by
  have key : ∀ A B : Ens X, transportCost c A B ≤ transportCost c B A := by
    intro A B
    obtain ⟨g, hg, hgc⟩ := exists_optimal_coupling hc B A
    have htr : IsCoupling A B (fun i j => g j i) :=
      ⟨fun i j => hg.nonneg j i, fun i => hg.col i, fun j => hg.row j⟩
    refine (transportCost_le_of_coupling hc htr).trans_eq ?_
    rw [hgc]
    simp only [planCost]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => by rw [hsymm]
  exact le_antisymm (key E F) (key F E)

/-! ## The gluing lemma -/

/-- The composition of two transport plans: mass is routed from `E` to `G` through the
intermediate ensemble `F`, splitting the mass arriving at each intermediate conformation in
proportion to where it must go next. -/
noncomputable def glue (E F G : Ens X) (g₁ : Fin E.card → Fin F.card → ℝ)
    (g₂ : Fin F.card → Fin G.card → ℝ) : Fin E.card → Fin G.card → ℝ :=
  fun i l => ∑ j, g₁ i j * g₂ j l / F.w j

lemma glue_isCoupling {E F G : Ens X} {g₁ : Fin E.card → Fin F.card → ℝ}
    {g₂ : Fin F.card → Fin G.card → ℝ} (h₁ : IsCoupling E F g₁) (h₂ : IsCoupling F G g₂) :
    IsCoupling E G (glue E F G g₁ g₂) := by
  refine ⟨fun i l => ?_, fun i => ?_, fun l => ?_⟩
  · exact Finset.sum_nonneg fun j _ =>
      div_nonneg (mul_nonneg (h₁.nonneg i j) (h₂.nonneg j l)) (F.w_nonneg j)
  · -- row marginals
    have : ∑ l, glue E F G g₁ g₂ i l = ∑ j, g₁ i j := by
      simp only [glue]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun j _ => ?_
      rcases eq_or_lt_of_le (F.w_nonneg j) with hw | hw
      · have h0 : g₁ i j = 0 := h₁.eq_zero_of_col_zero i hw.symm
        simp [h0]
      · have : ∑ l, g₁ i j * g₂ j l / F.w j = (g₁ i j / F.w j) * ∑ l, g₂ j l := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun l _ => by field_simp
        rw [this, h₂.row j]
        field_simp
    rw [this, h₁.row i]
  · -- column marginals
    have : ∑ i, glue E F G g₁ g₂ i l = ∑ j, g₂ j l := by
      simp only [glue]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun j _ => ?_
      rcases eq_or_lt_of_le (F.w_nonneg j) with hw | hw
      · have h0 : g₂ j l = 0 := h₂.eq_zero_of_row_zero hw.symm l
        simp [h0]
      · have : ∑ i, g₁ i j * g₂ j l / F.w j = (g₂ j l / F.w j) * ∑ i, g₁ i j := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun i _ => by field_simp
        rw [this, h₁.col j]
        field_simp
    rw [this, h₂.col l]

lemma planCost_glue_le {c : X → X → ℝ}
    (htri : ∀ x y z, c x z ≤ c x y + c y z) {E F G : Ens X}
    {g₁ : Fin E.card → Fin F.card → ℝ} {g₂ : Fin F.card → Fin G.card → ℝ}
    (h₁ : IsCoupling E F g₁) (h₂ : IsCoupling F G g₂) :
    planCost E G c (glue E F G g₁ g₂) ≤ planCost E F c g₁ + planCost F G c g₂ := by
  classical
  -- reorganise the cost of the glued plan as a sum over the intermediate conformation
  have hexp : planCost E G c (glue E F G g₁ g₂)
      = ∑ j, ∑ i, ∑ l, (g₁ i j * g₂ j l / F.w j) * c (E.pt i) (G.pt l) := by
    simp only [planCost, glue, Finset.sum_mul]
    rw [Finset.sum_comm (γ := Fin F.card)]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.sum_comm]
  have hA : planCost E F c g₁ = ∑ j, ∑ i, g₁ i j * c (E.pt i) (F.pt j) := by
    simp only [planCost]; rw [Finset.sum_comm]
  have hB : planCost F G c g₂ = ∑ j, ∑ l, g₂ j l * c (F.pt j) (G.pt l) := by
    simp only [planCost]
  rw [hexp, hA, hB, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun j _ => ?_
  rcases eq_or_lt_of_le (F.w_nonneg j) with hw | hw
  · -- an unpopulated intermediate conformation carries no mass at all
    have h1 : ∀ i, g₁ i j = 0 := fun i => h₁.eq_zero_of_col_zero i hw.symm
    have h2 : ∀ l, g₂ j l = 0 := fun l => h₂.eq_zero_of_row_zero hw.symm l
    simp [h1, h2]
  · have hwne : F.w j ≠ 0 := ne_of_gt hw
    -- bound the glued cost by the triangle inequality through `F.pt j`
    have step : ∑ i, ∑ l, (g₁ i j * g₂ j l / F.w j) * c (E.pt i) (G.pt l)
        ≤ ∑ i, ∑ l, (g₁ i j * g₂ j l / F.w j) *
            (c (E.pt i) (F.pt j) + c (F.pt j) (G.pt l)) := by
      refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun l _ => ?_
      exact mul_le_mul_of_nonneg_left (htri _ _ _)
        (div_nonneg (mul_nonneg (h₁.nonneg i j) (h₂.nonneg j l)) (F.w_nonneg j))
    refine step.trans_eq ?_
    have hsplit : ∀ i, ∑ l, (g₁ i j * g₂ j l / F.w j) *
          (c (E.pt i) (F.pt j) + c (F.pt j) (G.pt l))
        = g₁ i j * c (E.pt i) (F.pt j)
          + (g₁ i j / F.w j) * ∑ l, g₂ j l * c (F.pt j) (G.pt l) := by
      intro i
      have h1 : ∑ l, (g₁ i j * g₂ j l / F.w j) * c (E.pt i) (F.pt j)
          = g₁ i j * c (E.pt i) (F.pt j) := by
        have : ∑ l, (g₁ i j * g₂ j l / F.w j) * c (E.pt i) (F.pt j)
            = ((g₁ i j / F.w j) * ∑ l, g₂ j l) * c (E.pt i) (F.pt j) := by
          rw [Finset.mul_sum, Finset.sum_mul]
          exact Finset.sum_congr rfl fun l _ => by field_simp
        rw [this, h₂.row j]
        field_simp
      have h2 : ∑ l, (g₁ i j * g₂ j l / F.w j) * c (F.pt j) (G.pt l)
          = (g₁ i j / F.w j) * ∑ l, g₂ j l * c (F.pt j) (G.pt l) := by
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun l _ => by field_simp
      calc ∑ l, (g₁ i j * g₂ j l / F.w j) * (c (E.pt i) (F.pt j) + c (F.pt j) (G.pt l))
          = ∑ l, ((g₁ i j * g₂ j l / F.w j) * c (E.pt i) (F.pt j)
              + (g₁ i j * g₂ j l / F.w j) * c (F.pt j) (G.pt l)) := by
            exact Finset.sum_congr rfl fun l _ => by ring
        _ = g₁ i j * c (E.pt i) (F.pt j)
              + (g₁ i j / F.w j) * ∑ l, g₂ j l * c (F.pt j) (G.pt l) := by
            rw [Finset.sum_add_distrib, h1, h2]
    rw [Finset.sum_congr rfl fun i _ => hsplit i, Finset.sum_add_distrib]
    congr 1
    rw [← Finset.sum_mul]
    have : ∑ i, g₁ i j / F.w j = 1 := by
      rw [← Finset.sum_div, h₁.col j]
      field_simp
    rw [this, one_mul]

/-- **The gluing lemma / triangle inequality.**  Structural error accumulates along a chain
of ensembles exactly as a distance should: the cost of turning `E` into `G` never exceeds
the cost of going through any intermediate ensemble `F`.  Only now is it legitimate to talk
about a model being "`1.5 Å` from the truth". -/
theorem transportCost_triangle {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y)
    (htri : ∀ x y z, c x z ≤ c x y + c y z) (E F G : Ens X) :
    transportCost c E G ≤ transportCost c E F + transportCost c F G := by
  obtain ⟨g₁, h₁, hc₁⟩ := exists_optimal_coupling hc E F
  obtain ⟨g₂, h₂, hc₂⟩ := exists_optimal_coupling hc F G
  rw [hc₁, hc₂]
  exact (transportCost_le_of_coupling hc (glue_isCoupling h₁ h₂)).trans
    (planCost_glue_le htri h₁ h₂)

/-! ## Zero distance means observational identity -/

/-- If the transport cost vanishes for a structural metric, the two ensembles are
observationally identical: no experiment can tell them apart. -/
theorem same_of_transportCost_eq_zero {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y)
    (hsep : ∀ x y, c x y = 0 → x = y) {E F : Ens X} (h : transportCost c E F = 0) :
    E.Same F := by
  obtain ⟨g, hg, hgc⟩ := exists_optimal_coupling hc E F
  have hzero : planCost E F c g = 0 := by rw [← hgc, h]
  have hterm : ∀ i j, g i j * c (E.pt i) (F.pt j) = 0 := by
    intro i j
    have hnn : ∀ i' ∈ (univ : Finset (Fin E.card)), 0 ≤ ∑ j', g i' j' * c (E.pt i') (F.pt j') :=
      fun i' _ => Finset.sum_nonneg fun j' _ => mul_nonneg (hg.nonneg i' j') (hc _ _)
    have h1 : ∑ j', g i j' * c (E.pt i) (F.pt j') = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg hnn).1 hzero i (mem_univ i)
    exact (Finset.sum_eq_zero_iff_of_nonneg
      (fun j' _ => mul_nonneg (hg.nonneg i j') (hc _ _))).1 h1 j (mem_univ j)
  have hpt : ∀ i j, g i j ≠ 0 → E.pt i = F.pt j := by
    intro i j hne
    exact hsep _ _ (by
      rcases mul_eq_zero.1 (hterm i j) with h' | h'
      · exact absurd h' hne
      · exact h')
  intro f
  have h1 : E.expect f = ∑ i, ∑ j, g i j * f (E.pt i) := by
    simp only [Ens.expect]
    exact Finset.sum_congr rfl fun i _ => by rw [← Finset.sum_mul, hg.row i]
  have h2 : F.expect f = ∑ j, ∑ i, g i j * f (F.pt j) := by
    simp only [Ens.expect]
    exact Finset.sum_congr rfl fun j _ => by rw [← Finset.sum_mul, hg.col j]
  rw [h1, h2, Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i _ => ?_
  by_cases hij : g i j = 0
  · simp [hij]
  · rw [hpt i j hij]

/-- The plan that matches equal conformations to each other, splitting the mass at a
conformation between the two ensembles in proportion to their component weights. -/
noncomputable def matchPlan (E F : Ens X) : Fin E.card → Fin F.card → ℝ :=
  fun i j => if E.pt i = F.pt j then E.w i * F.w j / E.prob (E.pt i) else 0

lemma prob_eq_sum_ite (E : Ens X) (x : X) :
    E.prob x = ∑ i, (if E.pt i = x then E.w i else 0) := by
  simp only [Ens.prob, Ens.expect]
  exact Finset.sum_congr rfl fun i _ => by by_cases hx : E.pt i = x <;> simp [hx]

lemma w_le_prob (E : Ens X) (i : Fin E.card) : E.w i ≤ E.prob (E.pt i) := by
  rw [prob_eq_sum_ite]
  have hterm : (if E.pt i = E.pt i then E.w i else 0)
      ≤ ∑ i', (if E.pt i' = E.pt i then E.w i' else 0) :=
    Finset.single_le_sum (f := fun i' => if E.pt i' = E.pt i then E.w i' else 0)
      (fun i' _ => by by_cases hx : E.pt i' = E.pt i <;> simp [hx, E.w_nonneg]) (mem_univ i)
  simpa using hterm

lemma matchPlan_nonneg (E F : Ens X) (i : Fin E.card) (j : Fin F.card) :
    0 ≤ matchPlan E F i j := by
  unfold matchPlan
  by_cases hij : E.pt i = F.pt j
  · rw [if_pos hij]
    exact div_nonneg (mul_nonneg (E.w_nonneg i) (F.w_nonneg j)) (E.prob_nonneg _)
  · rw [if_neg hij]

lemma matchPlan_isCoupling {E F : Ens X} (h : E.Same F) : IsCoupling E F (matchPlan E F) := by
  have hprob : ∀ x, E.prob x = F.prob x := Ens.prob_eq_of_same h
  refine ⟨matchPlan_nonneg E F, fun i => ?_, fun j => ?_⟩
  · rcases eq_or_lt_of_le (E.w_nonneg i) with hw | hw
    · have hz : ∀ j, matchPlan E F i j = 0 := by
        intro j
        unfold matchPlan
        by_cases hij : E.pt i = F.pt j
        · rw [if_pos hij, ← hw]; simp
        · rw [if_neg hij]
      simp [hz, ← hw]
    · have hp : 0 < E.prob (E.pt i) := lt_of_lt_of_le hw (w_le_prob E i)
      have hsum : ∑ j, matchPlan E F i j
          = (E.w i / E.prob (E.pt i)) * ∑ j, (if F.pt j = E.pt i then F.w j else 0) := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun j _ => ?_
        unfold matchPlan
        by_cases hij : E.pt i = F.pt j
        · rw [if_pos hij, if_pos hij.symm]; ring
        · rw [if_neg hij, if_neg (fun hx : F.pt j = E.pt i => hij hx.symm)]; ring
      rw [hsum, ← prob_eq_sum_ite F (E.pt i), ← hprob]
      field_simp
  · rcases eq_or_lt_of_le (F.w_nonneg j) with hw | hw
    · have hz : ∀ i, matchPlan E F i j = 0 := by
        intro i
        unfold matchPlan
        by_cases hij : E.pt i = F.pt j
        · rw [if_pos hij, ← hw]; simp
        · rw [if_neg hij]
      simp [hz, ← hw]
    · have hp : 0 < F.prob (F.pt j) := lt_of_lt_of_le hw (w_le_prob F j)
      have hpE : 0 < E.prob (F.pt j) := by rw [hprob]; exact hp
      have hsum : ∑ i, matchPlan E F i j
          = F.w j * ∑ i, (if E.pt i = F.pt j then E.w i else 0) / E.prob (F.pt j) := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun i _ => ?_
        unfold matchPlan
        by_cases hij : E.pt i = F.pt j
        · rw [if_pos hij, if_pos hij, hij]; ring
        · rw [if_neg hij, if_neg hij]; ring
      rw [hsum, ← Finset.sum_div, ← prob_eq_sum_ite E (F.pt j)]
      field_simp

/-- Conversely, observationally identical ensembles are at zero transport cost: the plan
that matches equal conformations to each other is available. -/
theorem transportCost_eq_zero_of_same {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y)
    (hcd : ∀ x, c x x = 0) {E F : Ens X} (h : E.Same F) : transportCost c E F = 0 := by
  refine le_antisymm ?_ (transportCost_nonneg hc E F)
  refine (transportCost_le_of_coupling hc (matchPlan_isCoupling h)).trans_eq ?_
  simp only [planCost]
  refine Finset.sum_eq_zero fun i _ => Finset.sum_eq_zero fun j _ => ?_
  unfold matchPlan
  by_cases hij : E.pt i = F.pt j
  · rw [if_pos hij, hij, hcd]; ring
  · rw [if_neg hij]; ring


/-- **The transport cost is a metric on ensembles modulo experiment.**  It vanishes exactly
on observationally identical pairs. -/
theorem transportCost_eq_zero_iff_same {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y)
    (hcd : ∀ x, c x x = 0) (hsep : ∀ x y, c x y = 0 → x = y) (E F : Ens X) :
    transportCost c E F = 0 ↔ E.Same F :=
  ⟨same_of_transportCost_eq_zero hc hsep, transportCost_eq_zero_of_same hc hcd⟩

/-- **Summary: the structural error of an ensemble model is a distance.**  For any
structural metric `c` on conformation space the transport cost is nonnegative, attained by
an explicit optimal matching, symmetric, additive-subadditive along chains, and zero
exactly on ensembles no experiment can distinguish. -/
theorem transportCost_pseudoMetric {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y)
    (hcd : ∀ x, c x x = 0) (hsymm : ∀ x y, c x y = c y x)
    (htri : ∀ x y z, c x z ≤ c x y + c y z) (hsep : ∀ x y, c x y = 0 → x = y) :
    (∀ E F : Ens X, 0 ≤ transportCost c E F) ∧
    (∀ E F : Ens X, ∃ g, IsCoupling E F g ∧ transportCost c E F = planCost E F c g) ∧
    (∀ E F : Ens X, transportCost c E F = transportCost c F E) ∧
    (∀ E F G : Ens X, transportCost c E G ≤ transportCost c E F + transportCost c F G) ∧
    (∀ E F : Ens X, transportCost c E F = 0 ↔ E.Same F) :=
  ⟨fun E F => transportCost_nonneg hc E F,
   fun E F => exists_optimal_coupling hc E F,
   fun E F => transportCost_comm hc hsymm E F,
   fun E F G => transportCost_triangle hc htri E F G,
   fun E F => transportCost_eq_zero_iff_same hc hcd hsep E F⟩

end IDR
