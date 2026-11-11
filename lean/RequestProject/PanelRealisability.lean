/-
# When is a measured pairwise panel realisable by *some* IDR ensemble?

An intrinsically disordered region is never a single structure, so every experimental restraint on
it is an **ensemble average**.  The most common such restraints are binary-state averages: for each
residue `i` a per-residue occupancy `p i` (HDX protection, NMR order parameters, a helical
propensity), and for each pair `i, j` a co-occupancy `q i j` (crosslinking, PRE, contact
frequencies).  A modelling pipeline that claims to "fully capture" the region has to answer a prior
question: *is the measured panel realisable at all?*  If no ensemble reproduces it, the panel is
internally inconsistent and any fit to it is fitting noise.

This file gives the exact answer for the pairwise-disagreement panel
`d i j = P(state of i ≠ state of j)` (equivalently `d i j = p i + p j - 2 q i j`, proved as
`disagreePanel_eq_occupancy`):

* `realisable_iff_mem_cutPolytope` — the realisable panels are **exactly** the cut polytope of the
  complete graph on the residues, i.e. the convex hull of the cut vectors of the two-colourings of
  the chain.
* `disagree_triangle` and `disagree_triangle_sum_le_two` — the triangle facets.  These are
  *falsifiable consistency tests* an experimentalist can run on a panel before any modelling:
  `d i j ≤ d i k + d k j` and `d i j + d j k + d k i ≤ 2`.  A panel violating either cannot come
  from any ensemble whatsoever, no matter how flexible the model.
* `panel_optimum_eq_maxcut` — weighted MAX-CUT is exactly the problem of maximising a linear
  functional of the panel over all ensembles.  The optimum over ensembles is attained at a single
  conformation, and equals the maximum cut of the weight matrix.

The last item is a machine-checked reduction in the same style as the linear-ordering reduction of
`RequestProject.AUCHardness`: the *reduction* is proved here, while the NP-hardness of weighted
MAX-CUT itself (Karp) and the NP-hardness of deciding membership in the cut polytope
(Avis–Deza) are classical results cited informally, not machine-checked.  What is proved here is
the modelling-relevant content: the feasible set of measured panels is a cut polytope, so panel
consistency checking inherits the geometry (and the difficulty) of that polytope, and the triangle
tests above are valid necessary conditions.
-/
import Mathlib

set_option autoImplicit false

namespace IDR.PanelRealisability

open Finset

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- A finitely supported ensemble of binary conformational states on the residues `V`: a
probability distribution over two-colourings `V → Bool` (for instance "residue `i` is protected /
exposed", or "helical / coil"). -/
structure Ensemble (V : Type*) [Fintype V] [DecidableEq V] where
  /-- The population weight of each conformational state. -/
  weight : (V → Bool) → ℝ
  /-- Populations are non-negative. -/
  nonneg : ∀ s, 0 ≤ weight s
  /-- Populations sum to one. -/
  total : ∑ s, weight s = 1

namespace Ensemble

/-- The per-residue occupancy measured for residue `i`. -/
noncomputable def occupancy (E : Ensemble V) (i : V) : ℝ :=
  ∑ s, if s i then E.weight s else 0

/-- The pair co-occupancy measured for the pair `i, j`. -/
noncomputable def coOccupancy (E : Ensemble V) (i j : V) : ℝ :=
  ∑ s, if s i ∧ s j then E.weight s else 0

/-- The pairwise *disagreement* panel: the probability that residues `i` and `j` are in different
states. -/
noncomputable def disagreePanel (E : Ensemble V) : V → V → ℝ := fun i j =>
  ∑ s, if s i ≠ s j then E.weight s else 0

end Ensemble

open Ensemble

/-- The disagreement panel is determined by the occupancy/co-occupancy panel:
`d i j = p i + p j - 2 q i j`. -/
theorem disagreePanel_eq_occupancy (E : Ensemble V) (i j : V) :
    E.disagreePanel i j = E.occupancy i + E.occupancy j - 2 * E.coOccupancy i j := by
  unfold Ensemble.disagreePanel Ensemble.occupancy Ensemble.coOccupancy
  rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun s _ => ?_
  cases s i <;> cases s j <;> (simp; try ring)

theorem disagreePanel_nonneg (E : Ensemble V) (i j : V) : 0 ≤ E.disagreePanel i j :=
  Finset.sum_nonneg fun s _ => by
    split
    · exact E.nonneg s
    · exact le_rfl

theorem disagreePanel_le_one (E : Ensemble V) (i j : V) : E.disagreePanel i j ≤ 1 := by
  calc E.disagreePanel i j ≤ ∑ s, E.weight s := by
        refine Finset.sum_le_sum fun s _ => ?_
        split
        · exact le_rfl
        · exact E.nonneg s
    _ = 1 := E.total

theorem disagreePanel_self (E : Ensemble V) (i : V) : E.disagreePanel i i = 0 := by
  simp [Ensemble.disagreePanel]

theorem disagreePanel_comm (E : Ensemble V) (i j : V) :
    E.disagreePanel i j = E.disagreePanel j i := by
  refine Finset.sum_congr rfl fun s _ => ?_
  by_cases h : s i = s j <;> simp [h, Ne.symm]

/-- **Triangle test.**  Every realisable panel is a semimetric: `d i j ≤ d i k + d k j`. -/
theorem disagree_triangle (E : Ensemble V) (i j k : V) :
    E.disagreePanel i j ≤ E.disagreePanel i k + E.disagreePanel k j := by
  unfold Ensemble.disagreePanel
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun s _ => ?_
  have h := E.nonneg s
  cases s i <;> cases s j <;> cases s k <;> simp <;> linarith

/-- **Perimeter test.**  Every realisable panel satisfies the triangle facet
`d i j + d j k + d k i ≤ 2`: a two-colouring can separate at most two of the three pairs. -/
theorem disagree_triangle_sum_le_two (E : Ensemble V) (i j k : V) :
    E.disagreePanel i j + E.disagreePanel j k + E.disagreePanel k i ≤ 2 := by
  unfold Ensemble.disagreePanel
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  calc (∑ s, ((if s i ≠ s j then E.weight s else 0) + (if s j ≠ s k then E.weight s else 0)
          + (if s k ≠ s i then E.weight s else 0)))
      ≤ ∑ s, 2 * E.weight s := by
        refine Finset.sum_le_sum fun s _ => ?_
        have h := E.nonneg s
        cases s i <;> cases s j <;> cases s k <;> simp <;> linarith
    _ = 2 := by rw [← Finset.mul_sum, E.total, mul_one]

/-! ## The realisable panels form the cut polytope -/

/-- The panel of a single conformation `s`: the cut vector of the two-colouring `s`. -/
def cutVec (s : V → Bool) : V → V → ℝ := fun i j => if s i ≠ s j then 1 else 0

/-- The point-mass ensemble at a single conformation. -/
noncomputable def pointMass (s₀ : V → Bool) : Ensemble V where
  weight := fun s => if s = s₀ then 1 else 0
  nonneg := by intro s; split <;> norm_num
  total := by simp

theorem disagreePanel_pointMass (s₀ : V → Bool) :
    (pointMass s₀).disagreePanel = cutVec s₀ := by
  funext i j
  show (∑ s, if s i ≠ s j then (pointMass s₀).weight s else 0) = cutVec s₀ i j
  rw [Finset.sum_eq_single s₀]
  · simp [pointMass, cutVec]
  · intro b _ hb; simp [pointMass, hb]
  · intro h; exact absurd (Finset.mem_univ s₀) h

/-- The cut polytope of the complete graph on the residues. -/
def cutPolytope (V : Type*) [Fintype V] [DecidableEq V] : Set (V → V → ℝ) :=
  convexHull ℝ (Set.range (cutVec (V := V)))

/-- The set of panels that some ensemble realises. -/
def Realisable (V : Type*) [Fintype V] [DecidableEq V] : Set (V → V → ℝ) :=
  {d | ∃ E : Ensemble V, E.disagreePanel = d}

/-- The panel of an ensemble is the corresponding convex combination of cut vectors. -/
theorem disagreePanel_eq_sum_smul (E : Ensemble V) :
    E.disagreePanel = ∑ s, E.weight s • cutVec s := by
  funext i j
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, cutVec, mul_ite, mul_one, mul_zero]
  rfl

/-- Every realisable panel lies in the cut polytope. -/
theorem disagreePanel_mem_cutPolytope (E : Ensemble V) :
    E.disagreePanel ∈ cutPolytope V := by
  rw [disagreePanel_eq_sum_smul]
  refine (convex_convexHull ℝ _).sum_mem (fun s _ => E.nonneg s) E.total ?_
  intro s _
  exact subset_convexHull ℝ _ ⟨s, rfl⟩

/-- Mixing two ensembles gives an ensemble, so the realisable panels form a convex set. -/
theorem convex_realisable : Convex ℝ (Realisable V) := by
  rintro d ⟨E, rfl⟩ d' ⟨E', rfl⟩ a b ha hb hab
  refine ⟨⟨fun s => a * E.weight s + b * E'.weight s, ?_, ?_⟩, ?_⟩
  · intro s
    exact add_nonneg (mul_nonneg ha (E.nonneg s)) (mul_nonneg hb (E'.nonneg s))
  · rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, E.total, E'.total,
      mul_one, mul_one]
    exact hab
  · funext i j
    simp only [Ensemble.disagreePanel, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun s _ => ?_
    split <;> ring

/-- **The realisable measured panels are exactly the cut polytope.**  Consequently, checking
whether a measured pairwise panel can be reproduced by *any* ensemble is exactly the cut-polytope
membership problem, which is NP-hard (Avis–Deza); and the triangle tests above are the
corresponding necessary conditions that an experimentalist can check in polynomial time. -/
theorem realisable_iff_mem_cutPolytope (d : V → V → ℝ) :
    d ∈ Realisable V ↔ d ∈ cutPolytope V := by
  constructor
  · rintro ⟨E, rfl⟩
    exact disagreePanel_mem_cutPolytope E
  · intro h
    refine convexHull_min ?_ convex_realisable h
    rintro x ⟨s, rfl⟩
    exact ⟨pointMass s, disagreePanel_pointMass s⟩

/-! ## Weighted MAX-CUT is the ensemble optimum -/

/-- The cut value of a single conformation under a weight matrix. -/
noncomputable def cutValue (w : V → V → ℝ) (s : V → Bool) : ℝ :=
  ∑ i, ∑ j, if s i ≠ s j then w i j else 0

/-- The linear functional of the measured panel that a weight matrix defines. -/
noncomputable def panelValue (w : V → V → ℝ) (E : Ensemble V) : ℝ :=
  ∑ i, ∑ j, w i j * E.disagreePanel i j

/-- The maximum cut of a weight matrix. -/
noncomputable def maxCut (w : V → V → ℝ) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty (cutValue w)

theorem panelValue_eq_sum (w : V → V → ℝ) (E : Ensemble V) :
    panelValue w E = ∑ s, E.weight s * cutValue w s := by
  have step : ∀ i j : V, w i j * E.disagreePanel i j
      = ∑ s, (if s i ≠ s j then w i j * E.weight s else 0) := by
    intro i j
    simp only [Ensemble.disagreePanel, Finset.mul_sum, mul_ite, mul_zero]
  calc panelValue w E = ∑ i, ∑ j, ∑ s, (if s i ≠ s j then w i j * E.weight s else 0) :=
        Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => step i j
    _ = ∑ i, ∑ s, ∑ j, (if s i ≠ s j then w i j * E.weight s else 0) :=
        Finset.sum_congr rfl fun i _ => Finset.sum_comm
    _ = ∑ s, ∑ i, ∑ j, (if s i ≠ s j then w i j * E.weight s else 0) := Finset.sum_comm
    _ = ∑ s, E.weight s * cutValue w s := by
        refine Finset.sum_congr rfl fun s _ => ?_
        simp only [cutValue, Finset.mul_sum, mul_ite, mul_zero]
        exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by
          split
          · ring
          · rfl

theorem panelValue_pointMass (w : V → V → ℝ) (s₀ : V → Bool) :
    panelValue w (pointMass s₀) = cutValue w s₀ := by
  rw [panelValue_eq_sum, Finset.sum_eq_single s₀]
  · simp [pointMass]
  · intro b _ hb; simp [pointMass, hb]
  · intro h; exact absurd (Finset.mem_univ s₀) h

/-- **Weighted MAX-CUT is exactly the ensemble optimum.**  No ensemble beats the best single
conformation for a linear panel functional, and the best single conformation is achieved by an
ensemble.  Hence maximising a linear functional of a measured pairwise panel over all ensembles
is precisely weighted MAX-CUT, which is NP-hard. -/
theorem panel_optimum_eq_maxcut (w : V → V → ℝ) :
    (∀ E : Ensemble V, panelValue w E ≤ maxCut w) ∧
      ∃ E : Ensemble V, panelValue w E = maxCut w := by
  constructor
  · intro E
    rw [panelValue_eq_sum]
    calc (∑ s, E.weight s * cutValue w s) ≤ ∑ s, E.weight s * maxCut w := by
          refine Finset.sum_le_sum fun s _ => ?_
          exact mul_le_mul_of_nonneg_left
            (Finset.le_sup' (cutValue w) (Finset.mem_univ s)) (E.nonneg s)
      _ = maxCut w := by rw [← Finset.sum_mul, E.total, one_mul]
  · obtain ⟨s₀, -, hs₀⟩ := Finset.exists_mem_eq_sup' (Finset.univ_nonempty) (cutValue w)
    exact ⟨pointMass s₀, by rw [panelValue_pointMass, maxCut, hs₀]⟩

end IDR.PanelRealisability
