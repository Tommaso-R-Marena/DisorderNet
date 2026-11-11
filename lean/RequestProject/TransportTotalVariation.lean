/-
# The population-space `ℓ¹` law is the transport law of the crudest possible geometry

`RequestProject.Metric` measures a disorder model by the `ℓ¹` distance between population
vectors and derives the capacity and entropy laws from it; `RequestProject.Transport` points
out that `ℓ¹` is blind to structural similarity and replaces it by a transport cost.  The
two are not rival choices: this file proves they are the *same* construction, evaluated at
two different structural metrics.

`transportCost_unitDist_eq_half_ell1`: for the discrete structural metric -- every pair of
distinct conformations counted as one unit apart, i.e. the assumption that no two distinct
structures resemble each other at all -- the transport cost is exactly half the `ℓ¹`
distance, i.e. the total-variation distance.  So

* every earlier `ℓ¹`/capacity/entropy result is a statement about transport distance in the
  crudest geometry, and remains valid;
* and the transport cost against a real structural metric (RMSD, contact distance) is the
  same quantity computed in a geometry that knows how close two structures are.

The proof is an instance of strong Kantorovich duality: the upper bound is the explicit
maximal-overlap plan (keep the common population in place, redistribute the rest as a
product), the matching lower bound is the single dual test function `1_{p > q}`, and the
two meet.  Auxiliary results of independent use: `Ens.canon` (the canonical
one-component-per-conformation form of an ensemble) and `transportCost_congr_same` (the
transport cost only depends on ensembles through what experiment can see).
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Metric
import RequestProject.Transport
import RequestProject.TransportGeometry
import RequestProject.TransportDuality

namespace IDR

open Finset
open scoped Classical

variable {X : Type*}

/-! ## The discrete structural metric -/

/-- The crudest structural metric: distinct conformations are one unit apart. -/
noncomputable def unitDist (x y : X) : ℝ := if x = y then 0 else 1

lemma unitDist_nonneg (x y : X) : 0 ≤ unitDist x y := by
  unfold unitDist; split <;> norm_num

lemma unitDist_self (x : X) : unitDist x x = 0 := by simp [unitDist]

lemma unitDist_comm (x y : X) : unitDist x y = unitDist y x := by
  unfold unitDist
  by_cases h : x = y
  · simp [h]
  · simp [h, Ne.symm h]

lemma unitDist_triangle (x y z : X) : unitDist x z ≤ unitDist x y + unitDist y z := by
  unfold unitDist
  by_cases hxz : x = z
  · simp [hxz]
    split <;> split <;> norm_num
  · by_cases hxy : x = y
    · subst hxy
      simp [hxz]
    · simp [hxz, hxy]
      split <;> norm_num

lemma unitDist_eq_zero {x y : X} (h : unitDist x y = 0) : x = y := by
  unfold unitDist at h
  by_contra hne
  simp [hne] at h

/-! ## Ensembles depend on nothing but their populations -/

/-- The transport cost only sees ensembles through what experiment can see. -/
theorem transportCost_congr_same {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) (hcd : ∀ x, c x x = 0)
    (htri : ∀ x y z, c x z ≤ c x y + c y z) {E E' F F' : Ens X}
    (hE : E.Same E') (hF : F.Same F') : transportCost c E F = transportCost c E' F' := by
  have hz1 : transportCost c E E' = 0 := transportCost_eq_zero_of_same hc hcd hE
  have hz2 : transportCost c E' E = 0 := transportCost_eq_zero_of_same hc hcd hE.symm
  have hz3 : transportCost c F F' = 0 := transportCost_eq_zero_of_same hc hcd hF
  have hz4 : transportCost c F' F = 0 := transportCost_eq_zero_of_same hc hcd hF.symm
  refine le_antisymm ?_ ?_
  · calc transportCost c E F ≤ transportCost c E E' + transportCost c E' F :=
          transportCost_triangle hc htri _ _ _
      _ ≤ 0 + (transportCost c E' F' + transportCost c F' F) := by
          rw [hz1]
          have := transportCost_triangle hc htri E' F' F
          linarith
      _ = transportCost c E' F' := by rw [hz4]; ring
  · calc transportCost c E' F' ≤ transportCost c E' E + transportCost c E F' :=
          transportCost_triangle hc htri _ _ _
      _ ≤ 0 + (transportCost c E F + transportCost c F F') := by
          rw [hz2]
          have := transportCost_triangle hc htri E F F'
          linarith
      _ = transportCost c E F := by rw [hz3]; ring

namespace Ens

variable [Fintype X]

/-- The canonical form of an ensemble: one component per conformation of the (finite)
conformation space, carrying that conformation's population. -/
noncomputable def canon (E : Ens X) : Ens X where
  card := Fintype.card X
  pt := (Fintype.equivFin X).symm
  w := fun i => E.prob ((Fintype.equivFin X).symm i)
  w_nonneg := fun i => E.prob_nonneg _
  w_sum := by
    rw [Equiv.sum_comp (Fintype.equivFin X).symm (fun x => E.prob x)]
    exact E.sum_prob

@[simp] lemma canon_prob (E : Ens X) (x : X) : (E.canon).prob x = E.prob x := by
  have hrfl : (E.canon).prob x = ∑ i : Fin (Fintype.card X),
      E.prob ((Fintype.equivFin X).symm i) * (if (Fintype.equivFin X).symm i = x then (1 : ℝ)
        else 0) := rfl
  rw [hrfl, Equiv.sum_comp (Fintype.equivFin X).symm
    (fun y => E.prob y * (if y = x then (1 : ℝ) else 0))]
  rw [Finset.sum_eq_single x] <;> simp +contextual

lemma canon_same (E : Ens X) : (E.canon).Same E :=
  (same_iff_prob_eq _ _).2 fun x => canon_prob E x

end Ens

/-! ## Maximal overlap: the explicit optimal plan -/

section TV

variable [Fintype X]

/-- The population that the two ensembles genuinely share. -/
noncomputable def overlap (E F : Ens X) : ℝ := ∑ x, min (E.prob x) (F.prob x)

/-- The population that must move: one half of the `ℓ¹` distance. -/
lemma one_sub_overlap_eq (E F : Ens X) : 1 - overlap E F = Ens.ell1 E F / 2 := by
  have habs : ∀ x : X, |E.prob x - F.prob x|
      = E.prob x + F.prob x - 2 * min (E.prob x) (F.prob x) := by
    intro x
    rcases le_total (E.prob x) (F.prob x) with h | h
    · rw [abs_of_nonpos (by linarith), min_eq_left h]; ring
    · rw [abs_of_nonneg (by linarith), min_eq_right h]; ring
  rw [Ens.ell1, Finset.sum_congr rfl fun x _ => habs x]
  rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, E.sum_prob, F.sum_prob, ← Finset.mul_sum,
    ← overlap]
  ring

lemma overlap_le_one (E F : Ens X) : overlap E F ≤ 1 := by
  calc overlap E F ≤ ∑ x, E.prob x :=
        Finset.sum_le_sum fun x _ => min_le_left _ _
    _ = 1 := E.sum_prob

/-- The population of a conformation that the model over-supplies. -/
noncomputable def resE (E F : Ens X) (x : X) : ℝ := E.prob x - min (E.prob x) (F.prob x)

/-- The population of a conformation that the model under-supplies. -/
noncomputable def resF (E F : Ens X) (x : X) : ℝ := F.prob x - min (E.prob x) (F.prob x)

omit [Fintype X] in
lemma resE_nonneg (E F : Ens X) (x : X) : 0 ≤ resE E F x := by
  simp [resE]

omit [Fintype X] in
lemma resF_nonneg (E F : Ens X) (x : X) : 0 ≤ resF E F x := by
  simp [resF]

lemma sum_resE (E F : Ens X) : ∑ x, resE E F x = 1 - overlap E F := by
  simp only [resE, overlap]
  rw [Finset.sum_sub_distrib, E.sum_prob]

lemma sum_resF (E F : Ens X) : ∑ x, resF E F x = 1 - overlap E F := by
  simp only [resF, overlap]
  rw [Finset.sum_sub_distrib, F.sum_prob]

/-- The enumeration of conformation space used by the canonical form. -/
noncomputable def enum (X : Type*) [Fintype X] : Fin (Fintype.card X) → X :=
  (Fintype.equivFin X).symm

lemma sum_resE_fin (E F : Ens X) :
    ∑ i : Fin (Fintype.card X), resE E F (enum X i) = 1 - overlap E F := by
  rw [enum, Equiv.sum_comp (Fintype.equivFin X).symm (fun x => resE E F x), sum_resE]

lemma sum_resF_fin (E F : Ens X) :
    ∑ i : Fin (Fintype.card X), resF E F (enum X i) = 1 - overlap E F := by
  rw [enum, Equiv.sum_comp (Fintype.equivFin X).symm (fun x => resF E F x), sum_resF]

/-- The maximal-overlap plan on canonical ensembles: keep the shared population where it
is, and move the rest as a product plan. -/
noncomputable def tvPlan (E F : Ens X) :
    Fin (Fintype.card X) → Fin (Fintype.card X) → ℝ :=
  fun i j =>
    (if i = j then min (E.prob (enum X i)) (F.prob (enum X i)) else 0)
      + resE E F (enum X i) * resF E F (enum X j) / (1 - overlap E F)

lemma canon_w (E : Ens X) (i : Fin (Fintype.card X)) : (E.canon).w i = E.prob (enum X i) := rfl

lemma canon_pt (E : Ens X) (i : Fin (Fintype.card X)) : (E.canon).pt i = enum X i := rfl

lemma tvPlan_isCoupling (E F : Ens X) : IsCoupling (E.canon) (F.canon) (tvPlan E F) := by
  have hR0 : 0 ≤ 1 - overlap E F := by linarith [overlap_le_one E F]
  refine ⟨fun i j => ?_, fun i => ?_, fun j => ?_⟩
  · refine add_nonneg ?_ (div_nonneg (mul_nonneg (resE_nonneg E F _) (resF_nonneg E F _)) hR0)
    split
    · exact le_min (E.prob_nonneg _) (F.prob_nonneg _)
    · exact le_rfl
  · show ∑ j : Fin (Fintype.card X), tvPlan E F i j = E.prob (enum X i)
    have hsplit : ∑ j : Fin (Fintype.card X), tvPlan E F i j
        = min (E.prob (enum X i)) (F.prob (enum X i))
          + resE E F (enum X i) * (1 - overlap E F) / (1 - overlap E F) := by
      simp only [tvPlan]
      rw [Finset.sum_add_distrib]
      congr 1
      · simp
      · rw [← Finset.sum_div, ← Finset.mul_sum, sum_resF_fin]
    rw [hsplit]
    rcases eq_or_lt_of_le hR0 with hR | hR
    · have hzero : resE E F (enum X i) = 0 := by
        have hsum : ∑ x, resE E F x = 0 := by rw [sum_resE]; linarith
        exact (Finset.sum_eq_zero_iff_of_nonneg
          (fun y _ => resE_nonneg E F y)).1 hsum _ (mem_univ _)
      have hmin : min (E.prob (enum X i)) (F.prob (enum X i)) = E.prob (enum X i) := by
        simp only [resE, sub_eq_zero] at hzero
        exact hzero.symm
      rw [hzero, hmin]
      simp
    · rw [mul_div_assoc, div_self (ne_of_gt hR), mul_one]
      simp [resE]
  · show ∑ i : Fin (Fintype.card X), tvPlan E F i j = F.prob (enum X j)
    have hsplit : ∑ i : Fin (Fintype.card X), tvPlan E F i j
        = min (E.prob (enum X j)) (F.prob (enum X j))
          + (1 - overlap E F) * resF E F (enum X j) / (1 - overlap E F) := by
      simp only [tvPlan]
      rw [Finset.sum_add_distrib]
      congr 1
      · simp
      · rw [← Finset.sum_div, ← Finset.sum_mul, sum_resE_fin]
    rw [hsplit]
    rcases eq_or_lt_of_le hR0 with hR | hR
    · have hzero : resF E F (enum X j) = 0 := by
        have hsum : ∑ x, resF E F x = 0 := by rw [sum_resF]; linarith
        exact (Finset.sum_eq_zero_iff_of_nonneg
          (fun y _ => resF_nonneg E F y)).1 hsum _ (mem_univ _)
      have hmin : min (E.prob (enum X j)) (F.prob (enum X j)) = F.prob (enum X j) := by
        simp only [resF, sub_eq_zero] at hzero
        exact hzero.symm
      rw [hzero, hmin]
      simp
    · rw [mul_comm (1 - overlap E F), mul_div_assoc, div_self (ne_of_gt hR), mul_one]
      simp [resF]

lemma planCost_tvPlan_le (E F : Ens X) :
    planCost (E.canon) (F.canon) unitDist (tvPlan E F) ≤ 1 - overlap E F := by
  have hR0 : 0 ≤ 1 - overlap E F := by linarith [overlap_le_one E F]
  have hterm : ∀ i j : Fin (Fintype.card X),
      tvPlan E F i j * unitDist ((E.canon).pt i) ((F.canon).pt j)
        ≤ resE E F (enum X i) * resF E F (enum X j) / (1 - overlap E F) := by
    intro i j
    by_cases hij : i = j
    · subst hij
      rw [canon_pt, canon_pt, unitDist_self, mul_zero]
      exact div_nonneg (mul_nonneg (resE_nonneg E F _) (resF_nonneg E F _)) hR0
    · have hd : unitDist ((E.canon).pt i) ((F.canon).pt j) ≤ 1 := by
        unfold unitDist; split <;> norm_num
      have hnn : 0 ≤ tvPlan E F i j := (tvPlan_isCoupling E F).nonneg i j
      calc tvPlan E F i j * unitDist ((E.canon).pt i) ((F.canon).pt j)
          ≤ tvPlan E F i j * 1 := mul_le_mul_of_nonneg_left hd hnn
        _ = tvPlan E F i j := by ring
        _ = resE E F (enum X i) * resF E F (enum X j) / (1 - overlap E F) := by
            simp [tvPlan, hij]
  calc planCost (E.canon) (F.canon) unitDist (tvPlan E F)
      ≤ ∑ i : Fin (Fintype.card X), ∑ j : Fin (Fintype.card X),
          resE E F (enum X i) * resF E F (enum X j) / (1 - overlap E F) :=
        Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => hterm i j
    _ ≤ 1 - overlap E F := by
        have hrow : ∀ i : Fin (Fintype.card X),
            ∑ j : Fin (Fintype.card X),
              resE E F (enum X i) * resF E F (enum X j) / (1 - overlap E F)
            = resE E F (enum X i) * (1 - overlap E F) / (1 - overlap E F) := by
          intro i
          rw [← Finset.sum_div, ← Finset.mul_sum, sum_resF_fin]
        rw [Finset.sum_congr rfl fun i _ => hrow i]
        rcases eq_or_lt_of_le hR0 with hR | hR
        · simp [← hR]
        · rw [Finset.sum_congr rfl fun i _ => by
            rw [mul_div_assoc, div_self (ne_of_gt hR), mul_one]]
          rw [sum_resE_fin]

/-- The dual test function that certifies the lower bound: the indicator of the
conformations the model over-populates. -/
noncomputable def overSet (E F : Ens X) : X → ℝ :=
  fun x => if F.prob x < E.prob x then (1 : ℝ) else 0

omit [Fintype X] in
lemma overSet_lipschitz (E F : Ens X) (x y : X) :
    |overSet E F x - overSet E F y| ≤ 1 * unitDist x y := by
  unfold overSet unitDist
  by_cases hxy : x = y
  · simp [hxy]
  · rw [if_neg hxy, one_mul]
    split <;> split <;> norm_num

lemma expect_overSet_diff (E F : Ens X) :
    E.expect (overSet E F) - F.expect (overSet E F) = Ens.ell1 E F / 2 := by
  have hE := E.expect_eq_sum_prob (overSet E F)
  have hF := F.expect_eq_sum_prob (overSet E F)
  rw [hE, hF, ← Finset.sum_sub_distrib]
  have hterm : ∀ x : X, E.prob x * overSet E F x - F.prob x * overSet E F x
      = max (E.prob x - F.prob x) 0 := by
    intro x
    unfold overSet
    by_cases h : F.prob x < E.prob x
    · rw [if_pos h, max_eq_left (by linarith)]; ring
    · rw [if_neg h, max_eq_right (by linarith [not_lt.1 h])]; ring
  rw [Finset.sum_congr rfl fun x _ => hterm x]
  have hmax : ∀ x : X, max (E.prob x - F.prob x) 0
      = (|E.prob x - F.prob x| + (E.prob x - F.prob x)) / 2 := by
    intro x
    rcases le_total (F.prob x) (E.prob x) with h | h
    · rw [max_eq_left (by linarith), abs_of_nonneg (by linarith)]; ring
    · rw [max_eq_right (by linarith), abs_of_nonpos (by linarith)]; ring
  rw [Finset.sum_congr rfl fun x _ => hmax x, ← Finset.sum_div, Finset.sum_add_distrib,
    Finset.sum_sub_distrib, E.sum_prob, F.sum_prob, ← Ens.ell1]
  ring

/-- **Total variation is the transport cost of the discrete geometry.**  For the structural
metric that treats every pair of distinct conformations as maximally different, the
transport distance between two ensembles equals half their `ℓ¹` population distance.  The
`ℓ¹` theory of model capacity is therefore a special case of the transport theory, and the
transport cost against a real structural metric is the same law computed in a geometry that
knows which structures resemble each other. -/
theorem transportCost_unitDist_eq_half_ell1 (E F : Ens X) :
    transportCost unitDist E F = Ens.ell1 E F / 2 := by
  have hcanon : transportCost unitDist E F = transportCost unitDist (E.canon) (F.canon) :=
    transportCost_congr_same unitDist_nonneg unitDist_self unitDist_triangle
      (E.canon_same).symm (F.canon_same).symm
  refine le_antisymm ?_ ?_
  · rw [hcanon, ← one_sub_overlap_eq]
    exact (transportCost_le_of_coupling unitDist_nonneg (tvPlan_isCoupling E F)).trans
      (planCost_tvPlan_le E F)
  · have hdual := transportCost_ge_of_observable (c := unitDist (X := X)) unitDist_nonneg
      (L := 1) one_pos (f := overSet E F) (overSet_lipschitz E F) E F
    rw [div_one, expect_overSet_diff, abs_of_nonneg (by have := Ens.ell1_nonneg E F; linarith)] at hdual
    exact hdual

end TV

end IDR
