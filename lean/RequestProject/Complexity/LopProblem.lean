/-
# The weighted linear ordering problem as a decision problem

`lopProblem` packages the optimisation `lopOpt` of `AUCHardness.lean` as a decision problem: given
a weight matrix and a target, is there a linear order of the items collecting at least the target
weight?  Weights are measured in unary in the instance size, which is the encoding under which the
reduction to score tables stays polynomial.

`rankValue` restates the objective in terms of an injective ranking function instead of a
permutation; `rankValue_le_lopOpt` and `exists_rank_eq_lopOpt` show the two descriptions have the
same optimum.  Rankings are much easier to construct than permutations, and that is how the
reduction from independent set builds its linear order.
-/
import RequestProject.Complexity.IndependentSet
import RequestProject.AUCHardnessDecision

set_option autoImplicit false

namespace IDR.Complexity

open Finset IDR.GroupedAUC.Hardness

/-- An instance of the weighted linear ordering problem. -/
structure LopInst where
  /-- The number of items. -/
  K : ℕ
  /-- The weight matrix. -/
  W : Fin K → Fin K → ℕ
  /-- The target value. -/
  target : ℕ

/-- The size of a linear ordering instance: items, total weight (in unary) and target. -/
def LopInst.size (I : LopInst) : ℕ := I.K + (∑ k : Fin I.K, ∑ l : Fin I.K, I.W k l) + I.target

/-- **The weighted linear ordering problem.** -/
def lopProblem : Problem where
  Inst := LopInst
  size := LopInst.size
  Yes := fun I => I.target ≤ lopOpt I.W

/-! ## Linear orders as rankings -/

section Rank

variable {K : ℕ}

/-- The value of a ranking: a pair `(k, l)` pays `W k l` when `k` is ranked above `l`. -/
def rankValue (W : Fin K → Fin K → ℕ) (r : Fin K → ℕ) : ℕ :=
  ∑ k : Fin K, ∑ l : Fin K, if r l < r k then W k l else 0

/-- Every injective ranking is realised by a permutation, so its value is a lower bound for the
linear ordering optimum. -/
theorem rankValue_le_lopOpt (W : Fin K → Fin K → ℕ) (r : Fin K → ℕ)
    (hinj : Function.Injective r) : rankValue W r ≤ lopOpt W := by
  classical
  set s := Tuple.sort r with hs
  have hmono : StrictMono (r ∘ ⇑s) :=
    (Tuple.monotone_sort r).strictMono_of_injective (hinj.comp s.injective)
  refine le_trans (le_of_eq ?_) (lopValue_le_lopOpt W s⁻¹)
  unfold rankValue lopValue
  refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => ?_
  have hiff : (r l < r k) ↔ (s⁻¹ l < s⁻¹ k) := by
    have hl : r l = (r ∘ ⇑s) (s⁻¹ l) := by simp
    have hk : r k = (r ∘ ⇑s) (s⁻¹ k) := by simp
    rw [hl, hk, hmono.lt_iff_lt]
  exact if_congr hiff rfl rfl

/-- The optimum is attained by an injective ranking. -/
theorem exists_rank_eq_lopOpt (W : Fin K → Fin K → ℕ) :
    ∃ r : Fin K → ℕ, Function.Injective r ∧ rankValue W r = lopOpt W := by
  obtain ⟨pi, hpi⟩ := exists_lopValue_eq W
  refine ⟨fun k => (pi k : ℕ), fun a b h => pi.injective (Fin.val_injective h), ?_⟩
  rw [← hpi]
  unfold rankValue lopValue
  exact Finset.sum_congr rfl fun k _ =>
    Finset.sum_congr rfl fun l _ => if_congr Fin.lt_def.symm rfl rfl

end Rank

end IDR.Complexity
