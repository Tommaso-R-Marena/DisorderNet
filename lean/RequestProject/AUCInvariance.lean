/-
# Theorem 1 — the per-protein bias term is invisible to the within-protein AUC

A multi-task head that adds a learned per-protein constant `b k` to every residue of protein `k`
changes no same-protein comparison: the Mann–Whitney kernel sees only differences of scores, and
inside one protein the constant cancels.  Hence

* `U_within_shift`, `auc_within_shift_invariant` — **`AUC_within` is exactly invariant** under
  `s ↦ s + b∘grp`, for every `b : G → ℝ` and every score field `s`;
* `U_pooled_shift_diff`, `auc_pooled_shift_diff` — the whole effect of the bias on the pooled
  statistic is carried by the cross-protein pairs: the change in the pooled statistic equals the
  change in the between statistic, and in AUC terms the pooled change is `w_between` times the
  between change;
* `auc_within_strictMono_invariant` — the same holds, more generally, for any per-protein
  *strictly monotone* recalibration, not just an additive constant;
* `shift_const` — a bias that is the same for every protein changes nothing at all.

So the bias term is safe by construction, not merely by unit test: whatever it does, it can only
move the between-protein part of the benchmark number.
-/
import RequestProject.AUCCore

set_option autoImplicit false

namespace IDR.GroupedAUC

open Finset

section Shift

variable {I G : Type*} [Fintype I] [DecidableEq G]

/-- Adding a per-protein constant `b k` to every residue of protein `k`. -/
def shift (grp : I → G) (b : G → ℝ) (s : I → ℝ) : I → ℝ := fun i => s i + b (grp i)

/-- Applying a per-protein strictly monotone recalibration. -/
def recal (grp : I → G) (f : G → ℝ → ℝ) (s : I → ℝ) : I → ℝ := fun i => f (grp i) (s i)

/-- **Theorem 1 (statistic form).** The within-protein Mann–Whitney statistic is exactly invariant
under per-protein biases. -/
theorem U_within_shift (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ) :
    U (withinPairs lab grp) (shift grp b s) = U (withinPairs lab grp) s := by
  unfold U
  refine Finset.sum_congr rfl (fun q hq => ?_)
  have hq' : grp q.1 = grp q.2 := (Finset.mem_filter.mp hq).2
  simp only [shift, hq']
  exact kern_add_const _ _ _

/-- **Theorem 1.** `AUC_within` is exactly invariant under per-protein biases. -/
theorem auc_within_shift_invariant (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ) :
    auc (withinPairs lab grp) (shift grp b s) = auc (withinPairs lab grp) s := by
  unfold auc
  rw [U_within_shift]

/-- The same invariance for an arbitrary per-protein strictly monotone recalibration. -/
theorem auc_within_strictMono_invariant (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    (f : G → ℝ → ℝ) (hf : ∀ k, StrictMono (f k)) :
    auc (withinPairs lab grp) (recal grp f s) = auc (withinPairs lab grp) s := by
  unfold auc U
  congr 1
  refine Finset.sum_congr rfl (fun q hq => ?_)
  have hq' : grp q.1 = grp q.2 := (Finset.mem_filter.mp hq).2
  simp only [recal, hq']
  unfold kern
  have h1 : f (grp q.2) (s q.2) < f (grp q.2) (s q.1) ↔ s q.2 < s q.1 :=
    (hf (grp q.2)).lt_iff_lt
  have h2 : f (grp q.2) (s q.1) = f (grp q.2) (s q.2) ↔ s q.1 = s q.2 :=
    (hf (grp q.2)).injective.eq_iff
  simp [h1, h2]

/-- **Theorem 1 (between form).** Every effect of a per-protein bias on the pooled statistic is an
effect on the cross-protein pairs. -/
theorem U_pooled_shift_diff (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ) :
    U (allPairs lab) (shift grp b s) - U (allPairs lab) s
      = U (betweenPairs lab grp) (shift grp b s) - U (betweenPairs lab grp) s := by
  rw [U_split lab grp, U_split lab grp, U_within_shift]
  ring

/-- In AUC terms: the pooled AUC moves by `w_between` times the move in the between AUC. -/
theorem auc_pooled_shift_diff (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ)
    (hb : (betweenPairs lab grp).card ≠ 0) :
    auc (allPairs lab) (shift grp b s) - auc (allPairs lab) s
      = ((betweenPairs lab grp).card / (allPairs lab).card)
          * (auc (betweenPairs lab grp) (shift grp b s) - auc (betweenPairs lab grp) s) := by
  have hbc : ((betweenPairs lab grp).card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hb
  have hac : ((allPairs lab).card : ℝ) ≠ 0 := by
    have : (betweenPairs lab grp).card ≤ (allPairs lab).card :=
      Finset.card_filter_le _ _
    have h0 : 0 < (allPairs lab).card := lt_of_lt_of_le (Nat.pos_of_ne_zero hb) this
    exact Nat.cast_ne_zero.mpr h0.ne'
  unfold auc
  rw [div_sub_div_same, div_sub_div_same, U_pooled_shift_diff lab grp s b]
  field_simp

omit [Fintype I] [DecidableEq G] in
/-- A bias that is constant across proteins changes nothing whatsoever. -/
theorem shift_const (grp : I → G) (s : I → ℝ) (c : ℝ) (X : Finset (I × I)) :
    U X (shift grp (fun _ => c) s) = U X s := by
  unfold U shift
  exact Finset.sum_congr rfl (fun q _ => kern_add_const _ _ _)

end Shift

end IDR.GroupedAUC
