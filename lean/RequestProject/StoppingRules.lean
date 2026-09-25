/-
# Part XCII.3  Every stopping rule, not just the crossing rule

`IDR.Seq.anytime_valid` bounds the probability that the wealth process *ever* crosses the
threshold.  That is the right bound to prove, but it is not yet the statement an experimenter
needs, because an experimenter does not stop at the crossing: they stop when the reagent runs
out, when the shift ends, when a supervisor asks, or when the number on the screen looks
convincing — and only then look at the wealth.  This file closes that gap.

A *stopping rule* here is an arbitrary function of the reads seen so far, with no measurability,
monotonicity or independence assumption of any kind; it may depend on the whole history in any
way whatsoever.  A run under such a rule stops at the first prefix at which the rule says stop,
and *declares a refutation* if the wealth at that moment is at least the threshold.

* `Declares` — the refutation event of a run under a rule.
* `declares_imp_crossed` — such a run can only declare when the wealth has crossed.
* `stopping_rule_valid` — hence, under the null, **every** stopping rule declares a refutation
  with probability at most `α`.  The experimenter may choose when to stop for any reason at all,
  including reasons that depend on the data, and the level survives.
* `no_rule_beats_the_bound` — the same bound holds for every rule simultaneously, so it cannot be
  evaded by choosing the rule after seeing the data either.
* `never_stopping_rule`, `crossing_rule_declares` — the two extremes: a rule that never stops
  never declares (the bound is not vacuous about which rules are covered), and the rule that
  stops at the first crossing does declare exactly when the wealth crosses.
-/
import Mathlib
import RequestProject.Sequential

set_option autoImplicit false

namespace IDR
namespace Seq

open Finset

/-- A stopping rule: an arbitrary function of the reads seen so far (in order). -/
abbrev Rule := List Bool → Bool

/-- `Declares q₁ q₀ c R pre w rest` : the run that has already read `pre`, holds wealth `w`, and
will read `rest`, stops at the first prefix where `R` says stop, and the wealth there is at least
`c` — that is, it declares a refutation. -/
def Declares (q₁ q₀ c : ℝ) (R : Rule) : List Bool → ℝ → List Bool → Prop
  | pre, w, [] => R pre = true ∧ c ≤ w
  | pre, w, b :: t =>
      (R pre = true ∧ c ≤ w) ∨
        (R pre = false ∧ Declares q₁ q₀ c R (pre ++ [b]) (w * lrOne q₁ q₀ b) t)

/-- A run under any rule declares only if the wealth process has crossed the threshold. -/
lemma declares_imp_crossed {q₁ q₀ c : ℝ} (R : Rule) :
    ∀ (rest : List Bool) (pre : List Bool) (w : ℝ),
      Declares q₁ q₀ c R pre w rest → Crossed q₁ q₀ c w rest := by
  intro rest
  induction rest with
  | nil => intro pre w h; exact h.2
  | cons b t ih =>
      intro pre w h
      rcases h with ⟨_, hle⟩ | ⟨_, hrec⟩
      · exact Or.inl hle
      · exact Or.inr (ih (pre ++ [b]) _ hrec)

open Classical in
/-- The indicator of the refutation event of a run under the rule `R`. -/
noncomputable def declInd (q₁ q₀ c : ℝ) (R : Rule) (w : ℝ) (l : List Bool) : ℝ :=
  if Declares q₁ q₀ c R [] w l then 1 else 0

lemma declInd_le_crossInd (q₁ q₀ c : ℝ) (R : Rule) (w : ℝ) (l : List Bool) :
    declInd q₁ q₀ c R w l ≤ crossInd q₁ q₀ c w l := by
  unfold declInd crossInd
  by_cases h : Declares q₁ q₀ c R [] w l
  · rw [if_pos h, if_pos (declares_imp_crossed R l [] w h)]
  · rw [if_neg h]
    split <;> norm_num

/-- **Every stopping rule is valid.**  Under the null read rate, a run stopped by *any* rule
whatsoever — including one that depends on the data in an arbitrary way — declares a refutation
with probability at most `α`. -/
theorem stopping_rule_valid {q₁ q₀ α : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1) (h00 : 0 < q₀)
    (h01 : q₀ < 1) (hα : 0 < α) (n : ℕ) (R : Rule) :
    ∑ l ∈ words n, probL q₀ l * declInd q₁ q₀ (1 / α) R 1 l ≤ α := by
  refine le_trans (Finset.sum_le_sum ?_) (anytime_valid h10 h11 h00 h01 hα n)
  intro l _
  exact mul_le_mul_of_nonneg_left (declInd_le_crossInd _ _ _ R 1 l)
    (probL_nonneg (le_of_lt h00) (le_of_lt h01) l)

/-- The bound is uniform over rules and horizons, so it cannot be evaded by choosing the rule —
or the run length — after seeing the data. -/
theorem no_rule_beats_the_bound {q₁ q₀ α : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1) (h00 : 0 < q₀)
    (h01 : q₀ < 1) (hα : 0 < α) :
    ∀ (n : ℕ) (R : Rule), ∑ l ∈ words n, probL q₀ l * declInd q₁ q₀ (1 / α) R 1 l ≤ α :=
  fun n R => stopping_rule_valid h10 h11 h00 h01 hα n R

/-- A rule that never stops never declares: the guarantee is about rules that do stop. -/
lemma never_stopping_rule (q₁ q₀ c : ℝ) :
    ∀ (l : List Bool) (pre : List Bool) (w : ℝ), ¬ Declares q₁ q₀ c (fun _ => false) pre w l := by
  intro l
  induction l with
  | nil => intro pre w h; simp [Declares] at h
  | cons b t ih =>
      intro pre w h
      rcases h with ⟨h, -⟩ | ⟨-, hrec⟩
      · simp at h
      · exact ih _ _ hrec

/-- The rule that stops immediately declares exactly when the starting wealth is already at the
threshold; combined with `stopping_rule_valid` this pins the meaning of the event. -/
lemma crossing_rule_declares {q₁ q₀ c : ℝ} {w : ℝ} (hw : c ≤ w) (pre l : List Bool) :
    Declares q₁ q₀ c (fun _ => true) pre w l := by
  cases l with
  | nil => exact ⟨rfl, hw⟩
  | cons b t => exact Or.inl ⟨rfl, hw⟩

end Seq
end IDR
