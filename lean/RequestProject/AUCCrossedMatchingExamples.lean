/-
# A worked instance of the crossed-matching bound, and why it beats the ceiling

The six-residue, two-protein instance of `AUCCeilingExamples.lean`: each protein holds positives
at `0` and `20` and a negative at `19`.  There are four cross-protein comparisons, with gaps
`19, −1` from protein `0` to protein `1` and `19, −1` back.

* the plain ceiling is `U_within + |between| = 2 + 4 = 6`;
* a single crossed pair (`ceiling_gap_of_crossed`) brings that down to `5`;
* the **maximum crossed matching** has size `4` — the gap `19` on one side crosses the gap `−1` on
  the other, in both directions — and brings it down to `4` (`matching_bound`);
* and `4` is exactly attained, already by the zero bias (`matching_bound_attained`).

So on this instance the crossed-matching bound is tight while the ceiling is not: the reachable
pooled AUC is `1/2`, not the `3/4` the ceiling allows.
-/
import RequestProject.AUCCrossedMatching
import RequestProject.AUCCeilingExamples

set_option autoImplicit false

namespace IDR.GroupedAUC

namespace Crossed

open Finset

/-- The two crossed configurations: the comparison with gap `19` on one side against the
comparison with gap `−1` on the other, in both directions. -/
def pairing : Fin 2 ⊕ Fin 2 → Fin 6 × Fin 6 :=
  Sum.elim ![((0 : Fin 6), (5 : Fin 6)), ((1 : Fin 6), (5 : Fin 6))]
    ![((4 : Fin 6), (2 : Fin 6)), ((3 : Fin 6), (2 : Fin 6))]

theorem pairing_injective : Function.Injective pairing := by decide

theorem pairing_crossed : ∀ i : Fin 2,
    CrossedConfig lab grp sc (pairing (Sum.inl i)) (pairing (Sum.inr i)) := by
  intro i
  fin_cases i
  · exact ⟨by decide, by decide, by decide, by decide, by decide, by norm_num [pairing]⟩
  · exact ⟨by decide, by decide, by decide, by decide, by decide, by norm_num [pairing]⟩

/-- All four cross-protein comparisons are matched: the maximum crossed matching has size `4`. -/
theorem four_le_maxCrossedCard : 4 ≤ maxCrossedCard lab grp sc := by
  have h := maxCrossedCard_ge_of_pairing lab grp sc pairing pairing_injective pairing_crossed
  simpa using h

/-- **The crossed-matching bound on this instance.**  No per-protein bias can push the pooled
statistic above `4`, one pair below what the single-crossed-pair bound gives and two below the
ceiling. -/
theorem matching_bound (b : Fin 2 → ℝ) : U (allPairs lab) (shift grp b sc) ≤ 4 := by
  have h := ceiling_gap_of_max_crossed_matching lab grp sc b
  rw [U_within, card_between] at h
  have h4 : (4 : ℝ) ≤ (maxCrossedCard lab grp sc : ℝ) := by
    exact_mod_cast four_le_maxCrossedCard
  norm_num at h
  linarith

/-- The pooled AUC after any bias is at most `1/2` on this instance. -/
theorem matching_auc_le (b : Fin 2 → ℝ) : auc (allPairs lab) (shift grp b sc) ≤ 1 / 2 := by
  rw [auc, card_all, div_le_div_iff₀ (by norm_num) (by norm_num)]
  have := matching_bound b
  norm_num
  linarith

lemma all_eq : allPairs lab =
    ({(0, 2), (0, 5), (1, 2), (1, 5), (3, 2), (3, 5), (4, 2), (4, 5)} :
      Finset (Fin 6 × Fin 6)) := by decide

/-- **The bound is tight here**: the zero bias already reaches `4`. -/
theorem matching_bound_attained : U (allPairs lab) (shift grp (fun _ => 0) sc) = 4 := by
  rw [all_eq, U, Finset.sum_insert (by decide), Finset.sum_insert (by decide),
    Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_insert (by decide),
    Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_singleton]
  norm_num [kern, shift]

/-- Hence the reachable pooled AUC on this instance is exactly `1/2`. -/
theorem matching_auc_attained : auc (allPairs lab) (shift grp (fun _ => 0) sc) = 1 / 2 := by
  rw [auc, card_all, matching_bound_attained]
  norm_num

end Crossed

end IDR.GroupedAUC
