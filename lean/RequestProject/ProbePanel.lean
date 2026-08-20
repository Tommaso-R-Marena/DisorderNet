/-
# Part XC.5  Combining probes: when a panel helps and when it destroys the contrast

Part XC prices a study in terms of one number, the rate contrast `τ·J`, and shows that a poor
reporter is expensive in molecules and, past a point, fatal.  The experimentalist's response is
to combine probes: label two positions, add a second crosslink, score a molecule positive if
*any* probe fires (an OR panel) or only if *all* of them do (an AND panel).  This file works out
what that does to `J`, and the answer contains a warning.

Independent probes are modelled the usual way, by multiplying the per-probe miss probabilities;
that product formula is the definition of the combined reporter here, and independence is an
assumption about the experiment, not a theorem.

* `youden_or_eq`, `youden_and_eq` — closed forms.  Youden's index of an OR panel is
  `∏ sp_i − ∏ (1 − se_i)`, and of an AND panel `∏ se_i − ∏ (1 − sp_i)`.  For a single probe both
  reduce to `se + sp − 1`, so this is the natural generalisation of the contrast.
* `youden_or_ge_of_clean` — adding a probe with no false positives can only help.
* `youden_or_lt_iff` — and the exact criterion for when adding a second probe *hurts*:
  `sp₁·(1 − sp₂) > (1 − se₁)·se₂`, i.e. when the false positives it brings outweigh the true
  positives it adds.
* `youden_or_le_pow`, `youden_and_le_pow` — the warning.  With `k` identical imperfect probes the
  OR panel's index is at most `sp^k` and the AND panel's at most `se^k`: both vanish
  exponentially in the number of probes.  Piling probes onto a panel is not a way to buy
  contrast; the molecule count `⌈1/(α(τJ)²)⌉` then grows exponentially in `k`.
* `panel_samples_le` — the design number for a panel, obtained from the general theory by
  substituting the panel's index.

The design rule that follows is specific: combine probes only when each added probe is
essentially free of false positives on the omitted states, and otherwise spend the effort on one
clean probe rather than on several dirty ones.
-/
import Mathlib
import RequestProject.NoisyDetection

set_option autoImplicit false

namespace IDR
namespace Panel

open Finset Noisy

/-! ## 1. The two panel rules -/

variable {k : ℕ}

/-- Sensitivity of the OR panel: it fires unless every probe misses. -/
noncomputable def orSe (se : Fin k → ℝ) : ℝ := 1 - ∏ i, (1 - se i)

/-- Specificity of the OR panel: it is silent only if every probe is silent. -/
noncomputable def orSp (sp : Fin k → ℝ) : ℝ := ∏ i, sp i

/-- Sensitivity of the AND panel: every probe must fire. -/
noncomputable def andSe (se : Fin k → ℝ) : ℝ := ∏ i, se i

/-- Specificity of the AND panel: a false positive needs every probe to misfire. -/
noncomputable def andSp (sp : Fin k → ℝ) : ℝ := 1 - ∏ i, (1 - sp i)

/-- **Youden's index of an OR panel.** -/
theorem youden_or_eq (se sp : Fin k → ℝ) :
    youden (orSe se) (orSp sp) = (∏ i, sp i) - ∏ i, (1 - se i) := by
  simp only [youden, orSe, orSp]; ring

/-- **Youden's index of an AND panel.** -/
theorem youden_and_eq (se sp : Fin k → ℝ) :
    youden (andSe se) (andSp sp) = (∏ i, se i) - ∏ i, (1 - sp i) := by
  simp only [youden, andSe, andSp]; ring

/-- With one probe both panels are that probe. -/
theorem youden_or_single (se sp : Fin 1 → ℝ) :
    youden (orSe se) (orSp sp) = youden (se 0) (sp 0) := by
  rw [youden_or_eq]
  simp [youden]
  ring

/-! ## 2. Two probes: when the panel helps -/

/-- Youden's index of a two-probe OR panel, in the probes' own numbers. -/
theorem youden_or_pair (se₁ sp₁ se₂ sp₂ : ℝ) :
    youden (orSe ![se₁, se₂]) (orSp ![sp₁, sp₂]) = sp₁ * sp₂ - (1 - se₁) * (1 - se₂) := by
  simp only [youden_or_eq, Fin.prod_univ_two]
  norm_num

/-- **A probe with no false positives can only help.**  If the second probe never fires outside
the omitted states, the OR panel's index is at least the first probe's. -/
theorem youden_or_ge_of_clean {se₁ sp₁ se₂ : ℝ} (hse₂ : 0 ≤ se₂) (hse₁ : se₁ ≤ 1) :
    youden se₁ sp₁ ≤ youden (orSe ![se₁, se₂]) (orSp ![sp₁, 1]) := by
  rw [youden_or_pair]
  simp only [youden]
  nlinarith

/-- **And the exact criterion for when it hurts**: the second probe degrades the panel precisely
when the false positives it contributes outweigh the true positives it adds. -/
theorem youden_or_lt_iff (se₁ sp₁ se₂ sp₂ : ℝ) :
    youden (orSe ![se₁, se₂]) (orSp ![sp₁, sp₂]) < youden se₁ sp₁
      ↔ (1 - se₁) * se₂ < sp₁ * (1 - sp₂) := by
  rw [youden_or_pair]
  simp only [youden]
  constructor <;> intro h <;> nlinarith

/-! ## 3. Many probes: both rules decay exponentially -/

/-- **The OR panel's index is at most `∏ sp_i`.**  With `k` identical probes of specificity
`sp < 1` it is at most `sp^k`: an OR panel of many imperfect probes has almost no contrast. -/
theorem youden_or_le_pow {se sp : Fin k → ℝ} (hse : ∀ i, se i ≤ 1) :
    youden (orSe se) (orSp sp) ≤ ∏ i, sp i := by
  rw [youden_or_eq]
  have : 0 ≤ ∏ i, (1 - se i) := Finset.prod_nonneg fun i _ => by linarith [hse i]
  linarith

/-- **The AND panel's index is at most `∏ se_i`**, so it too decays exponentially in the number
of probes. -/
theorem youden_and_le_pow {se sp : Fin k → ℝ} (hsp : ∀ i, sp i ≤ 1) :
    youden (andSe se) (andSp sp) ≤ ∏ i, se i := by
  rw [youden_and_eq]
  have : 0 ≤ ∏ i, (1 - sp i) := Finset.prod_nonneg fun i _ => by linarith [hsp i]
  linarith

/-- Identical probes: the OR panel's index is at most `sp^k`. -/
theorem youden_or_identical_le {se sp : ℝ} (hse : se ≤ 1) :
    youden (orSe (fun _ : Fin k => se)) (orSp (fun _ : Fin k => sp)) ≤ sp ^ k := by
  have h := youden_or_le_pow (se := fun _ : Fin k => se) (sp := fun _ : Fin k => sp)
    (fun _ => hse)
  simpa using h

/-! ## 4. The design number for a panel -/

/-- **The panel's sample size.**  Substituting the panel's Youden index into the design of
`RequestProject.NoisyDetection`: a panel with index `J` needs `⌈1/(α·(τJ)²)⌉` molecules, and
since the index of a many-probe panel is exponentially small in the number of probes, so is the
reciprocal of that requirement. -/
theorem panel_samples_le {tau alpha : ℝ} {se sp : Fin k → ℝ} (htau : 0 < tau)
    (hJ : 0 < youden (orSe se) (orSp sp)) (ha : 0 < alpha) (hse : ∀ i, se i ≤ 1) :
    samplesFor alpha (tau * ∏ i, sp i)
      ≤ samplesFor alpha (tau * youden (orSe se) (orSp sp)) :=
  samplesFor_antitone ha (mul_pos htau hJ)
    (mul_le_mul_of_nonneg_left (youden_or_le_pow hse) htau.le)

/-- **Part XC.5 in one statement**: closed forms for both panel rules, the exact criterion for a
second probe to help, and the exponential decay of a many-probe panel's contrast. -/
theorem probe_panel_laws (se sp : Fin k → ℝ) (se₁ sp₁ se₂ sp₂ : ℝ) (hse : ∀ i, se i ≤ 1)
    (hsp : ∀ i, sp i ≤ 1) :
    youden (orSe se) (orSp sp) = (∏ i, sp i) - ∏ i, (1 - se i) ∧
    youden (andSe se) (andSp sp) = (∏ i, se i) - ∏ i, (1 - sp i) ∧
    (youden (orSe ![se₁, se₂]) (orSp ![sp₁, sp₂]) < youden se₁ sp₁
      ↔ (1 - se₁) * se₂ < sp₁ * (1 - sp₂)) ∧
    youden (orSe se) (orSp sp) ≤ ∏ i, sp i ∧
    youden (andSe se) (andSp sp) ≤ ∏ i, se i :=
  ⟨youden_or_eq se sp, youden_and_eq se sp, youden_or_lt_iff se₁ sp₁ se₂ sp₂,
    youden_or_le_pow hse, youden_and_le_pow hsp⟩

end Panel
end IDR
