/-
# Part XC.3  The reporter is a physical object

`RequestProject.NoisyDetection` treats the read-out as a single pair of numbers, a sensitivity
and a specificity.  A real reporter is not that: it is a molecular event with a firing
probability that varies from conformation to conformation, and it is attached to only a fraction
of the molecules in the tube.  Both facts change the contrast the design must be computed with,
and one of them can change its sign.

* `rate_eq` — the positive rate of a state-dependent reporter is `(1-sp) + Σ_{x∈A} w_x(p_x+sp-1)`.
* `contrast_eq` — hence the contrast is `τ·(se_eff + sp - 1)` where `se_eff` is the
  **population-weighted mean** firing probability over the omitted states.  The design must be
  computed with that average, not with a sensitivity measured on a reference construct: two
  reporters with the same nominal sensitivity but different state preferences give different
  contrasts on the same target.
* `contrast_le_of_dark`, `contrast_neg_of_dark` — a reporter blind to part of the omitted set
  does not merely lose signal.  Dark states contribute `-(1-sp)` per unit population, so once the
  dark population exceeds `sp/(1-sp)` times the bright one the contrast is **negative**: the
  count moves the wrong way and the counting test is anti-conservative, failing to reject a
  baseline that is wrong.  This is a design rule, not a caveat: the probe must be validated for
  coverage of the omitted states, not only for affinity.
* `rate_labelled`, `contrast_labelled` — an incompletely labelled sample scales every rate, hence
  the contrast, by the labelled fraction `d`; the molecule count therefore scales as `1/d²`.
* `design_power` — the resulting sample size, with both error probabilities bounded at finite
  `n`, in terms of the one number a realistic design can actually claim to know:
  `Δ = d·τ·(se_eff + sp - 1)`.
* `reporter_physics_laws` — the four statements together.

As everywhere in this development, no number here is a measurement; what is proved is which
average enters the design and what happens when part of the target is invisible to the probe.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Metric
import RequestProject.NoisyDetection

set_option autoImplicit false

namespace IDR
namespace Reporter

open Finset Noisy

variable {X : Type*} [Fintype X] [DecidableEq X]

/-! ## 1. A state-dependent reporter -/

/-- The firing profile of a reporter that fires with conformation-dependent probability `p x` on
the states of `A` and with probability `1 - sp` (a false positive) on every other state. -/
noncomputable def fireOf (A : Finset X) (p : X → ℝ) (sp : ℝ) : X → ℝ :=
  fun x => if x ∈ A then p x else 1 - sp

/-- The population of the omitted set. -/
noncomputable def mass (E : Ens X) (A : Finset X) : ℝ := ∑ x ∈ A, E.prob x

omit [Fintype X] [DecidableEq X] in
lemma mass_nonneg (E : Ens X) (A : Finset X) : 0 ≤ mass E A :=
  Finset.sum_nonneg fun x _ => E.prob_nonneg x

/-- **The observed positive rate of a state-dependent reporter.** -/
theorem rate_eq (E : Ens X) (A : Finset X) (p : X → ℝ) (sp : ℝ) :
    E.expect (fireOf A p sp) = (1 - sp) + ∑ x ∈ A, E.prob x * (p x + sp - 1) := by
  classical
  rw [E.expect_eq_sum_prob]
  have hsplit : ∑ x, E.prob x * fireOf A p sp x
      = ∑ x ∈ A, E.prob x * fireOf A p sp x + ∑ x ∈ Aᶜ, E.prob x * fireOf A p sp x := by
    rw [Finset.sum_add_sum_compl]
  have hA : ∑ x ∈ A, E.prob x * fireOf A p sp x = ∑ x ∈ A, E.prob x * p x :=
    Finset.sum_congr rfl fun x hx => by simp [fireOf, hx]
  have hAc : ∑ x ∈ Aᶜ, E.prob x * fireOf A p sp x = (1 - mass E A) * (1 - sp) := by
    have h : ∑ x ∈ Aᶜ, E.prob x * fireOf A p sp x = ∑ x ∈ Aᶜ, E.prob x * (1 - sp) := by
      refine Finset.sum_congr rfl fun x hx => ?_
      have : x ∉ A := Finset.mem_compl.mp hx
      simp [fireOf, this]
    have hcompl : ∑ x ∈ Aᶜ, E.prob x = 1 - mass E A := by
      have := Finset.sum_add_sum_compl A E.prob
      rw [E.sum_prob] at this
      simp only [mass]
      linarith
    rw [h, ← Finset.sum_mul, hcompl]
  have hexp : ∑ x ∈ A, E.prob x * (p x + sp - 1)
      = ∑ x ∈ A, E.prob x * p x - mass E A * (1 - sp) := by
    have : ∀ x ∈ A, E.prob x * (p x + sp - 1) = E.prob x * p x - E.prob x * (1 - sp) := by
      intro x _; ring
    rw [Finset.sum_congr rfl this, Finset.sum_sub_distrib, ← Finset.sum_mul]
    rfl
  rw [hsplit, hA, hAc, hexp]
  ring

/-- The population-weighted mean firing probability over the omitted states: the sensitivity a
realistic design must use. -/
noncomputable def effSe (E : Ens X) (A : Finset X) (p : X → ℝ) : ℝ :=
  (∑ x ∈ A, E.prob x * p x) / mass E A

/-- **The contrast of a state-dependent reporter** is the omitted population times Youden's
index computed from the *population-weighted mean* sensitivity. -/
theorem contrast_eq (E : Ens X) (A : Finset X) (p : X → ℝ) (sp : ℝ) (hA : 0 < mass E A) :
    E.expect (fireOf A p sp) - (1 - sp) = mass E A * youden (effSe E A p) sp := by
  rw [rate_eq]
  have hexp : ∑ x ∈ A, E.prob x * (p x + sp - 1)
      = (∑ x ∈ A, E.prob x * p x) - mass E A * (1 - sp) := by
    have h : ∀ x ∈ A, E.prob x * (p x + sp - 1) = E.prob x * p x - E.prob x * (1 - sp) := by
      intro x _; ring
    rw [Finset.sum_congr rfl h, Finset.sum_sub_distrib, ← Finset.sum_mul]
    rfl
  have hse : ∑ x ∈ A, E.prob x * p x = mass E A * effSe E A p := by
    have hm : mass E A ≠ 0 := ne_of_gt hA
    rw [effSe, mul_div_cancel₀ _ hm]
  simp only [youden]
  rw [hexp, hse]
  ring

/-! ## 2. Dark states -/

/-- **Dark states subtract.**  If the reporter never fires on the states of `D ⊆ A`, then those
states contribute `-(1-sp)` per unit population, and the contrast is at most
`sp·(τ - δ) - (1-sp)·δ`. -/
theorem contrast_le_of_dark (E : Ens X) {A D : Finset X} {p : X → ℝ} {sp : ℝ} (hDA : D ⊆ A)
    (hp1 : ∀ x, p x ≤ 1) (hdark : ∀ x ∈ D, p x = 0) :
    E.expect (fireOf A p sp) - (1 - sp)
      ≤ sp * (mass E A - mass E D) - (1 - sp) * mass E D := by
  classical
  rw [rate_eq]
  have hsplit : ∑ x ∈ A, E.prob x * (p x + sp - 1)
      = ∑ x ∈ A \ D, E.prob x * (p x + sp - 1) + ∑ x ∈ D, E.prob x * (p x + sp - 1) := by
    rw [Finset.sum_sdiff hDA]
  have hD : ∑ x ∈ D, E.prob x * (p x + sp - 1) = (sp - 1) * mass E D := by
    have h : ∀ x ∈ D, E.prob x * (p x + sp - 1) = E.prob x * (sp - 1) := by
      intro x hx; rw [hdark x hx]; ring
    rw [Finset.sum_congr rfl h, ← Finset.sum_mul]
    simp only [mass]
    ring
  have hbright : ∑ x ∈ A \ D, E.prob x * (p x + sp - 1) ≤ sp * (mass E A - mass E D) := by
    have hmass : mass E A - mass E D = ∑ x ∈ A \ D, E.prob x := by
      have := Finset.sum_sdiff (f := E.prob) hDA
      simp only [mass]
      linarith
    rw [hmass, Finset.mul_sum]
    refine Finset.sum_le_sum fun x _ => ?_
    have := hp1 x
    have hpx := E.prob_nonneg x
    nlinarith
  have := hsplit
  rw [hD] at this
  simp only [mass] at *
  linarith [hbright]

/-- **A reporter blind to too much of the omitted set inverts the test.**  If the dark population
outweighs the bright one in the ratio `sp : (1-sp)`, the contrast is strictly negative: the
positive count *falls* when the omitted states are populated, so the counting test rejects the
disorder-free baseline less often when the baseline is wrong than when it is right. -/
theorem contrast_neg_of_dark (E : Ens X) {A D : Finset X} {p : X → ℝ} {sp : ℝ} (hDA : D ⊆ A)
    (hp1 : ∀ x, p x ≤ 1) (hdark : ∀ x ∈ D, p x = 0)
    (hbad : sp * (mass E A - mass E D) < (1 - sp) * mass E D) :
    E.expect (fireOf A p sp) - (1 - sp) < 0 := by
  have h := contrast_le_of_dark (sp := sp) E hDA hp1 hdark
  linarith

/-! ## 3. Incomplete labelling -/

/-- The firing profile when only a fraction `d` of the molecules carries the reporter and an
unlabelled molecule never fires. -/
noncomputable def labelled (d : ℝ) (fire : X → ℝ) : X → ℝ := fun x => d * fire x

omit [Fintype X] [DecidableEq X] in
/-- Incomplete labelling scales every rate by the labelled fraction. -/
theorem rate_labelled (E : Ens X) (d : ℝ) (fire : X → ℝ) :
    E.expect (labelled d fire) = d * E.expect fire :=
  E.expect_smul d fire

/-- …hence it scales the contrast by the labelled fraction, so the molecule count of the design
scales as `1/d²`. -/
theorem contrast_labelled (E : Ens X) (A : Finset X) (p : X → ℝ) (sp d : ℝ) (hA : 0 < mass E A) :
    E.expect (labelled d (fireOf A p sp)) - d * (1 - sp)
      = d * (mass E A * youden (effSe E A p) sp) := by
  rw [rate_labelled, ← contrast_eq E A p sp hA]
  ring

/-! ## 4. The design contrast, and the sample size it dictates -/

/-- **The contrast a realistic design may claim**: labelled fraction times omitted population
times Youden's index of the *population-weighted mean* sensitivity. -/
noncomputable def designContrast (E : Ens X) (A : Finset X) (p : X → ℝ) (sp d : ℝ) : ℝ :=
  d * (mass E A * youden (effSe E A p) sp)

/-- **The realistic sample size.**  Write `q₀ = d(1-sp)` for the rate on a disorder-free system
and `q₁` for the rate on the real one.  If the design contrast is positive, then
`⌈1/(α·Δ²)⌉` scored molecules hold both error probabilities of the midpoint counting test at
`α`, where `Δ` is that contrast — with the state-dependent sensitivity and the labelled fraction
already folded in. -/
theorem design_power (E : Ens X) (A : Finset X) (p : X → ℝ) {sp d alpha : ℝ}
    (hA : 0 < mass E A) (hd0 : 0 ≤ d) (hd1 : d ≤ 1) (hsp0 : 0 ≤ sp) (hsp1 : sp ≤ 1)
    (hp0 : ∀ x, 0 ≤ p x) (hp1 : ∀ x, p x ≤ 1)
    (hpos : 0 < designContrast E A p sp d) (ha : 0 < alpha) {n : ℕ}
    (hn : samplesFor alpha (designContrast E A p sp d) ≤ n) :
    ∑ s ∈ accepts n (n * (d * (1 - sp) + E.expect (labelled d (fireOf A p sp))) / 2),
        recProb (E.expect (labelled d (fireOf A p sp))) s ≤ alpha ∧
      ∑ s ∈ rejects n (n * (d * (1 - sp) + E.expect (labelled d (fireOf A p sp))) / 2),
        recProb (d * (1 - sp)) s ≤ alpha := by
  have hgap : E.expect (labelled d (fireOf A p sp)) - d * (1 - sp)
      = designContrast E A p sp d := contrast_labelled E A p sp d hA
  have h0 : 0 ≤ d * (1 - sp) := mul_nonneg hd0 (by linarith)
  have hfire0 : ∀ x, 0 ≤ fireOf A p sp x := by
    intro x
    by_cases hx : x ∈ A
    · simpa [fireOf, hx] using hp0 x
    · simp only [fireOf, hx, if_false]; linarith
  have hfire1 : ∀ x, fireOf A p sp x ≤ 1 := by
    intro x
    by_cases hx : x ∈ A
    · simpa [fireOf, hx] using hp1 x
    · simp only [fireOf, hx, if_false]; linarith
  have hrate1 : E.expect (labelled d (fireOf A p sp)) ≤ 1 := by
    rw [rate_labelled]
    have hle : E.expect (fireOf A p sp) ≤ 1 := by
      have := E.expect_mono (f := fireOf A p sp) (g := fun _ => 1) hfire1
      simpa using this
    have hge : 0 ≤ E.expect (fireOf A p sp) := E.expect_nonneg hfire0
    nlinarith
  have hlt : d * (1 - sp) < E.expect (labelled d (fireOf A p sp)) := by linarith [hgap ▸ hpos]
  refine samplesFor_spec h0 hrate1 hlt ha ?_
  rw [hgap]
  exact hn

/-- **Part XC.3 in one statement.**  For a state-dependent reporter on an incompletely labelled
sample: the contrast is the labelled fraction times the omitted population times Youden's index
of the population-weighted mean sensitivity; dark states subtract from it and can invert its
sign; and at positive contrast the counting test's two error probabilities are bounded by `α` at
the computed sample size. -/
theorem reporter_physics_laws (E : Ens X) (A : Finset X) (p : X → ℝ) {sp d alpha : ℝ}
    (hA : 0 < mass E A) (hd0 : 0 ≤ d) (hd1 : d ≤ 1) (hsp0 : 0 ≤ sp) (hsp1 : sp ≤ 1)
    (hp0 : ∀ x, 0 ≤ p x) (hp1 : ∀ x, p x ≤ 1)
    (hpos : 0 < designContrast E A p sp d) (ha : 0 < alpha) {n : ℕ}
    (hn : samplesFor alpha (designContrast E A p sp d) ≤ n) :
    (E.expect (fireOf A p sp) - (1 - sp) = mass E A * youden (effSe E A p) sp) ∧
    (E.expect (labelled d (fireOf A p sp)) - d * (1 - sp) = designContrast E A p sp d) ∧
    (∀ D : Finset X, D ⊆ A → (∀ x ∈ D, p x = 0) →
      E.expect (fireOf A p sp) - (1 - sp)
        ≤ sp * (mass E A - mass E D) - (1 - sp) * mass E D) ∧
    (∑ s ∈ accepts n (n * (d * (1 - sp) + E.expect (labelled d (fireOf A p sp))) / 2),
        recProb (E.expect (labelled d (fireOf A p sp))) s ≤ alpha ∧
      ∑ s ∈ rejects n (n * (d * (1 - sp) + E.expect (labelled d (fireOf A p sp))) / 2),
        recProb (d * (1 - sp)) s ≤ alpha) :=
  ⟨contrast_eq E A p sp hA,
    contrast_labelled E A p sp d hA,
    fun _ hDA hdark => contrast_le_of_dark E hDA hp1 hdark,
    design_power E A p hA hd0 hd1 hsp0 hsp1 hp0 hp1 hpos ha hn⟩

end Reporter
end IDR
