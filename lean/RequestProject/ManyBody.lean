/-
# Part XXV.1  Many-body solvation: what a pairwise-additive potential cannot represent

The Hamiltonian of Part XXIV is *pairwise additive*: its solvent-averaged energy is a sum of
functions of the individual interatomic distances.  That is an assumption, and this file
prices it exactly.

For a triple of atoms let `W(a,b,c)` be the potential of mean force as a function of the
three separations.  Additivity means `W a b c = f a + g b + h c`.  The obstruction is the
mixed second difference

  `mixed W a a' b b' c = W a' b' c - W a' b c - W a b' c + W a b c`,

which vanishes identically for every additive `W` (`mixed_eq_zero_of_additive`) and is
strictly negative for the cooperative desolvation form `-exp(-a) * exp(-b)`
(`mixed_cooperative_neg`).  Hence `not_additive_cooperative`: no pairwise decomposition of
any kind -- no choice of pair potentials, however fitted -- reproduces a cooperative
three-body potential of mean force.  Explicit solvent, or an explicit three-body term, is a
different model, not a better parameterisation of this one.
-/
import Mathlib

namespace IDR

namespace ManyBody

open Real

/-- A three-body potential of mean force, as a function of the three separations. -/
abbrev PMF3 := ℝ → ℝ → ℝ → ℝ

/-- `W` is pairwise additive if it splits into one function of each separation. -/
def IsPairwiseAdditive (W : PMF3) : Prop :=
  ∃ f g h : ℝ → ℝ, ∀ a b c : ℝ, W a b c = f a + g b + h c

/-- The mixed second difference in the first two separations. -/
def mixed (W : PMF3) (a a' b b' c : ℝ) : ℝ :=
  W a' b' c - W a' b c - W a b' c + W a b c

/-- **Additivity has no mixed difference.** -/
theorem mixed_eq_zero_of_additive {W : PMF3} (hW : IsPairwiseAdditive W)
    (a a' b b' c : ℝ) : mixed W a a' b b' c = 0 := by
  obtain ⟨f, g, h, hfgh⟩ := hW
  unfold mixed
  rw [hfgh, hfgh, hfgh, hfgh]
  ring

/-- A cooperative desolvation potential of mean force: the cost of removing water from the
first contact is reduced when the second contact is already formed.  This is the standard
shape of a hydrophobic three-body term. -/
noncomputable def cooperative : PMF3 := fun a b _ => -(Real.exp (-a) * Real.exp (-b))

/-- The cooperative form has a strictly nonzero mixed difference. -/
theorem mixed_cooperative_neg (c : ℝ) : mixed cooperative 0 1 0 1 c < 0 := by
  unfold mixed cooperative
  have h1 : Real.exp (-(1:ℝ)) < 1 := by
    rw [show (1:ℝ) = Real.exp 0 by simp]
    exact Real.exp_lt_exp.mpr (by norm_num)
  have h0 : (0:ℝ) < Real.exp (-(1:ℝ)) := Real.exp_pos _
  have hsq : 0 < (Real.exp (-(1:ℝ)) - 1) ^ 2 := by nlinarith
  simp only [neg_zero, Real.exp_zero, mul_one, one_mul]
  nlinarith [hsq]

/-- **A cooperative three-body potential of mean force is not pairwise additive.**  No
choice of pair potentials reproduces it: the failure is not a fitting error but a
representability obstruction. -/
theorem not_additive_cooperative : ¬ IsPairwiseAdditive cooperative := by
  intro hW
  have := mixed_eq_zero_of_additive hW 0 1 0 1 0
  have hlt := mixed_cooperative_neg 0
  rw [this] at hlt
  exact lt_irrefl 0 hlt

/-- **The obstruction is generic**, not an artefact of one form: any `W` with a nonzero
mixed difference at even a single quintuple of arguments is outside the additive model. -/
theorem not_additive_of_mixed_ne {W : PMF3} {a a' b b' c : ℝ}
    (h : mixed W a a' b b' c ≠ 0) : ¬ IsPairwiseAdditive W :=
  fun hW => h (mixed_eq_zero_of_additive hW a a' b b' c)

/-- **The best additive approximation still errs.**  For every pairwise-additive `V` the
supremum error against a cooperative `W` is at least a quarter of the mixed difference: the
non-additivity cannot be absorbed anywhere. -/
theorem additive_error_lower_bound {W V : PMF3} (hV : IsPairwiseAdditive V)
    (a a' b b' c : ℝ) :
    |mixed W a a' b b' c| / 4 ≤
      max (max |W a' b' c - V a' b' c| |W a' b c - V a' b c|)
        (max |W a b' c - V a b' c| |W a b c - V a b c|) := by
  set M := max (max |W a' b' c - V a' b' c| |W a' b c - V a' b c|)
    (max |W a b' c - V a b' c| |W a b c - V a b c|) with hM
  have h1 : |W a' b' c - V a' b' c| ≤ M := le_trans (le_max_left _ _) (le_max_left _ _)
  have h2 : |W a' b c - V a' b c| ≤ M := le_trans (le_max_right _ _) (le_max_left _ _)
  have h3 : |W a b' c - V a b' c| ≤ M := le_trans (le_max_left _ _) (le_max_right _ _)
  have h4 : |W a b c - V a b c| ≤ M := le_trans (le_max_right _ _) (le_max_right _ _)
  have hVz : mixed V a a' b b' c = 0 := mixed_eq_zero_of_additive hV a a' b b' c
  have hsplit : mixed W a a' b b' c
      = (W a' b' c - V a' b' c) - (W a' b c - V a' b c) - (W a b' c - V a b' c)
        + (W a b c - V a b c) + mixed V a a' b b' c := by
    unfold mixed at hVz ⊢
    ring
  rw [hsplit, hVz, add_zero, div_le_iff₀ (by norm_num : (0:ℝ) < 4)]
  obtain ⟨l1, u1⟩ := abs_le.mp h1
  obtain ⟨l2, u2⟩ := abs_le.mp h2
  obtain ⟨l3, u3⟩ := abs_le.mp h3
  obtain ⟨l4, u4⟩ := abs_le.mp h4
  exact abs_le.mpr ⟨by linarith, by linarith⟩

end ManyBody

end IDR
