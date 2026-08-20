/-
# Fragment pipelines VI: imperfect fragments, and the bias–variance split

Everything so far assumed each fragment was fitted *exactly*.  In practice a fragment panel is
a simulation or a measurement, and it is wrong by some amount.  This file shows how those
errors travel through the assembly.

`CondClose δ p p'` says that two chains have head conditionals agreeing to within `δ` in `ℓ¹`
at every seam, and leading panels agreeing to within `δ`.  The theorem
`mproj_l1_stability` says the assembled models are then within `(n+1)·δ` of each other: the
error of a fragment pipeline built from imperfect panels grows only **linearly** in the number
of fragments, with no amplification along the chain.

Combined with the additive design law this gives the clean split (`fitted_pipeline_error`)

    total error  ≤  √(2 · total seam information)  +  (number of fragments) · (panel error),

a bias–variance decomposition for modular ensemble construction: the first term is the price
of cutting the chain at all, paid even with perfect data, and the second is the price of
imperfect fragment data, paid once per fragment.  The two move in opposite directions as the
cuts are made denser — more cuts raise the first term (`merged_seam_le_sum`) and shrink the
fragments the second term is measured on — which is exactly why an optimal fragment length
exists.
-/
import Mathlib
import RequestProject.ChainPipelineLimits

namespace RequestProject.ChainPipeline

open Finset IDR.Pinsker RequestProject.Modular

universe u

variable {A : Type u} [Fintype A] [Nonempty A]

/-! ## Panels that agree to within `δ` -/

/-- Two chains whose fragment panels agree to within `δ`: at every seam the conditional law of
the block before the seam, given the seam, differs by at most `δ` in `ℓ¹`, and the trailing
two-block panel differs by at most `δ`. -/
def CondClose (δ : ℝ) : ∀ {n : ℕ}, (Blocks A n → ℝ) → (Blocks A n → ℝ) → Prop
  | 0, p, p' => ∑ a : A, |p a - p' a| ≤ δ
  | 1, p, p' => ∑ x : Blocks A 1, |p x - p' x| ≤ δ
  | (_ + 2), p, p' =>
      (∀ b : A, ∑ a : A, |margXY (cur p) a b / margY (cur p) b
        - margXY (cur p') a b / margY (cur p') b| ≤ δ) ∧ CondClose δ (dropFirst p) (dropFirst p')

omit [Nonempty A] in
lemma condClose_succ2 {δ : ℝ} {n : ℕ} (p p' : Blocks A (n + 2) → ℝ) :
    CondClose δ p p' ↔
      ((∀ b : A, ∑ a : A, |margXY (cur p) a b / margY (cur p) b
        - margXY (cur p') a b / margY (cur p') b| ≤ δ)
        ∧ CondClose δ (dropFirst p) (dropFirst p')) := Iff.rfl

/-! ## Errors travel linearly along the chain -/

/-- **Stability of assembly.**  If two chains have fragment panels agreeing to within `δ`,
their pipeline models are within `(n+1)·δ` in `ℓ¹`.  A panel error committed on one fragment
is felt once, not amplified: the assembled error is the sum of the fragment errors. -/
theorem mproj_l1_stability {δ : ℝ} : ∀ {n : ℕ} {p p' : Blocks A (n + 1) → ℝ},
    (∀ x, 0 < p x) → (∀ x, 0 < p' x) → ∑ x, p x = 1 → ∑ x, p' x = 1 → CondClose δ p p' →
    ∑ x : Blocks A (n + 1), |mproj p x - mproj p' x| ≤ (n + 1 : ℝ) * δ
  | 0, p, p', _, _, _, _, h => by simpa using h
  | (n + 1), p, p', hp, hp', hs, hs', h => by
      obtain ⟨hcond, htail⟩ := h
      have hδ : 0 ≤ δ :=
        le_trans (Finset.sum_nonneg fun a _ => abs_nonneg _) (hcond (Classical.arbitrary A))
      have hdp : ∀ y, 0 < dropFirst p y := fun y => dropFirst_pos hp y
      have hdp' : ∀ y, 0 < dropFirst p' y := fun y => dropFirst_pos hp' y
      have hds : ∑ y : Blocks A (n + 1), dropFirst p y = 1 := by rw [sum_dropFirst]; exact hs
      have hds' : ∑ y : Blocks A (n + 1), dropFirst p' y = 1 := by rw [sum_dropFirst]; exact hs'
      have hIH : ∑ y : Blocks A (n + 1), |mproj (dropFirst p) y - mproj (dropFirst p') y|
          ≤ (n + 1 : ℝ) * δ := mproj_l1_stability hdp hdp' hds hds' htail
      set M : Blocks A (n + 1) → ℝ := mproj (dropFirst p) with hM
      set M' : Blocks A (n + 1) → ℝ := mproj (dropFirst p') with hM'
      set c : A → A → ℝ := fun a b => margXY (cur p) a b / margY (cur p) b with hc
      set c' : A → A → ℝ := fun a b => margXY (cur p') a b / margY (cur p') b with hc'
      have hMpos : ∀ y, 0 < M y := fun y => mproj_pos hdp y
      have hMsum : ∑ y : Blocks A (n + 1), M y = 1 := sum_mproj hdp hds
      have hc'sum : ∀ b : A, ∑ a : A, c' a b = 1 := by
        intro b
        rw [hc']
        simp only
        rw [← Finset.sum_div, ← margY_eq_sum_margXY (cur p') b,
          div_self (ne_of_gt (margY_cur_pos hp' b))]
      have hc'nonneg : ∀ a b : A, 0 ≤ c' a b := fun a b =>
        le_of_lt (div_pos (margXY_cur_pos hp' a b) (margY_cur_pos hp' b))
      have hterm : ∀ (a : A) (y : Blocks A (n + 1)),
          |mproj p ((a, y) : Blocks A (n + 2)) - mproj p' ((a, y) : Blocks A (n + 2))|
            ≤ |c a y.1 - c' a y.1| * M y + c' a y.1 * |M y - M' y| := by
        intro a y
        obtain ⟨b, w⟩ := y
        have h1 : mproj p ((a, b, w) : Blocks A (n + 2)) = c a b * M (b, w) := mproj_succ2 p a b w
        have h2 : mproj p' ((a, b, w) : Blocks A (n + 2)) = c' a b * M' (b, w) :=
          mproj_succ2 p' a b w
        rw [h1, h2]
        have hsplit : c a b * M (b, w) - c' a b * M' (b, w)
            = (c a b - c' a b) * M (b, w) + c' a b * (M (b, w) - M' (b, w)) := by ring
        calc |c a b * M (b, w) - c' a b * M' (b, w)|
            ≤ |(c a b - c' a b) * M (b, w)| + |c' a b * (M (b, w) - M' (b, w))| := by
              rw [hsplit]; exact abs_add_le _ _
          _ = |c a b - c' a b| * M (b, w) + c' a b * |M (b, w) - M' (b, w)| := by
              rw [abs_mul, abs_mul, abs_of_pos (hMpos (b, w)), abs_of_nonneg (hc'nonneg a b)]
      have hstep : ∑ x : Blocks A (n + 2), |mproj p x - mproj p' x|
          ≤ ∑ a : A, ∑ y : Blocks A (n + 1),
              (|c a y.1 - c' a y.1| * M y + c' a y.1 * |M y - M' y|) := by
        rw [sum_blocks_succ (fun x : Blocks A (n + 2) => |mproj p x - mproj p' x|)]
        exact Finset.sum_le_sum fun a _ => Finset.sum_le_sum fun y _ => hterm a y
      have hsplit2 : ∑ a : A, ∑ y : Blocks A (n + 1),
            (|c a y.1 - c' a y.1| * M y + c' a y.1 * |M y - M' y|)
          = (∑ y : Blocks A (n + 1), (∑ a : A, |c a y.1 - c' a y.1|) * M y)
            + ∑ y : Blocks A (n + 1), (∑ a : A, c' a y.1) * |M y - M' y| := by
        rw [← Finset.sum_add_distrib, Finset.sum_comm]
        refine Finset.sum_congr rfl fun y _ => ?_
        rw [Finset.sum_add_distrib, Finset.sum_mul, Finset.sum_mul]
      have hfirst : (∑ y : Blocks A (n + 1), (∑ a : A, |c a y.1 - c' a y.1|) * M y) ≤ δ := by
        have hb : ∀ y : Blocks A (n + 1), (∑ a : A, |c a y.1 - c' a y.1|) * M y ≤ δ * M y :=
          fun y => mul_le_mul_of_nonneg_right (hcond y.1) (le_of_lt (hMpos y))
        calc ∑ y : Blocks A (n + 1), (∑ a : A, |c a y.1 - c' a y.1|) * M y
            ≤ ∑ y : Blocks A (n + 1), δ * M y := Finset.sum_le_sum fun y _ => hb y
          _ = δ := by rw [← Finset.mul_sum, hMsum, mul_one]
      have hsecond : (∑ y : Blocks A (n + 1), (∑ a : A, c' a y.1) * |M y - M' y|)
          ≤ (n + 1 : ℝ) * δ := by
        have hb : ∀ y : Blocks A (n + 1), (∑ a : A, c' a y.1) * |M y - M' y| = |M y - M' y| :=
          fun y => by rw [hc'sum y.1, one_mul]
        rw [Finset.sum_congr rfl fun y _ => hb y]
        exact hIH
      have : ∑ x : Blocks A (n + 2), |mproj p x - mproj p' x| ≤ δ + (n + 1 : ℝ) * δ := by
        rw [hsplit2] at hstep
        linarith
      push_cast
      linarith

/-- **The bias–variance split for a fragment pipeline.**  A pipeline assembled from panels that
are accurate to `δ` is within `√(2·total seam information) + (number of fragments)·δ` of the
truth: the first term is the price of cutting the chain, paid even with perfect fragments, and
the second is the price of imperfect fragments, paid once per fragment and never amplified. -/
theorem fitted_pipeline_error {δ : ℝ} {n : ℕ} {p p' : Blocks A (n + 1) → ℝ} (hp : ∀ x, 0 < p x)
    (hp' : ∀ x, 0 < p' x) (hs : ∑ x, p x = 1) (hs' : ∑ x, p' x = 1) (h : CondClose δ p p') :
    ∑ x : Blocks A (n + 1), |p x - mproj p' x|
      ≤ Real.sqrt (2 * seamInfoTotal p) + (n + 1 : ℝ) * δ := by
  have h1 : ∑ x : Blocks A (n + 1), |p x - mproj p x| ≤ Real.sqrt (2 * seamInfoTotal p) :=
    pipeline_l1_le_sqrt_seamInfoTotal hp hs
  have h2 : ∑ x : Blocks A (n + 1), |mproj p x - mproj p' x| ≤ (n + 1 : ℝ) * δ :=
    mproj_l1_stability hp hp' hs hs' h
  have htri : ∑ x : Blocks A (n + 1), |p x - mproj p' x|
      ≤ (∑ x : Blocks A (n + 1), |p x - mproj p x|)
        + ∑ x : Blocks A (n + 1), |mproj p x - mproj p' x| := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun x _ => ?_
    calc |p x - mproj p' x| = |(p x - mproj p x) + (mproj p x - mproj p' x)| := by ring_nf
      _ ≤ |p x - mproj p x| + |mproj p x - mproj p' x| := abs_add_le _ _
  linarith

end RequestProject.ChainPipeline
