/-
# Fragment pipelines IV: longer fragments are never worse

A pipeline is specified by where the chain is cut.  Cutting more often makes each fragment
smaller — cheaper to simulate, cheaper to measure, cheaper to fit — but the previous files
showed that every cut is paid for in information.  This file proves the trade-off has no
hidden reward on the accuracy side: **merging two adjacent seams into one longer overlap never
increases the price.**

Formally, for a chain of at least four blocks, compare two pipelines:

* the *fine* one, which cuts at block `1` and again at block `2`, paying
  `cmi (cur p) + cmi (cur (dropFirst p))`;
* the *coarse* one, which treats blocks `0` and `1` as a single fragment and cuts only at
  block `2`, paying `cmi (pairCur p)`.

`merged_seam_le_sum` shows the coarse price never exceeds the fine one, and
`coarse_pipeline_le_fine` upgrades this to the whole chain: replacing any two adjacent cuts by
the single longer overlap they span lowers (or leaves unchanged) the total seam information.
Iterating, the price is monotone in how finely the chain is cut: **there is no free
refinement, and the shortest fragment a pipeline uses sets the floor on its error.**

The proof is a variational one, and does not need a chain rule for conditional mutual
information.  The two-seam glued model `twoSeamModel p` is an explicit ensemble that is
conditionally independent in the *coarse* sense (`condIndep_pairCur_twoSeamModel`), so the
optimality theorem for a single seam bounds the coarse price by the relative entropy of the
truth from it — and `klG_twoSeamModel` evaluates that relative entropy as exactly the sum of
the two fine seam informations.
-/
import Mathlib
import RequestProject.ChainPipelineLimits

namespace RequestProject.ChainPipeline

open Finset IDR.Pinsker RequestProject.Modular

universe u

variable {A : Type u} [Fintype A] [Nonempty A]

/-! ## The coarse view: two blocks read as one fragment -/

omit [Fintype A] [Nonempty A] in
/-- The chain of at least four blocks, read with the first two blocks merged into a single
fragment: head `(x₀,x₁)`, seam `x₂`, tail the rest. -/
def pairCur {n : ℕ} (p : Blocks A (n + 3) → ℝ) : (A × A) → A → Blocks A n → ℝ :=
  fun ab c w => p (ab.1, ab.2, c, w)

omit [Nonempty A] in
lemma sum_blocks_succ3 {n : ℕ} (f : Blocks A (n + 3) → ℝ) :
    ∑ x : Blocks A (n + 3), f x
      = ∑ a : A, ∑ b : A, ∑ c : A, ∑ w : Blocks A n, f (a, b, c, w) := by
  rw [sum_blocks_succ]
  exact Finset.sum_congr rfl fun a _ => sum_blocks_succ2 _

omit [Nonempty A] in
lemma sum_blocks_pair {n : ℕ} (f : Blocks A (n + 3) → ℝ) :
    ∑ x : Blocks A (n + 3), f x
      = ∑ ab : A × A, ∑ c : A, ∑ w : Blocks A n, f (ab.1, ab.2, c, w) := by
  rw [sum_blocks_succ3, Fintype.sum_prod_type
    (f := fun ab : A × A => ∑ c : A, ∑ w : Blocks A n, f (ab.1, ab.2, c, w))]

/-! ## The two-seam glued model -/

/-- The model a pipeline produces when it cuts at both of the first two seams and is exact
beyond them: the head conditional across the first seam, times the glued model of the chain
that starts at the second block. -/
noncomputable def twoSeamModel {n : ℕ} (p : Blocks A (n + 3) → ℝ) : Blocks A (n + 3) → ℝ :=
  fun x => (margXY (cur p) x.1 x.2.1 / margY (cur p) x.2.1)
    * glue (cur (dropFirst p)) x.2.1 x.2.2.1 x.2.2.2

omit [Nonempty A] in
lemma twoSeamModel_apply {n : ℕ} (p : Blocks A (n + 3) → ℝ) (a b c : A) (w : Blocks A n) :
    twoSeamModel p (a, b, c, w)
      = (margXY (cur p) a b / margY (cur p) b) * glue (cur (dropFirst p)) b c w := rfl

lemma twoSeamModel_pos {n : ℕ} {p : Blocks A (n + 3) → ℝ} (hp : ∀ x, 0 < p x)
    (x : Blocks A (n + 3)) : 0 < twoSeamModel p x := by
  obtain ⟨a, b, c, w⟩ := x
  rw [twoSeamModel_apply]
  refine mul_pos (div_pos (margXY_cur_pos hp a b) (margY_cur_pos hp b)) ?_
  exact glue_pos (fun x y z => le_of_lt (dropFirst_pos hp _))
    (dropFirst_pos hp ((b, c, w) : Blocks A (n + 2)))

lemma sum_twoSeamModel {n : ℕ} {p : Blocks A (n + 3) → ℝ} (hp : ∀ x, 0 < p x)
    (hs : ∑ x, p x = 1) : ∑ x : Blocks A (n + 3), twoSeamModel p x = 1 := by
  have hdp : ∀ x, 0 ≤ dropFirst p x := fun x => le_of_lt (dropFirst_pos hp x)
  have hdps : ∑ b : A, ∑ c : A, ∑ w : Blocks A n, dropFirst p (b, c, w) = 1 := by
    rw [← sum_blocks_succ2 (dropFirst p), sum_dropFirst]
    exact hs
  have hglue : ∑ b : A, ∑ c : A, ∑ w : Blocks A n, glue (cur (dropFirst p)) b c w = 1 :=
    sum_glue (p := cur (dropFirst p)) (fun x y z => hdp _) hdps
  rw [sum_blocks_succ3]
  have hstep : ∀ b : A, ∑ a : A, (margXY (cur p) a b / margY (cur p) b) = 1 := by
    intro b
    rw [← Finset.sum_div, ← margY_eq_sum_margXY (cur p) b,
      div_self (ne_of_gt (margY_cur_pos hp b))]
  calc ∑ a : A, ∑ b : A, ∑ c : A, ∑ w : Blocks A n, twoSeamModel p (a, b, c, w)
      = ∑ b : A, ∑ c : A, ∑ w : Blocks A n,
          (∑ a : A, margXY (cur p) a b / margY (cur p) b) * glue (cur (dropFirst p)) b c w := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun b _ => ?_
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun c _ => ?_
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun w _ => ?_
        rw [Finset.sum_mul]
        exact Finset.sum_congr rfl fun a _ => twoSeamModel_apply p a b c w
    _ = 1 := by simp only [hstep, one_mul]; exact hglue

omit [Nonempty A] in
/-- The two-seam model is conditionally independent **in the coarse view**: it is an ensemble
a pipeline that cuts only at the second seam can produce. -/
lemma condIndep_pairCur_twoSeamModel {n : ℕ} (p : Blocks A (n + 3) → ℝ) :
    CondIndep (pairCur (twoSeamModel p)) := by
  have hform : pairCur (twoSeamModel p) = fun (ab : A × A) (c : A) (w : Blocks A n) =>
      ((margXY (cur p) ab.1 ab.2 / margY (cur p) ab.2)
        * (margXY (cur (dropFirst p)) ab.2 c / margY (cur (dropFirst p)) c))
      * margYZ (cur (dropFirst p)) c w := by
    funext ab c w
    show twoSeamModel p (ab.1, ab.2, c, w) = _
    rw [twoSeamModel_apply, glue]
    ring
  rw [hform]
  exact condIndep_of_product _ _

/-! ## The price of the two-seam model -/

/-- Pointwise, the log-likelihood ratio of the truth to the two-seam model splits into the two
seam terms. -/
lemma log_ratio_split_twoSeam {n : ℕ} {p : Blocks A (n + 3) → ℝ} (hp : ∀ x, 0 < p x)
    (a b c : A) (w : Blocks A n) :
    p (a, b, c, w) * Real.log (p (a, b, c, w) / twoSeamModel p (a, b, c, w))
      = p (a, b, c, w) * Real.log (p (a, b, c, w) / glue (cur p) a b (c, w))
        + p (a, b, c, w) * Real.log (dropFirst p (b, c, w)
            / glue (cur (dropFirst p)) b c w) := by
  have hxy : 0 < margXY (cur p) a b := margXY_cur_pos hp a b
  have hy : 0 < margY (cur p) b := margY_cur_pos hp b
  have hcp : 0 < margXY (cur p) a b / margY (cur p) b := div_pos hxy hy
  have ht : 0 < dropFirst p ((b, c, w) : Blocks A (n + 2)) := dropFirst_pos hp _
  have hg : 0 < glue (cur (dropFirst p)) b c w :=
    glue_pos (fun x y z => le_of_lt (dropFirst_pos hp _)) ht
  have hpx : 0 < p (a, b, c, w) := hp _
  have hglue : glue (cur p) a b (c, w)
      = (margXY (cur p) a b / margY (cur p) b) * dropFirst p ((b, c, w) : Blocks A (n + 2)) := by
    rw [glue, dropFirst_eq_margYZ]
    field_simp
  rw [twoSeamModel_apply, hglue, ← mul_add]
  congr 1
  rw [Real.log_div (ne_of_gt hpx) (by positivity), Real.log_div (ne_of_gt hpx) (by positivity),
    Real.log_div (ne_of_gt ht) (ne_of_gt hg), Real.log_mul (ne_of_gt hcp) (ne_of_gt hg),
    Real.log_mul (ne_of_gt hcp) (ne_of_gt ht)]
  ring

/-- **The two-seam model costs exactly the two seam informations.** -/
theorem klG_twoSeamModel {n : ℕ} {p : Blocks A (n + 3) → ℝ} (hp : ∀ x, 0 < p x) :
    klG p (twoSeamModel p) = cmi (cur p) + cmi (cur (dropFirst p)) := by
  have hsplit : klG p (twoSeamModel p)
      = (∑ a : A, ∑ b : A, ∑ c : A, ∑ w : Blocks A n,
          p (a, b, c, w) * Real.log (p (a, b, c, w) / glue (cur p) a b (c, w)))
        + ∑ a : A, ∑ b : A, ∑ c : A, ∑ w : Blocks A n,
          p (a, b, c, w) * Real.log (dropFirst p (b, c, w)
            / glue (cur (dropFirst p)) b c w) := by
    rw [klG, sum_blocks_succ3, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun w _ => log_ratio_split_twoSeam hp a b c w
  have hfirst : (∑ a : A, ∑ b : A, ∑ c : A, ∑ w : Blocks A n,
      p (a, b, c, w) * Real.log (p (a, b, c, w) / glue (cur p) a b (c, w))) = cmi (cur p) := by
    rw [cmi_eq_sum_log_ratio (p := cur p) (fun a b z => le_of_lt (hp _))]
    exact Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ =>
      (sum_blocks_succ (fun z : Blocks A (n + 1) =>
        cur p a b z * Real.log (cur p a b z / glue (cur p) a b z))).symm
  have hsecond : (∑ a : A, ∑ b : A, ∑ c : A, ∑ w : Blocks A n,
      p (a, b, c, w) * Real.log (dropFirst p (b, c, w) / glue (cur (dropFirst p)) b c w))
      = cmi (cur (dropFirst p)) := by
    rw [cmi_eq_sum_log_ratio (p := cur (dropFirst p)) (fun b c w => le_of_lt (dropFirst_pos hp _))]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun w _ => ?_
    rw [show cur (dropFirst p) b c w = dropFirst p ((b, c, w) : Blocks A (n + 2)) from rfl,
      dropFirst, Finset.sum_mul]
  rw [hsplit, hfirst, hsecond]

/-! ## Merging two seams never costs more -/

/-- **No free refinement, locally.**  The information price of cutting the chain once, at the
second block, with the first two blocks kept together in one fragment, is at most the price of
cutting at both of the first two blocks.  Making fragments shorter can only cost more. -/
theorem merged_seam_le_sum {n : ℕ} {p : Blocks A (n + 3) → ℝ} (hp : ∀ x, 0 < p x)
    (hs : ∑ x, p x = 1) :
    cmi (pairCur p) ≤ cmi (cur p) + cmi (cur (dropFirst p)) := by
  have hps : ∑ ab : A × A, ∑ c : A, ∑ w : Blocks A n, pairCur p ab c w = 1 := by
    simp only [pairCur]
    rw [← sum_blocks_pair p]; exact hs
  have hqs : ∑ ab : A × A, ∑ c : A, ∑ w : Blocks A n, pairCur (twoSeamModel p) ab c w = 1 := by
    simp only [pairCur]
    rw [← sum_blocks_pair (twoSeamModel p)]
    exact sum_twoSeamModel hp hs
  have hbound := no_modular_model_beats_seam_information (p := pairCur p)
    (q := pairCur (twoSeamModel p)) (fun ab c w => le_of_lt (hp _))
    (fun ab c w => twoSeamModel_pos hp _) hps hqs (condIndep_pairCur_twoSeamModel p)
  have hkl : (∑ ab : A × A, ∑ c : A, ∑ w : Blocks A n,
      pairCur p ab c w * Real.log (pairCur p ab c w / pairCur (twoSeamModel p) ab c w))
      = klG p (twoSeamModel p) := by
    simp only [pairCur]
    rw [klG, sum_blocks_pair
      (fun x : Blocks A (n + 3) => p x * Real.log (p x / twoSeamModel p x))]
  rw [hkl, klG_twoSeamModel hp] at hbound
  exact hbound

/-- **No free refinement, for the whole chain.**  The pipeline that keeps the first two blocks
in a single fragment, and is otherwise identical, has total price at most that of the pipeline
that cuts at both.  Iterating over the seams: the total seam information of a pipeline is
monotone in how finely the chain is cut, so among pipelines the shortest fragment sets the
floor on the error. -/
theorem coarse_pipeline_le_fine {n : ℕ} {p : Blocks A (n + 3) → ℝ} (hp : ∀ x, 0 < p x)
    (hs : ∑ x, p x = 1) :
    cmi (pairCur p) + seamInfoTotal (dropFirst (dropFirst p)) ≤ seamInfoTotal p := by
  have h := merged_seam_le_sum hp hs
  rw [seamInfoTotal_succ2 p, seamInfoTotal_succ2 (dropFirst p)]
  linarith

end RequestProject.ChainPipeline
