/-
# Fragment pipelines I: the whole chain, seam by seam

A disordered region of any realistic length is never modelled in one piece.  The pipeline
that every practical method uses is: cut the chain into a sequence of overlapping fragments,
build (or measure) an ensemble for each fragment, and join consecutive fragments through the
block of chain they share.  `RequestProject.ModularGluing` settled the cost of **one** such
cut.  This file settles the cost of the whole pipeline, for a chain of arbitrarily many
blocks and arbitrarily many seams.

A conformation of a chain of `n+1` blocks is an element of `Blocks A n`, the `n+1`-fold
product of the block-state type `A`.  A pipeline model is built left to right: the first
fragment supplies the law of block `0` given block `1`, the next fragment the law of block
`1` given block `2`, and so on.  `mproj p` is the model the pipeline returns when every
fragment is fitted perfectly to the truth `p` — the *Markov projection* of `p` along the
chain.

The results:

* `mproj_pos`, `sum_mproj` — the pipeline output is a genuine ensemble.
* `mproj_dropFirst` — every fragment panel of the pipeline model reproduces the truth's,
  seam by seam: the pipeline is exactly right on everything a single fragment can see.
* `klG_mproj_eq_seamInfoTotal` — **the additive design law.**  The relative entropy of the
  truth from the pipeline model is *exactly* the sum of the conditional mutual informations
  at the individual seams.  The price of a fragment pipeline is the sum of the prices of its
  cuts; there are no cross terms, no interference between seams, and no cancellation.
* `seamInfoTotal_nonneg`, `mproj_eq_self_iff_isPipeline`, `seamInfoTotal_eq_zero_iff` — the
  price is nonnegative and vanishes exactly on the chains a pipeline can represent, i.e. the
  chains that are Markov along the sequence.

The consequences for design — optimality, Pinsker, and the extensivity of the error in chain
length — are in `RequestProject.ChainPipelineLimits`.
-/
import Mathlib
import RequestProject.ModularOptimality

namespace RequestProject.ChainPipeline

open Finset IDR.Pinsker RequestProject.Modular

universe u

/-! ## Chains of blocks -/

/-- A conformation of a chain of `n+1` coarse blocks, each in state type `A`. -/
def Blocks (A : Type u) : ℕ → Type u
  | 0 => A
  | (n + 1) => A × Blocks A n

instance blocksFintype (A : Type u) [Fintype A] : ∀ n, Fintype (Blocks A n)
  | 0 => ‹Fintype A›
  | (n + 1) => @instFintypeProd _ _ ‹Fintype A› (blocksFintype A n)

instance blocksNonempty (A : Type u) [Nonempty A] : ∀ n, Nonempty (Blocks A n)
  | 0 => ‹Nonempty A›
  | (n + 1) => @instNonemptyProd _ _ ‹Nonempty A› (blocksNonempty A n)

variable {A : Type u} [Fintype A] [Nonempty A]

omit [Nonempty A] in
/-- Summation over a one-block chain is summation over the block. -/
lemma sum_blocks_zero (f : Blocks A 0 → ℝ) : ∑ x : Blocks A 0, f x = ∑ a : A, f a := rfl

omit [Nonempty A] in
/-- Summation over a chain splits off the first block. -/
lemma sum_blocks_succ {n : ℕ} (f : Blocks A (n + 1) → ℝ) :
    ∑ x : Blocks A (n + 1), f x = ∑ a : A, ∑ w : Blocks A n, f (a, w) :=
  Fintype.sum_prod_type (f := (f : A × Blocks A n → ℝ))

omit [Nonempty A] in
/-- Summation over a chain of at least three blocks, split into head, seam and tail. -/
lemma sum_blocks_succ2 {n : ℕ} (f : Blocks A (n + 2) → ℝ) :
    ∑ x : Blocks A (n + 2), f x = ∑ a : A, ∑ b : A, ∑ w : Blocks A n, f (a, b, w) := by
  rw [sum_blocks_succ]
  exact Finset.sum_congr rfl fun a _ => sum_blocks_succ _

/-- The chain seen from the second block onwards: the law of the remaining blocks after the
first has been marginalised away.  This is the panel the next fragment in the pipeline is
fitted to. -/
def dropFirst {n : ℕ} (p : Blocks A (n + 1) → ℝ) : Blocks A n → ℝ := fun w => ∑ a, p (a, w)

/-- The head/seam/tail view of a chain of at least three blocks, in the curried form used by
`RequestProject.ModularGluing`. -/
def cur {n : ℕ} (p : Blocks A (n + 2) → ℝ) : A → A → Blocks A n → ℝ := fun a b w => p (a, b, w)

omit [Nonempty A] in
lemma dropFirst_eq_margYZ {n : ℕ} (p : Blocks A (n + 2) → ℝ) (b : A) (w : Blocks A n) :
    dropFirst p (b, w) = margYZ (cur p) b w := rfl

omit [Nonempty A] in
lemma dropFirst_nonneg {n : ℕ} {p : Blocks A (n + 1) → ℝ} (hp : ∀ x, 0 ≤ p x) (w : Blocks A n) :
    0 ≤ dropFirst p w :=
  Finset.sum_nonneg fun _ _ => hp _

lemma dropFirst_pos {n : ℕ} {p : Blocks A (n + 1) → ℝ} (hp : ∀ x, 0 < p x) (w : Blocks A n) :
    0 < dropFirst p w :=
  Finset.sum_pos (fun _ _ => hp _) ⟨Classical.arbitrary A, Finset.mem_univ _⟩

omit [Nonempty A] in
lemma sum_dropFirst {n : ℕ} (p : Blocks A (n + 1) → ℝ) :
    ∑ w : Blocks A n, dropFirst p w = ∑ x : Blocks A (n + 1), p x := by
  rw [sum_blocks_succ, Finset.sum_comm]
  rfl

/-! ## The pipeline model -/

/-- **The Markov projection of a chain ensemble**: the ensemble a fragment pipeline returns
when each fragment is fitted exactly to the truth.  Reading left to right, the first fragment
contributes the conditional law of block `0` given the seam block `1`, and the rest of the
chain is handled recursively by the same rule. -/
noncomputable def mproj : ∀ {n : ℕ}, (Blocks A n → ℝ) → (Blocks A n → ℝ)
  | 0, p => p
  | 1, p => p
  | (_ + 2), p => fun x =>
      (margXY (cur p) x.1 x.2.1 / margY (cur p) x.2.1) * mproj (dropFirst p) x.2

omit [Nonempty A] in
@[simp] lemma mproj_zero (p : Blocks A 0 → ℝ) : mproj p = p := rfl

omit [Nonempty A] in
@[simp] lemma mproj_one (p : Blocks A 1 → ℝ) : mproj p = p := rfl

omit [Nonempty A] in
lemma mproj_succ2 {n : ℕ} (p : Blocks A (n + 2) → ℝ) (a b : A) (w : Blocks A n) :
    mproj p (a, b, w) =
      (margXY (cur p) a b / margY (cur p) b) * mproj (dropFirst p) (b, w) := rfl

/-- **The total seam information of a chain**: the sum, over the seams of the pipeline, of
the conditional mutual information between what lies before the seam and what lies after it,
given the seam itself. -/
noncomputable def seamInfoTotal : ∀ {n : ℕ}, (Blocks A n → ℝ) → ℝ
  | 0, _ => 0
  | 1, _ => 0
  | (_ + 2), p => cmi (cur p) + seamInfoTotal (dropFirst p)

omit [Nonempty A] in
@[simp] lemma seamInfoTotal_zero (p : Blocks A 0 → ℝ) : seamInfoTotal p = 0 := rfl

omit [Nonempty A] in
@[simp] lemma seamInfoTotal_one (p : Blocks A 1 → ℝ) : seamInfoTotal p = 0 := rfl

omit [Nonempty A] in
lemma seamInfoTotal_succ2 {n : ℕ} (p : Blocks A (n + 2) → ℝ) :
    seamInfoTotal p = cmi (cur p) + seamInfoTotal (dropFirst p) := rfl

/-- The ensembles a fragment pipeline can express: those that are conditionally independent
across every seam, i.e. Markov along the chain of blocks. -/
def IsPipeline : ∀ {n : ℕ}, (Blocks A n → ℝ) → Prop
  | 0, _ => True
  | 1, _ => True
  | (_ + 2), m => CondIndep (cur m) ∧ IsPipeline (dropFirst m)

omit [Nonempty A] in
lemma isPipeline_succ2 {n : ℕ} (m : Blocks A (n + 2) → ℝ) :
    IsPipeline m ↔ (CondIndep (cur m) ∧ IsPipeline (dropFirst m)) := Iff.rfl

/-! ## The pipeline output is an ensemble -/

lemma margXY_cur_pos {n : ℕ} {p : Blocks A (n + 2) → ℝ} (hp : ∀ x, 0 < p x) (a b : A) :
    0 < margXY (cur p) a b :=
  Finset.sum_pos (fun _ _ => hp _) ⟨Classical.arbitrary (Blocks A n), Finset.mem_univ _⟩

lemma margY_cur_pos {n : ℕ} {p : Blocks A (n + 2) → ℝ} (hp : ∀ x, 0 < p x) (b : A) :
    0 < margY (cur p) b :=
  Finset.sum_pos (fun a _ => margXY_cur_pos hp a b) ⟨Classical.arbitrary A, Finset.mem_univ _⟩

/-- The pipeline model is strictly positive whenever the truth is. -/
theorem mproj_pos : ∀ {n : ℕ} {p : Blocks A n → ℝ}, (∀ x, 0 < p x) → ∀ x, 0 < mproj p x
  | 0, _, hp, x => hp x
  | 1, _, hp, x => hp x
  | (_ + 2), p, hp, x => by
      rw [show x = (x.1, x.2.1, x.2.2) from rfl, mproj_succ2]
      exact mul_pos (div_pos (margXY_cur_pos hp _ _) (margY_cur_pos hp _))
        (mproj_pos (fun w => dropFirst_pos hp w) _)

/-- The pipeline model is normalised: it is a genuine ensemble of the full-length chain. -/
theorem sum_mproj : ∀ {n : ℕ} {p : Blocks A n → ℝ}, (∀ x, 0 < p x) → ∑ x, p x = 1 →
    ∑ x, mproj p x = 1
  | 0, _, _, hs => hs
  | 1, _, _, hs => hs
  | (n + 2), p, hp, hs => by
      have hIH : ∑ w : Blocks A (n + 1), mproj (dropFirst p) w = 1 := by
        refine sum_mproj (fun w => dropFirst_pos hp w) ?_
        rw [sum_dropFirst]; exact hs
      have hsplit : ∑ x : Blocks A (n + 2), mproj p x
          = ∑ b : A, ∑ w : Blocks A n,
              (∑ a : A, margXY (cur p) a b / margY (cur p) b) * mproj (dropFirst p) (b, w) := by
        rw [sum_blocks_succ2]
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun b _ => ?_
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun w _ => ?_
        rw [Finset.sum_mul]
        exact Finset.sum_congr rfl fun a _ => mproj_succ2 p a b w
      have hcond : ∀ b : A, (∑ a : A, margXY (cur p) a b / margY (cur p) b) = 1 := by
        intro b
        rw [← Finset.sum_div, ← margY_eq_sum_margXY (cur p) b,
          div_self (ne_of_gt (margY_cur_pos hp b))]
      rw [hsplit]
      simp only [hcond, one_mul]
      rw [← sum_blocks_succ (fun w : Blocks A (n + 1) => mproj (dropFirst p) w)]
      exact hIH

/-! ## The pipeline reproduces every fragment panel -/

/-- Marginalising the first block out of the pipeline model returns the pipeline model of the
truncated chain: the pipeline is exactly right on every panel a downstream fragment sees. -/
theorem mproj_dropFirst {n : ℕ} {p : Blocks A (n + 2) → ℝ} (hp : ∀ x, 0 < p x)
    (w : Blocks A (n + 1)) : dropFirst (mproj p) w = mproj (dropFirst p) w := by
  obtain ⟨b, v⟩ := w
  have : dropFirst (mproj p) (b, v)
      = (∑ a : A, margXY (cur p) a b / margY (cur p) b) * mproj (dropFirst p) (b, v) := by
    rw [dropFirst, Finset.sum_mul]
    exact Finset.sum_congr rfl fun a _ => mproj_succ2 p a b v
  rw [this, ← Finset.sum_div, ← margY_eq_sum_margXY (cur p) b,
    div_self (ne_of_gt (margY_cur_pos hp b)), one_mul]

/-! ## The additive design law -/

/-- The one-seam step of the design law: relative to the pipeline model, the log-likelihood
ratio splits into the seam term and the ratio of the truncated chain to its own pipeline
model. -/
lemma log_ratio_split {n : ℕ} {p : Blocks A (n + 2) → ℝ} (hp : ∀ x, 0 < p x)
    (a b : A) (w : Blocks A n) :
    p (a, b, w) * Real.log (p (a, b, w) / mproj p (a, b, w))
      = p (a, b, w) * Real.log (p (a, b, w) / glue (cur p) a b w)
        + p (a, b, w) * Real.log (dropFirst p (b, w) / mproj (dropFirst p) (b, w)) := by
  have hxy : 0 < margXY (cur p) a b := margXY_cur_pos hp a b
  have hy : 0 < margY (cur p) b := margY_cur_pos hp b
  have ht : 0 < dropFirst p (b, w) := dropFirst_pos hp (b, w)
  have hM : 0 < mproj (dropFirst p) (b, w) := mproj_pos (fun v => dropFirst_pos hp v) (b, w)
  have hpx : 0 < p (a, b, w) := hp _
  have hglue : glue (cur p) a b w = (margXY (cur p) a b / margY (cur p) b) * dropFirst p (b, w) := by
    rw [glue, dropFirst_eq_margYZ]
    field_simp
  have hmp : mproj p (a, b, w)
      = (margXY (cur p) a b / margY (cur p) b) * mproj (dropFirst p) (b, w) := mproj_succ2 p a b w
  have hc : 0 < margXY (cur p) a b / margY (cur p) b := div_pos hxy hy
  rw [hglue, hmp, ← mul_add]
  congr 1
  rw [Real.log_div (ne_of_gt hpx) (by positivity), Real.log_div (ne_of_gt hpx) (by positivity),
    Real.log_div (ne_of_gt ht) (ne_of_gt hM), Real.log_mul (ne_of_gt hc) (ne_of_gt hM),
    Real.log_mul (ne_of_gt hc) (ne_of_gt ht)]
  ring

/-- **The additive design law for fragment pipelines.**  The relative entropy of the true
full-length ensemble from the model a fragment pipeline produces is *exactly* the sum of the
conditional mutual informations across the individual seams.

Every cut is paid for once, at the price set by the information the two sides of that cut
share once the shared block is known, and the total bill is the plain sum: seams do not
interfere, and no cut can be subsidised by another. -/
theorem klG_mproj_eq_seamInfoTotal : ∀ {n : ℕ} {p : Blocks A n → ℝ}, (∀ x, 0 < p x) →
    klG p (mproj p) = seamInfoTotal p
  | 0, p, hp => by
      simp only [mproj_zero, seamInfoTotal_zero, klG]
      exact Finset.sum_eq_zero fun x _ => by
        rw [div_self (ne_of_gt (hp x)), Real.log_one, mul_zero]
  | 1, p, hp => by
      simp only [mproj_one, seamInfoTotal_one, klG]
      exact Finset.sum_eq_zero fun x _ => by
        rw [div_self (ne_of_gt (hp x)), Real.log_one, mul_zero]
  | (n + 2), p, hp => by
      have hIH : klG (dropFirst p) (mproj (dropFirst p)) = seamInfoTotal (dropFirst p) :=
        klG_mproj_eq_seamInfoTotal (fun w => dropFirst_pos hp w)
      have hstep : klG p (mproj p)
          = (∑ a : A, ∑ b : A, ∑ w : Blocks A n,
              p (a, b, w) * Real.log (p (a, b, w) / glue (cur p) a b w))
            + ∑ a : A, ∑ b : A, ∑ w : Blocks A n,
              p (a, b, w) * Real.log (dropFirst p (b, w) / mproj (dropFirst p) (b, w)) := by
        rw [klG, sum_blocks_succ2]
        rw [← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun a _ => ?_
        rw [← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun b _ => ?_
        rw [← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun w _ => log_ratio_split hp a b w
      have hfirst : (∑ a : A, ∑ b : A, ∑ w : Blocks A n,
          p (a, b, w) * Real.log (p (a, b, w) / glue (cur p) a b w)) = cmi (cur p) :=
        (cmi_eq_sum_log_ratio (p := cur p) (fun a b w => le_of_lt (hp _))).symm
      have hsecond : (∑ a : A, ∑ b : A, ∑ w : Blocks A n,
          p (a, b, w) * Real.log (dropFirst p (b, w) / mproj (dropFirst p) (b, w)))
          = klG (dropFirst p) (mproj (dropFirst p)) := by
        rw [klG, sum_blocks_succ, Finset.sum_comm]
        refine Finset.sum_congr rfl fun b _ => ?_
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun w _ => ?_
        rw [dropFirst, Finset.sum_mul]
      rw [hstep, hfirst, hsecond, hIH, seamInfoTotal_succ2]

/-! ## Nonnegativity, and exactness -/

/-- Each seam term is nonnegative, hence so is the total price of a pipeline. -/
theorem seamInfoTotal_nonneg : ∀ {n : ℕ} {p : Blocks A n → ℝ}, (∀ x, 0 < p x) →
    ∑ x, p x = 1 → 0 ≤ seamInfoTotal p
  | 0, _, _, _ => le_of_eq rfl
  | 1, _, _, _ => le_of_eq rfl
  | (n + 2), p, hp, hs => by
      have hs3 : ∑ a : A, ∑ b : A, ∑ w : Blocks A n, p (a, b, w) = 1 := by
        rw [← sum_blocks_succ2 p]; exact hs
      have h1 : 0 ≤ cmi (cur p) :=
        cmi_nonneg (p := cur p) (fun a b w => le_of_lt (hp _)) hs3
      have h2 : 0 ≤ seamInfoTotal (dropFirst p) := by
        refine seamInfoTotal_nonneg (fun w => dropFirst_pos hp w) ?_
        rw [sum_dropFirst]; exact hs
      rw [seamInfoTotal_succ2]
      linarith

/-- A pipeline reproduces the truth exactly precisely when the chain is Markov along the
sequence of blocks — that is, when every seam decouples what precedes it from what follows
it. -/
theorem mproj_eq_self_iff_isPipeline : ∀ {n : ℕ} {p : Blocks A n → ℝ}, (∀ x, 0 < p x) →
    ((∀ x, mproj p x = p x) ↔ IsPipeline p)
  | 0, _, _ => ⟨fun _ => trivial, fun _ _ => rfl⟩
  | 1, _, _ => ⟨fun _ => trivial, fun _ _ => rfl⟩
  | (n + 2), p, hp => by
      constructor
      · intro h
        have hdrop : ∀ v : Blocks A (n + 1), mproj (dropFirst p) v = dropFirst p v := by
          intro v
          rw [← mproj_dropFirst hp v]
          exact Finset.sum_congr rfl fun a _ => h _
        have hglue : ∀ a b w, glue (cur p) a b w = cur p a b w := by
          intro a b w
          have hy' : margY (cur p) b ≠ 0 := ne_of_gt (margY_cur_pos hp b)
          have h1 := h (a, b, w)
          rw [mproj_succ2, hdrop (b, w)] at h1
          show glue (cur p) a b w = p (a, b, w)
          rw [glue, ← h1, dropFirst_eq_margYZ]
          field_simp
        refine ⟨(glue_eq_self_iff_condIndep (p := cur p)
          (fun a b w => le_of_lt (hp _))).1 hglue, ?_⟩
        exact (mproj_eq_self_iff_isPipeline (fun w => dropFirst_pos hp w)).1 hdrop
      · rintro ⟨hci, htail⟩ x
        have hdrop : ∀ v, mproj (dropFirst p) v = dropFirst p v :=
          (mproj_eq_self_iff_isPipeline (fun w => dropFirst_pos hp w)).2 htail
        obtain ⟨a, b, w⟩ := x
        have hy' : margY (cur p) b ≠ 0 := ne_of_gt (margY_cur_pos hp b)
        have hg : glue (cur p) a b w = p (a, b, w) :=
          (glue_eq_self_iff_condIndep (p := cur p) (fun a b w => le_of_lt (hp _))).2 hci a b w
        rw [mproj_succ2, hdrop (b, w), dropFirst_eq_margYZ, ← hg, glue]
        field_simp

/-- **The pipeline is free exactly when it is exact.**  The total seam information vanishes
if and only if the true chain is one a fragment pipeline can represent. -/
theorem seamInfoTotal_eq_zero_iff : ∀ {n : ℕ} {p : Blocks A n → ℝ}, (∀ x, 0 < p x) →
    ∑ x, p x = 1 → (seamInfoTotal p = 0 ↔ IsPipeline p)
  | 0, _, _, _ => ⟨fun _ => trivial, fun _ => rfl⟩
  | 1, _, _, _ => ⟨fun _ => trivial, fun _ => rfl⟩
  | (n + 2), p, hp, hs => by
      have hs3 : ∑ a : A, ∑ b : A, ∑ w : Blocks A n, p (a, b, w) = 1 := by
        rw [← sum_blocks_succ2 p]; exact hs
      have hsd : ∑ w : Blocks A (n + 1), dropFirst p w = 1 := by rw [sum_dropFirst]; exact hs
      have h1 : 0 ≤ cmi (cur p) :=
        cmi_nonneg (p := cur p) (fun a b w => le_of_lt (hp _)) hs3
      have h2 : 0 ≤ seamInfoTotal (dropFirst p) :=
        seamInfoTotal_nonneg (fun w => dropFirst_pos hp w) hsd
      have hiff := cmi_eq_zero_iff_condIndep (p := cur p) (fun a b w => le_of_lt (hp _)) hs3
      have hiff2 := seamInfoTotal_eq_zero_iff (fun w => dropFirst_pos hp w) hsd
      rw [seamInfoTotal_succ2]
      constructor
      · intro h
        exact ⟨hiff.1 (by linarith), hiff2.1 (by linarith)⟩
      · rintro ⟨hci, htail⟩
        rw [hiff.2 hci, hiff2.2 htail, add_zero]

end RequestProject.ChainPipeline
