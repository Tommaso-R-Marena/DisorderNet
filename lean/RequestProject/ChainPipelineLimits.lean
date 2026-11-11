/-
# Fragment pipelines II: what no pipeline can do

`RequestProject.ChainPipeline` showed that the model a fragment pipeline returns when every
fragment is fitted perfectly is at relative entropy exactly `seamInfoTotal p` — the sum of the
conditional mutual informations at the seams — from the truth.  This file turns that identity
into limits on what any pipeline whatsoever can achieve.

* `no_pipeline_beats_seamInfoTotal` — **no joining rule, and no amount of fitting, does
  better.**  Every strictly positive ensemble that is Markov along the chain — that is, every
  ensemble a fragment pipeline can express, however it was produced — is at relative entropy
  at least `seamInfoTotal p` from the truth.  The Markov projection attains the floor, so the
  price is a property of where the chain was cut, not of the algorithm that joined the pieces.
* `pipeline_l1_le_sqrt_seamInfoTotal`, `pipeline_design_rule`, `pipeline_seam_budget` — through
  Pinsker the price becomes an operational `ℓ¹` error, and the design rule for a whole
  pipeline: to build a full-length ensemble to accuracy `eps`, place the cuts so that the seam
  informations *sum* to at most `eps²/2`; with `k` seams a uniform per-seam budget of
  `eps²/(2k)` suffices.
* `seamInfoTotal_ge_of_seamFloor`, `pipeline_length_limit` — **the error of a fragment pipeline
  is extensive.**  If every seam of a long disordered chain carries at least `c` of
  information, the price of the pipeline grows linearly in the number of seams, and a pipeline
  model accurate to `eps` in relative entropy can span at most `eps/c` seams.  Long-range
  couplings do not merely perturb a fragment-based model of a disordered region; they place a
  hard ceiling on the length of chain such a model can describe.
* `isPipeline_mproj`, `fragment_panels_never_falsify_pipeline` — and the pipeline assumption
  cannot be tested by fragment data: the pipeline model reproduces every adjacent panel of the
  truth and carries zero seam information of its own, at every seam.
-/
import Mathlib
import RequestProject.ChainPipeline

namespace RequestProject.ChainPipeline

open Finset IDR.Pinsker RequestProject.Modular

universe u

variable {A : Type u} [Fintype A] [Nonempty A]

/-! ## Product form and conditional independence -/

omit [Nonempty A] in
/-- Any ensemble that factorises through the seam is conditionally independent across it.
This is the algebraic content of "a fragment pipeline can only produce Markov chains". -/
lemma condIndep_of_product {X Y Z : Type*} [Fintype X] [Fintype Y] [Fintype Z]
    (f : X → Y → ℝ) (g : Y → Z → ℝ) :
    CondIndep (fun x y z => f x y * g y z) := by
  intro x y z
  have hXY : margXY (fun x y z => f x y * g y z) x y = f x y * ∑ z, g y z := by
    rw [margXY, Finset.mul_sum]
  have hYZ : margYZ (fun x y z => f x y * g y z) y z = (∑ x, f x y) * g y z := by
    rw [margYZ, Finset.sum_mul]
  have hY : margY (fun x y z => f x y * g y z) y = (∑ x, f x y) * ∑ z, g y z := by
    rw [margY, Finset.sum_mul]
    exact Finset.sum_congr rfl fun x _ => by rw [Finset.mul_sum]
  rw [hXY, hYZ, hY]
  ring

/-- A positive ensemble that is conditionally independent across the seam factorises into the
head conditional and the law of everything from the seam onwards. -/
lemma factor_of_condIndep {n : ℕ} {m : Blocks A (n + 2) → ℝ} (hm : ∀ x, 0 < m x)
    (hci : CondIndep (cur m)) (a b : A) (w : Blocks A n) :
    m (a, b, w) = (margXY (cur m) a b / margY (cur m) b) * dropFirst m (b, w) := by
  have hy : 0 < margY (cur m) b := margY_cur_pos hm b
  have := hci a b w
  rw [dropFirst_eq_margYZ]
  field_simp
  exact this

/-! ## No pipeline beats the total seam information -/

/-- The pointwise split of the log-likelihood ratio against an arbitrary pipeline model:
the seam term, the mismatch of the head conditionals, and the ratio of the truncated chains. -/
lemma log_ratio_split_model {n : ℕ} {p m : Blocks A (n + 2) → ℝ} (hp : ∀ x, 0 < p x)
    (hm : ∀ x, 0 < m x) (hci : CondIndep (cur m)) (a b : A) (w : Blocks A n) :
    p (a, b, w) * Real.log (p (a, b, w) / m (a, b, w))
      = p (a, b, w) * Real.log (p (a, b, w) / glue (cur p) a b w)
        + p (a, b, w) * Real.log
            ((margXY (cur p) a b / margY (cur p) b) / (margXY (cur m) a b / margY (cur m) b))
        + p (a, b, w) * Real.log (dropFirst p (b, w) / dropFirst m (b, w)) := by
  have hcp : 0 < margXY (cur p) a b / margY (cur p) b :=
    div_pos (margXY_cur_pos hp a b) (margY_cur_pos hp b)
  have hcm : 0 < margXY (cur m) a b / margY (cur m) b :=
    div_pos (margXY_cur_pos hm a b) (margY_cur_pos hm b)
  have htp : 0 < dropFirst p (b, w) := dropFirst_pos hp (b, w)
  have htm : 0 < dropFirst m (b, w) := dropFirst_pos hm (b, w)
  have hpx : 0 < p (a, b, w) := hp _
  have hglue : glue (cur p) a b w
      = (margXY (cur p) a b / margY (cur p) b) * dropFirst p (b, w) := by
    rw [glue, dropFirst_eq_margYZ]
    field_simp
  have hmfac : m (a, b, w) = (margXY (cur m) a b / margY (cur m) b) * dropFirst m (b, w) :=
    factor_of_condIndep hm hci a b w
  rw [hglue, hmfac, ← mul_add, ← mul_add]
  congr 1
  rw [Real.log_div (ne_of_gt hpx) (by positivity), Real.log_div (ne_of_gt hpx) (by positivity),
    Real.log_div (ne_of_gt hcp) (ne_of_gt hcm), Real.log_div (ne_of_gt htp) (ne_of_gt htm),
    Real.log_mul (ne_of_gt hcp) (ne_of_gt htp), Real.log_mul (ne_of_gt hcm) (ne_of_gt htm)]
  ring

/-- The head conditionals of two ensembles differ at a nonnegative price: the average, over
the seam state, of the relative entropy between the two conditional laws of the block before
the seam. -/
lemma head_conditional_mismatch_nonneg {n : ℕ} {p m : Blocks A (n + 2) → ℝ}
    (hp : ∀ x, 0 < p x) (hm : ∀ x, 0 < m x) :
    0 ≤ ∑ a : A, ∑ b : A, margXY (cur p) a b * Real.log
      ((margXY (cur p) a b / margY (cur p) b) / (margXY (cur m) a b / margY (cur m) b)) := by
  have hswap : (∑ a : A, ∑ b : A, margXY (cur p) a b * Real.log
        ((margXY (cur p) a b / margY (cur p) b) / (margXY (cur m) a b / margY (cur m) b)))
      = ∑ b : A, margY (cur p) b * klG (fun a : A => margXY (cur p) a b / margY (cur p) b)
          (fun a : A => margXY (cur m) a b / margY (cur m) b) := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [klG, Finset.mul_sum]
    refine Finset.sum_congr rfl fun a _ => ?_
    have hy : margY (cur p) b ≠ 0 := ne_of_gt (margY_cur_pos hp b)
    field_simp
  rw [hswap]
  refine Finset.sum_nonneg fun b _ => ?_
  refine mul_nonneg (le_of_lt (margY_cur_pos hp b)) ?_
  refine klG_nonneg (fun a => le_of_lt (div_pos (margXY_cur_pos hp a b) (margY_cur_pos hp b)))
    (fun a => div_pos (margXY_cur_pos hm a b) (margY_cur_pos hm b)) ?_ ?_
  · rw [← Finset.sum_div, ← margY_eq_sum_margXY (cur p) b,
      div_self (ne_of_gt (margY_cur_pos hp b))]
  · rw [← Finset.sum_div, ← margY_eq_sum_margXY (cur m) b,
      div_self (ne_of_gt (margY_cur_pos hm b))]

/-- **No fragment pipeline beats the total seam information.**  Every strictly positive
ensemble that a fragment pipeline can express — Markov along the chain, however its pieces
were fitted or joined — is at relative entropy at least `seamInfoTotal p` from the truth.  The
Markov projection attains this floor exactly (`klG_mproj_eq_seamInfoTotal`), so the price of
building a disordered region out of fragments is set by where the chain is cut and by nothing
else. -/
theorem no_pipeline_beats_seamInfoTotal : ∀ {n : ℕ} {p m : Blocks A n → ℝ}, (∀ x, 0 < p x) →
    (∀ x, 0 < m x) → ∑ x, p x = 1 → ∑ x, m x = 1 → IsPipeline m →
    seamInfoTotal p ≤ klG p m
  | 0, p, m, hp, hm, hps, hms, _ => by
      rw [seamInfoTotal_zero]
      exact klG_nonneg (fun x => le_of_lt (hp x)) hm hps hms
  | 1, p, m, hp, hm, hps, hms, _ => by
      rw [seamInfoTotal_one]
      exact klG_nonneg (fun x => le_of_lt (hp x)) hm hps hms
  | (n + 2), p, m, hp, hm, hps, hms, hpipe => by
      obtain ⟨hci, htail⟩ := hpipe
      have hIH : seamInfoTotal (dropFirst p) ≤ klG (dropFirst p) (dropFirst m) := by
        refine no_pipeline_beats_seamInfoTotal (fun w => dropFirst_pos hp w)
          (fun w => dropFirst_pos hm w) ?_ ?_ htail
        · rw [sum_dropFirst]; exact hps
        · rw [sum_dropFirst]; exact hms
      have hsplit : klG p m
          = (∑ a : A, ∑ b : A, ∑ w : Blocks A n,
              p (a, b, w) * Real.log (p (a, b, w) / glue (cur p) a b w))
            + (∑ a : A, ∑ b : A, ∑ w : Blocks A n, p (a, b, w) * Real.log
                ((margXY (cur p) a b / margY (cur p) b) / (margXY (cur m) a b / margY (cur m) b)))
            + ∑ a : A, ∑ b : A, ∑ w : Blocks A n,
                p (a, b, w) * Real.log (dropFirst p (b, w) / dropFirst m (b, w)) := by
        rw [klG, sum_blocks_succ2]
        rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun a _ => ?_
        rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun b _ => ?_
        rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun w _ => log_ratio_split_model hp hm hci a b w
      have hfirst : (∑ a : A, ∑ b : A, ∑ w : Blocks A n,
          p (a, b, w) * Real.log (p (a, b, w) / glue (cur p) a b w)) = cmi (cur p) :=
        (cmi_eq_sum_log_ratio (p := cur p) (fun a b w => le_of_lt (hp _))).symm
      have hmid : (∑ a : A, ∑ b : A, ∑ w : Blocks A n, p (a, b, w) * Real.log
            ((margXY (cur p) a b / margY (cur p) b) / (margXY (cur m) a b / margY (cur m) b)))
          = ∑ a : A, ∑ b : A, margXY (cur p) a b * Real.log
            ((margXY (cur p) a b / margY (cur p) b) / (margXY (cur m) a b / margY (cur m) b)) :=
        sum_triple_margXY (cur p) _
      have hlast : (∑ a : A, ∑ b : A, ∑ w : Blocks A n,
          p (a, b, w) * Real.log (dropFirst p (b, w) / dropFirst m (b, w)))
          = klG (dropFirst p) (dropFirst m) := by
        rw [klG, sum_blocks_succ, Finset.sum_comm]
        refine Finset.sum_congr rfl fun b _ => ?_
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun w _ => ?_
        rw [dropFirst, Finset.sum_mul]
      have hmidnn := head_conditional_mismatch_nonneg (p := p) (m := m) hp hm
      rw [seamInfoTotal_succ2, hsplit, hfirst, hmid, hlast]
      linarith

/-- **If any seam carries information, no fragment pipeline captures the ensemble.**  When the
total seam information is positive, every pipeline model — every ensemble assembled from
fragments, however fitted — differs from the truth somewhere and sits at strictly positive
relative entropy from it.  Capturing such a disordered region fully requires a model that is
not built by joining fragments across a cut. -/
theorem no_pipeline_captures_chain {n : ℕ} {p m : Blocks A n → ℝ} (hp : ∀ x, 0 < p x)
    (hm : ∀ x, 0 < m x) (hps : ∑ x, p x = 1) (hms : ∑ x, m x = 1) (hpipe : IsPipeline m)
    (hpos : 0 < seamInfoTotal p) : 0 < klG p m ∧ ∃ x, p x ≠ m x := by
  have hle : seamInfoTotal p ≤ klG p m := no_pipeline_beats_seamInfoTotal hp hm hps hms hpipe
  have hkl : 0 < klG p m := lt_of_lt_of_le hpos hle
  refine ⟨hkl, ?_⟩
  by_contra hcon
  push_neg at hcon
  have : klG p m = 0 := by
    rw [klG]
    refine Finset.sum_eq_zero fun x _ => ?_
    rw [hcon x, div_self (ne_of_gt (hm x)), Real.log_one, mul_zero]
  linarith

/-! ## From the total price to operational error -/

/-- **Pinsker for the whole pipeline.**  The population-space `ℓ¹` error of the pipeline model
is at most the square root of twice the total seam information. -/
theorem pipeline_l1_le_sqrt_seamInfoTotal {n : ℕ} {p : Blocks A n → ℝ} (hp : ∀ x, 0 < p x)
    (hs : ∑ x, p x = 1) :
    ∑ x : Blocks A n, |p x - mproj p x| ≤ Real.sqrt (2 * seamInfoTotal p) := by
  have h := ell1_le_sqrt_two_klG (p := p) (q := mproj p) (fun x => le_of_lt (hp x))
    (fun x => mproj_pos hp x) hs (sum_mproj hp hs)
  rwa [klG_mproj_eq_seamInfoTotal hp] at h

/-- **The pipeline design rule.**  To build a full-length disordered ensemble out of fragments
to population accuracy `eps`, place the cuts so that the seam informations *sum* to at most
`eps²/2`. -/
theorem pipeline_design_rule {n : ℕ} {p : Blocks A n → ℝ} (hp : ∀ x, 0 < p x)
    (hs : ∑ x, p x = 1) {eps : ℝ} (heps : 0 ≤ eps) (hbudget : seamInfoTotal p ≤ eps ^ 2 / 2) :
    ∑ x : Blocks A n, |p x - mproj p x| ≤ eps := by
  have h1 := pipeline_l1_le_sqrt_seamInfoTotal hp hs
  have h2 : Real.sqrt (2 * seamInfoTotal p) ≤ eps := by
    have hle : 2 * seamInfoTotal p ≤ eps ^ 2 := by linarith
    calc Real.sqrt (2 * seamInfoTotal p) ≤ Real.sqrt (eps ^ 2) := Real.sqrt_le_sqrt hle
      _ = eps := Real.sqrt_sq heps
  linarith

/-! ## The seam budget: how the total is spread over the cuts -/

/-- Every seam of the chain carries at most `s` of information. -/
def SeamCap (s : ℝ) : ∀ {n : ℕ}, (Blocks A n → ℝ) → Prop
  | 0, _ => True
  | 1, _ => True
  | (_ + 2), p => cmi (cur p) ≤ s ∧ SeamCap s (dropFirst p)

/-- Every seam of the chain carries at least `c` of information. -/
def SeamFloor (c : ℝ) : ∀ {n : ℕ}, (Blocks A n → ℝ) → Prop
  | 0, _ => True
  | 1, _ => True
  | (_ + 2), p => c ≤ cmi (cur p) ∧ SeamFloor c (dropFirst p)

omit [Nonempty A] in
/-- A chain of `n+2` blocks has `n` seams, and a per-seam cap of `s` caps the total price at
`n·s`. -/
theorem seamInfoTotal_le_of_seamCap {s : ℝ} : ∀ {n : ℕ} {p : Blocks A (n + 1) → ℝ},
    SeamCap s p → seamInfoTotal p ≤ (n : ℝ) * s
  | 0, _, _ => by simp [seamInfoTotal_one]
  | (n + 1), p, hcap => by
      obtain ⟨h1, h2⟩ := hcap
      have hIH : seamInfoTotal (dropFirst p) ≤ (n : ℝ) * s := seamInfoTotal_le_of_seamCap h2
      rw [seamInfoTotal_succ2]
      push_cast
      linarith

omit [Nonempty A] in
/-- Symmetrically, a per-seam floor of `c` forces the total price to be at least `n·c`. -/
theorem seamInfoTotal_ge_of_seamFloor {c : ℝ} : ∀ {n : ℕ} {p : Blocks A (n + 1) → ℝ},
    SeamFloor c p → (n : ℝ) * c ≤ seamInfoTotal p
  | 0, _, _ => by simp [seamInfoTotal_one]
  | (n + 1), p, hfl => by
      obtain ⟨h1, h2⟩ := hfl
      have hIH : (n : ℝ) * c ≤ seamInfoTotal (dropFirst p) := seamInfoTotal_ge_of_seamFloor h2
      rw [seamInfoTotal_succ2]
      push_cast
      linarith

/-- **The uniform seam budget.**  A chain cut into fragments at `n` seams, each carrying at
most `eps²/(2n)` of information, is reproduced by the pipeline to population accuracy `eps`. -/
theorem pipeline_seam_budget {n : ℕ} (hn : 0 < n) {p : Blocks A (n + 1) → ℝ} (hp : ∀ x, 0 < p x)
    (hs : ∑ x, p x = 1) {eps : ℝ} (heps : 0 ≤ eps) (hcap : SeamCap (eps ^ 2 / (2 * n)) p) :
    ∑ x : Blocks A (n + 1), |p x - mproj p x| ≤ eps := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have h := seamInfoTotal_le_of_seamCap hcap
  have hbudget : seamInfoTotal p ≤ eps ^ 2 / 2 := by
    have : (n : ℝ) * (eps ^ 2 / (2 * n)) = eps ^ 2 / 2 := by
      field_simp
    linarith [this ▸ h]
  exact pipeline_design_rule hp hs heps hbudget

/-- **The error of a fragment pipeline is extensive, and that caps the chain length.**  If
every seam of the chain carries at least `c > 0` of information, then *no* pipeline model —
whatever the fitting procedure — is within relative entropy `eps` of the truth once the chain
has more than `eps/c` seams.  A fragment-based description of a disordered region is not
merely approximate: its error grows with length, so it has a maximum length beyond which it
cannot be trusted. -/
theorem pipeline_length_limit {n : ℕ} {c eps : ℝ} (hc : 0 < c) {p m : Blocks A (n + 1) → ℝ}
    (hp : ∀ x, 0 < p x) (hm : ∀ x, 0 < m x) (hps : ∑ x, p x = 1) (hms : ∑ x, m x = 1)
    (hpipe : IsPipeline m) (hfl : SeamFloor c p) (hacc : klG p m ≤ eps) :
    (n : ℝ) ≤ eps / c := by
  have h1 : (n : ℝ) * c ≤ seamInfoTotal p := seamInfoTotal_ge_of_seamFloor hfl
  have h2 : seamInfoTotal p ≤ klG p m :=
    no_pipeline_beats_seamInfoTotal hp hm hps hms hpipe
  rw [le_div_iff₀ hc]
  linarith

/-! ## Fragment data cannot test the pipeline assumption -/

/-- The law of the leading block of a chain. -/
def headMarg {n : ℕ} (q : Blocks A (n + 1) → ℝ) (b : A) : ℝ := ∑ w : Blocks A n, q (b, w)

omit [Nonempty A] in
lemma headMarg_dropFirst {n : ℕ} (p : Blocks A (n + 2) → ℝ) (b : A) :
    headMarg (dropFirst p) b = margY (cur p) b := by
  rw [headMarg, margY, Finset.sum_comm]
  rfl

/-- The pipeline model gets the law of every single block exactly right. -/
theorem headMarg_mproj : ∀ {n : ℕ} {q : Blocks A (n + 1) → ℝ}, (∀ x, 0 < q x) → ∀ b : A,
    headMarg (mproj q) b = headMarg q b
  | 0, _, _, _ => rfl
  | (n + 1), q, hq, b => by
      have hIH : ∀ b' : A, headMarg (mproj (dropFirst q)) b' = headMarg (dropFirst q) b' :=
        headMarg_mproj (fun w => dropFirst_pos hq w)
      have hexp : headMarg (mproj q) b
          = ∑ b' : A, (margXY (cur q) b b' / margY (cur q) b')
              * headMarg (mproj (dropFirst q)) b' := by
        rw [headMarg, sum_blocks_succ]
        refine Finset.sum_congr rfl fun b' _ => ?_
        rw [headMarg, Finset.mul_sum]
        exact Finset.sum_congr rfl fun w _ => mproj_succ2 q b b' w
      rw [hexp]
      have hstep : ∀ b' : A, (margXY (cur q) b b' / margY (cur q) b')
          * headMarg (mproj (dropFirst q)) b' = margXY (cur q) b b' := by
        intro b'
        rw [hIH b', headMarg_dropFirst q b', div_mul_cancel₀]
        exact ne_of_gt (margY_cur_pos hq b')
      rw [Finset.sum_congr rfl fun b' _ => hstep b', headMarg]
      exact (sum_blocks_succ (fun w : Blocks A (n + 1) => q (b, w))).symm

/-- The pipeline model reproduces the panel of the leading fragment — the joint law of the
first two blocks — exactly. -/
theorem margXY_cur_mproj {n : ℕ} {p : Blocks A (n + 2) → ℝ} (hp : ∀ x, 0 < p x) (a b : A) :
    margXY (cur (mproj p)) a b = margXY (cur p) a b := by
  have hexp : margXY (cur (mproj p)) a b
      = (margXY (cur p) a b / margY (cur p) b) * headMarg (mproj (dropFirst p)) b := by
    rw [margXY, headMarg, Finset.mul_sum]
    exact Finset.sum_congr rfl fun w _ => mproj_succ2 p a b w
  rw [hexp, headMarg_mproj (fun w => dropFirst_pos hp w) b, headMarg_dropFirst p b,
    div_mul_cancel₀]
  exact ne_of_gt (margY_cur_pos hp b)

/-- The Markov projection is itself a pipeline ensemble: it is conditionally independent
across every seam. -/
theorem isPipeline_mproj : ∀ {n : ℕ} {p : Blocks A n → ℝ}, (∀ x, 0 < p x) → IsPipeline (mproj p)
  | 0, _, _ => trivial
  | 1, _, _ => trivial
  | (n + 2), p, hp => by
      refine ⟨?_, ?_⟩
      · have hform : cur (mproj p) = fun a b (w : Blocks A n) =>
            (margXY (cur p) a b / margY (cur p) b) * mproj (dropFirst p) (b, w) := rfl
        rw [hform]
        exact condIndep_of_product (fun a b => margXY (cur p) a b / margY (cur p) b)
          (fun b w => mproj (dropFirst p) (b, w))
      · have hd : dropFirst (mproj p) = mproj (dropFirst p) :=
          funext fun w => mproj_dropFirst hp w
        rw [hd]
        exact isPipeline_mproj (fun w => dropFirst_pos hp w)

/-- **Fragment measurements can never falsify the pipeline assumption.**  Whatever the truth
is, the pipeline model reproduces the panel of the leading fragment exactly, marginalises to
the pipeline model of the truncated chain — hence reproduces *every* fragment panel along the
chain — and has zero seam information at every seam of its own.  There is therefore always an
exactly Markov ensemble consistent with all the fragment data, and the assumption that a
disordered region can be assembled from fragments is testable only by an observable that spans
a cut. -/
theorem fragment_panels_never_falsify_pipeline {n : ℕ} {p : Blocks A (n + 2) → ℝ}
    (hp : ∀ x, 0 < p x) (hs : ∑ x, p x = 1) :
    (∀ a b, margXY (cur (mproj p)) a b = margXY (cur p) a b) ∧
      dropFirst (mproj p) = mproj (dropFirst p) ∧
      seamInfoTotal (mproj p) = 0 :=
  ⟨fun a b => margXY_cur_mproj hp a b, funext fun w => mproj_dropFirst hp w,
    (seamInfoTotal_eq_zero_iff (fun x => mproj_pos hp x) (sum_mproj hp hs)).2
      (isPipeline_mproj hp)⟩

end RequestProject.ChainPipeline
