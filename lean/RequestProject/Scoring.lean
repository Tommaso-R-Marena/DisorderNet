/-
# Part XIV.1  Scoring an ensemble prediction: which metrics can select the right model

Every earlier part asks what a *model* of a disordered region must be.  This file asks the
complementary question, which is just as much a part of the design: by what number is the
prediction to be judged?  The choice is not innocent -- a model is whatever the score
selects, so a score that is minimised by something other than the truth guarantees that a
well-trained, well-evaluated pipeline converges on the wrong object.

The setting is a finite conformation library `X`; a prediction is a population vector
`p : X → ℝ` (`IsProbVec`), the truth is another one `q`, and a *score* `S p y` is the
penalty incurred when the prediction `p` meets an observed conformation `y`.  Its expected
value under the truth is `expScore S p q`, and the score is `StrictlyProper` when the truth
is the unique minimiser.

* `brier_strictly_proper` -- the quadratic (Brier) score over the conformation library is
  strictly proper, with the exact excess-risk identity
  `expScore brier p q - expScore brier q q = ∑ x, (p x - q x)^2`
  (`brier_excess`).  Strictly proper ensemble scores therefore exist, and they are cheap.
* `strictlyProper_rejects_point_prediction` -- under *any* strictly proper score, a
  single-structure prediction is strictly worse than the truth as soon as the truth does
  not put all of its population on one conformation.  This is the evaluation-side
  counterpart of the Part I error floor.
* `sqDistScore` -- the score in everyday use: draw a structure from the model and measure
  its (squared) deviation from the observed one.  `sqDistScore_expected` is the exact
  identity `E = var p + var q + (mean p - mean q)^2`; hence
  `sqDistScore_rewards_collapse`, at fixed mean the score strictly *decreases* as the
  prediction becomes less dispersed, so the truthful model is beaten by its own collapse,
  and `sqDistScore_not_strictlyProper` / `sqDist_three_state` make this explicit.
  `sqDistScore_depends_on_two_moments` isolates the reason: the score sees the prediction
  only through two numbers.
* `bestOfScore` -- the other score in everyday use: report `N` structures and keep the one
  closest to the observation.  `bestOfScore_eq_of_supp_eq`: it does not see the
  populations at all; `bestOfScore_antitone_supp` and `bestOfScore_expected_eq_zero`: any
  prediction whose support contains the truth's scores a perfect zero, so hedging is free
  and the truth is never strictly preferred (`bestOfScore_not_strictlyProper`).
-/
import Mathlib

namespace IDR

namespace Scoring

open Finset
open scoped BigOperators Classical

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- A population vector on the conformation library `X`. -/
def IsProbVec (p : X → ℝ) : Prop := (∀ x, 0 ≤ p x) ∧ ∑ x, p x = 1

/-- The expected score of the prediction `p` when the truth is `q`. -/
def expScore (S : (X → ℝ) → X → ℝ) (p q : X → ℝ) : ℝ := ∑ y, q y * S p y

/-- A score is *proper* when no prediction beats the truth. -/
def Proper (S : (X → ℝ) → X → ℝ) : Prop :=
  ∀ p q : X → ℝ, IsProbVec p → IsProbVec q → expScore S q q ≤ expScore S p q

/-- A score is *strictly proper* when the truth is the unique minimiser: the score can be
used to fit and to evaluate a model without biasing it towards anything else. -/
def StrictlyProper (S : (X → ℝ) → X → ℝ) : Prop :=
  ∀ p q : X → ℝ, IsProbVec p → IsProbVec q → p ≠ q → expScore S q q < expScore S p q

omit [DecidableEq X] in
lemma Proper_of_StrictlyProper {S : (X → ℝ) → X → ℝ} (h : StrictlyProper S) : Proper S := by
  intro p q hp hq
  rcases eq_or_ne p q with rfl | hne
  · exact le_refl _
  · exact (h p q hp hq hne).le

/-- The population vector of a single-structure ("point") prediction. -/
def pointVec (x₀ : X) : X → ℝ := fun x => if x = x₀ then 1 else 0

lemma isProbVec_pointVec (x₀ : X) : IsProbVec (pointVec (X := X) x₀) := by
  constructor
  · intro x; by_cases h : x = x₀ <;> simp [pointVec, h]
  · simp [pointVec]

/-! ## The quadratic (Brier) score -/

/-- The quadratic score of the predicted populations `p` against the observed
conformation `y`. -/
def brier (p : X → ℝ) (y : X) : ℝ := ∑ x, (p x - if x = y then (1 : ℝ) else 0) ^ 2

lemma brier_eq (p : X → ℝ) (y : X) :
    brier p y = (∑ x, p x ^ 2) - 2 * p y + 1 := by
  have h : ∀ x : X, (p x - if x = y then (1 : ℝ) else 0) ^ 2
      = p x ^ 2 - (if x = y then 2 * p x - 1 else 0) := by
    intro x; by_cases h : x = y <;> simp [h]; ring
  simp only [brier, h, Finset.sum_sub_distrib]
  rw [Finset.sum_ite_eq' Finset.univ y (fun x => 2 * p x - 1)]
  simp
  ring

lemma expScore_brier (p q : X → ℝ) (hq : IsProbVec q) :
    expScore brier p q = (∑ x, p x ^ 2) - 2 * (∑ x, q x * p x) + 1 := by
  simp only [expScore, brier_eq]
  have : ∀ y : X, q y * ((∑ x, p x ^ 2) - 2 * p y + 1)
      = q y * ((∑ x, p x ^ 2) + 1) - 2 * (q y * p y) := by
    intro y; ring
  simp only [this, Finset.sum_sub_distrib, ← Finset.sum_mul, ← Finset.mul_sum, hq.2]
  ring

/-- **Exact excess risk of the quadratic score.**  Predicting `p` when the truth is `q`
costs exactly the squared Euclidean distance between the two population vectors. -/
theorem brier_excess (p q : X → ℝ) (hq : IsProbVec q) :
    expScore brier p q - expScore brier q q = ∑ x, (p x - q x) ^ 2 := by
  rw [expScore_brier p q hq, expScore_brier q q hq]
  have h : ∀ x : X, (p x - q x) ^ 2 = p x ^ 2 - q x * p x - q x * p x + q x ^ 2 := by
    intro x; ring
  simp only [h, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  ring_nf

/-- **The quadratic score is strictly proper.**  A strictly proper score on ensembles
exists: the truth is its unique minimiser, so nothing but the true population vector is
rewarded. -/
theorem brier_strictly_proper : StrictlyProper (X := X) brier := by
  intro p q hp hq hne
  have hex := brier_excess p q hq
  have hpos : 0 < ∑ x, (p x - q x) ^ 2 := by
    rcases Function.ne_iff.1 hne with ⟨x₀, hx₀⟩
    refine Finset.sum_pos' (fun x _ => sq_nonneg _) ⟨x₀, Finset.mem_univ x₀, ?_⟩
    have hne0 : p x₀ - q x₀ ≠ 0 := sub_ne_zero.2 hx₀
    positivity
  linarith

/-- **Any strictly proper score rejects a single-structure prediction.**  If the truth does
not put all of its population on the conformation `x₀`, then predicting `x₀` alone is
strictly worse than reporting the true ensemble.  This is the evaluation-side form of the
error floor. -/
theorem strictlyProper_rejects_point_prediction {S : (X → ℝ) → X → ℝ}
    (hS : StrictlyProper S) (q : X → ℝ) (hq : IsProbVec q) (x₀ : X) (hx₀ : q x₀ < 1) :
    expScore S q q < expScore S (pointVec x₀) q := by
  refine hS _ q (isProbVec_pointVec x₀) hq ?_
  intro h
  have hval : (1 : ℝ) = q x₀ := by simpa [pointVec] using congrFun h x₀
  linarith

/-! ## What an honest score selects when the model cannot see the whole context -/

/-- The population-weighted average of a family of targets. -/
def mixVec {ι : Type*} [Fintype ι] (r : ι → ℝ) (q : ι → X → ℝ) : X → ℝ :=
  fun x => ∑ i, r i * q i x

omit [DecidableEq X] in
lemma isProbVec_mixVec {ι : Type*} [Fintype ι] {r : ι → ℝ} {q : ι → X → ℝ}
    (hr : IsProbVec r) (hq : ∀ i, IsProbVec (q i)) : IsProbVec (mixVec r q) := by
  refine ⟨fun x => Finset.sum_nonneg fun i _ => mul_nonneg (hr.1 i) ((hq i).1 x), ?_⟩
  simp only [mixVec]
  rw [Finset.sum_comm]
  have h : ∀ i : ι, ∑ x : X, r i * q i x = r i := by
    intro i
    rw [← Finset.mul_sum, (hq i).2, mul_one]
  exact (Finset.sum_congr rfl fun i _ => h i).trans hr.2

/-- The expected quadratic score against a family of targets is the score against their
average. -/
lemma expScore_brier_mix {ι : Type*} [Fintype ι] (p : X → ℝ) {r : ι → ℝ} {q : ι → X → ℝ}
    (hr : IsProbVec r) (hq : ∀ i, IsProbVec (q i)) :
    ∑ i, r i * expScore brier p (q i) = expScore brier p (mixVec r q) := by
  have hmix := isProbVec_mixVec hr hq
  simp only [fun i => expScore_brier p (q i) (hq i), expScore_brier p _ hmix]
  have h : ∀ i : ι, r i * ((∑ x, p x ^ 2) - 2 * (∑ x, q i x * p x) + 1)
      = ((∑ x, p x ^ 2) + 1) * r i - 2 * (∑ x, r i * (q i x * p x)) := by
    intro i
    rw [← Finset.mul_sum]
    ring
  simp only [h, Finset.sum_sub_distrib, ← Finset.mul_sum, hr.2]
  have hswap : ∑ i, r i * (∑ x, q i x * p x) = ∑ x, mixVec r q x * p x := by
    simp only [Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun x _ => ?_
    simp only [mixVec, Finset.sum_mul]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [hswap]
  ring

/-- **A model that cannot resolve the context is driven, by an honest score, to the
context-averaged ensemble.**  If one prediction has to serve a family of distinct targets
(the same input arising in several cellular contexts, say), then under the quadratic score
the unique optimum is their weighted average -- an ensemble that is generally the truth in
none of the contexts.  Only a model that conditions on the context can be optimal in each. -/
theorem brier_optimal_prediction_is_average {ι : Type*} [Fintype ι] {r : ι → ℝ}
    {q : ι → X → ℝ} (hr : IsProbVec r) (hq : ∀ i, IsProbVec (q i))
    (p : X → ℝ) (hp : IsProbVec p) (hne : p ≠ mixVec r q) :
    ∑ i, r i * expScore brier (mixVec r q) (q i) < ∑ i, r i * expScore brier p (q i) := by
  rw [expScore_brier_mix p hr hq, expScore_brier_mix (mixVec r q) hr hq]
  exact brier_strictly_proper p (mixVec r q) hp (isProbVec_mixVec hr hq) hne

/-! ## The sample-distance ("draw a structure and measure the deviation") score -/

/-- The ensemble average of a structural coordinate `c`. -/
def meanC (c : X → ℝ) (p : X → ℝ) : ℝ := ∑ x, p x * c x

/-- The ensemble variance of a structural coordinate `c`. -/
def varC (c : X → ℝ) (p : X → ℝ) : ℝ := ∑ x, p x * (c x - meanC c p) ^ 2

omit [DecidableEq X] in
lemma varC_nonneg (c p : X → ℝ) (hp : IsProbVec p) : 0 ≤ varC c p :=
  Finset.sum_nonneg fun x _ => mul_nonneg (hp.1 x) (sq_nonneg _)

omit [DecidableEq X] in
lemma varC_eq (c p : X → ℝ) (hp : IsProbVec p) :
    varC c p = (∑ x, p x * c x ^ 2) - (meanC c p) ^ 2 := by
  have h : ∀ x : X, p x * (c x - meanC c p) ^ 2
      = p x * c x ^ 2 - meanC c p * (p x * c x) - meanC c p * (p x * c x)
        + (meanC c p) ^ 2 * p x := by
    intro x; ring
  simp only [varC, h, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  rw [hp.2]
  simp only [meanC]
  ring

/-- The score in everyday use: a structure is drawn from the model and its squared
deviation from the observed structure, along the coordinate `c`, is reported. -/
def sqDistScore (c : X → ℝ) (p : X → ℝ) (y : X) : ℝ := ∑ x, p x * (c x - c y) ^ 2

omit [DecidableEq X] in
/-- **Exact expected sample-distance score.**  It is the sum of the *model's* dispersion,
the *truth's* dispersion, and the squared error of the mean. -/
theorem sqDistScore_expected (c p q : X → ℝ) (hp : IsProbVec p) (hq : IsProbVec q) :
    expScore (sqDistScore c) p q
      = varC c p + varC c q + (meanC c p - meanC c q) ^ 2 := by
  have hinner : ∀ y : X, sqDistScore c p y
      = (∑ x, p x * c x ^ 2) - c y * meanC c p - c y * meanC c p + c y ^ 2 := by
    intro y
    have h : ∀ x : X, p x * (c x - c y) ^ 2
        = p x * c x ^ 2 - c y * (p x * c x) - c y * (p x * c x) + c y ^ 2 * p x := by
      intro x; ring
    simp only [sqDistScore, h, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
    rw [hp.2]
    simp only [meanC]
    ring
  have hexp : expScore (sqDistScore c) p q
      = (∑ x, p x * c x ^ 2) - meanC c q * meanC c p - meanC c q * meanC c p
        + (∑ y, q y * c y ^ 2) := by
    simp only [expScore, hinner]
    have h : ∀ y : X, q y * ((∑ x, p x * c x ^ 2) - c y * meanC c p - c y * meanC c p + c y ^ 2)
        = (∑ x, p x * c x ^ 2) * q y - meanC c p * (q y * c y) - meanC c p * (q y * c y)
          + q y * c y ^ 2 := by
      intro y; ring
    simp only [h, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
    rw [hq.2]
    simp only [meanC]
    ring
  rw [hexp, varC_eq c p hp, varC_eq c q hq]
  ring

omit [DecidableEq X] in
/-- The sample-distance score sees the prediction only through two numbers -- its mean and
its variance along `c`.  Two models agreeing in those, however different their ensembles,
are scored identically against every target. -/
theorem sqDistScore_depends_on_two_moments (c p p' q : X → ℝ)
    (hp : IsProbVec p) (hp' : IsProbVec p') (hq : IsProbVec q)
    (hmean : meanC c p = meanC c p') (hvar : varC c p = varC c p') :
    expScore (sqDistScore c) p q = expScore (sqDistScore c) p' q := by
  rw [sqDistScore_expected c p q hp hq, sqDistScore_expected c p' q hp' hq, hmean, hvar]

omit [DecidableEq X] in
/-- **The sample-distance score rewards collapse.**  At a fixed mean, the score strictly
decreases as the prediction becomes *less* dispersed -- whatever the target.  A model is
therefore penalised precisely for reporting the disorder that is really there. -/
theorem sqDistScore_rewards_collapse (c p p' q : X → ℝ)
    (hp : IsProbVec p) (hp' : IsProbVec p') (hq : IsProbVec q)
    (hmean : meanC c p = meanC c p') (hvar : varC c p' < varC c p) :
    expScore (sqDistScore c) p' q < expScore (sqDistScore c) p q := by
  rw [sqDistScore_expected c p q hp hq, sqDistScore_expected c p' q hp' hq, hmean]
  linarith

/-! ## The best-of-`N` score -/

/-- The populated part of a prediction. -/
noncomputable def suppF (p : X → ℝ) : Finset X := Finset.univ.filter (fun x => 0 < p x)

omit [DecidableEq X] in
@[simp] lemma mem_suppF {p : X → ℝ} {x : X} : x ∈ suppF p ↔ 0 < p x := by
  simp [suppF]

omit [DecidableEq X] in
lemma suppF_nonempty {p : X → ℝ} (hp : IsProbVec p) : (suppF p).Nonempty := by
  by_contra hcon
  rw [Finset.not_nonempty_iff_eq_empty] at hcon
  have hzero : ∀ x : X, p x = 0 := by
    intro x
    have : x ∉ suppF p := by simp [hcon]
    rw [mem_suppF] at this
    exact le_antisymm (not_lt.1 this) (hp.1 x)
  have : (1 : ℝ) = 0 := by
    rw [← hp.2]; exact (Finset.sum_eq_zero fun x _ => hzero x)
  norm_num at this

/-- The best-of-`N` score: the model reports the structures it populates and is charged
only the deviation of its *closest* one from the observation. -/
noncomputable def bestOfScore (d : X → X → ℝ) (p : X → ℝ) (y : X) : ℝ :=
  if h : (suppF p).Nonempty then (suppF p).inf' h (fun x => d x y) else 0

omit [DecidableEq X] in
/-- **The best-of-`N` score is blind to the populations.**  Two models populating the same
conformations score identically, however different the weights they assign -- so the score
carries no information about the quantity a disorder model exists to predict. -/
theorem bestOfScore_eq_of_supp_eq (d : X → X → ℝ) {p p' : X → ℝ} (h : suppF p = suppF p') :
    bestOfScore d p = bestOfScore d p' := by
  funext y
  simp only [bestOfScore, h]

omit [DecidableEq X] in
/-- Hedging is free: enlarging the set of reported structures can only lower the score. -/
theorem bestOfScore_antitone_supp (d : X → X → ℝ) {p p' : X → ℝ}
    (hp : IsProbVec p) (hp' : IsProbVec p') (hsub : suppF p ⊆ suppF p') (y : X) :
    bestOfScore d p' y ≤ bestOfScore d p y := by
  simp only [bestOfScore, dif_pos (suppF_nonempty hp), dif_pos (suppF_nonempty hp')]
  refine Finset.le_inf' (suppF_nonempty hp) _ (fun x hx => ?_)
  exact Finset.inf'_le (fun x => d x y) (hsub hx)

omit [DecidableEq X] in
/-- A model that populates the observed conformation is charged nothing. -/
lemma bestOfScore_eq_zero_of_mem {d : X → X → ℝ} (hd0 : ∀ x y, 0 ≤ d x y)
    (hdrefl : ∀ y, d y y = 0) {p : X → ℝ} (hp : IsProbVec p) {y : X} (hy : y ∈ suppF p) :
    bestOfScore d p y = 0 := by
  simp only [bestOfScore, dif_pos (suppF_nonempty hp)]
  refine le_antisymm ?_ (Finset.le_inf' (suppF_nonempty hp) _ (fun x _ => hd0 x y))
  simpa [hdrefl y] using Finset.inf'_le (fun x => d x y) hy

omit [DecidableEq X] in
/-- **Any prediction that covers the truth scores a perfect zero.**  In particular the
truth itself, an arbitrarily badly weighted model on the same conformations, and the model
that simply lists the whole library are all tied at the optimum. -/
theorem bestOfScore_expected_eq_zero {d : X → X → ℝ} (hd0 : ∀ x y, 0 ≤ d x y)
    (hdrefl : ∀ y, d y y = 0) {p q : X → ℝ} (hp : IsProbVec p) (hq : IsProbVec q)
    (hsub : suppF q ⊆ suppF p) :
    expScore (bestOfScore d) p q = 0 := by
  refine Finset.sum_eq_zero (fun y _ => ?_)
  rcases lt_or_ge 0 (q y) with hy | hy
  · have hmem : y ∈ suppF p := hsub (mem_suppF.2 hy)
    rw [bestOfScore_eq_zero_of_mem hd0 hdrefl hp hmem, mul_zero]
  · have : q y = 0 := le_antisymm hy (hq.1 y)
    rw [this, zero_mul]

/-! ## Explicit witnesses -/

section Examples

/-- A three-state rotamer library, with a structural coordinate taking the values
`-1, 0, 1`. -/
def coord3 : Fin 3 → ℝ := fun i => (i : ℝ) - 1

/-- The disordered target: the three rotamers are equally populated. -/
noncomputable def unif3 : Fin 3 → ℝ := fun _ => 1/3

lemma isProbVec_unif3 : IsProbVec unif3 := by
  refine ⟨fun x => ?_, ?_⟩
  · fin_cases x <;> norm_num [unif3]
  · norm_num [unif3, Fin.sum_univ_three]

lemma meanC_unif3 : meanC coord3 unif3 = 0 := by
  norm_num [meanC, unif3, coord3, Fin.sum_univ_three]

lemma varC_unif3 : varC coord3 unif3 = 2/3 := by
  simp only [varC, meanC_unif3]
  norm_num [unif3, coord3, Fin.sum_univ_three]

lemma meanC_pointVec_one : meanC coord3 (pointVec (1 : Fin 3)) = 0 := by
  simp only [meanC, Fin.sum_univ_three, pointVec]
  norm_num [coord3, Fin.ext_iff]

lemma varC_pointVec_one : varC coord3 (pointVec (1 : Fin 3)) = 0 := by
  simp only [varC, meanC_pointVec_one, Fin.sum_univ_three, pointVec]
  norm_num [coord3, Fin.ext_iff]

/-- **The sample-distance score prefers the collapsed model on a disordered target.**  The
truthful three-state ensemble scores `4/3`; the single structure sitting at the mean --
which is wrong about every observable of the disorder -- scores `2/3`, i.e. exactly half.
The penalty the truth pays is precisely its own variance. -/
theorem sqDist_three_state :
    expScore (sqDistScore coord3) unif3 unif3 = 4/3 ∧
    expScore (sqDistScore coord3) (pointVec (1 : Fin 3)) unif3 = 2/3 := by
  constructor
  · rw [sqDistScore_expected coord3 unif3 unif3 isProbVec_unif3 isProbVec_unif3,
      varC_unif3, meanC_unif3]
    norm_num
  · rw [sqDistScore_expected coord3 (pointVec (1 : Fin 3)) unif3
      (isProbVec_pointVec _) isProbVec_unif3, varC_unif3, varC_pointVec_one,
      meanC_unif3, meanC_pointVec_one]
    norm_num

/-- Hence the sample-distance score is **not** strictly proper: it is not a metric one may
fit or rank disorder models by. -/
theorem sqDistScore_not_strictlyProper : ¬ StrictlyProper (sqDistScore coord3) := by
  intro hS
  have hne : pointVec (1 : Fin 3) ≠ unif3 := by
    intro h
    have hval := congrFun h 1
    norm_num [pointVec, unif3] at hval
  have hlt := hS (pointVec (1 : Fin 3)) unif3 (isProbVec_pointVec _) isProbVec_unif3 hne
  rw [sqDist_three_state.1, sqDist_three_state.2] at hlt
  norm_num at hlt

/-- Two conformations, and a distance between them. -/
def coordDist2 : Fin 2 → Fin 2 → ℝ := fun x y => |(x : ℝ) - (y : ℝ)|

/-- The truth: the two conformations are equally populated. -/
noncomputable def unif2 : Fin 2 → ℝ := fun _ => 1/2

/-- A model with the same two conformations but badly wrong populations. -/
noncomputable def skew2 : Fin 2 → ℝ := fun i => if i = 0 then 3/4 else 1/4

lemma isProbVec_unif2 : IsProbVec unif2 := by
  refine ⟨fun x => ?_, ?_⟩
  · fin_cases x <;> norm_num [unif2]
  · norm_num [unif2, Fin.sum_univ_two]

lemma isProbVec_skew2 : IsProbVec skew2 := by
  refine ⟨fun x => ?_, ?_⟩
  · fin_cases x <;> norm_num [skew2]
  · norm_num [skew2, Fin.sum_univ_two]

lemma suppF_unif2 : suppF unif2 = Finset.univ := by
  ext x
  fin_cases x <;> simp [unif2]

lemma suppF_skew2 : suppF skew2 = Finset.univ := by
  ext x
  fin_cases x <;> simp [skew2]

lemma coordDist2_nonneg (x y : Fin 2) : 0 ≤ coordDist2 x y := abs_nonneg _

lemma coordDist2_refl (y : Fin 2) : coordDist2 y y = 0 := by simp [coordDist2]

/-- **The best-of-`N` score cannot distinguish a badly weighted model from the truth.**
Both score a perfect zero, so the score is not strictly proper and, worse, carries no
information at all about the populations. -/
theorem bestOf_two_state :
    expScore (bestOfScore coordDist2) unif2 unif2 = 0 ∧
    expScore (bestOfScore coordDist2) skew2 unif2 = 0 := by
  refine ⟨bestOfScore_expected_eq_zero coordDist2_nonneg coordDist2_refl
      isProbVec_unif2 isProbVec_unif2 (Finset.Subset.refl _),
    bestOfScore_expected_eq_zero coordDist2_nonneg coordDist2_refl
      isProbVec_skew2 isProbVec_unif2 ?_⟩
  simp [suppF_unif2, suppF_skew2]

theorem bestOfScore_not_strictlyProper : ¬ StrictlyProper (bestOfScore coordDist2) := by
  intro hS
  have hne : skew2 ≠ unif2 := by
    intro h
    have hval := congrFun h 0
    norm_num [skew2, unif2] at hval
  have hlt := hS skew2 unif2 isProbVec_skew2 isProbVec_unif2 hne
  rw [bestOf_two_state.1, bestOf_two_state.2] at hlt
  norm_num at hlt

end Examples

end Scoring

end IDR
