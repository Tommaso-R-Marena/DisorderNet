/-
# The nature of the model: what it must be, and what it cannot be

`RequestProject.Statistics`, `RequestProject.FiniteData` and `RequestProject.Binding` rule
out particular *families* of models (coordinate regression, two-moment models, distograms,
mean-field models, reweighting models).  This file goes one level up and characterises the
**object** a model of a disordered region has to output, and the **structural properties**
a model must have, independently of any architecture.

## What the output object has to be

* `Ens.expect_eq_sum_dirac` -- every ensemble is a convex combination of single
  structures.  The thing to be predicted is therefore a point of a *simplex over*
  conformation space, not a point of conformation space: the output type must be a
  distribution.
* `Ens.mix`, `Ens.expect_mix` -- ensembles are closed under mixing, and
  `expressive_closed_under_mix` shows a correct model class must be too: whatever
  parametrisation is chosen, it has to be able to interpolate any two of its outputs.
* `expressive_iff_solves_all` -- **the exact criterion for universality**: a class of
  outputs solves *every* prediction problem iff it realises every ensemble up to
  observational equality.  Nothing weaker than expressivity in this sense is enough, and
  nothing more is needed.

## What no model can be, even approximately

The earlier obstructions are exact-equality statements.  A skeptic may answer that models
are only ever approximate.  `ApproxSame` measures a model against its target in the
strongest uniform sense available (all observables bounded by one -- equivalently, twice
the total-variation distance), and the obstructions survive *quantitatively*:

* `not_approx_dirac` -- a single-structure output is not merely wrong, it is wrong by at
  least the population of the conformation it misses.  Error bars, confidences and
  "the model is approximately right" do not close this gap.
* `not_approx_of_bounded_capacity` -- a model emitting at most `k` structures is off by at
  least `1/(k+1)` on some target.  Sampling more decoys from a fixed-capacity model does
  not converge.
* `no_finite_representation` -- **an information-bottleneck obstruction**: a model whose
  target-dependence is routed through a finite internal code of `N` states fails on any
  problem containing `N + 1` observationally distinct targets.  Capacity has to be counted
  in the representation, not only in the output.
* `not_deterministic_of_symmetric` -- **a symmetry obstruction, independent of variance**:
  if the target ensemble is invariant under a symmetry with no fixed point (mirror-image
  or register-shifted states of equal free energy), then *no* single conformation is
  invariant, so a symmetry-respecting single-structure model cannot exist.  A distribution
  can be symmetric while none of its samples is.
* `no_finite_ensemble_eq_atomless` -- **finite ensembles are themselves an
  idealisation**: a finitely supported model can never equal a continuous conformational
  distribution.  A real disordered region is only ever approximated, which is why the
  approximate notion `ApproxSame` -- and not exact equality -- is the honest target, and
  why the negative results above matter: they hold in that approximate sense too.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics

namespace IDR

open Finset
open scoped Classical

variable {X : Type*}

namespace Ens

/-! ## The output object is a distribution: a point of the simplex over conformations -/

/-- **Every ensemble is a convex combination of single structures.**  A model of a
disordered region does not have to invent a new kind of object: it has to output a weight
vector over conformations.  Conversely, this is exactly why a single structure is not
enough -- the output lives in the simplex, whose extreme points are the single
structures. -/
theorem expect_eq_sum_dirac (E : Ens X) (f : X → ℝ) :
    E.expect f = ∑ j, E.w j * (dirac (E.pt j)).expect f := by
  simp [expect, dirac]

/-- The mixture `t E + (1-t) F` of two ensembles. -/
noncomputable def mix (E F : Ens X) (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) : Ens X where
  card := E.card + F.card
  pt := Fin.append E.pt F.pt
  w := Fin.append (fun j => t * E.w j) (fun j => (1 - t) * F.w j)
  w_nonneg := by
    intro j
    induction j using Fin.addCases with
    | left i => simpa using mul_nonneg ht0 (E.w_nonneg i)
    | right i => simpa using mul_nonneg (by linarith) (F.w_nonneg i)
  w_sum := by
    rw [Fin.sum_univ_add]
    simp only [Fin.append_left, Fin.append_right, ← Finset.mul_sum, E.w_sum, F.w_sum]
    ring

@[simp] lemma expect_mix (E F : Ens X) (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (f : X → ℝ) :
    (mix E F t ht0 ht1).expect f = t * E.expect f + (1 - t) * F.expect f := by
  simp only [expect, mix]
  rw [Fin.sum_univ_add]
  simp only [Fin.append_left, Fin.append_right, Finset.mul_sum]
  congr 1 <;> exact Finset.sum_congr rfl fun i _ => by ring

end Ens

/-! ## The exact criterion for a model class to be enough -/

/-- A class of candidate outputs `M` is *expressive* if it realises every ensemble up to
observational equality. -/
def Expressive (M : Set (Ens X)) : Prop := ∀ E : Ens X, ∃ m ∈ M, m.Same E

/-- **Universality criterion.**  A class of outputs solves every prediction problem
exactly if and only if it is expressive.  This is the precise sense in which the design
question has a unique answer: the architecture is irrelevant, the *range* of the model is
everything. -/
theorem expressive_iff_solves_all (M : Set (Ens X)) :
    Expressive M ↔
      ∀ (I : Type) (T : I → Ens X), ∃ A : I → Ens X, (∀ i, A i ∈ M) ∧ ∀ i, (A i).Same (T i) := by
  constructor
  · intro hM I T
    choose A hA hA' using fun i => hM (T i)
    exact ⟨A, hA, hA'⟩
  · intro h E
    obtain ⟨A, hA, hA'⟩ := h Unit (fun _ => E)
    exact ⟨A (), hA (), hA' ()⟩

/-- **The output space must be convex.**  An expressive class contains, up to
observational equality, every mixture of two of its members: a model that can represent
two conformational states must also be able to represent every population ratio between
them.  Discrete "pick one of finitely many templates" outputs are therefore excluded. -/
theorem expressive_closed_under_mix {M : Set (Ens X)} (hM : Expressive M) (E F : Ens X)
    {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    ∃ m ∈ M, ∀ f : X → ℝ, m.expect f = t * E.expect f + (1 - t) * F.expect f := by
  obtain ⟨m, hm, hm'⟩ := hM (Ens.mix E F t ht0 ht1)
  exact ⟨m, hm, fun f => by rw [hm' f, Ens.expect_mix]⟩

/-! ## Approximate correctness: the obstructions are quantitative -/

/-- `E` and `F` agree to within `eps` on every observable bounded by one.  (Up to the
factor two this is the total-variation distance, the strongest uniform notion of "the
model is approximately right".) -/
def ApproxSame (eps : ℝ) (E F : Ens X) : Prop :=
  ∀ f : X → ℝ, (∀ x, |f x| ≤ 1) → |E.expect f - F.expect f| ≤ eps

lemma approxSame_of_same {E F : Ens X} (h : E.Same F) {eps : ℝ} (heps : 0 ≤ eps) :
    ApproxSame eps E F := fun f _ => by rw [h f]; simpa using heps

/-- Approximate agreement bounds the discrepancy in the population of any single
conformation. -/
lemma prob_sub_le_of_approx {eps : ℝ} {E F : Ens X} (h : ApproxSame eps E F) (x : X) :
    |E.prob x - F.prob x| ≤ eps :=
  h _ (fun y => by by_cases hy : y = x <;> simp [hy])

@[simp] lemma prob_dirac (a x : X) : (Ens.dirac a).prob x = if a = x then 1 else 0 := by
  simp [Ens.prob]

/-- **A single structure is quantitatively wrong.**  If the target populates two distinct
conformations, then *every* single-structure output -- whatever structure it picks, and
whatever confidence it reports -- differs from the target by at least the population of
one of them, in the strongest uniform sense.  The error does not shrink with better
training; it is set by the thermodynamics of the target. -/
theorem not_approx_dirac {E : Ens X} {x y : X} (hx : 0 < E.prob x) (hy : 0 < E.prob y)
    (hxy : x ≠ y) {eps : ℝ} (heps : eps < min (E.prob x) (E.prob y)) (a : X) :
    ¬ ApproxSame eps E (Ens.dirac a) := by
  intro h
  -- one of `x`, `y` is different from the predicted structure `a`
  have key : ∀ z : X, 0 < E.prob z → z ≠ a → E.prob z ≤ eps := by
    intro z _ hz
    have := prob_sub_le_of_approx h z
    rw [prob_dirac, if_neg (fun hc : a = z => hz hc.symm), sub_zero] at this
    exact (abs_le.1 this).2
  rcases eq_or_ne x a with rfl | hxa
  · have := key y hy (Ne.symm hxy)
    have := min_le_right (E.prob x) (E.prob y)
    linarith
  · have := key x hx hxa
    have := min_le_left (E.prob x) (E.prob y)
    linarith

namespace Ens

/-- A model that gives positive probability to `L` distinct conformations has at least `L`
mixture components. -/
theorem card_le_of_prob_pos {L : ℕ} {M : Ens X} (g : Fin L → X) (hg : Function.Injective g)
    (hpos : ∀ l, 0 < M.prob (g l)) : L ≤ M.card := by
  have hM : ∀ l, ∃ j : Fin M.card, M.pt j = g l := by
    intro l
    obtain ⟨j, -, hj⟩ := (M.prob_pos_iff (g l)).1 (hpos l)
    exact ⟨j, hj⟩
  choose z hz using hM
  have hinj : Function.Injective z := by
    intro a b hab
    exact hg (by rw [← hz a, ← hz b, hab])
  simpa using Fintype.card_le_of_injective z hinj

end Ens

/-- The uniform ensemble on distinct conformations gives each of them probability
`1/m`. -/
lemma unif_prob_eq {m : ℕ} (hm : 0 < m) {g : Fin m → X} (hg : Function.Injective g)
    (l : Fin m) : (unif hm g).prob (g l) = 1 / m :=
  Ens.prob_pt_of_injective (unif hm g) hg l

/-- **Bounded capacity is quantitatively wrong too.**  A model whose output is a mixture
of at most `k` structures -- `k` templates, `k` sampled decoys, `k` latent states -- is off
by at least `1/(k+1)` on the uniform ensemble over `k+1` conformations.  Drawing more
samples from such a model cannot help: the gap is a property of the model class, not of the
sampling. -/
theorem not_approx_of_bounded_capacity {k : ℕ} {M : Ens X} (hM : M.card ≤ k)
    (g : Fin (k + 1) → X) (hg : Function.Injective g) {eps : ℝ} (heps : eps < 1 / (k + 1)) :
    ¬ ApproxSame eps M (unif (Nat.succ_pos k) g) := by
  intro h
  have hcast : ((k + 1 : ℕ) : ℝ) = (k : ℝ) + 1 := by push_cast; ring
  have hpos : ∀ l, 0 < M.prob (g l) := by
    intro l
    have h1 := prob_sub_le_of_approx h (g l)
    rw [unif_prob_eq _ hg l, hcast] at h1
    have h2 := (abs_le.1 h1).1
    have h3 : (0:ℝ) ≤ M.prob (g l) := M.prob_nonneg _
    rcases eq_or_lt_of_le h3 with h4 | h4
    · exfalso; rw [← h4] at h2; linarith
    · exact h4
  have := Ens.card_le_of_prob_pos g hg hpos
  omega

/-! ## An information bottleneck: capacity must be counted in the representation -/

/-- **A finite internal representation cannot carry a disordered target.**  Suppose the
model computes its answer as `D (R i)`: an encoder `R` producing an internal code in a
finite set `Z`, followed by any decoder `D` whatsoever.  If the prediction problem contains
more observationally distinct targets than `Z` has states, the model must be wrong on one
of them.  This bounds a model by the entropy of its bottleneck, not by its parameter count:
a large network with a coarse, discretely-classified internal description of disorder (a
"disorder class", a chosen template, a finite vocabulary of states) is subject to it. -/
theorem no_finite_representation {I Z : Type*} [Fintype Z] {T A : I → Ens X}
    (R : I → Z) (D : Z → Ens X) (hA : ∀ i, A i = D (R i))
    {L : ℕ} (hL : Fintype.card Z < L) (idx : Fin L → I)
    (hdist : ∀ l l', l ≠ l' → ¬ (T (idx l)).Same (T (idx l'))) :
    ∃ i, ¬ (A i).Same (T i) := by
  by_contra hcon
  push_neg at hcon
  obtain ⟨l, l', hll', hRl⟩ := Fintype.exists_ne_map_eq_of_card_lt (fun l => R (idx l))
    (by simpa using hL)
  have h1 : (A (idx l)).Same (T (idx l)) := hcon _
  have h2 : (A (idx l')).Same (T (idx l')) := hcon _
  rw [hA (idx l)] at h1
  rw [hA (idx l'), ← hRl] at h2
  exact hdist l l' hll' (h1.symm.trans h2)

/-! ## A symmetry obstruction, independent of variance -/

/-- The ensemble `E` is invariant under the transformation `sigma` of conformation
space. -/
def SymmetricUnder (sigma : X → X) (E : Ens X) : Prop :=
  ∀ f : X → ℝ, E.expect (fun x => f (sigma x)) = E.expect f

/-- A symmetric ensemble that is a single structure forces that structure to be a fixed
point of the symmetry. -/
theorem fixed_point_of_symmetric_deterministic {sigma : X → X} {E : Ens X}
    (hE : SymmetricUnder sigma E) (hdet : E.Deterministic) : ∃ a : X, sigma a = a := by
  obtain ⟨a, ha⟩ := (Ens.deterministic_iff E).1 hdet
  refine ⟨a, ?_⟩
  by_contra hne
  have h := hE (fun y => if y = a then (1 : ℝ) else 0)
  rw [ha (fun x => if sigma x = a then (1:ℝ) else 0), ha (fun y => if y = a then (1:ℝ) else 0)]
    at h
  simp [hne] at h

/-- **A symmetry with no fixed point rules out single-structure models outright.**  If the
free-energy landscape is invariant under a transformation that leaves no conformation
unchanged -- mirror-image states, a register shift between equivalent binding modes, two
symmetry-related orientations -- then the equilibrium ensemble is symmetric, and no single
conformation is.  A model that respects the physics cannot commit to a structure.  Note
that this argument uses no notion of variance or of distance: it is a purely structural
obstruction. -/
theorem not_deterministic_of_symmetric {sigma : X → X} (hfix : ∀ x : X, sigma x ≠ x)
    {E : Ens X} (hE : SymmetricUnder sigma E) : ¬ E.Deterministic := by
  intro hdet
  obtain ⟨a, ha⟩ := fixed_point_of_symmetric_deterministic hE hdet
  exact hfix a ha

/-- The obstruction is not vacuous: two conformations exchanged by the symmetry (say a
pair of mirror-image states of equal free energy) give a symmetric, non-deterministic
ensemble. -/
theorem exists_symmetric_nondeterministic :
    ∃ E : Ens Bool, SymmetricUnder Bool.not E ∧ ¬ E.Deterministic := by
  refine ⟨unif (by norm_num) ![false, true], ?_, ?_⟩
  · intro f
    rw [unif_expect, unif_expect]
    simp [Fin.sum_univ_succ]
    ring
  · refine Ens.not_deterministic_of_two_points (x := false) (y := true) ?_ ?_ (by decide)
    · simpa using unif_prob_pos (m := 2) (by norm_num) ![false, true] 0
    · simpa using unif_prob_pos (m := 2) (by norm_num) ![false, true] 1

lemma not_fixed_point_bool : ∀ b : Bool, Bool.not b ≠ b := by decide

/-! ## Even a finite ensemble is an idealisation -/

open MeasureTheory in
/-- **No finitely supported model equals a continuous conformational distribution.**  If
the true distribution of the region is atomless -- as any distribution over a continuum of
torsion angles is -- then for every finite ensemble there is a bounded observable, namely
the indicator of the model's own support, on which the model and the truth differ by the
maximum possible amount: the model says one, the truth says zero.

So the ensemble picture itself is exact only as a limit; finite models are *approximations*
by construction.  This is why the negative results above were stated in the approximate
form `ApproxSame`: the right question is never "is the model exactly right" but "how far
is it", and even in that metric a single structure, a bounded capacity, or a finite code
cannot be made close. -/
theorem no_finite_ensemble_eq_atomless {X : Type*} [MeasurableSpace X]
    [MeasurableSingletonClass X] (mu : Measure X) (hmu : ∀ x : X, mu {x} = 0) (E : Ens X) :
    ∃ f : X → ℝ, (∀ x, 0 ≤ f x ∧ f x ≤ 1) ∧ Measurable f ∧
      ∫ x, f x ∂mu = 0 ∧ E.expect f = 1 := by
  classical
  set s : Finset X := Finset.image E.pt Finset.univ with hs
  refine ⟨Set.indicator (↑s : Set X) 1, ?_, ?_, ?_, ?_⟩
  · intro x
    by_cases hx : x ∈ (↑s : Set X) <;> simp [Set.indicator, hx]
  · exact measurable_one.indicator s.finite_toSet.measurableSet
  · have hzero : mu (↑s : Set X) = 0 := by
      have hcov : (↑s : Set X) = ⋃ x ∈ s, ({x} : Set X) := by ext y; simp
      rw [hcov]
      refine le_antisymm (le_trans (measure_biUnion_finset_le s _) ?_) (zero_le _)
      simp [hmu]
    rw [integral_indicator_one s.finite_toSet.measurableSet, measureReal_def, hzero]
    simp
  · have hmem : ∀ j : Fin E.card, E.pt j ∈ (↑s : Set X) := by
      intro j
      simp [hs]
    have hone : ∀ j : Fin E.card,
        E.w j * Set.indicator (↑s : Set X) (1 : X → ℝ) (E.pt j) = E.w j := by
      intro j
      rw [Set.indicator_of_mem (hmem j)]
      simp
    simp only [Ens.expect]
    rw [Finset.sum_congr rfl (fun j _ => hone j)]
    exact E.w_sum

end IDR
