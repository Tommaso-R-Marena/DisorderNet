/-
# Geometric statistics of a conformational ensemble

Specialising the general framework of `RequestProject.EnsembleCore` to the conformation
space `Conf n = Fin n → ℝ` used in `RequestProject.DisorderedRegions`, we introduce the
mean conformation, the per-degree-of-freedom variance, the covariance between two degrees
of freedom, and the notion of a *disordered* degree of freedom.

The main results here are

* `Ens.var_pos_of_two_points` -- two populated conformations that differ at `i` make `i`
  disordered;
* `mean_sq_lt_of_disordered` -- **averaging destroys geometry**: if every populated
  conformation satisfies a hard geometric constraint (here: lying on a sphere, the
  idealisation of fixed bond lengths / a fixed radius of gyration) and the ensemble is
  disordered, then the *mean* conformation strictly violates that constraint.  The
  structure minimising the mean squared error is therefore not even a physically
  admissible structure.
-/
import Mathlib
import RequestProject.DisorderedRegions
import RequestProject.EnsembleCore

namespace IDR

open Finset
open scoped Classical

namespace Ens

variable {n : ℕ} (E : Ens (Conf n))

/-- Mean value of degree of freedom `i`. -/
def mean (i : Fin n) : ℝ := E.expect (fun c => c i)

/-- Variance of degree of freedom `i`. -/
def var (i : Fin n) : ℝ := E.expect (fun c => (c i - E.mean i) ^ 2)

/-- Covariance of degrees of freedom `i` and `j`. -/
def cov (i j : Fin n) : ℝ := E.expect (fun c => (c i - E.mean i) * (c j - E.mean j))

/-- Total (trace) variance. -/
def totalVar : ℝ := ∑ i, E.var i

/-- Degree of freedom `i` is *disordered*: its marginal is non-degenerate. -/
def Disordered (i : Fin n) : Prop := 0 < E.var i

lemma var_nonneg (i : Fin n) : 0 ≤ E.var i := E.expect_nonneg fun _ => sq_nonneg _

lemma totalVar_nonneg : 0 ≤ E.totalVar := Finset.sum_nonneg fun i _ => E.var_nonneg i

lemma var_le_totalVar (i : Fin n) : E.var i ≤ E.totalVar :=
  Finset.single_le_sum (f := fun i => E.var i) (fun i _ => E.var_nonneg i) (Finset.mem_univ i)

lemma expect_congr_on_support {X : Type*} (E : Ens X) {f g : X → ℝ}
    (h : ∀ j, 0 < E.w j → f (E.pt j) = g (E.pt j)) : E.expect f = E.expect g := by
  refine Finset.sum_congr rfl fun j _ => ?_
  rcases eq_or_lt_of_le (E.w_nonneg j) with hw | hw
  · rw [← hw]; ring
  · rw [h j hw]

/-- Variance as second moment minus squared mean. -/
lemma var_eq_sq (i : Fin n) : E.var i = E.expect (fun c => (c i) ^ 2) - (E.mean i) ^ 2 := by
  have hexp : ∀ c : Conf n, (c i - E.mean i) ^ 2
      = (c i) ^ 2 + (-(2 * E.mean i)) * (c i) + (E.mean i) ^ 2 := fun c => by ring
  simp only [var, expect] at *
  rw [Finset.sum_congr rfl (fun j _ => by rw [hexp (E.pt j)])]
  have : ∀ j : Fin E.card, E.w j * ((E.pt j i) ^ 2 + (-(2 * E.mean i)) * (E.pt j i)
      + (E.mean i) ^ 2)
      = E.w j * (E.pt j i) ^ 2 + (-(2 * E.mean i)) * (E.w j * E.pt j i)
        + (E.mean i) ^ 2 * E.w j := fun j => by ring
  rw [Finset.sum_congr rfl (fun j _ => this j), Finset.sum_add_distrib, Finset.sum_add_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, E.w_sum]
  have hm : ∑ j, E.w j * E.pt j i = E.mean i := rfl
  rw [hm]
  ring

/-- Two populated conformations differing at `i` make `i` disordered. -/
theorem var_pos_of_two_points {x y : Conf n} {i : Fin n} (hx : 0 < E.prob x)
    (hy : 0 < E.prob y) (hxy : x i ≠ y i) : 0 < E.var i := by
  -- one of the two must differ from the mean
  have key : ∀ z : Conf n, 0 < E.prob z → z i ≠ E.mean i → 0 < E.var i := by
    intro z hz hne
    obtain ⟨j, hwj, hpj⟩ := (E.prob_pos_iff z).1 hz
    have hterm : 0 < E.w j * (E.pt j i - E.mean i) ^ 2 := by
      refine mul_pos hwj ?_
      have : E.pt j i - E.mean i ≠ 0 := by
        rw [hpj]; exact sub_ne_zero.mpr hne
      positivity
    refine lt_of_lt_of_le hterm ?_
    exact Finset.single_le_sum
      (f := fun j => E.w j * (E.pt j i - E.mean i) ^ 2)
      (fun j _ => mul_nonneg (E.w_nonneg j) (sq_nonneg _)) (Finset.mem_univ j)
  by_cases hxm : x i = E.mean i
  · exact key y hy (fun h => hxy (hxm.trans h.symm))
  · exact key x hx hxm

/-- **Averaging destroys geometry.**  Suppose every populated conformation lies on the
sphere of radius `r` (a stand-in for the hard geometric constraints of a real polypeptide:
fixed bond lengths, excluded volume, a definite radius of gyration).  If the ensemble has
a disordered degree of freedom, the mean conformation -- the unique minimiser of the mean
squared error, hence the target of any regression-trained single-structure predictor --
lies strictly inside the sphere and so is *not itself an admissible conformation*. -/
theorem mean_sq_lt_of_disordered {r : ℝ}
    (hE : ∀ j, 0 < E.w j → ∑ i, (E.pt j i) ^ 2 = r ^ 2)
    {i : Fin n} (hi : E.Disordered i) : ∑ i, (E.mean i) ^ 2 < r ^ 2 := by
  have h1 : E.expect (fun c => ∑ i, (c i) ^ 2) = r ^ 2 := by
    rw [expect_congr_on_support E (g := fun _ => r ^ 2) (fun j hj => hE j hj)]
    simp
  have h2 : ∑ i, E.expect (fun c => (c i) ^ 2) = r ^ 2 := by
    rw [← h1, expect_sum]
  have h3 : ∀ i : Fin n, E.expect (fun c => (c i) ^ 2) = E.var i + (E.mean i) ^ 2 := by
    intro i; rw [var_eq_sq]; ring
  have h4 : ∑ i, (E.var i + (E.mean i) ^ 2) = r ^ 2 := by
    rw [← h2]
    exact (Finset.sum_congr rfl fun i _ => (h3 i)).symm
  rw [Finset.sum_add_distrib] at h4
  have h5 : E.var i ≤ ∑ i, E.var i :=
    Finset.single_le_sum (f := fun i => E.var i) (fun i _ => E.var_nonneg i) (Finset.mem_univ i)
  have := hi
  simp only [Disordered] at this
  linarith

end Ens

/-! ## Bridge to the concrete setting of `RequestProject.DisorderedRegions` -/

/-- Every `Ensemble n` of the concrete development is an `Ens (Conf n)`. -/
def Ensemble.toEns {n : ℕ} (E : Ensemble n) : Ens (Conf n) where
  card := E.card
  pt := E.conf
  w := E.w
  w_nonneg := E.w_nonneg
  w_sum := E.w_sum

@[simp] lemma Ensemble.toEns_expect {n : ℕ} (E : Ensemble n) (f : Conf n → ℝ) :
    E.toEns.expect f = E.expect f := rfl

@[simp] lemma Ensemble.toEns_mean {n : ℕ} (E : Ensemble n) (i : Fin n) :
    E.toEns.mean i = E.mean i := rfl

@[simp] lemma Ensemble.toEns_var {n : ℕ} (E : Ensemble n) (i : Fin n) :
    E.toEns.var i = E.variance i := rfl

lemma Ensemble.toEns_disordered {n : ℕ} (E : Ensemble n) (i : Fin n) :
    E.toEns.Disordered i ↔ E.Disordered i := Iff.rfl

/-- ... and conversely. -/
def Ens.toEnsemble {n : ℕ} (E : Ens (Conf n)) : Ensemble n where
  card := E.card
  conf := E.pt
  w := E.w
  w_nonneg := E.w_nonneg
  w_sum := E.w_sum

@[simp] lemma Ens.toEnsemble_expect {n : ℕ} (E : Ens (Conf n)) (f : Conf n → ℝ) :
    E.toEnsemble.expect f = E.expect f := rfl

@[simp] lemma Ens.toEnsemble_variance {n : ℕ} (E : Ens (Conf n)) (i : Fin n) :
    E.toEnsemble.variance i = E.var i := rfl

/-- Mean squared deviation of the ensemble from a single predicted structure `x`. -/
def Ens.pointLoss {n : ℕ} (E : Ens (Conf n)) (x : Conf n) : ℝ :=
  E.expect (fun c => ∑ i, (c i - x i) ^ 2)

/-- **Bias-variance decomposition** in the general setting: the squared error of a
single-structure prediction is the total variance of the ensemble plus the squared bias.
The first term is irreducible. -/
theorem Ens.pointLoss_eq {n : ℕ} (E : Ens (Conf n)) (x : Conf n) :
    E.pointLoss x = E.totalVar + ∑ i, (E.mean i - x i) ^ 2 :=
  IDR.pointLoss_eq E.toEnsemble x

/-- Every single-structure prediction has squared error at least the variance of any one
degree of freedom; on a disordered region this is a strictly positive floor. -/
theorem Ens.var_le_pointLoss {n : ℕ} (E : Ens (Conf n)) (i : Fin n) (x : Conf n) :
    E.var i ≤ E.pointLoss x :=
  IDR.variance_le_pointLoss E.toEnsemble i x

/-- **The sharp boundary between order and disorder.**  A single-structure model is
exactly right on an ensemble if and only if every degree of freedom has zero variance,
i.e. the region is rigidly ordered.  Structure prediction is therefore correct precisely
on the folded case and incorrect on every disordered one. -/
theorem Ens.deterministic_iff_var_zero {n : ℕ} (E : Ens (Conf n)) :
    E.Deterministic ↔ ∀ i, E.var i = 0 := by
  rw [Ens.deterministic_iff]
  constructor
  · intro h
    exact fun i => (variance_all_zero_iff_dirac E.toEnsemble).2 h i
  · intro h
    exact (variance_all_zero_iff_dirac E.toEnsemble).1 h

end IDR
