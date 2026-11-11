/-
# Part XCVII  Nonlinear reports: what a ratio, a mode or a fitted exponent is worth

Part LXX characterises exactly which *linear* functionals of an ensemble the data determine: the
span of normalisation and the measured observables, and nothing else.  The assumptions list
recorded the restriction: "Nonlinear summaries (a mode of a distribution, a ratio of populations,
an exponent extracted by fitting) are not covered by the characterisation."  Almost every number
quoted from an ensemble is of that kind.  This file covers them.

The object is the **feasible set** `feasible g p`: every genuine ensemble reproducing the measured
averages of the target.  A report `F` — now an arbitrary function of the ensemble, not necessarily
linear — is *determined* when it takes the same value everywhere on that set.

* `feasible_convex` — the feasible set is convex, hence connected.  This is what makes the
  nonlinear theory possible at all.
* `determinedF_comp` — **the positive half, and it is a usable rule.**  *Any* function whatsoever
  of finitely many identifiable linear functionals is determined: a ratio of two secondary-structure
  contents, a nonlinear calibration curve applied to a measured average, a difference of logarithms.
  So nonlinearity is not itself the problem; leaving the measured span is.
* `not_determinedF_ratio` — **the negative half, on the canonical nonlinear summary.**  A ratio of
  two conformer populations is not determined whenever the data are blind to transfer of population
  between those two conformers — even though every measured average is reproduced exactly, and even
  at an interior target.  The proof exhibits the competing ensemble.
* `reachable_interval` — **and the failure is not a two-point ambiguity but an interval.**  For a
  continuous report, if some feasible ensemble gives a different value then *every* value in
  between is also attained by a feasible ensemble.  A non-identifiable nonlinear number therefore
  has a whole interval of values consistent with the data, which is the honest error bar and is
  not the one a fitting procedure reports.
* `reachable_interval_of_ratio` — the two combined for the ratio: a genuine continuum of
  population ratios fits the data equally well.

The protocol this proves correct is the one Part LXX already recommended, now extended: quote a
nonlinear number only when it is exhibited as a function of identifiable linear functionals.  If it
cannot be so exhibited, what should be reported is the interval, not the fitted value.
-/
import Mathlib
import RequestProject.Identifiability

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR

namespace NonlinearIdentify

open Finset IDR.Restraint IDR.Identify

variable {m k : ℕ}

/-- Every genuine ensemble that reproduces the measured averages of the target. -/
def feasible (g : Fin k → Fin m → ℝ) (p : Fin m → ℝ) : Set (Fin m → ℝ) :=
  {q | IsEns q ∧ ∀ j, obs (g j) q = obs (g j) p}

/-- An arbitrary — possibly nonlinear — report of the ensemble is *determined* when it takes the
same value on every ensemble consistent with the data. -/
def DeterminedF (g : Fin k → Fin m → ℝ) (p : Fin m → ℝ) (F : (Fin m → ℝ) → ℝ) : Prop :=
  ∀ q ∈ feasible g p, F q = F p

lemma mem_feasible_self {g : Fin k → Fin m → ℝ} {p : Fin m → ℝ} (hp : IsEns p) :
    p ∈ feasible g p := ⟨hp, fun _ => rfl⟩

/-- The feasible set is convex: it is the simplex cut by linear equations. -/
theorem feasible_convex (g : Fin k → Fin m → ℝ) (p : Fin m → ℝ) : Convex ℝ (feasible g p) := by
  intro q1 hq1 q2 hq2 a b ha hb hab
  refine ⟨⟨fun i => ?_, ?_⟩, fun j => ?_⟩
  · have h1 := hq1.1.1 i
    have h2 := hq2.1.1 i
    have : a * q1 i + b * q2 i ≥ 0 := by nlinarith
    simpa using this
  · have h1 := hq1.1.2
    have h2 := hq2.1.2
    have : ∑ i, (a * q1 i + b * q2 i) = a * (∑ i, q1 i) + b * (∑ i, q2 i) := by
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    simpa [h1, h2, hab] using this
  · have h1 := hq1.2 j
    have h2 := hq2.2 j
    have hlin : obs (g j) (fun i => a * q1 i + b * q2 i)
        = a * obs (g j) q1 + b * obs (g j) q2 := by
      simp only [obs, Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    have : obs (g j) (fun i => a * q1 i + b * q2 i) = obs (g j) p := by
      rw [hlin, h1, h2, ← add_mul, hab, one_mul]
    simpa using this

/-- **Any function of identifiable linear functionals is identifiable.**  Nonlinearity is not the
obstruction; leaving the measured span is. -/
theorem determinedF_comp {r : ℕ} (g : Fin k → Fin m → ℝ) {p : Fin m → ℝ} (hp1 : ∑ i, p i = 1)
    (f : Fin r → (Fin m → ℝ)) (hf : ∀ l, f l ∈ measured g) (Phi : (Fin r → ℝ) → ℝ) :
    DeterminedF g p (fun q => Phi (fun l => obs (f l) q)) := by
  intro q hq
  have : ∀ l, obs (f l) q = obs (f l) p := fun l =>
    determined_of_mem g hp1 (hf l) q hq.1 hq.2
  simp only []
  congr 1
  funext l
  exact this l

/-- The direction that moves population from conformation `j` to conformation `i`. -/
noncomputable def transferDir (i j : Fin m) : Fin m → ℝ :=
  fun t => (if t = i then (1:ℝ) else 0) - (if t = j then (1:ℝ) else 0)

lemma transferDir_sum (i j : Fin m) : ∑ t, transferDir i j t = 0 := by
  unfold transferDir
  rw [Finset.sum_sub_distrib]
  simp

lemma transferDir_apply_self {i j : Fin m} (hij : i ≠ j) : transferDir i j i = 1 := by
  unfold transferDir
  simp [hij]

lemma transferDir_apply_other {i j : Fin m} (hij : i ≠ j) : transferDir i j j = -1 := by
  unfold transferDir
  simp [Ne.symm hij]

lemma transferDir_ne_zero {i j : Fin m} (hij : i ≠ j) : transferDir i j ≠ 0 := by
  intro h
  have h1 : transferDir i j i = 0 := by rw [h]; rfl
  rw [transferDir_apply_self hij] at h1
  norm_num at h1

/-- **A ratio of populations is not determined** whenever the data are blind to transfer of
population between the two conformers, however many restraints there are and however interior the
target. -/
theorem not_determinedF_ratio (g : Fin k → Fin m → ℝ) {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d)
    (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1) {i j : Fin m} (hij : i ≠ j)
    (hblind : ∀ l, obs (g l) (transferDir i j) = 0) :
    ¬ DeterminedF g p (fun q => q i / q j) := by
  obtain ⟨c, hc, hens, -⟩ :=
    exists_perturbation hd hp hp1 (transferDir_ne_zero hij) (transferDir_sum i j)
  set q : Fin m → ℝ := fun t => p t + c * transferDir i j t with hq
  have hqi : q i = p i + c := by
    rw [hq]; simp only []; rw [transferDir_apply_self hij]; ring
  have hqj : q j = p j - c := by
    rw [hq]; simp only []; rw [transferDir_apply_other hij]; ring
  have hfeas : q ∈ feasible g p := by
    refine ⟨hens, fun l => ?_⟩
    rw [hq, obs_add_smul, hblind l, mul_zero, add_zero]
  have hpi : 0 < p i := lt_of_lt_of_le hd (hp i)
  have hpj : 0 < p j := lt_of_lt_of_le hd (hp j)
  intro hdet
  have hval : q i / q j = p i / p j := hdet q hfeas
  rw [hqi, hqj] at hval
  rcases le_or_gt (p j - c) 0 with hle | hlt
  · -- the competing ensemble empties conformation `j`
    have hzero : p j - c = 0 := le_antisymm hle (by simpa [hqj] using hens.1 j)
    rw [hzero, div_zero] at hval
    have hpos : 0 < p i / p j := by positivity
    rw [← hval] at hpos
    exact lt_irrefl _ hpos
  · -- or it strictly increases the ratio
    have hgt : p i / p j < (p i + c) / (p j - c) := by
      rw [div_lt_div_iff₀ hpj hlt]
      nlinarith
    rw [hval] at hgt
    exact lt_irrefl _ hgt

/-- **A non-identifiable continuous report has an interval of values, not two.**  If some feasible
ensemble gives a different value of a continuous report, every intermediate value is also given by
a feasible ensemble. -/
theorem reachable_interval (g : Fin k → Fin m → ℝ) {p : Fin m → ℝ} (hp : IsEns p)
    {F : (Fin m → ℝ) → ℝ} (hF : ContinuousOn F (feasible g p)) {q : Fin m → ℝ}
    (hq : q ∈ feasible g p) :
    Set.Icc (F p) (F q) ⊆ F '' (feasible g p) :=
  (feasible_convex g p).isPreconnected.intermediate_value (mem_feasible_self hp) hq hF

/-- The two halves combined on the ratio: between the fitted ratio and the competing one, every
value fits the data equally well. -/
theorem reachable_interval_of_ratio (g : Fin k → Fin m → ℝ) {p : Fin m → ℝ} (hp : IsEns p)
    {i j : Fin m} (hj : ∀ q ∈ feasible g p, q j ≠ 0)
    {q : Fin m → ℝ} (hq : q ∈ feasible g p) :
    Set.Icc (p i / p j) (q i / q j) ⊆ (fun x : Fin m → ℝ => x i / x j) '' (feasible g p) :=
  reachable_interval g hp
    (ContinuousOn.div (Continuous.continuousOn (continuous_apply i))
      (Continuous.continuousOn (continuous_apply j)) hj) hq

end NonlinearIdentify

end IDR
