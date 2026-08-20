/-
# Part CXLIV  Which properties of a disordered ensemble a titration actually measures

Part CXLII showed that an arithmetic ladder of enough conditions determines the internal-distance
distribution exactly, and Part CXLIII that the inversion amplifies reading error by an explicit
constant which diverges as two distances approach each other.  Both statements are about the
*weights themselves*.  A model of a disordered region is rarely asked for a weight; it is asked for
a property of the ensemble — a population, a mean, a fraction of expanded conformers — that is, for
a linear functional `L(w) = ∑_t c_t · w_t` of the distance distribution.  Different functionals are
measured with wildly different reliability, and this part says exactly how reliable each one is.

* **The representer.**  A functional is measured through the ladder only in so far as it can be
  written as a combination of the readings.  `functional_stability` — if a polynomial `P` of degree
  below the number of conditions satisfies the *representer equation*

  `P(e^{−h t}) = c_t · t · e^{κ₀ t}`  at every distance `t` of the ensemble,

  then for any two distributions whose ladder readings agree to within `δ`,

  `|L(w) − L(w')| ≤ ‖P‖₁ · δ`,   `‖P‖₁ = ∑_{j<n} |P.coeff j|`.

  The amplification of a property of the ensemble is the ℓ¹ norm of its representer on the nodes —
  not a property of the experiment alone and not a property of the functional alone, but of the two
  together.  Part CXLIII's weight bound is the case where `c` is a point mass and the representer is
  the Lagrange basis polynomial.

* **The measured properties are the screened populations.**  `screened_population_stable` — the
  functional `w ↦ ∑_t w_t e^{−κ_j t}/t`, the screened population read at the `j`-th condition of the
  ladder, has representer `X^j`, hence amplification exactly `1`: it is measured with no error
  amplification whatsoever, at every condition of the ladder.  These functionals — and their
  combinations, with the ℓ¹ cost of the combination — are what a titration reports.  Anything else
  is an extrapolation whose price is its representer norm, and Part CXLII showed that for the
  individual weights of nearby distances that price is unbounded.

* `ensemble_functional_law` collects the two, with the resulting rule: **report the functional and
  its representer norm.**  A model of a disordered region calibrated on a titration should quote the
  properties it claims to have measured together with the amplification each one carries; a property
  whose representer norm is large is a prediction of the model, not a measurement.
-/
import Mathlib
import RequestProject.SaltResponsiveEnsemble

set_option autoImplicit false

namespace IDR
namespace EnsembleFunctionals

open Finset Polynomial

/-- The ℓ¹ norm of the first `n` coefficients of a representer polynomial: the factor by which the
functional it represents amplifies reading error. -/
noncomputable def representerNorm (P : ℝ[X]) (n : ℕ) : ℝ :=
  ∑ j ∈ Finset.range n, |P.coeff j|

/-- The difference of two ladder readings, as a power sum in the Vandermonde nodes. -/
theorem reading_diff_pow (T T' : Finset ℝ) (w w' : ℝ → ℝ) (kappa0 hstep : ℝ) (j : ℕ) :
    DistanceEnsemble.measCurve T w (TitrationLadder.ladderPoint kappa0 hstep j)
      - DistanceEnsemble.measCurve T' w' (TitrationLadder.ladderPoint kappa0 hstep j)
      = ∑ t ∈ T ∪ T',
          (((DistanceEnsemble.ext T w t - DistanceEnsemble.ext T' w' t) / t)
              * Real.exp (-(kappa0 * t)))
            * (Real.exp (-(hstep * t))) ^ j := by
  rw [SaltResponsive.curve_diff_expSum T T' w w']
  refine Finset.sum_congr rfl fun t _ => ?_
  have hpow : Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t))
      = Real.exp (-(kappa0 * t)) * (Real.exp (-(hstep * t))) ^ j := by
    rw [← Real.exp_nat_mul, ← Real.exp_add, TitrationLadder.ladderPoint]
    ring_nf
  rw [hpow]
  ring

/-- **The stability of a property of the ensemble is the norm of its representer.**  Let
`L(w) = ∑_t c_t w_t` be a linear functional of the distance distribution, and let `P` be a
polynomial of degree below the number `n` of ladder conditions solving the representer equation
`P(e^{−ht}) = c_t · t · e^{κ₀t}` at every distance realised by either ensemble.  If the ladder
readings of two distributions agree to within `δ`, the functional agrees to within
`‖P‖₁ · δ`. -/
theorem functional_stability {T T' : Finset ℝ} {w w' c : ℝ → ℝ} {P : ℝ[X]}
    {kappa0 hstep delta : ℝ} {n : ℕ}
    (hpos : ∀ t ∈ T ∪ T', 0 < t) (hdeg : P.natDegree < n)
    (hrep : ∀ t ∈ T ∪ T', P.eval (Real.exp (-(hstep * t))) = c t * t * Real.exp (kappa0 * t))
    (h : ∀ j < n, |DistanceEnsemble.measCurve T w (TitrationLadder.ladderPoint kappa0 hstep j)
      - DistanceEnsemble.measCurve T' w' (TitrationLadder.ladderPoint kappa0 hstep j)| ≤ delta) :
    |∑ t ∈ T ∪ T', c t * DistanceEnsemble.ext T w t
        - ∑ t ∈ T ∪ T', c t * DistanceEnsemble.ext T' w' t|
      ≤ representerNorm P n * delta := by
  classical
  set U : Finset ℝ := T ∪ T' with hU
  set B : ℝ → ℝ := fun t =>
    ((DistanceEnsemble.ext T w t - DistanceEnsemble.ext T' w' t) / t)
      * Real.exp (-(kappa0 * t)) with hB
  set x : ℝ → ℝ := fun t => Real.exp (-(hstep * t)) with hx
  -- the functional difference is the `P`-combination of the reading differences
  have hkey : ∑ t ∈ U, c t * DistanceEnsemble.ext T w t
      - ∑ t ∈ U, c t * DistanceEnsemble.ext T' w' t
      = ∑ j ∈ Finset.range n, P.coeff j *
          (DistanceEnsemble.measCurve T w (TitrationLadder.ladderPoint kappa0 hstep j)
            - DistanceEnsemble.measCurve T' w' (TitrationLadder.ladderPoint kappa0 hstep j)) := by
    have e1 : ∀ j ∈ Finset.range n,
        P.coeff j * (DistanceEnsemble.measCurve T w (TitrationLadder.ladderPoint kappa0 hstep j)
          - DistanceEnsemble.measCurve T' w' (TitrationLadder.ladderPoint kappa0 hstep j))
          = ∑ t ∈ U, P.coeff j * (B t * (x t) ^ j) := by
      intro j _
      rw [reading_diff_pow T T' w w' kappa0 hstep j, ← hU, Finset.mul_sum]
    rw [Finset.sum_congr rfl e1, Finset.sum_comm]
    have e2 : ∀ t ∈ U, ∑ j ∈ Finset.range n, P.coeff j * (B t * (x t) ^ j)
        = B t * P.eval (x t) := by
      intro t _
      rw [eval_eq_sum_range' hdeg, Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [Finset.sum_congr rfl e2, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun t ht => ?_
    have htpos : 0 < t := hpos t ht
    have hEne : Real.exp (kappa0 * t) ≠ 0 := Real.exp_ne_zero _
    rw [hrep t ht]
    show c t * DistanceEnsemble.ext T w t - c t * DistanceEnsemble.ext T' w' t
        = ((DistanceEnsemble.ext T w t - DistanceEnsemble.ext T' w' t) / t
            * Real.exp (-(kappa0 * t))) * (c t * t * Real.exp (kappa0 * t))
    rw [Real.exp_neg]
    field_simp
  rw [hkey]
  calc |∑ j ∈ Finset.range n, P.coeff j *
          (DistanceEnsemble.measCurve T w (TitrationLadder.ladderPoint kappa0 hstep j)
            - DistanceEnsemble.measCurve T' w' (TitrationLadder.ladderPoint kappa0 hstep j))|
      ≤ ∑ j ∈ Finset.range n, |P.coeff j *
          (DistanceEnsemble.measCurve T w (TitrationLadder.ladderPoint kappa0 hstep j)
            - DistanceEnsemble.measCurve T' w' (TitrationLadder.ladderPoint kappa0 hstep j))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j ∈ Finset.range n, |P.coeff j| * delta := by
        refine Finset.sum_le_sum fun j hj => ?_
        rw [abs_mul]
        exact mul_le_mul_of_nonneg_left (h j (Finset.mem_range.1 hj)) (abs_nonneg _)
    _ = representerNorm P n * delta := by rw [representerNorm, Finset.sum_mul]

/-- **The screened populations are measured with no amplification.**  The functional
`w ↦ ∑_t w_t e^{−κ_j t}/t` — the screened population at the `j`-th condition — has representer
`X^j` and therefore amplification exactly `1`: two ensembles whose ladder readings agree to within
`δ` agree on it to within `δ`.  These are the properties of a disordered ensemble that a salt
titration measures; every other property is read off through its representer. -/
theorem screened_population_stable {j n : ℕ} (hj : j < n) :
    representerNorm ((X : ℝ[X]) ^ j) n = 1 ∧
    ∀ (kappa0 hstep : ℝ) (t : ℝ),
      ((X : ℝ[X]) ^ j).eval (Real.exp (-(hstep * t)))
        = (Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t)) / t) * t
            * Real.exp (kappa0 * t) ∨ t = 0 := by
  constructor
  · rw [representerNorm]
    have hcoeff : ∀ i, ((X : ℝ[X]) ^ j).coeff i = if i = j then 1 else 0 := by
      intro i
      simp [coeff_X_pow, eq_comm]
    rw [Finset.sum_congr rfl (fun i _ => by rw [hcoeff i])]
    simp [apply_ite (fun r : ℝ => |r|), Finset.sum_ite_eq', Finset.mem_range.2 hj]
  · intro kappa0 hstep t
    rcases eq_or_ne t 0 with rfl | ht
    · exact Or.inr rfl
    · refine Or.inl ?_
      have hpow : Real.exp (-(TitrationLadder.ladderPoint kappa0 hstep j * t))
          = Real.exp (-(kappa0 * t)) * (Real.exp (-(hstep * t))) ^ j := by
        rw [← Real.exp_nat_mul, ← Real.exp_add, TitrationLadder.ladderPoint]
        ring_nf
      have hEne : Real.exp (kappa0 * t) ≠ 0 := Real.exp_ne_zero _
      rw [eval_pow, eval_X, hpow, Real.exp_neg]
      field_simp
      rw [← Real.exp_add]
      simp

/-- **The functional law.**  What a finite salt titration of a disordered ensemble reports is a
list of screened populations, measured with unit amplification; every other linear property of the
distance distribution is measured through a representer polynomial, with the ℓ¹ norm of that
representer as its error amplification.  A calibrated model should therefore quote, for each
ensemble property it claims to have measured, the norm of that property's representer — the
individual weights of nearby distances having, by Part CXLII, unbounded norm. -/
theorem ensemble_functional_law {kappa0 hstep : ℝ} {n : ℕ} :
    (∀ (T T' : Finset ℝ) (w w' c : ℝ → ℝ) (P : ℝ[X]) (delta : ℝ),
        (∀ t ∈ T ∪ T', 0 < t) → P.natDegree < n →
        (∀ t ∈ T ∪ T', P.eval (Real.exp (-(hstep * t))) = c t * t * Real.exp (kappa0 * t)) →
        (∀ j < n, |DistanceEnsemble.measCurve T w (TitrationLadder.ladderPoint kappa0 hstep j)
            - DistanceEnsemble.measCurve T' w'
                (TitrationLadder.ladderPoint kappa0 hstep j)| ≤ delta) →
        |∑ t ∈ T ∪ T', c t * DistanceEnsemble.ext T w t
            - ∑ t ∈ T ∪ T', c t * DistanceEnsemble.ext T' w' t|
          ≤ representerNorm P n * delta) ∧
    (∀ j < n, representerNorm ((X : ℝ[X]) ^ j) n = 1) := by
  refine ⟨?_, ?_⟩
  · intro T T' w w' c P delta hpos hdeg hrep h
    exact functional_stability hpos hdeg hrep h
  · intro j hj
    exact (screened_population_stable hj).1

end EnsembleFunctionals
end IDR
