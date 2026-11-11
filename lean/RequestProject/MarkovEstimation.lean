/-
# Part C  What an estimated Markov state model is worth

Part LXII proves what a Markov state model of a disordered region reports when the transition
matrix is known *exactly*: the coarse thermodynamics is right, the timescales are one-sided, the
answer depends on the lag.  Real models are estimated from a finite number of observed
transitions, and the assumptions list recorded the gap: "Statistical error in the counts, the
choice of clustering, and the continuous-time case are outside what is proved."  This file closes
the first and the third.

Everything rests on one inequality, and it is the same contraction that governs ageing
(`RequestProject.Ageing`): a Doeblin-minorised step shrinks population differences by `1 − eps`.

* `stationary_perturb` — **the error propagation theorem.**  If every estimated row is within
  `eta` (in ℓ¹) of the true row, and the true chain is `eps`-minorised, then the estimated
  stationary populations are within `eta / eps` of the true ones.  The statistical error of the
  counts enters the reported populations *divided by the mixing rate*: a slowly mixing model — the
  generic case for a disordered region — amplifies count noise, and by exactly this factor.
* `observable_perturb` — hence any reported average of an observable bounded by `G` is accurate to
  `G·eta/eps`, which is the number a Markov state model should quote.
* `rows_within_of_counts` — the elementary counting half: a row is within `eta` in ℓ¹ as soon as
  every destination frequency is within `eta/|S|`, so the accuracy requirement is on the counts
  out of each *state*, not on the length of the trajectory.  (The concentration statement that
  converts a number of observed transitions into such an entrywise accuracy is the Chebyshev bound
  of Part XC, applied per destination.)
* `generator_stationary_iff` — **the continuous-time bridge.**  For a generator `Q` (rows summing
  to zero, nonnegative off-diagonal rates) and a step `h` small enough that `I + hQ` is a genuine
  stochastic matrix, a population vector is invariant for the continuous-time dynamics (`πQ = 0`)
  **iff** it is invariant for that discrete step.  Every statement of Part LXII and of this file
  therefore applies verbatim to a continuous-time Markov state model, read at any admissible step.
* `uniformised_isStoch` — and the admissible steps are exactly the uniformisation ones,
  `h ≤ 1/max_x (−Q x x)`.
-/
import Mathlib
import RequestProject.Ageing

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset

namespace IDR.MSMEstimation

open IDR.Ageing

variable {S : Type*} [Fintype S] [DecidableEq S]

omit [DecidableEq S] in
lemma l1_add_le (u v : S → ℝ) : l1 (fun x => u x + v x) ≤ l1 u + l1 v := by
  unfold l1
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_le_sum fun x _ => abs_add_le _ _

omit [DecidableEq S] in
/-- The ℓ¹ size of the population change caused by a perturbation of the transition matrix, for a
probability vector `q`, is at most the worst row error. -/
lemma l1_vecMul_perturb {q : S → ℝ} (hq : ∀ x, 0 ≤ q x) (hq1 : ∑ x, q x = 1)
    (D : S → S → ℝ) {eta : ℝ} (hD : ∀ x, ∑ z, |D x z| ≤ eta) :
    l1 (vecMul q D) ≤ eta := by
  unfold l1 vecMul
  calc ∑ z, |∑ x, q x * D x z| ≤ ∑ z, ∑ x, q x * |D x z| := by
        refine Finset.sum_le_sum fun z _ => ?_
        calc |∑ x, q x * D x z| ≤ ∑ x, |q x * D x z| := Finset.abs_sum_le_sum_abs _ _
          _ = ∑ x, q x * |D x z| := by
              refine Finset.sum_congr rfl fun x _ => ?_
              rw [abs_mul, abs_of_nonneg (hq x)]
    _ = ∑ x, q x * ∑ z, |D x z| := by
        rw [Finset.sum_comm]
        exact Finset.sum_congr rfl fun x _ => by rw [Finset.mul_sum]
    _ ≤ ∑ x, q x * eta := Finset.sum_le_sum fun x _ => mul_le_mul_of_nonneg_left (hD x) (hq x)
    _ = eta := by rw [← Finset.sum_mul, hq1, one_mul]

omit [DecidableEq S] in
/-- **The error propagation theorem for an estimated Markov state model.**  Row errors of size
`eta` become stationary-population errors of size `eta / eps`, with `eps` the minorisation
constant of the true chain: count noise is amplified by the mixing time. -/
theorem stationary_perturb {M Mhat : S → S → ℝ} {pi pihat : S → ℝ} {eps eta : ℝ}
    (heps0 : 0 < eps)
    (hcontract : ∀ v : S → ℝ, ∑ x, v x = 0 → l1 (vecMul v M) ≤ (1 - eps) * l1 v)
    (hpi : vecMul pi M = pi) (hpihat : vecMul pihat Mhat = pihat)
    (hpihat0 : ∀ x, 0 ≤ pihat x) (hpihat1 : ∑ x, pihat x = 1) (hpi1 : ∑ x, pi x = 1)
    (hrows : ∀ x, ∑ z, |Mhat x z - M x z| ≤ eta) :
    l1 (fun x => pihat x - pi x) ≤ eta / eps := by
  set D : S → S → ℝ := fun x z => Mhat x z - M x z with hD
  have hzero : ∑ x, (pihat x - pi x) = 0 := by
    rw [Finset.sum_sub_distrib, hpihat1, hpi1, sub_self]
  have hsplit : ∀ z, pihat z - pi z = vecMul pihat D z + vecMul (fun x => pihat x - pi x) M z := by
    intro z
    have h1 : pihat z = vecMul pihat Mhat z := by rw [hpihat]
    have h2 : pi z = vecMul pi M z := by rw [hpi]
    rw [h1, h2]
    unfold vecMul
    rw [hD]
    rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun x _ => by ring
  have hfun : (fun z => pihat z - pi z)
      = fun z => vecMul pihat D z + vecMul (fun x => pihat x - pi x) M z := funext hsplit
  have hbound : l1 (fun x => pihat x - pi x)
      ≤ eta + (1 - eps) * l1 (fun x => pihat x - pi x) := by
    calc l1 (fun x => pihat x - pi x)
        = l1 (fun z => vecMul pihat D z + vecMul (fun x => pihat x - pi x) M z) := by rw [← hfun]
      _ ≤ l1 (vecMul pihat D) + l1 (vecMul (fun x => pihat x - pi x) M) :=
          l1_add_le _ _
      _ ≤ eta + (1 - eps) * l1 (fun x => pihat x - pi x) := by
          have h1 := l1_vecMul_perturb hpihat0 hpihat1 D hrows
          have h2 := hcontract (fun x => pihat x - pi x) hzero
          linarith
  rw [le_div_iff₀ heps0]
  linarith

omit [DecidableEq S] in
/-- Any reported average of an observable bounded by `G` inherits the accuracy `G·eta/eps`. -/
theorem observable_perturb {pi pihat : S → ℝ} {g : S → ℝ} {G err : ℝ} (hG : ∀ x, |g x| ≤ G)
    (hG0 : 0 ≤ G) (hclose : l1 (fun x => pihat x - pi x) ≤ err) :
    |(∑ x, pihat x * g x) - ∑ x, pi x * g x| ≤ G * err := by
  have hstep : |(∑ x, pihat x * g x) - ∑ x, pi x * g x| ≤ ∑ x, |pihat x - pi x| * G := by
    rw [← Finset.sum_sub_distrib]
    calc |∑ x, (pihat x * g x - pi x * g x)| ≤ ∑ x, |pihat x * g x - pi x * g x| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ x, |pihat x - pi x| * G := by
          refine Finset.sum_le_sum fun x _ => ?_
          have : pihat x * g x - pi x * g x = (pihat x - pi x) * g x := by ring
          rw [this, abs_mul]
          exact mul_le_mul_of_nonneg_left (hG x) (abs_nonneg _)
  have : ∑ x, |pihat x - pi x| * G = l1 (fun x => pihat x - pi x) * G := by
    unfold l1
    rw [Finset.sum_mul]
  rw [this] at hstep
  calc |(∑ x, pihat x * g x) - ∑ x, pi x * g x| ≤ l1 (fun x => pihat x - pi x) * G := hstep
    _ ≤ err * G := mul_le_mul_of_nonneg_right hclose hG0
    _ = G * err := by ring

omit [DecidableEq S] in
/-- The counting half: a row is within `eta` in ℓ¹ once every destination frequency is within
`eta / |S|`. -/
theorem rows_within_of_counts {Mhat M : S → S → ℝ} {eta : ℝ} (hcard : 0 < Fintype.card S)
    (hentry : ∀ x z, |Mhat x z - M x z| ≤ eta / Fintype.card S) :
    ∀ x, ∑ z, |Mhat x z - M x z| ≤ eta := by
  intro x
  have hc : (0:ℝ) < (Fintype.card S : ℝ) := by exact_mod_cast hcard
  calc ∑ z, |Mhat x z - M x z| ≤ ∑ _z : S, eta / Fintype.card S :=
        Finset.sum_le_sum fun z _ => hentry x z
    _ = (Fintype.card S : ℝ) * (eta / Fintype.card S) := by
        rw [Finset.sum_const, nsmul_eq_mul, Finset.card_univ]
    _ = eta := by field_simp

/-! ### Continuous time -/

/-- A generator: nonnegative off-diagonal rates, rows summing to zero. -/
def IsGenerator (Q : S → S → ℝ) : Prop :=
  (∀ x z, x ≠ z → 0 ≤ Q x z) ∧ (∀ x, ∑ z, Q x z = 0)

/-- The uniformised step `I + hQ`. -/
def uniformised (Q : S → S → ℝ) (h : ℝ) : S → S → ℝ :=
  fun x z => (if z = x then (1:ℝ) else 0) + h * Q x z

/-- For a small enough step the uniformisation is a genuine stochastic matrix. -/
theorem uniformised_isStoch {Q : S → S → ℝ} (hQ : IsGenerator Q) {h : ℝ} (hh : 0 < h)
    (hsmall : ∀ x, h * (-(Q x x)) ≤ 1) : IsStoch (uniformised Q h) := by
  constructor
  · intro x z
    unfold uniformised
    by_cases hzx : z = x
    · subst hzx
      have hz := hsmall z
      have : (if (z:S) = z then (1:ℝ) else 0) = 1 := by simp
      rw [this]
      linarith
    · simp only [if_neg hzx, zero_add]
      exact mul_nonneg hh.le (hQ.1 x z (Ne.symm hzx))
  · intro x
    unfold uniformised
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, hQ.2 x, mul_zero, add_zero]
    simp

/-- **The continuous-time bridge.**  A population vector is invariant for the generator exactly
when it is invariant for the uniformised discrete step, so the discrete-time theory transfers
verbatim. -/
theorem generator_stationary_iff {Q : S → S → ℝ} {h : ℝ} (hh : h ≠ 0) (p : S → ℝ) :
    (∀ z, ∑ x, p x * Q x z = 0) ↔ vecMul p (uniformised Q h) = p := by
  constructor
  · intro hgen
    funext z
    unfold vecMul uniformised
    have hsum : ∑ x, p x * ((if z = x then (1:ℝ) else 0) + h * Q x z)
        = (∑ x, p x * (if z = x then (1:ℝ) else 0)) + h * ∑ x, p x * Q x z := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun x _ => by ring
    rw [hsum, hgen z, mul_zero, add_zero]
    simp
  · intro hfix z
    have hz : vecMul p (uniformised Q h) z = p z := by rw [hfix]
    unfold vecMul uniformised at hz
    have hsum : ∑ x, p x * ((if z = x then (1:ℝ) else 0) + h * Q x z)
        = (∑ x, p x * (if z = x then (1:ℝ) else 0)) + h * ∑ x, p x * Q x z := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun x _ => by ring
    rw [hsum] at hz
    have hdelta : ∑ x, p x * (if z = x then (1:ℝ) else 0) = p z := by simp
    rw [hdelta] at hz
    have : h * ∑ x, p x * Q x z = 0 := by linarith
    rcases mul_eq_zero.mp this with hcase | hcase
    · exact absurd hcase hh
    · exact hcase

end IDR.MSMEstimation
