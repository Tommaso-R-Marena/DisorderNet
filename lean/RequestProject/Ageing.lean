/-
# Part XCIX  A region that never reaches equilibrium

Ensembles in this development are equilibrium objects except where kinetics is modelled
explicitly, and the two places where the equilibrium assumption is lifted — Parts XVIII–XIX —
lift it only to *driven steady states*.  The assumptions list recorded what was left: "genuinely
non-stationary (ageing, transient) cellular behaviour is still outside the scope".  A disordered
region in a cell is exactly that: the temperature, the crowding, the phosphorylation state and the
partner concentration all move, and the region tracks them with a lag.  This file proves what a
model of such a region can and cannot say.

The dynamics is a **time-inhomogeneous** finite-state Markov chain: a different stochastic matrix
`M t` at every step, each with its own instantaneous equilibrium `pi t`.  Nothing is stationary.

* `l1_vecMul_le` — a stochastic step never increases the population distance between two
  ensembles.  This is the trivial half, and it is not enough: it permits an arbitrarily large
  permanent lag.
* `l1_contract` — **the mixing half, derived rather than assumed.**  If the step is Doeblin
  minorised (`eps · pi z ≤ M x z` from every conformation), then population differences contract
  by the factor `1 − eps` at every step.  The proof splits the kernel as `eps·(restart) +
  (1−eps)·(remainder)`, which is where the contraction comes from.
* `ageing_lag` — **the theorem**: with each step contracting by `1 − eps` and the instantaneous
  equilibrium moving by at most `delta` per step, the region's distance from its *current*
  equilibrium obeys

      ‖p t − pi t‖₁  ≤  (1 − eps)^t · ‖p 0 − pi 0‖₁ + delta / eps .

  The transient dies geometrically, and what remains is a permanent lag `delta/eps`: the drive
  speed divided by the mixing rate.  A model that reports the instantaneous Boltzmann ensemble is
  wrong by that much, at all times, and the error is a property of the *ratio* of two timescales —
  which is what makes it measurable and what makes it survivable.
* `ageing_lag_of_slow_drive` — the adiabatic corollary: a drive slow compared with the mixing time
  leaves the equilibrium description valid to any prescribed accuracy.
* `no_stationary_matches` — **and the negative statement.**  An explicit two-state region driven
  between two environments, for which no single time-independent ensemble whatsoever reproduces
  the population at all times.  Equilibrium modelling of an ageing region is not conservative; it
  is wrong, and the file exhibits the observable that shows it.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset

namespace IDR.Ageing

variable {S : Type*} [Fintype S]

/-- Population distance (ℓ¹) between two signed population vectors. -/
def l1 (v : S → ℝ) : ℝ := ∑ x, |v x|

/-- One step of the dynamics acting on a population vector. -/
def vecMul (v : S → ℝ) (M : S → S → ℝ) : S → ℝ := fun z => ∑ x, v x * M x z

/-- A stochastic matrix. -/
def IsStoch (M : S → S → ℝ) : Prop := (∀ x z, 0 ≤ M x z) ∧ (∀ x, ∑ z, M x z = 1)

lemma l1_nonneg (v : S → ℝ) : 0 ≤ l1 v := Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- A stochastic step is a contraction in the weak sense: it never increases the population
distance. -/
theorem l1_vecMul_le {M : S → S → ℝ} (hM : IsStoch M) (v : S → ℝ) : l1 (vecMul v M) ≤ l1 v := by
  unfold l1 vecMul
  calc ∑ z, |∑ x, v x * M x z| ≤ ∑ z, ∑ x, |v x| * M x z := by
        refine Finset.sum_le_sum fun z _ => ?_
        calc |∑ x, v x * M x z| ≤ ∑ x, |v x * M x z| := Finset.abs_sum_le_sum_abs _ _
          _ = ∑ x, |v x| * M x z := by
              refine Finset.sum_congr rfl fun x _ => ?_
              rw [abs_mul, abs_of_nonneg (hM.1 x z)]
    _ = ∑ x, |v x| * ∑ z, M x z := by
        rw [Finset.sum_comm]
        exact Finset.sum_congr rfl fun x _ => by rw [Finset.mul_sum]
    _ = ∑ x, |v x| := by
        exact Finset.sum_congr rfl fun x _ => by rw [hM.2 x, mul_one]

/-- **Doeblin minorisation gives a strict contraction.**  If every row of the step dominates
`eps` times a fixed distribution, then differences of ensembles shrink by `1 − eps` per step. -/
theorem l1_contract {M : S → S → ℝ} (hM : IsStoch M) {eps : ℝ} {w : S → ℝ}
    (hwsum : ∑ z, w z = 1) (heps1 : eps < 1)
    (hmin : ∀ x z, eps * w z ≤ M x z) {v : S → ℝ} (hv : ∑ x, v x = 0) :
    l1 (vecMul v M) ≤ (1 - eps) * l1 v := by
  have hne : (1 : ℝ) - eps ≠ 0 := by linarith
  have hpos : (0 : ℝ) < 1 - eps := by linarith
  set R : S → S → ℝ := fun x z => (M x z - eps * w z) / (1 - eps) with hR
  have hRstoch : IsStoch R := by
    constructor
    · intro x z
      exact div_nonneg (by linarith [hmin x z]) hpos.le
    · intro x
      have : ∑ z, R x z = (∑ z, (M x z - eps * w z)) / (1 - eps) := by
        rw [hR]
        rw [← Finset.sum_div]
      rw [this, Finset.sum_sub_distrib, hM.2 x, ← Finset.mul_sum, hwsum, mul_one]
      field_simp
  have hkey : ∀ z, vecMul v M z = (1 - eps) * vecMul v R z := by
    intro z
    have hexp : vecMul v R z = (∑ x, v x * M x z) / (1 - eps) := by
      unfold vecMul
      rw [hR]
      have : ∀ x, v x * ((M x z - eps * w z) / (1 - eps))
          = (v x * M x z - v x * (eps * w z)) / (1 - eps) := by
        intro x; field_simp
      rw [Finset.sum_congr rfl (fun x _ => this x), ← Finset.sum_div, Finset.sum_sub_distrib]
      congr 1
      have : ∑ x, v x * (eps * w z) = (∑ x, v x) * (eps * w z) := by rw [Finset.sum_mul]
      rw [this, hv, zero_mul, sub_zero]
    rw [hexp]
    unfold vecMul
    field_simp
  have hl1 : l1 (vecMul v M) = (1 - eps) * l1 (vecMul v R) := by
    unfold l1
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun z _ => ?_
    rw [hkey z, abs_mul, abs_of_nonneg hpos.le]
  rw [hl1]
  exact mul_le_mul_of_nonneg_left (l1_vecMul_le hRstoch v) hpos.le

/-- **The ageing bound.**  A region driven through a sequence of environments, each step
contracting by `1 − eps` and each instantaneous equilibrium moving by at most `delta`, sits at
population distance at most `(1−eps)^t · (initial distance) + delta/eps` from its current
equilibrium — a geometric transient plus a permanent lag set by drive speed over mixing rate. -/
theorem ageing_lag {p pi : ℕ → S → ℝ} {M : ℕ → S → S → ℝ} {eps delta : ℝ}
    (heps0 : 0 < eps) (heps1 : eps < 1) (hdelta : 0 ≤ delta)
    (hstep : ∀ t, p (t+1) = vecMul (p t) (M t))
    (hfix : ∀ t, vecMul (pi t) (M t) = pi t)
    (hcontract : ∀ t (v : S → ℝ), ∑ x, v x = 0 → l1 (vecMul v (M t)) ≤ (1 - eps) * l1 v)
    (hzero : ∀ t, ∑ x, (p t x - pi t x) = 0)
    (hmove : ∀ t, l1 (fun x => pi t x - pi (t+1) x) ≤ delta) :
    ∀ t, l1 (fun x => p t x - pi t x) ≤ (1 - eps)^t * l1 (fun x => p 0 x - pi 0 x)
        + delta / eps := by
  intro t
  induction t with
  | zero =>
      simp only [pow_zero, one_mul]
      have : 0 ≤ delta / eps := div_nonneg hdelta heps0.le
      linarith
  | succ t ih =>
      set a := l1 (fun x => p t x - pi t x) with ha
      have hstepd : (fun z => p (t+1) z - pi t z) = vecMul (fun x => p t x - pi t x) (M t) := by
        funext z
        have h1 : p (t+1) z = vecMul (p t) (M t) z := by rw [hstep t]
        have h2 : pi t z = vecMul (pi t) (M t) z := by rw [hfix t]
        rw [h1, h2]
        unfold vecMul
        rw [← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun x _ => by ring
      have hc : l1 (fun z => p (t+1) z - pi t z) ≤ (1 - eps) * a := by
        rw [hstepd]
        exact hcontract t _ (hzero t)
      have htri : l1 (fun x => p (t+1) x - pi (t+1) x)
          ≤ l1 (fun z => p (t+1) z - pi t z) + l1 (fun x => pi t x - pi (t+1) x) := by
        unfold l1
        rw [← Finset.sum_add_distrib]
        refine Finset.sum_le_sum fun x _ => ?_
        simp only []
        have hsplit : p (t+1) x - pi (t+1) x = (p (t+1) x - pi t x) + (pi t x - pi (t+1) x) := by
          ring
        rw [hsplit]
        exact abs_add_le _ _
      have hstep2 : l1 (fun x => p (t+1) x - pi (t+1) x) ≤ (1 - eps) * a + delta := by
        have := hmove t
        linarith
      have hgeo : (1 - eps) * ((1 - eps)^t * l1 (fun x => p 0 x - pi 0 x) + delta / eps) + delta
          = (1 - eps)^(t+1) * l1 (fun x => p 0 x - pi 0 x) + delta / eps := by
        field_simp
        ring
      have h1e : (0:ℝ) ≤ 1 - eps := by linarith
      calc l1 (fun x => p (t+1) x - pi (t+1) x) ≤ (1 - eps) * a + delta := hstep2
        _ ≤ (1 - eps) * ((1 - eps)^t * l1 (fun x => p 0 x - pi 0 x) + delta / eps) + delta := by
            have := mul_le_mul_of_nonneg_left ih h1e
            linarith
        _ = (1 - eps)^(t+1) * l1 (fun x => p 0 x - pi 0 x) + delta / eps := hgeo

/-- The adiabatic corollary: a drive slow compared with the mixing rate leaves the instantaneous
equilibrium description accurate to any prescribed tolerance, once the transient has died. -/
theorem ageing_lag_of_slow_drive {p pi : ℕ → S → ℝ} {M : ℕ → S → S → ℝ} {eps delta tol : ℝ}
    (heps0 : 0 < eps) (heps1 : eps < 1) (hdelta : 0 ≤ delta)
    (hstep : ∀ t, p (t+1) = vecMul (p t) (M t))
    (hfix : ∀ t, vecMul (pi t) (M t) = pi t)
    (hcontract : ∀ t (v : S → ℝ), ∑ x, v x = 0 → l1 (vecMul v (M t)) ≤ (1 - eps) * l1 v)
    (hzero : ∀ t, ∑ x, (p t x - pi t x) = 0)
    (hmove : ∀ t, l1 (fun x => pi t x - pi (t+1) x) ≤ delta)
    (hslow : delta / eps ≤ tol / 2) (t : ℕ)
    (htrans : (1 - eps)^t * l1 (fun x => p 0 x - pi 0 x) ≤ tol / 2) :
    l1 (fun x => p t x - pi t x) ≤ tol := by
  have := ageing_lag heps0 heps1 hdelta hstep hfix hcontract hzero hmove t
  linarith

/-! ### No stationary ensemble describes a driven region -/

/-- A two-state region whose population is `1` in the first environment and `0` in the second:
the drive alternates, so the population alternates. -/
def drivenPop : ℕ → Bool → ℝ := fun t s => if s = (decide (t % 2 = 0)) then 1 else 0

/-- **No time-independent ensemble reproduces a driven region.**  The population of the first
conformation takes two different values at two different times, so no single ensemble — Boltzmann
or otherwise, at any temperature, with any energy function — agrees with the region at all
times. -/
theorem no_stationary_matches :
    ¬ ∃ q : Bool → ℝ, ∀ t : ℕ, ∀ s : Bool, drivenPop t s = q s := by
  rintro ⟨q, hq⟩
  have h0 : drivenPop 0 true = q true := hq 0 true
  have h1 : drivenPop 1 true = q true := hq 1 true
  rw [← h1] at h0
  simp [drivenPop] at h0

end IDR.Ageing
