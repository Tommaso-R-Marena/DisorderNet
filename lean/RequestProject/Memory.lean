/-
# Part XXI  One relaxation time is not enough

Fluorescence correlation, single-molecule FRET and NMR relaxation of disordered regions all
report *hierarchies* of timescales -- correlation functions that are visibly not single
exponentials.  A two-state kinetic model is the default interpretation of almost every such
experiment.  This file proves that the two are incompatible, exactly.

* `twoP_stochastic`, `piTwo_stationary`, `twoP_detailedBalance` -- the two-state chain is a
  perfectly good reversible kinetics: unlike the driven cycle of Part XVIII, nothing here is
  wrong with its equilibrium.
* `prop_eq` -- the conditional expectation of any observable relaxes as
  `mean + (f i − mean)·λ^t` with the single eigenvalue `λ = 1 − a − b`;
* `autocorr_eq` -- so the equilibrium autocorrelation of *every* observable of a two-state
  model is one geometric, `Var(f)·λ^t`, with the same `λ` for all observables: a two-state
  model has exactly one relaxation time, and it is a property of the model, not of the probe.
* `no_single_geometric` -- but a genuine two-timescale decay is not a geometric: matching
  `c₁λ₁^t + c₂λ₂^t` (positive weights, distinct rates) at `t = 0, 1, 2` already forces
  `c₁c₂(λ₁ − λ₂)² = 0`, by the strict Cauchy--Schwarz inequality.
* `two_state_model_cannot_fit_two_timescales` -- hence no two-state model, with any rates and
  any observable, reproduces a two-exponential correlation function.  Distinct measured
  timescales are a *capacity* statement about the kinetic model, exactly as distinct populated
  conformations are a capacity statement about the ensemble (Part I).
-/
import Mathlib
import RequestProject.Kinetics

namespace IDR

open Finset

namespace Memory

/-- The general two-state kinetics: leave state 0 with probability `a`, state 1 with
probability `b`. -/
noncomputable def twoP (a b : ℝ) : Fin 2 → Fin 2 → ℝ := ![![1 - a, a], ![b, 1 - b]]

/-- Its equilibrium populations. -/
noncomputable def piTwo (a b : ℝ) : Fin 2 → ℝ := ![b / (a + b), a / (a + b)]

theorem twoP_stochastic {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (ha1 : a ≤ 1) (hb1 : b ≤ 1) :
    Kinetics.IsStochastic (twoP a b) := by
  constructor
  · intro i j
    fin_cases i <;> fin_cases j <;> simp [twoP] <;> linarith
  · intro i
    fin_cases i <;> simp [twoP, Fin.sum_univ_two]

theorem piTwo_stationary {a b : ℝ} (hab : 0 < a + b) :
    Kinetics.Stationary (twoP a b) (piTwo a b) := by
  have h : a + b ≠ 0 := hab.ne'
  funext j
  simp only [Kinetics.evolve, Fin.sum_univ_two]
  fin_cases j <;> simp [twoP, piTwo] <;> field_simp <;> ring

/-- A two-state chain is always reversible: there is no Kolmogorov obstruction with only two
states, which is why Part XVIII needed three. -/
theorem twoP_detailedBalance {a b : ℝ} (hab : 0 < a + b) :
    Kinetics.DetailedBalance (twoP a b) (piTwo a b) := by
  have h : a + b ≠ 0 := hab.ne'
  intro i j
  fin_cases i <;> fin_cases j <;> simp [twoP, piTwo] <;> field_simp

/-- The conditional expectation of `f` after `t` steps, as a function of the initial state. -/
noncomputable def prop (P : Fin 2 → Fin 2 → ℝ) (f : Fin 2 → ℝ) : ℕ → Fin 2 → ℝ
  | 0 => f
  | t + 1 => fun i => ∑ j, P i j * prop P f t j

/-- The equilibrium mean of an observable. -/
noncomputable def mean (a b : ℝ) (f : Fin 2 → ℝ) : ℝ := ∑ i, piTwo a b i * f i

/-- The equilibrium variance of an observable. -/
noncomputable def varTwo (a b : ℝ) (f : Fin 2 → ℝ) : ℝ :=
  ∑ i, piTwo a b i * (f i - mean a b f) ^ 2

/-- The equilibrium autocorrelation function of an observable. -/
noncomputable def autocorr (a b : ℝ) (f : Fin 2 → ℝ) (t : ℕ) : ℝ :=
  ∑ i, piTwo a b i * (f i - mean a b f) * (prop (twoP a b) f t i - mean a b f)

/-- **A two-state model has exactly one eigenvalue.**  Every conditional expectation relaxes
to the mean geometrically with ratio `λ = 1 − a − b`. -/
theorem prop_eq {a b : ℝ} (hab : 0 < a + b) (f : Fin 2 → ℝ) (t : ℕ) (i : Fin 2) :
    prop (twoP a b) f t i = mean a b f + (f i - mean a b f) * (1 - a - b) ^ t := by
  have h : a + b ≠ 0 := hab.ne'
  induction t generalizing i with
  | zero => simp [prop]
  | succ t ih =>
      have hstep : ∀ i : Fin 2,
          ∑ j, twoP a b i j * (f j - mean a b f) = (1 - a - b) * (f i - mean a b f) := by
        intro i
        simp only [Fin.sum_univ_two, mean, piTwo]
        fin_cases i <;> simp [twoP] <;> field_simp <;> ring
      have : prop (twoP a b) f (t + 1) i = ∑ j, twoP a b i j * prop (twoP a b) f t j := rfl
      rw [this]
      have hrw : ∀ j, twoP a b i j * prop (twoP a b) f t j
          = twoP a b i j * mean a b f
            + (1 - a - b) ^ t * (twoP a b i j * (f j - mean a b f)) := by
        intro j
        rw [ih j]; ring
      have hrow : ∑ j, twoP a b i j = 1 := by
        fin_cases i <;> simp [twoP, Fin.sum_univ_two]
      rw [Finset.sum_congr rfl (fun j _ => hrw j), Finset.sum_add_distrib, ← Finset.sum_mul,
        ← Finset.mul_sum, hstep i, hrow, pow_succ]
      ring

/-- **The autocorrelation of every observable is a single geometric**, with the same rate. -/
theorem autocorr_eq {a b : ℝ} (hab : 0 < a + b) (f : Fin 2 → ℝ) (t : ℕ) :
    autocorr a b f t = varTwo a b f * (1 - a - b) ^ t := by
  simp only [autocorr, varTwo, Finset.sum_mul]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [prop_eq hab f t i]
  ring

/-- **A genuine two-timescale decay is not a geometric.**  Two positive weights and two
distinct rates cannot be matched by a single geometric even at three consecutive times: the
gap is exactly the strict Cauchy--Schwarz defect `c₁c₂(λ₁ − λ₂)²`. -/
theorem no_single_geometric {c₁ c₂ l₁ l₂ c l : ℝ} (hc₁ : 0 < c₁) (hc₂ : 0 < c₂) (hne : l₁ ≠ l₂)
    (h0 : c = c₁ + c₂) (h1 : c * l = c₁ * l₁ + c₂ * l₂)
    (h2 : c * l ^ 2 = c₁ * l₁ ^ 2 + c₂ * l₂ ^ 2) : False := by
  have hd : (l₁ - l₂) ^ 2 > 0 := by
    have : l₁ - l₂ ≠ 0 := sub_ne_zero.mpr hne
    positivity
  have hpos : 0 < c₁ * c₂ * (l₁ - l₂) ^ 2 := by positivity
  have key : (c₁ * l₁ + c₂ * l₂) ^ 2 = (c₁ + c₂) * (c₁ * l₁ ^ 2 + c₂ * l₂ ^ 2) := by
    rw [← h1, ← h0, ← h2]; ring
  have hzero : c₁ * c₂ * (l₁ - l₂) ^ 2 = 0 := by linear_combination -key
  linarith

/-- **The design statement.**  No two-state kinetic model -- whatever its rates, whatever
observable is probed -- reproduces a correlation function with two distinct relaxation rates
and positive amplitudes.  A measured hierarchy of timescales is a lower bound on the number of
kinetic states, exactly as a measured spread of conformations is a lower bound on the number
of components of the ensemble. -/
theorem two_state_model_cannot_fit_two_timescales {a b c₁ c₂ l₁ l₂ : ℝ} (hab : 0 < a + b)
    (f : Fin 2 → ℝ) (hc₁ : 0 < c₁) (hc₂ : 0 < c₂) (hne : l₁ ≠ l₂) :
    ¬ ∀ t : ℕ, autocorr a b f t = c₁ * l₁ ^ t + c₂ * l₂ ^ t := by
  intro h
  have e0 := h 0
  have e1 := h 1
  have e2 := h 2
  rw [autocorr_eq hab] at e0 e1 e2
  refine no_single_geometric (c := varTwo a b f) (l := 1 - a - b) hc₁ hc₂ hne ?_ ?_ ?_
  · simpa using e0
  · simpa using e1
  · simpa using e2

end Memory

end IDR
