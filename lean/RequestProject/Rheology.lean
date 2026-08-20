/-
# Part XLIII.1  The material state: why a viscous-droplet model of a condensate fails

The condensates that disordered regions form are described, in the first instance, as liquid
droplets: a single viscosity, a single relaxation time, Stokes--Einstein diffusion inside.
Every quantitative rheology experiment on them says otherwise -- the relaxation modulus decays
as a broad power law rather than an exponential, the probe mean-square displacement grows
sublinearly, and both change with the age of the droplet.  This file proves what those three
observations *exclude*, at the level of the models actually written down.

**A finite Maxwell spectrum cannot be a power law.**  With relaxation modulus
`maxwell g tau t = Σ_k g_k e^{-t/tau_k}` (`g_k ≥ 0`, `tau_k > 0` -- any finite set of modes,
which is what a coarse-grained model with finitely many slow variables produces):

* `maxwell_le_exp` -- `G(t) ≤ G(0)·e^{-t/T}` where `T` is the slowest mode: the decay is
  exponential with the slowest rate, whatever the spectrum;
* `maxwell_lt_power_law` -- hence for *every* power law `C·t^{-alpha}` the model eventually
  falls strictly below it: no finite spectrum reproduces power-law rheology at long times, and
  the failure is one-sided (the model is always too fast);
* `maxwell_integrableOn`, `maxwell_viscosity` -- a finite spectrum always has a finite terminal
  viscosity, `η = Σ_k g_k tau_k`;
* `power_law_not_integrableOn` -- while a power law with `alpha ≤ 1` has none.  So "the
  condensate has a viscosity" is not an approximation to power-law rheology; it is a
  qualitatively different statement.

**Sublinear diffusion excludes memoryless steps.**  For a probe whose displacement is the sum
of `n` steps with covariance `C`,

* `msd_of_uncorrelated` -- uncorrelated stationary steps give exactly `MSD(n) = n·σ²`, linear,
  with no freedom at all;
* `subdiffusion_forces_anticorrelation` -- so any sublinear MSD forces the off-diagonal
  covariance to sum to a negative number, and `subdiffusion_forces_memory` -- in particular
  some pair of steps is correlated;
* `power_law_msd_forces_memory` -- for a measured `MSD(n) = A·n^alpha` with `alpha < 1` this
  applies at every `n ≥ 2`.  A model of motion inside a condensate that samples independent
  steps is excluded by the data, not merely inaccurate.

**Aging excludes stationarity.**  `aging_forces_error` -- if the modulus measured at two
waiting times differs at some lag, any model that predicts a single waiting-time-independent
curve is wrong by at least half that difference on one of them.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

open Finset MeasureTheory

namespace Rheology

/-! ## The relaxation modulus of a finite Maxwell spectrum -/

variable {n : ℕ}

/-- The relaxation modulus of a finite Maxwell spectrum: moduli `g` and relaxation times
`tau`. -/
noncomputable def maxwell (g tau : Fin n → ℝ) (t : ℝ) : ℝ :=
  ∑ k, g k * Real.exp (-(t / tau k))

@[simp] lemma maxwell_zero (g tau : Fin n → ℝ) : maxwell g tau 0 = ∑ k, g k := by
  simp [maxwell]

/-- **A finite spectrum decays exponentially at the slowest rate.**  If every relaxation time
is at most `T`, the modulus is at most `G(0)·e^{-t/T}`. -/
theorem maxwell_le_exp {g tau : Fin n → ℝ} {T : ℝ} (hg : ∀ k, 0 ≤ g k) (hpos : ∀ k, 0 < tau k)
    (hT : ∀ k, tau k ≤ T) {t : ℝ} (ht : 0 ≤ t) :
    maxwell g tau t ≤ (∑ k, g k) * Real.exp (-(t / T)) := by
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun k _ => ?_
  have hle : Real.exp (-(t / tau k)) ≤ Real.exp (-(t / T)) := by
    refine Real.exp_le_exp.mpr ?_
    have : t / T ≤ t / tau k := div_le_div_of_nonneg_left ht (hpos k) (hT k)
    linarith
  exact mul_le_mul_of_nonneg_left hle (hg k)

/-- **No finite Maxwell spectrum is a power law.**  For every power law `C·t^{-alpha}` with
`C > 0` (any exponent), the modulus of a finite spectrum is eventually strictly below it. -/
theorem maxwell_lt_power_law {g tau : Fin n → ℝ} {T : ℝ} (hg : ∀ k, 0 ≤ g k)
    (hpos : ∀ k, 0 < tau k) (hT : ∀ k, tau k ≤ T) (hTpos : 0 < T) {C alpha : ℝ} (hC : 0 < C) :
    ∃ t₀ : ℝ, 0 < t₀ ∧ ∀ t ≥ t₀, maxwell g tau t < C * t ^ (-alpha) := by
  set S := ∑ k, g k with hS
  have hSnn : 0 ≤ S := Finset.sum_nonneg fun k _ => hg k
  have hlim : Filter.Tendsto (fun t : ℝ => (S + 1) * (t ^ alpha * Real.exp (-(1 / T) * t)))
      Filter.atTop (nhds 0) := by
    have := tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero alpha (1 / T) (by positivity)
    simpa using this.const_mul (S + 1)
  obtain ⟨t₁, ht₁⟩ := Filter.eventually_atTop.mp
    (hlim.eventually (gt_mem_nhds (show (0 : ℝ) < C by exact hC)))
  refine ⟨max t₁ 1, lt_of_lt_of_le one_pos (le_max_right _ _), fun t ht => ?_⟩
  have ht1 : t₁ ≤ t := le_trans (le_max_left _ _) ht
  have htpos : 0 < t := lt_of_lt_of_le one_pos (le_trans (le_max_right _ _) ht)
  have hkey := ht₁ t ht1
  have hexp : maxwell g tau t ≤ S * Real.exp (-(t / T)) :=
    maxwell_le_exp hg hpos hT htpos.le
  have hrw : Real.exp (-(t / T)) = Real.exp (-(1 / T) * t) := by
    congr 1
    field_simp
  have hpowpos : 0 < t ^ alpha := Real.rpow_pos_of_pos htpos alpha
  have hstep : S * Real.exp (-(1 / T) * t) * t ^ alpha < C := by
    have hless : S * Real.exp (-(1 / T) * t) * t ^ alpha
        ≤ (S + 1) * (t ^ alpha * Real.exp (-(1 / T) * t)) := by
      have : (0 : ℝ) ≤ Real.exp (-(1 / T) * t) * t ^ alpha :=
        mul_nonneg (Real.exp_pos _).le hpowpos.le
      nlinarith [Real.exp_pos (-(1 / T) * t)]
    linarith
  have hfinal : maxwell g tau t * t ^ alpha < C := by
    calc maxwell g tau t * t ^ alpha ≤ S * Real.exp (-(t / T)) * t ^ alpha :=
          mul_le_mul_of_nonneg_right hexp hpowpos.le
      _ = S * Real.exp (-(1 / T) * t) * t ^ alpha := by rw [hrw]
      _ < C := hstep
  have hneg : t ^ (-alpha) = (t ^ alpha)⁻¹ := by
    rw [Real.rpow_neg htpos.le]
  rw [hneg, ← div_eq_mul_inv, lt_div_iff₀ hpowpos]
  exact hfinal

/-! ### Terminal viscosity -/

lemma exp_mode_integrableOn {tau : ℝ} (hpos : 0 < tau) :
    IntegrableOn (fun t : ℝ => Real.exp (-(t / tau))) (Set.Ioi 0) := by
  have hrw : (fun t : ℝ => Real.exp (-(t / tau))) = fun t : ℝ => Real.exp (-(1 / tau) * t) := by
    funext t
    congr 1
    field_simp
  rw [hrw]
  exact exp_neg_integrableOn_Ioi 0 (by positivity)

lemma integral_exp_mode {tau : ℝ} (hpos : 0 < tau) :
    ∫ t in Set.Ioi (0 : ℝ), Real.exp (-(t / tau)) = tau := by
  have hrw : (fun t : ℝ => Real.exp (-(t / tau)))
      = fun t : ℝ => (fun x : ℝ => Real.exp (-x)) ((1 / tau) * t) := by
    funext t
    congr 1
    field_simp
  have hcomp := integral_comp_mul_left_Ioi (fun x : ℝ => Real.exp (-x)) 0
    (show (0 : ℝ) < 1 / tau by positivity)
  simp only [mul_zero] at hcomp
  rw [hrw, hcomp, integral_exp_neg_Ioi]
  simp

/-- A finite Maxwell spectrum is integrable in time: it has a terminal viscosity. -/
theorem maxwell_integrableOn {g tau : Fin n → ℝ} (hpos : ∀ k, 0 < tau k) :
    IntegrableOn (maxwell g tau) (Set.Ioi 0) := by
  unfold maxwell
  refine integrable_finset_sum _ fun k _ => ?_
  exact ((exp_mode_integrableOn (hpos k)).const_mul (g k))

/-- **The terminal viscosity of a finite spectrum is `Σ g_k tau_k`** -- always finite. -/
theorem maxwell_viscosity {g tau : Fin n → ℝ} (hpos : ∀ k, 0 < tau k) :
    ∫ t in Set.Ioi (0 : ℝ), maxwell g tau t = ∑ k, g k * tau k := by
  unfold maxwell
  rw [integral_finset_sum _ (fun k _ => (exp_mode_integrableOn (hpos k)).const_mul (g k))]
  exact Finset.sum_congr rfl fun k _ => by
    rw [MeasureTheory.integral_const_mul, integral_exp_mode (hpos k)]

/-- **A power-law modulus with `alpha ≤ 1` has no terminal viscosity at all.**  So a model with
finitely many relaxation modes and a power-law material are not close: they differ on whether a
viscosity exists. -/
theorem power_law_not_integrableOn {C alpha : ℝ} (hC : C ≠ 0) (halpha : alpha ≤ 1) :
    ¬ IntegrableOn (fun t : ℝ => C * t ^ (-alpha)) (Set.Ioi 1) := by
  intro hint
  have h : IntegrableOn (fun t : ℝ => t ^ (-alpha)) (Set.Ioi 1) := by
    have h2 := hint.const_mul C⁻¹
    have hid : ∀ t : ℝ, C⁻¹ * (C * t ^ (-alpha)) = t ^ (-alpha) := by
      intro t
      field_simp
    simpa [hid] using h2
  rw [integrableOn_Ioi_rpow_iff (by norm_num)] at h
  linarith

/-! ## Sublinear diffusion excludes memoryless steps -/

/-- The mean-square displacement after `n` steps whose covariance is `C`. -/
def msd (C : ℕ → ℕ → ℝ) (m : ℕ) : ℝ := ∑ i ∈ Finset.range m, ∑ j ∈ Finset.range m, C i j

/-- **Uncorrelated stationary steps diffuse exactly linearly.** -/
theorem msd_of_uncorrelated {C : ℕ → ℕ → ℝ} {s2 : ℝ} (hdiag : ∀ i, C i i = s2)
    (hoff : ∀ i j, i ≠ j → C i j = 0) (m : ℕ) : msd C m = m * s2 := by
  unfold msd
  have : ∀ i ∈ Finset.range m, ∑ j ∈ Finset.range m, C i j = s2 := by
    intro i hi
    rw [Finset.sum_eq_single i]
    · exact hdiag i
    · intro j _ hj
      exact hoff i j (Ne.symm hj)
    · intro hcon
      exact absurd hi hcon
  rw [Finset.sum_congr rfl this]
  simp [mul_comm]

/-- **Sublinear diffusion forces net anticorrelation between steps.**  The deficit of the
measured mean-square displacement below the memoryless value is exactly the sum of the
off-diagonal covariances. -/
theorem subdiffusion_forces_anticorrelation {C : ℕ → ℕ → ℝ} {s2 : ℝ} (hdiag : ∀ i, C i i = s2)
    (m : ℕ) (h : msd C m < m * s2) :
    ∑ i ∈ Finset.range m, ∑ j ∈ Finset.range m, (if i = j then 0 else C i j) < 0 := by
  have hsplit : msd C m
      = m * s2 + ∑ i ∈ Finset.range m, ∑ j ∈ Finset.range m, (if i = j then 0 else C i j) := by
    unfold msd
    have hterm : ∀ i ∈ Finset.range m, ∑ j ∈ Finset.range m, C i j
        = s2 + ∑ j ∈ Finset.range m, (if i = j then 0 else C i j) := by
      intro i hi
      have h1 : ∑ j ∈ Finset.range m, C i j
          = ∑ j ∈ Finset.range m, ((if i = j then C i j else 0)
              + (if i = j then 0 else C i j)) := by
        refine Finset.sum_congr rfl fun j _ => ?_
        by_cases hij : i = j <;> simp [hij]
      rw [h1, Finset.sum_add_distrib]
      congr 1
      rw [Finset.sum_ite_eq (Finset.range m) i (fun j => C i j)]
      simp [hi, hdiag i]
    rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib]
    simp [mul_comm]
  linarith [hsplit ▸ h]

/-- **Sublinear diffusion excludes independent steps**: some pair of steps is correlated. -/
theorem subdiffusion_forces_memory {C : ℕ → ℕ → ℝ} {s2 : ℝ} (hdiag : ∀ i, C i i = s2)
    (m : ℕ) (h : msd C m < m * s2) : ∃ i j, i ≠ j ∧ C i j ≠ 0 := by
  by_contra hcon
  push_neg at hcon
  have hzero : ∀ i j, i ≠ j → C i j = 0 := fun i j hij => hcon i j hij
  rw [msd_of_uncorrelated hdiag hzero m] at h
  exact lt_irrefl _ h

/-- **A measured power-law mean-square displacement with exponent below one forces memory at
every number of steps.**  Here `A` is fixed by the one-step displacement `C 0 0 = A`. -/
theorem power_law_msd_forces_memory {C : ℕ → ℕ → ℝ} {A alpha : ℝ} (hA : 0 < A)
    (halpha : alpha < 1) (hdiag : ∀ i, C i i = A)
    (hmsd : ∀ m : ℕ, msd C m = A * (m : ℝ) ^ alpha) {m : ℕ} (hm : 2 ≤ m) :
    ∃ i j, i ≠ j ∧ C i j ≠ 0 := by
  refine subdiffusion_forces_memory hdiag m ?_
  rw [hmsd m]
  have hm1 : (1 : ℝ) < (m : ℝ) := by exact_mod_cast hm.trans_lt' one_lt_two
  have hlt : (m : ℝ) ^ alpha < (m : ℝ) ^ (1 : ℝ) :=
    Real.rpow_lt_rpow_of_exponent_lt hm1 halpha
  rw [Real.rpow_one] at hlt
  nlinarith

/-! ## Aging excludes a stationary model -/

/-- **A model with no waiting-time dependence is wrong about an ageing material.**  If the
modulus measured at lag `t` differs between two waiting times, any single predicted curve `M`
is off by at least half that difference on one of them. -/
theorem aging_forces_error (G : ℝ → ℝ → ℝ) (M : ℝ → ℝ) (tw₁ tw₂ t : ℝ) :
    |G tw₁ t - G tw₂ t| / 2 ≤ max |M t - G tw₁ t| (|M t - G tw₂ t|) := by
  have htri : |G tw₁ t - G tw₂ t| ≤ |M t - G tw₁ t| + |M t - G tw₂ t| := by
    have h : G tw₁ t - G tw₂ t = (M t - G tw₂ t) - (M t - G tw₁ t) := by ring
    calc |G tw₁ t - G tw₂ t| = |(M t - G tw₂ t) - (M t - G tw₁ t)| := by rw [h]
      _ ≤ |M t - G tw₂ t| + |M t - G tw₁ t| := abs_sub _ _
      _ = |M t - G tw₁ t| + |M t - G tw₂ t| := by ring
  have h1 : |M t - G tw₁ t| ≤ max |M t - G tw₁ t| (|M t - G tw₂ t|) := le_max_left _ _
  have h2 : |M t - G tw₂ t| ≤ max |M t - G tw₁ t| (|M t - G tw₂ t|) := le_max_right _ _
  linarith

end Rheology

end IDR
