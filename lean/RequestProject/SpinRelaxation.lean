/-
# Part XXXII.1  Spin relaxation: what a nuclear relaxation rate can see of the motion

Nuclear spin relaxation is the standard probe of *dynamics* in a disordered region: for each
backbone amide one measures a longitudinal rate `R₁`, a transverse rate `R₂` and a
heteronuclear NOE, and every one of them is a fixed linear combination of the *spectral
density* `J` evaluated at a handful of frequencies fixed by the spectrometer.  The motion of
the chain enters only through

  `J(ω) = Σ_k w_k · 2τ_k / (1 + (ω τ_k)²)`   (`specDens`),

the Fourier transform of a bond-vector correlation function written as a finite mixture of
exponentials with correlation times `τ_k` and weights `w_k` (`lorentz` is one Lorentzian).
This file asks what that map does and does not determine.

Exact facts about a single Lorentzian:

* `lorentz_zero`, `specDens_zero` -- `J(0) = 2⟨τ⟩` is exactly twice the *mean* correlation
  time; combined with `specDens_ge_term` (`J(0) ≥ 2 w_k τ_k` for every component) this is the
  precise sense in which a minority slow state dominates transverse relaxation:
  `minority_slow_state_dominates` shows a 1% population that is 10³ times slower than the bulk
  raises `J(0)` above ten times the bulk value.
* `lorentz_strictAnti_freq`, `specDens_strictAnti_freq` -- the spectral density is strictly
  decreasing in frequency, so `J(0)` is the largest value the experiment can return.
* `lorentz_le_inv_freq` -- at fixed frequency the Lorentzian is at most `1/ω`, attained exactly
  when `ω τ = 1`: relaxation is maximally sensitive to motions at the Larmor frequency and is
  progressively blind on either side.
* `lorentz_tau_ambiguity`, `lorentz_eq_iff` -- and it is *two-to-one*: `τ` and `1/(ω²τ)` give
  exactly the same value.  A single rate does not distinguish a slow motion from the
  corresponding fast one; that is the classical `τ_c` ambiguity, and it is exact.

What a full experiment determines:

* `twoComponent_identifiable` -- on two known correlation times whose product is not `1/ω²` the
  populations *are* identified by one frequency.  The obstruction below is not softness of the
  data but the geometry of the map.
* `relaxation_underdetermined` -- two explicit four-component motional models, both with
  strictly positive populations, predict *identical* spectral densities at three distinct
  frequencies — as many independent numbers as `R₁`, `R₂` and the NOE provide at one field —
  while their mean correlation times, and hence `J(0)`, differ by `77/240` of the unit.  A
  relaxation data set at one field therefore does not determine the distribution of
  correlation times, and any timescale quoted from it is a property of the fitting model.
* `modelFree_eq_specDens` -- the Lipari--Szabo "model-free" form is exactly the two-component
  case of `specDens`, with populations `S²` and `1 − S²` and correlation times `τ_m` and
  `τ_m τ_e/(τ_m + τ_e)`.  It is a *choice of two components*, not a model-independent readout,
  and `modelFree_underdetermined` exhibits two different `(S², τ_e)` pairs that a measurement
  at a given frequency cannot tell apart.

Design consequence for a model of a disordered region: relaxation data constrain the ensemble's
dynamics only through a few values of `J`, they weight the slow tail of the correlation-time
distribution enormously, and they are formally degenerate.  A model must be compared with the
*rates*, forward-modelled at the fields used, and never with a fitted `S²` or `τ_c`.
-/
import Mathlib

set_option autoImplicit false

namespace Relax

open Finset

/-- One Lorentzian: the contribution of a motion of correlation time `tau` to the spectral
density at angular frequency `om`. -/
noncomputable def lorentz (tau om : ℝ) : ℝ := 2 * tau / (1 + (om * tau) ^ 2)

/-- The spectral density of a motion described by a finite mixture of exponential correlation
functions with populations `w` and correlation times `tau`. -/
noncomputable def specDens {m : ℕ} (w tau : Fin m → ℝ) (om : ℝ) : ℝ :=
  ∑ k, w k * lorentz (tau k) om

@[simp] lemma lorentz_zero (tau : ℝ) : lorentz tau 0 = 2 * tau := by
  simp [lorentz]

lemma lorentz_pos {tau om : ℝ} (h : 0 < tau) : 0 < lorentz tau om := by
  have : (0 : ℝ) < 1 + (om * tau) ^ 2 := by positivity
  exact div_pos (by linarith) this

lemma lorentz_nonneg {tau om : ℝ} (h : 0 ≤ tau) : 0 ≤ lorentz tau om := by
  have : (0 : ℝ) < 1 + (om * tau) ^ 2 := by positivity
  exact div_nonneg (by linarith) this.le

/-- **`J(0)` is twice the mean correlation time.** -/
theorem specDens_zero {m : ℕ} (w tau : Fin m → ℝ) :
    specDens w tau 0 = 2 * ∑ k, w k * tau k := by
  simp [specDens, Finset.mul_sum]
  refine Finset.sum_congr rfl ?_
  intro k _
  ring

/-- Every component contributes to `J(0)`: a slow component cannot be hidden. -/
theorem specDens_ge_term {m : ℕ} {w tau : Fin m → ℝ} (hw : ∀ k, 0 ≤ w k) (htau : ∀ k, 0 ≤ tau k)
    (j : Fin m) : 2 * (w j * tau j) ≤ specDens w tau 0 := by
  rw [specDens_zero]
  have : w j * tau j ≤ ∑ k, w k * tau k :=
    Finset.single_le_sum (f := fun k => w k * tau k)
      (fun k _ => mul_nonneg (hw k) (htau k)) (Finset.mem_univ j)
  linarith

/-- **A minority slow state dominates transverse relaxation.**  A population of `1/100` that is
`1000` times slower than the bulk already raises `J(0)` above ten times the bulk value. -/
theorem minority_slow_state_dominates {tau0 : ℝ} (h : 0 < tau0) :
    10 * (2 * tau0) < specDens ![99/100, 1/100] ![tau0, 1000 * tau0] 0 := by
  rw [specDens_zero]
  simp [Fin.sum_univ_two]
  linarith

/-- The Lorentzian is strictly decreasing in frequency. -/
theorem lorentz_strictAnti_freq {tau om1 om2 : ℝ} (htau : 0 < tau) (h0 : 0 ≤ om1)
    (h : om1 < om2) : lorentz tau om2 < lorentz tau om1 := by
  have hd1 : (0 : ℝ) < 1 + (om1 * tau) ^ 2 := by positivity
  have hd2 : (0 : ℝ) < 1 + (om2 * tau) ^ 2 := by positivity
  rw [lorentz, lorentz, div_lt_div_iff₀ hd2 hd1]
  have hsq : (om1 * tau) ^ 2 < (om2 * tau) ^ 2 := by
    have h1 : 0 ≤ om1 * tau := mul_nonneg h0 htau.le
    have h2 : om1 * tau < om2 * tau := by
      exact mul_lt_mul_of_pos_right h htau
    nlinarith
  nlinarith

/-- **The spectral density is strictly decreasing in frequency**, so `J(0)` is the largest value
the experiment can return. -/
theorem specDens_strictAnti_freq {m : ℕ} [NeZero m] {w tau : Fin m → ℝ} (hw : ∀ k, 0 < w k)
    (htau : ∀ k, 0 < tau k) {om1 om2 : ℝ} (h0 : 0 ≤ om1) (h : om1 < om2) :
    specDens w tau om2 < specDens w tau om1 := by
  refine Finset.sum_lt_sum_of_nonempty ?_ ?_
  · exact Finset.univ_nonempty
  · intro k _
    exact mul_lt_mul_of_pos_left (lorentz_strictAnti_freq (htau k) h0 h) (hw k)

/-- **Relaxation is maximally sensitive to motion at the Larmor frequency**: at frequency `om`
the Lorentzian never exceeds `1/om`, and it attains that value exactly when `om * tau = 1`. -/
theorem lorentz_le_inv_freq {tau om : ℝ} (htau : 0 < tau) (hom : 0 < om) :
    lorentz tau om ≤ 1 / om := by
  have hd : (0 : ℝ) < 1 + (om * tau) ^ 2 := by positivity
  rw [lorentz, div_le_div_iff₀ hd hom]
  nlinarith [sq_nonneg (om * tau - 1)]

theorem lorentz_eq_inv_freq {tau om : ℝ} (hom : 0 < om) (h : om * tau = 1) :
    lorentz tau om = 1 / om := by
  have htau : tau = 1 / om := by field_simp at h ⊢; linarith
  subst htau
  rw [lorentz]
  field_simp
  ring

/-- **The `τ_c` ambiguity, exactly.**  At a fixed frequency, `tau` and `1/(om² tau)` give the
same Lorentzian: one rate does not distinguish a slow motion from the matching fast one. -/
theorem lorentz_tau_ambiguity {tau om : ℝ} (htau : 0 < tau) (hom : 0 < om) :
    lorentz (1 / (om ^ 2 * tau)) om = lorentz tau om := by
  have h1 : om ^ 2 * tau ≠ 0 := by positivity
  have hd : (0 : ℝ) < 1 + (om * tau) ^ 2 := by positivity
  rw [lorentz, lorentz]
  rw [div_eq_div_iff (by positivity) (ne_of_gt hd)]
  field_simp
  ring

/-- The Lorentzian is two-to-one: it identifies `tau` only up to the reflection
`tau ↦ 1/(om² tau)`. -/
theorem lorentz_eq_iff {t1 t2 om : ℝ} (h1 : 0 < t1) (h2 : 0 < t2) :
    lorentz t1 om = lorentz t2 om ↔ t1 = t2 ∨ om ^ 2 * t1 * t2 = 1 := by
  have hd1 : (0 : ℝ) < 1 + (om * t1) ^ 2 := by positivity
  have hd2 : (0 : ℝ) < 1 + (om * t2) ^ 2 := by positivity
  rw [lorentz, lorentz, div_eq_div_iff (ne_of_gt hd1) (ne_of_gt hd2)]
  constructor
  · intro h
    have key : (t1 - t2) * (1 - om ^ 2 * t1 * t2) = 0 := by nlinarith
    rcases mul_eq_zero.mp key with h' | h'
    · exact Or.inl (by linarith)
    · exact Or.inr (by linarith)
  · rintro (rfl | h)
    · ring
    · nlinarith

/-- **Two known correlation times are identifiable.**  If the two correlation times are not
related by the `τ_c` reflection, the populations are determined by one frequency. -/
theorem twoComponent_identifiable {t1 t2 om a b : ℝ} (h1 : 0 < t1) (h2 : 0 < t2)
    (hne : t1 ≠ t2) (hrefl : om ^ 2 * t1 * t2 ≠ 1)
    (h : specDens ![a, 1 - a] ![t1, t2] om = specDens ![b, 1 - b] ![t1, t2] om) :
    a = b := by
  have hL : lorentz t1 om ≠ lorentz t2 om := by
    intro hcon
    rcases (lorentz_eq_iff h1 h2).mp hcon with h' | h'
    · exact hne h'
    · exact hrefl h'
  simp only [specDens, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one] at h
  have : (a - b) * (lorentz t1 om - lorentz t2 om) = 0 := by ring_nf; ring_nf at h; linarith
  rcases mul_eq_zero.mp this with h' | h'
  · linarith
  · exact absurd (by linarith : lorentz t1 om = lorentz t2 om) hL

/-! ### The Lipari--Szabo model-free form is a two-Lorentzian mixture -/

/-- The "model-free" spectral density: an order parameter `S²`, an overall tumbling time `taum`
and an internal correlation time `taue`. -/
noncomputable def modelFree (S2 taum taue om : ℝ) : ℝ :=
  S2 * lorentz taum om + (1 - S2) * lorentz (taum * taue / (taum + taue)) om

/-- **The model-free form is exactly the two-component case of a mixture of Lorentzians**: `S²`
and `1 − S²` are populations, and the "internal" correlation time is the harmonic combination
of the tumbling and internal times. -/
theorem modelFree_eq_specDens (S2 taum taue om : ℝ) :
    modelFree S2 taum taue om
      = specDens ![S2, 1 - S2] ![taum, taum * taue / (taum + taue)] om := by
  simp [modelFree, specDens, Fin.sum_univ_two]

/-- **`S²` is not read off a measurement.**  At `om = 1` with tumbling time `taum = 1`, the pairs
`(S², τ_e) = (1/2, 1)` and `(3/4, 1/2)` give exactly the same spectral density; the order
parameter is a property of the two-component fit, not a measured number. -/
theorem modelFree_underdetermined :
    modelFree (1/2) 1 1 1 = modelFree (3/4) 1 (1/2) 1 := by
  simp [modelFree, lorentz]
  norm_num

/-! ### What a one-field data set leaves open -/

/-- Four correlation times, spanning a decade and a half around the inverse Larmor frequency. -/
noncomputable def tauW : Fin 4 → ℝ := ![1/4, 1/3, 3, 4]

/-- One motional model: the four components equally populated. -/
noncomputable def pW : Fin 4 → ℝ := ![1/4, 1/4, 1/4, 1/4]

/-- A second motional model, with strictly positive and quite different populations. -/
noncomputable def qW : Fin 4 → ℝ := ![1/8, 117/320, 43/320, 3/8]

lemma pW_pos (k : Fin 4) : 0 < pW k := by fin_cases k <;> norm_num [pW]

lemma qW_pos (k : Fin 4) : 0 < qW k := by fin_cases k <;> norm_num [qW]

lemma pW_sum : ∑ k, pW k = 1 := by simp [pW, Fin.sum_univ_four]; norm_num

lemma qW_sum : ∑ k, qW k = 1 := by simp [qW, Fin.sum_univ_four]; norm_num

lemma tauW_pos (k : Fin 4) : 0 < tauW k := by fin_cases k <;> norm_num [tauW]

/-- **A one-field relaxation data set does not determine the motion.**  Two four-component
motional models with strictly positive populations over the same correlation times predict
*identical* spectral densities at three distinct frequencies — as many independent numbers as
`R₁`, `R₂` and the heteronuclear NOE provide at one field — yet their mean correlation times,
i.e. their values of `J(0)`, differ by `77/240`. -/
theorem relaxation_underdetermined :
    (∀ k, 0 < pW k) ∧ (∀ k, 0 < qW k) ∧ ∑ k, pW k = 1 ∧ ∑ k, qW k = 1 ∧
      specDens pW tauW (1/2) = specDens qW tauW (1/2) ∧
      specDens pW tauW 1 = specDens qW tauW 1 ∧
      specDens pW tauW 2 = specDens qW tauW 2 ∧
      specDens qW tauW 0 - specDens pW tauW 0 = 77/240 := by
  refine ⟨pW_pos, qW_pos, pW_sum, qW_sum, ?_, ?_, ?_, ?_⟩ <;>
    simp [specDens, lorentz, Fin.sum_univ_four, pW, qW, tauW] <;> norm_num

end Relax
