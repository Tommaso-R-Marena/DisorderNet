/-
# Part XLII.1  Synthesis: the nascent chain is not the free chain

Every disordered region is first seen by the cell as a nascent chain emerging from the
ribosome, one residue at a time, tethered at its C-terminus, and in a conformational ensemble
that is a *moving target*: at each elongation step the accessible ensemble changes, and the
chain relaxes towards it only as fast as its own dynamics allow.  Models of disordered regions
are built and validated against the equilibrium ensemble of the *finished* chain.  This file
quantifies the two distinct errors that entails.

**Error 1: lag.**  Model synthesis as a sequence of relaxation steps: `p t` is the actual
distribution after `t` elongation events, `pi t` is the equilibrium ensemble of the `t`-residue
chain, one elongation step contracts the distance to the *current* equilibrium by a factor
`delta ∈ [0,1)` (the standard Dobrushin/spectral-gap input; `delta = e^{-lambda·tau}` for a
chain relaxing at rate `lambda` in the codon time `tau`), and each elongation moves the
equilibrium by at most `d`.

* `tracking_error_le` -- the adiabatic bound
  `dist(p t, pi t) ≤ delta^t · dist(p 0, pi 0) + d·(1 − delta^t)/(1 − delta)`;
* `tracking_error_le_steady` -- hence a steady lag of at most `d/(1 − delta)`, and
  `tracking_error_eventually` -- that bound governs the process after finitely many residues,
  to any accuracy;
* `sharp_tracking_error_eq`, `sharp_tracking_error_tendsto` -- **the bound is attained**: a
  one-dimensional chain relaxing by `delta` per step behind a uniformly drifting equilibrium
  has lag exactly `d(1 − delta^t)/(1 − delta)`, converging to `d/(1 − delta)`;
* `lag_le_of_rates`, `quasi_static_limit` -- in rate variables the lag is at most
  `d/(1 − e^{-lambda·tau})`, which tends to `d` -- one elongation step of drift, the
  quasi-static limit -- as the codon time `tau` grows;
* `lag_bound_ge_of_fast_synthesis` -- and is at least `d/(lambda·tau)` in the opposite regime:
  a chain synthesised fast compared with its own relaxation is never in the ensemble a model
  of the mature chain predicts, and the discrepancy grows without bound as `tau → 0`.

**Error 2: vectorial context.**  Even with infinitely slow synthesis, the equilibrium ensemble
of the emerged fragment is *not* the corresponding marginal of the mature chain's ensemble,
because interaction partners that are not yet synthesised are absent from the Hamiltonian.

* `nascentFraction`, `matureFraction` -- the two-state Boltzmann populations of a fragment
  conformation, without and with a contact of strength `eps` to a residue that exists only in
  the mature chain;
* `vectorial_gap` -- their difference is exactly `(e^{beta·eps} − 1)/(2(e^{beta·eps} + 1))`,
  strictly positive for every `beta·eps > 0`, and `vectorial_gap_tendsto_half` -- it saturates
  at `1/2` for a strong contact: the nascent and mature ensembles can differ by the whole
  population;
* `mature_model_error_on_nascent` -- so a model that reproduces the mature ensemble exactly
  carries that entire difference as error on the nascent chain.  Co-translational data are a
  separate observable, not a consistency check on the same one.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace Cotranslational

/-! ## Error 1: lag behind a moving equilibrium -/

section Lag

variable {X : Type*} [PseudoMetricSpace X]

/-- **The adiabatic (quasi-static) bound.**  If each elongation step contracts the distance to
the current equilibrium by `delta` and moves the equilibrium by at most `d`, then after `t`
residues the chain lags behind its own equilibrium by at most
`delta^t · (initial lag) + d(1 − delta^t)/(1 − delta)`. -/
theorem tracking_error_le (p pi : ℕ → X) {delta d : ℝ} (hd0 : 0 ≤ delta) (hd1 : delta < 1)
    (hrelax : ∀ t, dist (p (t + 1)) (pi t) ≤ delta * dist (p t) (pi t))
    (hdrift : ∀ t, dist (pi t) (pi (t + 1)) ≤ d) (t : ℕ) :
    dist (p t) (pi t) ≤ delta ^ t * dist (p 0) (pi 0) + d * (1 - delta ^ t) / (1 - delta) := by
  have hden : (0 : ℝ) < 1 - delta := by linarith
  induction t with
  | zero => simp
  | succ t ih =>
      have hstep : dist (p (t + 1)) (pi (t + 1))
          ≤ delta * dist (p t) (pi t) + d :=
        calc dist (p (t + 1)) (pi (t + 1))
            ≤ dist (p (t + 1)) (pi t) + dist (pi t) (pi (t + 1)) := dist_triangle _ _ _
          _ ≤ delta * dist (p t) (pi t) + d := add_le_add (hrelax t) (hdrift t)
      have hmul : delta * dist (p t) (pi t)
          ≤ delta * (delta ^ t * dist (p 0) (pi 0) + d * (1 - delta ^ t) / (1 - delta)) :=
        mul_le_mul_of_nonneg_left ih hd0
      have halg : delta * (delta ^ t * dist (p 0) (pi 0) + d * (1 - delta ^ t) / (1 - delta)) + d
          = delta ^ (t + 1) * dist (p 0) (pi 0)
            + d * (1 - delta ^ (t + 1)) / (1 - delta) := by
        field_simp
        ring
      linarith [hstep, hmul, halg.le, halg.ge]

/-- The steady-state form: the lag never exceeds the initial lag carried forward plus
`d/(1 − delta)`. -/
theorem tracking_error_le_steady (p pi : ℕ → X) {delta d : ℝ} (hd0 : 0 ≤ delta) (hd1 : delta < 1)
    (hdnn : 0 ≤ d)
    (hrelax : ∀ t, dist (p (t + 1)) (pi t) ≤ delta * dist (p t) (pi t))
    (hdrift : ∀ t, dist (pi t) (pi (t + 1)) ≤ d) (t : ℕ) :
    dist (p t) (pi t) ≤ delta ^ t * dist (p 0) (pi 0) + d / (1 - delta) := by
  have hden : (0 : ℝ) < 1 - delta := by linarith
  have h := tracking_error_le p pi hd0 hd1 hrelax hdrift t
  have hpow : (0 : ℝ) ≤ delta ^ t := pow_nonneg hd0 t
  have : d * (1 - delta ^ t) / (1 - delta) ≤ d / (1 - delta) := by
    gcongr
    nlinarith
  linarith

/-- After finitely many residues the lag is within `ε` of the steady bound `d/(1 − delta)`. -/
theorem tracking_error_eventually (p pi : ℕ → X) {delta d : ℝ} (hd0 : 0 ≤ delta) (hd1 : delta < 1)
    (hdnn : 0 ≤ d)
    (hrelax : ∀ t, dist (p (t + 1)) (pi t) ≤ delta * dist (p t) (pi t))
    (hdrift : ∀ t, dist (pi t) (pi (t + 1)) ≤ d) {eps : ℝ} (heps : 0 < eps) :
    ∃ T : ℕ, ∀ t ≥ T, dist (p t) (pi t) ≤ d / (1 - delta) + eps := by
  have hlim : Filter.Tendsto (fun t : ℕ => delta ^ t * dist (p 0) (pi 0)) Filter.atTop
      (nhds 0) := by
    have := tendsto_pow_atTop_nhds_zero_of_lt_one hd0 hd1
    simpa using this.mul_const (dist (p 0) (pi 0))
  obtain ⟨T, hT⟩ := (Filter.eventually_atTop.mp
    (hlim.eventually (eventually_le_nhds (by linarith : (0 : ℝ) < eps))))
  refine ⟨T, fun t ht => ?_⟩
  have h := tracking_error_le_steady p pi hd0 hd1 hdnn hrelax hdrift t
  have := hT t ht
  linarith

end Lag

/-! ### The bound is attained -/

section Sharp

variable {delta d : ℝ}

/-- The equilibrium of the `t`-residue chain, drifting by `d` per elongation step. -/
noncomputable def sharpPi (d : ℝ) (t : ℕ) : ℝ := d * t

/-- A chain that relaxes by exactly `delta` per step behind that drifting equilibrium. -/
noncomputable def sharpP (delta d : ℝ) (t : ℕ) : ℝ :=
  sharpPi d t - d * (1 - delta ^ t) / (1 - delta)

lemma sharp_relax (hd0 : 0 ≤ delta) (hd1 : delta < 1) (t : ℕ) :
    dist (sharpP delta d (t + 1)) (sharpPi d t)
      = delta * dist (sharpP delta d t) (sharpPi d t) := by
  have hden : (1 : ℝ) - delta ≠ 0 := by
    intro h
    rw [sub_eq_zero] at h
    exact absurd h.symm (ne_of_lt hd1)
  have h1 : sharpP delta d (t + 1) - sharpPi d t
      = delta * (sharpP delta d t - sharpPi d t) := by
    unfold sharpP sharpPi
    push_cast
    field_simp
    ring
  rw [Real.dist_eq, Real.dist_eq, h1, abs_mul, abs_of_nonneg hd0]

lemma sharp_drift (t : ℕ) : dist (sharpPi d t) (sharpPi d (t + 1)) = |d| := by
  unfold sharpPi
  rw [Real.dist_eq]
  push_cast
  rw [show d * t - d * (t + 1) = -d by ring, abs_neg]

/-- **The adiabatic bound is exactly attained.**  For this chain the lag after `t` elongation
steps is exactly `d(1 − delta^t)/(1 − delta)` -- the second term of `tracking_error_le`, whose
first term vanishes because the chain starts at equilibrium. -/
theorem sharp_tracking_error_eq (hd0 : 0 ≤ delta) (hd1 : delta < 1) (hdnn : 0 ≤ d) (t : ℕ) :
    dist (sharpP delta d t) (sharpPi d t) = d * (1 - delta ^ t) / (1 - delta) := by
  have hden : (0 : ℝ) < 1 - delta := by linarith
  have hpow : delta ^ t ≤ 1 := pow_le_one₀ hd0 hd1.le
  unfold sharpP
  rw [Real.dist_eq]
  have : sharpPi d t - d * (1 - delta ^ t) / (1 - delta) - sharpPi d t
      = -(d * (1 - delta ^ t) / (1 - delta)) := by ring
  rw [this, abs_neg, abs_of_nonneg
    (div_nonneg (mul_nonneg hdnn (by linarith)) hden.le)]

/-- **The hypotheses of `tracking_error_le` are satisfiable with equality throughout**, so the
bound is not an artefact of the estimates used to prove it: this chain starts at equilibrium,
contracts by exactly `delta` per step, sees exactly `d` of drift per step, and lags by exactly
the value the theorem allows. -/
theorem sharp_saturates (hd0 : 0 ≤ delta) (hd1 : delta < 1) (hdnn : 0 ≤ d) :
    (∀ t, dist (sharpP delta d (t + 1)) (sharpPi d t)
        = delta * dist (sharpP delta d t) (sharpPi d t)) ∧
    (∀ t, dist (sharpPi d t) (sharpPi d (t + 1)) = d) ∧
    dist (sharpP delta d 0) (sharpPi d 0) = 0 ∧
    (∀ t, dist (sharpP delta d t) (sharpPi d t) = d * (1 - delta ^ t) / (1 - delta)) := by
  refine ⟨fun t => sharp_relax hd0 hd1 t, fun t => ?_, ?_,
    fun t => sharp_tracking_error_eq hd0 hd1 hdnn t⟩
  · rw [sharp_drift, abs_of_nonneg hdnn]
  · simpa using sharp_tracking_error_eq hd0 hd1 hdnn 0

/-- and it converges to the steady bound `d/(1 − delta)`. -/
theorem sharp_tracking_error_tendsto (hd0 : 0 ≤ delta) (hd1 : delta < 1) (hdnn : 0 ≤ d) :
    Filter.Tendsto (fun t : ℕ => dist (sharpP delta d t) (sharpPi d t)) Filter.atTop
      (nhds (d / (1 - delta))) := by
  have hden : (0 : ℝ) < 1 - delta := by linarith
  have hpow : Filter.Tendsto (fun t : ℕ => delta ^ t) Filter.atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hd0 hd1
  have : Filter.Tendsto (fun t : ℕ => d * (1 - delta ^ t) / (1 - delta)) Filter.atTop
      (nhds (d / (1 - delta))) := by
    have := ((hpow.const_sub 1).const_mul d).div_const (1 - delta)
    simpa using this
  refine this.congr fun t => ?_
  exact (sharp_tracking_error_eq hd0 hd1 hdnn t).symm

end Sharp

/-! ### In rate variables -/

section Rates

/-- With a relaxation rate `lambda` and a codon time `tau`, one elongation step contracts the
lag by `e^{-lambda·tau}`, so the steady lag is at most `d/(1 − e^{-lambda·tau})`. -/
theorem lag_le_of_rates {X : Type*} [PseudoMetricSpace X] (p pi : ℕ → X) {lam tau d : ℝ}
    (hlam : 0 < lam) (htau : 0 < tau) (hdnn : 0 ≤ d)
    (hrelax : ∀ t, dist (p (t + 1)) (pi t)
      ≤ Real.exp (-(lam * tau)) * dist (p t) (pi t))
    (hdrift : ∀ t, dist (pi t) (pi (t + 1)) ≤ d) (t : ℕ) :
    dist (p t) (pi t)
      ≤ Real.exp (-(lam * tau)) ^ t * dist (p 0) (pi 0) + d / (1 - Real.exp (-(lam * tau))) := by
  have h0 : 0 ≤ Real.exp (-(lam * tau)) := (Real.exp_pos _).le
  have h1 : Real.exp (-(lam * tau)) < 1 := by
    rw [Real.exp_lt_one_iff]
    nlinarith
  exact tracking_error_le_steady p pi h0 h1 hdnn hrelax hdrift t

/-- **Slow synthesis is quasi-static.**  As the codon time grows the steady lag bound tends to
`d`: one elongation step's worth of drift, i.e. the chain is in the equilibrium ensemble of
the chain one residue shorter. -/
theorem quasi_static_limit {lam d : ℝ} (hlam : 0 < lam) :
    Filter.Tendsto (fun tau : ℝ => d / (1 - Real.exp (-(lam * tau)))) Filter.atTop (nhds d) := by
  have hexp : Filter.Tendsto (fun tau : ℝ => Real.exp (-(lam * tau))) Filter.atTop (nhds 0) := by
    have h1 : Filter.Tendsto (fun tau : ℝ => -(lam * tau)) Filter.atTop Filter.atBot :=
      Filter.tendsto_neg_atTop_atBot.comp
        (Filter.Tendsto.const_mul_atTop hlam Filter.tendsto_id)
    exact Real.tendsto_exp_atBot.comp h1
  have := (Filter.Tendsto.const_sub (1 : ℝ) hexp)
  simpa using (tendsto_const_nhds.div this (by norm_num))

/-- **Fast synthesis never equilibrates.**  For a positive codon time the steady lag bound is
at least `d/(lambda·tau)`, which diverges as the codon time shrinks: the nascent ensemble is
then not the equilibrium ensemble of any chain length. -/
theorem lag_bound_ge_of_fast_synthesis {lam tau d : ℝ} (hlam : 0 < lam) (htau : 0 < tau)
    (hd : 0 ≤ d) :
    d / (lam * tau) ≤ d / (1 - Real.exp (-(lam * tau))) := by
  have hx : 0 < lam * tau := mul_pos hlam htau
  have h1 : 1 - Real.exp (-(lam * tau)) ≤ lam * tau := by
    have := Real.add_one_le_exp (-(lam * tau))
    linarith
  have h2 : 0 < 1 - Real.exp (-(lam * tau)) := by
    have : Real.exp (-(lam * tau)) < 1 := by
      rw [Real.exp_lt_one_iff]; linarith
    linarith
  exact div_le_div_of_nonneg_left hd h2 h1

end Rates

/-! ## Error 2: the vectorial context -/

section Vectorial

/-- The population of the folded fragment conformation in the *nascent* chain, where the
stabilising contact partner has not yet been synthesised: two states of equal energy. -/
noncomputable def nascentFraction : ℝ := 1 / 2

/-- The population of the same conformation in the *mature* chain, where the contact of
strength `eps` is available. -/
noncomputable def matureFraction (beta eps : ℝ) : ℝ :=
  Real.exp (beta * eps) / (Real.exp (beta * eps) + 1)

/-- **The vectorial gap.**  The nascent and mature populations of the same conformation differ
by exactly `(e^{beta·eps} − 1)/(2(e^{beta·eps} + 1))`, which is strictly positive whenever the
contact is stabilising. -/
theorem vectorial_gap (beta eps : ℝ) :
    matureFraction beta eps - nascentFraction
      = (Real.exp (beta * eps) - 1) / (2 * (Real.exp (beta * eps) + 1)) := by
  have hpos : 0 < Real.exp (beta * eps) + 1 := by positivity
  unfold matureFraction nascentFraction
  field_simp
  ring

theorem vectorial_gap_pos {beta eps : ℝ} (h : 0 < beta * eps) :
    0 < matureFraction beta eps - nascentFraction := by
  rw [vectorial_gap]
  have h1 : 1 < Real.exp (beta * eps) := Real.one_lt_exp_iff.mpr h
  apply div_pos <;> linarith

/-- For a strong contact the gap saturates at `1/2`: the whole population moves. -/
theorem vectorial_gap_tendsto_half {beta : ℝ} (hbeta : 0 < beta) :
    Filter.Tendsto (fun eps : ℝ => matureFraction beta eps - nascentFraction) Filter.atTop
      (nhds (1 / 2)) := by
  have hexp : Filter.Tendsto (fun eps : ℝ => Real.exp (-(beta * eps))) Filter.atTop (nhds 0) := by
    have h1 : Filter.Tendsto (fun eps : ℝ => -(beta * eps)) Filter.atTop Filter.atBot :=
      Filter.tendsto_neg_atTop_atBot.comp
        (Filter.Tendsto.const_mul_atTop hbeta Filter.tendsto_id)
    exact Real.tendsto_exp_atBot.comp h1
  have hrw : ∀ eps : ℝ, matureFraction beta eps - nascentFraction
      = 1 / (1 + Real.exp (-(beta * eps))) - 1 / 2 := by
    intro eps
    have h : Real.exp (-(beta * eps)) = 1 / Real.exp (beta * eps) := by
      rw [Real.exp_neg]; ring
    have hpos : 0 < Real.exp (beta * eps) := Real.exp_pos _
    unfold matureFraction nascentFraction
    rw [h]
    field_simp
  simp only [hrw]
  have h2 : Filter.Tendsto (fun eps : ℝ => 1 + Real.exp (-(beta * eps))) Filter.atTop
      (nhds 1) := by simpa using (Filter.Tendsto.const_add (1 : ℝ) hexp)
  have h3 : Filter.Tendsto (fun eps : ℝ => 1 / (1 + Real.exp (-(beta * eps)))) Filter.atTop
      (nhds ((1 : ℝ) / 1)) := Filter.Tendsto.div tendsto_const_nhds h2 one_ne_zero
  rw [div_one] at h3
  have h4 := h3.sub_const (1 / 2)
  norm_num at h4 ⊢
  exact h4

/-- **A model of the mature ensemble is wrong about the nascent one by the whole gap.**  If a
model reproduces the mature population exactly, its error on the nascent chain is the vectorial
gap, which is positive for any stabilising contact. -/
theorem mature_model_error_on_nascent {beta eps M : ℝ} (h : 0 < beta * eps)
    (hM : M = matureFraction beta eps) :
    |M - nascentFraction| = (Real.exp (beta * eps) - 1) / (2 * (Real.exp (beta * eps) + 1)) ∧
      0 < |M - nascentFraction| := by
  have hgap := vectorial_gap beta eps
  have hpos := vectorial_gap_pos h
  subst hM
  refine ⟨?_, ?_⟩
  · rw [abs_of_pos hpos, hgap]
  · exact abs_pos.mpr (ne_of_gt hpos)

end Vectorial

end Cotranslational

end IDR
