import Mathlib

/-!
# Part CXLVI — Turnover: a steady-state level is one number about two rates

Intrinsically disordered regions are, in the cell, short-lived: they carry degrons, they are
recognised without unfolding by the proteasome, and their abundance is a steady state between
synthesis and degradation rather than a property of a folding funnel.  Any model that claims
to capture a disordered region *in vivo* is therefore claiming something about a
production–degradation balance, and this file states exactly what such a claim contains.

The abundance obeys `m' = s - d·m`, whose solution is

    level s d m0 t = s/d + (m0 - s/d) · exp (-d t).

We prove:

* `hasDerivAt_level` — it does solve the equation, and `level_zero` fixes the initial value;
* `tendsto_level` — it relaxes to `s/d`, the steady state;
* `steady_state_blind` and `rate_rescaling` — the steady state is invariant under rescaling
  *both* rates by the same factor, and that rescaling is exactly a rescaling of time.  So an
  abundance measurement, however precise, determines only the ratio `s/d`: a fast protein made
  fast and a slow protein made slowly are indistinguishable at steady state;
* `relaxation_identifies` — but a *time course* separates them completely: if two
  parameter pairs generate the same trajectory from the same starting point, and the starting
  point is not already the steady state, the pairs coincide;
* `halfLife` results — the approach time is `log 2 / d`, set by degradation alone and not by
  synthesis, which is the experimental handle;
* `degron_lowers_steady_state` — adding a degron (multiplying `d`) strictly lowers the steady
  state, and `degron_ratio` gives the factor exactly.

The design consequence: a model of a disordered proteome fitted to abundances is fitting one
number per protein and cannot be tested against, or predict, either rate separately.  A
pulse–chase or a decay curve is not a refinement of the abundance measurement; it is the
measurement that makes the model identifiable.
-/

noncomputable section

namespace RequestProject.Turnover

open Real Filter Topology

/-- Abundance at time `t` of a species produced at constant rate `s`, degraded with
first-order rate `d`, starting from `m0`. -/
def level (s d m0 t : ℝ) : ℝ := s / d + (m0 - s / d) * Real.exp (-d * t)

@[simp] lemma level_zero (s d m0 : ℝ) : level s d m0 0 = m0 := by
  unfold level; simp

/-- **The trajectory solves the production–degradation equation.** -/
theorem hasDerivAt_level (s d m0 t : ℝ) (hd : d ≠ 0) :
    HasDerivAt (level s d m0) (s - d * level s d m0 t) t := by
  have h1 : HasDerivAt (fun u : ℝ => -d * u) (-d) t := by
    simpa using (hasDerivAt_id t).const_mul (-d)
  have hexp : HasDerivAt (fun u : ℝ => Real.exp (-d * u)) (Real.exp (-d * t) * (-d)) t :=
    (Real.hasDerivAt_exp (-d * t)).comp t h1
  have h := (hexp.const_mul (m0 - s / d)).const_add (s / d)
  have hval : (m0 - s / d) * (Real.exp (-d * t) * (-d)) = s - d * level s d m0 t := by
    unfold level
    field_simp
    ring
  rw [hval] at h
  exact h

/-- **Relaxation to the steady state.** -/
theorem tendsto_level (s d m0 : ℝ) (hd : 0 < d) :
    Tendsto (level s d m0) atTop (𝓝 (s / d)) := by
  have h0 : Tendsto (fun t : ℝ => d * t) atTop atTop := Filter.tendsto_id.const_mul_atTop hd
  have hlin : Tendsto (fun t : ℝ => -(d * t)) atTop atBot :=
    Filter.tendsto_neg_atTop_atBot.comp h0
  have hexp : Tendsto (fun t : ℝ => Real.exp (-(d * t))) atTop (𝓝 0) :=
    Real.tendsto_exp_atBot.comp hlin
  have h := ((hexp.const_mul (m0 - s / d)).const_add (s / d))
  rw [mul_zero, add_zero] at h
  unfold level
  simp only [neg_mul]
  exact h

/-- The steady-state abundance. -/
def steadyState (s d : ℝ) : ℝ := s / d

/-! ### What an abundance measurement is worth -/

/-- **The steady state is blind to the overall time scale.**  Speeding up synthesis and
degradation by the same factor leaves the abundance untouched. -/
theorem steady_state_blind (s d : ℝ) {k : ℝ} (hk : k ≠ 0) :
    steadyState (k * s) (k * d) = steadyState s d := by
  unfold steadyState
  rw [mul_div_mul_left _ _ hk]

/-- Rescaling both rates is exactly a rescaling of time: the two trajectories are the same
curve run at different speeds. -/
theorem rate_rescaling (s d m0 t : ℝ) {k : ℝ} (hk : k ≠ 0) :
    level (k * s) (k * d) m0 t = level s d m0 (k * t) := by
  unfold level
  rw [mul_div_mul_left _ _ hk]
  ring_nf

/-- **One abundance is one number.**  For every target degradation rate there is a synthesis
rate reproducing the observed steady state exactly. -/
theorem steady_state_underdetermined (s d : ℝ) {d' : ℝ} (hd' : d' ≠ 0) :
    ∃ s' : ℝ, steadyState s' d' = steadyState s d :=
  ⟨d' * (s / d), by unfold steadyState; field_simp⟩

/-! ### What a time course is worth -/

/-- **A decay curve identifies both rates.**  If two production–degradation models generate
the same trajectory from the same initial abundance, and that abundance is not already the
steady state, then the models are equal. -/
theorem relaxation_identifies {s d s' d' m0 : ℝ} (hd : 0 < d) (hd' : 0 < d')
    (hne : m0 ≠ s / d)
    (h : ∀ t : ℝ, 0 ≤ t → level s d m0 t = level s' d' m0 t) :
    s = s' ∧ d = d' := by
  -- the two steady states agree, being the common limit of one function
  have hlim : s / d = s' / d' := by
    have h1 : Tendsto (level s d m0) atTop (𝓝 (s / d)) := tendsto_level s d m0 hd
    have h2 : Tendsto (level s' d' m0) atTop (𝓝 (s' / d')) := tendsto_level s' d' m0 hd'
    have heq : level s d m0 =ᶠ[atTop] level s' d' m0 :=
      Filter.eventually_atTop.2 ⟨0, fun t ht => h t ht⟩
    exact tendsto_nhds_unique h1 (h2.congr' heq.symm)
  have hB : m0 - s / d ≠ 0 := sub_ne_zero.2 hne
  -- with equal steady states the exponentials must agree
  have hexp : ∀ t : ℝ, 0 ≤ t → Real.exp (-d * t) = Real.exp (-d' * t) := by
    intro t ht
    have := h t ht
    unfold level at this
    rw [← hlim] at this
    have h2 : (m0 - s / d) * Real.exp (-d * t) = (m0 - s / d) * Real.exp (-d' * t) := by
      linarith
    exact mul_left_cancel₀ hB h2
  have hd_eq : d = d' := by
    have h1 := hexp 1 zero_le_one
    have h2 : -d * 1 = -d' * 1 := Real.exp_injective h1
    linarith
  refine ⟨?_, hd_eq⟩
  rw [hd_eq] at hlim
  field_simp at hlim
  linarith

/-! ### The experimental handle -/

/-- The time at which the abundance has covered half the distance to its steady state. -/
def halfLife (d : ℝ) : ℝ := Real.log 2 / d

/-- **The relaxation time is set by degradation alone.**  Synthesis fixes where the level
goes, not how fast it gets there. -/
theorem level_halfLife (s d m0 : ℝ) (hd : 0 < d) :
    level s d m0 (halfLife d) = s / d + (m0 - s / d) / 2 := by
  unfold level halfLife
  have hexp : Real.exp (-d * (Real.log 2 / d)) = 1 / 2 := by
    rw [show -d * (Real.log 2 / d) = -Real.log 2 by field_simp]
    rw [Real.exp_neg, Real.exp_log (by norm_num : (0:ℝ) < 2)]
    norm_num
  rw [hexp]
  ring

/-! ### Degrons -/

/-- **A degron strictly lowers the steady state.** -/
theorem degron_lowers_steady_state {s d k : ℝ} (hs : 0 < s) (hd : 0 < d) (hk : 1 < k) :
    steadyState s (k * d) < steadyState s d := by
  unfold steadyState
  rw [div_lt_div_iff₀ (by positivity) hd]
  nlinarith [mul_pos hs hd, sub_pos.2 hk]

/-- The exact factor by which a degron lowers the steady state. -/
theorem degron_ratio {s d k : ℝ} (hd : d ≠ 0) (hk : k ≠ 0) :
    steadyState s (k * d) = steadyState s d / k := by
  unfold steadyState
  field_simp

/-- **The turnover design law.**  For a disordered region whose abundance is a
production–degradation steady state:

1. the trajectory solves the balance equation and relaxes to `s/d`;
2. the steady state is invariant under a common rescaling of both rates — which is nothing
   but a rescaling of time — so an abundance measurement determines only the ratio, and for
   *every* degradation rate there is a synthesis rate matching the observed abundance;
3. a time course from a non-stationary start identifies both rates uniquely;
4. the approach half-time is `log 2 / d`, independent of synthesis, and a degron of strength
   `k` divides the steady state by exactly `k`. -/
theorem turnover_design_law {s d m0 : ℝ} (hd : 0 < d) (hne : m0 ≠ s / d) :
    Tendsto (level s d m0) atTop (𝓝 (steadyState s d)) ∧
    (∀ k : ℝ, k ≠ 0 → steadyState (k * s) (k * d) = steadyState s d ∧
      ∀ t : ℝ, level (k * s) (k * d) m0 t = level s d m0 (k * t)) ∧
    (∀ d' : ℝ, d' ≠ 0 → ∃ s' : ℝ, steadyState s' d' = steadyState s d) ∧
    (∀ s' d' : ℝ, 0 < d' → (∀ t : ℝ, 0 ≤ t → level s d m0 t = level s' d' m0 t) →
      s = s' ∧ d = d') ∧
    level s d m0 (halfLife d) = s / d + (m0 - s / d) / 2 ∧
    (∀ k : ℝ, k ≠ 0 → steadyState s (k * d) = steadyState s d / k) :=
  ⟨tendsto_level s d m0 hd,
    fun _ hk => ⟨steady_state_blind s d hk, fun t => rate_rescaling s d m0 t hk⟩,
    fun _ hd' => steady_state_underdetermined s d hd',
    fun _ _ hd' hcurve => relaxation_identifies hd hd' hne hcurve,
    level_halfLife s d m0 hd,
    fun _ hk => degron_ratio hd.ne' hk⟩

end RequestProject.Turnover
