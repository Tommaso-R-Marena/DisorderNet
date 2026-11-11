/-
# Part XX  The ensemble you sample is the integrator's, not the model's

Every reported ensemble of a disordered region that came from a simulation came from a
*discrete* integrator run at a finite timestep.  That is not a detail of implementation: a
finite timestep changes the stationary distribution.  This file proves it exactly, in the one
case where everything can be computed in closed form -- an overdamped Langevin (Euler--
Maruyama) step in a harmonic well of stiffness `k` at inverse temperature `beta`, which is the
normal-mode description of a flexible chain (`RequestProject.Rouse`).

The update `x ↦ (1 − k·dt) x + sqrt(2 dt/beta) ξ` acts on the variance by
`varStep : v ↦ (1 − k·dt)^2 v + 2 dt/beta`.

* `varSeq_eq` -- the variance after `n` steps is `discVar + r^n (v₀ − discVar)` with
  `r = (1 − k·dt)^2`;
* `varSeq_tendsto` -- so for `0 < k·dt < 2` the simulation equilibrates, geometrically, to
  `discVar = 2 / (beta·k·(2 − k·dt))`;
* `exactVar_lt_discVar`, `bias_eq` -- which is **not** the Boltzmann variance `1/(beta·k)`:
  the sampled ensemble is too broad, by exactly `dt / (beta (2 − k·dt))`;
* `discVar_eq_effK`, `effK_lt`, `integrator_samples_a_softer_force_field` -- and the excess is
  not noise.  The simulation is *exactly* the equilibrium ensemble of a different force field,
  one with stiffness `effK = k(2 − k·dt)/2 < k`.  A finite timestep is a silent change of
  Hamiltonian, so the ensemble reported for a sequence belongs to the integrator as much as to
  the energy function.
* `bias_tendsto_zero_at_zero_timestep`, `bias_pos` -- the bias vanishes only as `dt → 0`, and
  is strictly positive at every `dt > 0`;
* `no_sampling_removes_the_timestep_bias` -- and it is a *bias*, not a variance: however long
  the run, the sampled variance eventually stays a fixed distance away from the target.  The
  statistical theory of Part V governs the fluctuation around the wrong answer.
* `unstable_of_large_timestep` -- past the stability limit `k·dt > 2` the chain does not
  merely mis-sample: the variance diverges.

And the positive half of the statement:

* `unbiased_iff_consistency` -- for *any* scheme acting affinely on the variance,
  `v ↦ r v + s`, sampling the Boltzmann ensemble exactly is the single algebraic condition
  `s = (1 − r)/(beta·k)` matching the noise to the damping;
* `ou_unbiased` -- the exact Ornstein--Uhlenbeck propagator satisfies it at *every* timestep.
  So the bias is a property of the chosen scheme, not of discreteness as such, and
  `reported_ensemble_belongs_to_the_integrator` is the design statement: name the integrator,
  and prefer one that meets the consistency condition.
-/
import Mathlib

namespace IDR

namespace Integrator

/-- One Euler--Maruyama step of overdamped Langevin dynamics in a harmonic well acts on the
variance of the (Gaussian) population by this affine map. -/
noncomputable def varStep (k beta dt v : ℝ) : ℝ := (1 - k * dt) ^ 2 * v + 2 * dt / beta

/-- The variance after `n` integrator steps, starting from `v₀`. -/
noncomputable def varSeq (k beta dt v₀ : ℝ) : ℕ → ℝ
  | 0 => v₀
  | n + 1 => varStep k beta dt (varSeq k beta dt v₀ n)

/-- The Boltzmann variance of the harmonic well: what the model *means*. -/
noncomputable def exactVar (k beta : ℝ) : ℝ := 1 / (beta * k)

/-- The stationary variance of the finite-timestep integrator: what the simulation
*samples*. -/
noncomputable def discVar (k beta dt : ℝ) : ℝ := 2 / (beta * k * (2 - k * dt))

/-- The stiffness of the force field the integrator is actually sampling. -/
noncomputable def effK (k dt : ℝ) : ℝ := k * (2 - k * dt) / 2

lemma discVar_eq_effK {k beta dt : ℝ} (hk : 0 < k) (hbeta : 0 < beta) (hstab : k * dt < 2) :
    discVar k beta dt = exactVar (effK k dt) beta := by
  have h2 : (2 : ℝ) - k * dt ≠ 0 := by linarith
  unfold discVar exactVar effK
  field_simp

lemma effK_lt {k dt : ℝ} (hk : 0 < k) (hdt : 0 < dt) : effK k dt < k := by
  have : 0 < k * (k * dt) := mul_pos hk (mul_pos hk hdt)
  unfold effK
  nlinarith

lemma effK_pos {k dt : ℝ} (hk : 0 < k) (hstab : k * dt < 2) : 0 < effK k dt := by
  unfold effK
  have : 0 < 2 - k * dt := by linarith
  positivity

/-- `discVar` is the fixed point of one integrator step. -/
lemma varStep_discVar {k beta dt : ℝ} (hk : 0 < k) (hbeta : 0 < beta) (hstab : k * dt < 2) :
    varStep k beta dt (discVar k beta dt) = discVar k beta dt := by
  have h2 : (2 : ℝ) - k * dt ≠ 0 := by linarith
  unfold varStep discVar
  field_simp
  ring

/-- The variance after `n` steps, in closed form: geometric relaxation to `discVar`. -/
theorem varSeq_eq {k beta dt v₀ : ℝ} (hk : 0 < k) (hbeta : 0 < beta) (hstab : k * dt < 2)
    (n : ℕ) :
    varSeq k beta dt v₀ n
      = discVar k beta dt + ((1 - k * dt) ^ 2) ^ n * (v₀ - discVar k beta dt) := by
  induction n with
  | zero => simp [varSeq]
  | succ n ih =>
      have hfix := varStep_discVar hk hbeta hstab
      unfold varSeq
      rw [ih]
      unfold varStep at hfix ⊢
      linear_combination hfix

/-- Inside the stability limit the integrator equilibrates, geometrically, to its own
stationary variance. -/
theorem varSeq_tendsto {k beta dt v₀ : ℝ} (hk : 0 < k) (hbeta : 0 < beta) (hdt : 0 < dt)
    (hstab : k * dt < 2) :
    Filter.Tendsto (varSeq k beta dt v₀) Filter.atTop (nhds (discVar k beta dt)) := by
  have hr0 : (0 : ℝ) ≤ (1 - k * dt) ^ 2 := sq_nonneg _
  have hr1 : (1 - k * dt) ^ 2 < 1 := by
    have h1 : -1 < 1 - k * dt := by nlinarith
    have h2 : 1 - k * dt < 1 := by nlinarith
    nlinarith
  have hpow : Filter.Tendsto (fun n : ℕ => ((1 - k * dt) ^ 2) ^ n) Filter.atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hr0 hr1
  have h0 : Filter.Tendsto
      (fun n : ℕ => discVar k beta dt + ((1 - k * dt) ^ 2) ^ n * (v₀ - discVar k beta dt))
      Filter.atTop (nhds (discVar k beta dt)) := by
    simpa using (hpow.mul_const (v₀ - discVar k beta dt)).const_add (discVar k beta dt)
  exact h0.congr fun n => (varSeq_eq hk hbeta hstab n).symm

/-- **The finite timestep is a systematic error.**  The sampled variance exceeds the
Boltzmann variance by exactly `dt / (beta (2 − k·dt))`. -/
theorem bias_eq {k beta dt : ℝ} (hk : 0 < k) (hbeta : 0 < beta) (hstab : k * dt < 2) :
    discVar k beta dt - exactVar k beta = dt / (beta * (2 - k * dt)) := by
  have h2 : (2 : ℝ) - k * dt ≠ 0 := by linarith
  unfold discVar exactVar
  field_simp
  ring

lemma bias_pos {k beta dt : ℝ} (hk : 0 < k) (hbeta : 0 < beta) (hdt : 0 < dt)
    (hstab : k * dt < 2) : 0 < discVar k beta dt - exactVar k beta := by
  rw [bias_eq hk hbeta hstab]
  have : 0 < beta * (2 - k * dt) := by
    have : 0 < 2 - k * dt := by linarith
    positivity
  positivity

/-- The simulated ensemble is strictly broader than the one the energy function specifies. -/
theorem exactVar_lt_discVar {k beta dt : ℝ} (hk : 0 < k) (hbeta : 0 < beta) (hdt : 0 < dt)
    (hstab : k * dt < 2) : exactVar k beta < discVar k beta dt := by
  linarith [bias_pos hk hbeta hdt hstab]

/-- The bias vanishes only in the limit of vanishing timestep, and then to first order. -/
theorem bias_tendsto_zero_at_zero_timestep {k beta : ℝ} (hbeta : 0 < beta) :
    Filter.Tendsto (fun dt : ℝ => dt / (beta * (2 - k * dt))) (nhds 0) (nhds 0) := by
  have hcont : ContinuousAt (fun dt : ℝ => dt / (beta * (2 - k * dt))) 0 := by
    refine ContinuousAt.div (by fun_prop) (by fun_prop) ?_
    simp only [mul_zero, sub_zero]
    positivity
  simpa using hcont.tendsto

/-- **The simulation is the equilibrium of a different force field.**  At timestep `dt` the
integrator samples exactly the Boltzmann ensemble of a harmonic well with the strictly smaller
stiffness `effK = k(2 − k·dt)/2`.  A finite timestep is a silent change of Hamiltonian. -/
theorem integrator_samples_a_softer_force_field {k beta dt : ℝ} (hk : 0 < k) (hbeta : 0 < beta)
    (hdt : 0 < dt) (hstab : k * dt < 2) :
    0 < effK k dt ∧ effK k dt < k ∧ discVar k beta dt = exactVar (effK k dt) beta :=
  ⟨effK_pos hk hstab, effK_lt hk hdt, discVar_eq_effK hk hbeta hstab⟩

/-- **No amount of sampling removes it.**  The sampled variance converges -- but to the wrong
number: beyond some run length it stays at least half the bias away from the Boltzmann
variance, for every initial condition. -/
theorem no_sampling_removes_the_timestep_bias {k beta dt v₀ : ℝ} (hk : 0 < k) (hbeta : 0 < beta)
    (hdt : 0 < dt) (hstab : k * dt < 2) :
    ∃ N : ℕ, ∀ n ≥ N,
      (discVar k beta dt - exactVar k beta) / 2 ≤ varSeq k beta dt v₀ n - exactVar k beta := by
  set B := discVar k beta dt - exactVar k beta with hB
  have hBpos : 0 < B := bias_pos hk hbeta hdt hstab
  have htend := varSeq_tendsto (v₀ := v₀) hk hbeta hdt hstab
  have hmem : Set.Ioi (discVar k beta dt - B / 2) ∈ nhds (discVar k beta dt) :=
    Ioi_mem_nhds (by linarith)
  obtain ⟨N, hN⟩ := (htend.eventually_mem hmem).exists_forall_of_atTop
  refine ⟨N, fun n hn => ?_⟩
  have := hN n hn
  simp only [Set.mem_Ioi] at this
  have : discVar k beta dt - B / 2 < varSeq k beta dt v₀ n := this
  linarith

/-- **Past the stability limit the integrator does not merely mis-sample.**  If `k·dt > 2`
the variance grows without bound from any starting point at or above the fixed point of the
recursion, so the trajectory leaves every bounded region. -/
theorem unstable_of_large_timestep {k beta dt v₀ : ℝ} (hbeta : 0 < beta) (hdt : 0 < dt)
    (hunstab : 2 < k * dt) (hv₀ : 0 ≤ v₀) :
    Filter.Tendsto (varSeq k beta dt v₀) Filter.atTop Filter.atTop := by
  have hr : (1 : ℝ) < (1 - k * dt) ^ 2 := by nlinarith
  have hc : 0 < 2 * dt / beta := by positivity
  -- the sequence dominates the geometric sequence `c * r^n`
  have key : ∀ n : ℕ, (2 * dt / beta) * ((1 - k * dt) ^ 2) ^ n ≤ varSeq k beta dt v₀ (n + 1) := by
    intro n
    induction n with
    | zero =>
        simp only [pow_zero, mul_one, varSeq, varStep]
        nlinarith [sq_nonneg (1 - k * dt)]
    | succ n ih =>
        have hstep : varSeq k beta dt v₀ (n + 2)
            = (1 - k * dt) ^ 2 * varSeq k beta dt v₀ (n + 1) + 2 * dt / beta := rfl
        rw [hstep, pow_succ]
        nlinarith [sq_nonneg (1 - k * dt)]
  have hgeo : Filter.Tendsto (fun n : ℕ => (2 * dt / beta) * ((1 - k * dt) ^ 2) ^ n)
      Filter.atTop Filter.atTop :=
    Filter.Tendsto.const_mul_atTop hc (tendsto_pow_atTop_atTop_of_one_lt hr)
  have hshift : Filter.Tendsto (fun n : ℕ => varSeq k beta dt v₀ (n + 1)) Filter.atTop
      Filter.atTop := Filter.tendsto_atTop_mono key hgeo
  exact (Filter.tendsto_add_atTop_iff_nat (f := varSeq k beta dt v₀) 1).mp hshift

/-! ## The bias is a property of the scheme, not of discreteness

Any integrator whose one-step action on the variance of a Gaussian population is affine --
which covers every scheme built from linear drift and additive noise -- has stationary
variance `s / (1 - r)`.  Sampling the *right* ensemble is then a single algebraic condition
on the scheme, and the exact Ornstein--Uhlenbeck propagator satisfies it at every timestep. -/

/-- The stationary variance of a general affine variance map `v ↦ r v + s`. -/
noncomputable def genStat (r s : ℝ) : ℝ := s / (1 - r)

lemma genStat_fixed {r s : ℝ} (hr : r ≠ 1) : r * genStat r s + s = genStat r s := by
  have h : (1 : ℝ) - r ≠ 0 := sub_ne_zero.mpr (Ne.symm hr)
  unfold genStat
  field_simp
  ring

/-- Euler--Maruyama in the harmonic well is the affine map with `r = (1 - k dt)^2`,
`s = 2 dt / beta`, and `discVar` is its stationary variance. -/
lemma discVar_eq_genStat {k beta dt : ℝ} (hk : 0 < k) (hbeta : 0 < beta) (hdt : 0 < dt)
    (hstab : k * dt < 2) :
    discVar k beta dt = genStat ((1 - k * dt) ^ 2) (2 * dt / beta) := by
  have h2 : (2 : ℝ) - k * dt ≠ 0 := by linarith
  have hkd : k * dt ≠ 0 := by positivity
  unfold discVar genStat
  rw [show (1 : ℝ) - (1 - k * dt) ^ 2 = k * dt * (2 - k * dt) by ring]
  field_simp

/-- **Unbiased sampling is one algebraic condition on the scheme.**  An affine integrator
reproduces the Boltzmann variance exactly iff its noise term is matched to its damping. -/
theorem unbiased_iff_consistency {k beta r s : ℝ} (hk : 0 < k) (hbeta : 0 < beta) (hr : r ≠ 1) :
    genStat r s = exactVar k beta ↔ s = (1 - r) / (beta * k) := by
  have h : (1 : ℝ) - r ≠ 0 := sub_ne_zero.mpr (Ne.symm hr)
  have hbk : beta * k ≠ 0 := by positivity
  unfold genStat exactVar
  rw [div_eq_div_iff h hbk]
  constructor
  · intro hs
    field_simp
    linarith [hs]
  · intro hs
    rw [hs]
    field_simp

/-- The exact (Ornstein--Uhlenbeck) propagator of the harmonic well: damping `exp(-k dt)` on
the coordinate, hence `r = exp(-2 k dt)` on the variance, with the matched noise. -/
noncomputable def ouR (k dt : ℝ) : ℝ := Real.exp (-(2 * k * dt))

noncomputable def ouS (k beta dt : ℝ) : ℝ := (1 - Real.exp (-(2 * k * dt))) / (beta * k)

lemma ouR_lt_one {k dt : ℝ} (hk : 0 < k) (hdt : 0 < dt) : ouR k dt < 1 := by
  unfold ouR
  exact Real.exp_lt_one_iff.mpr (by nlinarith)

/-- **The exact propagator samples the exact ensemble, at every timestep.**  So the
finite-timestep bias of Part XX is a property of the scheme, not of discreteness. -/
theorem ou_unbiased {k beta dt : ℝ} (hk : 0 < k) (hbeta : 0 < beta) (hdt : 0 < dt) :
    genStat (ouR k dt) (ouS k beta dt) = exactVar k beta :=
  (unbiased_iff_consistency hk hbeta (ouR_lt_one hk hdt).ne).mpr rfl

/-- **The design statement for a simulated ensemble.**  At any fixed timestep the naive
integrator samples the equilibrium of a softened force field, strictly broader than the
target, and no length of run repairs it; while a scheme satisfying the consistency condition
samples the target exactly.  A reported ensemble is therefore a statement about an integrator
as well as about an energy function, and the integrator must be part of the model. -/
theorem reported_ensemble_belongs_to_the_integrator {k beta dt : ℝ} (hk : 0 < k)
    (hbeta : 0 < beta) (hdt : 0 < dt) (hstab : k * dt < 2) :
    exactVar k beta < discVar k beta dt ∧
      discVar k beta dt - exactVar k beta = dt / (beta * (2 - k * dt)) ∧
      (0 < effK k dt ∧ effK k dt < k ∧ discVar k beta dt = exactVar (effK k dt) beta) ∧
      genStat (ouR k dt) (ouS k beta dt) = exactVar k beta :=
  ⟨exactVar_lt_discVar hk hbeta hdt hstab, bias_eq hk hbeta hstab,
    integrator_samples_a_softer_force_field hk hbeta hdt hstab, ou_unbiased hk hbeta hdt⟩

end Integrator

end IDR
