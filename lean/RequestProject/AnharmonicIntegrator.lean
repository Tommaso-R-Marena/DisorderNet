/-
# Part CVII  Integrators beyond the Gaussian case

Part XX prices the finite timestep for a linear (harmonic) mode, and the assumptions list said
what that leaves: "it is the normal-mode statement, not a theorem about anharmonic force fields".
A disordered region is exactly where the normal-mode picture is weakest — its slow motions are
large-amplitude and its potential along them is nothing like quadratic.  This file proves what
survives the anharmonicity, and what does not.

* `verlet_reversible` — **what survives is the structure.**  The velocity-Verlet step is exactly
  reversible for *every* force, anharmonic ones included: running the step with `−dt` on its output
  returns the input identically, with no error term.  This is the property that makes the scheme
  usable at all, and it is not a Gaussian fact.
* `harmonic_shadow_invariant` — **for a harmonic mode the scheme has an exactly conserved
  quantity**, `om²(1 − dt²om²/4)q² + p²`, a modified energy that is not the true energy: the
  finite-timestep bias of Part XX in its conserved form.
* `harmonic_bounded` — hence the classical stability condition `dt·om < 2` in its strongest form:
  below it the shadow energy is positive definite and *every* trajectory is bounded for all time,
  with explicit bounds on position and momentum.
* `quartic_unstable` — **and what does not survive is the stability condition.**  For the quartic
  oscillator `V = q⁴/4` there is no such timestep: for *every* `dt > 0` there is an initial
  condition whose trajectory grows at least like `5ⁿ`.  Stability of an anharmonic integrator is a
  statement about the trajectory as well as the timestep, and no timestep chosen from the harmonic
  frequencies of a force field is unconditionally safe.

The design consequence for a model of a disordered region: a timestep validated on the fast
harmonic modes of the force field carries no guarantee on the soft anharmonic ones, and an
integrator that has been stable for the length of a run is evidence about the region of phase
space visited, not about the scheme.  This is the honest replacement for the Gaussian statement.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR.AnharmonicIntegrator

/-! ## The velocity-Verlet step for an arbitrary force -/

/-- One velocity-Verlet step with force `F` (unit mass) at timestep `dt`, acting on the pair
`(position, momentum)`. -/
noncomputable def vstep (F : ℝ → ℝ) (dt : ℝ) (s : ℝ × ℝ) : ℝ × ℝ :=
  let ph := s.2 + dt / 2 * F s.1
  let q' := s.1 + dt * ph
  (q', ph + dt / 2 * F q')

/-- **The velocity-Verlet step is exactly reversible, for every force.**  No anharmonicity, and no
size of timestep, spoils it. -/
theorem verlet_reversible (F : ℝ → ℝ) (dt : ℝ) (s : ℝ × ℝ) :
    vstep F (-dt) (vstep F dt s) = s := by
  unfold vstep
  rw [Prod.ext_iff]
  constructor <;> simp <;> ring_nf

/-! ## The harmonic mode: an exactly conserved shadow energy -/

/-- The harmonic force of angular frequency `om`. -/
noncomputable def harmForce (om : ℝ) : ℝ → ℝ := fun q => -(om ^ 2 * q)

/-- The shadow energy exactly conserved by velocity Verlet on a harmonic mode. -/
noncomputable def shadowE (dt om : ℝ) (s : ℝ × ℝ) : ℝ :=
  om ^ 2 * (1 - dt ^ 2 * om ^ 2 / 4) * s.1 ^ 2 + s.2 ^ 2

/-- **The shadow energy is exactly invariant** — the finite-timestep bias in conserved form. -/
theorem harmonic_shadow_invariant (dt om : ℝ) (s : ℝ × ℝ) :
    shadowE dt om (vstep (harmForce om) dt s) = shadowE dt om s := by
  unfold shadowE vstep harmForce
  simp only
  ring

/-- The shadow energy is conserved along the whole trajectory. -/
theorem harmonic_shadow_iterate (dt om : ℝ) (s : ℝ × ℝ) (n : ℕ) :
    shadowE dt om ((vstep (harmForce om) dt)^[n] s) = shadowE dt om s := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Function.iterate_succ_apply', harmonic_shadow_invariant, ih]

/-- **Below the stability limit every harmonic trajectory is bounded for all time.** -/
theorem harmonic_bounded {dt om : ℝ} (hdt : 0 < dt) (hom : 0 < om) (hstab : dt * om < 2)
    (s : ℝ × ℝ) (n : ℕ) :
    ((vstep (harmForce om) dt)^[n] s).2 ^ 2 ≤ shadowE dt om s ∧
      om ^ 2 * (1 - dt ^ 2 * om ^ 2 / 4) * ((vstep (harmForce om) dt)^[n] s).1 ^ 2
        ≤ shadowE dt om s := by
  have hcoef : 0 < om ^ 2 * (1 - dt ^ 2 * om ^ 2 / 4) := by
    have h1 : dt ^ 2 * om ^ 2 < 4 := by nlinarith [mul_pos hdt hom]
    have h2 : 0 < om ^ 2 := by positivity
    nlinarith
  have hinv := harmonic_shadow_iterate dt om s n
  set t := (vstep (harmForce om) dt)^[n] s with ht
  have hE : om ^ 2 * (1 - dt ^ 2 * om ^ 2 / 4) * t.1 ^ 2 + t.2 ^ 2 = shadowE dt om s := hinv
  constructor
  · nlinarith [sq_nonneg t.1, sq_nonneg t.2]
  · nlinarith [sq_nonneg t.1, sq_nonneg t.2]

/-! ## The quartic mode: no timestep is safe -/

/-- The leapfrog (position-Verlet) trajectory of the quartic oscillator `V = q⁴/4`, as the pair of
consecutive positions, started from rest at `Q`. -/
noncomputable def lfPair (dt Q : ℝ) : ℕ → ℝ × ℝ
  | 0 => (Q, Q)
  | (n + 1) =>
      let s := lfPair dt Q n
      (s.2, 2 * s.2 - s.1 - dt ^ 2 * s.2 ^ 3)

/-- **For every timestep there is a quartic trajectory that blows up.**  With `Q = 3/dt` the
leapfrog positions grow at least geometrically, at rate `5` per step: an anharmonic force field
has no unconditionally stable timestep. -/
theorem quartic_unstable {dt : ℝ} (hdt : 0 < dt) :
    ∃ Q : ℝ, 0 < Q ∧ ∀ n : ℕ, 5 ^ n * Q ≤ |(lfPair dt Q n).2| := by
  refine ⟨3 / dt, by positivity, ?_⟩
  set Q : ℝ := 3 / dt with hQ
  have hQpos : 0 < Q := by positivity
  have hdtQ : dt ^ 2 * Q ^ 2 = 9 := by
    rw [hQ]
    field_simp
    norm_num
  have key : ∀ n : ℕ, Q ≤ |(lfPair dt Q n).1| ∧ |(lfPair dt Q n).1| ≤ |(lfPair dt Q n).2|
      ∧ 5 ^ n * Q ≤ |(lfPair dt Q n).2| := by
    intro n
    induction n with
    | zero => simp [lfPair, abs_of_pos hQpos]
    | succ n ih =>
        obtain ⟨h1, h2, h3⟩ := ih
        set a := (lfPair dt Q n).1 with ha
        set b := (lfPair dt Q n).2 with hb
        have hbQ : Q ≤ |b| := le_trans h1 h2
        have hnew : (lfPair dt Q (n + 1)) = (b, 2 * b - a - dt ^ 2 * b ^ 3) := by
          simp [lfPair, ← ha, ← hb]
        have habs : dt ^ 2 * |b| ^ 3 - (2 * |b| + |a|) ≤ |2 * b - a - dt ^ 2 * b ^ 3| := by
          have e1 : |2 * b - a - dt ^ 2 * b ^ 3| = |dt ^ 2 * b ^ 3 - (2 * b - a)| := by
            rw [← abs_neg]
            ring_nf
          rw [e1]
          have e2 : |dt ^ 2 * b ^ 3| - |2 * b - a| ≤ |dt ^ 2 * b ^ 3 - (2 * b - a)| :=
            abs_sub_abs_le_abs_sub _ _
          have e3 : |dt ^ 2 * b ^ 3| = dt ^ 2 * |b| ^ 3 := by
            rw [abs_mul, abs_of_pos (by positivity : (0:ℝ) < dt ^ 2), abs_pow]
          have e4 : |2 * b - a| ≤ 2 * |b| + |a| := by
            calc |2 * b - a| ≤ |2 * b| + |a| := abs_sub _ _
              _ = 2 * |b| + |a| := by rw [abs_mul]; norm_num
          linarith [e2, e3, e4]
        have hcube : 9 * |b| ≤ dt ^ 2 * |b| ^ 3 := by
          have hb0 : 0 < |b| := lt_of_lt_of_le hQpos hbQ
          have : Q ^ 2 ≤ |b| ^ 2 := by nlinarith
          nlinarith [sq_nonneg (dt * |b|)]
        have hgrow : 6 * |b| ≤ |2 * b - a - dt ^ 2 * b ^ 3| := by
          have : |a| ≤ |b| := h2
          linarith [habs, hcube]
        refine ⟨?_, ?_, ?_⟩
        · rw [hnew]
          exact hbQ
        · rw [hnew]
          simp only
          have hb0 : 0 ≤ |b| := abs_nonneg b
          linarith [hgrow]
        · rw [hnew]
          simp only
          have : (5 : ℝ) ^ (n + 1) * Q = 5 * (5 ^ n * Q) := by ring
          rw [this]
          linarith [hgrow, h3]
  intro n
  exact (key n).2.2

end IDR.AnharmonicIntegrator
