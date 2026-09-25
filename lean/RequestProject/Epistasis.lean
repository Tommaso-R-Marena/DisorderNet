/-
# Part XXIII.1  Mutations: the additive quantity is the energy, never the population

A model of a disordered region is used, in practice, to predict the *effect of a mutation*:
how much helix a substitution removes, how much a phosphomimetic shifts a population, how
two substitutions combine.  The default analysis is additive -- single-mutant effects are
summed, and a deviation from the sum ("epistasis") is read as evidence of an interaction
between the two sites.  This file shows that the additive analysis is applied to the wrong
quantity, and quantifies what goes wrong.

The setting is the two-state form that every population report takes: a region is ordered
with probability `sig (-(beta * E))`, where `sig` is the logistic function and `E` the free
energy of the ordered state relative to the disordered one.  A double-mutant cycle of a
quantity `f` is `cyc f x a b = f (x+a+b) + f x - f (x+a) - f (x+b)`.

* `cyc_of_affine` -- an affine quantity has zero cycle: additivity is exactly the absence of
  a cycle, so cycles are the right diagnostic *provided* they are taken of the right
  quantity.
* `energy_cycle_eq_coupling` -- for a two-site energy `h1 s1 + h2 s2 + J s1 s2` the energy
  cycle is exactly the coupling `J`.
* `logit_pop`, `logodds_cycle_eq_coupling`, `coupling_iff_logodds_cycle` -- the *log-odds*
  of the population is the energy up to `-beta`, so the log-odds cycle recovers `beta * J`
  exactly, and vanishes precisely when the two sites are uncoupled.  This is the positive
  design law: a sequence-to-ensemble model must be additive in energy, and it is therefore
  non-linear in everything it reports.
* `sig_cyc_neg` -- but the *population* cycle of two independent, uncoupled substitutions is
  strictly negative.  Non-additivity of measured populations is not evidence of an
  interaction (`population_cycle_not_evidence_of_coupling`): it is the curvature of the
  Boltzmann map.
* `cyc_reflect`, `epistasis_sign_depends_on_background` -- worse, the *sign* of the measured
  epistasis is a property of the background: the same two substitutions show negative
  epistasis on one background and positive epistasis on its mirror image.
* `saturation` -- and on an extreme background arbitrarily large energetic effects produce
  arbitrarily small population changes, so population data lose all sensitivity to the
  quantity the model is fitted on.
-/
import Mathlib

namespace IDR

namespace Epistasis

open Real

/-! ## The logistic map from energies to populations -/

/-- The logistic function: the ordered population of a two-state region as a function of
its (negative, in units of `kT`) free energy. -/
noncomputable def sig (x : ℝ) : ℝ := Real.exp x / (1 + Real.exp x)

lemma one_add_exp_pos (x : ℝ) : 0 < 1 + Real.exp x := by positivity

lemma sig_pos (x : ℝ) : 0 < sig x := by
  unfold sig
  exact div_pos (Real.exp_pos x) (one_add_exp_pos x)

lemma sig_lt_one (x : ℝ) : sig x < 1 := by
  unfold sig
  rw [div_lt_one (one_add_exp_pos x)]
  linarith

lemma one_sub_sig (x : ℝ) : 1 - sig x = 1 / (1 + Real.exp x) := by
  unfold sig
  field_simp
  ring

lemma sig_zero : sig 0 = 1 / 2 := by
  unfold sig
  rw [Real.exp_zero]
  norm_num

lemma sig_neg (x : ℝ) : sig (-x) = 1 - sig x := by
  rw [one_sub_sig]
  unfold sig
  rw [Real.exp_neg]
  have hx : (0 : ℝ) < Real.exp x := Real.exp_pos x
  field_simp
  ring

/-- The log-odds (logit) of a population. -/
noncomputable def logit (p : ℝ) : ℝ := Real.log (p / (1 - p))

/-- **The logistic map is inverted by the log-odds.** -/
lemma logit_sig (x : ℝ) : logit (sig x) = x := by
  unfold logit
  rw [one_sub_sig]
  have h : sig x / (1 / (1 + Real.exp x)) = Real.exp x := by
    unfold sig
    field_simp
  rw [h, Real.log_exp]

/-! ## Double-mutant cycles -/

/-- The double-mutant cycle of a quantity `f`: the effect of both substitutions minus the
sum of their separate effects, on the background `x`. -/
def cyc (f : ℝ → ℝ) (x a b : ℝ) : ℝ := f (x + a + b) + f x - f (x + a) - f (x + b)

/-- **Additivity is exactly the absence of a cycle.**  An affine quantity has zero cycle on
every background. -/
lemma cyc_of_affine (c d x a b : ℝ) : cyc (fun y => c * y + d) x a b = 0 := by
  unfold cyc
  ring

/-! ## Two sites, with and without a coupling -/

/-- A two-site energy: fields `h1`, `h2` and a coupling `J`. -/
def energy (h1 h2 J s1 s2 : ℝ) : ℝ := h1 * s1 + h2 * s2 + J * s1 * s2

/-- **The energy cycle is the coupling.** -/
theorem energy_cycle_eq_coupling (h1 h2 J : ℝ) :
    energy h1 h2 J 1 1 + energy h1 h2 J 0 0
      - energy h1 h2 J 1 0 - energy h1 h2 J 0 1 = J := by
  unfold energy
  ring

/-- The ordered population of a two-state region of energy `E` at inverse temperature
`beta`. -/
noncomputable def pop (beta E : ℝ) : ℝ := sig (-(beta * E))

/-- The log-odds of the population is the energy, up to `-beta`. -/
lemma logit_pop (beta E : ℝ) : logit (pop beta E) = -(beta * E) := logit_sig _

/-- **The log-odds cycle recovers the coupling exactly.**  Reported in log-odds, a
double-mutant cycle measures `beta * J` and nothing else. -/
theorem logodds_cycle_eq_coupling (beta h1 h2 J : ℝ) :
    logit (pop beta (energy h1 h2 J 1 1)) + logit (pop beta (energy h1 h2 J 0 0))
      - logit (pop beta (energy h1 h2 J 1 0)) - logit (pop beta (energy h1 h2 J 0 1))
      = -(beta * J) := by
  simp only [logit_pop, energy]
  ring

/-- ... and it vanishes exactly when the two sites are uncoupled. -/
theorem coupling_iff_logodds_cycle {beta : ℝ} (hbeta : beta ≠ 0) (h1 h2 J : ℝ) :
    J = 0 ↔
      logit (pop beta (energy h1 h2 J 1 1)) + logit (pop beta (energy h1 h2 J 0 0))
        - logit (pop beta (energy h1 h2 J 1 0)) - logit (pop beta (energy h1 h2 J 0 1)) = 0 := by
  rw [logodds_cycle_eq_coupling]
  constructor
  · intro h; rw [h]; ring
  · intro h
    have : beta * J = 0 := by linarith
    rcases mul_eq_zero.1 this with h' | h'
    · exact absurd h' hbeta
    · exact h'

/-! ## The population cycle of two uncoupled substitutions is not zero -/

/-- **Global epistasis.**  Two strictly favourable, energetically independent substitutions
have a strictly negative *population* cycle: the populations they produce do not add. -/
theorem sig_cyc_neg {a b : ℝ} (ha : 0 < a) (hb : 0 < b) : cyc sig 0 a b < 0 := by
  set u := Real.exp a with hu_def
  set v := Real.exp b with hv_def
  have hu : 1 < u := Real.one_lt_exp_iff.mpr ha
  have hv : 1 < v := Real.one_lt_exp_iff.mpr hb
  have hu0 : (0 : ℝ) < u := lt_trans zero_lt_one hu
  have hv0 : (0 : ℝ) < v := lt_trans zero_lt_one hv
  have hab : Real.exp (0 + a + b) = u * v := by
    rw [zero_add, Real.exp_add]
  have h1 : sig (0 + a + b) = u * v / (1 + u * v) := by
    unfold sig; rw [hab]
  have h2 : sig (0 + a) = u / (1 + u) := by
    unfold sig; rw [zero_add]
  have h3 : sig (0 + b) = v / (1 + v) := by
    unfold sig; rw [zero_add]
  have hkey : u / (1 + u) + v / (1 + v) - (u * v / (1 + u * v) + 1 / 2)
      = ((u - 1) * (v - 1) * (u * v - 1)) / (2 * (1 + u) * (1 + v) * (1 + u * v)) := by
    field_simp
    ring
  have hpos : 0 < ((u - 1) * (v - 1) * (u * v - 1)) / (2 * (1 + u) * (1 + v) * (1 + u * v)) := by
    apply div_pos
    · have h4 : 0 < u - 1 := by linarith
      have h5 : 0 < v - 1 := by linarith
      have h6 : 0 < u * v - 1 := by nlinarith
      positivity
    · positivity
  unfold cyc
  rw [h1, h2, h3, sig_zero]
  linarith [hkey ▸ hpos]

/-- The cycle is odd under reflection of the background: the logistic curve is
point-symmetric, so the same pair of substitutions has the opposite epistasis on the
mirrored background. -/
lemma cyc_reflect (x a b : ℝ) : cyc sig (-(x + a + b)) a b = -cyc sig x a b := by
  unfold cyc
  have e1 : -(x + a + b) + a + b = -x := by ring
  have e2 : -(x + a + b) + a = -(x + b) := by ring
  have e3 : -(x + a + b) + b = -(x + a) := by ring
  rw [e1, e2, e3, sig_neg, sig_neg, sig_neg, sig_neg]
  ring

/-- **The sign of measured epistasis is a property of the background.**  The same two
substitutions show strictly negative epistasis in populations on one background and
strictly positive epistasis on its mirror image, with no change in any energy. -/
theorem epistasis_sign_depends_on_background {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    cyc sig 0 a b < 0 ∧ 0 < cyc sig (-(0 + a + b)) a b := by
  refine ⟨sig_cyc_neg ha hb, ?_⟩
  rw [cyc_reflect]
  linarith [sig_cyc_neg ha hb]

/-- **A population cycle is not evidence of a coupling.**  Two uncoupled sites (`J = 0`),
each stabilising the ordered state by `1 kT`: the energy cycle is zero, and the population
cycle is strictly negative. -/
theorem population_cycle_not_evidence_of_coupling :
    energy (-1) (-1) 0 1 1 + energy (-1) (-1) 0 0 0
        - energy (-1) (-1) 0 1 0 - energy (-1) (-1) 0 0 1 = 0 ∧
      pop 1 (energy (-1) (-1) 0 1 1) + pop 1 (energy (-1) (-1) 0 0 0)
        - pop 1 (energy (-1) (-1) 0 1 0) - pop 1 (energy (-1) (-1) 0 0 1) < 0 := by
  constructor
  · rw [energy_cycle_eq_coupling]
  · have hcyc := sig_cyc_neg (a := 1) (b := 1) one_pos zero_lt_one
    unfold cyc at hcyc
    have e11 : pop 1 (energy (-1) (-1) 0 1 1) = sig (0 + 1 + 1) := by
      unfold pop energy; norm_num
    have e00 : pop 1 (energy (-1) (-1) 0 0 0) = sig 0 := by
      unfold pop energy; norm_num
    have e10 : pop 1 (energy (-1) (-1) 0 1 0) = sig (0 + 1) := by
      unfold pop energy; norm_num
    have e01 : pop 1 (energy (-1) (-1) 0 0 1) = sig (0 + 1) := by
      unfold pop energy; norm_num
    rw [e11, e00, e10, e01]
    linarith

/-! ## Saturation: populations stop reporting energies -/

/-- **Saturation.**  On a sufficiently stabilising background every further stabilisation,
however large in energy, moves the population by less than `eps`.  Population data
therefore carry vanishing information about the very quantity a sequence-to-ensemble model
must get right. -/
theorem saturation {eps : ℝ} (heps : 0 < eps) :
    ∃ x₀ : ℝ, ∀ a : ℝ, 0 < a → sig (x₀ + a) - sig x₀ < eps := by
  obtain ⟨x₀, hx₀⟩ : ∃ x₀ : ℝ, Real.exp (-x₀) < eps := by
    refine ⟨Real.log (2 / eps), ?_⟩
    rw [Real.exp_neg, Real.exp_log (by positivity)]
    rw [inv_lt_iff_one_lt_mul₀ (by positivity)]
    field_simp
    norm_num
  refine ⟨x₀, fun a _ => ?_⟩
  have h1 : sig (x₀ + a) < 1 := sig_lt_one _
  have h2 : 1 - sig x₀ = 1 / (1 + Real.exp x₀) := one_sub_sig x₀
  have h3 : 1 / (1 + Real.exp x₀) < Real.exp (-x₀) := by
    rw [Real.exp_neg]
    have hx : (0 : ℝ) < Real.exp x₀ := Real.exp_pos x₀
    rw [div_lt_iff₀ (one_add_exp_pos x₀), inv_mul_eq_div, lt_div_iff₀ hx]
    nlinarith
  linarith

end Epistasis

end IDR
