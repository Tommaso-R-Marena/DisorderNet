/-
# Part XXIII  Mutations: what a sequence-to-ensemble model must be linear in

Every earlier part is about one region in one context.  A model is nevertheless used to
predict *differences*: the effect of a substitution, and how two substitutions combine.
This part fixes the quantity in which that arithmetic is allowed to be done.

The development is `RequestProject.Epistasis`; here the results are bundled.

* `IDR.mutational_design_laws` -- the additive object is the energy: the log-odds cycle of
  a two-site model is exactly `-beta * J`, and it vanishes precisely when the sites are
  uncoupled; the population cycle of two *uncoupled* substitutions is strictly negative, so
  non-additive populations are no evidence of an interaction; the sign of the measured
  epistasis flips with the background; and on an extreme background arbitrarily large
  energetic effects move the population by arbitrarily little.

Read with Part IV.3 (`EnergyModels.lean`): a model must carry an energy function, its
parameters must be fitted and reported in energy, and every quantity it predicts is a
non-linear function of them.
-/
import Mathlib
import RequestProject.Epistasis

namespace IDR

open Epistasis

/-- **The design laws of mutational prediction.**

1. *Additivity is the absence of a cycle*, and the energy cycle of a two-site model is
   exactly its coupling `J`.
2. *The additive quantity is the energy.*  The population's log-odds is the energy up to
   `-beta`, so a double-mutant cycle taken in log-odds returns `-beta * J` exactly, and
   (at non-zero temperature) vanishes if and only if the sites are uncoupled.
3. *Populations do not add even without a coupling.*  Two strictly favourable, uncoupled
   substitutions have a strictly negative population cycle; explicitly, two substitutions
   worth `1 kT` each show zero energy cycle and a strictly negative population cycle.
4. *The sign of epistasis is a property of the background*, not of the pair of
   substitutions: the same pair is negatively epistatic on one background and positively
   epistatic on the mirrored one.
5. *Saturation.*  On a sufficiently stabilising background every further stabilisation,
   however large, moves the population by less than `eps`.

Together: a sequence-to-ensemble model must be additive in energy and must report energies;
population differences are a non-linear, background-dependent, saturating read-out of them,
and cannot be summed, fitted or interpreted as if they were energies. -/
theorem mutational_design_laws :
    -- 1
    ((∀ c d x a b : ℝ, cyc (fun y => c * y + d) x a b = 0) ∧
      (∀ h1 h2 J : ℝ, energy h1 h2 J 1 1 + energy h1 h2 J 0 0
        - energy h1 h2 J 1 0 - energy h1 h2 J 0 1 = J)) ∧
    -- 2
    ((∀ beta h1 h2 J : ℝ,
        logit (pop beta (energy h1 h2 J 1 1)) + logit (pop beta (energy h1 h2 J 0 0))
          - logit (pop beta (energy h1 h2 J 1 0)) - logit (pop beta (energy h1 h2 J 0 1))
          = -(beta * J)) ∧
      (∀ beta : ℝ, beta ≠ 0 → ∀ h1 h2 J : ℝ, (J = 0 ↔
        logit (pop beta (energy h1 h2 J 1 1)) + logit (pop beta (energy h1 h2 J 0 0))
          - logit (pop beta (energy h1 h2 J 1 0))
          - logit (pop beta (energy h1 h2 J 0 1)) = 0))) ∧
    -- 3
    ((∀ a b : ℝ, 0 < a → 0 < b → cyc sig 0 a b < 0) ∧
      (energy (-1) (-1) 0 1 1 + energy (-1) (-1) 0 0 0
          - energy (-1) (-1) 0 1 0 - energy (-1) (-1) 0 0 1 = 0 ∧
        pop 1 (energy (-1) (-1) 0 1 1) + pop 1 (energy (-1) (-1) 0 0 0)
          - pop 1 (energy (-1) (-1) 0 1 0) - pop 1 (energy (-1) (-1) 0 0 1) < 0)) ∧
    -- 4
    (∀ a b : ℝ, 0 < a → 0 < b →
      cyc sig 0 a b < 0 ∧ 0 < cyc sig (-(0 + a + b)) a b) ∧
    -- 5
    (∀ eps : ℝ, 0 < eps → ∃ x₀ : ℝ, ∀ a : ℝ, 0 < a → sig (x₀ + a) - sig x₀ < eps) :=
  ⟨⟨fun c d x a b => cyc_of_affine c d x a b, fun h1 h2 J => energy_cycle_eq_coupling h1 h2 J⟩,
    ⟨fun beta h1 h2 J => logodds_cycle_eq_coupling beta h1 h2 J,
      fun _ hbeta h1 h2 J => coupling_iff_logodds_cycle hbeta h1 h2 J⟩,
    ⟨fun _ _ ha hb => sig_cyc_neg ha hb, population_cycle_not_evidence_of_coupling⟩,
    fun _ _ ha hb => epistasis_sign_depends_on_background ha hb,
    fun _ heps => saturation heps⟩

end IDR
