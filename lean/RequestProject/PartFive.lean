/-
# Part V capstone: the statistical design laws

Parts I--III say what a model of an intrinsically disordered region *is* and how big it must
be; Part IV adds the thermodynamics and the measurement theory.  Part V is about the two
things a practitioner actually controls -- the **loss** that is optimised and the **data**
that are available -- and it closes the development by connecting them to the error that
the earlier laws speak in.

`statistical_design_laws` bundles eight clauses:

1. **The loss controls the error** (Pinsker).  `‖p - q‖₁² ≤ 2 KL(p‖q)`: a model trained to
   relative entropy `eps²/2` is within `eps` in the operational metric, so every capacity,
   resolution and data bound proved earlier applies to a KL-trained model.  The same holds
   for force-field (variational free energy) training, with `‖p - q‖₁ ≤ sqrt (2 β ΔF)`.
2. **Reweighting has an exact price.**  Kish's effective sample size of an importance
   reweighting is exactly `N/(1 + χ²)`, and at most `N·exp(-KL)`: refining a simulated
   ensemble by `K` nats costs a factor `e^K` in simulation length.
3. **Response is fluctuation.**  In a perturbed (tilted) ensemble, `d⟨f⟩/dλ = Cov(f, A)` for
   every observable: susceptibilities to ligands, crowders and modifications are properties
   of the unperturbed fluctuations, and a rigid region has none.
4. **A model is a certified free energy** (Gibbs--Bogoliubov--Feynman), with slack exactly
   `(1/β)·KL`.
5. **Data are needed, unconditionally.**  For every estimator, on a pair of ensembles at
   distance `4 eps`, `eps`-accuracy forces `n ≥ 1/(4 eps)` samples; such pairs exist on any
   library with two conformations.  Sharpening the argument -- additivity of relative
   entropy over independent frames, then Pinsker -- upgrades this to the true rate
   `n ≥ 1/(48 eps²)`, quadratic in the tolerance.
6. **And they suffice.**  The empirical ensemble attains expected `ℓ¹` risk `sqrt (m/n)`.
7. **Unless the model can only reuse what it has seen**, in which case accuracy `eps` on a
   maximally disordered target requires `n ≥ m(1 - eps)` -- exponentially many frames in the
   length of the region.
-/
import Mathlib
import RequestProject.Reweighting
import RequestProject.Response
import RequestProject.SampleComplexity
import RequestProject.Estimation
import RequestProject.Pinsker
import RequestProject.SharpBound
import RequestProject.PartFour

namespace IDR

open Finset
open scoped Classical

/-- **The statistical design laws for a model of an intrinsically disordered region.**
Each clause is an instance of a theorem proved in Part V; together with
`IDR.physical_design_laws`, `IDR.quantitative_design_laws` and
`IDR.model_must_be` / `IDR.model_cannot_be` they are the specification this development
delivers. -/
theorem statistical_design_laws :
    -- (1) the training loss controls the operational error
    (∀ (m : ℕ) (p q : Fin m → ℝ), (∀ j, 0 ≤ p j) → (∀ j, 0 < q j) →
        ∑ j, p j = 1 → ∑ j, q j = 1 →
        (∑ j, |p j - q j|) ^ 2 ≤ 2 * klDiv p q) ∧
    (∀ (m : ℕ) (p q : Fin m → ℝ), (∀ j, 0 ≤ p j) → (∀ j, 0 < q j) →
        ∑ j, p j = 1 → ∑ j, q j = 1 → ∀ eps : ℝ, 0 ≤ eps → klDiv p q ≤ eps ^ 2 / 2 →
        ∑ j, |p j - q j| ≤ eps) ∧
    -- (2) the exact price of reweighting a simulated ensemble
    (∀ (m : ℕ) (p q : Fin m → ℝ), (∀ j, 0 ≤ p j) → (∀ j, 0 < q j) →
        ∑ j, p j = 1 → ∑ j, q j = 1 →
        Reweight.essFrac p q = 1 / (1 + Reweight.chiSq p q) ∧
          Reweight.essFrac p q ≤ Real.exp (-(klDiv p q))) ∧
    -- (3) response is a fluctuation, and a rigid region has none
    (∀ (n : ℕ), 0 < n → ∀ q A f : Fin n → ℝ, (∀ j, 0 < q j) → ∀ lam : ℝ,
        HasDerivAt (Response.meanObs q A f) (Response.cov q A f A lam) lam) ∧
    (∀ (n : ℕ), 0 < n → ∀ q A : Fin n → ℝ, (∀ j, 0 < q j) → ∀ lam c : ℝ,
        (∀ j, A j = c) → Response.var q A A lam = 0) ∧
    -- (4) a tractable model is a certified upper bound on the free energy
    (∀ (n : ℕ), 0 < n → ∀ beta : ℝ, 0 < beta → ∀ U U0 : Fin n → ℝ,
        FreeEnergy.freeEnergy beta U (FreeEnergy.boltz beta U)
          ≤ FreeEnergy.freeEnergy beta U0 (FreeEnergy.boltz beta U0)
            + ∑ j, FreeEnergy.boltz beta U0 j * (U j - U0 j)) ∧
    -- (5) no estimator escapes the data requirement
    (∀ (m : ℕ), 2 ≤ m → ∀ eps : ℝ, 0 < eps → eps ≤ 1/4 →
        ∃ p q : Fin m → ℝ, (∀ j, 0 ≤ p j) ∧ (∀ j, 0 ≤ q j) ∧ (∑ j, p j = 1) ∧
          (∑ j, q j = 1) ∧ Learn.l1 p q = 4 * eps ∧
          ∀ (n : ℕ) (T : (Fin n → Fin m) → (Fin m → ℝ)),
            Learn.risk p T ≤ eps → Learn.risk q T ≤ eps → 1 / (4 * eps) ≤ (n : ℝ)) ∧
    -- (5b) and the true dependence on the tolerance is quadratic
    (∀ (n : ℕ) (eps : ℝ), 0 < eps → eps ≤ 1/4 → ∀ T : (Fin n → Fin 2) → (Fin 2 → ℝ),
        Learn.risk (Learn.bern eps) T ≤ eps → Learn.risk (Learn.bern (-eps)) T ≤ eps →
          1 / (48 * eps ^ 2) ≤ (n : ℝ)) ∧
    -- (6) and counting the samples already achieves the order
    (∀ (m n : ℕ), 0 < n → ∀ p : Fin m → ℝ, (∀ j, 0 ≤ p j) → ∑ j, p j = 1 →
        Learn.risk p (Learn.emp (n := n)) ≤ Real.sqrt ((m : ℝ) / n)) ∧
    -- (7) models that can only reuse observed conformations must observe nearly all of them
    (∀ (m n : ℕ), 0 < m → ∀ T : (Fin n → Fin m) → (Fin m → ℝ), Learn.SupportHonest T →
        ∀ eps : ℝ, Learn.risk (Learn.unifW m) T ≤ eps → (m : ℝ) * (1 - eps) ≤ n) := by
  refine ⟨fun m p q hp hq hps hqs => Pinsker.pinsker hp hq hps hqs,
    fun m p q hp hq hps hqs eps heps h => Pinsker.ell1_le_of_kl_le hp hq hps hqs heps h,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro m p q hp hq hps hqs
    exact ⟨Reweight.essFrac_eq hq hps hqs, Reweight.essFrac_le_exp_neg_kl hp hq hps hqs⟩
  · intro n hn q A f hq lam
    exact Response.linear_response hn hq lam
  · intro n hn q A hq lam c hconst
    exact Response.rigid_no_response hn hq lam c hconst
  · intro n hn beta hbeta U U0
    exact Response.bogoliubov hn hbeta U U0
  · intro m hm eps heps heps'
    obtain ⟨p, q, hp, hq, hps, hqs, hD⟩ := Learn.exists_hard_pair hm heps heps'
    exact ⟨p, q, hp, hq, hps, hqs, hD, fun n T hTp hTq =>
      Learn.sample_complexity_two_point hp hq hps hqs T heps hD hTp hTq⟩
  · intro n eps heps heps' T hTp hTq
    exact Learn.sharp_sample_complexity heps heps' T hTp hTq
  · intro m n hn p hp hps
    exact Learn.empirical_risk_le hp hps hn
  · intro m n hm T hT eps hrisk
    exact Learn.support_honest_needs_coverage hT hm hrisk

end IDR
