/-
# Part VI capstone: the precision design laws, and the complete specification

Parts I--III fix what a model of an intrinsically disordered region is and how large it must
be; Part IV adds the thermodynamics and the measurement theory; Part V the loss and the
data.  Part VI closes the development with the *inverse* problem that motivates most
experiments on disorder: not "what is the ensemble?" but "what is the ensemble telling me
about the conditions?" -- a ligand, a denaturant, a modification, a crowder.

`precision_design_laws` bundles seven clauses:

1. **Information is fluctuation.**  The Fisher information a single conformation carries
   about the strength of a perturbation is the variance of the conjugate observable, which
   by Part V.2 is also the susceptibility.  Response, fluctuation and information are one
   quantity.
2. **Free energy is convex in the coupling**, so the family is a regular exponential family
   and the susceptibility can never be negative: thermodynamic stability.
3. **The Cramér--Rao bound.**  Every unbiased estimate of the perturbation strength from a
   conformation has variance at least `1/fisher`.
4. **Rigidity is blindness.**  If the perturbation cannot move the region, no unbiased
   estimator of it exists at all, at any sample size.  Conversely two conformations that
   differ along the coupling already give strictly positive information.  The disorder is
   the measurement channel.
5. **Chapman--Robbins**, the assumption-free version: for *any* readout of a conformation,
   the shift between two contexts is at most `sqrt (Var · chi²)`.
6. **The chi-squared cost tensorises**: `n` frames turn `chi²` into `(1 + chi²)^n - 1`.
7. **Detection and reweighting are reciprocal.**  Separating two contexts by `delta` with a
   readout that fluctuates by `V` needs `n ≥ log (1 + delta²/V) / log (1 + chi²)` frames --
   the reciprocal of the reweighting cost that Part V.1 shows is the price of *moving*
   between the same two contexts.

Together with `IDR.model_must_be`, `IDR.model_cannot_be`, `IDR.quantitative_design_laws`,
`IDR.physical_design_laws` and `IDR.statistical_design_laws`, this completes the
specification the development delivers.
-/
import Mathlib
import RequestProject.Fisher
import RequestProject.Repeats
import RequestProject.PartFive
import RequestProject.Verdict
import RequestProject.Design

namespace IDR

open Finset
open scoped Classical

/-- **The precision design laws for a model of an intrinsically disordered region.**
Each clause is an instance of a theorem proved in Part VI. -/
theorem precision_design_laws :
    -- (1) information carried about a perturbation is the fluctuation of its conjugate
    (∀ (n : ℕ), 0 < n → ∀ q A : Fin n → ℝ, (∀ j, 0 < q j) → ∀ lam : ℝ,
        HasDerivAt (Response.meanObs q A A) (Fisher.fisher q A lam) lam) ∧
    -- (2) the free energy is convex in the coupling
    (∀ (n : ℕ), 0 < n → ∀ q A : Fin n → ℝ, (∀ j, 0 < q j) →
        ConvexOn ℝ Set.univ (fun l => Real.log (Response.part q A l))) ∧
    -- (3) the Cramér--Rao bound
    (∀ (n : ℕ), 0 < n → ∀ q A T : Fin n → ℝ, (∀ j, 0 < q j) →
        (∀ l, Response.meanObs q A T l = l) → ∀ lam : ℝ,
          1 ≤ Response.var q A T lam * Fisher.fisher q A lam) ∧
    -- (4) a rigid region admits no unbiased estimator of its context, a disordered one does
    (∀ (n : ℕ), 0 < n → ∀ q A : Fin n → ℝ, (∀ j, 0 < q j) → ∀ c : ℝ, (∀ j, A j = c) →
        ¬ ∃ T : Fin n → ℝ, ∀ l, Response.meanObs q A T l = l) ∧
    (∀ (n : ℕ), 0 < n → ∀ q A : Fin n → ℝ, (∀ j, 0 < q j) → ∀ lam : ℝ, ∀ j₁ j₂ : Fin n,
        A j₁ ≠ A j₂ → 0 < Fisher.fisher q A lam) ∧
    -- (5) Chapman--Robbins: no assumptions at all
    (∀ (m : ℕ) (p q T : Fin m → ℝ), (∀ j, 0 < q j) → ∑ j, p j = 1 → ∑ j, q j = 1 →
        |(∑ j, p j * T j) - ∑ j, q j * T j|
          ≤ Real.sqrt (Fisher.varW q T * Reweight.chiSq p q)) ∧
    -- (6) the chi-squared cost tensorises multiplicatively over independent frames
    (∀ (m n : ℕ) (p q : Fin m → ℝ), (∀ j, 0 < q j) → ∑ j, p j = 1 → ∑ j, q j = 1 →
        1 + Fisher.chiSqG (Learn.prodP (n := n) p) (Learn.prodP (n := n) q)
          = (1 + Reweight.chiSq p q) ^ n) ∧
    -- (7) so the frames needed to detect a change of context invert the reweighting cost
    (∀ (m n : ℕ) (p q : Fin m → ℝ) (T : (Fin n → Fin m) → ℝ), (∀ j, 0 < q j) →
        ∑ j, p j = 1 → ∑ j, q j = 1 → ∀ delta V : ℝ, 0 ≤ delta → 0 < V →
        Fisher.varW (Learn.prodP (n := n) q) T ≤ V →
        delta ≤ |(∑ s, Learn.prodP p s * T s) - ∑ s, Learn.prodP q s * T s| →
          Real.log (1 + delta ^ 2 / V) ≤ n * Real.log (1 + Reweight.chiSq p q)) := by
  refine ⟨fun n hn q A hq lam => Fisher.fisher_eq_susceptibility hn hq lam,
    fun n hn q A hq => Fisher.logPart_convexOn hn hq,
    fun n hn q A T hq hub lam => Fisher.cramer_rao hn hq hub lam,
    fun n hn q A hq c hconst => Fisher.no_unbiased_estimator_of_rigid hn hq c hconst,
    fun n hn q A hq lam j₁ j₂ hne => Fisher.fisher_pos_of_disordered hn hq lam hne,
    fun m p q T hq hps hqs => ?_,
    fun m n p q hq hps hqs => Fisher.chiSqG_prodP hq hps hqs,
    fun m n p q T hq hps hqs delta V hdelta hV hvar hshift =>
      Fisher.log_frames_needed T hq hps hqs hdelta hV hvar hshift⟩
  have h := Fisher.context_discrimination (T := T) hq hps hqs
  rwa [Fisher.chiSqG_eq_chiSq] at h

end IDR
