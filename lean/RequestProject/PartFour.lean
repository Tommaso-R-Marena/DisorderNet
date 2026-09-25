/-
# Part IV capstone: the physical design laws

`RequestProject.Design` collects the *informational* laws -- how error is measured, how
capacity must scale, how refinement works, what invariances cost.  Part IV adds the laws
that come from the physics and from the measurement process, and this file bundles them
into one statement, `physical_design_laws`.

Read as a specification, its six clauses say:

1. **Train by free energy; it is training by KL.**  The excess variational free energy of a
   model over the Boltzmann ensemble is exactly `1/β` times its relative entropy to the
   truth, and the Boltzmann ensemble is the unique minimiser.  A variational fit is
   therefore unbiased as an objective: whatever it gets wrong is the model class, not the
   loss.
2. **Disorder is entropic, and the entropy has a price.**  A conformation can hold half the
   population only if its energy beats its `|D|` competitors by `(1/β) log |D|`.  Absent
   that gap -- the definition of an intrinsically disordered region -- no single structure
   dominates.
3. **Size the model geometrically.**  Against a target with `m` substates pairwise `Δ`
   apart, transport distortion `D` requires `m(1 - 2D/Δ)` structures -- a rate--distortion
   law that, unlike the `ℓ¹` bounds, cannot be evaded by predicting near-copies.
4. **The numbers, for a real disordered polymer.**  For the freely jointed chain of `N`
   bonds: single-structure error at least `N b²`, exact correctness requires `2^N`
   components, distortion `D` still requires `2^N(1 - 2D)`.
5. **Experiments contract.**  Measurement can only reduce the distinguishability of two
   candidate ensembles (data processing), and an experiment of insensitivity `alpha` with
   precision `eps` leaves a whole ball of radius `eps/(1-alpha)` of ensembles fitting the
   data equally well.  Fitting is never recovery of a unique answer.
6. **Coupling costs latent capacity.**  A region whose parts are coupled through `m`
   mutually exclusive substates needs exactly `m` latent states -- one per substate -- and
   in particular no factorised model at all.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.ModelNature
import RequestProject.EnergyModels
import RequestProject.Metric
import RequestProject.MaxEnt
import RequestProject.Invariance
import RequestProject.Transport
import RequestProject.Design
import RequestProject.FreeEnergy
import RequestProject.Quantization
import RequestProject.Chain
import RequestProject.Channel
import RequestProject.LatentRank

namespace IDR

open Finset
open scoped Classical

/-- **The physical design laws for a model of an intrinsically disordered region.**  Each
clause is an instance of a theorem proved in Part IV; together with
`IDR.quantitative_design_laws` and `IDR.model_must_be` / `IDR.model_cannot_be` they are the
specification this development delivers. -/
theorem physical_design_laws :
    -- (1) variational training is training by KL, with a unique optimum
    (∀ (n : ℕ), 0 < n → ∀ (beta : ℝ), 0 < beta → ∀ (U p : Fin n → ℝ),
        (∀ j, 0 ≤ p j) → ∑ j, p j = 1 →
        FreeEnergy.freeEnergy beta U p - FreeEnergy.freeEnergy beta U (FreeEnergy.boltz beta U)
            = klDiv p (FreeEnergy.boltz beta U) / beta ∧
          (FreeEnergy.freeEnergy beta U p
              ≤ FreeEnergy.freeEnergy beta U (FreeEnergy.boltz beta U) →
            p = FreeEnergy.boltz beta U)) ∧
    -- (2) a dominant conformation requires an energy gap paying for the entropy
    (∀ (n : ℕ) (beta : ℝ), 0 < beta → 0 < n → ∀ (U : Fin n → ℝ) (j0 : Fin n)
        (D : Finset (Fin n)), D.Nonempty → j0 ∉ D → ∀ Uu : ℝ, (∀ j ∈ D, U j ≤ Uu) →
        1 / 2 ≤ FreeEnergy.boltz beta U j0 → Real.log D.card / beta ≤ Uu - U j0) ∧
    -- (3) the geometric rate--distortion law
    (∀ (X : Type) (m k : ℕ) (hm : 0 < m) (c : X → X → ℝ), StructDist c →
        ∀ (g : Fin m → X) (Delta Dist : ℝ), 0 < Delta → Separated c g Delta →
        ∀ M : Ens X, M.card ≤ k →
        transportCost c M (unif hm g) ≤ Dist →
        (m : ℝ) * (1 - 2 * Dist / Delta) ≤ k) ∧
    -- (4) the ideal chain: the numbers in `N`
    (∀ (N : ℕ) (b r : ℝ),
        (N : ℝ) * b ^ 2 ≤ (chainEns N).expect (fun s => (endToEnd b s - r) ^ 2)) ∧
    (∀ (N : ℕ) (M : Ens (Chain N)), M.Same (chainEns N) → 2 ^ N ≤ M.card) ∧
    -- (5) experiments contract, and leave a ball of ensembles fitting equally well
    (∀ (n p : ℕ) (K : Fin n → Fin p → ℝ), Channel.IsChannel K → ∀ u v : Fin n → ℝ,
        (∀ j, 0 ≤ u j) → (∀ j, 0 < v j) →
        klDiv (Channel.push K u) (Channel.push K v) ≤ klDiv u v) ∧
    (∀ (n p : ℕ) (K : Fin n → Fin p → ℝ), Channel.IsChannel K → ∀ (alpha eps : ℝ)
        (nu : Fin p → ℝ), ∑ y, nu y = 1 → alpha < 1 → (∀ j y, alpha * nu y ≤ K j y) →
        ∀ u v : Fin n → ℝ, ∑ j, u j = ∑ j, v j → ∑ j, |u j - v j| ≤ eps / (1 - alpha) →
        ∑ y, |Channel.push K u y - Channel.push K v y| ≤ eps) ∧
    -- (6) inter-segment coupling costs exactly one latent state per coupled substate
    (∀ (X Y : Type) (m : ℕ) (hm : 0 < m) (xs : Fin m → X) (ys : Fin m → Y),
        Function.Injective xs → Function.Injective ys →
        MixtureOfProducts m (diagEns hm xs ys) ∧
          ∀ k : ℕ, MixtureOfProducts k (diagEns hm xs ys) → m ≤ k) := by
  refine ⟨?_, ?_, ?_, fun N b r => chain_single_structure_floor b r,
    fun _ _ h => chain_capacity h, ?_, ?_, ?_⟩
  · intro n hn beta hbeta U p hp hps
    exact ⟨FreeEnergy.freeEnergy_gap hn hbeta hp hps,
      fun hmin => FreeEnergy.freeEnergy_min_unique hn hbeta hp hps hmin⟩
  · intro n beta hbeta hn U j0 D hD hj0 Uu hU hhalf
    exact FreeEnergy.folded_needs_entropic_gap hbeta hn U j0 D hD hj0 Uu hU hhalf
  · intro X m k hm c hc g Delta Dist hDelta hsep M hM hD
    exact quantization_capacity hm hc hDelta hsep hM hD
  · intro n p K hK u v hu hv
    exact Channel.dataProcessing hK hu hv
  · intro n p K hK alpha eps nu hnu halpha hmin u v hsum hclose
    exact Channel.indistinguishable_radius hK hnu halpha hmin hsum hclose
  · intro X Y m hm xs ys hx hy
    exact latent_rank_theorem hm hx hy

end IDR
