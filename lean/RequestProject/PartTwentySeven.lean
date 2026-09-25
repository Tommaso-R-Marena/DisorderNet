/-
# Part XXVII  Sampling: what it takes to turn an energy function into conformations

Part XXVI leaves the model with a dilemma: the ensemble it denotes lives on exponentially
many conformations, and only a factorized model has a computable configuration sum -- while
the factorizations that are cheap are exactly the ones that cannot carry a long-range contact.
This part treats the escape route that real programs take: do not compute the normalization at
all, and generate conformations with a Markov chain.

* `RequestProject.Metropolis` -- the Metropolis kernel of a target under a symmetric proposal
  is a transition kernel, is reversible with respect to the target, and therefore leaves it
  stationary; and it depends on the target only through *ratios*, so the kernel built from the
  normalized Boltzmann populations is literally the same kernel as the one built from the bare
  weights `exp(-beta E)`.  The intractable sum of Part XXVI never appears, and the acceptance
  probability is a function of the energy difference alone.
* `RequestProject.Mixing` -- what that does not buy is time.  The bottleneck lemma bounds the
  population that can leave a region of conformation space per step, so the escape time is at
  least the reciprocal of the escape probability; on a two-well landscape with a barrier of
  height `B` the Metropolis chain needs at least `e^{beta B}` steps to equilibrate the wells.

`IDR.sampling_laws` bundles the four statements.  Together with Part XXVI they complete the
computational half of the specification: a model of a disordered region must come with a
sampler, the sampler needs only unnormalized weights, and the ensemble it actually produces in
a finite run is the one the model may claim -- not the Boltzmann ensemble of its energy
function, unless the run is long compared with the exponential of its barriers.
-/
import Mathlib
import RequestProject.Metropolis
import RequestProject.Mixing

namespace IDR

open Metropolis Mixing

/-- **The design laws of sampling.**

1. *The Metropolis chain samples the model's ensemble.*  With a symmetric proposal the kernel
   is stochastic, reversible with respect to the target, and leaves it stationary.
2. *The partition function never appears.*  Rescaling the target leaves the kernel unchanged,
   so the kernel of a Boltzmann ensemble is the kernel of its unnormalized weights, and the
   acceptance probability depends only on the energy difference.
3. *Bottlenecks bound the sampler.*  If the chain leaves a region with probability at most
   `eps` per step, then after `t` steps started inside it the population outside is at most
   `t · eps`, so reaching a population `m` outside takes at least `m / eps` steps.
4. *A barrier costs exponentially.*  On the two-well landscape the escape probability is
   `e^{-beta B}/2`, and equilibrating the wells takes at least `e^{beta B}` steps. -/
theorem sampling_laws :
    -- 1  the Metropolis chain is a sampler for its target
    (∀ (n : ℕ) (p : Fin n → ℝ) (q : Fin n → Fin n → ℝ), (∀ i, 0 < p i) →
      (∀ i j, 0 ≤ q i j) → (∀ i, ∑ j, q i j = 1) → (∀ i j, q i j = q j i) →
      Kinetics.IsStochastic (mhK p q) ∧ Kinetics.DetailedBalance (mhK p q) p ∧
        Kinetics.Stationary (mhK p q) p) ∧
    -- 2  only ratios enter: no partition function
    ((∀ (n : ℕ) (p : Fin n → ℝ) (q : Fin n → Fin n → ℝ) (c : ℝ), 0 < c →
        mhK (fun l => c * p l) q = mhK p q) ∧
      (∀ (n : ℕ) (beta : ℝ) (E : Fin n → ℝ) (q : Fin n → Fin n → ℝ) (Zc : ℝ), 0 < Zc →
        mhK (fun i => Real.exp (-beta * E i) / Zc) q
          = mhK (fun i => Real.exp (-beta * E i)) q) ∧
      (∀ (n : ℕ) (beta : ℝ) (E : Fin n → ℝ) (i j : Fin n),
        acc (fun l => Real.exp (-beta * E l)) i j
          = min 1 (Real.exp (-beta * (E j - E i))))) ∧
    -- 3  a bottleneck bounds how fast the sampler can move
    (∀ (n : ℕ) (P : Fin n → Fin n → ℝ) (w : Fin n → ℝ) (S : Finset (Fin n)) (eps : ℝ),
      Kinetics.IsStochastic P → (∀ i, 0 ≤ w i) → (∑ i, w i = 1) →
      (∀ i ∈ S, ∑ j ∈ Finset.univ \ S, P i j ≤ eps) → 0 ≤ eps →
      ∀ t : ℕ, massOut S (iterate P t w) ≤ massOut S w + t * eps) ∧
    -- 4  crossing a barrier of height B takes at least exp (beta B) steps
    (∀ beta B : ℝ, 0 ≤ beta → 0 ≤ B →
      (∀ i ∈ ({0} : Finset (Fin 3)),
          ∑ j ∈ Finset.univ \ ({0} : Finset (Fin 3)), barrierK beta B i j
            ≤ Real.exp (-beta * B) / 2) ∧
        ∀ t : ℕ, (1:ℝ)/2 ≤ massOut {0}
            (iterate (barrierK beta B) t (fun i => if i = 0 then 1 else 0)) →
          Real.exp (beta * B) ≤ t) :=
  ⟨fun _ _ _ hp hq0 hq1 hq =>
      ⟨mhK_stochastic hp hq0 hq1, mhK_detailedBalance hp hq, mhK_stationary hp hq0 hq1 hq⟩,
    ⟨fun _ p q _ hc => mhK_smul p q hc,
      fun _ beta E q _ hZ => mhK_of_unnormalized beta E q hZ,
      fun _ beta E i j => acc_boltzmann beta E i j⟩,
    fun _ _ _ _ _ hP hw hw1 heps heps0 t => massOut_iterate_le hP hw hw1 heps heps0 t,
    fun _ _ hbeta hB =>
      ⟨barrier_escape_le hbeta hB, fun _ hhalf => barrier_steps_needed hbeta hB hhalf⟩⟩

end IDR
