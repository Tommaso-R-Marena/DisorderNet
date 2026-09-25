/-
# Part LXVIII  The full ladder

Part LXV treats one exchanging pair.  A production run of a disordered region is a ladder of `K`
replicas, and the natural question is whether the two conclusions of Part LXV -- exactness of the
sampled distribution, and the impossibility of repairing a disconnected move set -- are artefacts
of the two-replica case.  They are not.  `RequestProject.ReplicaLadder` proves both for `K`
replicas, through a general lemma that isolates what makes them true: a replica exchange is a
Metropolis move *along an involution* of the extended state space, and every such move is
reversible with respect to any positive weight.

`IDR.replica_ladder_laws` bundles four statements:

1. *The general mechanism.*  For any finite state space, any positive weight and any involution
   `s`, the Metropolis move `x ↦ s x` is a transition kernel, is reversible with respect to the
   weight, leaves it stationary, and accepts at a rate exactly equal to the overlap of the weight
   with its image under `s`.  Exchanging the configurations of two replicas is such a move; so is
   any parallel exchange of several disjoint pairs.
2. *The acceptance collapses to the pair.*  On the ladder the acceptance probability is
   `min 1 exp((b c − b a)(E (x c) − E (x a)))`: no partition function, and no dependence on the
   other `K − 2` replicas.
3. *The ladder is unbiased.*  In the extended ensemble the average of any observable of the
   configuration held by replica `k` is exactly its Boltzmann average at replica `k`'s own
   temperature, and every exchange preserves the extended ensemble.
4. *And the conserved count survives the whole ladder.*  If no within-temperature move crosses
   between a set `A` of conformations and its complement, then the number of replicas holding a
   configuration in `A` is unchanged by every exchange and every within-temperature update, so a
   run started with all `K` replicas inside `A` never leaves `A`, at any temperature and after
   any number of sweeps.

The last statement is the one worth acting on.  Adding rungs to the ladder, or widening the
temperature range, cannot help: the obstruction is a conserved quantity of the move set, and the
temperatures do not enter it.  Before a tempering run is trusted to have sampled a disordered
region, what has to be argued is that the move set connects the conformations -- and that is a
statement about the moves, not about the thermostat.
-/
import Mathlib
import RequestProject.ReplicaLadder

set_option autoImplicit false

namespace IDR

open IDR.Ladder

/-- **The replica-ladder laws.**

1. the Metropolis move along any involution of any finite state space is a transition kernel,
   reversible for any positive weight, stationary for it, and accepting at exactly the overlap
   rate;
2. on a `K`-replica ladder the exchange acceptance is `min 1 exp((b c − b a)(E (x c) − E (x a)))`;
3. the extended ensemble reproduces each replica's own Boltzmann average exactly, and the
   exchange preserves it;
4. and if no within-temperature move crosses the boundary of `A`, a run started with every
   replica inside `A` never leaves it. -/
theorem replica_ladder_laws :
    (∀ (S : Type) (_ : Fintype S) (_ : DecidableEq S) (pi : S → ℝ), (∀ x, 0 < pi x) →
        ∀ (s : S → S), Function.Involutive s →
          (∀ x, ∑ y, invK pi s x y = 1) ∧
          (∀ x y, pi x * invK pi s x y = pi y * invK pi s y x) ∧
          (∀ y, ∑ x, pi x * invK pi s x y = pi y) ∧
          (∑ x, pi x * invAcc pi s x = ∑ x, min (pi x) (pi (s x)))) ∧
    (∀ (K n : ℕ) (_ : NeZero n) (b : Fin K → ℝ) (E : Fin n → ℝ) (a c : Fin K), a ≠ c →
        ∀ x : Fin K → Fin n,
          invAcc (ladderW (fun k => RE.boltzW (b k) E)) (swapPair a c) x
            = min 1 (Real.exp ((b c - b a) * (E (x c) - E (x a))))) ∧
    (∀ (K n : ℕ) (p : Fin K → Fin n → ℝ), (∀ k, ∑ i, p k i = 1) →
        (∀ (k0 : Fin K) (f : Fin n → ℝ), ∑ x, ladderW p x * f (x k0) = ∑ i, p k0 i * f i) ∧
        ((∀ k i, 0 < p k i) → ∀ (a c : Fin K) (y : Fin K → Fin n),
          ∑ x, ladderW p x * invK (ladderW p) (swapPair a c) x y = ladderW p y)) ∧
    (∀ (K n : ℕ) (A : Finset (Fin n)) (lam : ℝ) (P : Fin K → Fin n → Fin n → ℝ)
        (p : Fin K → Fin n → ℝ) (a c : Fin K),
        (∀ k, RE.Blocked A (P k)) →
        ∀ w : (Fin K → Fin n) → ℝ, (∀ x, w x ≠ 0 → ∀ k, x k ∈ A) →
          ∀ (m : ℕ) (y : Fin K → Fin n) (k : Fin K), y k ∉ A →
            iter (ladderK lam P p a c) m w y = 0) := by
  refine ⟨fun S _ _ pi hpi s hs =>
      ⟨fun x => invK_stochastic hpi s x, fun x y => invK_detailedBalance hpi hs x y,
        fun y => invK_stationary hpi hs y, invAcc_mean_eq_overlap hpi s⟩,
    fun K n _ b E a c hac x => ladder_acc_boltz b E hac x,
    fun K n p hp1 => ⟨fun k0 f => ladder_unbiased hp1 k0 f,
      fun hp a c y => ladder_swap_stationary hp a c y⟩,
    fun K n A lam P p a c hP w hw m y k hk =>
      ladder_cannot_repair_ergodicity hP a c hw m y k hk⟩

end IDR
