/-
# Part XXVI  Tractability: what makes an ensemble model usable, and what that costs

Parts I–XXV say what a model of a disordered region must *denote*: a context-conditional
distribution over conformations, with enough capacity to carry the populated states, judged
by a strictly proper score and reporting a calibrated, informative uncertainty.  None of that
says the distribution can be evaluated.  This part supplies the missing requirement and
prices it.

* `RequestProject.TransferMatrix` -- the configuration space of a residue-level chain model is
  exponential (`k^(n+1)` conformations), so populations are ratios of exponentially long sums.
  A nearest-neighbour factorization makes those sums exactly computable -- the transfer-matrix
  theorem, `n` matrix multiplications -- and the normalized measure is a genuine ensemble.
  But locality is a real restriction: a three-residue ensemble with a long-range contact
  between its ends has *no* nearest-neighbour factorization, for any transfer matrix and any
  boundary conditions.
* `RequestProject.Tensorization` -- the standard escape route, sampling from a reference model
  and reweighting onto the target, is priced exactly: for product models the reweighting cost
  tensorizes, so a per-residue mismatch `c > 0` leaves an effective sample size fraction of
  exactly `(1 + c)^(-n)` and demands `(1 + c)^n` frames.  Relative entropy is additive over
  sites; the chi-squared cost is multiplicative.

`IDR.tractability_laws` bundles the four statements.  Together they turn the qualitative
design advice of the earlier parts into a constraint on model *architecture*: the model must
be factorized enough to be evaluated and sampled, and the factorization itself must already
contain the long-range structure the model is meant to predict -- because neither a local
factorization nor a reweighting of a mis-specified reference can supply it afterwards.
-/
import Mathlib
import RequestProject.TransferMatrix
import RequestProject.Tensorization

namespace IDR

open Transfer Tensor

/-- **The design laws of tractability.**

1. *Exponential space, polynomial evaluation.*  A chain of `n+1` residues with `k` local
   states has `k^(n+1)` conformations, and yet for any nearest-neighbour weight the whole
   configuration sum equals `v · Mⁿ · u`: the model is evaluable in a number of arithmetic
   operations linear in the length of the region.  With strictly positive local weights the
   normalized nearest-neighbour Gibbs measure is a genuine ensemble: strictly positive on
   every conformation and summing to one.
2. *Locality is a restriction, not a convenience.*  In a nearest-neighbour three-residue model
   the two ends are conditionally independent given the middle residue; the explicit contact
   ensemble violates that identity, is nevertheless a bona fide probability distribution with
   correlated ends, and therefore differs from **every** nearest-neighbour Gibbs model at some
   conformation.
3. *Reweighting cost tensorizes.*  For product references the chi-squared cost multiplies over
   sites, so `n` residues with the same per-site mismatch `c` give `1 + χ² = (1 + c)^n` and an
   effective sample size fraction of exactly `(1 + c)^(-n)`: retaining `neff` effective frames
   requires `neff · (1 + c)^n` frames of simulation.
4. *And relative entropy adds.*  The same statement from the entropy side: a per-residue
   mismatch of `K` nats is a chain-level mismatch of `n · K` nats. -/
theorem tractability_laws :
    -- 1  exponentially many conformations, computable configuration sum, genuine ensemble
    ((∀ (k n : ℕ), Fintype.card (Fin (n + 1) → Fin k) = k ^ (n + 1)) ∧
      (∀ (k n : ℕ) (v u : Fin k → ℝ) (M : Matrix (Fin k) (Fin k) ℝ),
        Z (n := n) v M u = ∑ b, (Matrix.vecMul v (M ^ n)) b * u b) ∧
      (∀ (k n : ℕ) (_ : NeZero k) (v u : Fin k → ℝ) (M : Matrix (Fin k) (Fin k) ℝ),
        (∀ a, 0 < v a) → (∀ a b, 0 < M a b) → (∀ a, 0 < u a) →
        (∑ x : Fin (n + 1) → Fin k, chainProb v M u x = 1) ∧
          ∀ x : Fin (n + 1) → Fin k, 0 < chainProb v M u x)) ∧
    -- 2  a long-range contact has no nearest-neighbour factorization
    ((∀ (P : (Fin 3 → Fin 2) → ℝ), PairFactored P → ∀ a a' b c c' : Fin 2,
        P ![a, b, c] * P ![a', b, c'] = P ![a, b, c'] * P ![a', b, c]) ∧
      (∑ x : Fin 3 → Fin 2, contactDist x = 1) ∧
      ¬ PairFactored contactDist ∧
      (∀ (v u : Fin 2 → ℝ) (M : Matrix (Fin 2) (Fin 2) ℝ),
        ∃ x : Fin 3 → Fin 2, chainProb (n := 2) v M u x ≠ contactDist x)) ∧
    -- 3  the reweighting cost is exponential in the length of the region
    (∀ (k n : ℕ) (p₀ q₀ : Fin k → ℝ), (∀ s, 0 < q₀ s) → ∑ s, p₀ s = 1 → ∑ s, q₀ s = 1 →
      essFracT (prodDist (fun _ : Fin n => p₀)) (prodDist (fun _ : Fin n => q₀))
          = 1 / (1 + chiSqT p₀ q₀) ^ n ∧
        ∀ N neff : ℝ,
          neff ≤ N * essFracT (prodDist (fun _ : Fin n => p₀))
              (prodDist (fun _ : Fin n => q₀)) →
            neff * (1 + chiSqT p₀ q₀) ^ n ≤ N) ∧
    -- 4  relative entropy is additive over residues
    (∀ (k n : ℕ) (p q : Fin n → Fin k → ℝ), (∀ i s, 0 < p i s) → (∀ i s, 0 < q i s) →
      (∀ i, ∑ s, p i s = 1) →
      klT (prodDist p) (prodDist q) = ∑ i, klT (p i) (q i)) :=
  ⟨⟨fun _ n => card_chain_states n,
      fun _ n v u M => Z_eq_vecMul_pow M u n v,
      fun _ _ _ _ _ _ hv hM hu =>
        ⟨chainProb_sum_one hv hM hu, fun x => chainProb_pos hv hM hu x⟩⟩,
    ⟨fun _ hP a a' b c c' => pairFactored_cross hP a a' b c c',
      contactDist_sum_one,
      contactDist_not_pairFactored,
      fun v u M => chainProb_ne_contactDist v u M⟩,
    fun _ n _ _ hq hps hqs =>
      ⟨essFrac_prodDist_iid hq hps hqs n,
        fun _ _ h => frames_needed_prodDist hq hps hqs n h⟩,
    fun _ _ _ _ hp hq hps => kl_prodDist hp hq hps⟩

end IDR
