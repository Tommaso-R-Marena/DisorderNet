/-
# Part LXV  Replica exchange: unbiased, and not a cure

Every reported conformational ensemble of a disordered region comes out of an enhanced-sampling
run, and in practice that means replica exchange.  `RequestProject.ReplicaExchange` treats the
two-replica extended system exactly, on a finite conformation space, and settles four questions
about it.

`IDR.replica_exchange_laws` bundles five statements:

1. *It is unbiased.*  The swap move is reversible with respect to the product of the two
   Boltzmann ensembles, hence leaves it stationary, and the marginal at each temperature is
   exactly the Boltzmann ensemble at that temperature.  Together with the within-temperature
   moves (`reK_stationary`) the whole chain preserves the extended ensemble.  Nothing here is
   approximate: the correction to the reported populations from swapping is exactly zero.
2. *The acceptance rate is an overlap and nothing else.*  The mean acceptance is exactly
   `1 - TV(pi, pi ∘ swap)`, the overlap between the extended weight and its swap; and the swap
   acceptance is `min 1 exp((b2-b1)(E j - E i))`, in which no partition function appears.
3. *A temperature step costs exponentially in the energy gap.*  If the two energy histograms are
   separated by `g`, the mean acceptance is at most `exp(-(b2-b1) g)` -- attained exactly in a
   two-state instance -- so holding acceptance at `al` forces `(b2-b1) g <= log(1/al)`, and a
   ladder spanning `[b 0, b K]` needs `K >= (b K - b 0) g / log(1/al)` rungs.  The energy gap
   between two temperatures is extensive in the number of residues; the replica count therefore
   grows with the size of the disordered region, which is why tempering a large IDR is expensive.
4. *And a high acceptance rate is no evidence of good sampling.*  At equal temperatures every
   swap is accepted, while the two replicas sample the same ensemble and the exchange conveys
   nothing.  Acceptance measures overlap, and overlap is necessary, not sufficient.
5. *Tempering cannot repair broken ergodicity.*  If the within-temperature move set never
   crosses between a set `A` of conformations and its complement -- at any temperature -- then
   the number of replicas inside `A` is a conserved quantity of the whole extended chain, so a
   run started with every replica inside `A` never visits the complement, for any number of
   sweeps, although the Boltzmann ensemble puts positive weight there.  The swap move permutes
   the configurations currently held; it cannot manufacture one.  Raising the temperature lowers
   barriers, but a barrier the move set cannot cross at any temperature -- a topological trap, a
   knot, a chirality, a broken covalent constraint -- is not a barrier at all, and no amount of
   exchange will cross it.

Read with Part XXV (broken ergodicity) and `RequestProject.Mixing` (the exponential cost of a
barrier), this closes the loop on what a simulation protocol can promise: unbiasedness is free,
efficiency is not, and connectivity of the move set is an assumption, never a consequence.
-/
import Mathlib
import RequestProject.ReplicaExchange

set_option autoImplicit false

namespace IDR

open IDR.RE

/-- **The replica-exchange laws.**

1. the swap move is reversible with respect to the extended (product) Boltzmann ensemble, hence
   stationary for it, with the correct marginal at each temperature;
2. the mean acceptance rate is exactly `1 - TV(pi, pi ∘ swap)`, and the acceptance probability
   is a function of the energies and temperatures alone;
3. separated energy histograms make the acceptance at most `exp(-(b2-b1) g)`, an equality in an
   explicit two-state instance, so a ladder holding acceptance `al` needs
   `(b K - b 0) g / log(1/al)` rungs;
4. equal temperatures give acceptance `1` while the two replicas sample the same ensemble;
5. and if no within-temperature move crosses the boundary of `A`, a run started inside `A` never
   leaves it, at any temperature and after any number of sweeps. -/
theorem replica_exchange_laws :
    (∀ (n : ℕ) (p1 p2 : Fin n → ℝ), (∀ i, 0 < p1 i) → (∀ j, 0 < p2 j) →
        (∑ i, p1 i = 1) → (∑ j, p2 j = 1) →
        (∀ s t, prodW p1 p2 s * swapK p1 p2 s t = prodW p1 p2 t * swapK p1 p2 t s) ∧
        (∀ t, ∑ s, prodW p1 p2 s * swapK p1 p2 s t = prodW p1 p2 t) ∧
        (∀ i, ∑ j, prodW p1 p2 (i, j) = p1 i) ∧ (∀ j, ∑ i, prodW p1 p2 (i, j) = p2 j)) ∧
    (∀ (n : ℕ) (p1 p2 : Fin n → ℝ), (∀ i, 0 < p1 i) → (∀ j, 0 < p2 j) →
        (∑ i, p1 i = 1) → (∑ j, p2 j = 1) →
        meanAcc p1 p2 = 1 - (1/2) * ∑ s, |prodW p1 p2 s - prodW p1 p2 (Prod.swap s)|) ∧
    (∀ (n : ℕ) (_ : NeZero n) (b1 b2 : ℝ) (E : Fin n → ℝ) (s : Fin n × Fin n),
        accSwap (boltzW b1 E) (boltzW b2 E) s
          = min 1 (Real.exp ((b2 - b1) * (E s.2 - E s.1)))) ∧
    ((∀ (n : ℕ) (p1 p2 E : Fin n → ℝ) (b1 b2 a g : ℝ), b1 ≤ b2 →
        (∀ i, 0 ≤ p1 i) → (∀ j, 0 ≤ p2 j) → (∑ i, p1 i = 1) → (∑ j, p2 j = 1) →
        (∀ i, p1 i ≠ 0 → a ≤ E i) → (∀ j, p2 j ≠ 0 → E j ≤ a - g) →
        ∑ s, prodW p1 p2 s * accE b1 b2 E s ≤ Real.exp (-((b2 - b1) * g))) ∧
      (∀ b1 b2 g : ℝ, b1 ≤ b2 → 0 ≤ g →
        ∑ s, prodW (![0, 1] : Fin 2 → ℝ) (![1, 0] : Fin 2 → ℝ) s
            * accE b1 b2 (![0, g] : Fin 2 → ℝ) s = Real.exp (-((b2 - b1) * g))) ∧
      (∀ (b : ℕ → ℝ) (K : ℕ) (g al : ℝ), 0 < g → 0 < al → al < 1 →
        (∀ k, k < K → (b (k + 1) - b k) * g ≤ Real.log (1 / al)) →
        (b K - b 0) * g / Real.log (1 / al) ≤ K)) ∧
    (∀ (n : ℕ) (_ : NeZero n) (b : ℝ) (E : Fin n → ℝ), meanAcc (boltzW b E) (boltzW b E) = 1) ∧
    (∀ (n : ℕ) (_ : NeZero n) (A : Finset (Fin n)) (lam : ℝ) (P1 P2 : Fin n → Fin n → ℝ)
        (b1 b2 : ℝ) (E : Fin n → ℝ) (w : (Fin n × Fin n) → ℝ),
        Blocked A P1 → Blocked A P2 → (∀ s, w s ≠ 0 → (s.1 ∈ A ∧ s.2 ∈ A)) →
        ∀ (m : ℕ) (t : Fin n × Fin n), t.1 ∉ A →
          iter (reK lam P1 P2 (boltzW b1 E) (boltzW b2 E)) m w t = 0) := by
  refine ⟨fun n p1 p2 h1 h2 hs1 hs2 =>
      ⟨fun s t => swapK_detailedBalance h1 h2 s t, fun t => swapK_stationary h1 h2 t,
        fun i => prodW_marginal_fst p1 p2 hs2 i, fun j => prodW_marginal_snd p1 p2 hs1 j⟩,
    fun n p1 p2 h1 h2 hs1 hs2 => meanAcc_eq_one_sub_tv h1 h2 hs1 hs2,
    fun n _ b1 b2 E s => accSwap_boltz b1 b2 E s,
    ⟨fun n p1 p2 E b1 b2 a g hb hp1 hp2 hs1 hs2 hE1 hE2 =>
        meanAcc_le_exp_neg_gap hb hp1 hp2 hs1 hs2 hE1 hE2,
      fun b1 b2 g hb hg => meanAcc_two_state hb hg,
      fun b K g al hg hal hal1 hd => replicas_needed b K hg hal hal1 hd⟩,
    fun n _ b E => meanAcc_flat_eq_one b E,
    fun n _ A lam P1 P2 b1 b2 E w hA1 hA2 hw m t ht =>
      (replica_exchange_cannot_repair_ergodicity (lam := lam) (b1 := b1) (b2 := b2) (E := E)
        hA1 hA2 hw m).1 t ht⟩

end IDR
