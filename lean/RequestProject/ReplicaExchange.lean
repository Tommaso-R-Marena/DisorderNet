/-
# Part LXV  Replica exchange: what tempering buys, and what it cannot buy

Every serious simulation of a disordered region is run with an enhanced-sampling protocol, and
the protocol is almost always replica exchange (parallel tempering): several copies of the
system are propagated at different temperatures and neighbouring copies periodically attempt to
swap configurations.  The claims made for it are (i) that it is *unbiased* -- each temperature
still samples its own Boltzmann ensemble -- and (ii) that it *fixes* the sampling problem that
motivated it.  This file proves the first exactly, and shows the second is false as stated: the
swap move is a similarity transformation on the set of configurations currently held, so it can
redistribute configurations among temperatures but can never manufacture one.

The state of the extended system is a pair `(i, j)` -- the configuration of the hot replica and
the configuration of the cold replica -- with extended weight `prodW p1 p2 (i,j) = p1 i * p2 j`.

* `swapK_detailedBalance`, `swapK_stochastic`, `swapK_stationary`, `prodW_marginal_fst/snd` --
  the swap move is reversible with respect to the product of the two Boltzmann ensembles, so
  the marginal at each temperature is exactly the Boltzmann ensemble at that temperature.
* `accSwap_boltz` -- the acceptance probability is `min 1 exp((b2-b1)(E j - E i))`: no partition
  function appears, which is why the move is implementable.
* `meanAcc_eq_overlap`, `meanAcc_eq_one_sub_tv` -- the mean acceptance rate is *exactly* the
  overlap of the extended weight with its swap, `1 - TV(pi, pi∘swap)`.  Acceptance is a measure
  of histogram overlap and of nothing else.
* `meanAcc_le_exp_neg_gap`, `meanAcc_two_state`, `deltaBeta_le_of_meanAcc`, `replicas_needed` --
  if the two energy histograms are separated by a gap `g`, the acceptance is at most
  `exp(-(b2-b1) g)`, an equality in an explicit instance; hence holding a target acceptance `al`
  forces `(b2-b1) g <= log (1/al)`, and a ladder spanning `[bmin, bmax]` needs at least
  `(bmax-bmin) * g / log(1/al)` rungs.  Since the energy gap between two temperatures is
  extensive, the replica count grows with system size.
* `meanAcc_flat_eq_one` -- and a high acceptance rate is no evidence of anything: at equal
  temperatures every swap is accepted while the two ensembles are identical.
* `reK_conserves_count`, `re_iterate_support`, `replica_exchange_cannot_repair_ergodicity` --
  the negative result.  If the within-temperature move set never crosses between `A` and its
  complement (at *any* temperature), then the number of replicas inside `A` is conserved by the
  whole extended chain, so a run started with every replica in `A` never leaves `A` -- although
  the true Boltzmann ensemble puts positive weight outside.  Tempering lowers barriers; it does
  not connect what the move set disconnects.
-/
import Mathlib
import RequestProject.Kinetics
import RequestProject.Metropolis

set_option autoImplicit false

namespace IDR

namespace RE

open Finset

variable {n : ℕ}

/-! ## The extended ensemble -/

/-- The weight of a two-replica configuration in the extended ensemble. -/
def prodW (p1 p2 : Fin n → ℝ) (s : Fin n × Fin n) : ℝ := p1 s.1 * p2 s.2

lemma prodW_nonneg {p1 p2 : Fin n → ℝ} (h1 : ∀ i, 0 ≤ p1 i) (h2 : ∀ j, 0 ≤ p2 j)
    (s : Fin n × Fin n) : 0 ≤ prodW p1 p2 s :=
  mul_nonneg (h1 _) (h2 _)

lemma prodW_pos {p1 p2 : Fin n → ℝ} (h1 : ∀ i, 0 < p1 i) (h2 : ∀ j, 0 < p2 j)
    (s : Fin n × Fin n) : 0 < prodW p1 p2 s :=
  mul_pos (h1 _) (h2 _)

lemma prodW_sum {p1 p2 : Fin n → ℝ} (h1 : ∑ i, p1 i = 1) (h2 : ∑ j, p2 j = 1) :
    ∑ s, prodW p1 p2 s = 1 := by
  rw [Fintype.sum_prod_type]
  simp only [prodW]
  rw [← Finset.sum_mul_sum, h1, h2, one_mul]

/-- The hot-replica marginal of the extended ensemble is the hot Boltzmann ensemble. -/
lemma prodW_marginal_fst (p1 p2 : Fin n → ℝ) (h2 : ∑ j, p2 j = 1) (i : Fin n) :
    ∑ j, prodW p1 p2 (i, j) = p1 i := by
  simp only [prodW]
  rw [← Finset.mul_sum, h2, mul_one]

/-- The cold-replica marginal of the extended ensemble is the cold Boltzmann ensemble. -/
lemma prodW_marginal_snd (p1 p2 : Fin n → ℝ) (h1 : ∑ i, p1 i = 1) (j : Fin n) :
    ∑ i, prodW p1 p2 (i, j) = p2 j := by
  simp only [prodW]
  rw [← Finset.sum_mul, h1, one_mul]

lemma sum_swap (f : Fin n × Fin n → ℝ) : ∑ s, f (Prod.swap s) = ∑ s, f s :=
  Fintype.sum_equiv (Equiv.prodComm (Fin n) (Fin n)) _ _ (fun _ => rfl)

/-! ## The swap move -/

/-- The Metropolis acceptance probability of a configuration swap. -/
noncomputable def accSwap (p1 p2 : Fin n → ℝ) (s : Fin n × Fin n) : ℝ :=
  min 1 (prodW p1 p2 (Prod.swap s) / prodW p1 p2 s)

lemma accSwap_le_one (p1 p2 : Fin n → ℝ) (s : Fin n × Fin n) : accSwap p1 p2 s ≤ 1 :=
  min_le_left _ _

lemma accSwap_nonneg {p1 p2 : Fin n → ℝ} (h1 : ∀ i, 0 < p1 i) (h2 : ∀ j, 0 < p2 j)
    (s : Fin n × Fin n) : 0 ≤ accSwap p1 p2 s :=
  le_min zero_le_one (div_nonneg (prodW_pos h1 h2 _).le (prodW_pos h1 h2 _).le)

/-- **The swap kernel**: attempt to exchange the configurations of the two replicas, accepting
with the Metropolis probability and otherwise leaving the pair alone. -/
noncomputable def swapK (p1 p2 : Fin n → ℝ) (s t : Fin n × Fin n) : ℝ :=
  if t = Prod.swap s then accSwap p1 p2 s
  else if t = s then 1 - accSwap p1 p2 s else 0

lemma swapK_nonneg {p1 p2 : Fin n → ℝ} (h1 : ∀ i, 0 < p1 i) (h2 : ∀ j, 0 < p2 j)
    (s t : Fin n × Fin n) : 0 ≤ swapK p1 p2 s t := by
  unfold swapK
  split
  · exact accSwap_nonneg h1 h2 s
  · split
    · linarith [accSwap_le_one p1 p2 s]
    · exact le_rfl

/-- The swap kernel is a transition kernel. -/
theorem swapK_stochastic {p1 p2 : Fin n → ℝ} (h1 : ∀ i, 0 < p1 i) (h2 : ∀ j, 0 < p2 j)
    (s : Fin n × Fin n) :
    ∑ t, swapK p1 p2 s t = 1 := by
  by_cases hs : Prod.swap s = s
  · have : ∀ t : Fin n × Fin n, swapK p1 p2 s t = if t = s then accSwap p1 p2 s else 0 := by
      intro t
      unfold swapK
      rw [hs]
      split_ifs <;> rfl
    rw [Finset.sum_congr rfl (fun t _ => this t)]
    rw [Finset.sum_ite_eq' Finset.univ s (fun _ => accSwap p1 p2 s)]
    simp only [Finset.mem_univ, if_true]
    have : prodW p1 p2 (Prod.swap s) / prodW p1 p2 s = 1 ∨ prodW p1 p2 s = 0 := by
      rcases eq_or_ne (prodW p1 p2 s) 0 with h | h
      · exact Or.inr h
      · exact Or.inl (by rw [hs]; exact div_self h)
    unfold accSwap
    rcases this with h | h
    · rw [h, min_self]
    · exact absurd h (prodW_pos h1 h2 s).ne'
  · rw [Finset.sum_eq_add_of_mem (Prod.swap s) s (Finset.mem_univ _) (Finset.mem_univ _) hs]
    · unfold swapK
      rw [if_pos rfl, if_neg (Ne.symm hs), if_pos rfl]
      ring
    · intro c _ hc
      unfold swapK
      rw [if_neg hc.1, if_neg hc.2]

/-- **Detailed balance for the swap move.**  The extended ensemble is reversible under it. -/
theorem swapK_detailedBalance {p1 p2 : Fin n → ℝ} (h1 : ∀ i, 0 < p1 i) (h2 : ∀ j, 0 < p2 j)
    (s t : Fin n × Fin n) :
    prodW p1 p2 s * swapK p1 p2 s t = prodW p1 p2 t * swapK p1 p2 t s := by
  by_cases hts : t = Prod.swap s
  · subst hts
    have hst : s = Prod.swap (Prod.swap s) := by simp
    unfold swapK
    rw [if_pos rfl, if_pos hst.symm.symm]
    · unfold accSwap
      have e1 : prodW p1 p2 s * min 1 (prodW p1 p2 (Prod.swap s) / prodW p1 p2 s)
          = min (prodW p1 p2 s) (prodW p1 p2 (Prod.swap s)) :=
        Metropolis.mul_acc (prodW_pos h1 h2 s)
      have e2 : prodW p1 p2 (Prod.swap s)
            * min 1 (prodW p1 p2 (Prod.swap (Prod.swap s)) / prodW p1 p2 (Prod.swap s))
          = min (prodW p1 p2 (Prod.swap s)) (prodW p1 p2 (Prod.swap (Prod.swap s))) :=
        Metropolis.mul_acc (prodW_pos h1 h2 _)
      rw [e1, e2]
      simp [min_comm]
  · by_cases hts' : t = s
    · rw [hts']
    · have h1' : swapK p1 p2 s t = 0 := by unfold swapK; rw [if_neg hts, if_neg hts']
      have h2' : swapK p1 p2 t s = 0 := by
        unfold swapK
        rw [if_neg, if_neg (Ne.symm hts')]
        intro h
        exact hts (by rw [h]; simp)
      rw [h1', h2', mul_zero, mul_zero]

/-- Consequently the extended ensemble is stationary: replica exchange is **unbiased**. -/
theorem swapK_stationary {p1 p2 : Fin n → ℝ} (h1 : ∀ i, 0 < p1 i) (h2 : ∀ j, 0 < p2 j)
    (t : Fin n × Fin n) :
    ∑ s, prodW p1 p2 s * swapK p1 p2 s t = prodW p1 p2 t := by
  calc ∑ s, prodW p1 p2 s * swapK p1 p2 s t
      = ∑ s, prodW p1 p2 t * swapK p1 p2 t s :=
        Finset.sum_congr rfl (fun s _ => swapK_detailedBalance h1 h2 s t)
    _ = prodW p1 p2 t * ∑ s, swapK p1 p2 t s := by rw [Finset.mul_sum]
    _ = prodW p1 p2 t := by rw [swapK_stochastic h1 h2 t, mul_one]

/-! ## The acceptance probability, and what it measures -/

/-- The Boltzmann weights of an energy function at inverse temperature `beta`. -/
noncomputable def boltzW (beta : ℝ) (E : Fin n → ℝ) (i : Fin n) : ℝ :=
  Real.exp (-beta * E i) / ∑ k, Real.exp (-beta * E k)

lemma boltzZ_pos [NeZero n] (beta : ℝ) (E : Fin n → ℝ) : 0 < ∑ k, Real.exp (-beta * E k) := by
  have : (Finset.univ : Finset (Fin n)).Nonempty := Finset.univ_nonempty
  exact Finset.sum_pos (fun k _ => Real.exp_pos _) this

lemma boltzW_pos [NeZero n] (beta : ℝ) (E : Fin n → ℝ) (i : Fin n) : 0 < boltzW beta E i :=
  div_pos (Real.exp_pos _) (boltzZ_pos beta E)

lemma boltzW_sum [NeZero n] (beta : ℝ) (E : Fin n → ℝ) : ∑ i, boltzW beta E i = 1 := by
  unfold boltzW
  rw [← Finset.sum_div, div_self (boltzZ_pos beta E).ne']

/-- The swap acceptance probability written in terms of the energies alone. -/
noncomputable def accE (b1 b2 : ℝ) (E : Fin n → ℝ) (s : Fin n × Fin n) : ℝ :=
  min 1 (Real.exp ((b2 - b1) * (E s.2 - E s.1)))

/-- **No partition function appears.**  The Metropolis acceptance for the swap depends only on
the two inverse temperatures and the two energies. -/
theorem accSwap_boltz [NeZero n] (b1 b2 : ℝ) (E : Fin n → ℝ) (s : Fin n × Fin n) :
    accSwap (boltzW b1 E) (boltzW b2 E) s = accE b1 b2 E s := by
  have hZ1 := (boltzZ_pos b1 E).ne'
  have hZ2 := (boltzZ_pos b2 E).ne'
  unfold accSwap accE prodW boltzW
  congr 1
  rw [Prod.fst_swap, Prod.snd_swap]
  have hcancel : (Real.exp (-b1 * E s.2) / (∑ k, Real.exp (-b1 * E k))) *
        (Real.exp (-b2 * E s.1) / (∑ k, Real.exp (-b2 * E k))) /
      ((Real.exp (-b1 * E s.1) / (∑ k, Real.exp (-b1 * E k))) *
        (Real.exp (-b2 * E s.2) / (∑ k, Real.exp (-b2 * E k))))
      = (Real.exp (-b1 * E s.2) * Real.exp (-b2 * E s.1)) /
          (Real.exp (-b1 * E s.1) * Real.exp (-b2 * E s.2)) := by
    field_simp
  rw [hcancel, ← Real.exp_add, ← Real.exp_add, ← Real.exp_sub]
  congr 1
  ring

/-- The mean acceptance rate of the swap move in the extended ensemble. -/
noncomputable def meanAcc (p1 p2 : Fin n → ℝ) : ℝ := ∑ s, prodW p1 p2 s * accSwap p1 p2 s

/-- **Acceptance is overlap.**  The mean acceptance rate is exactly the overlap mass between the
extended weight and its swap. -/
theorem meanAcc_eq_overlap {p1 p2 : Fin n → ℝ} (h1 : ∀ i, 0 < p1 i) (h2 : ∀ j, 0 < p2 j) :
    meanAcc p1 p2 = ∑ s, min (prodW p1 p2 s) (prodW p1 p2 (Prod.swap s)) := by
  unfold meanAcc accSwap
  exact Finset.sum_congr rfl (fun s _ => Metropolis.mul_acc (prodW_pos h1 h2 s))

/-- Equivalently, the mean acceptance rate is `1 - TV(pi, pi ∘ swap)`. -/
theorem meanAcc_eq_one_sub_tv {p1 p2 : Fin n → ℝ} (h1 : ∀ i, 0 < p1 i) (h2 : ∀ j, 0 < p2 j)
    (hs1 : ∑ i, p1 i = 1) (hs2 : ∑ j, p2 j = 1) :
    meanAcc p1 p2
      = 1 - (1/2) * ∑ s, |prodW p1 p2 s - prodW p1 p2 (Prod.swap s)| := by
  rw [meanAcc_eq_overlap h1 h2]
  have key : ∀ s : Fin n × Fin n,
      min (prodW p1 p2 s) (prodW p1 p2 (Prod.swap s))
        = (prodW p1 p2 s + prodW p1 p2 (Prod.swap s)
            - |prodW p1 p2 s - prodW p1 p2 (Prod.swap s)|) / 2 := by
    intro s
    rcases le_total (prodW p1 p2 s) (prodW p1 p2 (Prod.swap s)) with h | h
    · rw [min_eq_left h, abs_of_nonpos (by linarith)]; ring
    · rw [min_eq_right h, abs_of_nonneg (by linarith)]; ring
  rw [Finset.sum_congr rfl (fun s _ => key s)]
  rw [← Finset.sum_div, Finset.sum_sub_distrib, Finset.sum_add_distrib]
  rw [sum_swap (fun s => prodW p1 p2 s), prodW_sum hs1 hs2]
  ring

/-- At equal temperatures every swap is accepted -- and nothing is learned, since the two
replicas then sample the same ensemble.  A high acceptance rate is not evidence of efficiency. -/
theorem meanAcc_flat_eq_one [NeZero n] (b : ℝ) (E : Fin n → ℝ) :
    meanAcc (boltzW b E) (boltzW b E) = 1 := by
  unfold meanAcc
  have hacc : ∀ s : Fin n × Fin n, accSwap (boltzW b E) (boltzW b E) s = 1 := by
    intro s
    rw [accSwap_boltz b b E s]
    unfold accE
    simp
  rw [Finset.sum_congr rfl (fun s _ => by rw [hacc s, mul_one])]
  exact prodW_sum (boltzW_sum b E) (boltzW_sum b E)

/-! ## Separated histograms: the exponential cost of a temperature gap -/

/-- **If the energy histograms are separated by a gap, acceptance is exponentially small.** -/
theorem meanAcc_le_exp_neg_gap {p1 p2 : Fin n → ℝ} {E : Fin n → ℝ} {b1 b2 a g : ℝ}
    (hb : b1 ≤ b2)
    (hp1 : ∀ i, 0 ≤ p1 i) (hp2 : ∀ j, 0 ≤ p2 j)
    (hs1 : ∑ i, p1 i = 1) (hs2 : ∑ j, p2 j = 1)
    (hE1 : ∀ i, p1 i ≠ 0 → a ≤ E i) (hE2 : ∀ j, p2 j ≠ 0 → E j ≤ a - g) :
    ∑ s, prodW p1 p2 s * accE b1 b2 E s ≤ Real.exp (-((b2 - b1) * g)) := by
  have hterm : ∀ s : Fin n × Fin n,
      prodW p1 p2 s * accE b1 b2 E s ≤ prodW p1 p2 s * Real.exp (-((b2 - b1) * g)) := by
    intro s
    rcases eq_or_ne (prodW p1 p2 s) 0 with h | h
    · rw [h, zero_mul, zero_mul]
    · have h1 : p1 s.1 ≠ 0 := fun hh => h (by unfold prodW; rw [hh, zero_mul])
      have h2 : p2 s.2 ≠ 0 := fun hh => h (by unfold prodW; rw [hh, mul_zero])
      have hEd : E s.2 - E s.1 ≤ -g := by
        have := hE1 s.1 h1
        have := hE2 s.2 h2
        linarith
      have hmul : (b2 - b1) * (E s.2 - E s.1) ≤ -((b2 - b1) * g) := by
        have hb' : 0 ≤ b2 - b1 := by linarith
        nlinarith
      have : accE b1 b2 E s ≤ Real.exp (-((b2 - b1) * g)) :=
        le_trans (min_le_right _ _) (Real.exp_le_exp.2 hmul)
      exact mul_le_mul_of_nonneg_left this (prodW_nonneg hp1 hp2 s)
  calc ∑ s, prodW p1 p2 s * accE b1 b2 E s
      ≤ ∑ s, prodW p1 p2 s * Real.exp (-((b2 - b1) * g)) := Finset.sum_le_sum (fun s _ => hterm s)
    _ = Real.exp (-((b2 - b1) * g)) := by
        rw [← Finset.sum_mul, prodW_sum hs1 hs2, one_mul]

/-- The bound is attained: two sharply peaked replicas separated by an energy gap `g` accept
swaps at exactly the rate `exp(-(b2-b1) g)`. -/
theorem meanAcc_two_state {b1 b2 g : ℝ} (hb : b1 ≤ b2) (hg : 0 ≤ g) :
    ∑ s, prodW (![0, 1] : Fin 2 → ℝ) (![1, 0] : Fin 2 → ℝ) s
        * accE b1 b2 (![0, g] : Fin 2 → ℝ) s
      = Real.exp (-((b2 - b1) * g)) := by
  have hle : Real.exp ((b2 - b1) * (0 - g)) ≤ 1 := by
    have h0 : (b2 - b1) * (0 - g) ≤ 0 := by nlinarith
    calc Real.exp ((b2 - b1) * (0 - g)) ≤ Real.exp 0 := Real.exp_le_exp.2 h0
      _ = 1 := Real.exp_zero
  simp only [Fintype.sum_prod_type, Fin.sum_univ_two, prodW, accE, Matrix.cons_val_zero,
    Matrix.cons_val_one]
  rw [min_eq_right hle]
  norm_num

/-- **Holding an acceptance target caps the temperature step.** -/
theorem deltaBeta_le_of_meanAcc {p1 p2 : Fin n → ℝ} {E : Fin n → ℝ} {b1 b2 a g al : ℝ}
    (hb : b1 ≤ b2) (hal : 0 < al)
    (hp1 : ∀ i, 0 ≤ p1 i) (hp2 : ∀ j, 0 ≤ p2 j)
    (hs1 : ∑ i, p1 i = 1) (hs2 : ∑ j, p2 j = 1)
    (hE1 : ∀ i, p1 i ≠ 0 → a ≤ E i) (hE2 : ∀ j, p2 j ≠ 0 → E j ≤ a - g)
    (hacc : al ≤ ∑ s, prodW p1 p2 s * accE b1 b2 E s) :
    (b2 - b1) * g ≤ Real.log (1 / al) := by
  have h := le_trans hacc (meanAcc_le_exp_neg_gap hb hp1 hp2 hs1 hs2 hE1 hE2)
  have := Real.log_le_log hal h
  rw [Real.log_exp] at this
  rw [Real.log_div one_ne_zero hal.ne', Real.log_one, zero_sub]
  linarith

/-- Telescoping: a ladder whose rungs are at most `d` apart and which spans `b 0` to `b K`
must have at least `(b K - b 0)/d` rungs. -/
theorem ladder_span (b : ℕ → ℝ) (d : ℝ) (K : ℕ) (hd : ∀ k, k < K → b (k + 1) - b k ≤ d) :
    b K - b 0 ≤ K * d := by
  induction K with
  | zero => simp
  | succ m ih =>
      have hm : b m - b 0 ≤ m * d := ih (fun k hk => hd k (by omega))
      have hstep : b (m + 1) - b m ≤ d := hd m (by omega)
      push_cast
      linarith

/-- **How many replicas a run needs.**  If every rung of the ladder must keep acceptance at
least `al` against an energy gap `g > 0`, the ladder spanning `[b 0, b K]` needs
`K ≥ (b K - b 0) * g / log(1/al)` rungs. -/
theorem replicas_needed (b : ℕ → ℝ) (K : ℕ) {g al : ℝ} (hg : 0 < g) (hal : 0 < al) (hal1 : al < 1)
    (hd : ∀ k, k < K → (b (k + 1) - b k) * g ≤ Real.log (1 / al)) :
    (b K - b 0) * g / Real.log (1 / al) ≤ K := by
  have hlog : 0 < Real.log (1 / al) := by
    rw [Real.log_div one_ne_zero hal.ne', Real.log_one, zero_sub, neg_pos]
    exact Real.log_neg hal hal1
  have hstep : ∀ k, k < K → b (k + 1) - b k ≤ Real.log (1 / al) / g := by
    intro k hk
    have := hd k hk
    rw [le_div_iff₀ hg]
    exact this
  have := ladder_span b (Real.log (1 / al) / g) K hstep
  rw [div_le_iff₀ hlog]
  have hK : (b K - b 0) * g ≤ K * Real.log (1 / al) := by
    have h' : (b K - b 0) * g ≤ (K * (Real.log (1 / al) / g)) * g :=
      mul_le_mul_of_nonneg_right this hg.le
    calc (b K - b 0) * g ≤ (K * (Real.log (1 / al) / g)) * g := h'
      _ = K * Real.log (1 / al) := by field_simp
  linarith

/-! ## Replica exchange cannot repair broken ergodicity -/

/-- A move set that never crosses between `A` and its complement. -/
def Blocked (A : Finset (Fin n)) (P : Fin n → Fin n → ℝ) : Prop :=
  ∀ i j, ((i ∈ A ∧ j ∉ A) ∨ (i ∉ A ∧ j ∈ A)) → P i j = 0

/-- The number of replicas currently holding a configuration inside `A`. -/
def cntA (A : Finset (Fin n)) (s : Fin n × Fin n) : ℕ :=
  (if s.1 ∈ A then 1 else 0) + (if s.2 ∈ A then 1 else 0)

/-- Independent within-temperature updates of the two replicas. -/
def withinK (P1 P2 : Fin n → Fin n → ℝ) (s t : Fin n × Fin n) : ℝ := P1 s.1 t.1 * P2 s.2 t.2

/-- **The replica-exchange chain**: with probability `lam` update each replica at its own
temperature, otherwise attempt a swap. -/
noncomputable def reK (lam : ℝ) (P1 P2 : Fin n → Fin n → ℝ) (p1 p2 : Fin n → ℝ)
    (s t : Fin n × Fin n) : ℝ :=
  lam * withinK P1 P2 s t + (1 - lam) * swapK p1 p2 s t

/-- The within-temperature part preserves the extended ensemble. -/
theorem withinK_stationary {P1 P2 : Fin n → Fin n → ℝ} {p1 p2 : Fin n → ℝ}
    (h1 : ∀ j, ∑ i, p1 i * P1 i j = p1 j) (h2 : ∀ j, ∑ i, p2 i * P2 i j = p2 j)
    (t : Fin n × Fin n) :
    ∑ s, prodW p1 p2 s * withinK P1 P2 s t = prodW p1 p2 t := by
  unfold prodW withinK
  rw [Fintype.sum_prod_type]
  calc ∑ i, ∑ j, p1 i * p2 j * (P1 i t.1 * P2 j t.2)
      = ∑ i, (p1 i * P1 i t.1) * ∑ j, (p2 j * P2 j t.2) := by
        refine Finset.sum_congr rfl (fun i _ => ?_)
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl (fun j _ => by ring)
    _ = (∑ i, p1 i * P1 i t.1) * ∑ j, (p2 j * P2 j t.2) := by rw [Finset.sum_mul]
    _ = p1 t.1 * p2 t.2 := by rw [h1 t.1, h2 t.2]

/-- The whole replica-exchange chain preserves the extended ensemble: each temperature keeps
sampling its own Boltzmann ensemble. -/
theorem reK_stationary {lam : ℝ} {P1 P2 : Fin n → Fin n → ℝ} {p1 p2 : Fin n → ℝ}
    (hp1 : ∀ i, 0 < p1 i) (hp2 : ∀ j, 0 < p2 j)
    (h1 : ∀ j, ∑ i, p1 i * P1 i j = p1 j) (h2 : ∀ j, ∑ i, p2 i * P2 i j = p2 j)
    (t : Fin n × Fin n) :
    ∑ s, prodW p1 p2 s * reK lam P1 P2 p1 p2 s t = prodW p1 p2 t := by
  unfold reK
  have : ∀ s : Fin n × Fin n,
      prodW p1 p2 s * (lam * withinK P1 P2 s t + (1 - lam) * swapK p1 p2 s t)
        = lam * (prodW p1 p2 s * withinK P1 P2 s t)
          + (1 - lam) * (prodW p1 p2 s * swapK p1 p2 s t) := by
    intro s; ring
  rw [Finset.sum_congr rfl (fun s _ => this s), Finset.sum_add_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, withinK_stationary h1 h2 t, swapK_stationary hp1 hp2 t]
  ring

/-- **The conservation law.**  If neither temperature's move set crosses the `A`-boundary, the
extended chain cannot change the number of replicas inside `A`. -/
theorem reK_conserves_count {A : Finset (Fin n)} {lam : ℝ} {P1 P2 : Fin n → Fin n → ℝ}
    {p1 p2 : Fin n → ℝ} (hA1 : Blocked A P1) (hA2 : Blocked A P2)
    (s t : Fin n × Fin n) (h : cntA A t ≠ cntA A s) :
    reK lam P1 P2 p1 p2 s t = 0 := by
  have hwithin : withinK P1 P2 s t = 0 := by
    unfold withinK
    by_cases h1 : s.1 ∈ A <;> by_cases h1' : t.1 ∈ A <;>
      by_cases h2 : s.2 ∈ A <;> by_cases h2' : t.2 ∈ A
    all_goals
      first
        | (exfalso; apply h; unfold cntA; simp [h1, h1', h2, h2']; done)
        | (rw [hA1 s.1 t.1 (by tauto), zero_mul])
        | (rw [hA2 s.2 t.2 (by tauto), mul_zero])
  have hswap : swapK p1 p2 s t = 0 := by
    unfold swapK
    rw [if_neg, if_neg]
    · intro hts; exact h (by rw [hts])
    · intro hts
      apply h
      rw [hts]
      unfold cntA
      simp [Nat.add_comm]
  rw [reK, hwithin, hswap, mul_zero, mul_zero, add_zero]

/-- Iterating a kernel on the extended state space. -/
noncomputable def iter (K : (Fin n × Fin n) → (Fin n × Fin n) → ℝ) :
    ℕ → ((Fin n × Fin n) → ℝ) → ((Fin n × Fin n) → ℝ)
  | 0, w => w
  | m + 1, w => fun t => ∑ s, iter K m w s * K s t

/-- The support of the run stays inside one level set of the conserved count. -/
theorem re_iterate_support {A : Finset (Fin n)} {lam : ℝ} {P1 P2 : Fin n → Fin n → ℝ}
    {p1 p2 : Fin n → ℝ} (hA1 : Blocked A P1) (hA2 : Blocked A P2)
    {w : (Fin n × Fin n) → ℝ} {c : ℕ} (hw : ∀ s, w s ≠ 0 → cntA A s = c) (m : ℕ) :
    ∀ t, cntA A t ≠ c → iter (reK lam P1 P2 p1 p2) m w t = 0 := by
  induction m with
  | zero => intro t ht; by_contra hne; exact ht (hw t hne)
  | succ k ih =>
      intro t ht
      refine Finset.sum_eq_zero (fun s _ => ?_)
      rcases eq_or_ne (cntA A s) c with hs | hs
      · rw [reK_conserves_count hA1 hA2 s t (by rw [hs]; exact ht), mul_zero]
      · rw [ih s hs, zero_mul]

/-- **Tempering does not connect what the move set disconnects.**  With both replicas started
inside `A`, no state in which some replica sits outside `A` is ever reached, at any temperature
and for any number of sweeps -- even though the Boltzmann ensemble puts positive weight there. -/
theorem replica_exchange_cannot_repair_ergodicity [NeZero n] {A : Finset (Fin n)} {lam : ℝ}
    {P1 P2 : Fin n → Fin n → ℝ} {b1 b2 : ℝ} {E : Fin n → ℝ}
    (hA1 : Blocked A P1) (hA2 : Blocked A P2)
    {w : (Fin n × Fin n) → ℝ} (hw : ∀ s, w s ≠ 0 → (s.1 ∈ A ∧ s.2 ∈ A)) (m : ℕ) :
    (∀ t : Fin n × Fin n, t.1 ∉ A →
        iter (reK lam P1 P2 (boltzW b1 E) (boltzW b2 E)) m w t = 0)
      ∧ (∀ i, i ∉ A → 0 < boltzW b1 E i) := by
  constructor
  · intro t ht
    refine re_iterate_support (c := 2) hA1 hA2 (fun s hs => ?_) m t ?_
    · have := hw s hs
      unfold cntA
      simp [this.1, this.2]
    · unfold cntA
      by_cases h2 : t.2 ∈ A <;> simp [ht, h2]
  · intro i _
    exact boltzW_pos b1 E i

end RE

end IDR
