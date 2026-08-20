/-
# Part LXVIII  The full ladder: `K` replicas, and the invariant that survives

Part LXV treats one exchanging pair, which is what a single exchange attempt is.  A production
run is a *ladder*: `K` replicas at inverse temperatures `b 0 < ... < b (K-1)`, each carrying its
own configuration, with neighbouring pairs attempting exchanges.  This file proves the same three
facts for the ladder, and does so through a general lemma that isolates what makes them true.

* `invK_detailedBalance`, `invK_stochastic`, `invK_stationary`, `invAcc_mean_eq_overlap` -- the
  general statement: the Metropolis move *along an involution* `s` of any finite state space is a
  transition kernel, is reversible with respect to any positive weight, leaves it stationary, and
  accepts at a rate exactly equal to the overlap of the weight with its image under `s`.  A
  replica exchange is exactly such a move, the involution being "swap the configurations held by
  replicas `a` and `c`".
* `ladderW_ratio`, `ladder_acc_boltz` -- on the ladder the acceptance ratio collapses to the two
  replicas involved: `min 1 exp((b c − b a)(E (x c) − E (x a)))`.  No partition function, and no
  dependence on the other `K−2` replicas.
* `ladder_unbiased` -- **the ladder is unbiased**: in the extended ensemble the average of any
  observable of replica `k`'s configuration is exactly its Boltzmann average at that replica's own
  temperature, for every `k`, and the swap move preserves the extended ensemble exactly.
* `ladder_swap_conserves_count`, `ladder_within_conserves_count`, `ladder_iterate_support`,
  `ladder_cannot_repair_ergodicity` -- **and the invariant survives the whole ladder**: if no
  within-temperature move crosses between a set `A` of conformations and its complement, the
  number of replicas holding a configuration in `A` is conserved by every exchange and every
  within-temperature update, so a run started with all `K` replicas inside `A` never leaves `A`,
  at any temperature and after any number of sweeps.  Adding rungs to the ladder does not help:
  the conserved quantity is a property of the move set, not of the temperature range.
-/
import Mathlib
import RequestProject.ReplicaExchange

set_option autoImplicit false

namespace IDR

namespace Ladder

open Finset

/-! ## Metropolis along an involution -/

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- The Metropolis acceptance probability for the move `x ↦ s x`. -/
noncomputable def invAcc (pi : S → ℝ) (s : S → S) (x : S) : ℝ := min 1 (pi (s x) / pi x)

omit [Fintype S] [DecidableEq S] in
lemma invAcc_le_one (pi : S → ℝ) (s : S → S) (x : S) : invAcc pi s x ≤ 1 := min_le_left _ _

omit [Fintype S] [DecidableEq S] in
lemma invAcc_nonneg {pi : S → ℝ} (hpi : ∀ x, 0 < pi x) (s : S → S) (x : S) :
    0 ≤ invAcc pi s x :=
  le_min zero_le_one (div_nonneg (hpi _).le (hpi _).le)

/-- The Metropolis kernel of the move `x ↦ s x`. -/
noncomputable def invK (pi : S → ℝ) (s : S → S) (x y : S) : ℝ :=
  if y = s x then invAcc pi s x else if y = x then 1 - invAcc pi s x else 0

/-- The move along an involution is a transition kernel. -/
theorem invK_stochastic {pi : S → ℝ} (hpi : ∀ x, 0 < pi x) (s : S → S) (x : S) :
    ∑ y, invK pi s x y = 1 := by
  by_cases hs : s x = x
  · have hval : ∀ y : S, invK pi s x y = if y = x then invAcc pi s x else 0 := by
      intro y
      unfold invK
      rw [hs]
      split_ifs <;> rfl
    rw [Finset.sum_congr rfl (fun y _ => hval y),
      Finset.sum_ite_eq' Finset.univ x (fun _ => invAcc pi s x)]
    simp only [Finset.mem_univ, if_true]
    unfold invAcc
    rw [hs, div_self (hpi x).ne', min_self]
  · rw [Finset.sum_eq_add_of_mem (s x) x (Finset.mem_univ _) (Finset.mem_univ _) hs]
    · unfold invK
      rw [if_pos rfl, if_neg (Ne.symm hs), if_pos rfl]
      ring
    · intro c _ hc
      unfold invK
      rw [if_neg hc.1, if_neg hc.2]

omit [Fintype S] in
/-- **Detailed balance along an involution.** -/
theorem invK_detailedBalance {pi : S → ℝ} (hpi : ∀ x, 0 < pi x) {s : S → S}
    (hs : Function.Involutive s) (x y : S) :
    pi x * invK pi s x y = pi y * invK pi s y x := by
  by_cases hyx : y = s x
  · subst hyx
    have hxs : x = s (s x) := (hs x).symm
    unfold invK
    rw [if_pos rfl, if_pos hxs]
    unfold invAcc
    rw [Metropolis.mul_acc (hpi x), Metropolis.mul_acc (hpi (s x)), ← hxs, min_comm]
  · by_cases hyx' : y = x
    · rw [hyx']
    · have h1 : invK pi s x y = 0 := by unfold invK; rw [if_neg hyx, if_neg hyx']
      have h2 : invK pi s y x = 0 := by
        unfold invK
        rw [if_neg, if_neg (Ne.symm hyx')]
        intro h
        exact hyx (by rw [h, hs y])
      rw [h1, h2, mul_zero, mul_zero]

/-- Hence the weight is stationary: the move is unbiased. -/
theorem invK_stationary {pi : S → ℝ} (hpi : ∀ x, 0 < pi x) {s : S → S}
    (hs : Function.Involutive s) (y : S) :
    ∑ x, pi x * invK pi s x y = pi y := by
  calc ∑ x, pi x * invK pi s x y = ∑ x, pi y * invK pi s y x :=
        Finset.sum_congr rfl (fun x _ => invK_detailedBalance hpi hs x y)
    _ = pi y * ∑ x, invK pi s y x := by rw [Finset.mul_sum]
    _ = pi y := by rw [invK_stochastic hpi s y, mul_one]

omit [DecidableEq S] in
/-- The mean acceptance rate of the move is exactly the overlap of the weight with its image. -/
theorem invAcc_mean_eq_overlap {pi : S → ℝ} (hpi : ∀ x, 0 < pi x) (s : S → S) :
    ∑ x, pi x * invAcc pi s x = ∑ x, min (pi x) (pi (s x)) :=
  Finset.sum_congr rfl (fun x _ => Metropolis.mul_acc (hpi x))

/-! ## The ladder -/

variable {K n : ℕ}

/-- Exchange the configurations held by replicas `a` and `c`. -/
def swapPair (a c : Fin K) (x : Fin K → Fin n) : Fin K → Fin n :=
  fun k => if k = a then x c else if k = c then x a else x k

lemma swapPair_involutive (a c : Fin K) : Function.Involutive (swapPair (n := n) a c) := by
  intro x
  funext k
  simp only [swapPair]
  by_cases hka : k = a
  · subst hka
    by_cases hca : c = k
    · simp [hca]
    · simp [hca]
  · by_cases hkc : k = c
    · subst hkc
      simp [hka]
    · simp [hka, hkc]

/-- The weight of a ladder configuration in the extended ensemble. -/
noncomputable def ladderW (p : Fin K → Fin n → ℝ) (x : Fin K → Fin n) : ℝ := ∏ k, p k (x k)

lemma ladderW_pos {p : Fin K → Fin n → ℝ} (hp : ∀ k i, 0 < p k i) (x : Fin K → Fin n) :
    0 < ladderW p x :=
  Finset.prod_pos (fun k _ => hp k (x k))

lemma prod_update_mem (p : Fin K → Fin n → ℝ) (y : Fin K → Fin n) (i : Fin K) (v : Fin n)
    {s : Finset (Fin K)} (hi : i ∈ s) :
    ∏ k ∈ s, p k (Function.update y i v k) = p i v * ∏ k ∈ s.erase i, p k (y k) := by
  rw [← Finset.prod_erase_mul s _ hi]
  rw [Function.update_self]
  rw [Finset.prod_congr rfl (fun k hk => by
    rw [Function.update_of_ne (Finset.mem_erase.mp hk).1])]
  ring

lemma swapPair_eq_update {a c : Fin K} (hac : a ≠ c) (x : Fin K → Fin n) :
    swapPair a c x = Function.update (Function.update x c (x a)) a (x c) := by
  funext k
  simp only [swapPair]
  by_cases hka : k = a
  · subst hka
    simp [Function.update_self]
  · by_cases hkc : k = c
    · subst hkc
      simp [hka]
    · simp [hka, hkc]

/-- **The exchange ratio collapses to the two replicas involved.** -/
theorem ladderW_ratio {p : Fin K → Fin n → ℝ} (hp : ∀ k i, 0 < p k i) {a c : Fin K} (hac : a ≠ c)
    (x : Fin K → Fin n) :
    ladderW p (swapPair a c x) / ladderW p x
      = (p a (x c) * p c (x a)) / (p a (x a) * p c (x c)) := by
  have hcmem : c ∈ (Finset.univ : Finset (Fin K)).erase a :=
    Finset.mem_erase.mpr ⟨Ne.symm hac, Finset.mem_univ c⟩
  have hswap : ladderW p (swapPair a c x)
      = p a (x c) * (p c (x a) * ∏ k ∈ (Finset.univ.erase a).erase c, p k (x k)) := by
    unfold ladderW
    rw [swapPair_eq_update hac,
      prod_update_mem p (Function.update x c (x a)) a (x c) (Finset.mem_univ a)]
    congr 1
    rw [prod_update_mem p x c (x a) hcmem]
  have hid : ladderW p x
      = p a (x a) * (p c (x c) * ∏ k ∈ (Finset.univ.erase a).erase c, p k (x k)) := by
    unfold ladderW
    rw [← Finset.prod_erase_mul Finset.univ (fun k => p k (x k)) (Finset.mem_univ a),
      ← Finset.prod_erase_mul (Finset.univ.erase a) (fun k => p k (x k)) hcmem]
    ring
  have hrest : (0:ℝ) < ∏ k ∈ (Finset.univ.erase a).erase c, p k (x k) :=
    Finset.prod_pos (fun k _ => hp k (x k))
  rw [hswap, hid]
  have h1 := (hp a (x a)).ne'
  have h2 := (hp c (x c)).ne'
  field_simp

/-- **The acceptance probability of an exchange on the ladder.** -/
theorem ladder_acc_boltz [NeZero n] (b : Fin K → ℝ) (E : Fin n → ℝ) {a c : Fin K} (hac : a ≠ c)
    (x : Fin K → Fin n) :
    invAcc (ladderW (fun k => RE.boltzW (b k) E)) (swapPair a c) x
      = min 1 (Real.exp ((b c - b a) * (E (x c) - E (x a)))) := by
  have hp : ∀ (k : Fin K) (i : Fin n), 0 < RE.boltzW (b k) E i := fun k i => RE.boltzW_pos _ _ _
  unfold invAcc
  rw [ladderW_ratio hp hac x]
  congr 1
  unfold RE.boltzW
  have hZa := (RE.boltzZ_pos (b a) E).ne'
  have hZc := (RE.boltzZ_pos (b c) E).ne'
  have hcancel :
      (Real.exp (-b a * E (x c)) / (∑ k, Real.exp (-b a * E k))) *
          (Real.exp (-b c * E (x a)) / (∑ k, Real.exp (-b c * E k))) /
        ((Real.exp (-b a * E (x a)) / (∑ k, Real.exp (-b a * E k))) *
          (Real.exp (-b c * E (x c)) / (∑ k, Real.exp (-b c * E k))))
        = (Real.exp (-b a * E (x c)) * Real.exp (-b c * E (x a))) /
            (Real.exp (-b a * E (x a)) * Real.exp (-b c * E (x c))) := by
    field_simp
  rw [hcancel, ← Real.exp_add, ← Real.exp_add, ← Real.exp_sub]
  congr 1
  ring

/-! ## Unbiasedness of the ladder -/

lemma ladder_prod_split (p : Fin K → Fin n → ℝ) (f : Fin n → ℝ) (k0 : Fin K)
    (x : Fin K → Fin n) :
    (∏ k, (if k = k0 then p k (x k) * f (x k) else p k (x k))) = ladderW p x * f (x k0) := by
  unfold ladderW
  rw [← Finset.prod_erase_mul Finset.univ (fun k => if k = k0 then p k (x k) * f (x k)
      else p k (x k)) (Finset.mem_univ k0)]
  rw [← Finset.prod_erase_mul Finset.univ (fun k => p k (x k)) (Finset.mem_univ k0)]
  rw [if_pos rfl]
  rw [Finset.prod_congr rfl (fun k hk => by rw [if_neg (Finset.mem_erase.mp hk).1])]
  ring

/-- **The ladder is unbiased.**  In the extended ensemble the average of an observable of the
configuration held by replica `k0` is exactly its Boltzmann average at replica `k0`'s own
temperature -- whatever the other replicas are doing. -/
theorem ladder_unbiased {p : Fin K → Fin n → ℝ} (hp1 : ∀ k, ∑ i, p k i = 1) (k0 : Fin K)
    (f : Fin n → ℝ) :
    ∑ x, ladderW p x * f (x k0) = ∑ i, p k0 i * f i := by
  have hkey : ∑ x : Fin K → Fin n, ∏ k, (if k = k0 then p k (x k) * f (x k) else p k (x k))
      = ∏ k, ∑ i, (if k = k0 then p k i * f i else p k i) := by
    rw [Finset.prod_univ_sum (fun _ => (Finset.univ : Finset (Fin n)))
      (fun k i => if k = k0 then p k i * f i else p k i), Fintype.piFinset_univ]
  calc ∑ x : Fin K → Fin n, ladderW p x * f (x k0)
      = ∑ x : Fin K → Fin n, ∏ k, (if k = k0 then p k (x k) * f (x k) else p k (x k)) :=
        Finset.sum_congr rfl (fun x _ => (ladder_prod_split p f k0 x).symm)
    _ = ∏ k, ∑ i, (if k = k0 then p k i * f i else p k i) := hkey
    _ = ∑ i, p k0 i * f i := by
        rw [← Finset.prod_erase_mul Finset.univ _ (Finset.mem_univ k0)]
        rw [Finset.prod_congr rfl (fun k hk => by
          rw [Finset.sum_congr rfl (fun i _ => by rw [if_neg (Finset.mem_erase.mp hk).1]),
            hp1 k])]
        simp

/-- The exchange move preserves the extended ensemble of the ladder. -/
theorem ladder_swap_stationary {p : Fin K → Fin n → ℝ} (hp : ∀ k i, 0 < p k i) (a c : Fin K)
    (y : Fin K → Fin n) :
    ∑ x, ladderW p x * invK (ladderW p) (swapPair a c) x y = ladderW p y :=
  invK_stationary (fun x => ladderW_pos hp x) (swapPair_involutive a c) y

/-! ## The conserved count, on the whole ladder -/

/-- The number of replicas currently holding a configuration inside `A`. -/
def cntA (A : Finset (Fin n)) (x : Fin K → Fin n) : ℕ :=
  ∑ k, if x k ∈ A then 1 else 0

/-- An exchange permutes the configurations held, so it cannot change the count. -/
theorem ladder_swap_conserves_count (A : Finset (Fin n)) (a c : Fin K) (x : Fin K → Fin n) :
    cntA A (swapPair a c x) = cntA A x := by
  unfold cntA swapPair
  by_cases hac : a = c
  · subst hac
    refine Finset.sum_congr rfl (fun k _ => ?_)
    by_cases hka : k = a <;> simp [hka]
  · rw [← Equiv.sum_comp (Equiv.swap a c)
      (fun k => if (if k = a then x c else if k = c then x a else x k) ∈ A then 1 else 0)]
    refine Finset.sum_congr rfl (fun k _ => ?_)
    by_cases hka : k = a
    · subst hka
      simp [Equiv.swap_apply_left, Ne.symm hac]
    · by_cases hkc : k = c
      · subst hkc
        simp [Equiv.swap_apply_right]
      · rw [Equiv.swap_apply_of_ne_of_ne hka hkc]
        simp [hka, hkc]

/-- A within-temperature update that never crosses the boundary of `A` cannot change the count
either. -/
theorem ladder_within_conserves_count {A : Finset (Fin n)} {P : Fin K → Fin n → Fin n → ℝ}
    (hP : ∀ k, RE.Blocked A (P k)) (x y : Fin K → Fin n) (h : cntA A y ≠ cntA A x) :
    (∏ k, P k (x k) (y k)) = 0 := by
  by_contra hne
  apply h
  have hmem : ∀ k, (x k ∈ A ↔ y k ∈ A) := by
    intro k
    by_contra hk
    have hzero : P k (x k) (y k) = 0 := by
      rcases Classical.em (x k ∈ A) with hx | hx
      · exact hP k (x k) (y k) (Or.inl ⟨hx, fun hy => hk ⟨fun _ => hy, fun _ => hx⟩⟩)
      · have hy : y k ∈ A := by
          by_contra hy
          exact hk ⟨fun hh => absurd hh hx, fun hh => absurd hh hy⟩
        exact hP k (x k) (y k) (Or.inr ⟨hx, hy⟩)
    exact hne (Finset.prod_eq_zero (Finset.mem_univ k) hzero)
  unfold cntA
  refine Finset.sum_congr rfl (fun k _ => ?_)
  by_cases hk : x k ∈ A
  · rw [if_pos hk, if_pos ((hmem k).mp hk)]
  · rw [if_neg hk, if_neg (fun hy => hk ((hmem k).mpr hy))]

/-- The kernel of one sweep of the ladder: with probability `lam` every replica takes a
within-temperature step, otherwise the pair `(a, c)` attempts an exchange. -/
noncomputable def ladderK (lam : ℝ) (P : Fin K → Fin n → Fin n → ℝ) (p : Fin K → Fin n → ℝ)
    (a c : Fin K) (x y : Fin K → Fin n) : ℝ :=
  lam * (∏ k, P k (x k) (y k)) + (1 - lam) * invK (ladderW p) (swapPair a c) x y

theorem ladderK_conserves_count {A : Finset (Fin n)} {lam : ℝ} {P : Fin K → Fin n → Fin n → ℝ}
    {p : Fin K → Fin n → ℝ} (hP : ∀ k, RE.Blocked A (P k)) (a c : Fin K)
    (x y : Fin K → Fin n) (h : cntA A y ≠ cntA A x) :
    ladderK lam P p a c x y = 0 := by
  have hswap : invK (ladderW p) (swapPair a c) x y = 0 := by
    unfold invK
    rw [if_neg, if_neg]
    · intro hyx; exact h (by rw [hyx])
    · intro hyx
      apply h
      rw [hyx, ladder_swap_conserves_count]
  rw [ladderK, ladder_within_conserves_count hP x y h, hswap, mul_zero, mul_zero, add_zero]

/-- Iterating the ladder kernel. -/
noncomputable def iter (Q : (Fin K → Fin n) → (Fin K → Fin n) → ℝ) :
    ℕ → ((Fin K → Fin n) → ℝ) → ((Fin K → Fin n) → ℝ)
  | 0, w => w
  | m + 1, w => fun y => ∑ x, iter Q m w x * Q x y

theorem ladder_iterate_support {A : Finset (Fin n)} {lam : ℝ} {P : Fin K → Fin n → Fin n → ℝ}
    {p : Fin K → Fin n → ℝ} (hP : ∀ k, RE.Blocked A (P k)) (a c : Fin K)
    {w : (Fin K → Fin n) → ℝ} {m0 : ℕ} (hw : ∀ x, w x ≠ 0 → cntA A x = m0) (m : ℕ) :
    ∀ y, cntA A y ≠ m0 → iter (ladderK lam P p a c) m w y = 0 := by
  induction m with
  | zero => intro y hy; by_contra hne; exact hy (hw y hne)
  | succ j ih =>
      intro y hy
      refine Finset.sum_eq_zero (fun x _ => ?_)
      rcases eq_or_ne (cntA A x) m0 with hx | hx
      · rw [ladderK_conserves_count hP a c x y (by rw [hx]; exact hy), mul_zero]
      · rw [ih x hx, zero_mul]

/-- **Adding rungs does not help.**  With every replica started inside `A` and no
within-temperature move crossing the boundary of `A`, no configuration outside `A` is ever
reached, at any temperature and after any number of sweeps. -/
theorem ladder_cannot_repair_ergodicity {A : Finset (Fin n)} {lam : ℝ}
    {P : Fin K → Fin n → Fin n → ℝ} {p : Fin K → Fin n → ℝ} (hP : ∀ k, RE.Blocked A (P k))
    (a c : Fin K) {w : (Fin K → Fin n) → ℝ} (hw : ∀ x, w x ≠ 0 → ∀ k, x k ∈ A) (m : ℕ) :
    ∀ (y : Fin K → Fin n) (k : Fin K), y k ∉ A → iter (ladderK lam P p a c) m w y = 0 := by
  intro y k hk
  refine ladder_iterate_support (m0 := K) hP a c (fun x hx => ?_) m y ?_
  · unfold cntA
    rw [Finset.sum_congr rfl (fun j _ => by rw [if_pos (hw x hx j)])]
    simp
  · unfold cntA
    intro hsum
    have hlt : ∑ j, (if y j ∈ A then 1 else 0) < K := by
      calc ∑ j, (if y j ∈ A then 1 else 0)
          < ∑ _j : Fin K, 1 := by
            refine Finset.sum_lt_sum (fun j _ => by split <;> omega)
              ⟨k, Finset.mem_univ k, by rw [if_neg hk]; omega⟩
        _ = K := by simp
    omega

end Ladder

end IDR
