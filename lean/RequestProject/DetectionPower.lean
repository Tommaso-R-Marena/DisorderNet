/-
# Part LXXXIX.1  How many measurements the capacity prediction needs: a power calculation

`RequestProject.CapacityExact` proves *what* an under-capacity model gets wrong: it assigns
population zero to an identifiable set of conformations carrying at least `tail P k` of the
true population (`IDR.Capacity.missed_states_of_under_capacity`).  That is a prediction about
the world, but by itself it is not yet an experiment: an experiment observes finitely many
molecules, and the question a referee asks first is *how many*.

This file answers that question exactly, so that the pre-registered test of the capacity law
can state its sample size in advance rather than after the fact.

* `missProb_eq` — the probability that `n` independent draws from the true ensemble all avoid
  the missed set is exactly `(1 - τ)ⁿ`, where `τ` is the true population of that set.
* `missProb_le_exp` — hence at most `exp(-nτ)`.
* `samplesFor`, `samplesFor_spec` — `⌈log(1/α)/τ⌉` draws suffice to drive that probability
  below `α`.  This is the number to write into the protocol before any data are taken.
* `likelihood_zero_of_hit` — a *single* observation inside the missed set refutes the
  under-capacity model outright: the model's likelihood for that data set is exactly `0`, not
  merely small.  The test is therefore a clean refutation, not a threshold on a fit statistic.
* `detection_power` — the capstone: for any model with at most `k < m` components there is a
  set of conformations the model calls unoccupied, which the truth occupies with probability
  at least `tail P k`, and `samplesFor α (tail P k)` independent observations of the real
  system land in it — refuting the model — with probability at least `1 - α`.
* `missProb_ge_one_sub` — the honest converse (Bernoulli): with `n` draws the probability of
  seeing nothing is at least `1 - nτ`, so a run shorter than the stated size is not evidence
  for the model.  Under-powered non-observation must not be reported as confirmation.

Nothing here is a claim about any real system: `τ` is whatever the independent measurement of
the populations says it is.  What is proved is the bridge from the population profile to a
sample size, which is the piece a data run needs and a proof can actually supply.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Metric
import RequestProject.CapacityExact

set_option autoImplicit false

namespace IDR
namespace Power

open Finset
open scoped Classical

variable {X : Type*} [Fintype X] [DecidableEq X]

/-! ## The law of `n` independent observations -/

/-- The probability of observing the sequence of conformations `s` in `n` independent draws
from the ensemble `E`. -/
noncomputable def pathProb (E : Ens X) {n : ℕ} (s : Fin n → X) : ℝ := ∏ i, E.prob (s i)

omit [Fintype X] [DecidableEq X] in
lemma pathProb_nonneg (E : Ens X) {n : ℕ} (s : Fin n → X) : 0 ≤ pathProb E s :=
  Finset.prod_nonneg fun i _ => E.prob_nonneg (s i)

/-- The sequences of `n` observations that never enter the set `A`. -/
def avoiding (A : Finset X) (n : ℕ) : Finset (Fin n → X) :=
  Fintype.piFinset (fun _ => Aᶜ)

/-- The probability that `n` independent draws from `E` all avoid `A`. -/
noncomputable def missProb (E : Ens X) (A : Finset X) (n : ℕ) : ℝ :=
  ∑ s ∈ avoiding A n, pathProb E s

/-- The probability that `n` independent draws from `E` enter `A` at least once. -/
noncomputable def hitProb (E : Ens X) (A : Finset X) (n : ℕ) : ℝ :=
  ∑ s ∈ Finset.univ \ avoiding A n, pathProb E s

omit [DecidableEq X] in
lemma pathProb_sum_one (E : Ens X) (n : ℕ) : ∑ s : Fin n → X, pathProb E s = 1 := by
  classical
  have h : ∑ s ∈ Fintype.piFinset (fun _ : Fin n => (Finset.univ : Finset X)),
      ∏ i, E.prob (s i) = (∑ x, E.prob x) ^ n := by
    rw [← Finset.prod_univ_sum (fun _ : Fin n => (Finset.univ : Finset X))
      (fun _ x => E.prob x)]
    simp
  simp only [pathProb]
  simpa [Fintype.piFinset_univ, E.sum_prob] using h

/-- **The miss probability is exactly `(1 - τ)ⁿ`**, where `τ` is the population of `A`. -/
theorem missProb_eq (E : Ens X) (A : Finset X) (n : ℕ) :
    missProb E A n = (1 - ∑ x ∈ A, E.prob x) ^ n := by
  classical
  have hcompl : ∑ x ∈ Aᶜ, E.prob x = 1 - ∑ x ∈ A, E.prob x := by
    have := Finset.sum_add_sum_compl A E.prob
    rw [E.sum_prob] at this
    linarith
  have h : ∑ s ∈ Fintype.piFinset (fun _ : Fin n => Aᶜ), ∏ i, E.prob (s i)
      = (∑ x ∈ Aᶜ, E.prob x) ^ n := by
    rw [← Finset.prod_univ_sum (fun _ : Fin n => Aᶜ) (fun _ x => E.prob x)]
    simp
  simp only [missProb, avoiding, pathProb]
  rw [h, hcompl]

lemma missProb_nonneg (E : Ens X) (A : Finset X) (n : ℕ) : 0 ≤ missProb E A n :=
  Finset.sum_nonneg fun s _ => pathProb_nonneg E s

/-- Hitting and missing are complementary. -/
theorem hitProb_eq (E : Ens X) (A : Finset X) (n : ℕ) :
    hitProb E A n = 1 - missProb E A n := by
  classical
  have hsub : avoiding A n ⊆ (Finset.univ : Finset (Fin n → X)) := Finset.subset_univ _
  have := Finset.sum_sdiff (f := fun s : Fin n → X => pathProb E s) hsub
  rw [pathProb_sum_one E n] at this
  rw [hitProb, missProb]
  linarith

omit [DecidableEq X] in
/-- The population of a subset never exceeds one. -/
lemma mass_le_one (E : Ens X) (A : Finset X) : ∑ x ∈ A, E.prob x ≤ 1 := by
  classical
  have h := Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ A)
    (fun x _ _ => E.prob_nonneg x)
  simpa [E.sum_prob] using h

omit [Fintype X] [DecidableEq X] in
lemma mass_nonneg (E : Ens X) (A : Finset X) : 0 ≤ ∑ x ∈ A, E.prob x :=
  Finset.sum_nonneg fun x _ => E.prob_nonneg x

/-! ## The exponential bound and the sample size -/

lemma pow_one_sub_le_exp {tau : ℝ} (h1 : tau ≤ 1) (n : ℕ) :
    (1 - tau) ^ n ≤ Real.exp (-((n : ℝ) * tau)) := by
  calc (1 - tau) ^ n ≤ (Real.exp (-tau)) ^ n := by
        gcongr
        · linarith
        · linarith [Real.add_one_le_exp (-tau)]
    _ = Real.exp (-((n : ℝ) * tau)) := by rw [← Real.exp_nat_mul]; ring_nf

/-- **Missing the missed set is exponentially unlikely.** -/
theorem missProb_le_exp (E : Ens X) (A : Finset X) (n : ℕ) :
    missProb E A n ≤ Real.exp (-((n : ℝ) * ∑ x ∈ A, E.prob x)) := by
  rw [missProb_eq]
  exact pow_one_sub_le_exp (mass_le_one E A) n

/-- **The pre-registered sample size.**  To refute an under-capacity model at significance
`alpha`, given that the states it omits carry population `tau`, this many independent
observations suffice. -/
noncomputable def samplesFor (alpha tau : ℝ) : ℕ := ⌈Real.log (1 / alpha) / tau⌉₊

/-- **The sample size does what it says.** -/
theorem samplesFor_spec {alpha tau : ℝ} (ha : 0 < alpha) (htau : 0 < tau) (htau1 : tau ≤ 1)
    {n : ℕ} (hn : samplesFor alpha tau ≤ n) : (1 - tau) ^ n ≤ alpha := by
  have hstep : Real.log (1 / alpha) / tau ≤ (n : ℝ) :=
    le_trans (Nat.le_ceil _) (by exact_mod_cast hn)
  have hlog : Real.log (1 / alpha) ≤ (n : ℝ) * tau := by
    rw [div_le_iff₀ htau] at hstep; exact hstep
  have h1 : (1 - tau) ^ n ≤ Real.exp (-((n : ℝ) * tau)) :=
    pow_one_sub_le_exp htau1 n
  have h2 : Real.exp (-((n : ℝ) * tau)) ≤ alpha := by
    have hmono : Real.exp (-((n : ℝ) * tau)) ≤ Real.exp (-Real.log (1 / alpha)) :=
      Real.exp_le_exp.2 (by linarith)
    rwa [← Real.log_inv, one_div, inv_inv, Real.exp_log ha] at hmono
  linarith

/-- A more populated missed set is missed less often. -/
lemma missProb_le_of_mass_ge (E : Ens X) (A : Finset X) {tau : ℝ}
    (hmass : tau ≤ ∑ x ∈ A, E.prob x) (n : ℕ) : missProb E A n ≤ (1 - tau) ^ n := by
  rw [missProb_eq]
  have hle : 1 - ∑ x ∈ A, E.prob x ≤ 1 - tau := by linarith
  have hnn : 0 ≤ 1 - ∑ x ∈ A, E.prob x := by linarith [mass_le_one E A]
  gcongr

/-! ## One observation is a refutation, not a nudge -/

omit [Fintype X] [DecidableEq X] in
/-- **A single observation inside the missed set gives the model likelihood exactly zero.**
The model does not merely fit badly; it assigns probability `0` to data that occurred. -/
theorem likelihood_zero_of_hit {M : Ens X} {A : Finset X} (hA : ∀ x ∈ A, M.prob x = 0)
    {n : ℕ} {s : Fin n → X} {i : Fin n} (hi : s i ∈ A) : pathProb M s = 0 := by
  classical
  exact Finset.prod_eq_zero (Finset.mem_univ i) (hA (s i) hi)

/-- Every sequence outside `avoiding A n` visits `A`. -/
lemma exists_mem_of_not_avoiding {A : Finset X} {n : ℕ} {s : Fin n → X}
    (hs : s ∉ avoiding A n) : ∃ i, s i ∈ A := by
  classical
  by_contra hcon
  push_neg at hcon
  exact hs (Fintype.mem_piFinset.2 fun i => Finset.mem_compl.2 (hcon i))

/-- Every sequence that refutes the model has likelihood zero under it. -/
theorem likelihood_zero_off_avoiding {M : Ens X} {A : Finset X} (hA : ∀ x ∈ A, M.prob x = 0)
    {n : ℕ} {s : Fin n → X} (hs : s ∉ avoiding A n) : pathProb M s = 0 := by
  obtain ⟨i, hi⟩ := exists_mem_of_not_avoiding hs
  exact likelihood_zero_of_hit hA hi

/-! ## The capstone: the power of the pre-registered test -/

/-- **The power calculation for the capacity test.**  Let a target populate `m` states with
profile `P`, and let `M` be *any* model with at most `k < m` components — for instance the
practitioner's fixed three-component mixture on a system with more than three populated
states.  Then there is a set `A` of conformations such that

1. the model assigns population zero to every conformation in `A`;
2. the true ensemble occupies `A` with probability at least `tail P k`;
3. observing any conformation of `A` even once makes the model's likelihood exactly zero; and
4. `samplesFor alpha (tail P k)` independent observations of the real system fail to refute
   the model with probability at most `alpha`.

Item 4 is the number a protocol needs: it is fixed by the independently measured populations
and the chosen significance level, before any model is trained. -/
theorem detection_power {m k n : ℕ} (P : Capacity.Profile m) {g : Fin m → X}
    (hg : Function.Injective g) {M : Ens X} (hM : M.card ≤ k) (hk : k < m)
    {alpha : ℝ} (ha : 0 < alpha) (hn : samplesFor alpha (P.tail k) ≤ n) :
    ∃ A : Finset X,
      (∀ x ∈ A, M.prob x = 0) ∧
      P.tail k ≤ ∑ x ∈ A, (Capacity.target P g).prob x ∧
      (∀ s : Fin n → X, s ∉ avoiding A n → pathProb M s = 0) ∧
      missProb (Capacity.target P g) A n ≤ alpha ∧
      1 - alpha ≤ hitProb (Capacity.target P g) A n := by
  classical
  obtain ⟨A, hA0, hAmass⟩ := Capacity.missed_states_of_under_capacity P hg hM
  have htau_pos : 0 < P.tail k := P.tail_pos hk
  have htau_le : P.tail k ≤ 1 := le_trans hAmass (mass_le_one _ A)
  have hmiss : missProb (Capacity.target P g) A n ≤ alpha := by
    refine le_trans (missProb_le_of_mass_ge _ A hAmass n) ?_
    exact samplesFor_spec ha htau_pos htau_le hn
  refine ⟨A, hA0, hAmass, fun s hs => likelihood_zero_off_avoiding hA0 hs, hmiss, ?_⟩
  rw [hitProb_eq]
  linarith

/-! ## The honest converse: an under-powered run is not evidence -/

/-- **Bernoulli's inequality as a warning.**  With `n` draws, the probability of never seeing
the missed set is at least `1 - n·τ`.  A run with `n ≪ 1/τ` therefore misses it most of the
time even though the model is wrong, so non-observation at that sample size must not be
reported as support for the model. -/
theorem missProb_ge_one_sub (E : Ens X) (A : Finset X) (n : ℕ) :
    1 - (n : ℝ) * (∑ x ∈ A, E.prob x) ≤ missProb E A n := by
  have hle : ∑ x ∈ A, E.prob x ≤ 1 := mass_le_one E A
  have h0 : 0 ≤ ∑ x ∈ A, E.prob x := mass_nonneg E A
  have hb := one_add_mul_le_pow (a := -(∑ x ∈ A, E.prob x)) (by linarith) n
  rw [missProb_eq]
  have : (1 : ℝ) + -(∑ x ∈ A, E.prob x) = 1 - ∑ x ∈ A, E.prob x := by ring
  rw [this] at hb
  linarith [hb]

end Power
end IDR
