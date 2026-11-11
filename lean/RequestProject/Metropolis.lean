/-
# Part XXVII.1  Sampling without the partition function: the Metropolis kernel

Part XXVI shows that a model of a disordered region denotes a distribution over exponentially
many conformations, and that only a factorized model has a computable configuration sum.
There is a second, weaker route to using an intractable model: never compute the
normalization at all, and generate conformations by a Markov chain whose stationary
distribution *is* the target.

This file builds that chain in the finite setting and proves the three facts that justify it.

* `mhK` -- the Metropolis kernel of a target `p` under a symmetric proposal `q`: propose,
  accept with probability `min 1 (p j / p i)`, otherwise stay.
* `mhK_detailedBalance`, `mhK_stochastic`, `mhK_stationary` -- it is a transition kernel, it
  obeys detailed balance with respect to `p`, and therefore (via
  `Kinetics.stationary_of_detailedBalance`) leaves `p` invariant.
* `mhK_smul`, `mhK_of_unnormalized`, `acc_boltzmann` -- **the partition function never
  appears**: the kernel built from the normalized Boltzmann populations and the kernel built
  from the bare weights `exp(-beta E)` are the same kernel, and the acceptance probability is
  a function of the energy *difference* alone.

That is the precise sense in which an energy function can be turned into an ensemble without
the intractable sum of Part XXVI.  What it does not buy is speed: `RequestProject.Mixing`
prices the time this chain needs.
-/
import Mathlib
import RequestProject.Kinetics

namespace IDR

open Finset

namespace Metropolis

variable {n : ℕ}

/-- The Metropolis acceptance probability for a move from `i` to `j`. -/
noncomputable def acc (p : Fin n → ℝ) (i j : Fin n) : ℝ := min 1 (p j / p i)

lemma acc_nonneg {p : Fin n → ℝ} (hp : ∀ i, 0 < p i) (i j : Fin n) : 0 ≤ acc p i j :=
  le_min zero_le_one (div_nonneg (hp j).le (hp i).le)

lemma acc_le_one (p : Fin n → ℝ) (i j : Fin n) : acc p i j ≤ 1 := min_le_left _ _

/-- The elementary identity behind detailed balance: `a · min 1 (b/a) = min a b`. -/
lemma mul_acc {a b : ℝ} (ha : 0 < a) : a * min 1 (b / a) = min a b := by
  rcases le_total a b with hab | hab
  · rw [min_eq_left (one_le_div ha |>.2 hab), mul_one, min_eq_left hab]
  · rw [min_eq_right (div_le_one ha |>.2 hab), mul_div_cancel₀ _ ha.ne', min_eq_right hab]

/-- **The Metropolis kernel** of the target `p` under the proposal `q`: an accepted proposal
moves the chain, a rejected one leaves it where it is. -/
noncomputable def mhK (p : Fin n → ℝ) (q : Fin n → Fin n → ℝ) : Fin n → Fin n → ℝ :=
  fun i j => if i = j then 1 - ∑ l ∈ Finset.univ.erase i, q i l * acc p i l
    else q i j * acc p i j

lemma mhK_off_diag {p : Fin n → ℝ} {q : Fin n → Fin n → ℝ} {i j : Fin n} (h : i ≠ j) :
    mhK p q i j = q i j * acc p i j := by
  rw [mhK, if_neg h]

/-- **Detailed balance.**  With a symmetric proposal the Metropolis kernel is reversible with
respect to its target. -/
theorem mhK_detailedBalance {p : Fin n → ℝ} {q : Fin n → Fin n → ℝ} (hp : ∀ i, 0 < p i)
    (hq : ∀ i j, q i j = q j i) : Kinetics.DetailedBalance (mhK p q) p := by
  intro i j
  rcases eq_or_ne i j with rfl | hij
  · rfl
  · rw [mhK_off_diag hij, mhK_off_diag hij.symm]
    calc p i * (q i j * acc p i j) = q i j * (p i * min 1 (p j / p i)) := by rw [acc]; ring
      _ = q i j * min (p i) (p j) := by rw [mul_acc (hp i)]
      _ = q j i * min (p j) (p i) := by rw [hq i j, min_comm]
      _ = p j * (q j i * acc p j i) := by
            rw [acc]
            rw [show p j * (q j i * min 1 (p i / p j)) = q j i * (p j * min 1 (p i / p j)) by ring,
              mul_acc (hp j)]

/-- The Metropolis kernel is a transition kernel: nonnegative, with rows summing to one (the
rejection probability is exactly what is left on the diagonal). -/
theorem mhK_stochastic {p : Fin n → ℝ} {q : Fin n → Fin n → ℝ} (hp : ∀ i, 0 < p i)
    (hq0 : ∀ i j, 0 ≤ q i j) (hq1 : ∀ i, ∑ j, q i j = 1) :
    Kinetics.IsStochastic (mhK p q) := by
  have hsplit : ∀ i : Fin n, ∑ l ∈ Finset.univ.erase i, q i l = 1 - q i i := by
    intro i
    have := Finset.add_sum_erase Finset.univ (q i) (Finset.mem_univ i)
    rw [hq1 i] at this
    linarith
  constructor
  · intro i j
    rcases eq_or_ne i j with rfl | hij
    · rw [mhK, if_pos rfl]
      have hle : ∑ l ∈ Finset.univ.erase i, q i l * acc p i l
          ≤ ∑ l ∈ Finset.univ.erase i, q i l := by
        refine Finset.sum_le_sum fun l _ => ?_
        calc q i l * acc p i l ≤ q i l * 1 :=
              mul_le_mul_of_nonneg_left (acc_le_one p i l) (hq0 i l)
          _ = q i l := mul_one _
      have := hsplit i
      have := hq0 i i
      linarith
    · rw [mhK_off_diag hij]
      exact mul_nonneg (hq0 i j) (acc_nonneg hp i j)
  · intro i
    rw [← Finset.add_sum_erase Finset.univ (mhK p q i) (Finset.mem_univ i), mhK, if_pos rfl]
    have hrest : ∑ l ∈ Finset.univ.erase i, mhK p q i l
        = ∑ l ∈ Finset.univ.erase i, q i l * acc p i l :=
      Finset.sum_congr rfl fun l hl => mhK_off_diag (Finset.ne_of_mem_erase hl).symm
    rw [hrest]
    ring

/-- **The target is stationary.**  The Metropolis chain samples the model's ensemble. -/
theorem mhK_stationary {p : Fin n → ℝ} {q : Fin n → Fin n → ℝ} (hp : ∀ i, 0 < p i)
    (hq0 : ∀ i j, 0 ≤ q i j) (hq1 : ∀ i, ∑ j, q i j = 1) (hq : ∀ i j, q i j = q j i) :
    Kinetics.Stationary (mhK p q) p :=
  Kinetics.stationary_of_detailedBalance (mhK_stochastic hp hq0 hq1)
    (mhK_detailedBalance hp hq)

/-! ### The partition function never appears -/

lemma acc_smul {p : Fin n → ℝ} {c : ℝ} (hc : 0 < c) (i j : Fin n) :
    acc (fun l => c * p l) i j = acc p i j := by
  rw [acc, acc, mul_div_mul_left _ _ hc.ne']

/-- **Rescaling the target does not change the sampler.**  Only ratios of populations enter
the Metropolis kernel. -/
theorem mhK_smul (p : Fin n → ℝ) (q : Fin n → Fin n → ℝ) {c : ℝ} (hc : 0 < c) :
    mhK (fun l => c * p l) q = mhK p q := by
  funext i j
  rw [mhK, mhK]
  simp only [acc_smul hc]

/-- **The Metropolis chain of a Boltzmann ensemble needs only the unnormalized weights.**  The
kernel built from the normalized populations `exp(-beta E)/Zc` is *the same kernel* as the one
built from `exp(-beta E)`: the intractable configuration sum of Part XXVI cancels. -/
theorem mhK_of_unnormalized (beta : ℝ) (E : Fin n → ℝ) (q : Fin n → Fin n → ℝ) {Zc : ℝ}
    (hZ : 0 < Zc) :
    mhK (fun i => Real.exp (-beta * E i) / Zc) q = mhK (fun i => Real.exp (-beta * E i)) q := by
  have : (fun i => Real.exp (-beta * E i) / Zc)
      = fun i => Zc⁻¹ * Real.exp (-beta * E i) := by
    funext i; rw [div_eq_inv_mul]
  rw [this]
  exact mhK_smul _ q (inv_pos.2 hZ)

/-- The acceptance probability depends on the energy *difference* alone. -/
theorem acc_boltzmann (beta : ℝ) (E : Fin n → ℝ) (i j : Fin n) :
    acc (fun l => Real.exp (-beta * E l)) i j = min 1 (Real.exp (-beta * (E j - E i))) := by
  rw [acc, ← Real.exp_sub]
  ring_nf

end Metropolis

end IDR
