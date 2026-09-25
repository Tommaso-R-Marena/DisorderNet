/-
# Part XXVII.2  What sampling still costs: bottlenecks

`RequestProject.Metropolis` removes the partition function from the problem: an unnormalized
energy is enough to build a chain whose stationary distribution is the model's ensemble.  It
says nothing about how long that chain must run, and this file supplies the missing bound.

* `massOut_evolve_le` -- **the bottleneck lemma**: if from every conformation of a region `S`
  the one-step probability of leaving `S` is at most `eps`, then one step of the chain
  increases the population outside `S` by at most `eps`;
* `massOut_iterate_le` -- hence after `t` steps started inside `S` the population outside `S`
  is at most `t · eps`, and `steps_needed` inverts this: reaching a population `m` outside `S`
  takes at least `m / eps` steps.

The bound is then applied to the smallest honest caricature of a rugged landscape, a
three-state chain with a barrier of height `B` between two equally populated wells:

* `barrier_escape_le` -- the escape probability from the initial well is `e^{-beta B}/2`;
* `barrier_mass_le` -- so after `t` Metropolis steps started in that well the population of
  everything else is at most `t · e^{-beta B}/2`;
* `barrier_steps_needed` -- and equilibrating the two wells (population `1/2` outside) takes
  at least `e^{beta B}` steps.

Evaluation of a factorized model is linear in the length of the region (Part XXVI); sampling
a rugged one is exponential in its barriers.  A model of a disordered region must therefore
report *which* ensemble its sampler actually produced -- the design conclusion that
`RequestProject.BrokenErgodicity` reaches from the continuous-time side, here in the discrete
Markov-chain Monte Carlo setting the models are actually run in.
-/
import Mathlib
import RequestProject.Kinetics
import RequestProject.Metropolis

namespace IDR

open Finset

namespace Mixing

variable {n : ℕ}

/-- The population outside a region `S` of conformation space. -/
noncomputable def massOut (S : Finset (Fin n)) (w : Fin n → ℝ) : ℝ :=
  ∑ j ∈ Finset.univ \ S, w j

/-- The population vector after `t` steps of the chain. -/
noncomputable def iterate (P : Fin n → Fin n → ℝ) : ℕ → (Fin n → ℝ) → (Fin n → ℝ)
  | 0, w => w
  | t + 1, w => Kinetics.evolve P (iterate P t w)

lemma evolve_nonneg {P : Fin n → Fin n → ℝ} {w : Fin n → ℝ} (hP : Kinetics.IsStochastic P)
    (hw : ∀ i, 0 ≤ w i) (j : Fin n) : 0 ≤ Kinetics.evolve P w j :=
  Finset.sum_nonneg fun i _ => mul_nonneg (hw i) (hP.1 i j)

lemma evolve_sum_one {P : Fin n → Fin n → ℝ} {w : Fin n → ℝ} (hP : Kinetics.IsStochastic P)
    (hw : ∑ i, w i = 1) : ∑ j, Kinetics.evolve P w j = 1 := by
  simp only [Kinetics.evolve]
  rw [Finset.sum_comm]
  rw [Finset.sum_congr rfl fun i (_ : i ∈ Finset.univ) => by
    rw [← Finset.mul_sum, hP.2 i, mul_one]]
  exact hw

lemma iterate_nonneg {P : Fin n → Fin n → ℝ} {w : Fin n → ℝ} (hP : Kinetics.IsStochastic P)
    (hw : ∀ i, 0 ≤ w i) : ∀ t i, 0 ≤ iterate P t w i := by
  intro t
  induction t with
  | zero => exact hw
  | succ t ih => exact fun i => evolve_nonneg hP ih i

lemma iterate_sum_one {P : Fin n → Fin n → ℝ} {w : Fin n → ℝ} (hP : Kinetics.IsStochastic P)
    (hw : ∑ i, w i = 1) : ∀ t, ∑ i, iterate P t w i = 1 := by
  intro t
  induction t with
  | zero => exact hw
  | succ t ih => exact evolve_sum_one hP ih

/-- **The bottleneck lemma.**  If the chain leaves the region `S` with probability at most
`eps` from every conformation in `S`, one step raises the population outside `S` by at most
`eps`. -/
theorem massOut_evolve_le {P : Fin n → Fin n → ℝ} {w : Fin n → ℝ} {S : Finset (Fin n)}
    {eps : ℝ} (hP : Kinetics.IsStochastic P) (hw : ∀ i, 0 ≤ w i) (hw1 : ∑ i, w i = 1)
    (heps : ∀ i ∈ S, ∑ j ∈ Finset.univ \ S, P i j ≤ eps) (heps0 : 0 ≤ eps) :
    massOut S (Kinetics.evolve P w) ≤ massOut S w + eps := by
  have hrow : ∀ i : Fin n, ∑ j ∈ Finset.univ \ S, P i j ≤ 1 := by
    intro i
    calc ∑ j ∈ Finset.univ \ S, P i j ≤ ∑ j, P i j :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
            (fun j _ _ => hP.1 i j)
      _ = 1 := hP.2 i
  have hrow0 : ∀ i : Fin n, 0 ≤ ∑ j ∈ Finset.univ \ S, P i j :=
    fun i => Finset.sum_nonneg fun j _ => hP.1 i j
  have hswap : massOut S (Kinetics.evolve P w)
      = ∑ i, w i * ∑ j ∈ Finset.univ \ S, P i j := by
    simp only [massOut, Kinetics.evolve]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ => by rw [Finset.mul_sum]
  rw [hswap, ← Finset.sum_sdiff (Finset.subset_univ S)]
  have hout : ∑ i ∈ Finset.univ \ S, w i * ∑ j ∈ Finset.univ \ S, P i j ≤ massOut S w := by
    refine Finset.sum_le_sum fun i _ => ?_
    calc w i * ∑ j ∈ Finset.univ \ S, P i j ≤ w i * 1 :=
          mul_le_mul_of_nonneg_left (hrow i) (hw i)
      _ = w i := mul_one _
  have hin : ∑ i ∈ S, w i * ∑ j ∈ Finset.univ \ S, P i j ≤ eps := by
    have h1 : ∑ i ∈ S, w i * ∑ j ∈ Finset.univ \ S, P i j ≤ ∑ i ∈ S, w i * eps :=
      Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (heps i hi) (hw i)
    have h2 : ∑ i ∈ S, w i ≤ 1 := by
      rw [← hw1]
      exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun i _ _ => hw i
    calc ∑ i ∈ S, w i * ∑ j ∈ Finset.univ \ S, P i j ≤ ∑ i ∈ S, w i * eps := h1
      _ = (∑ i ∈ S, w i) * eps := by rw [Finset.sum_mul]
      _ ≤ 1 * eps := mul_le_mul_of_nonneg_right h2 heps0
      _ = eps := one_mul _
  linarith

/-- **After `t` steps the population outside a bottlenecked region is at most `t · eps`.** -/
theorem massOut_iterate_le {P : Fin n → Fin n → ℝ} {w : Fin n → ℝ} {S : Finset (Fin n)}
    {eps : ℝ} (hP : Kinetics.IsStochastic P) (hw : ∀ i, 0 ≤ w i) (hw1 : ∑ i, w i = 1)
    (heps : ∀ i ∈ S, ∑ j ∈ Finset.univ \ S, P i j ≤ eps) (heps0 : 0 ≤ eps) :
    ∀ t : ℕ, massOut S (iterate P t w) ≤ massOut S w + t * eps := by
  intro t
  induction t with
  | zero => simp [iterate]
  | succ t ih =>
      have hstep := massOut_evolve_le (P := P) (w := iterate P t w) (S := S) hP
        (iterate_nonneg hP hw t) (iterate_sum_one hP hw1 t) heps heps0
      have : massOut S (iterate P (t + 1) w) ≤ massOut S (iterate P t w) + eps := hstep
      push_cast
      linarith

/-- **How long escaping a bottleneck takes.**  Starting inside `S`, reaching a population `m`
outside it requires at least `m / eps` steps. -/
theorem steps_needed {P : Fin n → Fin n → ℝ} {w : Fin n → ℝ} {S : Finset (Fin n)}
    {eps m : ℝ} (hP : Kinetics.IsStochastic P) (hw : ∀ i, 0 ≤ w i) (hw1 : ∑ i, w i = 1)
    (hstart : massOut S w = 0) (heps : ∀ i ∈ S, ∑ j ∈ Finset.univ \ S, P i j ≤ eps)
    (heps0 : 0 < eps) {t : ℕ} (hm : m ≤ massOut S (iterate P t w)) :
    m / eps ≤ t := by
  have h := massOut_iterate_le hP hw hw1 heps heps0.le t
  rw [hstart, zero_add] at h
  rw [div_le_iff₀ heps0]
  linarith

/-! ### A rugged landscape: two wells separated by a barrier -/

/-- Energies of the three-state landscape: two degenerate wells separated by a barrier of
height `B`. -/
noncomputable def barrierE (B : ℝ) : Fin 3 → ℝ := ![0, B, 0]

/-- The nearest-neighbour proposal: the chain may only step to an adjacent state. -/
noncomputable def barrierProp : Fin 3 → Fin 3 → ℝ :=
  fun i j => !![1/2, 1/2, 0; 1/2, 0, 1/2; 0, 1/2, 1/2] i j

lemma barrierProp_nonneg (i j : Fin 3) : 0 ≤ barrierProp i j := by
  fin_cases i <;> fin_cases j <;> norm_num [barrierProp]

lemma barrierProp_symm (i j : Fin 3) : barrierProp i j = barrierProp j i := by
  fin_cases i <;> fin_cases j <;>
    simp [barrierProp, Matrix.cons_val_two, Matrix.tail_cons]

lemma barrierProp_row (i : Fin 3) : ∑ j, barrierProp i j = 1 := by
  fin_cases i <;>
    simp [barrierProp, Fin.sum_univ_three, Matrix.cons_val_two, Matrix.tail_cons] <;> norm_num

/-- The unnormalized Boltzmann weights of the landscape. -/
noncomputable def barrierWeight (beta B : ℝ) : Fin 3 → ℝ :=
  fun i => Real.exp (-beta * barrierE B i)

lemma barrierWeight_pos (beta B : ℝ) (i : Fin 3) : 0 < barrierWeight beta B i :=
  Real.exp_pos _

/-- The Metropolis chain of the barrier landscape. -/
noncomputable def barrierK (beta B : ℝ) : Fin 3 → Fin 3 → ℝ :=
  Metropolis.mhK (barrierWeight beta B) barrierProp

lemma barrierK_stochastic (beta B : ℝ) : Kinetics.IsStochastic (barrierK beta B) :=
  Metropolis.mhK_stochastic (barrierWeight_pos beta B) barrierProp_nonneg barrierProp_row

/-- **The escape probability from the initial well is `e^{-beta B}/2`.** -/
theorem barrier_escape_le {beta B : ℝ} (hbeta : 0 ≤ beta) (hB : 0 ≤ B) :
    ∀ i ∈ ({0} : Finset (Fin 3)),
      ∑ j ∈ Finset.univ \ ({0} : Finset (Fin 3)), barrierK beta B i j
        ≤ Real.exp (-beta * B) / 2 := by
  intro i hi
  have hi0 : i = 0 := Finset.mem_singleton.1 hi
  subst hi0
  have hset : (Finset.univ \ ({0} : Finset (Fin 3))) = {1, 2} := by decide
  have h01 : barrierK beta B 0 1 = Real.exp (-beta * B) / 2 := by
    rw [barrierK, Metropolis.mhK_off_diag (by decide : (0 : Fin 3) ≠ 1), Metropolis.acc,
      barrierWeight, barrierWeight]
    have hE0 : barrierE B 0 = 0 := by simp [barrierE]
    have hE1 : barrierE B 1 = B := by simp [barrierE]
    rw [hE0, hE1]
    have : Real.exp (-beta * 0) = 1 := by simp
    rw [this, div_one]
    have hle : Real.exp (-beta * B) ≤ 1 := by
      apply Real.exp_le_one_iff.2
      nlinarith
    rw [min_eq_right hle]
    norm_num [barrierProp]
    ring
  have h02 : barrierK beta B 0 2 = 0 := by
    rw [barrierK, Metropolis.mhK_off_diag (by decide : (0 : Fin 3) ≠ 2)]
    have : barrierProp 0 2 = 0 := by
      simp [barrierProp, Matrix.cons_val_two, Matrix.tail_cons]
    rw [this, zero_mul]
  rw [hset]
  rw [Finset.sum_pair (by decide : (1 : Fin 3) ≠ 2), h01, h02, add_zero]

/-- **After `t` Metropolis steps started in one well, the population of the rest of the
landscape is at most `t · e^{-beta B}/2`.** -/
theorem barrier_mass_le {beta B : ℝ} (hbeta : 0 ≤ beta) (hB : 0 ≤ B) (t : ℕ) :
    massOut {0} (iterate (barrierK beta B) t (fun i => if i = 0 then 1 else 0))
      ≤ t * (Real.exp (-beta * B) / 2) := by
  have hw : ∀ i : Fin 3, 0 ≤ (if i = 0 then (1:ℝ) else 0) := by
    intro i; split <;> norm_num
  have hw1 : ∑ i : Fin 3, (if i = 0 then (1:ℝ) else 0) = 1 := by simp
  have hstart : massOut ({0} : Finset (Fin 3)) (fun i => if i = 0 then (1:ℝ) else 0) = 0 := by
    have hset : (Finset.univ \ ({0} : Finset (Fin 3))) = {1, 2} := by decide
    rw [massOut, hset, Finset.sum_pair (by decide : (1 : Fin 3) ≠ 2)]
    norm_num [show (1 : Fin 3) ≠ 0 by decide, show (2 : Fin 3) ≠ 0 by decide]
  have h := massOut_iterate_le (P := barrierK beta B)
    (w := fun i => if i = 0 then (1:ℝ) else 0) (S := {0})
    (barrierK_stochastic beta B) hw hw1 (barrier_escape_le hbeta hB)
    (by positivity) t
  rw [hstart, zero_add] at h
  exact h

/-- **Equilibrating across the barrier takes at least `e^{beta B}` Metropolis steps.**  The
sampling time is exponential in the barrier height, however cheap each step is. -/
theorem barrier_steps_needed {beta B : ℝ} (hbeta : 0 ≤ beta) (hB : 0 ≤ B) {t : ℕ}
    (hhalf : (1:ℝ)/2 ≤ massOut {0} (iterate (barrierK beta B) t
      (fun i => if i = 0 then 1 else 0))) :
    Real.exp (beta * B) ≤ t := by
  have h := barrier_mass_le hbeta hB t
  have hpos : (0:ℝ) < Real.exp (-beta * B) := Real.exp_pos _
  have h2 : (1:ℝ)/2 ≤ t * (Real.exp (-beta * B) / 2) := le_trans hhalf h
  have h3 : 1 ≤ t * Real.exp (-beta * B) := by linarith
  have hinv : Real.exp (-beta * B) * Real.exp (beta * B) = 1 := by
    rw [← Real.exp_add]
    simp
  nlinarith [Real.exp_pos (beta * B), Real.exp_pos (-beta * B)]

end Mixing

end IDR
