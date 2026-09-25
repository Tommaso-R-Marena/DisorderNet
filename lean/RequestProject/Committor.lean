/-
# Part XLIX.1  The committor: the mechanism is not in the landscape either

`RequestProject.FirstPassage` shows that the equilibrium profile of a coordinate does not fix
the rate.  This file shows that it does not fix the *mechanism* either.  The object that
defines mechanism in a hopping model is the **committor**: `q i` is the probability that a chain
started at state `i` reaches the product end `n` before returning to the reactant end `0`, and
"the transition state" is by definition the place where `q = 1/2`.

The setting is the one of `RequestProject.FirstPassage`: states `0, ..., n` along a reaction
coordinate, forward rates `kp`, backward rates `km`, equilibrium weights `p` obeying detailed
balance, but now with *both* ends absorbing.  `IsCommittor` is the standard harmonic system for
`q` (first-step analysis with `q 0 = 0`, `q n = 1`).

* `res`, `resSum` -- the local "resistance" `1/(p k · kp k)` and its partial sums; the committor
  is a ratio of resistances, exactly as in the electrical analogy.
* `committorFun_isCommittor`, `committor_unique` -- **the closed form**
  `q i = (Σ_{k<i} 1/(p k · kp k)) / (Σ_{k<n} 1/(p k · kp k))`
  is a solution of the harmonic system, and the only one.  The proof is the constancy of the
  reactive flux `p i · kp i · (q (i+1) - q i)`, which is where detailed balance enters.
* `committor_strictMono`, `committor_mem_Icc` -- the committor increases strictly along the
  coordinate and lies in `[0, 1]`; `committor_zero`, `committor_last` give the boundary values.
* `committor_uniform_rate` -- **when, and only when, the landscape is enough**: if the forward
  rate is the same at every step (a constant diffusion coefficient) the committor is a functional
  of the profile alone, `q i = (Σ_{k<i} 1/p k) / (Σ_{k<n} 1/p k)`, dominated by the states of
  lowest equilibrium weight -- this is the sense in which "the transition state is the top of the
  barrier" is true.
* `mechanism_not_determined_by_landscape` -- and it is exactly the sense in which it is false in
  general: on one and the same flat landscape, two rate profiles differing only in the second
  step put the transition state at `q 1 = 1/2` and at `q 1 = 1/3`.  Which conformations are
  "committed" is not read off the free-energy profile.

Read as a design constraint: a model of a disordered region that reports a free-energy landscape
along a coordinate has not thereby reported a mechanism.  Mechanistic statements -- transition
states, commitment, pathway -- require the kinetic profile as an independent input, exactly as
rates do.
-/
import Mathlib
import RequestProject.FirstPassage

set_option autoImplicit false

namespace IDR

namespace Committor

open Finset
open IDR.FirstPassage

/-- The local resistance of step `k`: the reciprocal of the equilibrium flux capacity
`p k · kp k`. -/
noncomputable def res (p kp : ℕ → ℝ) (k : ℕ) : ℝ := 1 / (p k * kp k)

/-- The total resistance of the first `i` steps. -/
noncomputable def resSum (p kp : ℕ → ℝ) (i : ℕ) : ℝ := ∑ k ∈ range i, res p kp k

lemma res_pos {p kp : ℕ → ℝ} {k : ℕ} (hp : 0 < p k) (hkp : 0 < kp k) : 0 < res p kp k := by
  rw [res]
  exact div_pos one_pos (mul_pos hp hkp)

lemma resSum_succ (p kp : ℕ → ℝ) (i : ℕ) :
    resSum p kp (i + 1) = resSum p kp i + res p kp i := by
  simp [resSum, Finset.sum_range_succ]

lemma resSum_pos {n : ℕ} {p kp : ℕ → ℝ} (hn : 0 < n) (hp : ∀ i, i ≤ n → 0 < p i)
    (hkp : ∀ i, i < n → 0 < kp i) : 0 < resSum p kp n := by
  refine Finset.sum_pos (fun k hk => ?_) ⟨0, by simpa using hn⟩
  have hk' : k < n := Finset.mem_range.mp hk
  exact res_pos (hp k hk'.le) (hkp k hk')

lemma resSum_strictMonoOn {n : ℕ} {p kp : ℕ → ℝ} (hp : ∀ i, i ≤ n → 0 < p i)
    (hkp : ∀ i, i < n → 0 < kp i) {i j : ℕ} (hij : i < j) (hj : j ≤ n) :
    resSum p kp i < resSum p kp j := by
  have hstep : ∀ k, k < n → resSum p kp k < resSum p kp (k + 1) := by
    intro k hk
    rw [resSum_succ]
    have := res_pos (hp k hk.le) (hkp k hk)
    linarith
  induction j with
  | zero => omega
  | succ j ih =>
      rcases Nat.lt_succ_iff_lt_or_eq.mp hij with h | h
      · exact lt_trans (ih h (by omega)) (hstep j (by omega))
      · subst h
        exact hstep i (by omega)

/-- The closed-form committor: the fraction of the total resistance lying below `i`. -/
noncomputable def committorFun (n : ℕ) (p kp : ℕ → ℝ) (i : ℕ) : ℝ :=
  resSum p kp i / resSum p kp n

/-- The harmonic system defining the committor of a hopping chain with both ends absorbing:
`q 0 = 0`, `q n = 1`, and at every interior state the value is the mean of the neighbouring
values weighted by the jump rates. -/
def IsCommittor (n : ℕ) (kp km q : ℕ → ℝ) : Prop :=
  q 0 = 0 ∧ q n = 1 ∧
    ∀ i, 0 < i → i < n → (kp i + km i) * q i = kp i * q (i + 1) + km i * q (i - 1)

lemma committor_zero (n : ℕ) (p kp : ℕ → ℝ) : committorFun n p kp 0 = 0 := by
  simp [committorFun, resSum]

lemma committor_last {n : ℕ} {p kp : ℕ → ℝ} (hn : 0 < n) (hp : ∀ i, i ≤ n → 0 < p i)
    (hkp : ∀ i, i < n → 0 < kp i) : committorFun n p kp n = 1 :=
  div_self (ne_of_gt (resSum_pos hn hp hkp))

/-- **The closed form solves the harmonic system**, so the committor exists. -/
theorem committorFun_isCommittor {n : ℕ} {p kp km : ℕ → ℝ} (hn : 0 < n)
    (hdb : DetailedBalance n p kp km) (hp : ∀ i, i ≤ n → 0 < p i) (hkp : ∀ i, i < n → 0 < kp i) :
    IsCommittor n kp km (committorFun n p kp) := by
  refine ⟨committor_zero n p kp, committor_last hn hp hkp, ?_⟩
  intro i hi0 hi
  obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
  have hj : j < n := by omega
  have hS : resSum p kp n ≠ 0 := ne_of_gt (resSum_pos hn hp hkp)
  have hpj : (0:ℝ) < p j := hp j hj.le
  have hkpj : (0:ℝ) < kp j := hkp j hj
  have hpi : (0:ℝ) < p (j + 1) := hp (j + 1) hi.le
  have hkpi : (0:ℝ) < kp (j + 1) := hkp (j + 1) hi
  have hdbj : p j * kp j = p (j + 1) * km (j + 1) := hdb j hj
  have hsub : (j + 1) - 1 = j := by omega
  rw [hsub]
  -- the increments of the closed form
  have e1 : committorFun n p kp (j + 1 + 1) - committorFun n p kp (j + 1)
      = res p kp (j + 1) / resSum p kp n := by
    rw [committorFun, committorFun, div_sub_div_same, resSum_succ]
    ring_nf
  have e2 : committorFun n p kp (j + 1) - committorFun n p kp j
      = res p kp j / resSum p kp n := by
    rw [committorFun, committorFun, div_sub_div_same, resSum_succ]
    ring_nf
  -- the reactive flux is the same across both bonds
  have hflux : kp (j + 1) * (res p kp (j + 1) / resSum p kp n)
      = km (j + 1) * (res p kp j / resSum p kp n) := by
    have hkm : km (j + 1) = p j * kp j / p (j + 1) := by
      field_simp [hdbj]
      linarith [hdbj]
    rw [hkm, res, res]
    field_simp
  linear_combination (-(kp (j + 1))) * e1 + km (j + 1) * e2 - hflux

/-- The reactive flux `p i · kp i · (q (i+1) - q i)` of any solution of the harmonic system is
independent of `i`: the electrical analogy, with detailed balance as Kirchhoff's law. -/
lemma flux_const {n : ℕ} {p kp km q : ℕ → ℝ} (hdb : DetailedBalance n p kp km)
    (hq : IsCommittor n kp km q) :
    ∀ i, i < n → p i * kp i * (q (i + 1) - q i) = p 0 * kp 0 * (q 1 - q 0) := by
  intro i
  induction i with
  | zero => intro _; rfl
  | succ j ih =>
      intro hi
      have hj : j < n := by omega
      have hIH := ih hj
      have heq := hq.2.2 (j + 1) (by omega) hi
      have hsub : (j + 1) - 1 = j := by omega
      rw [hsub] at heq
      have hdbj : p j * kp j = p (j + 1) * km (j + 1) := hdb j hj
      linear_combination (-(p (j + 1))) * heq + hIH - (q (j + 1) - q j) * hdbj

/-- **Uniqueness: every solution of the harmonic system is the closed form.** -/
theorem committor_unique {n : ℕ} {p kp km q : ℕ → ℝ} (hn : 0 < n)
    (hdb : DetailedBalance n p kp km) (hp : ∀ i, i ≤ n → 0 < p i) (hkp : ∀ i, i < n → 0 < kp i)
    (hq : IsCommittor n kp km q) : ∀ i, i ≤ n → q i = committorFun n p kp i := by
  have hflux := flux_const hdb hq
  set J : ℝ := p 0 * kp 0 * (q 1 - q 0) with hJ
  -- every increment is `J` times the local resistance
  have hincr : ∀ i, i < n → q (i + 1) - q i = J * res p kp i := by
    intro i hi
    have hne : p i * kp i ≠ 0 := ne_of_gt (mul_pos (hp i hi.le) (hkp i hi))
    have h := hflux i hi
    rw [res, mul_one_div, eq_div_iff hne]
    linear_combination h
  -- hence `q i = J * resSum i`
  have hsum : ∀ i, i ≤ n → q i = J * resSum p kp i := by
    intro i
    induction i with
    | zero => intro _; simp [hq.1, resSum]
    | succ j ih =>
        intro hj
        have hjn : j < n := by omega
        have h1 : q j = J * resSum p kp j := ih (by omega)
        have h2 := hincr j hjn
        rw [resSum_succ]
        linarith [h1, h2]
  -- the boundary condition fixes `J`
  have hSpos : 0 < resSum p kp n := resSum_pos hn hp hkp
  have hJval : J = 1 / resSum p kp n := by
    have h := hsum n le_rfl
    rw [hq.2.1] at h
    rw [eq_div_iff (ne_of_gt hSpos)]
    linear_combination -h
  intro i hi
  rw [hsum i hi, hJval, committorFun]
  ring

/-- The committor is strictly increasing along the coordinate. -/
theorem committor_strictMono {n : ℕ} {p kp : ℕ → ℝ} (hn : 0 < n) (hp : ∀ i, i ≤ n → 0 < p i)
    (hkp : ∀ i, i < n → 0 < kp i) {i j : ℕ} (hij : i < j) (hj : j ≤ n) :
    committorFun n p kp i < committorFun n p kp j := by
  have hS : 0 < resSum p kp n := resSum_pos hn hp hkp
  have hlt : resSum p kp i < resSum p kp j := resSum_strictMonoOn hp hkp hij hj
  rw [committorFun, committorFun]
  gcongr

/-- The committor takes values in `[0, 1]`. -/
theorem committor_mem_Icc {n : ℕ} {p kp : ℕ → ℝ} (hn : 0 < n) (hp : ∀ i, i ≤ n → 0 < p i)
    (hkp : ∀ i, i < n → 0 < kp i) {i : ℕ} (hi : i ≤ n) :
    0 ≤ committorFun n p kp i ∧ committorFun n p kp i ≤ 1 := by
  have hS : 0 < resSum p kp n := resSum_pos hn hp hkp
  have hmono : resSum p kp i ≤ resSum p kp n := by
    rcases eq_or_lt_of_le hi with h | h
    · simp [h]
    · exact (resSum_strictMonoOn hp hkp h le_rfl).le
  have hnn : 0 ≤ resSum p kp i := by
    refine Finset.sum_nonneg (fun k hk => ?_)
    have hk' : k < i := Finset.mem_range.mp hk
    exact (res_pos (hp k (by omega)) (hkp k (by omega))).le
  constructor
  · exact div_nonneg hnn hS.le
  · rw [committorFun, div_le_one hS]
    exact hmono

/-- **With a constant diffusion coefficient the landscape does determine the mechanism.**  If
the forward rate is the same at every step, the committor is the resistance profile of the
equilibrium weights alone, dominated by the states of lowest weight -- the barrier. -/
theorem committor_uniform_rate {n : ℕ} {p : ℕ → ℝ} {D : ℝ} (hD : D ≠ 0) (i : ℕ) :
    committorFun n p (fun _ => D) i
      = (∑ k ∈ range i, 1 / p k) / (∑ k ∈ range n, 1 / p k) := by
  rw [committorFun, resSum, resSum]
  simp only [res]
  have hi : (∑ k ∈ range i, 1 / (p k * D)) = (∑ k ∈ range i, 1 / p k) / D := by
    rw [Finset.sum_div]
    exact Finset.sum_congr rfl (fun k _ => by field_simp)
  have hnn : (∑ k ∈ range n, 1 / (p k * D)) = (∑ k ∈ range n, 1 / p k) / D := by
    rw [Finset.sum_div]
    exact Finset.sum_congr rfl (fun k _ => by field_simp)
  rw [hi, hnn, div_div_div_cancel_right₀]
  exact hD

/-- **The mechanism is not determined by the landscape.**  On the flat three-state landscape of
`RequestProject.FirstPassage`, the uniform rate profile puts the transition state exactly at the
middle state (`q 1 = 1/2`), while the profile whose second step is twice as slow gives
`q 1 = 1/3`: the same free energies, the same populations, a different committor. -/
theorem mechanism_not_determined_by_landscape :
    committorFun 2 flatP fastKp 1 = 1/2 ∧ committorFun 2 flatP slowKp 1 = 1/3 := by
  constructor
  · rw [committorFun]
    norm_num [resSum, res, Finset.sum_range_succ, flatP, fastKp]
  · rw [committorFun]
    norm_num [resSum, res, Finset.sum_range_succ, flatP, slowKp]

end Committor

end IDR
