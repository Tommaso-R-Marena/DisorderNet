/-
# Part XLVIII.1  The landscape does not determine the rate: exact first-passage times

Every account of a disordered region that speaks of "the free-energy landscape" is using a
one-dimensional profile: a reaction coordinate (an end-to-end distance, a helicity, a FRET
efficiency) and the equilibrium weights `p i` along it.  The profile is what an experiment or a
simulation reports, and it is routinely read as if it fixed the kinetics -- as if the height of
the barrier were the rate.  This file settles what the profile does and does not determine, by
solving the associated hopping dynamics exactly.

Setting.  States `0, 1, ..., n` along the coordinate; the chain hops `i -> i+1` at rate `kp i`
and `i -> i-1` at rate `km i`, with `km 0 = 0` (the coordinate is reflected at its lower end).
Equilibrium weights `p` satisfy detailed balance `p i * kp i = p (i+1) * km (i+1)`
(`DetailedBalance`), which is exactly the statement that the hopping dynamics has the profile
as its stationary distribution.  `IsMFPT` is the standard first-step-analysis system for the
mean first-passage time `T i` from `i` to the absorbing end state `n`.

* `mfptFun` -- the closed form
  `T m = Σ_{m ≤ i < n} (Σ_{j ≤ i} p j) / (p i · kp i)`,
  and `mfptFun_isMFPT` -- **it is a solution**: the first-passage system is satisfiable, so
  every statement below is about a real object.
* `mfpt_unique`, `mfpt_eq` -- **and it is the only one**: any solution of the system has
  exactly this value.  This is the discrete Kramers formula, proved here from detailed balance
  alone by a flux telescoping argument.
* `mfptFormula_pos` -- the time is strictly positive.
* `mfptFormula_smul`, `landscape_does_not_determine_rate` -- **the landscape does not determine
  the rate.**  Multiplying every rate by `c > 0` leaves the equilibrium profile, and hence every
  thermodynamic quantity, exactly unchanged, and divides every first-passage time by `c`.
* `kinetics_not_determined_by_profile` -- and not only by an overall factor: two rate profiles
  with the same equilibrium weights, differing only at one step, give times `3` and `5`.  A
  "diffusion coefficient" that varies along the coordinate is an independent input.
* `mfptFormula_ge_barrier`, `mfpt_ge_exp_barrier` -- **what the profile does determine is a
  lower bound, exponential in the barrier height**: for every intermediate state `b`,
  `T 0 ≥ p 0 / (p b · kp b)`, i.e. `T 0 ≥ exp (β (F b - F 0)) / kp b` for Boltzmann weights
  `p i = exp (-β F i)`.  Barrier crossing is slow for a thermodynamic reason; how slow also
  needs the kinetic prefactor.
* `mfptFormula_le` -- the matching upper bound `T 0 ≤ Σ_{i<n} 1 / (p i · kp i)` for a normalised
  profile.  Together with the previous item the barrier term brackets the answer.

The design consequence for a model of a disordered region: reporting the equilibrium ensemble --
even exactly, even as a function of a well-chosen coordinate -- underdetermines every kinetic
observable by an arbitrary factor.  A kinetic model must carry position-dependent rates
(equivalently, a diffusion profile) as data of its own.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace FirstPassage

open Finset

/-- The cumulative equilibrium weight up to and including state `i`. -/
noncomputable def cum (p : ℕ → ℝ) (i : ℕ) : ℝ := ∑ j ∈ range (i + 1), p j

lemma cum_zero (p : ℕ → ℝ) : cum p 0 = p 0 := by simp [cum]

lemma cum_succ (p : ℕ → ℝ) (i : ℕ) : cum p (i + 1) = cum p i + p (i + 1) := by
  simp [cum, Finset.sum_range_succ]

lemma cum_pos {p : ℕ → ℝ} {i : ℕ} (hp : ∀ j, j ≤ i → 0 < p j) : 0 < cum p i := by
  refine Finset.sum_pos (fun j hj => hp j (by simpa [Nat.lt_succ_iff] using hj)) ?_
  exact ⟨0, by simp⟩

lemma cum_ge_first {p : ℕ → ℝ} {i : ℕ} (hp : ∀ j, j ≤ i → 0 ≤ p j) : p 0 ≤ cum p i := by
  have : p 0 = ∑ j ∈ ({0} : Finset ℕ), p j := by simp
  rw [this]
  refine Finset.sum_le_sum_of_subset_of_nonneg (by simp) ?_
  intro j hj _
  exact hp j (by simpa [Nat.lt_succ_iff] using hj)

/-- Detailed balance for a hopping chain on `0, ..., n`: the equilibrium flux across each
bond vanishes, i.e. `p` is the stationary profile of the hopping dynamics. -/
def DetailedBalance (n : ℕ) (p kp km : ℕ → ℝ) : Prop :=
  ∀ i, i < n → p i * kp i = p (i + 1) * km (i + 1)

/-- The first-step-analysis system for the mean first-passage time to the absorbing state `n`:
`T n = 0` and, at every interior state, the expected waiting time plus the mean time from the
state jumped to.  (At `i = 0` the backward term is switched off by `km 0 = 0`, the reflecting
boundary; truncated subtraction makes the `i - 1` in that term harmless.) -/
def IsMFPT (n : ℕ) (kp km T : ℕ → ℝ) : Prop :=
  T n = 0 ∧ ∀ i, i < n → (kp i + km i) * T i = 1 + kp i * T (i + 1) + km i * T (i - 1)

/-- The closed-form first-passage time from state `m` to the absorbing end `n`. -/
noncomputable def mfptFun (n : ℕ) (p kp : ℕ → ℝ) (m : ℕ) : ℝ :=
  ∑ i ∈ Ico m n, cum p i / (p i * kp i)

/-- The closed-form first-passage time from the reflecting end to the absorbing end. -/
noncomputable def mfptFormula (n : ℕ) (p kp : ℕ → ℝ) : ℝ := mfptFun n p kp 0

lemma mfptFormula_eq (n : ℕ) (p kp : ℕ → ℝ) :
    mfptFormula n p kp = ∑ i ∈ range n, cum p i / (p i * kp i) := by
  rw [mfptFormula, mfptFun, Finset.range_eq_Ico]

lemma mfptFun_succ {n : ℕ} (p kp : ℕ → ℝ) {i : ℕ} (hi : i < n) :
    mfptFun n p kp i = cum p i / (p i * kp i) + mfptFun n p kp (i + 1) := by
  rw [mfptFun, mfptFun, Finset.sum_eq_sum_Ico_succ_bot hi]

lemma mfptFun_self (n : ℕ) (p kp : ℕ → ℝ) : mfptFun n p kp n = 0 := by
  simp [mfptFun]

/-- Detailed balance forces the backward rates to be positive wherever the forward rates and
the weights are. -/
lemma km_pos {n : ℕ} {p kp km : ℕ → ℝ} (hdb : DetailedBalance n p kp km)
    (hp : ∀ i, i ≤ n → 0 < p i) (hkp : ∀ i, i < n → 0 < kp i) {i : ℕ} (hi : i < n) :
    0 < km (i + 1) := by
  have h := hdb i hi
  have h1 : 0 < p i * kp i := mul_pos (hp i hi.le) (hkp i hi)
  have h2 : 0 < p (i + 1) := hp (i + 1) hi
  nlinarith

/-- **The closed form solves the first-passage system.**  Hence the system is satisfiable and
the statements below are not vacuous. -/
theorem mfptFun_isMFPT {n : ℕ} {p kp km : ℕ → ℝ} (hdb : DetailedBalance n p kp km)
    (hp : ∀ i, i ≤ n → 0 < p i) (hkp : ∀ i, i < n → 0 < kp i) (hkm0 : km 0 = 0) :
    IsMFPT n kp km (mfptFun n p kp) := by
  refine ⟨mfptFun_self n p kp, ?_⟩
  intro i hi
  have hpi : 0 < p i := hp i hi.le
  have hkpi : 0 < kp i := hkp i hi
  rcases Nat.eq_zero_or_pos i with hi0 | hipos
  · subst hi0
    rw [mfptFun_succ p kp hi, hkm0, cum_zero]
    have hp0 : p 0 ≠ 0 := ne_of_gt hpi
    have hk0 : kp 0 ≠ 0 := ne_of_gt hkpi
    have hmain : kp 0 * (p 0 / (p 0 * kp 0)) = 1 := by
      field_simp
    linear_combination hmain
  · obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
    have hj : j < n := by omega
    have hpj : 0 < p j := hp j hj.le
    have hkpj : 0 < kp j := hkp j hj
    have hkmi : 0 < km (j + 1) := km_pos hdb hp hkp hj
    have hdbj : p j * kp j = p (j + 1) * km (j + 1) := hdb j hj
    have hTj : mfptFun n p kp j = cum p j / (p j * kp j) + mfptFun n p kp (j + 1) :=
      mfptFun_succ p kp hj
    have hTi : mfptFun n p kp (j + 1)
        = cum p (j + 1) / (p (j + 1) * kp (j + 1)) + mfptFun n p kp (j + 1 + 1) :=
      mfptFun_succ p kp hi
    have hsub : (j + 1) - 1 = j := by omega
    rw [hsub, hTj, hTi]
    -- everything cancels except one identity: the forward term at `j+1` exceeds the backward
    -- term at `j` by exactly one unit of waiting time
    have hcum : cum p (j + 1) = cum p j + p (j + 1) := cum_succ p j
    have hmain : kp (j + 1) * (cum p (j + 1) / (p (j + 1) * kp (j + 1)))
        = 1 + km (j + 1) * (cum p j / (p j * kp j)) := by
      have h1 : p (j + 1) ≠ 0 := ne_of_gt (hp (j + 1) hi.le)
      have h2 : kp (j + 1) ≠ 0 := ne_of_gt (hkp (j + 1) hi)
      have hA : kp (j + 1) * (cum p (j + 1) / (p (j + 1) * kp (j + 1)))
          = cum p (j + 1) / p (j + 1) := by
        field_simp
      have hB : km (j + 1) * (cum p j / (p j * kp j)) = cum p j / p (j + 1) := by
        rw [hdbj]
        field_simp
      rw [hA, hB, hcum]
      field_simp
      ring
    linear_combination hmain

/-- The flux identity behind the closed form: for any solution of the first-passage system the
"current" `p i · kp i · (T i - T (i+1))` equals the cumulative weight below `i`. -/
lemma flux_eq {n : ℕ} {p kp km T : ℕ → ℝ} (hdb : DetailedBalance n p kp km) (hkm0 : km 0 = 0)
    (hT : IsMFPT n kp km T) :
    ∀ i, i < n → p i * kp i * (T i - T (i + 1)) = cum p i := by
  intro i
  induction i with
  | zero =>
      intro h0
      have heq := hT.2 0 h0
      rw [hkm0] at heq
      rw [cum_zero]
      linear_combination p 0 * heq
  | succ j ih =>
      intro hi
      have hj : j < n := by omega
      have hIH := ih hj
      have heq := hT.2 (j + 1) hi
      have hsub : (j + 1) - 1 = j := by omega
      rw [hsub] at heq
      have hdbj : p j * kp j = p (j + 1) * km (j + 1) := hdb j hj
      have hcum : cum p (j + 1) = cum p j + p (j + 1) := cum_succ p j
      -- multiply the balance equation at `j+1` by `p (j+1)` and use detailed balance
      linear_combination p (j + 1) * heq + hIH - (T j - T (j + 1)) * hdbj - hcum

/-- **Uniqueness: every solution of the first-passage system is the closed form.** -/
theorem mfpt_unique {n : ℕ} {p kp km T : ℕ → ℝ} (hdb : DetailedBalance n p kp km)
    (hp : ∀ i, i ≤ n → 0 < p i) (hkp : ∀ i, i < n → 0 < kp i) (hkm0 : km 0 = 0)
    (hT : IsMFPT n kp km T) : ∀ m, m ≤ n → T m = mfptFun n p kp m := by
  have hflux := flux_eq hdb hkm0 hT
  have key : ∀ k m, m + k = n → T m = mfptFun n p kp m := by
    intro k
    induction k with
    | zero =>
        intro m hm
        have : m = n := by omega
        subst this
        rw [hT.1, mfptFun_self]
    | succ k ih =>
        intro m hm
        have hmn : m < n := by omega
        have h1 : T (m + 1) = mfptFun n p kp (m + 1) := ih (m + 1) (by omega)
        have h2 := hflux m hmn
        have hpm : 0 < p m := hp m hmn.le
        have hkpm : 0 < kp m := hkp m hmn
        have hne : p m * kp m ≠ 0 := ne_of_gt (mul_pos hpm hkpm)
        have h3 : T m - T (m + 1) = cum p m / (p m * kp m) := by
          rw [eq_div_iff hne]
          linear_combination h2
        rw [mfptFun_succ p kp hmn, ← h1]
        linarith [h3]
  intro m hm
  exact key (n - m) m (by omega)

/-- **The discrete Kramers formula.**  The mean first-passage time from the reflecting end of a
one-dimensional profile to its absorbing end is
`Σ_{i<n} (Σ_{j ≤ i} p j) / (p i · kp i)`. -/
theorem mfpt_eq {n : ℕ} {p kp km T : ℕ → ℝ} (hdb : DetailedBalance n p kp km)
    (hp : ∀ i, i ≤ n → 0 < p i) (hkp : ∀ i, i < n → 0 < kp i) (hkm0 : km 0 = 0)
    (hT : IsMFPT n kp km T) : T 0 = mfptFormula n p kp :=
  mfpt_unique hdb hp hkp hkm0 hT 0 (Nat.zero_le n)

/-! ## What the formula says -/

/-- A crossing takes strictly positive time. -/
theorem mfptFormula_pos {n : ℕ} {p kp : ℕ → ℝ} (hn : 0 < n) (hp : ∀ i, i ≤ n → 0 < p i)
    (hkp : ∀ i, i < n → 0 < kp i) : 0 < mfptFormula n p kp := by
  rw [mfptFormula_eq]
  refine Finset.sum_pos (fun i hi => ?_) ⟨0, by simpa using hn⟩
  have hi' : i < n := Finset.mem_range.mp hi
  exact div_pos (cum_pos (fun j hj => hp j (by omega)))
    (mul_pos (hp i hi'.le) (hkp i hi'))

/-- Scaling every rate by `c` preserves detailed balance: the equilibrium profile, and with it
every thermodynamic quantity, is unchanged. -/
theorem detailedBalance_smul {n : ℕ} {p kp km : ℕ → ℝ} (c : ℝ)
    (hdb : DetailedBalance n p kp km) :
    DetailedBalance n p (fun i => c * kp i) (fun i => c * km i) := by
  intro i hi
  have h := hdb i hi
  simp only
  linear_combination c * h

/-- Scaling every rate by `c > 0` divides every first-passage time by `c`. -/
theorem mfptFormula_smul {n : ℕ} (p kp : ℕ → ℝ) {c : ℝ} (hc : c ≠ 0) :
    mfptFormula n p (fun i => c * kp i) = mfptFormula n p kp / c := by
  rw [mfptFormula_eq, mfptFormula_eq, Finset.sum_div]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  field_simp

/-- **The equilibrium profile does not determine the kinetics.**  With the same weights `p` --
hence the same free-energy landscape, the same populations, the same value of every equilibrium
observable -- the rates may be scaled by any `c > 0`, and every first-passage time is divided by
`c`.  No function of the landscape alone can return the rate. -/
theorem landscape_does_not_determine_rate {n : ℕ} {p kp km : ℕ → ℝ} {c : ℝ}
    (hn : 0 < n) (hc : 0 < c) (hc1 : c ≠ 1) (hdb : DetailedBalance n p kp km)
    (hp : ∀ i, i ≤ n → 0 < p i) (hkp : ∀ i, i < n → 0 < kp i) :
    DetailedBalance n p (fun i => c * kp i) (fun i => c * km i) ∧
      mfptFormula n p (fun i => c * kp i) = mfptFormula n p kp / c ∧
      mfptFormula n p (fun i => c * kp i) ≠ mfptFormula n p kp := by
  refine ⟨detailedBalance_smul c hdb, mfptFormula_smul p kp (ne_of_gt hc), ?_⟩
  rw [mfptFormula_smul p kp (ne_of_gt hc)]
  have hpos : 0 < mfptFormula n p kp := mfptFormula_pos hn hp hkp
  intro h
  rw [div_eq_iff (ne_of_gt hc)] at h
  have hz : mfptFormula n p kp * (c - 1) = 0 := by linear_combination -h
  rcases mul_eq_zero.mp hz with h0 | h1
  · exact absurd h0 (ne_of_gt hpos)
  · exact hc1 (by linarith)

/-! ### An explicit pair of kinetic profiles on the same landscape -/

/-- A flat three-state landscape. -/
noncomputable def flatP : ℕ → ℝ := fun _ => 1

/-- Uniform hopping rates on the flat landscape. -/
noncomputable def fastKp : ℕ → ℝ := fun _ => 1

/-- The backward rates detailed balance forces on `fastKp`. -/
noncomputable def fastKm : ℕ → ℝ := fun i => if i = 0 then 0 else 1

/-- The same landscape with the second step twice as slow: a position-dependent diffusion
coefficient. -/
noncomputable def slowKp : ℕ → ℝ := fun i => if i = 1 then 1/2 else 1

/-- The backward rates detailed balance forces on `slowKp`. -/
noncomputable def slowKm : ℕ → ℝ := fun i => if i = 0 then 0 else if i = 2 then 1/2 else 1

/-- **Not merely an overall factor: the profile leaves the kinetics free step by step.**  Two
rate profiles with the *same* equilibrium weights -- flat, so the same free energy at every
state -- and differing only in the rate of the second step give crossing times `3` and `5`.  A
position-dependent diffusion coefficient is an independent input to a kinetic model, not a
consequence of the landscape. -/
theorem kinetics_not_determined_by_profile :
    (∀ i, i ≤ 2 → 0 < flatP i) ∧ (∀ i, i < 2 → 0 < fastKp i) ∧ (∀ i, i < 2 → 0 < slowKp i) ∧
      DetailedBalance 2 flatP fastKp fastKm ∧ DetailedBalance 2 flatP slowKp slowKm ∧
      fastKm 0 = 0 ∧ slowKm 0 = 0 ∧
      mfptFormula 2 flatP fastKp = 3 ∧ mfptFormula 2 flatP slowKp = 5 := by
  refine ⟨fun i _ => by norm_num [flatP], fun i _ => by norm_num [fastKp],
    fun i _ => by simp only [slowKp]; split <;> norm_num, ?_, ?_, by norm_num [fastKm],
    by norm_num [slowKm], ?_, ?_⟩
  · intro i hi
    interval_cases i <;> norm_num [flatP, fastKp, fastKm]
  · intro i hi
    interval_cases i <;> norm_num [flatP, slowKp, slowKm]
  · rw [mfptFormula_eq]
    norm_num [Finset.sum_range_succ, cum, flatP, fastKp]
  · rw [mfptFormula_eq]
    norm_num [Finset.sum_range_succ, cum, flatP, slowKp]

/-- **The barrier bound.**  For every intermediate state `b`, the crossing time is at least
`p 0 / (p b · kp b)`: the ratio of the equilibrium weight of the starting well to that of the
state `b`, divided by the rate at `b`.  Taking `b` at the top of the barrier, this is the
Arrhenius factor. -/
theorem mfptFormula_ge_barrier {n : ℕ} {p kp : ℕ → ℝ} {b : ℕ} (hb : b < n)
    (hp : ∀ i, i ≤ n → 0 < p i) (hkp : ∀ i, i < n → 0 < kp i) :
    p 0 / (p b * kp b) ≤ mfptFormula n p kp := by
  rw [mfptFormula_eq]
  have hmem : b ∈ range n := Finset.mem_range.mpr hb
  have hbpos : 0 < p b * kp b := mul_pos (hp b hb.le) (hkp b hb)
  have hterm : p 0 / (p b * kp b) ≤ cum p b / (p b * kp b) := by
    have : p 0 ≤ cum p b := cum_ge_first (fun j hj => (hp j (by omega)).le)
    gcongr
  refine hterm.trans (Finset.single_le_sum (f := fun i => cum p i / (p i * kp i)) ?_ hmem)
  intro i hi
  have hi' : i < n := Finset.mem_range.mp hi
  exact le_of_lt (div_pos (cum_pos (fun j hj => hp j (by omega)))
    (mul_pos (hp i hi'.le) (hkp i hi')))

/-- **Kramers form.**  For Boltzmann weights `p i = exp (-β F i)` the crossing time is at least
`exp (β (F b - F 0)) / kp b`: exponential in the barrier height, with a prefactor that the
landscape does not supply. -/
theorem mfpt_ge_exp_barrier {n : ℕ} {kp F : ℕ → ℝ} {b : ℕ} {beta : ℝ} (hb : b < n)
    (hkp : ∀ i, i < n → 0 < kp i) :
    Real.exp (beta * (F b - F 0)) / kp b
      ≤ mfptFormula n (fun i => Real.exp (-(beta * F i))) kp := by
  have hp : ∀ i, i ≤ n → (0 : ℝ) < Real.exp (-(beta * F i)) := fun i _ => Real.exp_pos _
  have hmain := mfptFormula_ge_barrier (p := fun i => Real.exp (-(beta * F i))) hb hp hkp
  refine le_trans (le_of_eq ?_) hmain
  have hsplit : Real.exp (-(beta * F 0))
      = Real.exp (-(beta * F b)) * Real.exp (beta * (F b - F 0)) := by
    rw [← Real.exp_add]
    ring_nf
  simp only
  rw [hsplit, mul_comm (Real.exp (-(beta * F b))) (Real.exp (beta * (F b - F 0))),
    mul_comm (Real.exp (-(beta * F b))) (kp b),
    mul_div_mul_right _ _ (Real.exp_ne_zero (-(beta * F b)))]

/-- The matching upper bound for a normalised profile. -/
theorem mfptFormula_le {n : ℕ} {p kp : ℕ → ℝ} (hp : ∀ i, i ≤ n → 0 < p i)
    (hkp : ∀ i, i < n → 0 < kp i) (hnorm : ∀ i, i < n → cum p i ≤ 1) :
    mfptFormula n p kp ≤ ∑ i ∈ range n, 1 / (p i * kp i) := by
  rw [mfptFormula_eq]
  refine Finset.sum_le_sum (fun i hi => ?_)
  have hi' : i < n := Finset.mem_range.mp hi
  have hpos : 0 < p i * kp i := mul_pos (hp i hi'.le) (hkp i hi')
  have := hnorm i hi'
  gcongr

end FirstPassage

end IDR
