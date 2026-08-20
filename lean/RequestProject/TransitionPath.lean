/-
# Part LIV.1  Transition-path times: the crossing is not the waiting

Parts XLVIII--L solve the hopping dynamics of a reaction coordinate exactly and show that the
equilibrium profile fixes neither the rate nor the mechanism.  They left one object out, and it
is the one a single-molecule experiment on a disordered region actually resolves: the
*transition-path time*, the duration of the crossing itself, measured only over trajectories
that go from one end to the other without returning.  A mean first-passage time is dominated by
waiting in the reactant well; a transition-path time is not, and the two obey different laws.

This file constructs the conditioned (reactive) dynamics explicitly -- the Doob `h`-transform by
the committor -- and reduces it to the machinery already proved.

* `reactP`, `reactKp`, `reactKm` -- the reactive chain on the states `1, ..., n`, written with
  its lower end shifted to `0`: forward rate `kp i · q(i+1)/q(i)`, backward rate
  `km i · q(i-1)/q(i)`, and weights `p i · q(i)²`.  The lower end is automatically reflecting
  (`reactKm_zero`), because the committor vanishes at the reactant state: a reactive trajectory
  cannot return.
* `reactive_detailedBalance` -- **the reactive dynamics is again detailed-balanced**, with the
  reweighted profile `p q²`.  This is the whole content of the `h`-transform, and it is what
  makes the transition-path time computable: everything proved for first-passage times applies
  verbatim to the conditioned chain.
* `tpt_eq`, `tptFormula_pos` -- consequently the mean transition-path time is the *unique*
  solution of the reactive first-step system, and equals the discrete Kramers formula for the
  reweighted profile.
* `witness_mfpt`, `witness_tpt` -- the three-state witness, solved in closed form: on the
  profile `p = (1, e, 1)` with unit forward rates the mean first-passage time is
  `1 + (1+e)/e` and the mean transition-path time is `e/(1+e)`.
* `slow_reaction_fast_paths` -- **the law they violate together.**  For every `M` there is a
  detailed-balanced model whose mean first-passage time exceeds `M` while its mean
  transition-path time is below `1/M`.  Raising the barrier makes the reaction slower and the
  crossings *faster*: no function of the first-passage time returns the transition-path time,
  and a measured transition-path time is not a measurement of a rate.
-/
import Mathlib
import RequestProject.FirstPassage
import RequestProject.Committor

set_option autoImplicit false

namespace IDR

namespace TransitionPath

open Finset
open IDR.FirstPassage
open IDR.Committor

/-- The equilibrium weights of the reactive (committor-conditioned) chain, with the reactant
state shifted away: `reactP p q i = p (i+1) · q (i+1)²`. -/
noncomputable def reactP (p q : ℕ → ℝ) (i : ℕ) : ℝ := p (i + 1) * q (i + 1) ^ 2

/-- The forward rates of the reactive chain: `kp i · q(i+1)/q(i)`. -/
noncomputable def reactKp (kp q : ℕ → ℝ) (i : ℕ) : ℝ := kp (i + 1) * q (i + 2) / q (i + 1)

/-- The backward rates of the reactive chain: `km i · q(i-1)/q(i)`, which vanishes at the lower
end because the committor vanishes at the reactant state. -/
noncomputable def reactKm (km q : ℕ → ℝ) (i : ℕ) : ℝ :=
  if i = 0 then 0 else km (i + 1) * q i / q (i + 1)

/-- A reactive trajectory never returns: the conditioned chain is reflecting at its lower end. -/
theorem reactKm_zero (km q : ℕ → ℝ) : reactKm km q 0 = 0 := by simp [reactKm]

/-- **The `h`-transform preserves detailed balance.**  Conditioning a detailed-balanced hopping
chain on reaching the product before returning to the reactant gives another detailed-balanced
hopping chain, whose profile is the reactive one `p q²`. -/
theorem reactive_detailedBalance {n : ℕ} {p kp km q : ℕ → ℝ}
    (hdb : DetailedBalance n p kp km) (hq : ∀ i, 1 ≤ i → i ≤ n → 0 < q i) :
    DetailedBalance (n - 1) (reactP p q) (reactKp kp q) (reactKm km q) := by
  intro i hi
  have hin : i + 1 < n := by omega
  have hq1 : 0 < q (i + 1) := hq (i + 1) (by omega) (by omega)
  have hq2 : 0 < q (i + 2) := hq (i + 2) (by omega) (by omega)
  have hstep := hdb (i + 1) hin
  simp only [reactP, reactKp, reactKm, if_neg (Nat.succ_ne_zero i)]
  have h1 : i + 1 + 1 = i + 2 := rfl
  rw [h1]
  field_simp
  nlinarith [hstep, hq1, hq2]

/-- The closed-form mean transition-path time. -/
noncomputable def tptFormula (n : ℕ) (p kp q : ℕ → ℝ) : ℝ :=
  mfptFormula (n - 1) (reactP p q) (reactKp kp q)

/-- **The mean transition-path time is the unique solution of the reactive first-step system**,
and it is the discrete Kramers formula for the reweighted profile `p q²`. -/
theorem tpt_eq {n : ℕ} {p kp km q T : ℕ → ℝ} (hn : 1 ≤ n) (hdb : DetailedBalance n p kp km)
    (hp : ∀ i, i ≤ n → 0 < p i) (hkp : ∀ i, i < n → 0 < kp i)
    (hq : ∀ i, 1 ≤ i → i ≤ n → 0 < q i)
    (hT : IsMFPT (n - 1) (reactKp kp q) (reactKm km q) T) :
    T 0 = tptFormula n p kp q := by
  refine mfpt_eq (reactive_detailedBalance hdb hq) ?_ ?_ (reactKm_zero km q) hT
  · intro i hi
    have : i + 1 ≤ n := by omega
    exact mul_pos (hp (i + 1) this) (pow_pos (hq (i + 1) (by omega) this) 2)
  · intro i hi
    have h1 : i + 1 < n := by omega
    exact div_pos (mul_pos (hkp (i + 1) h1) (hq (i + 2) (by omega) (by omega)))
      (hq (i + 1) (by omega) (by omega))

/-- A crossing takes strictly positive time. -/
theorem tptFormula_pos {n : ℕ} {p kp q : ℕ → ℝ} (hn : 1 < n) (hp : ∀ i, i ≤ n → 0 < p i)
    (hkp : ∀ i, i < n → 0 < kp i) (hq : ∀ i, 1 ≤ i → i ≤ n → 0 < q i) :
    0 < tptFormula n p kp q := by
  refine mfptFormula_pos (by omega) ?_ ?_
  · intro i hi
    have : i + 1 ≤ n := by omega
    exact mul_pos (hp (i + 1) this) (pow_pos (hq (i + 1) (by omega) this) 2)
  · intro i hi
    have h1 : i + 1 < n := by omega
    exact div_pos (mul_pos (hkp (i + 1) h1) (hq (i + 2) (by omega) (by omega)))
      (hq (i + 1) (by omega) (by omega))

/-! ### The three-state witness -/

/-- The profile `(1, e, 1)`: a single barrier of weight `e`. -/
noncomputable def wP (e : ℝ) : ℕ → ℝ := fun i => if i = 1 then e else 1

/-- Unit forward rates. -/
noncomputable def wKp : ℕ → ℝ := fun _ => 1

/-- The backward rates forced by detailed balance. -/
noncomputable def wKm (e : ℝ) : ℕ → ℝ := fun i => if i = 0 then 0 else if i = 1 then 1 / e else e

theorem wP_pos {e : ℝ} (he : 0 < e) : ∀ i, i ≤ 2 → 0 < wP e i := by
  intro i _
  by_cases h : i = 1 <;> simp [wP, h, he]

theorem wKp_pos : ∀ i, i < 2 → 0 < wKp i := by intro i _; simp [wKp]

theorem w_detailedBalance {e : ℝ} (he : 0 < e) : DetailedBalance 2 (wP e) wKp (wKm e) := by
  intro i hi
  interval_cases i
  · simp only [wP, wKp, wKm]
    norm_num
    field_simp
  · simp [wP, wKp, wKm]

/-- The committor of the witness at the barrier state. -/
theorem witness_committor {e : ℝ} (he : 0 < e) :
    committorFun 2 (wP e) wKp 1 = e / (1 + e) := by
  simp only [committorFun, resSum, res, wP, wKp, Finset.sum_range_succ, Finset.sum_range_zero]
  norm_num
  field_simp
  ring

/-- **The mean first-passage time of the witness**: `1 + (1+e)/e`, which diverges as the
barrier grows. -/
theorem witness_mfpt (e : ℝ) :
    mfptFormula 2 (wP e) wKp = 1 + (1 + e) / e := by
  simp only [mfptFormula_eq, cum, wP, wKp, Finset.sum_range_succ, Finset.sum_range_zero]
  norm_num

/-- **The mean transition-path time of the witness**: `e/(1+e)`, which *shrinks* as the barrier
grows. -/
theorem witness_tpt {e : ℝ} (he : 0 < e) :
    tptFormula 2 (wP e) wKp (committorFun 2 (wP e) wKp) = e / (1 + e) := by
  have hq1 : committorFun 2 (wP e) wKp 1 = e / (1 + e) := witness_committor he
  have hq2 : committorFun 2 (wP e) wKp 2 = 1 :=
    committor_last (by norm_num) (wP_pos he) wKp_pos
  have hq1pos : 0 < committorFun 2 (wP e) wKp 1 := by rw [hq1]; positivity
  simp only [tptFormula, mfptFormula_eq, cum, reactP, reactKp, wKp,
    Finset.sum_range_succ, Finset.sum_range_zero]
  norm_num
  rw [hq1, hq2]
  have hne : (1 : ℝ) + e ≠ 0 := by positivity
  field_simp [wP]
  simp [wP, ne_of_gt he]

/-- **Slow reaction, fast paths.**  For every `M` there is a detailed-balanced three-state model
whose mean first-passage time exceeds `M` and whose mean transition-path time is below `1/M`:
raising a barrier slows the reaction and speeds up the crossings.  No function of the one
returns the other. -/
theorem slow_reaction_fast_paths (M : ℝ) (hM : 0 < M) :
    ∃ e : ℝ, 0 < e ∧ DetailedBalance 2 (wP e) wKp (wKm e) ∧
      M ≤ mfptFormula 2 (wP e) wKp ∧
      tptFormula 2 (wP e) wKp (committorFun 2 (wP e) wKp) ≤ 1 / M := by
  refine ⟨min 1 (1 / M), by positivity, w_detailedBalance (by positivity), ?_, ?_⟩
  · rw [witness_mfpt]
    set e := min 1 (1 / M) with he
    have hepos : 0 < e := by positivity
    have hele : e ≤ 1 / M := min_le_right _ _
    have hMe : M ≤ 1 / e := by
      rw [le_div_iff₀ hepos]
      calc M * e ≤ M * (1 / M) := by nlinarith
        _ = 1 := by field_simp
    have h1 : (1 + e) / e = 1 / e + 1 := by field_simp
    linarith
  · rw [witness_tpt (by positivity)]
    set e := min 1 (1 / M) with he
    have hepos : 0 < e := by positivity
    have hele : e ≤ 1 / M := min_le_right _ _
    calc e / (1 + e) ≤ e := by
          rw [div_le_iff₀ (by positivity)]
          nlinarith
      _ ≤ 1 / M := hele

end TransitionPath

end IDR
