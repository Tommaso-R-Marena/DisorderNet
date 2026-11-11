/-
# Part CXXVII  The same collapse for the standard Debye–Hückel chain kernel

Part CXXVI proved that screening destroys charge patterning, using the linear chain kernel
damped by `exp (−κ d)`.  The kernel that the polymer-electrostatics literature actually uses,
and the one already defined in Part LXXIII, is the Debye–Hückel interaction evaluated at the
root-mean-square separation of a Gaussian chain,

  `w d = exp (−κ b √d) / (b √d)`,

`Pattern.debye N b κ q = ∑_{i<j} w (j−i) · q i · q j`.

This part proves the collapse for that kernel too, so the conclusion does not depend on the
convenient exponential model.

* `debye_kern_le` — the pointwise bound `w d ≤ 27 / (κ³ b⁴ d²)`, obtained from `x³/27 ≤ exp x`
  (`cube_div_le_exp`); the screening factor beats every power.
* `sum_inv_sq_le` — `∑_{d=1}^{m} 1/d² ≤ 2`, proved from scratch by telescoping.
* `abs_debye_le` — **the Debye screening bound.**  For every `κ > 0`, `b > 0` and every sequence
  of unit charges, `|debye N b κ q| ≤ 54 N / (κ³ b⁴)`: the electrostatic energy of a charged
  disordered region is at most *linear* in its length, whatever its charge pattern.
* `debye_tendsto_zero`, `debye_pattern_collapse` — hence at fixed length the energy of every
  pattern tends to zero as the salt concentration grows, and **any two sequences whatsoever**
  differ by at most `108 N / (κ³ b⁴)`: the whole patterning landscape flattens, not merely the
  gap between the two extremal patterns of Part CXXVI.

Together with Part CXXVI this says: charge patterning is a low-salt phenomenon in every one of
these models, and a model of a disordered region that reports a patterning number without being
told the ionic strength is reporting a quantity whose value it cannot know.
-/
import Mathlib
import RequestProject.ChargePatterning

set_option autoImplicit false

namespace IDR
namespace Salt

open Finset

/-! ## 1. Two elementary inequalities -/

/-- `x³/27 ≤ exp x` for `x ≥ 0`: obtained by cubing `x/3 ≤ exp (x/3)`. -/
lemma cube_div_le_exp {x : ℝ} (hx : 0 ≤ x) : x ^ 3 / 27 ≤ Real.exp x := by
  have h1 : x / 3 ≤ Real.exp (x / 3) := by
    have := Real.add_one_le_exp (x / 3)
    linarith
  have h0 : 0 ≤ x / 3 := by linarith
  have h2 : (x / 3) ^ 3 ≤ (Real.exp (x / 3)) ^ 3 := by
    exact pow_le_pow_left₀ h0 h1 3
  have h3 : (Real.exp (x / 3)) ^ 3 = Real.exp x := by
    rw [← Real.exp_nat_mul]
    congr 1
    push_cast
    ring
  have h4 : (x / 3) ^ 3 = x ^ 3 / 27 := by ring
  linarith [h2, h3.symm.le, h3.le, h4.symm.le, h4.le]

/-- `∑_{i<m} 1/(i+1)² ≤ 2 − 1/m` for `m ≥ 1`, by telescoping. -/
lemma sum_inv_sq_le_aux : ∀ m : ℕ, 1 ≤ m →
    ∑ i ∈ range m, (1 : ℝ) / ((i : ℝ) + 1) ^ 2 ≤ 2 - 1 / (m : ℝ) := by
  intro m
  induction m with
  | zero => intro h; exact absurd h (by norm_num)
  | succ m ih =>
      intro _
      rcases Nat.eq_zero_or_pos m with hm | hm
      · subst hm; norm_num
      · have hmR : (1 : ℝ) ≤ (m : ℝ) := by exact_mod_cast hm
        have hpos : (0 : ℝ) < (m : ℝ) := by linarith
        have hstep : (1 : ℝ) / ((m : ℝ) + 1) ^ 2 ≤ 1 / (m : ℝ) - 1 / ((m : ℝ) + 1) := by
          rw [div_sub_div _ _ (ne_of_gt hpos) (by linarith : ((m : ℝ) + 1) ≠ 0),
            div_le_div_iff₀ (by positivity) (by positivity)]
          ring_nf
          nlinarith [hpos]
        have h := ih hm
        rw [Finset.sum_range_succ]
        push_cast
        linarith

/-- `∑_{d=1}^{m} 1/d² ≤ 2`. -/
lemma sum_inv_sq_le (m : ℕ) : ∑ i ∈ range m, (1 : ℝ) / ((i : ℝ) + 1) ^ 2 ≤ 2 := by
  rcases Nat.eq_zero_or_pos m with hm | hm
  · subst hm; norm_num
  · have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hm
    have hinv : (0 : ℝ) < 1 / (m : ℝ) := by positivity
    linarith [sum_inv_sq_le_aux m hm]

/-! ## 2. The Debye kernel is dominated by an inverse square -/

/-- **The screened Coulomb coupling decays faster than any power.**  At inverse screening length
`κ > 0` on a chain of bond length `b > 0`, the Debye–Hückel coupling between charges `d ≥ 1`
residues apart is at most `27 / (κ³ b⁴ d²)`. -/
lemma debye_kern_le {kappa b : ℝ} (hk : 0 < kappa) (hb : 0 < b) {d : ℕ} (hd : 1 ≤ d) :
    Real.exp (-(kappa * (b * Real.sqrt d))) / (b * Real.sqrt d)
      ≤ 27 / (kappa ^ 3 * b ^ 4 * (d : ℝ) ^ 2) := by
  have hdR : (1 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
  have hs : 0 < Real.sqrt d := Real.sqrt_pos.2 (by linarith)
  have hsq : Real.sqrt d ^ 2 = (d : ℝ) := Real.sq_sqrt (by linarith)
  set a : ℝ := kappa * (b * Real.sqrt d) with ha
  have hapos : 0 < a := by positivity
  have hcube : a ^ 3 / 27 ≤ Real.exp a := cube_div_le_exp hapos.le
  have hexp : Real.exp (-a) = (Real.exp a)⁻¹ := Real.exp_neg a
  have hEpos : 0 < Real.exp a := Real.exp_pos a
  have ha3 : a ^ 3 = kappa ^ 3 * b ^ 3 * (Real.sqrt d) ^ 3 := by rw [ha]; ring
  have hexp_le : Real.exp (-a) ≤ 27 / (kappa ^ 3 * b ^ 3 * (Real.sqrt d) ^ 3) := by
    rw [hexp, ← ha3, le_div_iff₀ (by positivity)]
    rw [inv_mul_eq_div, div_le_iff₀ hEpos]
    nlinarith [hcube, hEpos]
  calc Real.exp (-a) / (b * Real.sqrt d)
      ≤ (27 / (kappa ^ 3 * b ^ 3 * (Real.sqrt d) ^ 3)) / (b * Real.sqrt d) := by
        exact div_le_div_of_nonneg_right hexp_le (by positivity)
    _ = 27 / (kappa ^ 3 * b ^ 4 * (d : ℝ) ^ 2) := by
        rw [div_div]
        congr 1
        have : (Real.sqrt d) ^ 3 * Real.sqrt d = (d : ℝ) ^ 2 := by
          have : (Real.sqrt d) ^ 3 * Real.sqrt d = (Real.sqrt d ^ 2) ^ 2 := by ring
          rw [this, hsq]
        linear_combination (kappa ^ 3 * b ^ 4) * this

/-! ## 3. The Debye screening bound -/

/-- **The Debye screening bound.**  At inverse screening length `κ > 0` and bond length `b > 0`,
the Debye–Hückel energy of any sequence of unit charges is at most `54 N / (κ³ b⁴)` in absolute
value: linear in the length of the region, and uniformly over all charge patterns. -/
theorem abs_debye_le {N : ℕ} {b kappa : ℝ} (hk : 0 < kappa) (hb : 0 < b) {q : ℕ → ℝ}
    (hq : ∀ i, |q i| ≤ 1) :
    |Pattern.debye N b kappa q| ≤ 54 * N / (kappa ^ 3 * b ^ 4) := by
  set C : ℝ := 27 / (kappa ^ 3 * b ^ 4) with hC
  have hCpos : 0 < C := by rw [hC]; positivity
  have hkern : ∀ j : ℕ, ∀ i ∈ range j,
      |Real.exp (-(kappa * (b * Real.sqrt ((j - i : ℕ) : ℝ)))) / (b * Real.sqrt ((j - i : ℕ) : ℝ))
        * (q i * q j)| ≤ C * (1 / (((j - i - 1 : ℕ) : ℝ) + 1) ^ 2) := by
    intro j i hi
    have hij : i < j := Finset.mem_range.1 hi
    have hd1 : 1 ≤ j - i := by omega
    have hd : (((j - i - 1 : ℕ) : ℝ) + 1) = ((j - i : ℕ) : ℝ) := by
      have : (j - i - 1 : ℕ) + 1 = j - i := by omega
      exact_mod_cast congrArg (fun n : ℕ => (n : ℝ)) this
    have hnonneg : 0 ≤ Real.exp (-(kappa * (b * Real.sqrt ((j - i : ℕ) : ℝ))))
        / (b * Real.sqrt ((j - i : ℕ) : ℝ)) := by positivity
    have hqq : |q i * q j| ≤ 1 := by
      rw [abs_mul]
      nlinarith [abs_nonneg (q i), abs_nonneg (q j), hq i, hq j]
    rw [abs_mul, abs_of_nonneg hnonneg]
    have hstep := debye_kern_le hk hb hd1
    calc Real.exp (-(kappa * (b * Real.sqrt ((j - i : ℕ) : ℝ)))) / (b * Real.sqrt ((j - i : ℕ) : ℝ))
          * |q i * q j|
        ≤ Real.exp (-(kappa * (b * Real.sqrt ((j - i : ℕ) : ℝ)))) / (b * Real.sqrt ((j - i : ℕ) : ℝ))
            * 1 := mul_le_mul_of_nonneg_left hqq hnonneg
      _ ≤ 27 / (kappa ^ 3 * b ^ 4 * ((j - i : ℕ) : ℝ) ^ 2) := by rw [mul_one]; exact hstep
      _ = C * (1 / (((j - i - 1 : ℕ) : ℝ) + 1) ^ 2) := by
          rw [hd, hC]; field_simp
  have hinner : ∀ j ∈ range N,
      |∑ i ∈ range j, Real.exp (-(kappa * (b * Real.sqrt ((j - i : ℕ) : ℝ))))
          / (b * Real.sqrt ((j - i : ℕ) : ℝ)) * (q i * q j)| ≤ 2 * C := by
    intro j _
    have h1 := Finset.abs_sum_le_sum_abs
      (fun i => Real.exp (-(kappa * (b * Real.sqrt ((j - i : ℕ) : ℝ))))
        / (b * Real.sqrt ((j - i : ℕ) : ℝ)) * (q i * q j)) (range j)
    have h2 : ∑ i ∈ range j, |Real.exp (-(kappa * (b * Real.sqrt ((j - i : ℕ) : ℝ))))
        / (b * Real.sqrt ((j - i : ℕ) : ℝ)) * (q i * q j)|
        ≤ ∑ i ∈ range j, C * (1 / (((j - i - 1 : ℕ) : ℝ) + 1) ^ 2) :=
      Finset.sum_le_sum (hkern j)
    have h3 : ∑ i ∈ range j, C * (1 / (((j - i - 1 : ℕ) : ℝ) + 1) ^ 2)
        = C * ∑ i ∈ range j, (1 / (((j - i - 1 : ℕ) : ℝ) + 1) ^ 2) := by
      rw [Finset.mul_sum]
    have h4 : ∑ i ∈ range j, (1 / (((j - i - 1 : ℕ) : ℝ) + 1) ^ 2)
        = ∑ i ∈ range j, (1 : ℝ) / ((i : ℝ) + 1) ^ 2 := by
      rw [← Finset.sum_range_reflect]
      refine Finset.sum_congr rfl fun i hi => ?_
      have hij : i < j := Finset.mem_range.1 hi
      have : j - (j - 1 - i) - 1 = i := by omega
      rw [this]
    have h5 : ∑ i ∈ range j, (1 : ℝ) / ((i : ℝ) + 1) ^ 2 ≤ 2 := sum_inv_sq_le j
    have : C * ∑ i ∈ range j, (1 / (((j - i - 1 : ℕ) : ℝ) + 1) ^ 2) ≤ C * 2 := by
      rw [h4]
      exact mul_le_mul_of_nonneg_left h5 hCpos.le
    linarith [h1, h2, h3.le, h3.ge]
  have hsum : |Pattern.debye N b kappa q| ≤ ∑ _j ∈ range N, 2 * C := by
    unfold Pattern.debye Pattern.pairEnergy
    calc |∑ j ∈ range N, ∑ i ∈ range j,
            Real.exp (-(kappa * (b * Real.sqrt ((j - i : ℕ) : ℝ))))
              / (b * Real.sqrt ((j - i : ℕ) : ℝ)) * (q i * q j)|
        ≤ ∑ j ∈ range N, |∑ i ∈ range j,
            Real.exp (-(kappa * (b * Real.sqrt ((j - i : ℕ) : ℝ))))
              / (b * Real.sqrt ((j - i : ℕ) : ℝ)) * (q i * q j)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _j ∈ range N, 2 * C := Finset.sum_le_sum hinner
  have hconst : ∑ _j ∈ range N, 2 * C = 54 * N / (kappa ^ 3 * b ^ 4) := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, hC]
    field_simp
    ring
  linarith [hsum, hconst.le, hconst.ge]

/-- **Screening destroys the Debye–Hückel energy.**  At fixed length and bond length, the energy
of every unit-charge sequence tends to zero as the salt concentration grows. -/
theorem debye_tendsto_zero (N : ℕ) (b : ℝ) (hb : 0 < b) (q : ℕ → ℝ) (hq : ∀ i, |q i| ≤ 1) :
    Filter.Tendsto (fun kappa : ℝ => Pattern.debye N b kappa q) Filter.atTop (nhds 0) := by
  have hbound : Filter.Tendsto (fun kappa : ℝ => 54 * N / (kappa ^ 3 * b ^ 4))
      Filter.atTop (nhds 0) := by
    have h3 : Filter.Tendsto (fun kappa : ℝ => kappa ^ 3 * b ^ 4) Filter.atTop Filter.atTop := by
      have hpow : Filter.Tendsto (fun kappa : ℝ => kappa ^ 3) Filter.atTop Filter.atTop :=
        Filter.tendsto_pow_atTop (by norm_num)
      simpa using hpow.atTop_mul_const (by positivity : (0 : ℝ) < b ^ 4)
    exact Filter.Tendsto.div_atTop tendsto_const_nhds h3
  refine squeeze_zero_norm' ?_ hbound
  filter_upwards [Filter.eventually_gt_atTop (0 : ℝ)] with kappa hkappa
  simpa [Real.norm_eq_abs] using abs_debye_le hkappa hb hq

/-- **The whole patterning landscape flattens.**  Any two unit-charge sequences of the same
length -- not merely the two extremal patterns -- have Debye–Hückel energies within
`108 N / (κ³ b⁴)` of each other.  At high salt the order of the charges no longer matters at
all. -/
theorem debye_pattern_collapse {N : ℕ} {b kappa : ℝ} (hk : 0 < kappa) (hb : 0 < b)
    {q q' : ℕ → ℝ} (hq : ∀ i, |q i| ≤ 1) (hq' : ∀ i, |q' i| ≤ 1) :
    |Pattern.debye N b kappa q - Pattern.debye N b kappa q'| ≤ 108 * N / (kappa ^ 3 * b ^ 4) := by
  have h1 := abs_debye_le (N := N) hk hb hq
  have h2 := abs_debye_le (N := N) hk hb hq'
  have h3 : |Pattern.debye N b kappa q - Pattern.debye N b kappa q'|
      ≤ |Pattern.debye N b kappa q| + |Pattern.debye N b kappa q'| := abs_sub _ _
  have hpos : 0 < kappa ^ 3 * b ^ 4 := by positivity
  have : 54 * (N : ℝ) / (kappa ^ 3 * b ^ 4) + 54 * (N : ℝ) / (kappa ^ 3 * b ^ 4)
      = 108 * N / (kappa ^ 3 * b ^ 4) := by field_simp; ring
  linarith

end Salt
end IDR
