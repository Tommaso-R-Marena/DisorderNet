/-
# Part XXXIX  Tightening the entropy per residue

Part XII bracketed the conformational entropy per residue of a self-avoiding chain on the square
lattice between `log 2` and `log 4`, the upper bound coming from the exact count `c₄ = 100` of
four-bond conformations.  The upper end of that bracket is weak: `log 4` is the ideal-chain value
itself, and even the crudest excluded-volume constraint — a chain may not immediately retrace a
bond — leaves only three continuations of each bond and so at most `4·3^{n−1}` conformations, an
entropy per residue of `log 3` (that elementary count is quoted as the point of comparison; it is
not formalised here).  This file proves that the true value is strictly below that
naive figure.

The mechanism is the exact conformation count of a seven-bond chain, `cnt_seven : c₇ = 2172`,
established by exhaustive enumeration of all `4⁷ = 16384` bond sequences (the enumeration is a
kernel computation, so no additional axioms are used).  Since `c₇ = 2172 < 2187 = 3⁷` and the
count is submultiplicative, `connectiveConstant ≤ (log c₇)/7 < log 3`.

* `cnt_seven` -- the exact count.
* `connectiveConstant_lt_log_three` -- excluded volume costs strictly more entropy per residue
  than forbidding immediate retraction of the previous bond.
* `connectiveConstant_mem_Ico` -- the tightened bracket `log 2 ≤ μ < log 3`.

The design consequence is unchanged in kind and sharper in degree: the conformational entropy a
disordered chain actually has is not the ideal-chain entropy, and is not even the entropy left
after the one-bond correction.  A generative model whose per-residue branching is calibrated on
either figure over-counts conformations exponentially in the length of the region.
-/
import Mathlib
import RequestProject.SelfAvoiding

set_option maxRecDepth 200000
set_option maxHeartbeats 4000000

namespace IDR.SAW

/-- There are exactly `2172` self-avoiding conformations of a seven-bond chain on the square
lattice, against `4⁷ = 16384` unrestricted ones and `3⁷ = 2187` allowed by the no-retraction
rule alone.  Proved by exhaustive kernel enumeration. -/
theorem cnt_seven : cnt 7 = 2172 := by decide

/-- **Excluded volume costs more than the no-retraction rule.**  The conformational entropy per
residue of a self-avoiding chain on the square lattice is strictly below `log 3`, the value left
by forbidding only the immediate retraction of the previous bond. -/
theorem connectiveConstant_lt_log_three : connectiveConstant < Real.log 3 := by
  have h7 : connectiveConstant ≤ logCnt 7 / 7 := connectiveConstant_le_div (by norm_num)
  have hval : logCnt 7 = Real.log 2172 := by
    rw [logCnt, logCntOf, show cntOf dir 7 = 2172 from cnt_seven]
    norm_num
  have hlt : Real.log 2172 < Real.log 2187 := Real.log_lt_log (by norm_num) (by norm_num)
  have h2187 : Real.log 2187 = 7 * Real.log 3 := by
    rw [show (2187 : ℝ) = 3 ^ (7 : ℕ) by norm_num, Real.log_pow]
    norm_num
  rw [hval] at h7
  have h7' : 7 * connectiveConstant ≤ Real.log 2172 := by
    have := h7
    rw [le_div_iff₀ (by norm_num : (0:ℝ) < 7)] at this
    linarith
  linarith

/-- The tightened bracket: `log 2 ≤ μ < log 3`. -/
theorem connectiveConstant_mem_Ico :
    connectiveConstant ∈ Set.Ico (Real.log 2) (Real.log 3) :=
  ⟨log_two_le_connectiveConstant, connectiveConstant_lt_log_three⟩

end IDR.SAW
