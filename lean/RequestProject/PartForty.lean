/-
# Part XL  The entropy per residue, bracketed from both sides by finite counts

`RequestProject.ConnectiveBound` (Part XXXIX) and `RequestProject.Bridge` (Part XL) together
close the entropy per residue of a self-avoiding chain on the square lattice between two numbers
that a finite computation certifies.

The upper bound uses submultiplicativity of the conformation count and the exact seven-bond count
`c₇ = 2172 < 3⁷`, giving `μ < log 3`: excluded volume costs strictly more entropy per residue
than the naive no-retraction rule.  The lower bound uses Hammersley's bridges: a bridge is a
self-avoiding chain lying strictly to the right of its start and weakly to the left of its end in
one coordinate, two bridges concatenate to a bridge, so bridge counts are *super*multiplicative
and the six-bond count `101` gives `μ ≥ (log 101)/6`, an improvement on the directed-chain bound
`log 2`.

`IDR.entropy_per_residue_laws` bundles the five statements.
-/
import Mathlib
import RequestProject.ConnectiveBound
import RequestProject.Bridge

set_option autoImplicit false

namespace IDR

/-- **The entropy per residue of a self-avoiding chain, bracketed.**

1. *A finite count above*: there are exactly `2172` self-avoiding seven-bond conformations, and
   `2172 < 3⁷`.
2. *Hence*: the entropy per residue is strictly below `log 3`.
3. *A finite count below*: there are exactly `101` six-bond bridges.
4. *Hence*: the entropy per residue is at least `(log 101)/6`, which strictly improves the
   directed-chain bound `log 2`.
5. *The bracket*: `(log 101)/6 ≤ μ < log 3`, both ends strictly inside the ideal-chain value
   `log 4`. -/
theorem entropy_per_residue_laws :
    (SAW.cnt 7 = 2172 ∧ (2172 : ℕ) < 3 ^ 7) ∧
    SAW.connectiveConstant < Real.log 3 ∧
    (SAW.brCnt 6 = 101 ∧ (64 : ℕ) < 101) ∧
    (Real.log 2 < Real.log 101 / 6 ∧ Real.log 101 / 6 ≤ SAW.connectiveConstant) ∧
    SAW.connectiveConstant ∈ Set.Ico (Real.log 101 / 6) (Real.log 3) := by
  refine ⟨⟨SAW.cnt_seven, by norm_num⟩, SAW.connectiveConstant_lt_log_three,
    ⟨SAW.brCnt_six, by norm_num⟩,
    ⟨SAW.log_two_lt_log_bridge_div_six, SAW.log_bridge_div_six_le_connectiveConstant⟩,
    ⟨SAW.log_bridge_div_six_le_connectiveConstant, SAW.connectiveConstant_lt_log_three⟩⟩

end IDR
