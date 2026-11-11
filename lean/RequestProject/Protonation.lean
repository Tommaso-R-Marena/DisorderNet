/-
# Part XXV.2  pH and charge regulation: the charge of a disordered region is not a constant

The force field of Part XXIV fixes a partial charge on every atom.  For a titratable group
that is an idealisation: the protonation state is itself an equilibrium, governed by the pH
of the solution, and the mean charge of a disordered region therefore varies continuously
with pH.  This file removes the fixed-charge idealisation and prices it.

* `protonatedFraction pKa pH = 1 / (1 + 10^(pH - pKa))` -- the Henderson-Hasselbalch
  occupancy, with `protonatedFraction_mem_Ioo`, `protonatedFraction_half` (half-titration
  exactly at `pH = pKa`) and `protonatedFraction_strictAnti_pH`.
* `netCharge` -- the mean charge of a set of titratable sites, and
  `netCharge_strictAnti_pH`: a region whose protonated states are the more positive ones has
  a strictly decreasing titration curve.
* `no_fixed_charge_model` -- **no constant charge reproduces the titration curve**: for any
  number `q` there is a pH at which the mean charge of a single acidic site differs from `q`.
  The pH is a coordinate of the model in exactly the sense of `Context.lean`.
* `linkage_cycle` and `pKa_shift_eq_binding_shift` -- the thermodynamic cycle closes:
  the pKa shift caused by binding equals, in energy, the binding-energy change caused by
  protonation.  A model that reports a pKa shift is thereby committed to a binding shift.
-/
import Mathlib

namespace IDR

namespace Protonation

open Real

/-- The Henderson-Hasselbalch protonated fraction of a site of acidity constant `pKa` at
solution `pH`. -/
noncomputable def protonatedFraction (pKa pH : ℝ) : ℝ := 1 / (1 + (10:ℝ) ^ (pH - pKa))

lemma denom_pos (pKa pH : ℝ) : 0 < 1 + (10:ℝ) ^ (pH - pKa) := by
  have := Real.rpow_pos_of_pos (x := (10:ℝ)) (by norm_num) (pH - pKa)
  linarith

/-- The occupancy is a genuine probability: strictly between 0 and 1. -/
theorem protonatedFraction_mem_Ioo (pKa pH : ℝ) :
    protonatedFraction pKa pH ∈ Set.Ioo (0:ℝ) 1 := by
  have hd := denom_pos pKa pH
  have h10 := Real.rpow_pos_of_pos (x := (10:ℝ)) (by norm_num) (pH - pKa)
  constructor
  · exact div_pos one_pos hd
  · rw [protonatedFraction, div_lt_one hd]
    linarith

/-- Half-titration is exactly at `pH = pKa`. -/
theorem protonatedFraction_half (pKa : ℝ) : protonatedFraction pKa pKa = 1/2 := by
  unfold protonatedFraction
  norm_num

/-- **The titration curve is strictly decreasing in pH.** -/
theorem protonatedFraction_strictAnti_pH {pKa pH pH' : ℝ} (h : pH < pH') :
    protonatedFraction pKa pH' < protonatedFraction pKa pH := by
  have hd := denom_pos pKa pH
  have hd' := denom_pos pKa pH'
  have hlt : (10:ℝ) ^ (pH - pKa) < (10:ℝ) ^ (pH' - pKa) :=
    (Real.rpow_lt_rpow_left_iff (by norm_num)).mpr (by linarith)
  unfold protonatedFraction
  exact div_lt_div_of_pos_left one_pos hd (by linarith)

/-! ## The mean charge of a titratable region -/

variable {n : ℕ}

/-- The mean charge of `n` independent titratable sites: site `i` carries `zProt i` when
protonated and `zDeprot i` when deprotonated. -/
noncomputable def netCharge (pKa zProt zDeprot : Fin n → ℝ) (pH : ℝ) : ℝ :=
  ∑ i, (protonatedFraction (pKa i) pH * zProt i
        + (1 - protonatedFraction (pKa i) pH) * zDeprot i)

/-- **Charge regulation.**  If protonation makes every site more positive, the net charge of
the region is strictly decreasing in pH: the charge of a disordered region is a function of
the solution, not a parameter of the sequence. -/
theorem netCharge_strictAnti_pH {pKa zProt zDeprot : Fin n → ℝ} (hn : 0 < n)
    (hz : ∀ i, zDeprot i < zProt i) {pH pH' : ℝ} (h : pH < pH') :
    netCharge pKa zProt zDeprot pH' < netCharge pKa zProt zDeprot pH := by
  haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
  refine Finset.sum_lt_sum_of_nonempty Finset.univ_nonempty (fun i _ => ?_)
  have hf := protonatedFraction_strictAnti_pH (pKa := pKa i) h
  nlinarith [hz i]

/-- A single acidic site: neutral when protonated, `-1` when deprotonated. -/
noncomputable def acidCharge (pKa pH : ℝ) : ℝ :=
  netCharge (n := 1) (fun _ => pKa) (fun _ => 0) (fun _ => -1) pH

/-- **No fixed-charge model.**  For every constant `q` there is a pH at which the mean
charge of a single acidic site differs from `q`.  A force field with fixed partial charges
is therefore a model *of one pH*, and the pH must be reported with it. -/
theorem no_fixed_charge_model (pKa q : ℝ) : ∃ pH : ℝ, acidCharge pKa pH ≠ q := by
  by_contra hcon
  push_neg at hcon
  have hlt : acidCharge pKa (pKa + 1) < acidCharge pKa pKa := by
    unfold acidCharge
    exact netCharge_strictAnti_pH (n := 1) (by norm_num) (fun _ => by norm_num)
      (by linarith)
  rw [hcon, hcon] at hlt
  exact lt_irrefl q hlt

/-! ## Linkage: a pKa shift is a binding shift -/

/-- **The thermodynamic cycle closes.**  With `G s b` the free energy of the state with
protonation `s` and ligand occupancy `b`, the protonation energy difference caused by
binding equals the binding energy difference caused by protonation.  A model that predicts
a pKa shift on binding is thereby committed to a pH dependence of the affinity, with the
same number. -/
theorem linkage_cycle (G : Bool → Bool → ℝ) :
    (G true true - G false true) - (G true false - G false false)
      = (G true true - G true false) - (G false true - G false false) := by
  ring

/-- The same statement in the units in which it is measured: the pKa shift on binding,
multiplied by `RT ln 10`, is the change in binding free energy on protonation. -/
theorem pKa_shift_eq_binding_shift (RTln10 : ℝ) (G : Bool → Bool → ℝ)
    (dpKa : ℝ) (hshift : RTln10 * dpKa = (G true true - G false true)
      - (G true false - G false false)) :
    RTln10 * dpKa = (G true true - G true false) - (G false true - G false false) := by
  rw [hshift, linkage_cycle G]

end Protonation

end IDR
