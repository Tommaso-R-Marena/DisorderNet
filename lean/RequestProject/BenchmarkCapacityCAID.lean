/-
# The capacity theorem, instantiated: a disorder challenge of CAID scale

The general statements live in `RequestProject.BenchmarkCapacity` (the capacity of a benchmark
with `n` targets, annotation error rate `eps` and effect size `delta`) and in
`RequestProject.BenchmarkCapacityLabels` (the Hamming instantiation, with the matching
converse: pairs no analysis can resolve).  This file is the demonstration: it puts numbers of
the scale a community disorder challenge works at into those theorems and reads off the
answer.

The numbers below are *parameters chosen for illustration*, not measurements: a challenge with
a few hundred targets, and an annotation error rate of a few percent.  Nothing in the
mathematics depends on the particular choice; each statement is a numerical instance of the
general theorem, and any other pair of numbers can be substituted.

* `capacity_200_targets_5pct` -- 200 targets, 5% annotation error, 1% effect size: **10**
  methods can be placed in a certified order.  Not 10 per cent -- 10 methods.
* `capacity_200_targets_10pct` -- doubling the annotation error to 10% halves the capacity
  to **5**.
* `capacity_1000_targets_5pct` -- multiplying the targets by five, at the same label quality,
  leaves the capacity at **10**: collecting targets does not buy resolution that the labels do
  not have.
* `capacity_residue_level` -- the same at residue level: 100000 scorable residues of which
  5000 are mislabelled certify a ranking of at most **10** methods.
* `four_way_tie_unresolvable` -- and a concrete unresolvable pair: two predictions on four
  residues, one mislabelled, whose ranking two admissible truths order in opposite ways.
-/
import Mathlib
import RequestProject.BenchmarkCapacity
import RequestProject.BenchmarkCapacityLabels

set_option autoImplicit false

namespace IDR
namespace BenchCapacity

open Finset
open IDR.LabelNoise

/-! ## Numerical instances of the general capacity formula -/

/-- **200 targets, 5% annotation error, 1% effect size: ten methods.** -/
theorem capacity_200_targets_5pct : benchCapacity 200 (5/100) (1/100) = 10 := by
  have hres : resolution (5/100 : ℝ) (1/100) = 1/10 := by
    simp only [resolution]; norm_num
  have hfl : ⌊resolution (5/100 : ℝ) (1/100) * (200:ℕ)⌋₊ = 20 := by
    rw [hres]; norm_num
  rw [benchCapacity, hfl]

/-- **Doubling the annotation error halves the capacity: five methods.** -/
theorem capacity_200_targets_10pct : benchCapacity 200 (10/100) (1/100) = 5 := by
  have hres : resolution (10/100 : ℝ) (1/100) = 1/5 := by
    simp only [resolution]; norm_num
  have hfl : ⌊resolution (10/100 : ℝ) (1/100) * (200:ℕ)⌋₊ = 40 := by
    rw [hres]; norm_num
  rw [benchCapacity, hfl]

/-- **Five times as many targets, same labels: still ten methods.**  Capacity is set by label
quality, not by sample size. -/
theorem capacity_1000_targets_5pct : benchCapacity 1000 (5/100) (1/100) = 10 := by
  have hres : resolution (5/100 : ℝ) (1/100) = 1/10 := by
    simp only [resolution]; norm_num
  have hfl : ⌊resolution (5/100 : ℝ) (1/100) * (1000:ℕ)⌋₊ = 100 := by
    rw [hres]; norm_num
  rw [benchCapacity, hfl]

/-- **The residue-level count.**  With 100000 scorable residues of which 5000 are mislabelled,
at most ten methods can be certifiably ranked. -/
theorem capacity_residue_level : capacityNat 100000 (2 * 5000) = 10 := by
  rw [capacityNat]

/-- Halving the annotation error doubles the capacity, at residue level. -/
theorem capacity_residue_level_halved_noise : capacityNat 100000 (2 * 2500) = 20 := by
  rw [capacityNat]

/-! ## A concrete unresolvable pair

Four residues; the annotation calls residues `0` and `1` disordered.  Method `P` predicts `{0}`
and method `Q` predicts `{1}`; both are measured to make one mistake.  One admissible truth
(the annotation with residue `1` removed) makes `P` strictly better, another (the annotation
with residue `0` removed) makes `Q` strictly better, and each differs from the annotation on a
single residue.  So a one-residue annotation error is enough to make this comparison
undecidable -- not undecided by the present analysis, but undecidable from these data. -/

/-- The annotation of the toy benchmark: residues `0` and `1` are called disordered. -/
def annotToy : Finset (Fin 4) := {0, 1}

/-- The first method: it predicts residue `0` only. -/
def predP : Finset (Fin 4) := {0}

/-- The second method: it predicts residue `1` only. -/
def predQ : Finset (Fin 4) := {1}

/-- **A concrete unresolvable comparison.**  Two truths, each one residue away from the
annotation, order the two methods in opposite ways. -/
theorem four_way_tie_unresolvable :
    (∃ T₁ : Finset (Fin 4), noise T₁ annotToy ≤ 1 ∧ errors T₁ predP < errors T₁ predQ) ∧
      (∃ T₂ : Finset (Fin 4), noise T₂ annotToy ≤ 1 ∧ errors T₂ predQ < errors T₂ predP) := by
  apply unresolvable_of_close (nu := 1) <;> decide

end BenchCapacity
end IDR
