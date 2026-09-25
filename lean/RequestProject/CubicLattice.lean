/-
# Part XII.2  Excluded volume in three dimensions

`RequestProject.SelfAvoiding` works on the square lattice, which is convenient for explicit
counts but is not where a polypeptide lives.  This file instantiates the same general theory
(`RequestProject.LatticeWalk`) on the **cubic lattice**, and shows that every conclusion drawn
in two dimensions survives in three: the conformational entropy per residue exists, it is
strictly below the ideal-chain value `log 6` -- so excluded volume costs a fixed amount of
entropy per residue -- and it is at least `log 3`, so the chain still has exponentially many
conformations and the capacity law of Part III still forces a generative model.

* `cnt3_submultiplicative`, `tendsto_connectiveConstant3` : subadditivity and Fekete.
* `three_pow_le_cnt3` : `3 ^ n ≤ cnt3 n` from the directed (positive-octant) walks.
* `log_three_le_connectiveConstant3`, `connectiveConstant3_lt_log_six` :
  `log 3 ≤ mu₃ < log 6`.
* `saw3_fraction_tendsto_zero` : the self-avoiding conformations of a three-dimensional chain
  are an exponentially vanishing fraction of the freely jointed ones.
-/
import Mathlib
import RequestProject.LatticeWalk

namespace IDR.SAW

open scoped BigOperators

/-- A position on the cubic lattice. -/
abbrev Site3 := ℤ × ℤ × ℤ

/-- The six unit bond vectors of the cubic lattice. -/
def dir3 : Fin 6 → Site3 :=
  ![(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

/-- The number of self-avoiding conformations of a three-dimensional chain of `n` bonds. -/
def cnt3 (n : ℕ) : ℕ := cntOf dir3 n

/-- **Subadditivity of conformational entropy** on the cubic lattice. -/
theorem cnt3_submultiplicative (m n : ℕ) : cnt3 (m + n) ≤ cnt3 m * cnt3 n :=
  cntOf_submultiplicative dir3 m n

/-- The "height" functional `x + y + z`, strictly increasing along a directed walk. -/
def hgt3 (p : Site3) : ℤ := p.1 + p.2.1 + p.2.2

lemma hgt3_add (a b : Site3) : hgt3 (a + b) = hgt3 a + hgt3 b := by
  simp only [hgt3, Prod.fst_add, Prod.snd_add]; ring

lemma hgt3_zero : hgt3 0 = 0 := by simp [hgt3]

/-- **A three-dimensional chain still has exponentially many conformations.**  The walks that
only step in the three positive directions are self-avoiding. -/
theorem three_pow_le_cnt3 (n : ℕ) : 3 ^ n ≤ cnt3 n := by
  refine pow_le_cntOf dir3 hgt3 hgt3_add hgt3_zero ![0, 2, 4] ?_ ?_ n
  · decide
  · decide

lemma cnt3_pos (n : ℕ) : 0 < cnt3 n :=
  lt_of_lt_of_le (Nat.pow_pos (by norm_num)) (three_pow_le_cnt3 n)

/-- The conformational entropy of a three-dimensional chain of `n` bonds, in nats. -/
noncomputable def logCnt3 (n : ℕ) : ℝ := logCntOf dir3 n

/-- **The connective constant of the cubic lattice**: the conformational entropy per residue
of a self-avoiding chain in three dimensions. -/
noncomputable def connectiveConstant3 : ℝ := connectiveConstantOf dir3 cnt3_pos

theorem tendsto_connectiveConstant3 :
    Filter.Tendsto (fun n => logCnt3 n / n) Filter.atTop (nhds connectiveConstant3) :=
  tendsto_connectiveConstantOf cnt3_pos

theorem connectiveConstant3_le_div {n : ℕ} (hn : n ≠ 0) :
    connectiveConstant3 ≤ logCnt3 n / n :=
  connectiveConstantOf_le_div cnt3_pos hn

/-- A two-bond chain in three dimensions has `30` self-avoiding conformations, against `36`
unrestricted ones. -/
theorem cnt3_two : cnt3 2 = 30 := by decide

/-- **Excluded volume costs a fixed amount of entropy per residue, in three dimensions too.**
The entropy per residue is strictly below the ideal-chain value `log 6`. -/
theorem connectiveConstant3_lt_log_six : connectiveConstant3 < Real.log 6 := by
  have h2 : connectiveConstant3 ≤ logCnt3 2 / 2 := connectiveConstant3_le_div (by norm_num)
  have hval : logCnt3 2 = Real.log 30 := by
    rw [logCnt3, logCntOf, show cntOf dir3 2 = 30 from cnt3_two]
    norm_num
  have hlt : Real.log 30 < Real.log 36 := Real.log_lt_log (by norm_num) (by norm_num)
  have h36 : Real.log 36 = 2 * Real.log 6 := by
    rw [show (36 : ℝ) = 6 ^ (2 : ℕ) by norm_num, Real.log_pow]
    norm_num
  rw [hval] at h2
  linarith [h2, hlt, h36]

/-- **Excluded volume does not collapse the three-dimensional ensemble.** -/
theorem log_three_le_connectiveConstant3 : Real.log 3 ≤ connectiveConstant3 := by
  have h := log_le_connectiveConstantOf (dir := dir3) cnt3_pos (r := 3) (by norm_num)
    three_pow_le_cnt3
  simpa [connectiveConstant3] using h

theorem connectiveConstant3_pos : 0 < connectiveConstant3 :=
  lt_of_lt_of_le (Real.log_pos (by norm_num)) log_three_le_connectiveConstant3

/-- Iterating the cut bound in three dimensions. -/
lemma cnt3_two_mul (m : ℕ) : cnt3 (2 * m) ≤ 30 ^ m := by
  induction m with
  | zero => decide
  | succ k ih =>
      have hcut : cnt3 (2 * (k + 1)) ≤ cnt3 2 * cnt3 (2 * k) := by
        have := cnt3_submultiplicative 2 (2 * k)
        simpa [Nat.mul_succ, Nat.add_comm] using this
      calc cnt3 (2 * (k + 1)) ≤ cnt3 2 * cnt3 (2 * k) := hcut
        _ ≤ 30 * 30 ^ k := by rw [cnt3_two]; exact Nat.mul_le_mul_left _ ih
        _ = 30 ^ (k + 1) := by ring

/-- **An ideal-chain generator is wrong about the support in three dimensions too.** -/
theorem saw3_fraction_le (m : ℕ) :
    (cnt3 (2 * m) : ℝ) / 6 ^ (2 * m) ≤ (5 / 6 : ℝ) ^ m := by
  have h1 : (cnt3 (2 * m) : ℝ) ≤ 30 ^ m := by exact_mod_cast cnt3_two_mul m
  have h2 : (6 : ℝ) ^ (2 * m) = 36 ^ m := by rw [pow_mul]; norm_num
  rw [h2, div_le_iff₀ (by positivity)]
  calc (cnt3 (2 * m) : ℝ) ≤ 30 ^ m := h1
    _ = (5 / 6 : ℝ) ^ m * 36 ^ m := by rw [← mul_pow]; norm_num

theorem saw3_fraction_tendsto_zero :
    Filter.Tendsto (fun m : ℕ => (cnt3 (2 * m) : ℝ) / 6 ^ (2 * m)) Filter.atTop (nhds 0) := by
  have hgeom : Filter.Tendsto (fun m : ℕ => (5 / 6 : ℝ) ^ m) Filter.atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  exact squeeze_zero (fun m => by positivity) (fun m => saw3_fraction_le m) hgeom

/-- **The three-dimensional design laws of excluded volume.**  Everything proved on the square
lattice holds where a polypeptide actually lives: the entropy per residue exists, lies
strictly between `log 3` and `log 6`, and the self-avoiding conformations are an exponentially
vanishing fraction of the freely jointed ones. -/
theorem cubic_excluded_volume_laws :
    Filter.Tendsto (fun n => logCnt3 n / n) Filter.atTop (nhds connectiveConstant3) ∧
    (Real.log 3 ≤ connectiveConstant3 ∧ connectiveConstant3 < Real.log 6) ∧
    (∀ n, 3 ^ n ≤ cnt3 n) ∧
    Filter.Tendsto (fun m : ℕ => (cnt3 (2 * m) : ℝ) / 6 ^ (2 * m)) Filter.atTop (nhds 0) :=
  ⟨tendsto_connectiveConstant3,
    ⟨log_three_le_connectiveConstant3, connectiveConstant3_lt_log_six⟩,
    three_pow_le_cnt3, saw3_fraction_tendsto_zero⟩

end IDR.SAW
