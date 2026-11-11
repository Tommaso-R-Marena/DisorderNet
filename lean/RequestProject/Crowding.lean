/-
# Part XV.1  Macromolecular crowding: the ensemble in the cell is not the ensemble in the tube

Every quantitative ensemble of an intrinsically disordered region that has ever been
measured was measured in dilute buffer, while the region itself works at a total
macromolecular concentration of some 300 g/L.  This file proves what that displacement
does, exactly, and that it cannot be repaired by refitting a temperature.

The physics is the standard one: a crowded background at osmotic pressure `Pi` costs each
conformation the work `Pi · v(x)` needed to open a cavity of its excluded volume `v(x)`, so
the in-cell populations are the dilute populations reweighted by `exp (-beta·Pi·v)` --
i.e. the exponentially tilted family of `RequestProject.Response` at coupling `-v` and
strength `beta·Pi` (`crowded`, `crowded_zero`).

* `cov_two_point_of_weights` -- the Hoeffding two-point form of a covariance, from which
  `cov_nonneg_of_comonotone` and `cov_pos_of_comonotone`: observables that vary together
  have nonnegative covariance, strictly positive as soon as one populated pair separates
  them.  (Chebyshev's sum inequality in the weighted form needed here.)
* `hasDerivAt_crowdedMean` -- `d⟨f⟩/dPi = -beta · Cov(f, v)`: crowding response is a
  fluctuation of the *dilute* ensemble, so the size of the effect is set by exactly the
  conformational heterogeneity that makes the region disordered.
* `crowding_strictly_compacts` -- if larger conformations exclude more volume, the mean
  size is a strictly decreasing function of the crowder pressure, at every pressure.
  Hence `in_cell_ne_in_vitro`: the measured (dilute) ensemble is *never* the functional
  (in-cell) one, for any disordered region whose members differ in excluded volume.
* `rigid_region_ignores_crowding` -- and the converse: a region with a single excluded
  volume (a folded domain) is untouched, which is why the correction is specific to
  disorder.
* `crowding_is_not_a_temperature` -- an explicit three-state region for which no inverse
  temperature whatsoever reproduces the crowded populations: crowding is a genuinely new
  parameter of the model, not a rescaling of an existing one.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.FreeEnergy
import RequestProject.Response

namespace IDR

open Finset
open scoped Classical

namespace Crowding

variable {n : ℕ}

/-! ## Weighted covariance of observables that vary together -/

/-- Two observables *vary together* (are comonotone / monovarying) when they order the
conformations the same way. -/
def Comonotone (f g : Fin n → ℝ) : Prop := ∀ i j, 0 ≤ (f i - f j) * (g i - g j)

/-- **Hoeffding's two-point identity.**  A covariance under any normalised weight vector is
half the double average of the product of increments. -/
theorem cov_two_point_of_weights {w f g : Fin n → ℝ} (hw : ∑ j, w j = 1) :
    (∑ j, w j * (f j * g j)) - (∑ j, w j * f j) * (∑ j, w j * g j)
      = (1 / 2) * ∑ i, ∑ j, w i * w j * ((f i - f j) * (g i - g j)) := by
  have key : ∀ i : Fin n, ∑ j, w i * w j * ((f i - f j) * (g i - g j))
      = (w i * (f i * g i)) * (∑ j, w j) - (w i * f i) * (∑ j, w j * g j)
        - (w i * g i) * (∑ j, w j * f j) + w i * (∑ j, w j * (f j * g j)) := by
    intro i
    have expand : ∀ j : Fin n, w i * w j * ((f i - f j) * (g i - g j))
        = (w i * (f i * g i)) * w j - (w i * f i) * (w j * g j)
          - (w i * g i) * (w j * f j) + w i * (w j * (f j * g j)) := by
      intro j; ring
    rw [Finset.sum_congr rfl fun j (_ : j ∈ Finset.univ) => expand j,
      Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
      ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum]
  rw [Finset.sum_congr rfl fun i (_ : i ∈ Finset.univ) => key i]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
    ← Finset.sum_mul, ← Finset.sum_mul, ← Finset.sum_mul, ← Finset.sum_mul, hw]
  ring

/-- Comonotone observables have nonnegative covariance (Chebyshev's sum inequality). -/
theorem cov_nonneg_of_comonotone {w f g : Fin n → ℝ} (hw : ∑ j, w j = 1)
    (hwnn : ∀ j, 0 ≤ w j) (hc : Comonotone f g) :
    0 ≤ (∑ j, w j * (f j * g j)) - (∑ j, w j * f j) * (∑ j, w j * g j) := by
  rw [cov_two_point_of_weights hw]
  refine mul_nonneg (by norm_num) (Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => ?_)
  exact mul_nonneg (mul_nonneg (hwnn i) (hwnn j)) (hc i j)

/-- And strictly positive covariance as soon as two *populated* conformations are separated
by both observables. -/
theorem cov_pos_of_comonotone {w f g : Fin n → ℝ} (hw : ∑ j, w j = 1)
    (hwnn : ∀ j, 0 ≤ w j) (hc : Comonotone f g) {i₀ j₀ : Fin n}
    (hi : 0 < w i₀) (hj : 0 < w j₀) (hsep : 0 < (f i₀ - f j₀) * (g i₀ - g j₀)) :
    0 < (∑ j, w j * (f j * g j)) - (∑ j, w j * f j) * (∑ j, w j * g j) := by
  rw [cov_two_point_of_weights hw]
  refine mul_pos (by norm_num) ?_
  have hterm : ∀ i : Fin n, 0 ≤ ∑ j, w i * w j * ((f i - f j) * (g i - g j)) :=
    fun i => Finset.sum_nonneg fun j _ => mul_nonneg (mul_nonneg (hwnn i) (hwnn j)) (hc i j)
  have hinner : 0 < ∑ j, w i₀ * w j * ((f i₀ - f j) * (g i₀ - g j)) := by
    refine Finset.sum_pos' (fun j _ => mul_nonneg (mul_nonneg (hwnn i₀) (hwnn j)) (hc i₀ j))
      ⟨j₀, Finset.mem_univ _, ?_⟩
    exact mul_pos (mul_pos hi hj) hsep
  exact Finset.sum_pos' (fun i _ => hterm i) ⟨i₀, Finset.mem_univ _, hinner⟩

/-! ## The crowded ensemble -/

/-- The in-cell populations: the dilute populations `q` reweighted by the work
`Pi · v(x)` of opening a cavity of the conformation's excluded volume, in units of `kT`. -/
noncomputable def crowded (q v : Fin n → ℝ) (beta Pi : ℝ) : Fin n → ℝ :=
  Response.tilted q (fun j => -v j) (beta * Pi)

/-- The in-cell average of a structural observable `f`. -/
noncomputable def crowdedMean (q v f : Fin n → ℝ) (beta Pi : ℝ) : ℝ :=
  Response.meanObs q (fun j => -v j) f (beta * Pi)

lemma crowdedMean_eq (q v f : Fin n → ℝ) (beta Pi : ℝ) :
    crowdedMean q v f beta Pi = ∑ j, crowded q v beta Pi j * f j := rfl

lemma tilted_pos {q A : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam : ℝ) (j : Fin n) :
    0 < Response.tilted q A lam j :=
  div_pos (mul_pos (hq j) (Real.exp_pos _)) (Response.part_pos hn hq lam)

lemma crowded_pos {q v : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j) (beta Pi : ℝ) (j : Fin n) :
    0 < crowded q v beta Pi j := tilted_pos hn hq _ j

lemma crowded_sum_one {q v : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j) (beta Pi : ℝ) :
    ∑ j, crowded q v beta Pi j = 1 := Response.tilted_sum_one hn hq _

/-- In dilute buffer the crowded ensemble *is* the measured ensemble. -/
lemma crowded_zero {q v : Fin n → ℝ} (hq1 : ∑ j, q j = 1) (beta : ℝ) :
    crowded q v beta 0 = q := by
  funext j
  have hpart : Response.part q (fun j => -v j) 0 = 1 := by
    simp [Response.part, hq1]
  simp [crowded, Response.tilted, mul_zero, hpart]

lemma crowdedMean_zero {q v f : Fin n → ℝ} (hq1 : ∑ j, q j = 1) (beta : ℝ) :
    crowdedMean q v f beta 0 = ∑ j, q j * f j := by
  rw [crowdedMean_eq, crowded_zero hq1]

/-! ## Response to crowding -/

lemma meanObs_neg_right (q A v : Fin n → ℝ) (lam : ℝ) :
    Response.meanObs q A (fun j => -v j) lam = -(Response.meanObs q A v lam) := by
  simp [Response.meanObs, Finset.sum_neg_distrib]

lemma cov_neg_right (q A f v : Fin n → ℝ) (lam : ℝ) :
    Response.cov q A f (fun j => -v j) lam = -(Response.cov q A f v lam) := by
  simp only [Response.cov, meanObs_neg_right, ← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- **Crowding response is a dilute fluctuation.**  The derivative of any structural
average with respect to the crowder pressure is `-beta` times its covariance with the
excluded volume, taken in the ensemble at that pressure. -/
theorem hasDerivAt_crowdedMean {q v f : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j)
    (beta Pi : ℝ) :
    HasDerivAt (crowdedMean q v f beta)
      (-(beta * Response.cov q (fun j => -v j) f v (beta * Pi))) Pi := by
  have hlin := Response.linear_response (q := q) (A := fun j => -v j) (f := f) hn hq (beta * Pi)
  have hmul : HasDerivAt (fun P : ℝ => beta * P) beta Pi := by
    simpa using (hasDerivAt_id Pi).const_mul beta
  have hcomp := hlin.comp Pi hmul
  have hval : Response.cov q (fun j => -v j) f (fun j => -v j) (beta * Pi) * beta
      = -(beta * Response.cov q (fun j => -v j) f v (beta * Pi)) := by
    rw [cov_neg_right]; ring
  rw [← hval]
  exact hcomp

/-- The covariance of a structural observable with the excluded volume is strictly positive
in every crowded ensemble, provided the two vary together and some pair of conformations is
separated by both. -/
theorem cov_size_volume_pos {q v f : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j)
    (hc : Comonotone f v) {i₀ j₀ : Fin n} (hsep : 0 < (f i₀ - f j₀) * (v i₀ - v j₀))
    (lam : ℝ) : 0 < Response.cov q (fun j => -v j) f v lam := by
  rw [Response.cov_eq hn hq]
  have hw := Response.tilted_sum_one (q := q) (A := fun j => -v j) hn hq lam
  have hwnn : ∀ j, 0 ≤ Response.tilted q (fun j => -v j) lam j :=
    fun j => (tilted_pos hn hq lam j).le
  exact cov_pos_of_comonotone hw hwnn hc (tilted_pos hn hq lam i₀) (tilted_pos hn hq lam j₀) hsep

/-- **Crowding strictly compacts.**  If larger conformations exclude more volume, then the
mean size of a disordered region is a strictly decreasing function of the crowder pressure:
there is no pressure at which the effect saturates or reverses. -/
theorem crowding_strictly_compacts {q v f : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j)
    {beta : ℝ} (hbeta : 0 < beta) (hc : Comonotone f v) {i₀ j₀ : Fin n}
    (hsep : 0 < (f i₀ - f j₀) * (v i₀ - v j₀)) :
    StrictAnti (crowdedMean q v f beta) := by
  have hderiv : ∀ Pi : ℝ, HasDerivAt (crowdedMean q v f beta)
      (-(beta * Response.cov q (fun j => -v j) f v (beta * Pi))) Pi :=
    fun Pi => hasDerivAt_crowdedMean hn hq beta Pi
  refine strictAnti_of_deriv_neg fun Pi => ?_
  rw [(hderiv Pi).deriv]
  have := cov_size_volume_pos hn hq hc hsep (beta * Pi)
  nlinarith

/-- **The measured ensemble is not the functional ensemble.**  At any positive crowder
pressure the mean size in the cell is strictly below the dilute value that experiments
report; in particular the two population vectors differ. -/
theorem in_cell_ne_in_vitro {q v f : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j)
    (hq1 : ∑ j, q j = 1) {beta : ℝ} (hbeta : 0 < beta) (hc : Comonotone f v) {i₀ j₀ : Fin n}
    (hsep : 0 < (f i₀ - f j₀) * (v i₀ - v j₀)) {Pi : ℝ} (hPi : 0 < Pi) :
    crowdedMean q v f beta Pi < ∑ j, q j * f j ∧ crowded q v beta Pi ≠ q := by
  have hlt : crowdedMean q v f beta Pi < crowdedMean q v f beta 0 :=
    crowding_strictly_compacts hn hq hbeta hc hsep hPi
  rw [crowdedMean_zero hq1] at hlt
  refine ⟨hlt, fun hEq => ?_⟩
  have : crowdedMean q v f beta Pi = ∑ j, q j * f j := by
    rw [crowdedMean_eq, hEq]
  exact absurd this hlt.ne

/-- Conversely: a region all of whose conformations exclude the same volume -- a folded
domain -- is untouched by crowding at every pressure.  The correction proved above is a
correction specific to conformational heterogeneity. -/
theorem rigid_region_ignores_crowding {q v : Fin n → ℝ} (hq1 : ∑ j, q j = 1) {c : ℝ}
    (hconst : ∀ j, v j = c) (beta Pi : ℝ) : crowded q v beta Pi = q := by
  have hv : (fun j => -v j) = (fun _ : Fin n => -c) := by
    funext i; rw [hconst]
  have hpart : Response.part q (fun _ : Fin n => -c) (beta * Pi)
      = Real.exp (beta * Pi * -c) := by
    simp only [Response.part, ← Finset.sum_mul, hq1, one_mul]
  funext j
  rw [crowded, hv, Response.tilted, hpart, mul_div_assoc,
    div_self (Real.exp_ne_zero _), mul_one]

/-! ## Crowding is a new parameter, not a temperature

A three-state region with equally spaced energies `0, 1, 2` (in units of `kT` at the
reference temperature) has, at *every* inverse temperature, populations in geometric
progression: `p₁² = p₀·p₂`.  Adding a crowder that penalises only the most expanded state
breaks that progression, so no refitted temperature can imitate it. -/

/-- The ladder energies of the witness region. -/
def ladderU : Fin 3 → ℝ := ![0, 1, 2]

/-- The excluded volume of the witness region: only the third (most expanded) conformation
opens an extra cavity. -/
def ladderV : Fin 3 → ℝ := ![0, 0, 1]

/-- **Crowding is a shift of the energy landscape.**  Reweighting Boltzmann populations by
the cavity work returns the Boltzmann populations of the landscape raised by `beta·Pi·v`. -/
lemma crowded_eq_boltz (hn : 0 < n) (U v : Fin n → ℝ) (beta Pi : ℝ) :
    crowded (FreeEnergy.boltz 1 U) v beta Pi
      = FreeEnergy.boltz 1 (fun j => U j + beta * Pi * v j) := by
  have hZ : (0:ℝ) < FreeEnergy.part 1 U := FreeEnergy.part_pos hn _ _
  have hZ' : (0:ℝ) < FreeEnergy.part 1 (fun j => U j + beta * Pi * v j) :=
    FreeEnergy.part_pos hn _ _
  have hterm : ∀ i : Fin n, FreeEnergy.boltz 1 U i * Real.exp (beta * Pi * -v i)
      = Real.exp (-1 * (U i + beta * Pi * v i)) / FreeEnergy.part 1 U := by
    intro i
    rw [FreeEnergy.boltz, div_mul_eq_mul_div]
    congr 1
    rw [← Real.exp_add]
    ring_nf
  have hP : Response.part (FreeEnergy.boltz 1 U) (fun j => -v j) (beta * Pi)
      = FreeEnergy.part 1 (fun j => U j + beta * Pi * v j) / FreeEnergy.part 1 U := by
    simp only [Response.part, FreeEnergy.part, Finset.sum_div]
    exact Finset.sum_congr rfl fun i _ => hterm i
  funext j
  rw [crowded, Response.tilted, hterm j, hP, FreeEnergy.boltz]
  field_simp

/-- The three-term test for a geometric progression, evaluated on Boltzmann populations:
the middle state is over-populated by exactly `exp (b·(W₀ + W₂ − 2W₁))`. -/
lemma boltz_geom_test (b : ℝ) (W : Fin 3 → ℝ) :
    FreeEnergy.boltz b W 1 ^ 2
      = Real.exp (b * (W 0 + W 2 - 2 * W 1))
        * (FreeEnergy.boltz b W 0 * FreeEnergy.boltz b W 2) := by
  have hZ : (0:ℝ) < FreeEnergy.part b W := FreeEnergy.part_pos (by norm_num) _ _
  have key : Real.exp (-b * W 1) ^ 2
      = Real.exp (b * (W 0 + W 2 - 2 * W 1)) * (Real.exp (-b * W 0) * Real.exp (-b * W 2)) := by
    rw [sq, ← Real.exp_add, ← Real.exp_add, ← Real.exp_add]
    ring_nf
  simp only [FreeEnergy.boltz, div_pow, key]
  field_simp

lemma ladder_flat : ladderU 0 + ladderU 2 - 2 * ladderU 1 = 0 := by
  simp [ladderU]

/-- Boltzmann populations on the equally spaced ladder are geometric, at *every*
temperature. -/
lemma boltz_ladder_geometric (beta : ℝ) :
    FreeEnergy.boltz beta ladderU 1 ^ 2
      = FreeEnergy.boltz beta ladderU 0 * FreeEnergy.boltz beta ladderU 2 := by
  rw [boltz_geom_test beta ladderU, ladder_flat]
  simp

/-- The crowded populations of the witness region are *not* geometric: the middle state is
over-populated by exactly the factor `exp (beta·Pi) > 1`. -/
lemma crowded_ladder_not_geometric {beta Pi : ℝ} (hbeta : 0 < beta) (hPi : 0 < Pi) :
    crowded (FreeEnergy.boltz 1 ladderU) ladderV beta Pi 0
        * crowded (FreeEnergy.boltz 1 ladderU) ladderV beta Pi 2
      < crowded (FreeEnergy.boltz 1 ladderU) ladderV beta Pi 1 ^ 2 := by
  set W : Fin 3 → ℝ := fun j => ladderU j + beta * Pi * ladderV j with hW
  have hc := crowded_eq_boltz (n := 3) (by norm_num) ladderU ladderV beta Pi
  rw [hc]
  have hgap : W 0 + W 2 - 2 * W 1 = beta * Pi := by
    simp [hW, ladderU, ladderV]
  have hpos : (0:ℝ) < FreeEnergy.boltz 1 W 0 * FreeEnergy.boltz 1 W 2 :=
    mul_pos (FreeEnergy.boltz_pos (by norm_num) 1 W 0)
      (FreeEnergy.boltz_pos (by norm_num) 1 W 2)
  rw [boltz_geom_test 1 W, hgap]
  have hexp : (1:ℝ) < Real.exp (1 * (beta * Pi)) := by
    rw [one_mul, ← Real.exp_zero]
    exact Real.exp_lt_exp.mpr (mul_pos hbeta hPi)
  nlinarith

/-- **Crowding is not a temperature.**  For the witness region, no inverse temperature
whatsoever reproduces the crowded populations: the in-cell ensemble lies outside the entire
temperature family of the in-vitro landscape.  A model of a disordered region must carry
the crowder pressure as a parameter of its own. -/
theorem crowding_is_not_a_temperature {beta Pi : ℝ} (hbeta : 0 < beta) (hPi : 0 < Pi)
    (beta' : ℝ) :
    crowded (FreeEnergy.boltz 1 ladderU) ladderV beta Pi ≠ FreeEnergy.boltz beta' ladderU := by
  intro hEq
  have h1 := crowded_ladder_not_geometric hbeta hPi
  rw [hEq, boltz_ladder_geometric beta'] at h1
  exact lt_irrefl _ h1

end Crowding

end IDR
