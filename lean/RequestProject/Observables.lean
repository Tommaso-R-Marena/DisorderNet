/-
# Part IX.2  What the experiments actually measure

Every restraint used to build an ensemble model of a disordered region is a *nonlinear*
average of a conformational quantity, and the nonlinearity is severe.  This file proves,
for the three experiments that carry most of the information in the field, exactly what
number comes out of the machine and how it is related to the ensemble.

**Small-angle X-ray scattering.**  `debye` is the Debye scattering function of `N` point
scatterers, `I(q)/I(0) = (1/N²) Σ_{ij} sinc(q r_ij)`.

* `debye_isometry_invariant` and `debye_perm_invariant` -- the intensity depends on the
  configuration only through its *unordered set of pair distances*.  Rigid motions and,
  more importantly, arbitrary relabelling of the monomers leave every scattering curve
  unchanged: SAXS carries no information about which residue is where.
* `guinier` -- the Guinier law with an explicit, non-asymptotic error bound:
  `|I(q)/I(0) - (1 - q²Rg²/3)| ≤ (5/96)(qD)³` whenever `q·D ≤ 1`, `D` the maximum pair
  distance.  So in the Guinier regime the *entire* measurement is one number, `Rg`.

**Paramagnetic relaxation enhancement and the NOE.**  These report an `r^{-6}` average.
`preApparent` is the distance one reads off from it.

* `preApparent_ge_min` and `preApparent_le_of_weight` -- the apparent distance is squeezed
  between the closest approach `d_k` and `w_k^{-1/6}·d_k`.  A conformer of weight 1% that
  brings the spin label to 15 Å forces an apparent distance below 32 Å *whatever the other
  99% of the ensemble does*: the observable is a minority-report, not an average.
* `preApparent_le_mean` -- Jensen: the apparent distance never exceeds the mean distance,
  so an `r^{-6}` restraint interpreted as a mean distance always makes the model too
  compact.  This is the formal content of the standard warning about NOE/PRE-derived
  "distances" in disordered systems.

**Single-molecule FRET.** `fretEff` is the transfer efficiency.
* `fret_ensemble_ne_mean` -- an explicit symmetric two-state ensemble whose mean distance is
  `R₀` transfers with efficiency `> 1/2`: the efficiency of the mean conformation.  Inverting
  a measured efficiency with the single-molecule formula therefore returns a distance that
  is not the mean distance of the ensemble, and the discrepancy is a *property of the width*
  of the ensemble, i.e. of exactly the thing that makes the region disordered.

The common design conclusion (`RequestProject.PartNine`): each experiment is a strongly
nonlinear functional of the ensemble, so restraints must be applied by *forward-modelling
the observable from the candidate ensemble*, never by fitting structures to "experimental
distances"; and the forward models here are the exact ones.
-/
import Mathlib
import RequestProject.Polymer

namespace IDR

open Finset

namespace Observables

/-! ## Small-angle scattering -/

/-- Third-order Taylor bound for the scattering kernel `sinc`. -/
theorem sinc_taylor_bound {x : ℝ} (hx : |x| ≤ 1) :
    |Real.sinc x - (1 - x ^ 2 / 6)| ≤ 5 / 96 * |x| ^ 3 := by
  rcases eq_or_ne x 0 with h | h
  · simp [h, Real.sinc]
  · have hx0 : |x| > 0 := abs_pos.mpr h
    have hs : Real.sinc x = Real.sin x / x := by simp [Real.sinc, h]
    have key : Real.sinc x - (1 - x ^ 2 / 6) = (Real.sin x - (x - x ^ 3 / 6)) / x := by
      rw [hs]; field_simp
    rw [key, abs_div, div_le_iff₀ hx0]
    calc |Real.sin x - (x - x ^ 3 / 6)| ≤ |x| ^ 4 * (5 / 96) := Real.sin_bound hx
      _ = 5 / 96 * |x| ^ 3 * |x| := by ring

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The Debye scattering function of `N` identical point scatterers, normalised to `1` at
`q = 0`: `I(q)/I(0) = (1/N²) Σ_{i,j} sinc(q‖r_i - r_j‖)`. -/
noncomputable def debye {N : ℕ} (q : ℝ) (r : Fin N → E) : ℝ :=
  (1 / (N : ℝ) ^ 2) * ∑ i, ∑ j, Real.sinc (q * ‖r i - r j‖)

omit [InnerProductSpace ℝ E] in
@[simp] lemma debye_zero {N : ℕ} (hN : 0 < N) (r : Fin N → E) : debye 0 r = 1 := by
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  unfold debye
  simp [Real.sinc]
  field_simp

/-- **Scattering is blind to rigid motions.** -/
theorem debye_isometry_invariant {N : ℕ} (q : ℝ) (r : Fin N → E) (g : E ≃ₗᵢ[ℝ] E) (v : E) :
    debye q (fun i => g (r i) + v) = debye q r := by
  unfold debye
  refine congrArg _ (Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun j _ => ?_)))
  congr 2
  rw [show g (r i) + v - (g (r j) + v) = g (r i - r j) by simp [map_sub]]
  exact g.norm_map _

omit [InnerProductSpace ℝ E] in
/-- **Scattering is blind to the identity of the monomers.**  Relabelling the chain leaves
every scattering curve unchanged, so SAXS constrains the distribution of pair distances and
nothing else. -/
theorem debye_perm_invariant {N : ℕ} (q : ℝ) (r : Fin N → E) (s : Equiv.Perm (Fin N)) :
    debye q (r ∘ s) = debye q r := by
  unfold debye
  congr 1
  simp only [Function.comp_apply]
  rw [← Equiv.sum_comp s (fun i => ∑ j, Real.sinc (q * ‖r i - r j‖))]
  exact Finset.sum_congr rfl
    (fun i _ => Equiv.sum_comp s (fun j => Real.sinc (q * ‖r (s i) - r j‖)))

/-- **The Guinier law, with an explicit error bound.**  If all pair distances are at most `D`
and `q·D ≤ 1`, then the scattering curve is `1 - q²Rg²/3` up to `(5/96)(qD)³`.  In the
Guinier regime the whole experiment is the single number `Rg`. -/
theorem guinier {N : ℕ} (hN : 0 < N) (r : Fin N → E) {q D : ℝ} (hq : 0 ≤ q)
    (hr : ∀ i j, ‖r i - r j‖ ≤ D) (hqD : q * D ≤ 1) :
    |debye q r - (1 - q ^ 2 * Polymer.gyrationSq r / 3)| ≤ 5 / 96 * (q * D) ^ 3 := by
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  have hNpos : (0 : ℝ) < (N : ℝ) ^ 2 := by positivity
  -- rewrite the Guinier polynomial as an average over pairs
  have hRg : ∑ i, ∑ j, ‖r i - r j‖ ^ 2 = 2 * (N : ℝ) ^ 2 * Polymer.gyrationSq r := by
    rw [Polymer.gyration_eq_pair_sum hN r]
    field_simp
  have hpoly : 1 - q ^ 2 * Polymer.gyrationSq r / 3
      = (1 / (N : ℝ) ^ 2) * ∑ i, ∑ j, (1 - (q * ‖r i - r j‖) ^ 2 / 6) := by
    have hconst : ∑ i, ∑ j, (q * ‖r i - r j‖) ^ 2 / 6
        = q ^ 2 / 6 * ∑ i, ∑ j, ‖r i - r j‖ ^ 2 := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl (fun i _ => ?_)
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl (fun j _ => by ring)
    have hsplit : ∑ i, ∑ j, (1 - (q * ‖r i - r j‖) ^ 2 / 6)
        = (∑ _i : Fin N, ∑ _j : Fin N, (1 : ℝ)) - ∑ i, ∑ j, (q * ‖r i - r j‖) ^ 2 / 6 := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl (fun i _ => Finset.sum_sub_distrib _ _)
    have hone : (∑ _i : Fin N, ∑ _j : Fin N, (1 : ℝ)) = (N : ℝ) ^ 2 := by
      simp [Finset.sum_const, Finset.card_univ]
      ring
    have : ∑ i, ∑ j, (1 - (q * ‖r i - r j‖) ^ 2 / 6)
        = (N : ℝ) ^ 2 - q ^ 2 / 6 * ∑ i, ∑ j, ‖r i - r j‖ ^ 2 := by
      rw [hsplit, hone, hconst]
    rw [this, hRg]
    field_simp
    ring
  -- the difference is an average of the pointwise Taylor errors
  have hdiff : debye q r - (1 - q ^ 2 * Polymer.gyrationSq r / 3)
      = (1 / (N : ℝ) ^ 2) *
        ∑ i, ∑ j, (Real.sinc (q * ‖r i - r j‖) - (1 - (q * ‖r i - r j‖) ^ 2 / 6)) := by
    unfold debye
    rw [hpoly, ← mul_sub]
    congr 1
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun i _ => (Finset.sum_sub_distrib _ _).symm)
  rw [hdiff, abs_mul, abs_of_nonneg (by positivity : (0:ℝ) ≤ 1 / (N : ℝ) ^ 2)]
  have hterm : ∀ i j : Fin N,
      |Real.sinc (q * ‖r i - r j‖) - (1 - (q * ‖r i - r j‖) ^ 2 / 6)| ≤ 5 / 96 * (q * D) ^ 3 := by
    intro i j
    have hx0 : 0 ≤ q * ‖r i - r j‖ := by positivity
    have hxD : q * ‖r i - r j‖ ≤ q * D := by
      exact mul_le_mul_of_nonneg_left (hr i j) hq
    have habs : |q * ‖r i - r j‖| ≤ 1 := by
      rw [abs_of_nonneg hx0]; linarith
    refine le_trans (sinc_taylor_bound habs) ?_
    rw [abs_of_nonneg hx0]
    have : (q * ‖r i - r j‖) ^ 3 ≤ (q * D) ^ 3 := by
      exact pow_le_pow_left₀ hx0 hxD 3
    linarith
  have hsum : |∑ i, ∑ j, (Real.sinc (q * ‖r i - r j‖) - (1 - (q * ‖r i - r j‖) ^ 2 / 6))|
      ≤ (N : ℝ) ^ 2 * (5 / 96 * (q * D) ^ 3) := by
    refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
    have : ∀ i : Fin N, |∑ j, (Real.sinc (q * ‖r i - r j‖) - (1 - (q * ‖r i - r j‖) ^ 2 / 6))|
        ≤ (N : ℝ) * (5 / 96 * (q * D) ^ 3) := by
      intro i
      refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
      calc ∑ j, |Real.sinc (q * ‖r i - r j‖) - (1 - (q * ‖r i - r j‖) ^ 2 / 6)|
          ≤ ∑ _j : Fin N, (5 / 96 * (q * D) ^ 3) :=
            Finset.sum_le_sum (fun j _ => hterm i j)
        _ = (N : ℝ) * (5 / 96 * (q * D) ^ 3) := by
            simp [Finset.sum_const, Finset.card_univ]
    calc ∑ i, |∑ j, (Real.sinc (q * ‖r i - r j‖) - (1 - (q * ‖r i - r j‖) ^ 2 / 6))|
        ≤ ∑ _i : Fin N, ((N : ℝ) * (5 / 96 * (q * D) ^ 3)) := Finset.sum_le_sum (fun i _ => this i)
      _ = (N : ℝ) ^ 2 * (5 / 96 * (q * D) ^ 3) := by
          simp [Finset.sum_const, Finset.card_univ]; ring
  calc 1 / (N : ℝ) ^ 2 *
        |∑ i, ∑ j, (Real.sinc (q * ‖r i - r j‖) - (1 - (q * ‖r i - r j‖) ^ 2 / 6))|
      ≤ 1 / (N : ℝ) ^ 2 * ((N : ℝ) ^ 2 * (5 / 96 * (q * D) ^ 3)) := by
        exact mul_le_mul_of_nonneg_left hsum (by positivity)
    _ = 5 / 96 * (q * D) ^ 3 := by field_simp

/-! ## `r^{-6}` averaging: PRE and the NOE -/

variable {m : ℕ}

/-- The apparent distance read off an `r^{-6}`-averaged measurement (PRE, NOE) from an
ensemble with weights `w` and distances `d`. -/
noncomputable def preApparent (w d : Fin m → ℝ) : ℝ :=
  (∑ i, w i * (d i) ^ (-6 : ℤ)) ^ (-(1 : ℝ) / 6)

/-- Taking the `-1/6` power undoes the `r^{-6}` average of a single distance. -/
lemma rpow_neg_sixth {x : ℝ} (hx : 0 < x) : (x ^ (-6 : ℤ)) ^ (-(1 : ℝ) / 6) = x := by
  rw [← Real.rpow_intCast x (-6), ← Real.rpow_mul hx.le]
  norm_num

lemma exists_pos_weight {w : Fin m → ℝ} (hw : ∀ i, 0 ≤ w i) (hsum : ∑ i, w i = 1) :
    ∃ k, 0 < w k := by
  by_contra hcon
  push_neg at hcon
  have h0 : ∑ i, w i = 0 := Finset.sum_eq_zero (fun i _ => le_antisymm (hcon i) (hw i))
  rw [hsum] at h0
  norm_num at h0

lemma preSum_pos {w d : Fin m → ℝ} (hw : ∀ i, 0 ≤ w i) (hsum : ∑ i, w i = 1)
    (hd : ∀ i, 0 < d i) : 0 < ∑ i, w i * (d i) ^ (-6 : ℤ) := by
  obtain ⟨k, hk⟩ := exists_pos_weight hw hsum
  refine lt_of_lt_of_le (mul_pos hk (zpow_pos (hd k) _)) (Finset.single_le_sum
    (fun i _ => mul_nonneg (hw i) (zpow_pos (hd i) _).le) (Finset.mem_univ k))

/-- **A single compact conformer dictates the reading.**  If conformer `k` has weight `w k`
and distance `d k`, the apparent distance is at most `w k ^{-1/6} · d k`, no matter how
expanded the rest of the ensemble is. -/
theorem preApparent_le_of_weight {w d : Fin m → ℝ} (hw : ∀ i, 0 ≤ w i) (hd : ∀ i, 0 < d i)
    (k : Fin m) (hk : 0 < w k) :
    preApparent w d ≤ (w k) ^ (-(1 : ℝ) / 6) * d k := by
  have hk' : 0 < w k * (d k) ^ (-6 : ℤ) := mul_pos hk (zpow_pos (hd k) _)
  have hge : w k * (d k) ^ (-6 : ℤ) ≤ ∑ i, w i * (d i) ^ (-6 : ℤ) :=
    Finset.single_le_sum
      (fun i _ => mul_nonneg (hw i) (zpow_pos (hd i) _).le) (Finset.mem_univ k)
  have h1 : preApparent w d ≤ (w k * (d k) ^ (-6 : ℤ)) ^ (-(1 : ℝ) / 6) :=
    Real.rpow_le_rpow_of_nonpos hk' hge (by norm_num)
  refine le_trans h1 (le_of_eq ?_)
  rw [Real.mul_rpow hk.le (zpow_pos (hd k) _).le, rpow_neg_sixth (hd k)]

/-- The apparent distance is never smaller than the closest approach in the ensemble.  With
`preApparent_le_of_weight` this pins the reading to the interval
`[d_k, w_k^{-1/6} d_k]` around the *compact* member of the ensemble. -/
theorem preApparent_ge_min {w d : Fin m → ℝ} (hw : ∀ i, 0 ≤ w i) (hsum : ∑ i, w i = 1)
    (hd : ∀ i, 0 < d i) {c : ℝ} (hc : 0 < c) (hmin : ∀ i, c ≤ d i) :
    c ≤ preApparent w d := by
  have hz : ∀ y : ℝ, 0 < y → y ^ (-6 : ℤ) = (y ^ (6 : ℕ))⁻¹ := by
    intro y _
    rw [show (-6 : ℤ) = -(6 : ℕ) by norm_num, zpow_neg, zpow_natCast]
  have hanti : ∀ i, (d i) ^ (-6 : ℤ) ≤ c ^ (-6 : ℤ) := by
    intro i
    rw [hz _ (hd i), hz _ hc]
    exact inv_anti₀ (by positivity) (pow_le_pow_left₀ hc.le (hmin i) 6)
  have hle : ∑ i, w i * (d i) ^ (-6 : ℤ) ≤ c ^ (-6 : ℤ) := by
    calc ∑ i, w i * (d i) ^ (-6 : ℤ) ≤ ∑ i, w i * c ^ (-6 : ℤ) :=
          Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left (hanti i) (hw i))
      _ = c ^ (-6 : ℤ) := by rw [← Finset.sum_mul, hsum, one_mul]
  have hSpos : 0 < ∑ i, w i * (d i) ^ (-6 : ℤ) := preSum_pos hw hsum hd
  have h1 : (c ^ (-6 : ℤ)) ^ (-(1 : ℝ) / 6) ≤ preApparent w d :=
    Real.rpow_le_rpow_of_nonpos hSpos hle (by norm_num)
  rwa [rpow_neg_sixth hc] at h1

/-- **Jensen: the `r^{-6}` reading is always too compact.**  The apparent distance never
exceeds the mean distance of the ensemble. -/
theorem preApparent_le_mean {w d : Fin m → ℝ} (hw : ∀ i, 0 ≤ w i) (hsum : ∑ i, w i = 1)
    (hd : ∀ i, 0 < d i) :
    preApparent w d ≤ ∑ i, w i * d i := by
  obtain ⟨k, hk⟩ := exists_pos_weight hw hsum
  have hmeanpos : 0 < ∑ i, w i * d i :=
    lt_of_lt_of_le (mul_pos hk (hd k)) (Finset.single_le_sum
      (fun i _ => mul_nonneg (hw i) (hd i).le) (Finset.mem_univ k))
  have hconv : ConvexOn ℝ (Set.Ioi (0 : ℝ)) (fun x : ℝ => x ^ (-6 : ℤ)) := convexOn_zpow (-6)
  have hjensen : (∑ i, w i * d i) ^ (-6 : ℤ) ≤ ∑ i, w i * (d i) ^ (-6 : ℤ) := by
    have hcm := hconv.map_centerMass_le (t := Finset.univ) (w := w) (p := d)
      (fun i _ => hw i) (by rw [hsum]; norm_num) (fun i _ => Set.mem_Ioi.mpr (hd i))
    simpa [Finset.centerMass, hsum, smul_eq_mul, Function.comp] using hcm
  have h1 : preApparent w d ≤ ((∑ i, w i * d i) ^ (-6 : ℤ)) ^ (-(1 : ℝ) / 6) :=
    Real.rpow_le_rpow_of_nonpos (zpow_pos hmeanpos _) hjensen (by norm_num)
  rwa [rpow_neg_sixth hmeanpos] at h1

/-! ## FRET -/

/-- The Förster transfer efficiency at a donor–acceptor distance `x` in units of `R₀`. -/
noncomputable def fretEff (x : ℝ) : ℝ := 1 / (1 + x ^ 6)

/-- **The transfer efficiency of an ensemble is not the efficiency of its mean.**  A
symmetric two-state ensemble with distances `R₀/2` and `3R₀/2` -- mean distance exactly `R₀`
-- transfers with efficiency strictly greater than `fretEff 1 = 1/2`.  Inverting the
measured efficiency therefore yields a distance strictly smaller than the mean distance:
the bias is a property of the *width* of the distribution. -/
theorem fret_ensemble_ne_mean :
    (1 / 2) * fretEff (1 / 2) + (1 / 2) * fretEff (3 / 2) > fretEff 1 := by
  unfold fretEff
  norm_num

end Observables

end IDR
