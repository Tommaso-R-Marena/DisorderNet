/-
# Part CXXVI  Salt screening: the exact crossover that destroys charge patterning

Parts CXX–CXXV solved a charged disordered region exactly and showed that its dimensions, its
dielectric response and its residual conformational freedom are all controlled by the *order*
of its charges rather than by their composition.  Every one of those statements was made at
zero ionic strength: the coupling between two charges was taken to be the bare, unscreened
kernel `w d = d` inherited from the ideal chain.

Real disordered regions live in salt.  Part LXXIX (`Screening.lean`) already analysed the salt
dependence of the pairwise model in the fugacity variable, and showed that at high salt only the
nearest-neighbour charge correlation survives, with reentrance as a consequence of the sign
pattern of the autocorrelation.  What is missing there, and supplied here, is the *size* of the
patterning effect as a function of the screening length: a bound uniform over all patterns, the
unscreened contrast it must be compared with, and the crossover between them.  The kernel is

  `kern κ d = d · exp (−κ d)`,

the same linear kernel damped by the Debye factor at inverse screening length `κ`, and the
energy is the pairwise charge energy of Part LXXIII with that kernel,
`energy N κ q = ∑_{i<j} kern κ (j−i) · q i · q j`.

* `energy_zero` — at `κ = 0` the model *is* the unscreened one: for a neutral sequence the
  energy is exactly minus the sum of squared prefix charges, the functional of Part CXX.
* `abs_energy_le` — **the screening bound.**  For every `κ > 0` and every sequence of unit
  charges, `|energy N κ q| ≤ 4 N / κ²`.  The energy of a screened chain is at most *linear*
  in the length of the region, whatever its pattern, and it vanishes as the salt concentration
  grows (`energy_tendsto_zero`).
* `unscreened_gap_ge` — at zero salt the diblock and the perfectly mixed sequence, which have
  identical composition, differ by at least `t³/3 − t`: the contrast is *cubic* in the length.
* `screened_gap_le` — at inverse screening length `κ` that same contrast is at most
  `16 t / κ²`.
* `patterning_needs_long_screening_length` — putting the two together: if the screened contrast
  is even half the unscreened one, then `κ · t ≤ 12`.  **Charge patterning is visible only while
  the Debye screening length is comparable to the length of the region itself**; at higher salt
  two sequences of identical composition become thermodynamically indistinguishable, however
  differently their charges are arranged.
* `patterning_survives_at_low_salt`, `crossover_two_sided` — the crossover in both directions:
  for `κ ≤ 1/(288 t)` the contrast is *still* at least half its unscreened value, while for
  `κ t > 12` it is less than half.  The transition therefore happens at `κ t` of order one.
* `no_salt_blind_prediction` — the modelling consequence, quantitatively.  Any predictor that
  assigns an energy to a sequence *and to a sequence only* is wrong by at least `(t³/3 − t)/8`
  on one of the four sequence/condition pairs above.

The consequence for model design is the same one this development has been making throughout,
now in its sharpest quantitative form: a sequence-patterning parameter is not an intrinsic
property of a disordered region.  It is a property of the region *in a solution condition*, and
a model that reports it without carrying the ionic strength as an explicit context variable is
predicting a number whose true value it has not been told.
-/
import Mathlib
import RequestProject.ChargePatterning
import RequestProject.ChargePatterningExact

set_option autoImplicit false

namespace IDR
namespace Salt

open Finset

/-! ## 1. The screened kernel -/

/-- The Debye-screened linear kernel: the ideal-chain coupling `d` between two charges `d`
residues apart, damped by the screening factor `exp (−κ d)`. -/
noncomputable def kern (kappa : ℝ) (d : ℕ) : ℝ := (d : ℝ) * Real.exp (-(kappa * d))

/-- The screened pairwise charge energy of a sequence `q` of `N` residues. -/
noncomputable def energy (N : ℕ) (kappa : ℝ) (q : ℕ → ℝ) : ℝ :=
  Pattern.pairEnergy N (kern kappa) q

lemma kern_zero (d : ℕ) : kern 0 d = (d : ℝ) := by simp [kern]

lemma kern_nonneg (kappa : ℝ) (d : ℕ) : 0 ≤ kern kappa d :=
  mul_nonneg (Nat.cast_nonneg d) (Real.exp_nonneg _)

/-! ## 2. At zero salt: the unscreened functional of Part CXX -/

/-- **At zero ionic strength the model is the unscreened one.**  For a neutral sequence the
energy is exactly minus the sum of squared prefix charges. -/
theorem energy_zero (N : ℕ) (q : ℕ → ℝ) (hQ : Charge.pre q N = 0) :
    energy N 0 q = -∑ k ∈ range N, (Charge.pre q k) ^ 2 := by
  have hrw : energy N 0 q
      = ∑ j ∈ range N, ∑ i ∈ range j, q i * q j * ((j : ℝ) - i) := by
    unfold energy Pattern.pairEnergy
    refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun i hi => ?_
    have hij : i < j := Finset.mem_range.1 hi
    rw [kern_zero]
    have : ((j - i : ℕ) : ℝ) = (j : ℝ) - i := by
      have : (i : ℝ) ≤ j := by exact_mod_cast hij.le
      push_cast [Nat.cast_sub hij.le]
      ring
    rw [this]; ring
  rw [hrw, Charge.neutral_lin_kernel q N hQ]

/-! ## 3. The screening bound -/

/-- Each screened coupling is dominated by a geometric term with ratio `exp (−κ/2)`. -/
lemma kern_le (kappa : ℝ) (hk : 0 < kappa) (d : ℕ) :
    kern kappa d ≤ (2 / kappa) * (Real.exp (-(kappa / 2))) ^ d := by
  have hx : (0 : ℝ) ≤ kappa * d / 2 := by positivity
  have hxe : kappa * d / 2 ≤ Real.exp (kappa * d / 2) := by
    have := Real.add_one_le_exp (kappa * d / 2)
    linarith
  have hexp : Real.exp (-(kappa / 2)) ^ d = Real.exp (-(kappa * d / 2)) := by
    rw [← Real.exp_nat_mul]
    ring_nf
  have hpos : 0 < Real.exp (kappa * d / 2) := Real.exp_pos _
  have hsplit : Real.exp (-(kappa * d)) = Real.exp (-(kappa * d / 2)) * Real.exp (-(kappa * d / 2)) := by
    rw [← Real.exp_add]; ring_nf
  have hinv : Real.exp (-(kappa * d / 2)) = (Real.exp (kappa * d / 2))⁻¹ :=
    Real.exp_neg _
  have key : (d : ℝ) * Real.exp (-(kappa * d / 2)) ≤ 2 / kappa := by
    rw [hinv, ← div_eq_mul_inv, div_le_div_iff₀ hpos hk]
    nlinarith [hxe]
  rw [hexp, kern, hsplit, ← mul_assoc]
  have hnn : 0 ≤ Real.exp (-(kappa * d / 2)) := Real.exp_nonneg _
  exact mul_le_mul_of_nonneg_right key hnn

/-- The geometric tail bound used to sum the screened couplings along the chain. -/
lemma geom_tail_le {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r < 1) :
    ∀ j : ℕ, ∑ i ∈ range j, r ^ (j - i) ≤ r / (1 - r) := by
  have h1r : 0 < 1 - r := by linarith
  intro j
  induction j with
  | zero => simp; positivity
  | succ j ih =>
      have hstep : ∑ i ∈ range (j + 1), r ^ (j + 1 - i)
          = r * ∑ i ∈ range j, r ^ (j - i) + r := by
        rw [Finset.sum_range_succ]
        have h1 : ∑ i ∈ range j, r ^ (j + 1 - i) = r * ∑ i ∈ range j, r ^ (j - i) := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun i hi => ?_
          have hij : i < j := Finset.mem_range.1 hi
          have : j + 1 - i = (j - i) + 1 := by omega
          rw [this, pow_succ]
          ring
        rw [h1]
        simp
      rw [hstep]
      have : r * ∑ i ∈ range j, r ^ (j - i) ≤ r * (r / (1 - r)) :=
        mul_le_mul_of_nonneg_left ih hr0
      have hfin : r * (r / (1 - r)) + r = r / (1 - r) := by
        field_simp
        ring
      linarith

/-- **The screening bound.**  At inverse screening length `κ > 0` the pairwise charge energy of
any sequence of unit charges is at most `4 N / κ²` in absolute value: linear in the length of
the region, and uniformly over all patterns. -/
theorem abs_energy_le {N : ℕ} {kappa : ℝ} (hk : 0 < kappa) {q : ℕ → ℝ}
    (hq : ∀ i, |q i| ≤ 1) :
    |energy N kappa q| ≤ 4 * N / kappa ^ 2 := by
  set r : ℝ := Real.exp (-(kappa / 2)) with hr
  have hr0 : 0 < r := Real.exp_pos _
  have hr1 : r < 1 := by
    rw [hr]
    have : -(kappa / 2) < 0 := by linarith
    calc Real.exp (-(kappa / 2)) < Real.exp 0 := Real.exp_lt_exp.2 this
      _ = 1 := Real.exp_zero
  -- geometric tail in terms of `κ`
  have htail : r / (1 - r) ≤ 2 / kappa := by
    have hinv : r = (Real.exp (kappa / 2))⁻¹ := by
      rw [hr, ← Real.exp_neg]
    have hE : kappa / 2 + 1 ≤ Real.exp (kappa / 2) := Real.add_one_le_exp _
    have hEpos : 0 < Real.exp (kappa / 2) := Real.exp_pos _
    have h1r : 0 < 1 - r := by linarith
    rw [div_le_div_iff₀ h1r hk, hinv]
    have : (Real.exp (kappa / 2))⁻¹ * kappa ≤ 2 * (1 - (Real.exp (kappa / 2))⁻¹) := by
      rw [inv_mul_eq_div, div_le_iff₀ hEpos]
      have hx : (Real.exp (kappa / 2))⁻¹ * Real.exp (kappa / 2) = 1 :=
        inv_mul_cancel₀ (ne_of_gt hEpos)
      nlinarith [hE, hEpos]
    exact this
  -- bound the inner sum for each `j`
  have hinner : ∀ j ∈ range N, |∑ i ∈ range j, kern kappa (j - i) * (q i * q j)|
      ≤ (2 / kappa) * (2 / kappa) := by
    intro j _
    have h1 : |∑ i ∈ range j, kern kappa (j - i) * (q i * q j)|
        ≤ ∑ i ∈ range j, |kern kappa (j - i) * (q i * q j)| :=
      Finset.abs_sum_le_sum_abs _ _
    have h2 : ∀ i ∈ range j, |kern kappa (j - i) * (q i * q j)| ≤ (2 / kappa) * r ^ (j - i) := by
      intro i _
      rw [abs_mul, abs_of_nonneg (kern_nonneg kappa (j - i))]
      have hqq : |q i * q j| ≤ 1 := by
        rw [abs_mul]
        have := hq i; have := hq j
        nlinarith [abs_nonneg (q i), abs_nonneg (q j), hq i, hq j]
      calc kern kappa (j - i) * |q i * q j| ≤ kern kappa (j - i) * 1 :=
            mul_le_mul_of_nonneg_left hqq (kern_nonneg _ _)
        _ = kern kappa (j - i) := by ring
        _ ≤ (2 / kappa) * r ^ (j - i) := kern_le kappa hk (j - i)
    have h3 : ∑ i ∈ range j, |kern kappa (j - i) * (q i * q j)|
        ≤ ∑ i ∈ range j, (2 / kappa) * r ^ (j - i) := Finset.sum_le_sum h2
    have h4 : ∑ i ∈ range j, (2 / kappa) * r ^ (j - i) ≤ (2 / kappa) * (r / (1 - r)) := by
      rw [← Finset.mul_sum]
      exact mul_le_mul_of_nonneg_left (geom_tail_le hr0.le hr1 j) (by positivity)
    have h5 : (2 / kappa) * (r / (1 - r)) ≤ (2 / kappa) * (2 / kappa) :=
      mul_le_mul_of_nonneg_left htail (by positivity)
    linarith
  have hsum : |energy N kappa q| ≤ ∑ _j ∈ range N, (2 / kappa) * (2 / kappa) := by
    unfold energy Pattern.pairEnergy
    calc |∑ j ∈ range N, ∑ i ∈ range j, kern kappa (j - i) * (q i * q j)|
        ≤ ∑ j ∈ range N, |∑ i ∈ range j, kern kappa (j - i) * (q i * q j)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _j ∈ range N, (2 / kappa) * (2 / kappa) := Finset.sum_le_sum hinner
  have hne : kappa ≠ 0 := ne_of_gt hk
  have hconst : ∑ _j ∈ range N, (2 / kappa) * (2 / kappa) = 4 * N / kappa ^ 2 := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    field_simp
    ring
  linarith [hsum, hconst.le, hconst.ge]

/-- **Screening destroys the energy altogether.**  At fixed length, the screened charge energy of
every unit-charge sequence tends to zero as the salt concentration grows. -/
theorem energy_tendsto_zero (N : ℕ) (q : ℕ → ℝ) (hq : ∀ i, |q i| ≤ 1) :
    Filter.Tendsto (fun kappa : ℝ => energy N kappa q) Filter.atTop (nhds 0) := by
  have hbound : Filter.Tendsto (fun kappa : ℝ => 4 * N / kappa ^ 2) Filter.atTop (nhds 0) := by
    have h2 : Filter.Tendsto (fun kappa : ℝ => kappa ^ 2) Filter.atTop Filter.atTop :=
      Filter.tendsto_pow_atTop (by norm_num)
    exact Filter.Tendsto.div_atTop tendsto_const_nhds h2
  refine squeeze_zero_norm' ?_ hbound
  filter_upwards [Filter.eventually_gt_atTop (0 : ℝ)] with kappa hkappa
  simpa [Real.norm_eq_abs] using abs_energy_le hkappa hq

/-! ## 4. The patterning contrast, screened and unscreened -/

open Charge

/-- The charge sequences of Part CXX, as real-valued sequences. -/
noncomputable def altR : ℕ → ℝ := fun k => (Charge.alt k : ℝ)

/-- The diblock sequence of Part CXX, as a real-valued sequence. -/
noncomputable def blkR (t : ℕ) : ℕ → ℝ := fun k => (Charge.blk t k : ℝ)

lemma pre_altR (k : ℕ) : Charge.pre altR k = ((Charge.preZ Charge.alt k : ℤ) : ℝ) :=
  Charge.pre_cast Charge.alt k

lemma pre_blkR (t k : ℕ) : Charge.pre (blkR t) k = ((Charge.preZ (Charge.blk t) k : ℤ) : ℝ) :=
  Charge.pre_cast (Charge.blk t) k

lemma altR_abs (k : ℕ) : |altR k| ≤ 1 := by
  unfold altR Charge.alt
  split <;> simp

lemma blkR_abs (t k : ℕ) : |blkR t k| ≤ 1 := by
  unfold blkR Charge.blk
  split <;> simp

/-- The patterning contrast: the energy difference between the perfectly mixed and the diblock
sequence, which have identical composition. -/
noncomputable def gap (t : ℕ) (kappa : ℝ) : ℝ :=
  energy (2 * t) kappa altR - energy (2 * t) kappa (blkR t)

/-- **At zero salt the contrast is cubic in the length of the region.** -/
theorem unscreened_gap_ge (t : ℕ) : (t : ℝ) ^ 3 / 3 - t ≤ gap t 0 := by
  have haltn : Charge.pre altR (2 * t) = 0 := by
    rw [pre_altR, Charge.alt_neutral (m := 2 * t) ⟨t, by ring⟩]
    simp
  have hblkn : Charge.pre (blkR t) (2 * t) = 0 := by
    rw [pre_blkR, Charge.blk_neutral t]
    simp
  have halt : energy (2 * t) 0 altR = -((t : ℝ)) := by
    rw [energy_zero _ _ haltn]
    have hs : ∑ k ∈ range (2 * t), (Charge.pre altR k) ^ 2
        = ((∑ k ∈ range (2 * t), (Charge.preZ Charge.alt k) ^ 2 : ℤ) : ℝ) := by
      push_cast
      exact Finset.sum_congr rfl fun k _ => by rw [pre_altR k]
    rw [hs, Charge.sum_pre_sq_alt (2 * t)]
    have hhalf : (2 * t) / 2 = t := by omega
    rw [hhalf]
    push_cast
    ring
  have hblk : energy (2 * t) 0 (blkR t)
      = -((∑ k ∈ range (2 * t), (Charge.preZ (Charge.blk t) k) ^ 2 : ℤ) : ℝ) := by
    rw [energy_zero _ _ hblkn]
    have hs : ∑ k ∈ range (2 * t), (Charge.pre (blkR t) k) ^ 2
        = ((∑ k ∈ range (2 * t), (Charge.preZ (Charge.blk t) k) ^ 2 : ℤ) : ℝ) := by
      push_cast
      exact Finset.sum_congr rfl fun k _ => by rw [pre_blkR t k]
    rw [hs]
  have hcube : (t : ℝ) ^ 3 ≤ 3 * ((∑ k ∈ range (2 * t), (Charge.preZ (Charge.blk t) k) ^ 2 : ℤ) : ℝ) := by
    have := Charge.sum_pre_sq_blk_ge t
    exact_mod_cast this
  unfold gap
  rw [halt, hblk]
  linarith

/-- **At inverse screening length `κ` the same contrast is at most `16 t / κ²`.** -/
theorem screened_gap_le {t : ℕ} {kappa : ℝ} (hk : 0 < kappa) :
    |gap t kappa| ≤ 16 * t / kappa ^ 2 := by
  have h1 : |energy (2 * t) kappa altR| ≤ 4 * (2 * t : ℕ) / kappa ^ 2 :=
    abs_energy_le hk altR_abs
  have h2 : |energy (2 * t) kappa (blkR t)| ≤ 4 * (2 * t : ℕ) / kappa ^ 2 :=
    abs_energy_le hk (blkR_abs t)
  have hc : ((2 * t : ℕ) : ℝ) = 2 * t := by push_cast; ring
  rw [hc] at h1 h2
  have : |gap t kappa| ≤ |energy (2 * t) kappa altR| + |energy (2 * t) kappa (blkR t)| := by
    unfold gap
    exact abs_sub _ _
  have hk2 : 0 < kappa ^ 2 := by positivity
  have : |gap t kappa| ≤ 4 * (2 * t) / kappa ^ 2 + 4 * (2 * t) / kappa ^ 2 := by linarith
  calc |gap t kappa| ≤ 4 * (2 * (t : ℝ)) / kappa ^ 2 + 4 * (2 * (t : ℝ)) / kappa ^ 2 := this
    _ = 16 * t / kappa ^ 2 := by field_simp; ring

/-- **The crossover.**  If the screened patterning contrast is even half of the unscreened one,
then `κ · t ≤ 12`: charge patterning is a thermodynamic effect only while the Debye screening
length `1/κ` is comparable to the length of the region itself.  At higher salt, two sequences of
identical composition and radically different charge order become indistinguishable. -/
theorem patterning_needs_long_screening_length {t : ℕ} {kappa : ℝ} (hk : 0 < kappa)
    (ht : 3 ≤ t) (hgap : ((t : ℝ) ^ 3 / 3 - t) / 2 ≤ gap t kappa) :
    kappa * t ≤ 12 := by
  have htR : (3 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
  have hk2 : 0 < kappa ^ 2 := by positivity
  have hbound : gap t kappa ≤ 16 * t / kappa ^ 2 :=
    le_trans (le_abs_self _) (screened_gap_le hk)
  have hchain : ((t : ℝ) ^ 3 / 3 - t) / 2 ≤ 16 * t / kappa ^ 2 := le_trans hgap hbound
  have hmul : (((t : ℝ) ^ 3 / 3 - t) / 2) * kappa ^ 2 ≤ 16 * t := by
    rw [le_div_iff₀ hk2] at hchain
    linarith
  -- `t³/3 − t ≥ (2/9) t³` for `t ≥ 3`
  have hcube : (2 / 9 : ℝ) * (t : ℝ) ^ 3 ≤ (t : ℝ) ^ 3 / 3 - t := by
    nlinarith [htR, sq_nonneg ((t : ℝ) - 3)]
  have hpos : (0 : ℝ) < (t : ℝ) := by linarith
  have hstep : ((2 / 9 : ℝ) * (t : ℝ) ^ 3 / 2) * kappa ^ 2 ≤ 16 * t := by
    have h := hmul
    nlinarith [hcube, hk2]
  have hfin : kappa ^ 2 * (t : ℝ) ^ 2 ≤ 144 := by
    have hexp : ((2 / 9 : ℝ) * (t : ℝ) ^ 3 / 2) * kappa ^ 2 = (1 / 9) * (t : ℝ) * (kappa ^ 2 * (t:ℝ) ^ 2) := by
      ring
    rw [hexp] at hstep
    nlinarith [hstep, hpos]
  nlinarith [hfin, hpos, hk, mul_pos hk hpos]

lemma cube_third_sub_pos {x : ℝ} (hx : 2 ≤ x) : 0 < x ^ 3 / 3 - x := by
  have h2 : (4 : ℝ) ≤ x ^ 2 := by nlinarith
  have h3 : 4 * x ≤ x ^ 3 := by nlinarith
  linarith

/-- The unscreened contrast is genuinely there: for a region of at least four residues the
diblock and the perfectly mixed sequence, of identical composition, have different energies. -/
theorem unscreened_gap_pos {t : ℕ} (ht : 2 ≤ t) : 0 < gap t 0 := by
  have htR : (2 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
  have h := unscreened_gap_ge t
  linarith [cube_third_sub_pos htR]

/-- **Salt erases patterning.**  Above the crossover, `κ · t > 12`, the screened contrast between
the diblock and the perfectly mixed sequence is less than half of its unscreened value: the two
sequences of identical composition have become thermodynamically alike. -/
theorem screening_kills_contrast {t : ℕ} {kappa : ℝ} (hk : 0 < kappa) (ht : 3 ≤ t)
    (h : 12 < kappa * t) : gap t kappa < ((t : ℝ) ^ 3 / 3 - t) / 2 := by
  by_contra hcon
  push_neg at hcon
  exact absurd (patterning_needs_long_screening_length hk ht hcon) (not_le.2 h)

/-- **The salt crossover, in one statement.**  For a neutral region of `2t` residues with
`t ≥ 3`: at zero ionic strength the diblock and the perfectly mixed pattern differ in energy by
at least `t³/3 − t`, a strictly positive, cubically growing amount; at inverse screening length
`κ` the difference is at most `16 t / κ²`; and once `κ t > 12` it has fallen below half its
unscreened value.  Charge patterning is therefore a property of a disordered region *in a
solution condition*, not of its sequence alone. -/
theorem salt_crossover {t : ℕ} {kappa : ℝ} (hk : 0 < kappa) (ht : 3 ≤ t) :
    (0 < (t : ℝ) ^ 3 / 3 - t ∧ (t : ℝ) ^ 3 / 3 - t ≤ gap t 0)
      ∧ |gap t kappa| ≤ 16 * t / kappa ^ 2
      ∧ (12 < kappa * t → gap t kappa < ((t : ℝ) ^ 3 / 3 - t) / 2) := by
  have htR : (3 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
  refine ⟨⟨cube_third_sub_pos (by linarith), unscreened_gap_ge t⟩, screened_gap_le hk, ?_⟩
  intro h
  exact screening_kills_contrast hk ht h

/-! ## 5. Below the crossover the patterning survives -/

lemma one_sub_exp_neg_le (x : ℝ) : 1 - Real.exp (-x) ≤ x := by
  have h := Real.add_one_le_exp (-x)
  linarith

/-- **The screened energy is a small perturbation of the unscreened one at low salt.**  For unit
charges, `|energy N κ q − energy N 0 q| ≤ κ N⁴`. -/
lemma energy_perturb {N : ℕ} {kappa : ℝ} (hk : 0 ≤ kappa) {q : ℕ → ℝ} (hq : ∀ i, |q i| ≤ 1) :
    |energy N kappa q - energy N 0 q| ≤ kappa * N ^ 4 := by
  have hdiff : energy N kappa q - energy N 0 q
      = ∑ j ∈ range N, ∑ i ∈ range j, (kern kappa (j - i) - kern 0 (j - i)) * (q i * q j) := by
    unfold energy Pattern.pairEnergy
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  have hterm : ∀ j ∈ range N, ∀ i ∈ range j,
      |(kern kappa (j - i) - kern 0 (j - i)) * (q i * q j)| ≤ kappa * (N : ℝ) ^ 2 := by
    intro j hj i hi
    have hjN : j < N := Finset.mem_range.1 hj
    have hij : i < j := Finset.mem_range.1 hi
    have hdN : ((j - i : ℕ) : ℝ) ≤ (N : ℝ) := by
      have : (j - i : ℕ) ≤ N := by omega
      exact_mod_cast this
    have hd0 : (0 : ℝ) ≤ ((j - i : ℕ) : ℝ) := Nat.cast_nonneg _
    have hexp : kern 0 (j - i) - kern kappa (j - i)
        ≤ kappa * ((j - i : ℕ) : ℝ) ^ 2 := by
      rw [kern_zero, kern]
      have h1 : 1 - Real.exp (-(kappa * ((j - i : ℕ) : ℝ))) ≤ kappa * ((j - i : ℕ) : ℝ) :=
        one_sub_exp_neg_le _
      nlinarith [h1, hd0]
    have hge : kern kappa (j - i) ≤ kern 0 (j - i) := by
      rw [kern_zero, kern]
      have : Real.exp (-(kappa * ((j - i : ℕ) : ℝ))) ≤ 1 := by
        rw [Real.exp_le_one_iff]
        have : 0 ≤ kappa * ((j - i : ℕ) : ℝ) := mul_nonneg hk hd0
        linarith
      nlinarith [hd0]
    have hqq : |q i * q j| ≤ 1 := by
      rw [abs_mul]
      nlinarith [abs_nonneg (q i), abs_nonneg (q j), hq i, hq j]
    have hsqle : kappa * ((j - i : ℕ) : ℝ) ^ 2 ≤ kappa * (N : ℝ) ^ 2 := by
      have : ((j - i : ℕ) : ℝ) ^ 2 ≤ (N : ℝ) ^ 2 := by nlinarith [hd0, hdN]
      exact mul_le_mul_of_nonneg_left this hk
    rw [abs_mul, abs_of_nonpos (by linarith : kern kappa (j - i) - kern 0 (j - i) ≤ 0)]
    have hbase : -(kern kappa (j - i) - kern 0 (j - i)) ≤ kappa * (N : ℝ) ^ 2 := by linarith
    calc -(kern kappa (j - i) - kern 0 (j - i)) * |q i * q j|
        ≤ -(kern kappa (j - i) - kern 0 (j - i)) * 1 :=
          mul_le_mul_of_nonneg_left hqq (by linarith)
      _ ≤ kappa * (N : ℝ) ^ 2 := by rw [mul_one]; exact hbase
  have hinner : ∀ j ∈ range N,
      |∑ i ∈ range j, (kern kappa (j - i) - kern 0 (j - i)) * (q i * q j)|
        ≤ (N : ℝ) * (kappa * (N : ℝ) ^ 2) := by
    intro j hj
    have hjN : j < N := Finset.mem_range.1 hj
    calc |∑ i ∈ range j, (kern kappa (j - i) - kern 0 (j - i)) * (q i * q j)|
        ≤ ∑ i ∈ range j, |(kern kappa (j - i) - kern 0 (j - i)) * (q i * q j)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i ∈ range j, kappa * (N : ℝ) ^ 2 := Finset.sum_le_sum (hterm j hj)
      _ = (j : ℝ) * (kappa * (N : ℝ) ^ 2) := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      _ ≤ (N : ℝ) * (kappa * (N : ℝ) ^ 2) := by
          have hjR : (j : ℝ) ≤ (N : ℝ) := by exact_mod_cast hjN.le
          have : 0 ≤ kappa * (N : ℝ) ^ 2 := by positivity
          exact mul_le_mul_of_nonneg_right hjR this
  rw [hdiff]
  calc |∑ j ∈ range N, ∑ i ∈ range j, (kern kappa (j - i) - kern 0 (j - i)) * (q i * q j)|
      ≤ ∑ j ∈ range N, |∑ i ∈ range j, (kern kappa (j - i) - kern 0 (j - i)) * (q i * q j)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _j ∈ range N, (N : ℝ) * (kappa * (N : ℝ) ^ 2) := Finset.sum_le_sum hinner
    _ = kappa * (N : ℝ) ^ 4 := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        ring

/-- **Below the crossover the patterning is still there.**  If `κ ≤ 1/(288 t)` then the contrast
between the diblock and the perfectly mixed sequence is still at least half its unscreened value.
Together with `screening_kills_contrast` this locates the crossover at `κ t` of order one. -/
theorem patterning_survives_at_low_salt {t : ℕ} {kappa : ℝ} (hk : 0 ≤ kappa) (ht : 3 ≤ t)
    (hlow : kappa ≤ 1 / (288 * t)) : ((t : ℝ) ^ 3 / 3 - t) / 2 ≤ gap t kappa := by
  have htR : (3 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
  have hpos : (0 : ℝ) < (t : ℝ) := by linarith
  have h1 : |energy (2 * t) kappa altR - energy (2 * t) 0 altR| ≤ kappa * ((2 * t : ℕ) : ℝ) ^ 4 :=
    energy_perturb hk altR_abs
  have h2 : |energy (2 * t) kappa (blkR t) - energy (2 * t) 0 (blkR t)|
      ≤ kappa * ((2 * t : ℕ) : ℝ) ^ 4 := energy_perturb hk (blkR_abs t)
  have hc : ((2 * t : ℕ) : ℝ) = 2 * (t : ℝ) := by push_cast; ring
  rw [hc] at h1 h2
  have hpow : kappa * (2 * (t : ℝ)) ^ 4 = 16 * kappa * (t : ℝ) ^ 4 := by ring
  rw [hpow] at h1 h2
  have e1 := abs_le.1 h1
  have e2 := abs_le.1 h2
  have hg0 : (t : ℝ) ^ 3 / 3 - t ≤ gap t 0 := unscreened_gap_ge t
  have hgapdef0 : gap t 0 = energy (2 * t) 0 altR - energy (2 * t) 0 (blkR t) := rfl
  have hgapdefk : gap t kappa = energy (2 * t) kappa altR - energy (2 * t) kappa (blkR t) := rfl
  have hstep : gap t 0 - 32 * kappa * (t : ℝ) ^ 4 ≤ gap t kappa := by
    rw [hgapdef0, hgapdefk]
    linarith [e1.1, e1.2, e2.1, e2.2]
  -- `32 κ t⁴ ≤ t³/9 ≤ (t³/3 − t)/2`
  have hsmall : 32 * kappa * (t : ℝ) ^ 4 ≤ (t : ℝ) ^ 3 / 9 := by
    have h288 : (0 : ℝ) < 288 * (t : ℝ) := by linarith
    have hk' : kappa * (288 * (t : ℝ)) ≤ 1 := by
      rw [le_div_iff₀ h288] at hlow
      linarith
    have hcube0 : (0 : ℝ) ≤ (t : ℝ) ^ 3 / 9 := by positivity
    calc 32 * kappa * (t : ℝ) ^ 4 = (kappa * (288 * (t : ℝ))) * ((t : ℝ) ^ 3 / 9) := by ring
      _ ≤ 1 * ((t : ℝ) ^ 3 / 9) := mul_le_mul_of_nonneg_right hk' hcube0
      _ = (t : ℝ) ^ 3 / 9 := one_mul _
  have h9 : (9 : ℝ) ≤ (t : ℝ) ^ 2 := by nlinarith [htR]
  have ht3 : (t : ℝ) ≤ (t : ℝ) ^ 3 / 9 := by nlinarith [hpos, h9]
  linarith

/-! ## 6. No salt-blind model of a disordered region can be right -/

/-- Above the crossover the screened contrast is smaller in absolute value than half the
unscreened one. -/
theorem abs_screened_gap_lt_half {t : ℕ} {kappa : ℝ} (hk : 0 < kappa) (ht : 4 ≤ t)
    (hcross : 12 < kappa * t) : |gap t kappa| < ((t : ℝ) ^ 3 / 3 - t) / 2 := by
  have htR : (4 : ℝ) ≤ (t : ℝ) := by exact_mod_cast ht
  have hpos : (0 : ℝ) < (t : ℝ) := by linarith
  have hk2 : 0 < kappa ^ 2 := by positivity
  have hkt : 144 < kappa ^ 2 * (t : ℝ) ^ 2 := by nlinarith [hcross, hk, hpos]
  have h1 : |gap t kappa| ≤ 16 * t / kappa ^ 2 := screened_gap_le hk
  have h2 : 16 * (t : ℝ) / kappa ^ 2 < (t : ℝ) ^ 3 / 9 := by
    rw [div_lt_iff₀ hk2]
    nlinarith [hkt, hpos]
  have h3 : (t : ℝ) ^ 3 / 9 < ((t : ℝ) ^ 3 / 3 - t) / 2 := by
    nlinarith [htR, hpos]
  linarith

/-- **A model that does not know the ionic strength must be wrong.**  Let `f` be any predictor
that assigns an energy to a sequence and to a sequence only.  Evaluated on the two sequences of
identical composition -- the perfectly mixed and the diblock pattern -- at zero salt and above the
screening crossover, its worst error is at least `(t³/3 − t)/8`, an amount growing like the cube
of the length of the region.  A patterning parameter is therefore not a function of sequence: a
correct model must take the solution condition as an explicit input. -/
theorem no_salt_blind_prediction {t : ℕ} {kappa eps : ℝ} (hk : 0 < kappa) (ht : 4 ≤ t)
    (hcross : 12 < kappa * t) (f : (ℕ → ℝ) → ℝ)
    (h1 : |f altR - energy (2 * t) 0 altR| ≤ eps)
    (h2 : |f (blkR t) - energy (2 * t) 0 (blkR t)| ≤ eps)
    (h3 : |f altR - energy (2 * t) kappa altR| ≤ eps)
    (h4 : |f (blkR t) - energy (2 * t) kappa (blkR t)| ≤ eps) :
    ((t : ℝ) ^ 3 / 3 - t) / 8 ≤ eps := by
  have hg0 : (t : ℝ) ^ 3 / 3 - t ≤ gap t 0 := unscreened_gap_ge t
  have hgk : |gap t kappa| < ((t : ℝ) ^ 3 / 3 - t) / 2 := abs_screened_gap_lt_half hk ht hcross
  have hgk' : gap t kappa ≤ ((t : ℝ) ^ 3 / 3 - t) / 2 := le_of_lt (lt_of_le_of_lt (le_abs_self _) hgk)
  have e1 := abs_le.1 h1
  have e2 := abs_le.1 h2
  have e3 := abs_le.1 h3
  have e4 := abs_le.1 h4
  have hgapdef0 : gap t 0 = energy (2 * t) 0 altR - energy (2 * t) 0 (blkR t) := rfl
  have hgapdefk : gap t kappa = energy (2 * t) kappa altR - energy (2 * t) kappa (blkR t) := rfl
  rw [hgapdef0] at hg0
  rw [hgapdefk] at hgk'
  linarith [e1.1, e1.2, e2.1, e2.2, e3.1, e3.2, e4.1, e4.2]

/-- **The crossover is two-sided.**  For a region of `2t` residues with `t ≥ 4`: at
`κ ≤ 1/(288 t)` the patterning contrast is at least half its unscreened value, and at
`κ > 12/t` it is less than half.  The transition happens at `κ t` of order one, that is, when
the Debye screening length is comparable to the length of the region. -/
theorem crossover_two_sided {t : ℕ} (ht : 4 ≤ t) :
    (∀ kappa : ℝ, 0 ≤ kappa → kappa ≤ 1 / (288 * t) →
        ((t : ℝ) ^ 3 / 3 - t) / 2 ≤ gap t kappa)
      ∧ (∀ kappa : ℝ, 0 < kappa → 12 < kappa * t →
        |gap t kappa| < ((t : ℝ) ^ 3 / 3 - t) / 2) :=
  ⟨fun _ hk hlow => patterning_survives_at_low_salt hk (by omega) hlow,
    fun _ hk hcross => abs_screened_gap_lt_half hk ht hcross⟩

end Salt
end IDR
