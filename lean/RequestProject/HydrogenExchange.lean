/-
# Part XXVIII.2  Hydrogen exchange: an average of rates, not a rate of averages

Amide hydrogen–deuterium exchange is the standard residue-resolved probe of local stability in
disordered and partially disordered regions.  The measurement is a *rate*, and the rate is a
population-weighted average of per-conformer opening probabilities.  Two things follow, and
both are routinely mis-stated in the literature; this file proves them.

**1.  The apparent protection free energy is not the mean local stability.**
Under the EX2 (fast-reclosing) regime the observed rate is `k_obs = k_int · ⟨p_open⟩`, where
`p_open k` is the probability that the amide is exchange-competent in conformer `k`.  The
number reported is `ΔG_app = -RT log ⟨p_open⟩`, whereas the per-conformer stabilities are
`ΔG_k = -RT log p_open k`.

* `protectionFactor_eq` -- the protection factor is the reciprocal of the mean openness.
* `deltaGapp_le_mean` -- Jensen for `log`: `ΔG_app ≤ ⟨ΔG⟩` always.  Interpreting a hydrogen
  exchange measurement as a mean local free energy *systematically overestimates* how open the
  ensemble is not -- the apparent stability is always the lower number.
* `deltaGapp_lt_mean_two` -- an explicit two-state instance with a strict gap, so the bias is
  real, not a boundary case.
* `deltaGapp_le_of_weight` -- the minority-report bound, the exchange analogue of the `r^{-6}`
  bound of Part IX.2: a conformer of weight `w_k` that is open with probability `p_k` caps the
  apparent protection at `ΔG_k - RT log w_k`, *whatever the rest of the ensemble does*.  A 1%
  fully-open state limits the apparent stability to `RT log 100 ≈ 2.7 kcal/mol` at 300 K
  however stable the other 99% is.

**2.  The measurement is not a functional of the equilibrium ensemble at all.**
With opening/closing rates `k_op, k_cl` and intrinsic chemistry `k_int`, the steady-state
exchange rate is `k_ex = k_op k_int / (k_cl + k_int)`.

* `kex_eq_EX2_sub` -- the exact deviation from the EX2 reading:
  `k_int·K_op - k_ex = k_int·K_op · k_int/(k_cl + k_int)`, `K_op = k_op/k_cl`.  EX2 is a limit,
  not an identity.
* `kex_le_kop` -- in the opposite (EX1) limit the measurement saturates at the opening rate and
  becomes independent of the stability entirely.
* `equilibrium_does_not_determine_kex` -- two kinetic schemes with *identical* equilibrium
  populations (`k_op/k_cl` equal, so the same Boltzmann ensemble) give different observed
  exchange rates.  Therefore no model that outputs only an equilibrium distribution can
  predict a hydrogen exchange experiment: the forward model must include the kinetics, or the
  data must be restricted to the verified EX2 regime.
-/
import Mathlib

set_option autoImplicit false

namespace HX

open Finset

variable {m : ℕ}

/-- Mean openness of the amide over the conformational ensemble: the EX2 observable, in units
of the intrinsic chemical exchange rate. -/
noncomputable def meanOpen (w p : Fin m → ℝ) : ℝ := ∑ k, w k * p k

/-- Observed EX2 exchange rate `k_obs = k_int ⟨p_open⟩`. -/
noncomputable def kobsEX2 (kint : ℝ) (w p : Fin m → ℝ) : ℝ := kint * meanOpen w p

/-- Protection factor `P = k_int / k_obs`. -/
noncomputable def protectionFactor (kint : ℝ) (w p : Fin m → ℝ) : ℝ :=
  kint / kobsEX2 kint w p

/-- Apparent protection free energy `ΔG_app = -RT log ⟨p_open⟩`. -/
noncomputable def deltaGapp (RT : ℝ) (w p : Fin m → ℝ) : ℝ := -RT * Real.log (meanOpen w p)

/-- Per-conformer local stability `ΔG_k = -RT log p_k`, averaged over the ensemble. -/
noncomputable def meanDeltaG (RT : ℝ) (w p : Fin m → ℝ) : ℝ :=
  ∑ k, w k * (-RT * Real.log (p k))

lemma meanOpen_pos {w p : Fin m → ℝ} (hw : ∀ k, 0 ≤ w k) (hsum : ∑ k, w k = 1)
    (hp : ∀ k, 0 < p k) : 0 < meanOpen w p := by
  obtain ⟨k0, hk0⟩ : ∃ k, 0 < w k := by
    by_contra hcon
    push_neg at hcon
    have : ∑ k, w k ≤ 0 := Finset.sum_nonpos fun k _ => hcon k
    rw [hsum] at this; linarith
  refine lt_of_lt_of_le (mul_pos hk0 (hp k0)) ?_
  exact Finset.single_le_sum (fun k _ => mul_nonneg (hw k) (hp k).le) (Finset.mem_univ k0)

/-- **The protection factor is the reciprocal of the mean openness.** -/
theorem protectionFactor_eq {kint : ℝ} (hk : 0 < kint) {w p : Fin m → ℝ}
    (hw : ∀ k, 0 ≤ w k) (hsum : ∑ k, w k = 1) (hp : ∀ k, 0 < p k) :
    protectionFactor kint w p = (meanOpen w p)⁻¹ := by
  have hm := meanOpen_pos hw hsum hp
  unfold protectionFactor kobsEX2
  field_simp

/-- **Jensen: the apparent stability is never the mean stability.**  `-RT log ⟨p⟩ ≤ ⟨-RT log p⟩`
for `RT > 0`: hydrogen exchange reports a free energy at or below the population average of the
local stabilities, because it averages *rates*, not free energies. -/
theorem deltaGapp_le_mean {RT : ℝ} (hRT : 0 < RT) {w p : Fin m → ℝ}
    (hw : ∀ k, 0 ≤ w k) (hsum : ∑ k, w k = 1) (hp : ∀ k, 0 < p k) :
    deltaGapp RT w p ≤ meanDeltaG RT w p := by
  have hconc : ConcaveOn ℝ (Set.Ioi (0 : ℝ)) Real.log := strictConcaveOn_log_Ioi.concaveOn
  have hjensen : ∑ k, w k * Real.log (p k) ≤ Real.log (∑ k, w k * p k) := by
    have := hconc.le_map_centerMass (t := Finset.univ) (w := w) (p := p)
      (fun k _ => hw k) (by rw [hsum]; norm_num) (fun k _ => Set.mem_Ioi.mpr (hp k))
    simpa [Finset.centerMass, hsum, smul_eq_mul, Function.comp] using this
  have hexp : meanDeltaG RT w p = -RT * ∑ k, w k * Real.log (p k) := by
    unfold meanDeltaG
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => by ring
  rw [deltaGapp, hexp, meanOpen]
  have : -RT * Real.log (∑ k, w k * p k) ≤ -RT * ∑ k, w k * Real.log (p k) :=
    mul_le_mul_of_nonpos_left hjensen (by linarith)
  exact this

/-- **The bias is strict, explicitly.**  A 50:50 ensemble of a fully open state and a state
open with probability `1/100` has an apparent stability strictly below the mean of the two
local stabilities. -/
theorem deltaGapp_lt_mean_two {RT : ℝ} (hRT : 0 < RT) :
    deltaGapp RT (![1 / 2, 1 / 2] : Fin 2 → ℝ) (![1, 1 / 100] : Fin 2 → ℝ)
      < meanDeltaG RT (![1 / 2, 1 / 2] : Fin 2 → ℝ) (![1, 1 / 100] : Fin 2 → ℝ) := by
  have hlog : Real.log (1 / 100 : ℝ) < 0 :=
    Real.log_neg (by norm_num) (by norm_num)
  have hmean : meanOpen (![1 / 2, 1 / 2] : Fin 2 → ℝ) (![1, 1 / 100] : Fin 2 → ℝ)
      = 101 / 200 := by
    simp [meanOpen, Fin.sum_univ_two]
    norm_num
  have hG : meanDeltaG RT (![1 / 2, 1 / 2] : Fin 2 → ℝ) (![1, 1 / 100] : Fin 2 → ℝ)
      = -RT * (Real.log (1 / 100) / 2) := by
    simp [meanDeltaG, Fin.sum_univ_two]
    ring
  -- strict Jensen for two points: `log` is strictly concave
  have hstrict : Real.log (1 / 100 : ℝ) / 2 < Real.log (101 / 200 : ℝ) := by
    have h := strictConcaveOn_log_Ioi.2 (Set.mem_Ioi.mpr (by norm_num : (0:ℝ) < 1))
      (Set.mem_Ioi.mpr (by norm_num : (0:ℝ) < 1 / 100)) (by norm_num)
      (by norm_num : (0:ℝ) < 1 / 2) (by norm_num : (0:ℝ) < 1 / 2) (by norm_num)
    norm_num [Real.log_one] at h
    linarith [h]
  rw [deltaGapp, hG, hmean]
  have : -RT * Real.log (101 / 200 : ℝ) < -RT * (Real.log (1 / 100) / 2) := by
    have hneg : -RT < 0 := by linarith
    exact mul_lt_mul_of_neg_left hstrict hneg
  exact this

/-- **A minority open state caps the apparent protection.**  If conformer `k` has weight `w k`
and openness `p k`, the apparent free energy cannot exceed `ΔG_k - RT log (w k)`, whatever the
rest of the ensemble does.  With `w k = 1/100` and `p k = 1` this is `RT log 100`. -/
theorem deltaGapp_le_of_weight {RT : ℝ} (hRT : 0 < RT) {w p : Fin m → ℝ}
    (hw : ∀ i, 0 ≤ w i) (hp : ∀ i, 0 < p i) (k : Fin m) (hk : 0 < w k) :
    deltaGapp RT w p ≤ -RT * Real.log (p k) - RT * Real.log (w k) := by
  have hle : w k * p k ≤ meanOpen w p :=
    Finset.single_le_sum (f := fun i => w i * p i)
      (fun i _ => mul_nonneg (hw i) (hp i).le) (Finset.mem_univ k)
  have hpos : 0 < w k * p k := mul_pos hk (hp k)
  have hlog : Real.log (w k * p k) ≤ Real.log (meanOpen w p) :=
    Real.log_le_log hpos hle
  have hsplit : Real.log (w k * p k) = Real.log (w k) + Real.log (p k) :=
    Real.log_mul (ne_of_gt hk) (ne_of_gt (hp k))
  rw [deltaGapp]
  nlinarith [hlog, hsplit]

/-! ### The kinetic regimes: exchange is not a function of the ensemble -/

/-- Steady-state (Linderstrøm-Lang) exchange rate for a closed ⇌ open amide with intrinsic
chemical rate `k_int`. -/
noncomputable def kex (kop kcl kint : ℝ) : ℝ := kop * kint / (kcl + kint)

/-- Equilibrium opening constant `K_op = k_op / k_cl`, the only thing an equilibrium ensemble
knows about the amide. -/
noncomputable def Kop (kop kcl : ℝ) : ℝ := kop / kcl

/-- **EX2 is a limit, not an identity.**  The exact defect of the EX2 formula
`k_ex ≈ k_int K_op` is `k_int K_op · k_int/(k_cl + k_int)`, which vanishes only as
`k_int / k_cl → 0`. -/
theorem kex_eq_EX2_sub {kop kcl kint : ℝ} (hcl : 0 < kcl) (hint : 0 < kint) :
    kint * Kop kop kcl - kex kop kcl kint
      = kint * Kop kop kcl * (kint / (kcl + kint)) := by
  have h1 : kcl + kint ≠ 0 := by positivity
  unfold kex Kop
  field_simp
  ring

/-- **The EX1 ceiling.**  The exchange rate can never exceed the opening rate; in the EX1 limit
`k_int ≫ k_cl` it approaches it and stops reporting on stability at all. -/
theorem kex_le_kop {kop kcl kint : ℝ} (hop : 0 ≤ kop) (hcl : 0 ≤ kcl) (hint : 0 < kint) :
    kex kop kcl kint ≤ kop := by
  have hden : 0 < kcl + kint := by linarith
  rw [kex, div_le_iff₀ hden]
  nlinarith

/-- **The equilibrium ensemble does not determine the measurement.**  Two amides with the same
opening equilibrium constant -- hence exactly the same equilibrium populations of open and
closed, and the same Boltzmann ensemble -- exchange at different observed rates.  A model that
outputs only a distribution over conformations therefore cannot predict hydrogen exchange
without a kinetic forward model. -/
theorem equilibrium_does_not_determine_kex :
    ∃ kop kcl kop' kcl' kint : ℝ, 0 < kop ∧ 0 < kcl ∧ 0 < kop' ∧ 0 < kcl' ∧ 0 < kint ∧
      Kop kop kcl = Kop kop' kcl' ∧ kex kop kcl kint ≠ kex kop' kcl' kint := by
  refine ⟨1, 1, 100, 100, 1, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num [Kop], ?_⟩
  norm_num [kex]

end HX
