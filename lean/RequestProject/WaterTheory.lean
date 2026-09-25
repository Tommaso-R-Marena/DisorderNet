/-
# Part CXII  A theory of water

The solvent enters this development twice: as the exact finite-solvent potential of mean force
(the integration that turns an explicit solvent into a solute-only Hamiltonian) and as the
implicit terms of `RequestProject.Solvation`.  Both take the solvent as given.  Neither is a
*theory of water*: neither says why water behaves as it does, and it is water's peculiar
behaviour — the hydrogen-bond network, the density anomaly, the entropic hydrophobic effect —
that decides whether a region of a protein is folded or disordered.  This file supplies a model
of water and solves it exactly.

Three pieces of physics, each an exactly solvable model with the consequences proved.

## 1.  The hydrogen-bond network (`Zchain_eq`, `bondFraction_*`)

A chain of water molecules, each with `card S` orientational states, neighbours making a hydrogen
bond of energy `−eps` when their orientations agree.  The transfer matrix has constant row sum
`lam = exp(eps/T) − 1 + card S`, so the partition function is *exactly* `card S · lam^n`
(`Zchain_eq`) and the free energy per bond is exactly `−T log lam`
(`log_Zchain_div_tendsto`).  The intact-bond fraction is `exp(eps/T)/lam`
(`bond_conditional_prob`): it is strictly between the fully bonded and the random values
(`bondFraction_lt_one`, `bondFraction_gt_random`), and it **melts continuously with
temperature** (`bondFraction_strictAnti`) — the network is neither an all-or-nothing lattice nor
absent, which is the property that makes water a poor theta-solvent for a polypeptide over a
narrow temperature range.

## 2.  The density anomaly (`density_anomaly`, `exists_density_maximum`)

Water molecules are taken to be in a low-energy, low-density, hydrogen-bonded *open* state or in
one of `g` degenerate *dense* states, each species expanding normally with temperature.  The open
fraction `pOpen` decreases strictly with temperature (`pOpen_strictAnti`), so heating converts
open to dense and *contracts* the liquid.  `density_anomaly` gives the exact condition under
which this beats normal expansion — the liquid then has negative thermal expansivity — and
`exists_density_maximum` turns it into the temperature of maximum density: an interior minimum of
the molar volume, which is what water has at 4 °C and what no simple liquid has.

## 3.  The hydrophobic effect, and cold denaturation (`solv_pos`, `solv_strictMono`, `cold_denat`)

A nonpolar surface removes orientations from the water in its first shell: the dense states of a
shell water are reduced from `g` to `g' < g`.  The resulting solvation free energy per shell
water is

    solv eps T g g' = −T · log ( (exp(eps/T) + g') / (exp(eps/T) + g) ) ,

and the theorems are the three signatures of hydrophobic hydration:

* `solv_pos` — it is **positive**: burying nonpolar surface is favourable at every temperature.
* `solv_le_entropic` — it is bounded by `T·log(g/g')`, the pure orientational entropy term, so it
  is **entropy dominated**: the model has no attractive enthalpy to offer.
* `solv_strictMono` — it **increases with temperature**, the anomalous signature that
  distinguishes the hydrophobic effect from ordinary solvophobicity.
* `solv_tendsto_zero_cold`, `cold_denat` — and it **vanishes as the temperature goes to zero**,
  so a structure held together only by burial of hydrophobic surface must come apart on cooling.
  This is cold denaturation, and it is the reason a disordered region cannot be explained by a
  temperature-independent hydrophobic term.

What is *not* claimed: this is a model of water, exactly solved, not a first-principles theory of
the liquid.  The three-dimensional hydrogen-bond network, the critical behaviour of the
liquid–liquid transition and the quantitative value of the temperature of maximum density are
outside it; what is inside it is the *mechanism* — orientational degeneracy against a bond
energy — and every consequence drawn here is a theorem about that mechanism.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR.Water

open Finset

/-! ## 1.  The hydrogen-bonded chain, solved exactly -/

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- The Boltzmann weight of a neighbouring pair of water molecules: a hydrogen bond of energy
`−eps` when the two orientations agree, nothing otherwise. -/
noncomputable def hbWeight (eps T : ℝ) (a b : S) : ℝ := if a = b then Real.exp (eps / T) else 1

/-- The transfer-matrix eigenvalue: `exp(eps/T) − 1 + (number of orientations)`. -/
noncomputable def lam (S : Type*) [Fintype S] (eps T : ℝ) : ℝ :=
  Real.exp (eps / T) - 1 + Fintype.card S

omit [DecidableEq S] in
lemma lam_pos [Nonempty S] (eps T : ℝ) : 0 < lam S eps T := by
  have h1 : 0 < Real.exp (eps / T) := Real.exp_pos _
  have h2 : 1 ≤ Fintype.card S := Fintype.card_pos
  unfold lam
  have : (1 : ℝ) ≤ (Fintype.card S : ℝ) := by exact_mod_cast h2
  linarith

/-- **Constant row sums**: the transfer matrix has the same total weight from every orientation.
This is what makes the chain exactly solvable. -/
theorem sum_hbWeight (eps T : ℝ) (a : S) : ∑ b, hbWeight eps T a b = lam S eps T := by
  have hsplit : ∀ b : S, hbWeight eps T a b
      = 1 + (if a = b then Real.exp (eps / T) - 1 else 0) := by
    intro b; unfold hbWeight; split <;> ring
  simp only [hsplit]
  rw [Finset.sum_add_distrib, Finset.sum_ite_eq Finset.univ a (fun _ => Real.exp (eps / T) - 1)]
  simp [lam, Finset.card_univ]
  ring

/-- The conditional partition function of a chain with `n` further bonds, given the orientation
of the molecule at the end. -/
noncomputable def Zcond (eps T : ℝ) : ℕ → S → ℝ
  | 0, _ => 1
  | n + 1, a => ∑ b, hbWeight eps T a b * Zcond eps T n b

/-- The conditional partition function is exactly a power of the eigenvalue. -/
theorem Zcond_eq (eps T : ℝ) (n : ℕ) (a : S) : Zcond eps T n a = lam S eps T ^ n := by
  induction n generalizing a with
  | zero => simp [Zcond]
  | succ k ih =>
      simp only [Zcond, ih]
      rw [← Finset.sum_mul, sum_hbWeight]
      ring

/-- The partition function of a chain of `n + 1` water molecules (`n` hydrogen bonds). -/
noncomputable def Zchain (eps T : ℝ) (n : ℕ) : ℝ := ∑ a : S, Zcond eps T n a

/-- **The chain is solved exactly.** -/
theorem Zchain_eq (eps T : ℝ) (n : ℕ) :
    Zchain (S := S) eps T n = (Fintype.card S : ℝ) * lam S eps T ^ n := by
  unfold Zchain
  rw [Finset.sum_congr rfl fun a _ => Zcond_eq eps T n a]
  simp [Finset.sum_const, Finset.card_univ]

/-- **The free energy per hydrogen bond is exactly `−T log lam`.** -/
theorem log_Zchain_div_tendsto [Nonempty S] (eps T : ℝ) :
    Filter.Tendsto (fun n : ℕ => Real.log (Zchain (S := S) eps T n) / n) Filter.atTop
      (nhds (Real.log (lam S eps T))) := by
  have hcard : (0 : ℝ) < (Fintype.card S : ℝ) := by
    exact_mod_cast Fintype.card_pos
  have hlam := lam_pos (S := S) eps T
  have key : ∀ n : ℕ, 1 ≤ n → Real.log (Fintype.card S : ℝ) / n + Real.log (lam S eps T)
      = Real.log (Zchain (S := S) eps T n) / n := by
    intro n hn
    have hn0 : (n : ℝ) ≠ 0 := by
      have : 0 < n := hn
      positivity
    rw [Zchain_eq, Real.log_mul hcard.ne' (by positivity), Real.log_pow]
    field_simp
  have h0 : Filter.Tendsto (fun n : ℕ => Real.log (Fintype.card S : ℝ) / n) Filter.atTop
      (nhds 0) := tendsto_const_div_atTop_nhds_zero_nat _
  have hlim := h0.add_const (Real.log (lam S eps T))
  rw [zero_add] at hlim
  exact hlim.congr' (Filter.eventually_atTop.2 ⟨1, fun n hn => key n hn⟩)

/-! ### The intact-bond fraction -/

/-- The fraction of intact hydrogen bonds: the conditional probability that a molecule matches
its neighbour. -/
noncomputable def bondFraction (S : Type*) [Fintype S] (eps T : ℝ) : ℝ :=
  Real.exp (eps / T) / lam S eps T

/-- **The bond fraction is the conditional probability of the transfer matrix.** -/
theorem bond_conditional_prob [Nonempty S] (eps T : ℝ) (a : S) :
    hbWeight eps T a a / (∑ b, hbWeight eps T a b) = bondFraction S eps T := by
  rw [sum_hbWeight]
  simp [hbWeight, bondFraction]

omit [DecidableEq S] in
theorem bondFraction_pos [Nonempty S] (eps T : ℝ) : 0 < bondFraction S eps T :=
  div_pos (Real.exp_pos _) (lam_pos eps T)

omit [DecidableEq S] in
/-- The network is never complete at finite temperature. -/
theorem bondFraction_lt_one [Nonempty S] {eps T : ℝ} (hcard : 2 ≤ Fintype.card S) :
    bondFraction S eps T < 1 := by
  have hlam := lam_pos (S := S) eps T
  have hc : (2 : ℝ) ≤ (Fintype.card S : ℝ) := by exact_mod_cast hcard
  rw [bondFraction, div_lt_one hlam, lam]
  linarith

omit [DecidableEq S] in
/-- And never as sparse as random orientations: hydrogen bonding is real. -/
theorem bondFraction_gt_random [Nonempty S] {eps T : ℝ} (heps : 0 < eps) (hT : 0 < T)
    (hcard : 2 ≤ Fintype.card S) :
    1 / (Fintype.card S : ℝ) < bondFraction S eps T := by
  have hlam := lam_pos (S := S) eps T
  have hc : (2 : ℝ) ≤ (Fintype.card S : ℝ) := by exact_mod_cast hcard
  have hcpos : (0 : ℝ) < (Fintype.card S : ℝ) := by linarith
  have hepsT : 0 < eps / T := by positivity
  have hexp : 1 < Real.exp (eps / T) := by
    have := Real.add_one_le_exp (eps / T)
    linarith
  rw [bondFraction, div_lt_div_iff₀ hcpos hlam, lam]
  nlinarith

omit [DecidableEq S] in
/-- **The network melts continuously**: the bond fraction is strictly decreasing in temperature.
-/
theorem bondFraction_strictAnti [Nonempty S] {eps T₁ T₂ : ℝ} (heps : 0 < eps) (hT : 0 < T₁)
    (hlt : T₁ < T₂) (hcard : 2 ≤ Fintype.card S) :
    bondFraction S eps T₂ < bondFraction S eps T₁ := by
  have hT₂ : 0 < T₂ := lt_trans hT hlt
  have hc : (2 : ℝ) ≤ (Fintype.card S : ℝ) := by exact_mod_cast hcard
  have hdiv : eps / T₂ < eps / T₁ := by
    apply div_lt_div_of_pos_left heps hT hlt
  have hexp : Real.exp (eps / T₂) < Real.exp (eps / T₁) := Real.exp_lt_exp.2 hdiv
  have h1 : 0 < Real.exp (eps / T₂) := Real.exp_pos _
  have hlam₁ := lam_pos (S := S) eps T₁
  have hlam₂ := lam_pos (S := S) eps T₂
  rw [bondFraction, bondFraction, div_lt_div_iff₀ hlam₂ hlam₁, lam, lam]
  nlinarith

/-! ## 2.  Two-state water: the density anomaly -/

/-- The fraction of water in the open, hydrogen-bonded, low-density state, when the dense state
has degeneracy `g`. -/
noncomputable def pOpen (eps T g : ℝ) : ℝ := Real.exp (eps / T) / (Real.exp (eps / T) + g)

lemma pOpen_denom_pos {g : ℝ} (hg : 0 < g) (eps T : ℝ) : 0 < Real.exp (eps / T) + g := by
  have := Real.exp_pos (eps / T); linarith

theorem pOpen_pos {g : ℝ} (hg : 0 < g) (eps T : ℝ) : 0 < pOpen eps T g :=
  div_pos (Real.exp_pos _) (pOpen_denom_pos hg eps T)

theorem pOpen_lt_one {g : ℝ} (hg : 0 < g) (eps T : ℝ) : pOpen eps T g < 1 := by
  rw [pOpen, div_lt_one (pOpen_denom_pos hg eps T)]
  linarith

/-- **Heating breaks the network**: the open fraction strictly decreases with temperature. -/
theorem pOpen_strictAnti {eps T₁ T₂ g : ℝ} (heps : 0 < eps) (hg : 0 < g) (hT : 0 < T₁)
    (hlt : T₁ < T₂) : pOpen eps T₂ g < pOpen eps T₁ g := by
  have hdiv : eps / T₂ < eps / T₁ := div_lt_div_of_pos_left heps hT hlt
  have hexp : Real.exp (eps / T₂) < Real.exp (eps / T₁) := Real.exp_lt_exp.2 hdiv
  have h1 : 0 < Real.exp (eps / T₂) := Real.exp_pos _
  have hd₁ := pOpen_denom_pos hg eps T₁
  have hd₂ := pOpen_denom_pos hg eps T₂
  rw [pOpen, pOpen, div_lt_div_iff₀ hd₂ hd₁]
  nlinarith

/-- The molar volume: the dense volume `vd`, plus the excess `vo − vd` carried by the open
fraction, plus the ordinary thermal expansion `a·T` common to both species. -/
noncomputable def meanVolume (eps g vd vo a T : ℝ) : ℝ :=
  vd + (vo - vd) * pOpen eps T g + a * T

/-- **The density anomaly.**  When the loss of open structure outruns ordinary expansion, water
*contracts on heating*: its thermal expansivity is negative. -/
theorem density_anomaly {eps g vd vo a T₁ T₂ : ℝ}
    (hcond : a * (T₂ - T₁) < (vo - vd) * (pOpen eps T₁ g - pOpen eps T₂ g)) :
    meanVolume eps g vd vo a T₂ < meanVolume eps g vd vo a T₁ := by
  unfold meanVolume
  nlinarith [hcond]

/-- The condition of `density_anomaly` is satisfiable: with an open state genuinely less dense
and a small enough ordinary expansivity, water contracts between any two temperatures. -/
theorem density_anomaly_nonvacuous {eps g vd vo T₁ T₂ : ℝ} (heps : 0 < eps) (hg : 0 < g)
    (hT : 0 < T₁) (hlt : T₁ < T₂) (hv : vd < vo) :
    ∃ a > 0, a * (T₂ - T₁) < (vo - vd) * (pOpen eps T₁ g - pOpen eps T₂ g) := by
  have hp := pOpen_strictAnti heps hg hT hlt
  have hrhs : 0 < (vo - vd) * (pOpen eps T₁ g - pOpen eps T₂ g) := by
    apply mul_pos (by linarith) (by linarith)
  refine ⟨(vo - vd) * (pOpen eps T₁ g - pOpen eps T₂ g) / (2 * (T₂ - T₁)),
    div_pos hrhs (by linarith), ?_⟩
  rw [div_mul_eq_mul_div, div_lt_iff₀ (by linarith)]
  nlinarith

lemma continuousOn_pOpen {eps g : ℝ} (hg : 0 < g) :
    ContinuousOn (fun T => pOpen eps T g) {T : ℝ | T ≠ 0} := by
  apply ContinuousOn.div
  · exact (Real.continuous_exp.comp_continuousOn
      (continuousOn_const.div continuousOn_id fun T hT => hT))
  · exact ((Real.continuous_exp.comp_continuousOn
      (continuousOn_const.div continuousOn_id fun T hT => hT)).add continuousOn_const)
  · intro T _
    exact (pOpen_denom_pos hg eps T).ne'

lemma continuousOn_meanVolume {eps g vd vo a : ℝ} (hg : 0 < g) :
    ContinuousOn (meanVolume eps g vd vo a) {T : ℝ | T ≠ 0} := by
  unfold meanVolume
  exact ((continuousOn_const.add (continuousOn_const.mul (continuousOn_pOpen hg))).add
    (continuousOn_const.mul continuousOn_id))

/-- **The temperature of maximum density.**  If the liquid contracts between `T₁` and `T₂` and
expands again by `T₃`, then the molar volume has an interior minimum on `[T₁, T₃]`: water has a
temperature of maximum density, and no monotone model of a liquid has one. -/
theorem exists_density_maximum {eps g vd vo a T₁ T₂ T₃ : ℝ} (hg : 0 < g) (hT : 0 < T₁)
    (h12 : T₁ < T₂) (h23 : T₂ < T₃)
    (hlow : meanVolume eps g vd vo a T₂ < meanVolume eps g vd vo a T₁)
    (hhigh : meanVolume eps g vd vo a T₂ < meanVolume eps g vd vo a T₃) :
    ∃ Tstar ∈ Set.Ioo T₁ T₃, IsMinOn (meanVolume eps g vd vo a) (Set.Icc T₁ T₃) Tstar := by
  have hsub : Set.Icc T₁ T₃ ⊆ {T : ℝ | T ≠ 0} := by
    intro T hT'
    have : 0 < T := lt_of_lt_of_le hT hT'.1
    exact ne_of_gt this
  have hcont : ContinuousOn (meanVolume eps g vd vo a) (Set.Icc T₁ T₃) :=
    (continuousOn_meanVolume hg).mono hsub
  have hne : (Set.Icc T₁ T₃).Nonempty := Set.nonempty_Icc.2 (by linarith)
  obtain ⟨Tstar, hTstar, hmin⟩ := (isCompact_Icc (a := T₁) (b := T₃)).exists_isMinOn hne hcont
  have hT₂mem : T₂ ∈ Set.Icc T₁ T₃ := ⟨le_of_lt h12, le_of_lt h23⟩
  have hle : meanVolume eps g vd vo a Tstar ≤ meanVolume eps g vd vo a T₂ := hmin hT₂mem
  refine ⟨Tstar, ⟨?_, ?_⟩, hmin⟩
  · rcases lt_or_eq_of_le hTstar.1 with h | h
    · exact h
    · exfalso; rw [← h] at hle; linarith
  · rcases lt_or_eq_of_le hTstar.2 with h | h
    · exact h
    · exfalso; rw [h] at hle; linarith

/-! ## 3.  Hydrophobic hydration and cold denaturation -/

/-- The solvation free energy of a nonpolar surface, per water in its first shell: the shell
water keeps its hydrogen-bonded open state but loses dense-state orientations, `g → g'`. -/
noncomputable def solv (eps T g g' : ℝ) : ℝ :=
  -T * Real.log ((Real.exp (eps / T) + g') / (Real.exp (eps / T) + g))

/-- **Hydrophobic hydration costs free energy** at every temperature. -/
theorem solv_pos {eps T g g' : ℝ} (hT : 0 < T) (hg' : 0 < g') (hgg : g' < g) :
    0 < solv eps T g g' := by
  have hd : 0 < Real.exp (eps / T) + g := by have := Real.exp_pos (eps / T); linarith
  have hd' : 0 < Real.exp (eps / T) + g' := by have := Real.exp_pos (eps / T); linarith
  have hratio : (Real.exp (eps / T) + g') / (Real.exp (eps / T) + g) < 1 := by
    rw [div_lt_one hd]; linarith
  have hpos : 0 < (Real.exp (eps / T) + g') / (Real.exp (eps / T) + g) := div_pos hd' hd
  have hlog : Real.log ((Real.exp (eps / T) + g') / (Real.exp (eps / T) + g)) < 0 :=
    Real.log_neg hpos hratio
  unfold solv
  nlinarith

/-- **It is entropy dominated**: bounded by the pure orientational entropy term `T·log(g/g')`. -/
theorem solv_le_entropic {eps T g g' : ℝ} (hT : 0 < T) (hg' : 0 < g') (hgg : g' < g) :
    solv eps T g g' ≤ T * Real.log (g / g') := by
  have he := Real.exp_pos (eps / T)
  have hd : 0 < Real.exp (eps / T) + g := by linarith
  have hd' : 0 < Real.exp (eps / T) + g' := by linarith
  have hstep : (Real.exp (eps / T) + g) / (Real.exp (eps / T) + g') ≤ g / g' := by
    rw [div_le_div_iff₀ hd' hg']
    nlinarith
  have hlog : Real.log ((Real.exp (eps / T) + g) / (Real.exp (eps / T) + g'))
      ≤ Real.log (g / g') :=
    Real.log_le_log (div_pos hd hd') hstep
  have hflip : solv eps T g g'
      = T * Real.log ((Real.exp (eps / T) + g) / (Real.exp (eps / T) + g')) := by
    have h := Real.log_inv ((Real.exp (eps / T) + g) / (Real.exp (eps / T) + g'))
    rw [inv_div] at h
    unfold solv
    rw [h]
    ring
  rw [hflip]
  exact mul_le_mul_of_nonneg_left hlog hT.le

/-- **And it strengthens with temperature** — the anomalous signature of hydrophobicity. -/
theorem solv_strictMono {eps T₁ T₂ g g' : ℝ} (heps : 0 < eps) (hT : 0 < T₁) (hlt : T₁ < T₂)
    (hg' : 0 < g') (hgg : g' < g) : solv eps T₁ g g' < solv eps T₂ g g' := by
  have hT₂ : 0 < T₂ := lt_trans hT hlt
  have hflip : ∀ T : ℝ, solv eps T g g'
      = T * Real.log ((Real.exp (eps / T) + g) / (Real.exp (eps / T) + g')) := by
    intro T
    have h := Real.log_inv ((Real.exp (eps / T) + g) / (Real.exp (eps / T) + g'))
    rw [inv_div] at h
    unfold solv
    rw [h]
    ring
  have hdiv : eps / T₂ < eps / T₁ := div_lt_div_of_pos_left heps hT hlt
  have hexp : Real.exp (eps / T₂) < Real.exp (eps / T₁) := Real.exp_lt_exp.2 hdiv
  have he₁ := Real.exp_pos (eps / T₁)
  have he₂ := Real.exp_pos (eps / T₂)
  -- the log factor increases with temperature
  have hratio : (Real.exp (eps / T₁) + g) / (Real.exp (eps / T₁) + g')
      < (Real.exp (eps / T₂) + g) / (Real.exp (eps / T₂) + g') := by
    rw [div_lt_div_iff₀ (by linarith) (by linarith)]
    nlinarith
  have hlogpos : 0 < Real.log ((Real.exp (eps / T₁) + g) / (Real.exp (eps / T₁) + g')) := by
    apply Real.log_pos
    rw [lt_div_iff₀ (by linarith)]
    linarith
  have hloglt : Real.log ((Real.exp (eps / T₁) + g) / (Real.exp (eps / T₁) + g'))
      < Real.log ((Real.exp (eps / T₂) + g) / (Real.exp (eps / T₂) + g')) :=
    Real.log_lt_log (div_pos (by linarith) (by linarith)) hratio
  rw [hflip T₁, hflip T₂]
  nlinarith

/-- **The hydrophobic effect vanishes on cooling.**  For every tolerance there is a temperature
below which the free energy of burying a shell water is smaller than it. -/
theorem solv_tendsto_zero_cold {eps g g' delta : ℝ} (hg' : 0 < g') (hgg : g' < g)
    (hdelta : 0 < delta) :
    ∃ Tc > 0, ∀ T, 0 < T → T < Tc → solv eps T g g' < delta := by
  have hlog : 0 < Real.log (g / g') := Real.log_pos (by rw [lt_div_iff₀ hg']; linarith)
  refine ⟨delta / Real.log (g / g'), by positivity, fun T hT hTc => ?_⟩
  have h1 : solv eps T g g' ≤ T * Real.log (g / g') := solv_le_entropic hT hg' hgg
  have h2 : T * Real.log (g / g') < delta := by
    rw [← lt_div_iff₀ hlog]
    exact hTc
  linarith

/-- **Cold denaturation.**  A structure whose only stabilisation is the burial of `m` shell
waters, and which needs `Greq > 0` of free energy to stay folded, is unstable below an explicit
temperature. -/
theorem cold_denat {eps g g' Greq : ℝ} (m : ℕ) (hm : 0 < m) (hg' : 0 < g') (hgg : g' < g)
    (hG : 0 < Greq) :
    ∃ Tc > 0, ∀ T, 0 < T → T < Tc → (m : ℝ) * solv eps T g g' < Greq := by
  have hmpos : (0 : ℝ) < m := by exact_mod_cast hm
  obtain ⟨Tc, hTc, hbound⟩ :=
    solv_tendsto_zero_cold (eps := eps) (g := g) (g' := g') (delta := Greq / m) hg' hgg
      (by positivity)
  refine ⟨Tc, hTc, fun T hT hlt => ?_⟩
  have := hbound T hT hlt
  rw [lt_div_iff₀ hmpos] at this
  linarith [this]

end IDR.Water
