/-
# Part IX.3  Charge patterning and screening: sequence-dependent physics

Roughly a third of the residues in a typical intrinsically disordered region are charged, and
the dominant sequence-dependent force acting on the chain is the screened Coulomb
interaction between them.  Two facts about it organise the field: the interaction depends on
the *pattern* of the charges and not only on their number, and it is switched off by salt.
This file proves both, with an explicit Debye--Hückel Hamiltonian.

Setting.  A sequence carries charges `q : Fin N → ℝ` (in units of the elementary charge).
For a Gaussian chain of bond length `b` the root-mean-square distance between residues `i`
and `j` is `b√|i-j|` (`RequestProject.Polymer.ideal_gyration` is the same statistic), so the
mean-field Debye--Hückel energy of the sequence at inverse screening length `kappa` is

  `E = Σ_{i<j} q_i q_j · exp(-kappa·b√|i-j|) / (b√|i-j|)`   (`screenedEnergy`).

Alongside it we take the standard sequence descriptor, the sequence charge decoration
`scd = (1/N) Σ_{i<j} q_i q_j √(j-i)` (`scd`), used in the polyampholyte literature to rank
sequences by compactness.

* `netCharge_perm_invariant`, `absCharge_perm_invariant` -- net charge and amino-acid
  composition are invariant under permuting the sequence.
* `scd_blocky_lt_alternating` -- the two permutations `(+ + - -)` and `(+ - + -)` of the same
  composition have *different* charge decoration, the blocky one lower by exactly `4-4√2`.
  Composition is therefore not a sufficient statistic for the physics: this is the concrete,
  Hamiltonian-level instance of the abstract `composition_blind_error` of Part III.
* `screenedEnergy_perm_not_invariant` -- and the same two sequences have different
  Debye--Hückel energies at zero screening, so no model whose input is the composition can
  predict the energy, let alone the ensemble.
* `screenedEnergy_abs_le` -- `|E| ≤ (Σ|q_i|)²·exp(-kappa·b)/b`: an explicit, exponentially
  small bound at high salt, and `screening_tendsto_zero`, `patterning_washed_out` -- as the
  ionic strength grows *every* sequence-dependent electrostatic energy, and every difference
  between two sequences, goes to zero.  Charge patterning is a salt-tunable effect, so a
  model of a charged disordered region must carry the ionic strength as an input; a model
  fitted at one salt concentration is not a model of the region.
-/
import Mathlib
import RequestProject.Polymer

namespace IDR

open Finset

namespace Electro

variable {N : ℕ}

/-- Root-mean-square separation of residues `i` and `j` along a Gaussian chain of bond
length `b`. -/
noncomputable def sep (b : ℝ) (i j : Fin N) : ℝ := b * Real.sqrt (Nat.dist i j)

/-- The mean-field Debye--Hückel energy of a charge sequence on a Gaussian chain, at inverse
screening length `kappa`. -/
noncomputable def screenedEnergy (b kappa : ℝ) (q : Fin N → ℝ) : ℝ :=
  ∑ i, ∑ j, if i < j then q i * q j * (Real.exp (-(kappa * sep b i j)) / sep b i j) else 0

/-- Sequence charge decoration. -/
noncomputable def scd (q : Fin N → ℝ) : ℝ :=
  (1 / (N : ℝ)) * ∑ i, ∑ j, if i < j then q i * q j * Real.sqrt (Nat.dist i j) else 0

/-- Net charge. -/
def netCharge (q : Fin N → ℝ) : ℝ := ∑ i, q i

/-! ## Composition is invariant under patterning, the physics is not -/

theorem netCharge_perm_invariant (q : Fin N → ℝ) (s : Equiv.Perm (Fin N)) :
    netCharge (q ∘ s) = netCharge q :=
  Equiv.sum_comp s q

/-- The multiset of charges -- i.e. the amino-acid composition, as far as charge is concerned
-- is invariant under permutation: every symmetric function of the charges, such as the
fraction of positive residues or the mean squared charge, is blind to the pattern. -/
theorem absCharge_perm_invariant (q : Fin N → ℝ) (s : Equiv.Perm (Fin N)) (f : ℝ → ℝ) :
    ∑ i, f ((q ∘ s) i) = ∑ i, f (q i) :=
  Equiv.sum_comp s (fun i => f (q i))

/-- The blocky sequence `(+ + - -)`. -/
def blocky : Fin 4 → ℝ := ![1, 1, -1, -1]

/-- The alternating sequence `(+ - + -)`: the same composition, permuted. -/
def alternating : Fin 4 → ℝ := ![1, -1, 1, -1]

lemma alternating_eq_perm : ∃ s : Equiv.Perm (Fin 4), alternating = blocky ∘ s := by
  refine ⟨Equiv.mk ![0, 2, 1, 3] ![0, 2, 1, 3] ?_ ?_, ?_⟩
  · decide
  · decide
  · funext i
    fin_cases i <;> simp [alternating, blocky]

lemma scd_blocky : scd blocky = (1 - 2 * Real.sqrt 2 - Real.sqrt 3) / 4 := by
  unfold scd blocky
  simp [Fin.sum_univ_four, Nat.dist]
  ring

lemma scd_alternating :
    scd alternating = (-3 + 2 * Real.sqrt 2 - Real.sqrt 3) / 4 := by
  unfold scd alternating
  simp [Fin.sum_univ_four, Nat.dist]
  ring

/-- **Charge patterning is physical.**  Two sequences of identical composition -- the blocky
and the alternating arrangement of two positive and two negative residues -- have different
charge decoration, the blocky sequence lower (hence more compact) by `4 - 4√2 < 0`. -/
theorem scd_blocky_lt_alternating : scd blocky < scd alternating := by
  rw [scd_blocky, scd_alternating]
  have h2 : (1 : ℝ) < Real.sqrt 2 := by
    have : Real.sqrt 1 < Real.sqrt 2 := Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
    simpa using this
  linarith

/-! ## Screening -/

lemma sep_nonneg {b : ℝ} (hb : 0 ≤ b) (i j : Fin N) : 0 ≤ sep b i j :=
  mul_nonneg hb (Real.sqrt_nonneg _)

lemma sep_ge {b : ℝ} (hb : 0 < b) {i j : Fin N} (hij : i < j) : b ≤ sep b i j := by
  unfold sep
  have h1 : 1 ≤ Nat.dist (i : ℕ) (j : ℕ) := by
    have : (i : ℕ) < (j : ℕ) := hij
    unfold Nat.dist
    omega
  have : (1 : ℝ) ≤ Real.sqrt (Nat.dist (i : ℕ) (j : ℕ)) := by
    have h1' : (1 : ℝ) ≤ (Nat.dist (i : ℕ) (j : ℕ) : ℝ) := by exact_mod_cast h1
    calc (1 : ℝ) = Real.sqrt 1 := by simp
      _ ≤ _ := Real.sqrt_le_sqrt h1'
  nlinarith

/-- The screened kernel is decreasing in the separation. -/
lemma kernel_le {kappa b s : ℝ} (hkappa : 0 ≤ kappa) (hb : 0 < b) (hs : b ≤ s) :
    Real.exp (-(kappa * s)) / s ≤ Real.exp (-(kappa * b)) / b := by
  have hspos : 0 < s := lt_of_lt_of_le hb hs
  have hexp : Real.exp (-(kappa * s)) ≤ Real.exp (-(kappa * b)) := by
    refine Real.exp_le_exp.mpr ?_
    nlinarith
  have hinv : 1 / s ≤ 1 / b := by
    exact one_div_le_one_div_of_le hb hs
  calc Real.exp (-(kappa * s)) / s = Real.exp (-(kappa * s)) * (1 / s) := by ring
    _ ≤ Real.exp (-(kappa * b)) * (1 / b) := by
        refine mul_le_mul hexp hinv (by positivity) (Real.exp_nonneg _)
    _ = Real.exp (-(kappa * b)) / b := by ring

/-- **High salt kills electrostatics.**  The Debye--Hückel energy of *any* sequence is bounded
by `(Σ|q_i|)²·exp(-kappa·b)/b`, exponentially small in the inverse screening length. -/
theorem screenedEnergy_abs_le {b kappa : ℝ} (hb : 0 < b) (hkappa : 0 ≤ kappa) (q : Fin N → ℝ) :
    |screenedEnergy b kappa q| ≤ (∑ i, |q i|) ^ 2 * (Real.exp (-(kappa * b)) / b) := by
  have hKpos : 0 < Real.exp (-(kappa * b)) / b := by positivity
  have hterm : ∀ i j : Fin N,
      |if i < j then q i * q j * (Real.exp (-(kappa * sep b i j)) / sep b i j) else 0|
        ≤ |q i| * |q j| * (Real.exp (-(kappa * b)) / b) := by
    intro i j
    by_cases hij : i < j
    · simp only [hij, if_true]
      rw [abs_mul, abs_mul]
      have hk : |Real.exp (-(kappa * sep b i j)) / sep b i j|
          ≤ Real.exp (-(kappa * b)) / b := by
        have hpos : 0 < sep b i j := lt_of_lt_of_le hb (sep_ge hb hij)
        rw [abs_of_nonneg (by positivity)]
        exact kernel_le hkappa hb (sep_ge hb hij)
      exact mul_le_mul_of_nonneg_left hk (by positivity)
    · simp only [hij, if_false, abs_zero]
      positivity
  calc |screenedEnergy b kappa q|
      ≤ ∑ i, ∑ j, |q i| * |q j| * (Real.exp (-(kappa * b)) / b) := by
        unfold screenedEnergy
        refine le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum (fun i _ => ?_))
        exact le_trans (Finset.abs_sum_le_sum_abs _ _)
          (Finset.sum_le_sum (fun j _ => hterm i j))
    _ = (∑ i, |q i|) ^ 2 * (Real.exp (-(kappa * b)) / b) := by
        have hinner : ∀ i : Fin N, ∑ j, |q i| * |q j| * (Real.exp (-(kappa * b)) / b)
            = |q i| * ((∑ j, |q j|) * (Real.exp (-(kappa * b)) / b)) := by
          intro i
          rw [Finset.sum_mul, Finset.mul_sum]
          exact Finset.sum_congr rfl (fun j _ => by ring)
        rw [Finset.sum_congr rfl (fun i _ => hinner i), ← Finset.sum_mul]
        ring

/-- The electrostatic energy of every sequence vanishes in the high-salt limit. -/
theorem screening_tendsto_zero {b : ℝ} (hb : 0 < b) (q : Fin N → ℝ) :
    Filter.Tendsto (fun kappa : ℝ => screenedEnergy b kappa q) Filter.atTop (nhds 0) := by
  have hbound : Filter.Tendsto
      (fun kappa : ℝ => (∑ i, |q i|) ^ 2 * (Real.exp (-(kappa * b)) / b)) Filter.atTop
      (nhds 0) := by
    have hexp : Filter.Tendsto (fun kappa : ℝ => Real.exp (-(kappa * b))) Filter.atTop
        (nhds 0) := by
      have h1 : Filter.Tendsto (fun kappa : ℝ => -(kappa * b)) Filter.atTop Filter.atBot :=
        Filter.tendsto_neg_atTop_atBot.comp (Filter.tendsto_id.atTop_mul_const hb)
      exact Real.tendsto_exp_atBot.comp h1
    simpa using ((hexp.div_const b).const_mul ((∑ i, |q i|) ^ 2))
  refine squeeze_zero_norm' ?_ hbound
  filter_upwards [Filter.eventually_ge_atTop 0] with kappa hkappa
  simpa [Real.norm_eq_abs] using screenedEnergy_abs_le hb hkappa q

/-- **Patterning is washed out by salt.**  Two sequences whose energies differ at low salt
have energies differing by an exponentially small amount at high salt. -/
theorem patterning_washed_out {b : ℝ} (hb : 0 < b) (q q' : Fin N → ℝ) :
    Filter.Tendsto (fun kappa : ℝ => screenedEnergy b kappa q - screenedEnergy b kappa q')
      Filter.atTop (nhds 0) := by
  simpa using (screening_tendsto_zero hb q).sub (screening_tendsto_zero hb q')

/-- At zero screening the blocky and alternating sequences have different Debye--Hückel
energies: the *energy itself*, and not only a descriptor, distinguishes sequences of
identical composition.  The gap is `(4 - 2√2)/b`. -/
theorem screenedEnergy_blocky_sub_alternating {b : ℝ} (hb : 0 < b) :
    screenedEnergy b 0 blocky - screenedEnergy b 0 alternating = (4 - 2 * Real.sqrt 2) / b := by
  have hb' : b ≠ 0 := ne_of_gt hb
  have h2 : Real.sqrt 2 > 0 := Real.sqrt_pos.mpr (by norm_num)
  have h2' : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  have h3 : Real.sqrt 3 > 0 := Real.sqrt_pos.mpr (by norm_num)
  unfold screenedEnergy blocky alternating sep
  simp [Fin.sum_univ_four, Nat.dist]
  field_simp
  ring_nf
  nlinarith [h2, h2', h3]

/-- **No model whose input is the composition can predict the electrostatic energy.** -/
theorem screenedEnergy_perm_not_invariant {b : ℝ} (hb : 0 < b) :
    screenedEnergy b 0 blocky ≠ screenedEnergy b 0 alternating := by
  intro hcon
  have hdiff := screenedEnergy_blocky_sub_alternating hb
  rw [hcon, sub_self] at hdiff
  have hlt : Real.sqrt 2 < 2 := by
    have : Real.sqrt 2 < Real.sqrt 4 := Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
    calc Real.sqrt 2 < Real.sqrt 4 := this
      _ = 2 := by
          rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  have hnum : (4 : ℝ) - 2 * Real.sqrt 2 > 0 := by linarith
  have : (4 - 2 * Real.sqrt 2) / b > 0 := div_pos hnum hb
  rw [← hdiff] at this
  exact lt_irrefl 0 this

end Electro

end IDR
