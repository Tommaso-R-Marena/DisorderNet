/-
# Part CXXX  Counterion condensation: the ceiling on the charge-patterning signal

Parts CXX–CXXIX treat the charges of a disordered region as fixed numbers.  Real polyelectrolytes
do not behave that way.  Above a critical linear charge density — the Manning threshold, one
elementary charge per Bjerrum length `lB` — counterions condense onto the chain and the *effective*
charge density stops rising: adding bare charge only adds condensed counterions.  In the standard
counterion-condensation picture the effective charge per residue is

    s_eff = s / max (1, ξ),        ξ = lB · s / b

with `b` the contour spacing between charged residues, `s` the bare charge per residue and `ξ` the
Manning parameter.  Part CXXX puts this into the exactly solved pairwise model of the previous
parts and draws the consequences for what a model can be fitted to.

* `effDensity_below`, `effDensity_above` — the two regimes: below the threshold the effective
  density is the bare one, above it, it is exactly `b / lB`, *independent of the bare charge*.
* `effDensity_le` — hence a universal ceiling `s_eff ≤ b / lB` at every bare density.
* `manningEnergy_eq` — the energy is the bare pattern energy scaled by `s_eff²`
  (`Salt.energy_smul`: the pairwise energy is quadratic in the charge scale).
* `manning_saturation`, `bare_charge_unidentifiable` — **the identifiability failure.**  Two
  regions of *different* bare charge density, both above the Manning threshold, have exactly the
  same energy at every ionic strength and under every separation kernel.  No thermodynamic
  measurement of this class can recover the bare charge density of a strongly charged region;
  only the condensed value `b / lB` is observable.
* `manning_quadratic_below` — below the threshold the signal does grow, exactly as `s²`; so the
  bare density *is* identifiable there, and a titration in charge density (mutagenesis) is
  informative only in the weakly charged regime.
* `manning_ceiling` — **the absolute ceiling.**  At inverse screening length `κ > 0` the
  electrostatic energy of a region of `N` residues obeys
  `|E| ≤ (b/lB)² · 4N / κ²` uniformly over every bare charge density and every charge pattern.
  Charge patterning has a bounded thermodynamic budget, set by the solvent (through `lB`) and the
  chain geometry (through `b`), not by the sequence.
* `patterning_contrast_ceiling` — the same ceiling for the contrast between two sequences, which
  is the quantity a patterning parameter claims to predict.

The design consequence: a model that carries a bare charge-density parameter, or a patterning
score computed from the bare sequence charges, is over-parameterised above the Manning threshold —
its extra freedom is not identifiable from any equilibrium measurement, and it must either
renormalise the charges before scoring or carry `lB` (that is, solvent and temperature) and the
charge spacing as explicit context.
-/
import Mathlib
import RequestProject.ChargePatterning
import RequestProject.SaltCrossover
import RequestProject.SaltTitration

set_option autoImplicit false

namespace IDR
namespace Manning

open Finset

/-! ## 1. The pairwise energy is quadratic in the charge scale -/

/-- Scaling every charge by `c` scales the pairwise energy by `c²`. -/
lemma pairEnergy_smul (N : ℕ) (w q : ℕ → ℝ) (c : ℝ) :
    Pattern.pairEnergy N w (fun i => c * q i) = c ^ 2 * Pattern.pairEnergy N w q := by
  unfold Pattern.pairEnergy
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- The screened energy is quadratic in the charge scale. -/
lemma energy_smul (N : ℕ) (kappa : ℝ) (q : ℕ → ℝ) (c : ℝ) :
    Salt.energy N kappa (fun i => c * q i) = c ^ 2 * Salt.energy N kappa q :=
  pairEnergy_smul N (Salt.kern kappa) q c

/-! ## 2. Counterion condensation -/

/-- The Manning parameter `ξ = lB · s / b`: the number of elementary charges per Bjerrum
length. -/
noncomputable def manningParam (lB b s : ℝ) : ℝ := lB * s / b

/-- The effective (condensed) charge density `s / max (1, ξ)`. -/
noncomputable def effDensity (lB b s : ℝ) : ℝ := s / max 1 (manningParam lB b s)

/-- **Below the Manning threshold nothing is condensed.** -/
theorem effDensity_below {lB b s : ℝ} (hlB : 0 < lB) (hb : 0 < b)
    (hthr : s ≤ b / lB) : effDensity lB b s = s := by
  have hxi : manningParam lB b s ≤ 1 := by
    rw [manningParam, div_le_one hb]
    have h1 : lB * s ≤ lB * (b / lB) := by nlinarith
    have h2 : lB * (b / lB) = b := by field_simp
    linarith
  rw [effDensity, max_eq_left hxi, div_one]

/-- **Above the Manning threshold the effective density is exactly `b / lB`,** whatever the bare
charge density is. -/
theorem effDensity_above {lB b s : ℝ} (hlB : 0 < lB) (hb : 0 < b) (hthr : b / lB ≤ s) :
    effDensity lB b s = b / lB := by
  have hs : 0 < s := lt_of_lt_of_le (by positivity) hthr
  have hxi : 1 ≤ manningParam lB b s := by
    rw [manningParam, le_div_iff₀ hb]
    have h1 : lB * (b / lB) = b := by field_simp
    have h2 : lB * (b / lB) ≤ lB * s := by nlinarith
    linarith
  rw [effDensity, max_eq_right hxi, manningParam]
  field_simp

/-- The effective charge density never exceeds `b / lB`. -/
theorem effDensity_le {lB b s : ℝ} (hlB : 0 < lB) (hb : 0 < b) :
    effDensity lB b s ≤ b / lB := by
  rcases le_total s (b / lB) with h | h
  · rw [effDensity_below hlB hb h]; exact h
  · rw [effDensity_above hlB hb h]

/-- The effective charge density is nonnegative. -/
theorem effDensity_nonneg {lB b s : ℝ} (hlB : 0 < lB) (hb : 0 < b) (hs : 0 ≤ s) :
    0 ≤ effDensity lB b s := by
  rcases le_total s (b / lB) with h | h
  · rw [effDensity_below hlB hb h]; exact hs
  · rw [effDensity_above hlB hb h]; positivity

/-! ## 3. The energy of a condensed chain -/

/-- The screened energy of a region whose bare charge pattern is `q` at bare density `s`, with
counterion condensation taken into account. -/
noncomputable def manningEnergy (N : ℕ) (kappa lB b s : ℝ) (q : ℕ → ℝ) : ℝ :=
  Salt.energy N kappa (fun i => effDensity lB b s * q i)

/-- The energy is the bare pattern energy scaled by the square of the effective density. -/
theorem manningEnergy_eq (N : ℕ) (kappa lB b s : ℝ) (q : ℕ → ℝ) :
    manningEnergy N kappa lB b s q = effDensity lB b s ^ 2 * Salt.energy N kappa q :=
  energy_smul N kappa q _

/-- **Saturation.**  Above the Manning threshold the energy does not depend on the bare charge
density at all. -/
theorem manning_saturation {lB b s s' : ℝ} (hlB : 0 < lB) (hb : 0 < b)
    (hs : b / lB ≤ s) (hs' : b / lB ≤ s') (N : ℕ) (kappa : ℝ) (q : ℕ → ℝ) :
    manningEnergy N kappa lB b s q = manningEnergy N kappa lB b s' q := by
  rw [manningEnergy_eq, manningEnergy_eq, effDensity_above hlB hb hs,
    effDensity_above hlB hb hs']

/-- **The bare charge density of a strongly charged region is not identifiable.**  Two distinct
bare densities above the threshold give the same energy at *every* ionic strength and under
*every* separation kernel. -/
theorem bare_charge_unidentifiable {lB b s s' : ℝ} (hlB : 0 < lB) (hb : 0 < b)
    (hs : b / lB ≤ s) (hs' : b / lB ≤ s') (hne : s ≠ s') (N : ℕ) (q : ℕ → ℝ) :
    s ≠ s' ∧ (∀ kappa : ℝ, manningEnergy N kappa lB b s q = manningEnergy N kappa lB b s' q) ∧
      (∀ w : ℕ → ℝ, Pattern.pairEnergy N w (fun i => effDensity lB b s * q i)
        = Pattern.pairEnergy N w (fun i => effDensity lB b s' * q i)) := by
  refine ⟨hne, fun kappa => manning_saturation hlB hb hs hs' N kappa q, fun w => ?_⟩
  rw [pairEnergy_smul, pairEnergy_smul, effDensity_above hlB hb hs, effDensity_above hlB hb hs']

/-- **Below the threshold the signal is exactly quadratic in the bare density,** so there the bare
density is recoverable from the energy. -/
theorem manning_quadratic_below {lB b s : ℝ} (hlB : 0 < lB) (hb : 0 < b)
    (hthr : s ≤ b / lB) (N : ℕ) (kappa : ℝ) (q : ℕ → ℝ) :
    manningEnergy N kappa lB b s q = s ^ 2 * Salt.energy N kappa q := by
  rw [manningEnergy_eq, effDensity_below hlB hb hthr]

/-! ## 4. The absolute ceiling on electrostatic patterning -/

/-- **The ceiling.**  At inverse screening length `κ > 0`, the electrostatic energy of a region of
`N` residues with unit charge pattern is at most `(b/lB)² · 4N/κ²`, uniformly over every bare
charge density: condensation caps the thermodynamic budget of charge patterning. -/
theorem manning_ceiling {N : ℕ} {kappa lB b s : ℝ} (hk : 0 < kappa) (hlB : 0 < lB) (hb : 0 < b)
    (hs : 0 ≤ s) {q : ℕ → ℝ} (hq : ∀ i, |q i| ≤ 1) :
    |manningEnergy N kappa lB b s q| ≤ (b / lB) ^ 2 * (4 * N / kappa ^ 2) := by
  rw [manningEnergy_eq, abs_mul, abs_of_nonneg (by positivity : (0:ℝ) ≤ effDensity lB b s ^ 2)]
  have h1 : effDensity lB b s ^ 2 ≤ (b / lB) ^ 2 := by
    have h0 := effDensity_nonneg hlB hb hs
    have h2 : effDensity lB b s ≤ b / lB := effDensity_le hlB hb
    nlinarith
  have h3 : |Salt.energy N kappa q| ≤ 4 * N / kappa ^ 2 := Salt.abs_energy_le hk hq
  have h4 : (0:ℝ) ≤ |Salt.energy N kappa q| := abs_nonneg _
  have h5 : (0:ℝ) ≤ (b / lB) ^ 2 := by positivity
  calc effDensity lB b s ^ 2 * |Salt.energy N kappa q|
      ≤ (b / lB) ^ 2 * |Salt.energy N kappa q| := by nlinarith
    _ ≤ (b / lB) ^ 2 * (4 * N / kappa ^ 2) := by nlinarith

/-- The same ceiling for the *contrast* between two charge patterns of the same region — the
quantity any charge-patterning parameter claims to predict. -/
theorem patterning_contrast_ceiling {N : ℕ} {kappa lB b s s' : ℝ} (hk : 0 < kappa) (hlB : 0 < lB)
    (hb : 0 < b) (hs : 0 ≤ s) (hs' : 0 ≤ s') {q q' : ℕ → ℝ}
    (hq : ∀ i, |q i| ≤ 1) (hq' : ∀ i, |q' i| ≤ 1) :
    |manningEnergy N kappa lB b s q - manningEnergy N kappa lB b s' q'|
      ≤ 2 * ((b / lB) ^ 2 * (4 * N / kappa ^ 2)) := by
  have h1 := manning_ceiling (N := N) (s := s) hk hlB hb hs hq
  have h2 := manning_ceiling (N := N) (s := s') hk hlB hb hs' hq'
  calc |manningEnergy N kappa lB b s q - manningEnergy N kappa lB b s' q'|
      ≤ |manningEnergy N kappa lB b s q| + |manningEnergy N kappa lB b s' q'| := abs_sub _ _
    _ ≤ 2 * ((b / lB) ^ 2 * (4 * N / kappa ^ 2)) := by linarith

/-! ## 5. Capstone -/

/-- **The counterion-condensation law for charge-patterning models.**  (1) Below the Manning
threshold the electrostatic signal grows as the square of the bare charge density; (2) above it,
the signal is frozen at the condensed value, so two different bare densities are
indistinguishable at every ionic strength and under every separation kernel; and (3) there is an
absolute ceiling `(b/lB)²·4N/κ²` on the energy, uniform over all bare densities and all
patterns. -/
theorem condensation_law {N : ℕ} {kappa lB b : ℝ} (hk : 0 < kappa) (hlB : 0 < lB) (hb : 0 < b)
    {q : ℕ → ℝ} (hq : ∀ i, |q i| ≤ 1) :
    (∀ s, 0 ≤ s → s ≤ b / lB →
        manningEnergy N kappa lB b s q = s ^ 2 * Salt.energy N kappa q) ∧
    (∀ s s', b / lB ≤ s → b / lB ≤ s' →
        (∀ k : ℝ, manningEnergy N k lB b s q = manningEnergy N k lB b s' q) ∧
          ∀ w : ℕ → ℝ, Pattern.pairEnergy N w (fun i => effDensity lB b s * q i)
            = Pattern.pairEnergy N w (fun i => effDensity lB b s' * q i)) ∧
    (∀ s, 0 ≤ s →
        |manningEnergy N kappa lB b s q| ≤ (b / lB) ^ 2 * (4 * N / kappa ^ 2)) := by
  refine ⟨fun s _ hthr => manning_quadratic_below hlB hb hthr N kappa q, fun s s' hs hs' => ?_,
    fun s hs => manning_ceiling hk hlB hb hs hq⟩
  refine ⟨fun k => manning_saturation hlB hb hs hs' N k q, fun w => ?_⟩
  rw [pairEnergy_smul, pairEnergy_smul, effDensity_above hlB hb hs, effDensity_above hlB hb hs']

end Manning
end IDR
