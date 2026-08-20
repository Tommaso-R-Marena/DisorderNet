/-
# Part CXXXI  Ionic strength, not salt concentration: the context variable a model must carry

Parts CXXVI–CXXX carry the screening as an inverse Debye length `κ`.  An experiment does not set
`κ`; it sets the composition of the buffer.  The link between the two is the Debye–Hückel
relation

    κ = A · √I,        I = ½ ∑_i c_i z_i²,

where the sum runs over the ionic species of the buffer, `c_i` are their concentrations and `z_i`
their valences, and `A` collects the temperature, the solvent permittivity and Avogadro's number.
The valences enter *squared*.  This part formalises the consequence for the interface of a model
of a disordered region.

* `ionicStrength`, `ionicStrength_single`, `ionicStrength_symmetric` — the definition, and the
  value `c·z²` for a symmetric `z:z` salt at concentration `c`.
* `equal_ionicStrength_equal_prediction` — **ionic strength is a sufficient context variable.**
  Two buffers of the same ionic strength give the same screening and hence exactly the same
  energy for every sequence, at every chain length, under the whole model class.
* `divalent_equivalent_concentration` — the calibration this implies: a `2:2` salt at
  concentration `c` screens exactly like a `1:1` salt at `4c`.
* `energy_dimer` — the elementary read-out used to separate buffers: two unit charges one residue
  apart have energy `e^{−κ}`, strictly decreasing in the screening.
* `concentration_blind_error` — **salt concentration is not a sufficient context variable.**  At
  the same concentration `c > 0`, a monovalent and a divalent buffer give energies differing by
  exactly `e^{−A√c} − e^{−2A√c} > 0`.
* `no_concentration_blind_model` — hence any predictor that is told only the salt concentration is
  wrong by at least half that gap on one of the two buffers; a model must carry the ionic strength
  (equivalently, the valence-weighted composition), and a fit calibrated in a monovalent buffer
  does not transfer to a divalent one.

This is the same lesson as the earlier context theorems of the development, at the level of the
laboratory: the *units of the context input matter*, and the correct one is fixed by the physics
(`I`, quadratic in valence), not by what is convenient to record on a tube.
-/
import Mathlib
import RequestProject.ChargePatterning
import RequestProject.SaltCrossover

set_option autoImplicit false

namespace IDR
namespace Ionic

open Finset

/-! ## 1. Ionic strength and the Debye–Hückel relation -/

/-- The ionic strength `I = ½ ∑ c_i z_i²` of a buffer with `n` ionic species of concentrations
`c` and valences `z`. -/
noncomputable def ionicStrength {n : ℕ} (c z : Fin n → ℝ) : ℝ := (1 / 2) * ∑ i, c i * (z i) ^ 2

/-- The inverse Debye length `κ = A √I`. -/
noncomputable def kappaOf (A I : ℝ) : ℝ := A * Real.sqrt I

lemma ionicStrength_nonneg {n : ℕ} {c z : Fin n → ℝ} (hc : ∀ i, 0 ≤ c i) :
    0 ≤ ionicStrength c z := by
  rw [ionicStrength]
  have : 0 ≤ ∑ i, c i * (z i) ^ 2 :=
    Finset.sum_nonneg fun i _ => mul_nonneg (hc i) (sq_nonneg _)
  linarith

/-- A single ionic species contributes `c z² / 2`. -/
lemma ionicStrength_single (c z : ℝ) :
    ionicStrength (fun _ : Fin 1 => c) (fun _ : Fin 1 => z) = c * z ^ 2 / 2 := by
  simp [ionicStrength]
  ring

/-- **A symmetric `z:z` salt at concentration `c` has ionic strength `c z²`** — the cation and the
anion each contribute `c z² / 2`. -/
lemma ionicStrength_symmetric (c z : ℝ) :
    ionicStrength ![c, c] ![z, -z] = c * z ^ 2 := by
  simp [ionicStrength, Fin.sum_univ_two]
  ring

/-! ## 2. Ionic strength is a sufficient context variable -/

/-- **Two buffers of equal ionic strength are indistinguishable.**  Whatever their composition,
they set the same screening and hence exactly the same energy for every sequence and every chain
length. -/
theorem equal_ionicStrength_equal_prediction {n m : ℕ} {c z : Fin n → ℝ} {c' z' : Fin m → ℝ}
    {A : ℝ} (h : ionicStrength c z = ionicStrength c' z') (N : ℕ) (q : ℕ → ℝ) :
    Salt.energy N (kappaOf A (ionicStrength c z)) q
      = Salt.energy N (kappaOf A (ionicStrength c' z')) q := by
  rw [h]

/-- **The divalent calibration.**  A `2:2` salt at concentration `c` screens exactly like a `1:1`
salt at concentration `4c`. -/
theorem divalent_equivalent_concentration (c : ℝ) :
    ionicStrength ![c, c] ![2, -2] = ionicStrength ![4 * c, 4 * c] ![1, -1] := by
  rw [ionicStrength_symmetric, ionicStrength_symmetric]
  ring

/-! ## 3. Salt concentration is not -/

/-- The energy of two unit charges one residue apart is `e^{−κ}`. -/
lemma energy_dimer (kappa : ℝ) :
    Salt.energy 2 kappa (fun _ => 1) = Real.exp (-kappa) := by
  rw [Salt.energy, Pattern.pairEnergy]
  norm_num [Finset.sum_range_succ, Salt.kern]

lemma energy_dimer_strictAnti {k k' : ℝ} (h : k < k') :
    Salt.energy 2 k' (fun _ => 1) < Salt.energy 2 k (fun _ => 1) := by
  rw [energy_dimer, energy_dimer]
  exact Real.exp_lt_exp.2 (by linarith)

/-- **Salt concentration is not a sufficient context variable.**  At the same concentration `c`, a
monovalent (`1:1`) and a divalent (`2:2`) buffer differ in predicted energy by exactly
`e^{−A√c} − e^{−2A√c}`, which is strictly positive whenever `A, c > 0`. -/
theorem concentration_blind_error {A c : ℝ} (hA : 0 < A) (hc : 0 < c) :
    Salt.energy 2 (kappaOf A (ionicStrength ![c, c] ![1, -1])) (fun _ => 1)
        - Salt.energy 2 (kappaOf A (ionicStrength ![c, c] ![2, -2])) (fun _ => 1)
      = Real.exp (-(A * Real.sqrt c)) - Real.exp (-(2 * (A * Real.sqrt c)))
    ∧ 0 < Real.exp (-(A * Real.sqrt c)) - Real.exp (-(2 * (A * Real.sqrt c))) := by
  have hsc : 0 < Real.sqrt c := Real.sqrt_pos.2 hc
  have h1 : ionicStrength ![c, c] ![1, -1] = c := by
    rw [ionicStrength_symmetric]; ring
  have h2 : ionicStrength ![c, c] ![2, -2] = 4 * c := by
    rw [ionicStrength_symmetric]; ring
  have hs4 : Real.sqrt (4 * c) = 2 * Real.sqrt c := by
    rw [show (4 : ℝ) * c = 2 ^ 2 * c by ring, Real.sqrt_mul (by positivity), Real.sqrt_sq (by norm_num)]
  constructor
  · rw [h1, h2, energy_dimer, energy_dimer, kappaOf, kappaOf, hs4]
    ring_nf
  · have : Real.exp (-(2 * (A * Real.sqrt c))) < Real.exp (-(A * Real.sqrt c)) := by
      apply Real.exp_lt_exp.2
      nlinarith
    linarith

/-- **No concentration-blind model.**  A predictor that is told only the salt concentration
assigns one number to both buffers, and is therefore off by at least half the gap on one of
them. -/
theorem no_concentration_blind_model {A c : ℝ} (hA : 0 < A) (hc : 0 < c) (pred : ℝ → ℝ) :
    (Real.exp (-(A * Real.sqrt c)) - Real.exp (-(2 * (A * Real.sqrt c)))) / 2
      ≤ max |pred c - Salt.energy 2 (kappaOf A (ionicStrength ![c, c] ![1, -1])) (fun _ => 1)|
            |pred c - Salt.energy 2 (kappaOf A (ionicStrength ![c, c] ![2, -2])) (fun _ => 1)| := by
  obtain ⟨hgap, hpos⟩ := concentration_blind_error hA hc
  set E1 := Salt.energy 2 (kappaOf A (ionicStrength ![c, c] ![1, -1])) (fun _ => 1) with hE1
  set E2 := Salt.energy 2 (kappaOf A (ionicStrength ![c, c] ![2, -2])) (fun _ => 1) with hE2
  have htri : |E1 - E2| ≤ |pred c - E1| + |pred c - E2| := by
    calc |E1 - E2| = |(pred c - E2) - (pred c - E1)| := by ring_nf
      _ ≤ |pred c - E2| + |pred c - E1| := abs_sub _ _
      _ = |pred c - E1| + |pred c - E2| := by ring
  have hd : E1 - E2 = Real.exp (-(A * Real.sqrt c)) - Real.exp (-(2 * (A * Real.sqrt c))) := hgap
  have habs : |E1 - E2| = Real.exp (-(A * Real.sqrt c)) - Real.exp (-(2 * (A * Real.sqrt c))) := by
    rw [hd, abs_of_pos hpos]
  rcases le_total |pred c - E1| |pred c - E2| with h | h
  · rw [max_eq_right h]; linarith [htri, habs.symm.le, habs.le]
  · rw [max_eq_left h]; linarith [htri, habs.symm.le, habs.le]

end Ionic
end IDR
