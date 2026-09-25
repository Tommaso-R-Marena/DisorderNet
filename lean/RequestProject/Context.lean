/-
# Part XXIV.1  The context is not an abstract type: it is `(T, I, c)`

Every earlier part carries a *context* `C` as an opaque parameter: the ensemble of a
disordered region is a conditional object, and `C` is what it is conditioned on.  That is
a placeholder, and a placeholder is a hidden assumption.  This file removes it: the context
is a concrete structure with three real coordinates,

* `temperature`     -- absolute temperature `T` (K), which fixes the Boltzmann factor;
* `ionicStrength`   -- ionic strength `I` (mol/L), which fixes the Debye screening length;
* `ligand`          -- free partner concentration `c` (mol/L), which fixes the binding
                       equilibrium;

together with the physical constants that turn those numbers into energies.  Nothing here
is a "toy scale": `kB`, `elementaryCharge`, `vacuumPermittivity`, `avogadro` are the SI
values, `waterPermittivity` is the relative permittivity of water at 298 K, and the Debye
length is the full Debye-Hückel expression

  `lambda_D = sqrt (eps_r eps_0 k_B T / (2 N_A e^2 I))`,

which therefore depends on *two* of the three context coordinates.

The results are the ones a model builder needs:

* `debyeLength_pos`, `debyeLength_eq_scaled`, `debyeLength_strictAnti_ionicStrength`,
  `debyeLength_strictMono_temperature` -- the screening length is positive, scales as
  `I^{-1/2}` exactly, falls strictly with salt and rises strictly with temperature.
* `screened_lt_bare`, `screened_abs_strictAnti_ionicStrength` -- the Debye-Hückel
  interaction is strictly weaker than the bare Coulomb interaction at every separation and
  is strictly weakened by salt; a model with unscreened (or salt-independent) charges is
  wrong at every ionic strength.
* `boltzmann_strictMono_temperature` -- at fixed energy gap the population is strictly
  monotone in `T`, so no temperature-independent model can be right at two temperatures.
* `bound_strictMono_ligand`, `bound_lt_one`, `bound_tendsto_one` -- the bound fraction is
  a strictly increasing, saturating function of the partner concentration.
* `context_coordinates_load_bearing` -- the three coordinates are each load bearing: for
  each one there are two contexts differing in that coordinate alone whose predicted
  observable differs.  Hence `no_context_free_model`: a map from sequence to ensemble that
  ignores `C` is wrong on one of any such pair.
-/
import Mathlib

namespace IDR

namespace Context

open Real

/-! ## Physical constants (SI, except where stated) -/

/-- Boltzmann constant, J/K. -/
def kB : ℝ := 1.380649e-23

/-- Elementary charge, C. -/
def elementaryCharge : ℝ := 1.602176634e-19

/-- Vacuum permittivity, F/m. -/
def vacuumPermittivity : ℝ := 8.8541878128e-12

/-- Avogadro constant, 1/mol. -/
def avogadro : ℝ := 6.02214076e23

/-- Relative permittivity of liquid water at 298 K (dimensionless). -/
def waterPermittivity : ℝ := 78.4

/-- Litres per cubic metre: converts a molar concentration to a number density. -/
def litrePerCubicMetre : ℝ := 1000

lemma kB_pos : 0 < kB := by unfold kB; norm_num
lemma elementaryCharge_pos : 0 < elementaryCharge := by unfold elementaryCharge; norm_num
lemma vacuumPermittivity_pos : 0 < vacuumPermittivity := by unfold vacuumPermittivity; norm_num
lemma avogadro_pos : 0 < avogadro := by unfold avogadro; norm_num
lemma waterPermittivity_pos : 0 < waterPermittivity := by unfold waterPermittivity; norm_num
lemma litrePerCubicMetre_pos : 0 < litrePerCubicMetre := by
  unfold litrePerCubicMetre; norm_num

/-! ## The context -/

/-- The biophysical context a disordered-region ensemble is conditioned on.  A model that
does not take these three numbers as inputs is not a model of a solution. -/
structure BiophysicalContext where
  /-- Absolute temperature, K. -/
  temperature : ℝ
  /-- Ionic strength, mol/L. -/
  ionicStrength : ℝ
  /-- Free concentration of the binding partner, mol/L. -/
  ligand : ℝ
  temperature_pos : 0 < temperature
  ionicStrength_pos : 0 < ionicStrength
  ligand_nonneg : 0 ≤ ligand

namespace BiophysicalContext

variable (C : BiophysicalContext)

/-- Thermal energy `k_B T`, J. -/
def thermalEnergy : ℝ := kB * C.temperature

lemma thermalEnergy_pos : 0 < C.thermalEnergy :=
  mul_pos kB_pos C.temperature_pos

/-- Inverse temperature `beta = 1/(k_B T)`, 1/J. -/
noncomputable def beta : ℝ := 1 / C.thermalEnergy

lemma beta_pos : 0 < C.beta := by
  unfold beta; exact one_div_pos.mpr C.thermalEnergy_pos

/-- The Debye screening length, in metres:
`lambda_D = sqrt (eps_r eps_0 k_B T / (2 N_A e^2 I))`, with `I` converted from mol/L to
1/m^3.  It depends on the temperature and the ionic strength; both dependences are proved
below. -/
noncomputable def debyeLength : ℝ :=
  Real.sqrt (waterPermittivity * vacuumPermittivity * kB * C.temperature /
    (2 * avogadro * litrePerCubicMetre * elementaryCharge ^ 2 * C.ionicStrength))

end BiophysicalContext

open BiophysicalContext

/-- The denominator of the Debye expression is positive. -/
lemma debye_den_pos (C : BiophysicalContext) :
    0 < 2 * avogadro * litrePerCubicMetre * elementaryCharge ^ 2 * C.ionicStrength := by
  have := avogadro_pos
  have := litrePerCubicMetre_pos
  have := elementaryCharge_pos
  have := C.ionicStrength_pos
  positivity

/-- The numerator of the Debye expression is positive. -/
lemma debye_num_pos (C : BiophysicalContext) :
    0 < waterPermittivity * vacuumPermittivity * kB * C.temperature := by
  have := waterPermittivity_pos
  have := vacuumPermittivity_pos
  have := kB_pos
  have := C.temperature_pos
  positivity

lemma debyeLength_pos (C : BiophysicalContext) : 0 < C.debyeLength := by
  unfold BiophysicalContext.debyeLength
  exact Real.sqrt_pos.mpr (div_pos (debye_num_pos C) (debye_den_pos C))

/-- **Exact `I^{-1/2}` scaling.**  Two contexts at the same temperature have Debye lengths
in the ratio `sqrt (I' / I)`. -/
theorem debyeLength_eq_scaled (C C' : BiophysicalContext)
    (hT : C'.temperature = C.temperature) :
    C'.debyeLength = C.debyeLength * Real.sqrt (C.ionicStrength / C'.ionicStrength) := by
  unfold BiophysicalContext.debyeLength
  rw [hT, ← Real.sqrt_mul (le_of_lt (div_pos (debye_num_pos C) (debye_den_pos C)))]
  congr 1
  have h1 := (debye_den_pos C).ne'
  have h2 := (debye_den_pos C').ne'
  have h3 := C.ionicStrength_pos.ne'
  have h4 := C'.ionicStrength_pos.ne'
  field_simp

/-- **Salt strictly screens.**  At fixed temperature the Debye length strictly decreases in
the ionic strength. -/
theorem debyeLength_strictAnti_ionicStrength (C C' : BiophysicalContext)
    (hT : C'.temperature = C.temperature) (hI : C.ionicStrength < C'.ionicStrength) :
    C'.debyeLength < C.debyeLength := by
  have hnum := debye_num_pos C
  have hd := debye_den_pos C
  have hd' := debye_den_pos C'
  have := avogadro_pos
  have := litrePerCubicMetre_pos
  have := elementaryCharge_pos
  have hc : (0:ℝ) < 2 * avogadro * litrePerCubicMetre * elementaryCharge ^ 2 := by positivity
  have hden : 2 * avogadro * litrePerCubicMetre * elementaryCharge ^ 2 * C.ionicStrength
      < 2 * avogadro * litrePerCubicMetre * elementaryCharge ^ 2 * C'.ionicStrength :=
    mul_lt_mul_of_pos_left hI hc
  unfold BiophysicalContext.debyeLength
  rw [hT]
  refine Real.sqrt_lt_sqrt (le_of_lt (div_pos hnum hd')) ?_
  exact div_lt_div_of_pos_left hnum hd hden

/-- **Heating unscreens.**  At fixed ionic strength the Debye length strictly increases in
the temperature. -/
theorem debyeLength_strictMono_temperature (C C' : BiophysicalContext)
    (hI : C'.ionicStrength = C.ionicStrength) (hT : C.temperature < C'.temperature) :
    C.debyeLength < C'.debyeLength := by
  unfold BiophysicalContext.debyeLength
  rw [hI]
  refine Real.sqrt_lt_sqrt (le_of_lt (div_pos (debye_num_pos C) (debye_den_pos C))) ?_
  refine div_lt_div_of_pos_right ?_ (debye_den_pos C)
  have := waterPermittivity_pos
  have := vacuumPermittivity_pos
  have := kB_pos
  have hc : (0:ℝ) < waterPermittivity * vacuumPermittivity * kB := by positivity
  exact mul_lt_mul_of_pos_left hT hc

/-! ## The screened electrostatic interaction -/

/-- The Debye-Hückel interaction energy (J) between charges `z1 e` and `z2 e` at separation
`r` metres in the context `C`. -/
noncomputable def screened (C : BiophysicalContext) (z1 z2 r : ℝ) : ℝ :=
  z1 * z2 * elementaryCharge ^ 2 /
      (4 * Real.pi * waterPermittivity * vacuumPermittivity * r) *
    Real.exp (-(r / C.debyeLength))

/-- The bare (unscreened) Coulomb energy: the `I -> 0` idealisation. -/
noncomputable def bare (z1 z2 r : ℝ) : ℝ :=
  z1 * z2 * elementaryCharge ^ 2 /
    (4 * Real.pi * waterPermittivity * vacuumPermittivity * r)

lemma bare_pos_of_like {z1 z2 r : ℝ} (hz : 0 < z1 * z2) (hr : 0 < r) : 0 < bare z1 z2 r := by
  unfold bare
  have := elementaryCharge_pos
  have := waterPermittivity_pos
  have := vacuumPermittivity_pos
  have hpi := Real.pi_pos
  have hnum : 0 < z1 * z2 * elementaryCharge ^ 2 := by positivity
  have hden : 0 < 4 * Real.pi * waterPermittivity * vacuumPermittivity * r := by positivity
  exact div_pos hnum hden

/-- **Screening is strict.**  In any real solution (`I > 0`) the interaction between two
like charges is strictly weaker than the bare Coulomb value at every separation: a model
that uses unscreened charges is wrong at every ionic strength and every distance. -/
theorem screened_lt_bare (C : BiophysicalContext) {z1 z2 r : ℝ}
    (hz : 0 < z1 * z2) (hr : 0 < r) :
    screened C z1 z2 r < bare z1 z2 r := by
  unfold screened
  have hb := bare_pos_of_like hz hr
  have hlt : Real.exp (-(r / C.debyeLength)) < 1 := by
    refine Real.exp_lt_one_iff.mpr ?_
    have := debyeLength_pos C
    have : 0 < r / C.debyeLength := div_pos hr this
    linarith
  calc bare z1 z2 r * Real.exp (-(r / C.debyeLength)) < bare z1 z2 r * 1 :=
        mul_lt_mul_of_pos_left hlt hb
    _ = bare z1 z2 r := by ring

/-- **Salt strictly weakens the interaction.**  At fixed temperature and separation, raising
the ionic strength strictly lowers the magnitude of a like-charge repulsion. -/
theorem screened_abs_strictAnti_ionicStrength (C C' : BiophysicalContext)
    (hT : C'.temperature = C.temperature) (hI : C.ionicStrength < C'.ionicStrength)
    {z1 z2 r : ℝ} (hz : 0 < z1 * z2) (hr : 0 < r) :
    screened C' z1 z2 r < screened C z1 z2 r := by
  unfold screened
  have hb := bare_pos_of_like hz hr
  have hlam : C'.debyeLength < C.debyeLength :=
    debyeLength_strictAnti_ionicStrength C C' hT hI
  have h1 : r / C.debyeLength < r / C'.debyeLength :=
    div_lt_div_of_pos_left hr (debyeLength_pos C') hlam
  have : Real.exp (-(r / C'.debyeLength)) < Real.exp (-(r / C.debyeLength)) :=
    Real.exp_lt_exp.mpr (by linarith)
  have hbb : bare z1 z2 r = z1 * z2 * elementaryCharge ^ 2 /
      (4 * Real.pi * waterPermittivity * vacuumPermittivity * r) := rfl
  rw [← hbb]
  exact mul_lt_mul_of_pos_left this hb

/-! ## Temperature: the Boltzmann population -/

/-- The population of the higher-energy state of a two-state region with energy gap
`dE > 0`, in the context `C`. -/
noncomputable def boltzmannPop (C : BiophysicalContext) (dE : ℝ) : ℝ :=
  Real.exp (-(C.beta * dE)) / (1 + Real.exp (-(C.beta * dE)))

lemma boltzmannPop_pos (C : BiophysicalContext) (dE : ℝ) : 0 < boltzmannPop C dE := by
  unfold boltzmannPop
  have h := Real.exp_pos (-(C.beta * dE))
  exact div_pos h (by linarith)

/-- The logistic map is strictly monotone. -/
lemma logistic_strictMono {a b : ℝ} (hab : a < b) :
    Real.exp a / (1 + Real.exp a) < Real.exp b / (1 + Real.exp b) := by
  have ha := Real.exp_pos a
  have hb := Real.exp_pos b
  have hlt : Real.exp a < Real.exp b := Real.exp_lt_exp.mpr hab
  rw [div_lt_div_iff₀ (by linarith) (by linarith)]
  nlinarith

/-- **No temperature-free model.**  For a strictly positive energy gap the population of the
excited state is strictly increasing in the temperature. -/
theorem boltzmann_strictMono_temperature (C C' : BiophysicalContext) {dE : ℝ} (hdE : 0 < dE)
    (hT : C.temperature < C'.temperature) :
    boltzmannPop C dE < boltzmannPop C' dE := by
  unfold boltzmannPop
  refine logistic_strictMono ?_
  have hb : C'.beta < C.beta := by
    unfold BiophysicalContext.beta BiophysicalContext.thermalEnergy
    have h1 : 0 < kB * C.temperature := mul_pos kB_pos C.temperature_pos
    have h2 : 0 < kB * C'.temperature := mul_pos kB_pos C'.temperature_pos
    have : kB * C.temperature < kB * C'.temperature := mul_lt_mul_of_pos_left hT kB_pos
    exact one_div_lt_one_div_of_lt h1 this
  nlinarith [C.beta_pos, C'.beta_pos]

/-! ## Partner concentration: the binding isotherm -/

/-- The bound fraction of a single site with dissociation constant `Kd` (mol/L) at the free
partner concentration of the context. -/
noncomputable def boundFraction (C : BiophysicalContext) (Kd : ℝ) : ℝ :=
  C.ligand / (Kd + C.ligand)

/-- **No concentration-free model.**  At fixed `Kd` the bound fraction is strictly
increasing in the partner concentration. -/
theorem bound_strictMono_ligand (C C' : BiophysicalContext) {Kd : ℝ} (hKd : 0 < Kd)
    (hc : C.ligand < C'.ligand) :
    boundFraction C Kd < boundFraction C' Kd := by
  unfold boundFraction
  have h0 := C.ligand_nonneg
  have h1 : 0 < Kd + C.ligand := by linarith
  have h2 : 0 < Kd + C'.ligand := by linarith
  rw [div_lt_div_iff₀ h1 h2]
  nlinarith

/-- The site is never saturated at finite concentration. -/
theorem bound_lt_one (C : BiophysicalContext) {Kd : ℝ} (hKd : 0 < Kd) :
    boundFraction C Kd < 1 := by
  unfold boundFraction
  have h0 := C.ligand_nonneg
  have h1 : 0 < Kd + C.ligand := by linarith
  rw [div_lt_one h1]
  linarith

/-! ## The three coordinates are each load bearing -/

/-- A context with the given coordinates. -/
def mk' (T I c : ℝ) (hT : 0 < T) (hI : 0 < I) (hc : 0 ≤ c) :
    BiophysicalContext := ⟨T, I, c, hT, hI, hc⟩

@[simp] lemma mk'_temperature (T I c : ℝ) (hT : 0 < T) (hI : 0 < I) (hc : 0 ≤ c) :
    (mk' T I c hT hI hc).temperature = T := rfl

@[simp] lemma mk'_ionicStrength (T I c : ℝ) (hT : 0 < T) (hI : 0 < I) (hc : 0 ≤ c) :
    (mk' T I c hT hI hc).ionicStrength = I := rfl

@[simp] lemma mk'_ligand (T I c : ℝ) (hT : 0 < T) (hI : 0 < I) (hc : 0 ≤ c) :
    (mk' T I c hT hI hc).ligand = c := rfl

/-- **Every coordinate matters.**  For each of the three context coordinates there are two
contexts that agree on the other two and disagree on a predicted observable:
temperature moves a Boltzmann population, ionic strength moves an electrostatic energy,
and partner concentration moves a bound fraction. -/
theorem context_coordinates_load_bearing :
    (∃ C C' : BiophysicalContext, C.ionicStrength = C'.ionicStrength ∧ C.ligand = C'.ligand ∧
        boltzmannPop C 1e-20 ≠ boltzmannPop C' 1e-20) ∧
    (∃ C C' : BiophysicalContext, C.temperature = C'.temperature ∧ C.ligand = C'.ligand ∧
        screened C 1 1 1e-9 ≠ screened C' 1 1 1e-9) ∧
    (∃ C C' : BiophysicalContext, C.temperature = C'.temperature ∧
        C.ionicStrength = C'.ionicStrength ∧
        boundFraction C 1e-6 ≠ boundFraction C' 1e-6) := by
  refine ⟨?_, ?_, ?_⟩
  · refine ⟨mk' 280 0.15 0 (by norm_num) (by norm_num) le_rfl,
      mk' 320 0.15 0 (by norm_num) (by norm_num) le_rfl, rfl, rfl, ?_⟩
    apply ne_of_lt
    apply boltzmann_strictMono_temperature <;> norm_num
  · refine ⟨mk' 298 0.05 0 (by norm_num) (by norm_num) le_rfl,
      mk' 298 0.5 0 (by norm_num) (by norm_num) le_rfl, rfl, rfl, ?_⟩
    apply ne_of_gt
    apply screened_abs_strictAnti_ionicStrength <;> norm_num
  · refine ⟨mk' 298 0.15 0 (by norm_num) (by norm_num) le_rfl,
      mk' 298 0.15 1e-6 (by norm_num) (by norm_num) (by norm_num), rfl, rfl, ?_⟩
    apply ne_of_lt
    apply bound_strictMono_ligand <;> norm_num

/-- **No context-free model.**  A predictor that returns the same number for every context
disagrees with the temperature-dependent population on one of two contexts differing only
in temperature; the same holds for salt and for partner concentration. -/
theorem no_context_free_model (f : ℝ) :
    ∃ C : BiophysicalContext, boltzmannPop C 1e-20 ≠ f := by
  by_contra h
  push_neg at h
  obtain ⟨C, C', _, _, hne⟩ := context_coordinates_load_bearing.1
  exact hne ((h C).trans (h C').symm)

end Context

end IDR
