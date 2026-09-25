/-
# Part XI.4  Well-posedness: the ensemble a sequence has, and how accurately it must be earned

Parts I--X take the target of prediction to be "the ensemble of the region in its context".
For the development to be about physics rather than about an abstraction, three things have
to be true of that object, and none of them has been proved so far:

1. **It exists and is unique.**  Given the energy function that the sequence and the context
   determine, there *is* a distribution over the continuous conformation space -- the Gibbs
   measure -- and it is a probability measure (`gibbs`, `isProbabilityMeasure_gibbs`).
2. **It is continuous, not discrete.**  Its populations of single conformations vanish, so
   the continuum obstruction of `RequestProject.Continuum` applies to the *real* target and
   not merely to a hypothetical atomless one (`noAtoms_gibbs`).
3. **It depends on the energy in a controlled way**, or the prediction problem would be
   ill-posed: an arbitrarily small error in the force field could produce an arbitrarily
   large error in the answer.  `gibbs_stability` proves the sharp exponential bound

     `p_U(A) ≤ e^{2d} · p_V(A)`  and  `|p_U(A) − p_V(A)| ≤ e^{2d} − 1`

   whenever the two energies differ by at most `d` (in units of `kT`) everywhere.

Clause 3 is the quantitative design law that the rest of the development needs and that
practitioners feel every day: **populations are exponential in the energy error**.  To place
a population to within a relative factor `1 + eps` the energy function must be right to about
`eps/2` in units of `kT` -- roughly `0.3 kJ/mol` at room temperature for `eps = 0.2`.  No
amount of sampling repairs an energy error; the bound is on the *target* the sampler is
converging to.  (The matching lower statement, that an energy gap of `dE` moves the
population ratio by exactly `e^{-dE}`, is `Thermo.log_pop_ratio` in Part IX.)
-/
import Mathlib

namespace IDR

open MeasureTheory
open scoped NNReal ENNReal

variable {X : Type*} [MeasurableSpace X]

/-- The configurational partition function of an energy `U` (in units of `kT`) against a
reference measure `lam` on conformation space -- the flat measure of the conformational
degrees of freedom. -/
noncomputable def Zpart (lam : Measure X) (U : X → ℝ) : ℝ := ∫ x, Real.exp (-(U x)) ∂lam

/-- The Boltzmann--Gibbs ensemble of the energy `U`: the thermodynamic state of the region. -/
noncomputable def gibbs (lam : Measure X) (U : X → ℝ) : Measure X :=
  (ENNReal.ofReal (Zpart lam U))⁻¹ •
    lam.withDensity (fun x => ENNReal.ofReal (Real.exp (-(U x))))

section Bounded

variable {lam : Measure X} [IsProbabilityMeasure lam] {U V : X → ℝ} {M : ℝ}

lemma expU_integrable (hU : Measurable U) (hM : ∀ x, |U x| ≤ M) :
    Integrable (fun x => Real.exp (-(U x))) lam := by
  refine ⟨(Real.measurable_exp.comp hU.neg).aestronglyMeasurable, ?_⟩
  refine (hasFiniteIntegral_const (Real.exp M)).mono (Filter.Eventually.of_forall fun x => ?_)
  have h1 : -(U x) ≤ M := by have := abs_le.1 (hM x); linarith [this.1]
  simp only [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  exact Real.exp_le_exp.2 h1

/-- The partition function of a bounded energy is bounded below by `e^{-M}`: in particular it
is strictly positive, so the Gibbs ensemble is well defined. -/
lemma Zpart_ge (hU : Measurable U) (hM : ∀ x, |U x| ≤ M) :
    Real.exp (-M) ≤ Zpart lam U := by
  have hint := expU_integrable (lam := lam) hU hM
  have hlb : ∀ x, Real.exp (-M) ≤ Real.exp (-(U x)) := fun x =>
    Real.exp_le_exp.2 (by linarith [(abs_le.1 (hM x)).2])
  have h := integral_mono (integrable_const (Real.exp (-M))) hint hlb
  rw [integral_const] at h
  simpa [Zpart, measureReal_def] using h

lemma Zpart_pos (hU : Measurable U) (hM : ∀ x, |U x| ≤ M) : 0 < Zpart lam U :=
  lt_of_lt_of_le (Real.exp_pos _) (Zpart_ge hU hM)

omit [IsProbabilityMeasure lam] in
/-- The population that the Gibbs ensemble assigns to a conformational event. -/
lemma gibbs_real_apply (hint : Integrable (fun x => Real.exp (-(U x))) lam)
    (hZ : 0 < Zpart lam U) {A : Set X} (hA : MeasurableSet A) :
    (gibbs lam U).real A = (∫ x in A, Real.exp (-(U x)) ∂lam) / Zpart lam U := by
  have hlint : ∫⁻ x in A, ENNReal.ofReal (Real.exp (-(U x))) ∂lam
      = ENNReal.ofReal (∫ x in A, Real.exp (-(U x)) ∂lam) :=
    (ofReal_integral_eq_lintegral_ofReal hint.restrict
      (Filter.Eventually.of_forall fun x => (Real.exp_pos _).le)).symm
  have hnn : 0 ≤ ∫ x in A, Real.exp (-(U x)) ∂lam :=
    integral_nonneg fun _ => (Real.exp_pos _).le
  simp only [gibbs, measureReal_def, Measure.smul_apply, smul_eq_mul, withDensity_apply _ hA,
    hlint]
  rw [ENNReal.toReal_mul, ← ENNReal.ofReal_inv_of_pos hZ, ENNReal.toReal_ofReal (by positivity),
    ENNReal.toReal_ofReal hnn]
  ring

/-- **The Gibbs ensemble exists.**  For a bounded measurable energy it is a probability
measure on conformation space. -/
theorem isProbabilityMeasure_gibbs (hU : Measurable U) (hM : ∀ x, |U x| ≤ M) :
    IsProbabilityMeasure (gibbs lam U) := by
  have hint := expU_integrable (lam := lam) hU hM
  have hZ : 0 < Zpart lam U := Zpart_pos hU hM
  constructor
  have hlint : ∫⁻ x, ENNReal.ofReal (Real.exp (-(U x))) ∂lam
      = ENNReal.ofReal (Zpart lam U) :=
    (ofReal_integral_eq_lintegral_ofReal hint
      (Filter.Eventually.of_forall fun x => (Real.exp_pos _).le)).symm
  simp only [gibbs, Measure.smul_apply, smul_eq_mul, withDensity_apply _ MeasurableSet.univ,
    Measure.restrict_univ, hlint]
  exact ENNReal.inv_mul_cancel
    (by rw [Ne, ENNReal.ofReal_eq_zero]; exact not_le.2 hZ) ENNReal.ofReal_ne_top

/-- **The physical target is continuous.**  If the reference measure of the conformational
degrees of freedom has no atoms -- torsion angles range over a torus, coordinates over a
region of space -- then neither does the Gibbs ensemble.  So the total-variation obstruction
`RequestProject.Continuum.tvDist_toMeasure_eq_one` applies to the real target: no finite
library of structures overlaps it at all. -/
instance noAtoms_gibbs [MeasurableSingletonClass X] (lam : Measure X) [NoAtoms lam]
    (U : X → ℝ) : NoAtoms (gibbs lam U) := by
  constructor
  intro x
  simp only [gibbs, Measure.smul_apply, smul_eq_mul]
  rw [withDensity_apply _ (measurableSet_singleton x),
    Measure.restrict_eq_zero.2 (measure_singleton x)]
  simp

end Bounded

/-! ## Stability: populations are exponential in the energy error -/

section Stability

variable {lam : Measure X} [IsProbabilityMeasure lam] {U V : X → ℝ} {M d : ℝ}

/-- **Force-field error controls population error, exponentially.**  If two energies differ
by at most `d` everywhere (in units of `kT`), every conformational population computed from
one is within a factor `e^{2d}` of the population computed from the other. -/
theorem gibbs_stability (hU : Measurable U) (hV : Measurable V)
    (hMU : ∀ x, |U x| ≤ M) (hMV : ∀ x, |V x| ≤ M) (hd : ∀ x, |U x - V x| ≤ d)
    {A : Set X} (hA : MeasurableSet A) :
    (gibbs lam U).real A ≤ Real.exp (2 * d) * (gibbs lam V).real A := by
  have hintU := expU_integrable (lam := lam) hU hMU
  have hintV := expU_integrable (lam := lam) hV hMV
  have hZU : 0 < Zpart lam U := Zpart_pos hU hMU
  have hZV : 0 < Zpart lam V := Zpart_pos hV hMV
  -- numerator: `e^{-U} ≤ e^{d} e^{-V}` pointwise
  have hnum : ∫ x in A, Real.exp (-(U x)) ∂lam
      ≤ Real.exp d * ∫ x in A, Real.exp (-(V x)) ∂lam := by
    rw [← integral_const_mul]
    refine integral_mono hintU.restrict ((hintV.const_mul (Real.exp d)).restrict) fun x => ?_
    rw [← Real.exp_add]
    exact Real.exp_le_exp.2 (by linarith [(abs_le.1 (hd x)).1, (abs_le.1 (hd x)).2])
  -- denominator: `Z_U ≥ e^{-d} Z_V`
  have hden : Real.exp (-d) * Zpart lam V ≤ Zpart lam U := by
    rw [Zpart, Zpart, ← integral_const_mul]
    refine integral_mono ((hintV.const_mul (Real.exp (-d)))) hintU fun x => ?_
    rw [← Real.exp_add]
    exact Real.exp_le_exp.2 (by linarith [(abs_le.1 (hd x)).1, (abs_le.1 (hd x)).2])
  rw [gibbs_real_apply hintU hZU hA, gibbs_real_apply hintV hZV hA, div_le_iff₀ hZU]
  have hnnV : 0 ≤ ∫ x in A, Real.exp (-(V x)) ∂lam :=
    integral_nonneg fun _ => (Real.exp_pos _).le
  have hexp : Real.exp (2 * d) * Real.exp (-d) = Real.exp d := by
    rw [← Real.exp_add]; ring_nf
  have hstep : Real.exp (2 * d) * ((∫ x in A, Real.exp (-(V x)) ∂lam) / Zpart lam V)
      * (Real.exp (-d) * Zpart lam V)
      = Real.exp d * ∫ x in A, Real.exp (-(V x)) ∂lam := by
    field_simp
    linear_combination (∫ x in A, Real.exp (-(V x)) ∂lam) * hexp
  calc ∫ x in A, Real.exp (-(U x)) ∂lam
      ≤ Real.exp d * ∫ x in A, Real.exp (-(V x)) ∂lam := hnum
    _ = Real.exp (2 * d) * ((∫ x in A, Real.exp (-(V x)) ∂lam) / Zpart lam V)
          * (Real.exp (-d) * Zpart lam V) := hstep.symm
    _ ≤ Real.exp (2 * d) * ((∫ x in A, Real.exp (-(V x)) ∂lam) / Zpart lam V)
          * Zpart lam U := by
        exact mul_le_mul_of_nonneg_left hden
          (mul_nonneg (Real.exp_pos _).le (div_nonneg hnnV hZV.le))

/-- The same statement as an absolute error: two force fields that agree to within `d` in
units of `kT` cannot disagree about any population by more than `e^{2d} − 1`. -/
theorem gibbs_population_error (hU : Measurable U) (hV : Measurable V)
    (hMU : ∀ x, |U x| ≤ M) (hMV : ∀ x, |V x| ≤ M) (hd : ∀ x, |U x - V x| ≤ d)
    {A : Set X} (hA : MeasurableSet A) :
    |(gibbs lam U).real A - (gibbs lam V).real A| ≤ Real.exp (2 * d) - 1 := by
  have hPU : IsProbabilityMeasure (gibbs lam U) := isProbabilityMeasure_gibbs (lam := lam) hU hMU
  have hPV : IsProbabilityMeasure (gibbs lam V) := isProbabilityMeasure_gibbs (lam := lam) hV hMV
  have hd' : ∀ x, |V x - U x| ≤ d := fun x => by rw [abs_sub_comm]; exact hd x
  have h1 := gibbs_stability (lam := lam) hU hV hMU hMV hd hA
  have h2 := gibbs_stability (lam := lam) hV hU hMV hMU hd' hA
  have hbU : (gibbs lam U).real A ≤ 1 := by
    simpa using measureReal_mono (μ := gibbs lam U) (Set.subset_univ A)
  have hbV : (gibbs lam V).real A ≤ 1 := by
    simpa using measureReal_mono (μ := gibbs lam V) (Set.subset_univ A)
  have hnU : 0 ≤ (gibbs lam U).real A := measureReal_nonneg
  have hnV : 0 ≤ (gibbs lam V).real A := measureReal_nonneg
  have hne : Nonempty X := nonempty_of_isProbabilityMeasure lam
  have hd0 : 0 ≤ d := le_trans (abs_nonneg _) (hd (Classical.arbitrary X))
  have hge : (0 : ℝ) ≤ Real.exp (2 * d) - 1 := by
    have : Real.exp 0 ≤ Real.exp (2 * d) := Real.exp_le_exp.2 (by linarith)
    simpa using this
  rw [abs_le]
  refine ⟨?_, ?_⟩
  · nlinarith [mul_le_mul_of_nonneg_left hbU hge]
  · nlinarith [mul_le_mul_of_nonneg_left hbV hge]

end Stability

/-! ## The reference measure is part of the model: the Jacobian is an energy -/

section Reparam

variable {lam : Measure X}

/-- **Changing the flat measure shifts the energy by the log of the density.**  A model that
samples torsion angles uniformly and a model that samples Cartesian coordinates uniformly are
not the same model of the same energy: they are models of energies differing by the
log-Jacobian.  The reference measure of the conformational degrees of freedom is therefore
part of the physics that has to be specified, not a convention. -/
theorem gibbs_reparam (lam : Measure X) {rho : X → ℝ≥0} (hrho : Measurable rho)
    (hpos : ∀ x, 0 < rho x) {U : X → ℝ} (hU : Measurable U) :
    gibbs (lam.withDensity (fun x => (rho x : ℝ≥0∞))) U
      = gibbs lam (fun x => U x - Real.log (rho x)) := by
  have key : ∀ x, Real.exp (-(U x - Real.log (rho x))) = (rho x : ℝ) * Real.exp (-(U x)) := by
    intro x
    rw [neg_sub, Real.exp_sub, Real.exp_log (by exact_mod_cast hpos x), Real.exp_neg]
    field_simp
  have hexp : Measurable fun x => ENNReal.ofReal (Real.exp (-(U x))) :=
    ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp hU.neg)
  have hZ : Zpart (lam.withDensity (fun x => (rho x : ℝ≥0∞))) U
      = Zpart lam (fun x => U x - Real.log (rho x)) := by
    rw [Zpart, Zpart, integral_withDensity_eq_integral_smul hrho]
    refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
    show (rho x) • Real.exp (-(U x)) = Real.exp (-(U x - Real.log (rho x)))
    rw [key x, NNReal.smul_def, smul_eq_mul]
  have hd : (fun x => (rho x : ℝ≥0∞)) * (fun x => ENNReal.ofReal (Real.exp (-(U x))))
      = fun x => ENNReal.ofReal (Real.exp (-(U x - Real.log (rho x)))) := by
    funext x
    simp only [Pi.mul_apply, key x]
    rw [ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_coe_nnreal]
  rw [gibbs, gibbs, hZ]
  congr 1
  rw [← hd, withDensity_mul _ hrho.coe_nnreal_ennreal hexp]

/-- **The price of getting the reference measure wrong**, quantified by the stability law:
if the log-density (log-Jacobian) varies within `d`, then ignoring it moves every population
by at most `e^{2d} − 1` -- and, by `gibbs_stability` being the only bound available, by that
much in the worst case.  A Jacobian of order `kT` is a modelling error of order the effect
being modelled. -/
theorem jacobian_population_error [IsProbabilityMeasure lam] {rho : X → ℝ≥0}
    (hrho : Measurable rho) (hpos : ∀ x, 0 < rho x) {U : X → ℝ} {M d : ℝ}
    (hU : Measurable U) (hM : ∀ x, |U x| ≤ M) (hlog : ∀ x, |Real.log (rho x)| ≤ d)
    {A : Set X} (hA : MeasurableSet A) :
    |(gibbs (lam.withDensity (fun x => (rho x : ℝ≥0∞))) U).real A
        - (gibbs lam U).real A| ≤ Real.exp (2 * d) - 1 := by
  have hne : Nonempty X := nonempty_of_isProbabilityMeasure lam
  have hd0 : 0 ≤ d := le_trans (abs_nonneg _) (hlog (Classical.arbitrary X))
  have hlogm : Measurable fun x => U x - Real.log (rho x) :=
    hU.sub (Real.measurable_log.comp hrho.coe_nnreal_real)
  rw [gibbs_reparam lam hrho hpos hU]
  have hMU : ∀ x, |U x| ≤ M + d := fun x => (hM x).trans (by linarith)
  have hMV : ∀ x, |U x - Real.log (rho x)| ≤ M + d := fun x =>
    (abs_sub _ _).trans (by linarith [hM x, hlog x])
  have hdd : ∀ x, |(U x - Real.log (rho x)) - U x| ≤ d := fun x => by
    have : (U x - Real.log (rho x)) - U x = -(Real.log (rho x)) := by ring
    rw [this, abs_neg]
    exact hlog x
  exact gibbs_population_error (lam := lam) hlogm hU hMV hMU hdd hA

end Reparam

end IDR
