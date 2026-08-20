/-
# Part XXIV.5  The Boltzmann-Gibbs measure of the continuous Hamiltonian

The target of a model of a disordered region is now pinned down completely: it is the
probability measure on `R^(3N)` with density `exp (-H(x)/k_B T) / Z(T)`, and this file
constructs it, proves it is a probability measure, proves it has a Radon-Nikodym derivative
with respect to Lebesgue measure, and proves the two identities a generative model actually
needs.

A `GibbsData` packages what the previous files supply: an inverse temperature, a measurable
energy, a container `D` of finite positive volume, and a lower bound for the energy on `D`
(supplied by `Hsolv_bddBelow`).  Nothing else is assumed.

* `Zpart_pos`, `Zpart_lt_top`, `gibbsMeasure_isProbabilityMeasure` -- the partition function
  is finite and strictly positive, so the Gibbs measure exists and is a probability measure.
* `gibbs_absolutelyContinuous`, `gibbs_rnDeriv` -- it is absolutely continuous with respect
  to Lebesgue measure and its Radon-Nikodym derivative is the Boltzmann density.  A
  generative model may therefore be asked for a density at all.
* `density_ratio` -- **the identity that removes the partition function**:
  `f(x1)/f(x2) = exp (-(H(x1) - H(x2))/k_B T)`.  Relative probabilities are computable
  without `Z`.
* `score_hasFDerivAt`, `score_eq_neg_beta_grad` -- **the score identity**: the gradient of
  `log f` is exactly `-beta` times the gradient of the energy, independent of `Z`.  This is
  the precise statement that makes score matching and diffusion training well posed.
* `gibbsMeasure_rigid_invariant` -- the Gibbs measure is invariant under the diagonal action
  of every rigid motion, given an invariant energy and container: the SE(3) invariance
  requirement, at the level of the measure rather than of a single number.
* `dirac_not_absolutelyContinuous` -- a single-structure prediction is not merely a bad
  density: it has none.  The output object of a model of a disordered region cannot be a
  point of `R^(3N)`.
-/
import Mathlib
import RequestProject.Solvation

namespace IDR

namespace Gibbs

open MeasureTheory Real MM
open scoped ENNReal

variable {N : ℕ}

/-! ## The Boltzmann weight -/

/-- The Boltzmann weight `exp (-beta U(x))`. -/
noncomputable def weight (beta : ℝ) (U : Conf N → ℝ) (x : Conf N) : ℝ :=
  Real.exp (-(beta * U x))

lemma weight_pos (beta : ℝ) (U : Conf N → ℝ) (x : Conf N) : 0 < weight beta U x :=
  Real.exp_pos _

/-- Everything the construction needs, and nothing more. -/
structure GibbsData (N : ℕ) where
  /-- Inverse temperature `1/(k_B T)`. -/
  beta : ℝ
  /-- The potential energy (for instance `Hsolv`). -/
  U : Conf N → ℝ
  /-- The container: the sample is in a vessel of finite volume. -/
  D : Set (Conf N)
  /-- A lower bound for the energy on the container (supplied by the stability theorem). -/
  B : ℝ
  beta_pos : 0 < beta
  U_meas : Measurable U
  D_meas : MeasurableSet D
  D_finite : volume D ≠ ⊤
  D_pos : 0 < volume D
  U_bddBelow : ∀ x ∈ D, -B ≤ U x

namespace GibbsData

variable (G : GibbsData N)

/-- The partition function of the container. -/
noncomputable def Zpart : ℝ := ∫ x in G.D, weight G.beta G.U x

lemma weight_measurable : Measurable (weight G.beta G.U) := by
  unfold weight
  exact (G.U_meas.const_mul G.beta).neg.exp

lemma weight_le_on_D {x : Conf N} (hx : x ∈ G.D) :
    weight G.beta G.U x ≤ Real.exp (G.beta * G.B) := by
  unfold weight
  apply Real.exp_le_exp.mpr
  have := G.U_bddBelow x hx
  nlinarith [G.beta_pos]

lemma integrableOn_weight : IntegrableOn (weight G.beta G.U) G.D volume := by
  refine Measure.integrableOn_of_bounded (M := Real.exp (G.beta * G.B)) G.D_finite
    (G.weight_measurable.aestronglyMeasurable) ?_
  filter_upwards [ae_restrict_mem G.D_meas] with x hx
  rw [Real.norm_eq_abs, abs_of_pos (weight_pos _ _ _)]
  exact G.weight_le_on_D hx

/-- **The partition function is strictly positive.** -/
theorem Zpart_pos : 0 < G.Zpart := by
  unfold Zpart
  rw [setIntegral_pos_iff_support_of_nonneg_ae ?_ G.integrableOn_weight]
  · have hsupp : Function.support (weight G.beta G.U) = Set.univ := by
      ext x
      simp [Function.mem_support, (weight_pos G.beta G.U x).ne']
    rw [hsupp, Set.univ_inter]
    exact G.D_pos
  · filter_upwards with x using (weight_pos _ _ x).le

lemma Zpart_ne_zero : G.Zpart ≠ 0 := G.Zpart_pos.ne'

/-! ## The Gibbs density and the Gibbs measure -/

/-- The Boltzmann-Gibbs probability density `exp (-beta U(x)) / Z`, supported on the
container. -/
noncomputable def density (x : Conf N) : ℝ :=
  Set.indicator G.D (fun y => weight G.beta G.U y / G.Zpart) x

lemma density_nonneg (x : Conf N) : 0 ≤ G.density x := by
  unfold density
  refine Set.indicator_nonneg (fun y _ => ?_) x
  exact le_of_lt (div_pos (weight_pos _ _ _) G.Zpart_pos)

lemma density_of_mem {x : Conf N} (hx : x ∈ G.D) :
    G.density x = weight G.beta G.U x / G.Zpart := Set.indicator_of_mem hx _

lemma density_measurable : Measurable G.density := by
  unfold density
  exact (G.weight_measurable.div_const _).indicator G.D_meas

/-- The Gibbs measure: Lebesgue measure weighted by the Boltzmann density. -/
noncomputable def gibbsMeasure : Measure (Conf N) :=
  volume.withDensity (fun x => ENNReal.ofReal (G.density x))

lemma lintegral_density : ∫⁻ x, ENNReal.ofReal (G.density x) = 1 := by
  have hint : Integrable G.density volume := by
    unfold density
    rw [integrable_indicator_iff G.D_meas]
    exact G.integrableOn_weight.div_const _
  have h1 : ∫ x, G.density x = 1 := by
    unfold density
    rw [integral_indicator G.D_meas, integral_div]
    exact div_self G.Zpart_ne_zero
  rw [← ofReal_integral_eq_lintegral_ofReal hint
    (Filter.Eventually.of_forall G.density_nonneg), h1]
  simp

instance gibbsMeasure_isProbabilityMeasure : IsProbabilityMeasure G.gibbsMeasure := by
  constructor
  unfold gibbsMeasure
  rw [withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ]
  exact G.lintegral_density

/-- **The model's answer is a density.**  The Gibbs measure is absolutely continuous with
respect to Lebesgue measure. -/
theorem gibbs_absolutelyContinuous : G.gibbsMeasure ≪ volume :=
  withDensity_absolutelyContinuous _ _

/-- **The Radon-Nikodym derivative is the Boltzmann density.** -/
theorem gibbs_rnDeriv :
    G.gibbsMeasure.rnDeriv volume =ᵐ[volume] fun x => ENNReal.ofReal (G.density x) :=
  Measure.rnDeriv_withDensity volume (G.density_measurable.ennreal_ofReal)

/-- The Gibbs measure of a measurable set is the integral of the density over it. -/
theorem gibbs_apply {A : Set (Conf N)} (hA : MeasurableSet A) :
    G.gibbsMeasure A = ∫⁻ x in A, ENNReal.ofReal (G.density x) := by
  unfold gibbsMeasure
  rw [withDensity_apply _ hA]

/-! ## The two identities a generative model needs -/

/-- **Relative probabilities need no partition function.**  For two conformations in the
container the ratio of densities is `exp (-beta (U x1 - U x2))`. -/
theorem density_ratio {x1 x2 : Conf N} (h1 : x1 ∈ G.D) (h2 : x2 ∈ G.D) :
    G.density x1 / G.density x2 = Real.exp (-(G.beta * (G.U x1 - G.U x2))) := by
  have hz := G.Zpart_ne_zero
  have hw2 := (weight_pos G.beta G.U x2).ne'
  rw [G.density_of_mem h1, G.density_of_mem h2]
  have hcancel : weight G.beta G.U x1 / G.Zpart / (weight G.beta G.U x2 / G.Zpart)
      = weight G.beta G.U x1 / weight G.beta G.U x2 := by
    field_simp
  rw [hcancel]
  unfold weight
  rw [← Real.exp_sub]
  congr 1
  ring

/-- The logarithm of the density is affine in the energy. -/
theorem log_density_of_mem {x : Conf N} (hx : x ∈ G.D) :
    Real.log (G.density x) = -(G.beta * G.U x) - Real.log G.Zpart := by
  rw [G.density_of_mem hx, Real.log_div (weight_pos _ _ _).ne' G.Zpart_ne_zero]
  unfold weight
  rw [Real.log_exp]

/-- **The score identity.**  Wherever the energy is differentiable and the container is a
neighbourhood, the gradient of `log f` is exactly `-beta` times the gradient of the energy.
The partition function has disappeared: this is what makes score matching and diffusion
training of an ensemble model well posed. -/
theorem score_hasFDerivAt {x : Conf N} (hx : G.D ∈ nhds x) {dU : Conf N →L[ℝ] ℝ}
    (hU : HasFDerivAt G.U dU x) :
    HasFDerivAt (fun y => Real.log (G.density y)) (-G.beta • dU) x := by
  have heq : (fun y => Real.log (G.density y))
      =ᶠ[nhds x] fun y => -G.beta * G.U y - Real.log G.Zpart := by
    filter_upwards [hx] with y hy
    rw [G.log_density_of_mem hy]
    ring_nf
  refine HasFDerivAt.congr_of_eventuallyEq ?_ heq
  exact (hU.const_mul (-G.beta)).sub_const (Real.log G.Zpart)

/-- The score in coordinates: `grad log f = -beta grad U`. -/
theorem score_eq_neg_beta_grad {x : Conf N} (hx : G.D ∈ nhds x) {dU : Conf N →L[ℝ] ℝ}
    (hU : HasFDerivAt G.U dU x) :
    fderiv ℝ (fun y => Real.log (G.density y)) x = -G.beta • dU :=
  (G.score_hasFDerivAt hx hU).fderiv

end GibbsData

/-! ## Rigid-motion invariance of the measure -/

/-- The diagonal action of a rigid motion preserves Lebesgue measure on `R^(3N)`. -/
theorem act_measurePreserving (g : RigidMotion) :
    MeasurePreserving (g.act : Conf N → Conf N) volume volume := by
  have hrot : MeasurePreserving (g.rot : Point → Point) volume volume :=
    g.rot.measurePreserving
  have htr : MeasurePreserving (fun p : Point => p + g.trans) volume volume :=
    measurePreserving_add_right volume g.trans
  have hmap : MeasurePreserving g.map volume volume := htr.comp hrot
  exact volume_preserving_pi (fun _ => hmap)

lemma act_measurable (g : RigidMotion) : Measurable (g.act : Conf N → Conf N) :=
  (act_measurePreserving g).measurable

/-- The action of a rigid motion as a measurable equivalence of conformation space. -/
noncomputable def actEquiv (g : RigidMotion) : Conf N ≃ᵐ Conf N where
  toFun := g.act
  invFun := g.inv.act
  left_inv := g.act_inv_act
  right_inv := fun x => by
    have := g.act_act_inv x
    simpa using this
  measurable_toFun := act_measurable g
  measurable_invFun := act_measurable g.inv

lemma act_measurableEmbedding (g : RigidMotion) :
    MeasurableEmbedding (g.act : Conf N → Conf N) :=
  (actEquiv g).measurableEmbedding

/-- A measure with an invariant density is invariant. -/
theorem map_withDensity_of_invariant {T : Conf N → Conf N}
    (hT : MeasurePreserving T volume volume) {f : Conf N → ℝ≥0∞} (hf : Measurable f)
    (hinv : ∀ x, f (T x) = f x) :
    Measure.map T (volume.withDensity f) = volume.withDensity f := by
  ext A hA
  rw [Measure.map_apply hT.measurable hA, withDensity_apply _ (hT.measurable hA),
    withDensity_apply _ hA, ← lintegral_indicator (hT.measurable hA),
    ← lintegral_indicator hA]
  have hpt : ∀ x, (T ⁻¹' A).indicator f x = (A.indicator f) (T x) := by
    intro x
    by_cases hx : T x ∈ A
    · rw [Set.indicator_of_mem (by exact hx) , Set.indicator_of_mem hx, hinv]
    · rw [Set.indicator_of_notMem (by exact hx), Set.indicator_of_notMem hx]
  calc ∫⁻ x, (T ⁻¹' A).indicator f x = ∫⁻ x, (A.indicator f) (T x) := by
        exact lintegral_congr hpt
    _ = ∫⁻ y, (A.indicator f) y := hT.lintegral_comp (hf.indicator hA)

/-- **SE(3) invariance of the target measure.**  If the energy and the container are
invariant under a rigid motion, so is the Gibbs measure: the probability of a set equals
the probability of its rotated, translated image. -/
theorem gibbsMeasure_rigid_invariant (G : GibbsData N) (g : RigidMotion)
    (hU : ∀ x, G.U (g.act x) = G.U x) (hD : ∀ x, (g.act x ∈ G.D ↔ x ∈ G.D)) :
    Measure.map g.act G.gibbsMeasure = G.gibbsMeasure := by
  unfold GibbsData.gibbsMeasure
  refine map_withDensity_of_invariant (act_measurePreserving g)
    (G.density_measurable.ennreal_ofReal) (fun x => ?_)
  congr 1
  unfold GibbsData.density weight
  by_cases hx : x ∈ G.D
  · rw [Set.indicator_of_mem ((hD x).mpr hx), Set.indicator_of_mem hx, hU]
  · rw [Set.indicator_of_notMem (fun hc => hx ((hD x).mp hc)),
      Set.indicator_of_notMem hx]

/-! ## A single structure is not an admissible answer -/

/-- **No density, not just a bad one.**  For `N ≥ 1` a point prediction -- the Dirac measure
at one conformation -- is not absolutely continuous with respect to Lebesgue measure, so it
has no Radon-Nikodym derivative at all.  The output of a model of a disordered region must
be a genuine density on `R^(3N)`. -/
theorem dirac_not_absolutelyContinuous (hN : 0 < N) (x : Conf N) :
    ¬ (Measure.dirac x ≪ (volume : Measure (Conf N))) := by
  intro hac
  haveI : Nonempty (Fin N) := ⟨⟨0, hN⟩⟩
  have hnull : volume ({x} : Set (Conf N)) = 0 := by
    have hset : ({x} : Set (Conf N)) = Set.pi Set.univ (fun i => {x i}) := by
      ext y; simp [funext_iff]
    rw [hset, volume_pi_pi]
    simp
    omega
  have := hac hnull
  rw [Measure.dirac_apply_of_mem (Set.mem_singleton x)] at this
  exact one_ne_zero this

end Gibbs

end IDR
