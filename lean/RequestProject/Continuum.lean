/-
# Part XI.1  The continuum limit: conformation space is not a finite library

Every earlier part of this development models a disordered region by an `Ens X`: finitely
many conformations carrying weights.  That is what an ensemble *model* is, and it is what a
simulation returns, but it is not what the region *is*.  The true state of a chain in
solution is a probability measure on a continuous conformation space -- a torus of torsion
angles, a bounded region of Cartesian coordinates -- and a physically realistic measure of
that state has **no atoms**: no individual conformation carries positive probability.

This file closes the gap between the two pictures, and the result is sharper than a routine
limit argument.

* `Ens.toMeasure` -- the bridge: a finite ensemble *is* a probability measure (a finite
  mixture of Dirac masses), with `Ens.integral_toMeasure` identifying its integrals with the
  `expect` used throughout Parts I--X.  Everything proved for `Ens` is therefore a statement
  about a genuine measure on conformation space, not about a different object.

* `tvDist_toMeasure_eq_one` -- **no finite model is close to a continuous truth in total
  variation**.  If the target has no atoms, then *every* finite ensemble, of any size, with
  any weights, sits at the maximal total-variation distance `1` from it: the model and the
  truth are mutually singular.  Since `RequestProject.Metric` shows that the uniform
  observational (`ℓ¹`) error *is* total variation, this says that the population-space error
  of Part III does not survive the continuum limit.  Taken alone it would be a counsel of
  despair: refinement, reweighting, and every finite-library method would be maximally wrong
  by construction.

* `exists_ens_lipschitz_approx` -- **but discretisation is legitimate in a transport
  metric**.  On a compact conformation space, for every resolution `eps > 0` there is a
  finite ensemble whose average of *every* `L`-Lipschitz observable is within `L * eps` of
  the truth.  The number of conformations it needs is the covering number of conformation
  space at resolution `eps` -- exactly the rate--distortion count of
  `RequestProject.Quantization`.

* `continuum_dichotomy` -- the two together: at every resolution there is a finite ensemble
  that is weakly `eps`-accurate and simultaneously at total-variation distance `1`.  This is
  the precise statement that a model of a disordered region *must* be scored in a metric
  that sees the geometry of conformation space (`RequestProject.Transport`), and that a
  likelihood-type or population-overlap score against a continuous truth is uninformative.
-/
import Mathlib
import RequestProject.EnsembleCore

namespace IDR

open MeasureTheory Metric Finset
open scoped NNReal ENNReal

variable {X : Type*}

/-! ## The bridge: a finite ensemble is a probability measure -/

namespace Ens

/-- A finite conformational ensemble, read as a probability measure on conformation space:
the finite mixture of Dirac masses at its conformations. -/
noncomputable def toMeasure [MeasurableSpace X] (E : Ens X) : Measure X :=
  ∑ j, ENNReal.ofReal (E.w j) • Measure.dirac (E.pt j)

instance [MeasurableSpace X] (E : Ens X) : IsProbabilityMeasure E.toMeasure := by
  constructor
  simp only [toMeasure, Measure.coe_finset_sum, Finset.sum_apply, Measure.smul_apply,
    smul_eq_mul, measure_univ, mul_one]
  rw [← ENNReal.ofReal_sum_of_nonneg (fun j _ => E.w_nonneg j), E.w_sum, ENNReal.ofReal_one]

/-- Integration against the measure of an ensemble is the ensemble average: the `expect` of
Parts I--X is an honest integral. -/
lemma integral_toMeasure [MeasurableSpace X] (E : Ens X) {f : X → ℝ}
    (hf : StronglyMeasurable f) : ∫ x, f x ∂E.toMeasure = E.expect f := by
  have hint : ∀ j : Fin E.card, Integrable f (Measure.dirac (E.pt j)) := by
    intro j
    refine ⟨hf.aestronglyMeasurable, ?_⟩
    rw [HasFiniteIntegral, lintegral_dirac' _ (by fun_prop)]
    exact ENNReal.coe_lt_top
  rw [toMeasure, integral_finset_sum_measure fun j _ =>
    (hint j).smul_measure ENNReal.ofReal_ne_top]
  simp only [integral_smul_measure, integral_dirac' _ _ hf, smul_eq_mul]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [ENNReal.toReal_ofReal (E.w_nonneg j)]

/-- All of the mass of a finite ensemble sits on its (finite) list of conformations. -/
lemma toMeasure_range [MeasurableSpace X] [MeasurableSingletonClass X] (E : Ens X) :
    E.toMeasure (Set.range E.pt) = 1 := by
  have hmeas : MeasurableSet (Set.range E.pt) := (Set.finite_range E.pt).measurableSet
  simp only [toMeasure, Measure.coe_finset_sum, Finset.sum_apply, Measure.smul_apply,
    smul_eq_mul, Measure.dirac_apply' _ hmeas,
    Set.indicator_of_mem (Set.mem_range_self _), Pi.one_apply, mul_one]
  rw [← ENNReal.ofReal_sum_of_nonneg (fun j _ => E.w_nonneg j), E.w_sum, ENNReal.ofReal_one]

end Ens

/-! ## Total variation, and the mutual singularity of a finite model and a continuous truth -/

/-- Total-variation distance between two measures: the largest disagreement they show on a
measurable event.  On a finite conformational library this is exactly half the `ℓ¹`
population error of `RequestProject.Metric`. -/
noncomputable def tvDist [MeasurableSpace X] (mu nu : Measure X) : ℝ :=
  sSup {r : ℝ | ∃ s : Set X, MeasurableSet s ∧ r = |mu.real s - nu.real s|}

variable [MeasurableSpace X]

lemma tvDist_bddAbove (mu nu : Measure X) [IsProbabilityMeasure mu] [IsProbabilityMeasure nu] :
    BddAbove {r : ℝ | ∃ s : Set X, MeasurableSet s ∧ r = |mu.real s - nu.real s|} := by
  refine ⟨1, ?_⟩
  rintro r ⟨s, -, rfl⟩
  have h1 : mu.real s ≤ 1 := by
    simpa using measureReal_mono (μ := mu) (Set.subset_univ s)
  have h2 : nu.real s ≤ 1 := by
    simpa using measureReal_mono (μ := nu) (Set.subset_univ s)
  rw [abs_le]
  constructor <;> [linarith [measureReal_nonneg (μ := mu) (s := s)];
    linarith [measureReal_nonneg (μ := nu) (s := s)]]

lemma tvDist_le_one (mu nu : Measure X) [IsProbabilityMeasure mu] [IsProbabilityMeasure nu] :
    tvDist mu nu ≤ 1 := by
  refine csSup_le ⟨0, ⟨∅, MeasurableSet.empty, by simp⟩⟩ ?_
  rintro r ⟨s, -, rfl⟩
  have h1 : mu.real s ≤ 1 := by
    simpa using measureReal_mono (μ := mu) (Set.subset_univ s)
  have h2 : nu.real s ≤ 1 := by
    simpa using measureReal_mono (μ := nu) (Set.subset_univ s)
  rw [abs_le]
  constructor <;> [linarith [measureReal_nonneg (μ := mu) (s := s)];
    linarith [measureReal_nonneg (μ := nu) (s := s)]]

/-- Two probability measures that are supported on complementary events are at the maximal
total-variation distance. -/
lemma tvDist_eq_one_of_singular {mu nu : Measure X} [IsProbabilityMeasure mu]
    [IsProbabilityMeasure nu] {s : Set X} (hs : MeasurableSet s) (hmu : mu s = 0)
    (hnu : nu s = 1) : tvDist mu nu = 1 := by
  refine le_antisymm (tvDist_le_one mu nu) ?_
  have hmem : (1 : ℝ) ∈ {r : ℝ | ∃ s : Set X, MeasurableSet s ∧ r = |mu.real s - nu.real s|} := by
    refine ⟨s, hs, ?_⟩
    simp [measureReal_def, hmu, hnu]
  exact le_csSup (tvDist_bddAbove mu nu) hmem

/-- **A finite model of a continuous truth is maximally wrong in total variation.**  If the
target ensemble has no atoms -- no single conformation carries positive probability, which
is the case for any distribution with a density on a continuous conformation space -- then
every finite ensemble, of every size and with every choice of weights, lies at
total-variation distance exactly `1` from it. -/
theorem tvDist_toMeasure_eq_one [MeasurableSingletonClass X] (mu : Measure X)
    [IsProbabilityMeasure mu] [NoAtoms mu] (E : Ens X) : tvDist mu E.toMeasure = 1 :=
  tvDist_eq_one_of_singular (Set.finite_range E.pt).measurableSet
    ((Set.finite_range E.pt).measure_zero mu) E.toMeasure_range

/-- Consequently no finite ensemble is *any* better than any other in total variation: the
population-space error of Part III is saturated by construction, and carries no information
about a continuous target. -/
theorem no_finite_ensemble_tv_approx [MeasurableSingletonClass X] (mu : Measure X)
    [IsProbabilityMeasure mu] [NoAtoms mu] {eps : ℝ} (heps : eps < 1) :
    ¬ ∃ E : Ens X, tvDist mu E.toMeasure ≤ eps := by
  rintro ⟨E, hE⟩
  rw [tvDist_toMeasure_eq_one mu E] at hE
  exact absurd hE (not_le.mpr heps)

/-- A continuous target assigns probability zero to every single structure: a
single-conformation prediction is not merely inaccurate, it is almost surely wrong. -/
lemma prob_singleton_eq_zero (mu : Measure X) [NoAtoms mu] (x : X) : mu {x} = 0 :=
  measure_singleton x

/-! ## Discretisation: a finite ensemble at resolution `eps` -/

section Discretise

variable [MetricSpace X] [BorelSpace X] [CompactSpace X]

omit [MeasurableSpace X] [BorelSpace X] in
/-- A finite `eps`-net of a compact conformation space, indexed by `Fin n`. -/
lemma exists_finite_net {eps : ℝ} (heps : 0 < eps) :
    ∃ (n : ℕ) (c : Fin n → X), ∀ x : X, ∃ i, dist x (c i) < eps := by
  obtain ⟨t, -, hfin, hcov⟩ := (isCompact_univ (X := X)).finite_cover_balls heps
  refine ⟨hfin.toFinset.card, fun i => ((hfin.toFinset.equivFin.symm i : X)), fun x => ?_⟩
  have hx : x ∈ ⋃ y ∈ t, ball y eps := hcov (Set.mem_univ x)
  simp only [Set.mem_iUnion, mem_ball, exists_prop] at hx
  obtain ⟨y, hy, hxy⟩ := hx
  exact ⟨hfin.toFinset.equivFin ⟨y, by simpa using hy⟩, by simp [hxy]⟩

/-- Assignment of a conformation to the first net point within `eps` of it: the
discretisation map. -/
noncomputable def netIndex {n : ℕ} (c : Fin n → X) (eps : ℝ)
    (hcov : ∀ x : X, ∃ i, dist x (c i) < eps) (x : X) : Fin n :=
  (Finset.univ.filter (fun i => dist x (c i) < eps)).min'
    (by obtain ⟨i, hi⟩ := hcov x; exact ⟨i, by simp [hi]⟩)

omit [MeasurableSpace X] [BorelSpace X] [CompactSpace X] in
lemma netIndex_dist {n : ℕ} (c : Fin n → X) (eps : ℝ)
    (hcov : ∀ x : X, ∃ i, dist x (c i) < eps) (x : X) :
    dist x (c (netIndex c eps hcov x)) < eps := by
  have := Finset.min'_mem (Finset.univ.filter (fun i => dist x (c i) < eps))
    (by obtain ⟨i, hi⟩ := hcov x; exact ⟨i, by simp [hi]⟩)
  simpa [netIndex] using this

omit [MeasurableSpace X] [BorelSpace X] [CompactSpace X] in
lemma netIndex_preimage {n : ℕ} (c : Fin n → X) (eps : ℝ)
    (hcov : ∀ x : X, ∃ i, dist x (c i) < eps) (i : Fin n) :
    (netIndex c eps hcov) ⁻¹' {i} =
      {x | dist x (c i) < eps} ∩
        ⋂ j ∈ Finset.univ.filter (· < i), {x | eps ≤ dist x (c j)} := by
  ext x
  simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_inter_iff, Set.mem_setOf_eq,
    Set.mem_iInter, mem_filter, mem_univ, true_and]
  constructor
  · rintro rfl
    refine ⟨netIndex_dist c eps hcov x, fun j hj => ?_⟩
    by_contra h
    push_neg at h
    exact absurd hj (not_lt.mpr (Finset.min'_le _ _ (by simp [h])))
  · rintro ⟨h1, h2⟩
    refine le_antisymm (Finset.min'_le _ _ (by simp [h1])) ?_
    by_contra hlt
    push_neg at hlt
    exact absurd (netIndex_dist c eps hcov x) (not_lt.mpr (h2 _ hlt))

lemma netIndex_measurable {n : ℕ} (c : Fin n → X) (eps : ℝ)
    (hcov : ∀ x : X, ∃ i, dist x (c i) < eps) : Measurable (netIndex c eps hcov) := by
  refine measurable_to_countable' fun i => ?_
  rw [netIndex_preimage]
  refine MeasurableSet.inter (measurableSet_lt (by fun_prop) measurable_const) ?_
  refine MeasurableSet.biInter (Set.to_countable _) fun j _ => ?_
  exact measurableSet_le measurable_const (by fun_prop)

omit [MetricSpace X] [BorelSpace X] [CompactSpace X] in
/-- The push-forward of an integral along a discretisation map is a finite weighted sum: the
mass that lands on net point `i` is the measure of its cell. -/
lemma integral_comp_index (mu : Measure X) [IsProbabilityMeasure mu] {n : ℕ} (c : Fin n → X)
    (g : X → Fin n) (hg : Measurable g) (f : X → ℝ) :
    ∫ x, f (c (g x)) ∂mu = ∑ i, (mu.real (g ⁻¹' {i})) * f (c i) := by
  have key : (fun x => f (c (g x)))
      = fun x => ∑ i, Set.indicator (g ⁻¹' {i}) (fun _ => f (c i)) x := by
    funext x
    rw [Finset.sum_eq_single (g x)]
    · simp [Set.indicator_of_mem]
    · intro i _ hi
      exact Set.indicator_of_notMem (by simp [hi.symm]) _
    · simp
  rw [key, integral_finset_sum _
    (fun i _ => (integrable_const (f (c i))).indicator (hg (measurableSet_singleton i)))]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [integral_indicator_const _ (hg (measurableSet_singleton i))]
  simp [mul_comm]

/-- **Discretisation at resolution `eps`.**  On a compact conformation space, every
probability measure -- however continuous -- is reproduced by a *finite* ensemble to within
`L * eps` on every `L`-Lipschitz observable.  This is what licenses the finite-library
framework of Parts I--X as a model of the continuum: not exactness, but exactness at a
stated resolution, in a metric that sees the geometry of conformation space. -/
theorem exists_ens_lipschitz_approx (mu : Measure X) [IsProbabilityMeasure mu] {eps : ℝ}
    (heps : 0 < eps) :
    ∃ E : Ens X, (∀ x : X, ∃ i, dist x (E.pt i) ≤ eps) ∧
      ∀ (L : ℝ≥0) (f : X → ℝ), LipschitzWith L f →
        |(∫ x, f x ∂mu) - E.expect f| ≤ L * eps := by
  obtain ⟨n, c, hcov⟩ := exists_finite_net (X := X) heps
  set g := netIndex c eps hcov with hgdef
  have hg : Measurable g := netIndex_measurable c eps hcov
  have hgd : ∀ x, dist x (c (g x)) ≤ eps := fun x => (netIndex_dist c eps hcov x).le
  have hwnn : ∀ i, 0 ≤ mu.real (g ⁻¹' {i}) := fun i => measureReal_nonneg
  have hwsum : ∑ i, mu.real (g ⁻¹' {i}) = 1 := by
    have hmeas : ∀ i : Fin n, MeasurableSet (g ⁻¹' {i}) := fun i =>
      hg (measurableSet_singleton i)
    have hdisj : Pairwise (Function.onFun Disjoint fun i : Fin n => g ⁻¹' {i}) := by
      intro i j hij
      exact Set.disjoint_left.2 fun x hx hx' => hij (by
        simp only [Set.mem_preimage, Set.mem_singleton_iff] at hx hx'
        rw [← hx, ← hx'])
    have hsum : mu (⋃ i, g ⁻¹' {i}) = ∑ i, mu (g ⁻¹' {i}) := by
      rw [measure_iUnion hdisj hmeas, tsum_fintype]
    have huniv : (⋃ i, g ⁻¹' ({i} : Set (Fin n))) = Set.univ := by
      ext x; simp
    rw [huniv, measure_univ] at hsum
    have hfin : ∑ i, mu.real (g ⁻¹' {i}) = (1 : ℝ≥0∞).toReal := by
      rw [hsum, ENNReal.toReal_sum (fun i _ => measure_ne_top mu _)]
      rfl
    simpa using hfin
  refine ⟨⟨n, c, fun i => mu.real (g ⁻¹' {i}), hwnn, hwsum⟩, fun x => ⟨g x, hgd x⟩, ?_⟩
  intro L f hf
  have hexp : (⟨n, c, fun i => mu.real (g ⁻¹' {i}), hwnn, hwsum⟩ : Ens X).expect f
      = ∫ x, f (c (g x)) ∂mu := by
    rw [integral_comp_index mu c g hg f]
    rfl
  rw [hexp]
  -- the two integrals differ by at most `L * eps`
  have hcont : Continuous f := hf.continuous
  have h1 : Integrable f mu :=
    hcont.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace f)
  have hmeasf : Measurable fun x => f (c (g x)) :=
    hcont.measurable.comp ((Measurable.of_discrete).comp hg)
  obtain ⟨C, hC⟩ : ∃ C, ∀ x, ‖f (c (g x))‖ ≤ C := by
    obtain ⟨C, hC⟩ := (isCompact_range hcont).isBounded.subset_closedBall 0
    exact ⟨C, fun x => by simpa [Real.norm_eq_abs] using hC (Set.mem_range_self (c (g x)))⟩
  have h2 : Integrable (fun x => f (c (g x))) mu :=
    ⟨hmeasf.aestronglyMeasurable, by
      refine (hasFiniteIntegral_const C).mono (Filter.Eventually.of_forall fun x => ?_)
      simpa using (hC x).trans (le_abs_self C)⟩
  rw [← integral_sub h1 h2]
  refine (abs_integral_le_integral_abs).trans ?_
  have hle : ∫ x, |f x - f (c (g x))| ∂mu ≤ ∫ _x, (L : ℝ) * eps ∂mu := by
    refine integral_mono ((h1.sub h2).abs) (integrable_const _) fun x => ?_
    have h3 := hf.dist_le_mul x (c (g x))
    rw [Real.dist_eq] at h3
    exact h3.trans (mul_le_mul_of_nonneg_left (hgd x) L.2)
  simpa using hle

omit [BorelSpace X] [CompactSpace X] in
/-- **What weak accuracy means geometrically.**  A finite ensemble that reproduces every
`L`-Lipschitz observable of the truth to within `L·eps` must have its structures within `eps`
of the truth *on average*: the mean distance from a conformation drawn from the target to the
nearest structure of the model is at most `eps`.  This is the bridge to the rate--distortion
capacity laws of `RequestProject.Quantization`: a model at resolution `eps` must carry enough
structures to cover the populated part of conformation space at that resolution. -/
theorem mean_infDist_le_of_lipschitz_approx (mu : Measure X) [IsProbabilityMeasure mu]
    (E : Ens X) {eps : ℝ}
    (h : ∀ (L : ℝ≥0) (f : X → ℝ), LipschitzWith L f →
      |(∫ x, f x ∂mu) - E.expect f| ≤ L * eps) :
    ∫ x, Metric.infDist x (Set.range E.pt) ∂mu ≤ eps := by
  have hlip : LipschitzWith 1 (fun x : X => Metric.infDist x (Set.range E.pt)) :=
    Metric.lipschitz_infDist_pt _
  have hzero : E.expect (fun x => Metric.infDist x (Set.range E.pt)) = 0 := by
    simp only [Ens.expect]
    refine Finset.sum_eq_zero fun j _ => ?_
    rw [Metric.infDist_zero_of_mem (Set.mem_range_self j), mul_zero]
  have := h 1 _ hlip
  rw [hzero, sub_zero] at this
  simpa using (abs_le.1 this).2

/-- **The continuum dichotomy.**  Against a continuous (atomless) target on a compact
conformation space, at every resolution `eps` there is a finite ensemble which

* reproduces every `L`-Lipschitz observable to within `L * eps` -- it is an excellent model
  in the transport sense; and yet
* sits at total-variation distance exactly `1` -- it is a maximally wrong model in the
  population sense.

So the choice of metric is not a matter of taste.  A model of a disordered region must be
built and scored against geometry-aware (Lipschitz / transport) functionals; overlap-type
scores against the true state are saturated and cannot distinguish a good finite ensemble
from a bad one. -/
theorem continuum_dichotomy [MeasurableSingletonClass X] (mu : Measure X)
    [IsProbabilityMeasure mu] [NoAtoms mu] {eps : ℝ} (heps : 0 < eps) :
    ∃ E : Ens X,
      (∀ (L : ℝ≥0) (f : X → ℝ), LipschitzWith L f →
        |(∫ x, f x ∂mu) - E.expect f| ≤ L * eps) ∧
      tvDist mu E.toMeasure = 1 := by
  obtain ⟨E, -, hE⟩ := exists_ens_lipschitz_approx mu heps
  exact ⟨E, hE, tvDist_toMeasure_eq_one mu E⟩

end Discretise

end IDR
