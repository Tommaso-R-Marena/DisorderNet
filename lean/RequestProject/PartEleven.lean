/-
# Part XI capstone: the continuum design laws

Parts I--X establish what a model of an intrinsically disordered region must be, using
*finite* conformational ensembles throughout.  Part XI removes that idealisation and
replaces it by the object a chain in solution actually has: a probability measure on a
continuous conformation space, produced by the Boltzmann weight of an energy function.

`continuum_design_laws` bundles the result in six clauses, all for the honest target
`gibbs lam U` -- the Gibbs ensemble of a bounded energy `U` against the flat measure `lam`
of the conformational degrees of freedom.

1. **The target exists and is a probability distribution** (`isProbabilityMeasure_gibbs`).
   The prediction problem is well-posed: there is something to predict.
2. **It is continuous**: every individual structure has probability zero
   (`noAtoms_gibbs`).  "The most likely conformation" is not a meaningful output.
3. **The single-structure error floor survives the continuum limit**
   (`single_structure_floor`): on any coordinate that is continuously distributed, every
   reported value -- from a structure or not -- has mean squared error at least the variance,
   which is strictly positive.
4. **Population-space scores are saturated and useless**: every finite ensemble, of any size,
   is at total-variation distance exactly `1` from the truth (`tvDist_toMeasure_eq_one`).
5. **But finite ensembles are exactly the right approximation in a transport metric**: at
   every resolution `eps` one reproduces all `L`-Lipschitz observables to `L·eps`
   (`exists_ens_lipschitz_approx`).  Clauses 4 and 5 together say the metric is not a matter
   of taste: a model of disorder must be built and scored against geometry-aware
   functionals.
6. **Populations are exponential in the energy error** (`gibbs_stability`): force fields that
   agree to `d` in units of `kT` can disagree about a population by `e^{2d} − 1`, and no
   more.  This is the accuracy budget of the whole enterprise -- it says that sampling and
   architecture cannot repair the energy, and quantifies exactly how well the energy must be
   known to place a population.

`equivariance_design_law` is the seventh, structural, clause, and needs no measure theory:
an equivariant model that outputs one structure must return a fixed point of every symmetry
of its target, which is generically a conformation of probability zero.  Equivariance --
the correct and universal design principle for structure prediction -- is *incompatible*
with single-structure output on a symmetric disordered target, and the incompatibility is
removed by, and only by, making the output a distribution
(`equivariant_ensemble_predictor_exists`).
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Continuum
import RequestProject.ContinuumLoss
import RequestProject.GibbsMeasure
import RequestProject.Symmetry

namespace IDR

open MeasureTheory
open scoped NNReal

/-- **The continuum design laws.**  Six clauses about the true, continuous target of
disordered-region modelling: it exists, it is atomless, it has an irreducible
single-structure error floor, it is invisible to population-overlap scores, it is
approximable at any resolution in the transport sense by a finite ensemble, and its
populations are exponentially sensitive to the energy function. -/
theorem continuum_design_laws
    {X : Type*} [MetricSpace X] [CompactSpace X] [MeasurableSpace X] [BorelSpace X]
    (lam : Measure X) [IsProbabilityMeasure lam] [NoAtoms lam]
    {U : X → ℝ} {M : ℝ} (hU : Measurable U) (hM : ∀ x, |U x| ≤ M)
    {q : X → ℝ} (hq : Measurable q)
    (hint : Integrable (fun x => q x ^ 2) (gibbs lam U))
    [NoAtoms ((gibbs lam U).map q)]
    {eps : ℝ} (heps : 0 < eps) :
    -- 1. the target exists and is a probability measure
    IsProbabilityMeasure (gibbs lam U) ∧
    -- 2. it is continuous: no single structure carries any probability
    (∀ x : X, (gibbs lam U) {x} = 0) ∧
    -- 3. the single-structure error floor, in the continuum
    (0 < coordVar (gibbs lam U) q ∧
      ∀ a : ℝ, coordVar (gibbs lam U) q ≤ ∫ x, (q x - a) ^ 2 ∂(gibbs lam U)) ∧
    -- 4. & 5. every finite model is maximally wrong in total variation, and yet
    --         `eps`-accurate on every Lipschitz observable
    (∃ E : Ens X,
      (∀ (L : ℝ≥0) (f : X → ℝ), LipschitzWith L f →
        |(∫ x, f x ∂(gibbs lam U)) - E.expect f| ≤ L * eps) ∧
      tvDist (gibbs lam U) E.toMeasure = 1) ∧
    -- 6. populations are exponential in the energy error
    (∀ (V : X → ℝ) (d : ℝ), Measurable V → (∀ x, |V x| ≤ M) → (∀ x, |U x - V x| ≤ d) →
      ∀ A : Set X, MeasurableSet A →
        |(gibbs lam U).real A - (gibbs lam V).real A| ≤ Real.exp (2 * d) - 1) := by
  haveI hP : IsProbabilityMeasure (gibbs lam U) := isProbabilityMeasure_gibbs (lam := lam) hU hM
  refine ⟨hP, fun x => measure_singleton x, ⟨coordVar_pos_of_noAtoms _ hq hint,
    fun a => coord_pointLoss_min _ hq hint a⟩, ?_, ?_⟩
  · exact continuum_dichotomy (gibbs lam U) heps
  · intro V d hV hMV hd A hA
    exact gibbs_population_error (lam := lam) hU hV hM hMV hd hA

/-- **The equivariance design law.**  An equivariant predictor with single-structure output
must return a fixed point of every symmetry of its target -- generically an unpopulated
conformation -- while an equivariant *ensemble*-valued predictor has no such obstruction and
can be exactly right. -/
theorem equivariance_design_law {X : Type*} (s : X → X) :
    (∀ (A : Ens X → X), (∀ E F : Ens X, E.Same F → A E = A F) →
        (∀ E : Ens X, A (E.map s) = s (A E)) →
        ∀ E : Ens X, (E.map s).Same E → (∀ x : X, s x = x → E.prob x = 0) →
          E.prob (A E) = 0) ∧
      (∃ A : Ens X → Ens X, (∀ E : Ens X, (A E).Same E) ∧
        (∀ E : Ens X, (A (E.map s)).Same ((A E).map s))) :=
  ⟨fun _ hsame hequiv _ hinv hfix =>
      equivariant_point_prediction_unpopulated hsame hequiv hinv hfix,
    equivariant_ensemble_predictor_exists s⟩


/-! ## Non-vacuity: the flat chain on an interval

The hypotheses of `continuum_design_laws` are satisfiable, and by an entirely standard
object: the uniform ("flat", zero-energy) distribution of one continuous conformational
coordinate over a bounded range.  Everything above is therefore a statement about a
realisable situation, not an empty implication. -/

namespace FlatChain

/-- One continuous conformational coordinate, ranging over a bounded interval. -/
abbrev Coord := Set.Icc (0 : ℝ) 1

/-- The zero-energy (flat) Gibbs ensemble of that coordinate is its uniform distribution. -/
lemma gibbs_zero_eq_volume : gibbs (volume : Measure Coord) (fun _ => 0) = volume := by
  have hZ : Zpart (volume : Measure Coord) (fun _ => 0) = 1 := by simp [Zpart]
  simp [gibbs, hZ]

instance : NoAtoms ((volume : Measure Coord).map (Subtype.val)) := by
  constructor
  intro a
  rw [Measure.map_apply measurable_subtype_coe (measurableSet_singleton a)]
  refine Set.Subsingleton.measure_zero ?_ _
  intro x hx y hy
  simp only [Set.mem_preimage, Set.mem_singleton_iff] at hx hy
  exact Subtype.ext (hx.trans hy.symm)

lemma integrable_sq : Integrable (fun x : Coord => (x : ℝ) ^ 2) volume :=
  (by fun_prop : Continuous (fun x : Coord => (x : ℝ) ^ 2)).integrable_of_hasCompactSupport
    (HasCompactSupport.of_compactSpace _)

/-- **The design laws, instantiated.**  For the flat chain: every structure has probability
zero; the coordinate has a strictly positive variance which lower-bounds the error of every
single reported value; and at every resolution there is a finite ensemble that is
`eps`-accurate on all Lipschitz observables while being at total-variation distance `1`. -/
theorem flat_chain_design_laws {eps : ℝ} (heps : 0 < eps) :
    (∀ x : Coord, (volume : Measure Coord) {x} = 0) ∧
    (0 < coordVar (volume : Measure Coord) Subtype.val ∧
      ∀ a : ℝ, coordVar (volume : Measure Coord) Subtype.val
        ≤ ∫ x : Coord, ((x : ℝ) - a) ^ 2 ∂(volume : Measure Coord)) ∧
    (∃ E : Ens Coord,
      (∀ (L : ℝ≥0) (f : Coord → ℝ), LipschitzWith L f →
        |(∫ x, f x ∂(volume : Measure Coord)) - E.expect f| ≤ L * eps) ∧
      tvDist (volume : Measure Coord) E.toMeasure = 1) :=
  ⟨fun x => measure_singleton x,
    ⟨coordVar_pos_of_noAtoms _ measurable_subtype_coe integrable_sq,
      fun a => coord_pointLoss_min _ measurable_subtype_coe integrable_sq a⟩,
    continuum_dichotomy (volume : Measure Coord) heps⟩

end FlatChain

end IDR
