/-
# Part XXIV.7  The generative operator: `T_# mu_Z = mu_C`

The earlier parts say what the answer *is* (a probability measure on `R^(3N)`, absolutely
continuous, Gibbs, SE(3)-invariant).  A model must also *produce* it.  This file formalises
the generative operator itself -- a measurable transport map from a simple latent noise
distribution to the physical target -- and proves that one always exists.

* `IsGenerator T muZ muC` -- the push-forward equation `Measure.map T muZ = muC`.
* `IsGenerator.comp` -- generators compose (the many-layer case).
* `IsGenerator.invariant` -- if the latent is invariant under a symmetry `S` and the map is
  equivariant, the generated target is invariant: equivariance of the network *transfers* to
  the physics.  This is the design rule for SE(3)-equivariant architectures.
* `not_isGenerator_of_const` -- a deterministic network with no latent noise generates a
  Dirac measure and therefore cannot generate any absolutely continuous target.  Latent
  randomness is not a stylistic choice.
* `quantile`, `map_quantile_unif` -- the inverse-transform construction: for **every** Borel
  probability measure on `R` the quantile map pushes the uniform latent onto it, proved from
  the monotone/right-continuous structure of the distribution function.
* `exists_generator` -- **the existence theorem**: for every Borel probability measure `mu`
  on the conformation space `R^(3N)` -- in particular for the Gibbs measure of the solvated
  Hamiltonian -- there is a measurable `T` with `T_# unif = mu`.  The generative programme is
  not obstructed at the level of representability; what the earlier parts obstruct is
  producing the *right* `mu` and reporting it honestly.
-/
import Mathlib
import RequestProject.GibbsField

namespace IDR

namespace Generative

open MeasureTheory ProbabilityTheory Set Filter Topology MM Gibbs

/-! ## The push-forward operator -/

/-- `T` generates the target `muC` from the latent `muZ`: `T_# muZ = muC`. -/
def IsGenerator {Z W : Type*} [MeasurableSpace Z] [MeasurableSpace W]
    (T : Z → W) (muZ : Measure Z) (muC : Measure W) : Prop :=
  Measure.map T muZ = muC

/-- Generators compose. -/
theorem IsGenerator.comp {Z W V : Type*} [MeasurableSpace Z] [MeasurableSpace W]
    [MeasurableSpace V] {T : Z → W} {S : W → V} {muZ : Measure Z} {muC : Measure W}
    {muD : Measure V} (hT : IsGenerator T muZ muC) (hS : IsGenerator S muC muD)
    (hTm : Measurable T) (hSm : Measurable S) :
    IsGenerator (S ∘ T) muZ muD := by
  unfold IsGenerator at *
  rw [← Measure.map_map hSm hTm, hT, hS]

/-- **Equivariance transfers.**  If the latent distribution is invariant under a symmetry of
the latent space and the generator intertwines that symmetry with a symmetry of
conformation space, the generated ensemble is invariant. -/
theorem IsGenerator.invariant {Z W : Type*} [MeasurableSpace Z] [MeasurableSpace W]
    {T : Z → W} {muZ : Measure Z} {muC : Measure W} (hT : IsGenerator T muZ muC)
    {a : Z → Z} {b : W → W} (ham : Measurable a) (hbm : Measurable b) (hTm : Measurable T)
    (hlat : Measure.map a muZ = muZ) (hequiv : ∀ z, T (a z) = b (T z)) :
    Measure.map b muC = muC := by
  unfold IsGenerator at hT
  rw [← hT, Measure.map_map hbm hTm]
  have hcomp : b ∘ T = T ∘ a := by funext z; exact (hequiv z).symm
  rw [hcomp, ← Measure.map_map hTm ham, hlat]

/-- **A generator without noise generates nothing.**  A constant map pushes any probability
measure to a Dirac measure, which (for `N ≥ 1`) is not absolutely continuous: a model whose
output is a deterministic function of the sequence alone cannot represent a disordered
ensemble. -/
theorem not_isGenerator_of_const {N : ℕ} (hN : 0 < N) {Z : Type*} [MeasurableSpace Z]
    (muZ : Measure Z) [IsProbabilityMeasure muZ] (x0 : Conf N) (muC : Measure (Conf N))
    (hac : muC ≪ volume) : ¬ IsGenerator (fun _ : Z => x0) muZ muC := by
  intro hgen
  unfold IsGenerator at hgen
  rw [Measure.map_const] at hgen
  simp only [measure_univ, one_smul] at hgen
  rw [← hgen] at hac
  exact dirac_not_absolutelyContinuous hN x0 hac

/-! ## The uniform latent -/

/-- The latent noise: the uniform distribution on `(0,1)`. -/
noncomputable def unif : Measure ℝ := volume.restrict (Ioo 0 1)

instance : IsProbabilityMeasure unif := by
  constructor
  unfold unif
  rw [Measure.restrict_apply_univ, Real.volume_Ioo]
  norm_num

/-! ## The inverse-transform construction -/

variable (ν : Measure ℝ) [IsProbabilityMeasure ν]

/-- The quantile (generalised inverse distribution) function of `ν`. -/
noncomputable def quantile (u : ℝ) : ℝ := sInf {x : ℝ | u ≤ cdf ν x}

omit [IsProbabilityMeasure ν] in
lemma cdfSet_nonempty {u : ℝ} (hu : u < 1) : {x : ℝ | u ≤ cdf ν x}.Nonempty := by
  have h := (tendsto_cdf_atTop ν).eventually_const_lt hu
  obtain ⟨x, hx⟩ := h.exists
  exact ⟨x, le_of_lt hx⟩

omit [IsProbabilityMeasure ν] in
lemma cdfSet_bddBelow {u : ℝ} (hu : 0 < u) : BddBelow {x : ℝ | u ≤ cdf ν x} := by
  have h := (tendsto_cdf_atBot ν).eventually_lt_const hu
  rw [eventually_atBot] at h
  obtain ⟨b, hb⟩ := h
  refine ⟨b, fun x hx => ?_⟩
  by_contra hcon
  push_neg at hcon
  exact absurd (hb x hcon.le) (not_lt.mpr hx)

omit [IsProbabilityMeasure ν] in
/-- The defining property of the quantile: `u ≤ F (quantile u)`.  This is where right
continuity of the distribution function is used. -/
lemma le_cdf_quantile {u : ℝ} (h0 : 0 < u) (h1 : u < 1) : u ≤ cdf ν (quantile ν u) := by
  have hne := cdfSet_nonempty ν h1
  have hbdd := cdfSet_bddBelow ν h0
  have hcont : ContinuousWithinAt (cdf ν) (Ici (quantile ν u)) (quantile ν u) :=
    (cdf ν).right_continuous _
  have htend : Tendsto (cdf ν) (𝓝[>] (quantile ν u)) (𝓝 (cdf ν (quantile ν u))) :=
    hcont.mono_left (nhdsWithin_mono _ Ioi_subset_Ici_self)
  refine ge_of_tendsto htend ?_
  filter_upwards [self_mem_nhdsWithin] with y hy
  obtain ⟨s, hs, hsy⟩ := exists_lt_of_csInf_lt hne hy
  exact le_trans hs ((cdf ν).mono hsy.le)

omit [IsProbabilityMeasure ν] in
/-- **The Galois property of the quantile.** -/
theorem quantile_le_iff {u x : ℝ} (h0 : 0 < u) (h1 : u < 1) :
    quantile ν u ≤ x ↔ u ≤ cdf ν x := by
  constructor
  · intro h
    exact le_trans (le_cdf_quantile ν h0 h1) ((cdf ν).mono h)
  · intro h
    exact csInf_le (cdfSet_bddBelow ν h0) h

/-- The generator built from the quantile function, with an arbitrary constant outside the
unit interval (a null set for the latent). -/
noncomputable def qgen (u : ℝ) : ℝ :=
  if 0 < u ∧ u < 1 then quantile ν u else quantile ν (1/2)

omit [IsProbabilityMeasure ν] in
lemma qgen_preimage_Iic (x : ℝ) :
    qgen ν ⁻¹' (Iic x)
      = (Ioo 0 1 ∩ Iic (cdf ν x)) ∪ ((Ioo 0 1)ᶜ ∩ {_u : ℝ | quantile ν (1/2) ≤ x}) := by
  ext u
  simp only [mem_preimage, mem_Iic, mem_union, mem_inter_iff, mem_Ioo, mem_compl_iff,
    mem_setOf_eq, qgen]
  by_cases hu : 0 < u ∧ u < 1
  · rw [if_pos hu]
    constructor
    · intro h
      exact Or.inl ⟨hu, (quantile_le_iff ν hu.1 hu.2).mp h⟩
    · rintro (⟨-, h⟩ | ⟨hc, -⟩)
      · exact (quantile_le_iff ν hu.1 hu.2).mpr h
      · exact absurd hu hc
  · rw [if_neg hu]
    constructor
    · intro h; exact Or.inr ⟨hu, h⟩
    · rintro (⟨hc, -⟩ | ⟨-, h⟩)
      · exact absurd hc hu
      · exact h

omit [IsProbabilityMeasure ν] in
lemma qgen_measurable : Measurable (qgen ν) := by
  refine measurable_of_Iic (fun x => ?_)
  rw [qgen_preimage_Iic]
  refine MeasurableSet.union ?_ ?_
  · exact (measurableSet_Ioo).inter measurableSet_Iic
  · by_cases h : quantile ν (1/2) ≤ x
    · have : {_u : ℝ | quantile ν (1/2) ≤ x} = univ := by
        ext u
        simp only [mem_setOf_eq, mem_univ, iff_true]
        exact h
      rw [this, inter_univ]
      exact measurableSet_Ioo.compl
    · have : {_u : ℝ | quantile ν (1/2) ≤ x} = ∅ := by
        ext u
        simp only [mem_setOf_eq, mem_empty_iff_false, iff_false]
        exact h
      rw [this, inter_empty]
      exact MeasurableSet.empty

/-- The uniform measure of an initial segment of the unit interval. -/
lemma unif_Iic {c : ℝ} (hc1 : c ≤ 1) :
    unif (Iic c) = ENNReal.ofReal c := by
  unfold unif
  rw [Measure.restrict_apply measurableSet_Iic]
  rcases eq_or_lt_of_le hc1 with h | h
  · rw [h]
    have hset : Iic (1:ℝ) ∩ Ioo 0 1 = Ioo (0:ℝ) 1 := by
      apply inter_eq_right.mpr
      intro y hy
      exact le_of_lt hy.2
    rw [hset, Real.volume_Ioo]
    norm_num
  · have hset : Iic c ∩ Ioo 0 1 = Ioc 0 c := by
      ext y
      simp only [mem_inter_iff, mem_Iic, mem_Ioo, mem_Ioc]
      constructor
      · rintro ⟨h1, h2, -⟩; exact ⟨h2, h1⟩
      · rintro ⟨h1, h2⟩; exact ⟨h2, h1, lt_of_le_of_lt h2 h⟩
    rw [hset, Real.volume_Ioc]
    simp

/-- **The inverse-transform theorem.**  The quantile map pushes the uniform latent onto any
Borel probability measure on the line. -/
theorem map_qgen_unif : Measure.map (qgen ν) unif = ν := by
  haveI : IsProbabilityMeasure (Measure.map (qgen ν) unif) :=
    Measure.isProbabilityMeasure_map (qgen_measurable ν).aemeasurable
  refine Measure.eq_of_cdf _ _ ?_
  refine StieltjesFunction.ext (fun x => ?_)
  rw [cdf_eq_real, cdf_eq_real]
  have hmap : Measure.map (qgen ν) unif (Iic x) = ENNReal.ofReal (cdf ν x) := by
    rw [Measure.map_apply (qgen_measurable ν) measurableSet_Iic, qgen_preimage_Iic]
    have hdisj : unif ((Ioo 0 1 ∩ Iic (cdf ν x))
        ∪ ((Ioo 0 1)ᶜ ∩ {_u : ℝ | quantile ν (1/2) ≤ x}))
        = unif (Ioo 0 1 ∩ Iic (cdf ν x)) := by
      have hnull : unif ((Ioo 0 1)ᶜ ∩ {_u : ℝ | quantile ν (1/2) ≤ x}) = 0 := by
        refine measure_mono_null inter_subset_left ?_
        unfold unif
        rw [Measure.restrict_apply (measurableSet_Ioo.compl)]
        simp
      refine le_antisymm ?_ (measure_mono subset_union_left)
      calc unif _ ≤ unif (Ioo 0 1 ∩ Iic (cdf ν x))
            + unif ((Ioo 0 1)ᶜ ∩ {_u : ℝ | quantile ν (1/2) ≤ x}) := measure_union_le _ _
        _ = unif (Ioo 0 1 ∩ Iic (cdf ν x)) := by rw [hnull, add_zero]
    rw [hdisj]
    have hrestr : unif (Ioo 0 1 ∩ Iic (cdf ν x)) = unif (Iic (cdf ν x)) := by
      unfold unif
      rw [Measure.restrict_apply (measurableSet_Ioo.inter measurableSet_Iic),
        Measure.restrict_apply measurableSet_Iic]
      congr 1
      ext y
      simp only [mem_inter_iff, mem_Ioo, mem_Iic]
      tauto
    rw [hrestr, unif_Iic (cdf_le_one ν x)]
  rw [Measure.real, hmap, ENNReal.toReal_ofReal (cdf_nonneg ν x), ← cdf_eq_real]

/-! ## Existence of a generator for the physical target -/

variable {N : ℕ}

lemma conf_uncountable (hN : 0 < N) : ¬ Countable (Conf N) := by
  intro hc
  have hinj : Function.Injective
      (fun t : ℝ => (fun _ : Fin N => t • (EuclideanSpace.single (0 : Fin 3) (1:ℝ))) :
        ℝ → Conf N) := by
    intro s t hst
    have h := congrFun hst ⟨0, hN⟩
    have hne : (EuclideanSpace.single (0 : Fin 3) (1:ℝ)) ≠ 0 := by
      simp [EuclideanSpace.single_eq_zero_iff]
    exact smul_left_injective ℝ hne h
  have : Countable ℝ := hinj.countable
  exact (Uncountable.not_countable (α := ℝ)) this

/-- **Existence of the generative operator.**  Every Borel probability measure on the
conformation space -- in particular the Boltzmann-Gibbs ensemble of the solvated
Hamiltonian -- is the push-forward of the uniform latent under a measurable map. -/
theorem exists_generator (hN : 0 < N) (mu : Measure (Conf N)) [IsProbabilityMeasure mu] :
    ∃ T : ℝ → Conf N, Measurable T ∧ IsGenerator T unif mu := by
  haveI := conf_uncountable (N := N) hN
  let e : Conf N ≃ᵐ ℝ :=
    PolishSpace.measurableEquivOfNotCountable (conf_uncountable hN)
      (Uncountable.not_countable (α := ℝ))
  haveI : IsProbabilityMeasure (Measure.map e mu) :=
    Measure.isProbabilityMeasure_map e.measurable.aemeasurable
  refine ⟨fun u => e.symm (qgen (Measure.map e mu) u),
    e.symm.measurable.comp (qgen_measurable _), ?_⟩
  unfold IsGenerator
  have hcomp : (fun u => e.symm (qgen (Measure.map e mu) u))
      = e.symm ∘ (qgen (Measure.map e mu)) := rfl
  rw [hcomp, ← Measure.map_map e.symm.measurable (qgen_measurable _),
    map_qgen_unif, e.map_symm_map]

end Generative

end IDR
