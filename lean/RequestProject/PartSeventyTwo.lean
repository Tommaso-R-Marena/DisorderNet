/-
# Part LXXII  The design and reporting theorem

Parts LXIX-LXXI price the ensemble: the restraint count needed to determine it is `m - 1`,
exponential in the conformational entropy; the verifiable functionals form a subspace of dimension
at most `k + 1`; and finite tolerance sets a resolution floor no further experiment removes.
`RequestProject.ReportDesign` turns those limits into a procedure, and proves the procedure
optimal.

`IDR.report_design_laws` bundles four statements:

1. *Identifiability comes with a certificate.*  A functional lies in the measured span exactly
   when it is an explicit combination `c0 + sum_j c j g j` of normalisation and the measured
   observables.
2. *And the certificate computes the answer from the data.*  Every ensemble consistent with the
   measured averages `D` reports `c0 + sum_j c j D j` for that functional.  The identifiable
   content of a fit is a linear readout of the data: the fitted ensemble contributes nothing to
   it, and any two consistent ensembles agree.  A reader can recompute the number from the
   deposited data without ever downloading the model -- which is what "verifiable" should mean.
3. *Measuring what you report makes it verifiable.*  Taking the reported functionals themselves
   as the experiment list determines all of them, and every combination of them with
   normalisation: `r` experiments buy `r + 1` verifiable dimensions.
4. *And that is optimal.*  Any experiment set whatsoever making all `r` reports verifiable
   against an interior target satisfies `finrank (span {1, f 1, ..., f r}) <= k + 1`; so if the
   reports and normalisation are linearly independent, at least `r` experiments are needed.  The
   minimum is exactly `r`, and it does not depend on the size of the conformational library.

This is the constructive answer the negative parts were pointing at.  Determining the *ensemble*
of a disordered region costs `m - 1` experiments and is hopeless; determining `r` *reported
numbers* costs exactly `r` and is routine -- provided the experiment list spans what will be
reported, and provided nothing outside that span is reported.  A model of an intrinsically
disordered region can therefore be made fully verifiable at a stated resolution, not by
determining the ensemble, but by matching every claim to the measurements through the criterion of
Part LXX and quoting it with its certificate.
-/
import Mathlib
import RequestProject.ReportDesign

set_option autoImplicit false

namespace IDR

open IDR.Restraint IDR.Identify IDR.Report

/-- **The design and reporting laws.**

1. membership in the measured span is exactly representability as `c0 + sum_j c j g j`;
2. such a functional is a linear readout of the data, identical for every consistent ensemble;
3. measuring the reported functionals makes them, and their span with normalisation, verifiable;
4. and no experiment set with fewer than `finrank (span {1, f}) - 1` members can do so, giving a
   minimum of exactly `r` experiments for `r` independent reports. -/
theorem report_design_laws :
    (∀ (m k : ℕ) (g : Fin k → Fin m → ℝ) (f : Fin m → ℝ),
        f ∈ measured g ↔ ∃ (c0 : ℝ) (c : Fin k → ℝ), f = combo c0 c g) ∧
    ((∀ (m k : ℕ) (g : Fin k → Fin m → ℝ) (c0 : ℝ) (c D : Fin k → ℝ) (q : Fin m → ℝ),
        (∑ i, q i = 1) → (∀ j, obs (g j) q = D j) →
          obs (combo c0 c g) q = c0 + ∑ j, c j * D j) ∧
      (∀ (m k : ℕ) (g : Fin k → Fin m → ℝ) (q q' : Fin m → ℝ), (∑ i, q i = 1) → (∑ i, q' i = 1) →
        (∀ j, obs (g j) q = obs (g j) q') → ∀ f ∈ measured g, obs f q = obs f q')) ∧
    (∀ (m r : ℕ) (f : Fin r → Fin m → ℝ) (p : Fin m → ℝ), (∑ i, p i = 1) →
        (∀ i, Determined f p (f i)) ∧ ∀ h ∈ measured f, Determined f p h) ∧
    ((∀ (m k r : ℕ) (g : Fin k → Fin m → ℝ) (f : Fin r → Fin m → ℝ) (p : Fin m → ℝ) (d : ℝ),
        0 < d → (∀ i, d ≤ p i) → (∑ i, p i = 1) → (∀ i, Determined g p (f i)) →
          Module.finrank ℝ (measured f) ≤ k + 1) ∧
      (∀ (m k r : ℕ) (g : Fin k → Fin m → ℝ) (f : Fin r → Fin m → ℝ),
        LinearIndependent ℝ (obsFam f) → ∀ (p : Fin m → ℝ) (d : ℝ),
        0 < d → (∀ i, d ≤ p i) → (∑ i, p i = 1) → (∀ i, Determined g p (f i)) → r ≤ k)) := by
  refine ⟨fun m k g f => mem_measured_iff_combo g f,
    ⟨fun m k g c0 c D q hq1 hdata => report_value_from_data g c0 c D hq1 hdata,
      fun m k g q q' hq1 hq1' hdata f hf => consistent_ensembles_agree g hq1 hq1' hdata hf⟩,
    fun m r f p hp1 => report_sufficient f hp1,
    ⟨fun m k r g f p d hd hp hp1 hdet => report_min_experiments g f hd hp hp1 hdet,
      fun m k r g f hindep p d hd hp hp1 hdet => report_needs_r g f hindep hd hp hp1 hdet⟩⟩

end IDR
