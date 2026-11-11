/-
# Part LXXII  The design and reporting theorem: what to measure, and how to report it

Parts LXIX-LXXI are negative: the restraint count is short, the verifiable functionals form a
small subspace, the tolerance sets a resolution floor.  This part is the positive counterpart, and
it turns those limits into a design procedure that is optimal.

Start from the end of the pipeline.  A study reports `r` numbers about a disordered region: the
population of a bound-like conformer, a helical content, a contact frequency, a mean radius of
gyration -- linear functionals `f 1, ..., f r` of the ensemble.  The questions are: what must be
measured to make those numbers verifiable, and what is the fitted ensemble contributing to them?

* `mem_measured_iff_combo` -- first, a normal form.  A functional lies in the measured span
  exactly when it is `c0 + sum_j c j * g j` for explicit coefficients.  Identifiability always
  comes with a certificate.
* `report_value_from_data` -- and the certificate *computes the answer from the data*.  If
  `f = c0 + sum_j c j g j`, then **every** ensemble consistent with the measured averages
  `D 1, ..., D k` reports `obs f = c0 + sum_j c j D j`.  The identifiable content of a fit is a
  linear readout of the data; the fitted ensemble contributes nothing to it, and two consistent
  ensembles necessarily agree (`consistent_ensembles_agree`).  This is the strongest possible
  form of verifiability: the reported number can be recomputed from the deposited data by a
  reader who never downloads the model.
* `report_sufficient` -- second, the design half.  Measuring the functionals one intends to
  report makes all of them -- and every combination of them and normalisation -- verifiable.
  `r` experiments buy `r + 1` verifiable dimensions.
* `report_necessary`, `report_min_experiments` -- and this is optimal.  Any experiment set
  whatsoever that makes all `r` reports verifiable must satisfy
  `finrank (span {1, f 1, ..., f r}) <= k + 1`; so when the reports and normalisation are
  linearly independent, at least `r` experiments are needed, and measuring the reports directly
  attains the bound.  The minimum is exactly `r`.

The synthesis with Part LXIX is the practical point.  Determining the *ensemble* costs `m - 1`
experiments, exponentially many in the conformational entropy, and is hopeless.  Determining `r`
*reported numbers* costs exactly `r` experiments, and is routine -- provided the experiments are
chosen to span what will be reported, and provided nothing outside that span is reported.  The
impossibility of the first is not an obstacle to the second: a disordered region can be modelled
verifiably, at a stated resolution, as long as the claims are matched to the measurements by the
criterion of Part LXX rather than to the ensemble that interpolates them.
-/
import Mathlib
import RequestProject.Identifiability

set_option autoImplicit false

namespace IDR

namespace Report

open Finset IDR.Restraint IDR.Identify

variable {m k r : ℕ}

/-- The explicit combination of normalisation and the measured observables. -/
noncomputable def combo (c0 : ℝ) (c : Fin k → ℝ) (g : Fin k → Fin m → ℝ) : Fin m → ℝ :=
  fun i => c0 + ∑ j, c j * g j i

lemma combo_eq (c0 : ℝ) (c : Fin k → ℝ) (g : Fin k → Fin m → ℝ) :
    combo c0 c g = c0 • (fun _ => (1:ℝ)) + ∑ j, c j • g j := by
  funext i
  simp only [combo, Pi.add_apply, Pi.smul_apply, smul_eq_mul, mul_one, Finset.sum_apply]

lemma combo_mem (c0 : ℝ) (c : Fin k → ℝ) (g : Fin k → Fin m → ℝ) : combo c0 c g ∈ measured g := by
  rw [combo_eq]
  exact Submodule.add_mem _ (Submodule.smul_mem _ _ (measured_mem_const g))
    (Submodule.sum_mem _ fun j _ => Submodule.smul_mem _ _ (measured_mem_obs g j))

/-- **Identifiability comes with a certificate.**  A functional lies in the measured span exactly
when it is an explicit combination of normalisation and the measured observables. -/
theorem mem_measured_iff_combo (g : Fin k → Fin m → ℝ) (f : Fin m → ℝ) :
    f ∈ measured g ↔ ∃ (c0 : ℝ) (c : Fin k → ℝ), f = combo c0 c g := by
  constructor
  · intro hf
    obtain ⟨a, ha⟩ := (Submodule.mem_span_range_iff_exists_fun ℝ).1 hf
    refine ⟨a 0, fun j => a j.succ, ?_⟩
    rw [combo_eq, ← ha, Fin.sum_univ_succ]
    simp [obsFam]
  · rintro ⟨c0, c, rfl⟩
    exact combo_mem c0 c g

/-- **A verifiable number is a readout of the data.**  If `f = c0 + sum_j c j g j`, then every
ensemble reproducing the measured averages `D` reports the same value `c0 + sum_j c j D j` for
`f` -- computable from the deposited data alone, without the fitted ensemble. -/
theorem report_value_from_data (g : Fin k → Fin m → ℝ) (c0 : ℝ) (c : Fin k → ℝ)
    (D : Fin k → ℝ) {q : Fin m → ℝ} (hq1 : ∑ i, q i = 1) (hdata : ∀ j, obs (g j) q = D j) :
    obs (combo c0 c g) q = c0 + ∑ j, c j * D j := by
  have h1 : ∀ i ∈ (univ : Finset (Fin m)),
      combo c0 c g i * q i = c0 * q i + ∑ j, c j * (g j i * q i) := by
    intro i _
    simp only [combo, add_mul, Finset.sum_mul]
    congr 1
    exact Finset.sum_congr rfl fun j _ => by ring
  calc obs (combo c0 c g) q = ∑ i, (c0 * q i + ∑ j, c j * (g j i * q i)) :=
        Finset.sum_congr rfl h1
  _ = c0 * (∑ i, q i) + ∑ j, c j * ∑ i, g j i * q i := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_comm]
        congr 1
        exact Finset.sum_congr rfl fun j _ => by rw [Finset.mul_sum]
  _ = c0 + ∑ j, c j * D j := by
        rw [hq1, mul_one]
        congr 1
        exact Finset.sum_congr rfl fun j _ => by rw [show ∑ i, g j i * q i = obs (g j) q from rfl,
          hdata j]

/-- Two ensembles consistent with the same data agree on every verifiable functional. -/
theorem consistent_ensembles_agree (g : Fin k → Fin m → ℝ) {q q' : Fin m → ℝ}
    (hq1 : ∑ i, q i = 1) (hq1' : ∑ i, q' i = 1) (hdata : ∀ j, obs (g j) q = obs (g j) q')
    {f : Fin m → ℝ} (hf : f ∈ measured g) : obs f q = obs f q' := by
  obtain ⟨c0, c, rfl⟩ := (mem_measured_iff_combo g f).1 hf
  rw [report_value_from_data g c0 c (fun j => obs (g j) q') hq1 hdata,
    report_value_from_data g c0 c (fun j => obs (g j) q') hq1' (fun _ => rfl)]

/-! ## Optimal experimental design for a fixed report -/

/-- **Measuring what you report makes it verifiable.**  With the `r` reported functionals
themselves as the experiment list, every report -- and every combination of the reports with
normalisation -- is determined by the data. -/
theorem report_sufficient (f : Fin r → Fin m → ℝ) {p : Fin m → ℝ} (hp1 : ∑ i, p i = 1) :
    (∀ i, Determined f p (f i)) ∧ ∀ h ∈ measured f, Determined f p h :=
  ⟨fun i => determined_of_mem f hp1 (measured_mem_obs f i),
    fun _ hh => determined_of_mem f hp1 hh⟩

/-- **And no experiment set can do it with fewer dimensions.**  If an experiment list `g` makes
every reported functional verifiable against an interior target, then the report span is contained
in the measured span. -/
theorem report_necessary (g : Fin k → Fin m → ℝ) (f : Fin r → Fin m → ℝ)
    {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1)
    (hdet : ∀ i, Determined g p (f i)) : measured f ≤ measured g := by
  refine Submodule.span_le.2 ?_
  rintro x ⟨j, rfl⟩
  refine Fin.cases ?_ ?_ j
  · simpa [obsFam] using measured_mem_const g
  · intro i
    have : f i ∈ measured g := (determined_iff_mem g hd hp hp1 (f i)).1 (hdet i)
    simpa [obsFam] using this

/-- **The exact cost of a report.**  Any experiment set making all `r` reported functionals
verifiable has `finrank (span {1, f 1, ..., f r}) ≤ k + 1`; in particular, if the reports together
with normalisation are linearly independent, at least `r` experiments are needed -- and measuring
the reports themselves attains it.  The minimum number of experiments needed to verify `r`
independent reported numbers is exactly `r`, independently of the size of the conformational
library. -/
theorem report_min_experiments (g : Fin k → Fin m → ℝ) (f : Fin r → Fin m → ℝ)
    {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1)
    (hdet : ∀ i, Determined g p (f i)) :
    Module.finrank ℝ (measured f) ≤ k + 1 := by
  have hle := report_necessary g f hd hp hp1 hdet
  exact le_trans (Submodule.finrank_mono hle) (identifiable_dim g)

/-- With linearly independent reports, the count is exactly `r`: fewer than `r` experiments cannot
make them all verifiable. -/
theorem report_needs_r (g : Fin k → Fin m → ℝ) (f : Fin r → Fin m → ℝ)
    (hindep : LinearIndependent ℝ (obsFam f))
    {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1)
    (hdet : ∀ i, Determined g p (f i)) : r ≤ k := by
  have hdim : Module.finrank ℝ (measured f) = r + 1 := by
    rw [measured, finrank_span_eq_card hindep]
    simp
  have := report_min_experiments g f hd hp hp1 hdet
  rw [hdim] at this
  omega

end Report

end IDR
