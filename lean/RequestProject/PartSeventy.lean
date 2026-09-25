/-
# Part LXX  What is verifiable in a reported ensemble

Part LXIX counted the restraints an ensemble costs and found the count short in almost every
application.  This part answers the question that then becomes unavoidable: of the numbers
extracted from a fitted ensemble of a disordered region, *which are consequences of the data?*

`RequestProject.Identifiability` settles it exactly, on a finite conformational library, for an
interior target -- every conformation populated, which is what disorder means -- and for the
linear functionals of the ensemble, which is what a reported number is: a substate population, a
secondary-structure content, a contact frequency, a mean radius of gyration.

`IDR.identifiability_laws` bundles five statements:

1. *Averaging is linear in the observable*, so the identifiable functionals form a subspace.
2. *Everything in the measured span is verifiable.*  If `f` is a linear combination of
   normalisation and the measured observables, every ensemble matching the data reports the same
   average for `f`.  Normalisation counts as a restraint; it is why the constant is included.
3. *Nothing else is.*  If `f` lies outside that span, there is a genuine ensemble -- nonnegative,
   normalised -- reproducing every measurement and reporting a different value of `f`.  Because
   the target is interior, the discrepancy is not infinitesimal: the reported value moves by a
   finite amount at no cost in fit.
4. *Hence the criterion*: against an interior target, `f` is determined by the data **iff**
   `f ∈ span {1, g 1, ..., g k}`.  Identifiability is membership in a subspace fixed by the
   experiment list alone -- not a matter of regularisation, of prior width, or of goodness of fit.
5. *And the budget*: that subspace has dimension at most `k + 1`, so `k` experiments support at
   most `k + 1` independent verifiable numbers; and if `k + 1 < m`, at least one
   single-conformation population is not among them.

Together with Part LXIX this gives a complete and checkable protocol for reporting an ensemble
model of a disordered region.  Publish the library size `m` and the observable list; quote as a
result only a functional exhibited as a combination of the constant and the measured observables,
and give the combination.  A number so exhibited is a theorem about the experiment.  Any other
number -- however stable across fits, however tight its error bar from bootstrap resampling -- is
a property of the reference ensemble, which is exactly what the maximum-entropy analysis of
Part III predicts and what the counting law of Part LXIX quantifies.
-/
import Mathlib
import RequestProject.Identifiability

set_option autoImplicit false

namespace IDR

open IDR.Restraint IDR.Identify

/-- **The identifiability laws.**

1. ensemble averaging is linear in the observable;
2. every functional in the span of normalisation and the measured observables is determined by
   the data;
3. against an interior target, no other functional is;
4. hence determinacy is exactly membership in the measured span;
5. that span has dimension at most `k + 1`, and below the restraint threshold some
   single-conformation population is not determined. -/
theorem identifiability_laws :
    (∀ (m : ℕ) (f h q : Fin m → ℝ) (c : ℝ),
        obs (f + h) q = obs f q + obs h q ∧ obs (c • f) q = c * obs f q) ∧
    (∀ (m k : ℕ) (g : Fin k → Fin m → ℝ) (p : Fin m → ℝ), (∑ i, p i = 1) →
        ∀ f : Fin m → ℝ, f ∈ measured g → Determined g p f) ∧
    (∀ (m k : ℕ) (g : Fin k → Fin m → ℝ) (p : Fin m → ℝ) (d : ℝ), 0 < d → (∀ i, d ≤ p i) →
        (∑ i, p i = 1) → ∀ f : Fin m → ℝ, f ∉ measured g → ¬ Determined g p f) ∧
    (∀ (m k : ℕ) (g : Fin k → Fin m → ℝ) (p : Fin m → ℝ) (d : ℝ), 0 < d → (∀ i, d ≤ p i) →
        (∑ i, p i = 1) → ∀ f : Fin m → ℝ, (Determined g p f ↔ f ∈ measured g)) ∧
    ((∀ (m k : ℕ) (g : Fin k → Fin m → ℝ), Module.finrank ℝ (measured g) ≤ k + 1) ∧
      (∀ (m k : ℕ), k + 1 < m → ∀ (g : Fin k → Fin m → ℝ) (p : Fin m → ℝ) (d : ℝ), 0 < d →
        (∀ i, d ≤ p i) → (∑ i, p i = 1) →
          ∃ i : Fin m, ¬ Determined g p (Pi.single i (1:ℝ)))) := by
  refine ⟨fun m f h q c => ⟨obs_add f h q, obs_smul c f q⟩,
    fun m k g p hp1 f hf => determined_of_mem g hp1 hf,
    fun m k g p d hd hp hp1 f hf => not_determined_of_notMem g hd hp hp1 hf,
    fun m k g p d hd hp hp1 f => determined_iff_mem g hd hp hp1 f,
    ⟨fun m k g => identifiable_dim g,
      fun m k hk g p d hd hp hp1 => exists_population_not_determined hk g hd hp hp1⟩⟩

end IDR
