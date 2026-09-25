/-
# Part LXX  What is verifiable: the identifiable functionals are exactly the measured span

Part LXIX counted how many restraints an ensemble costs, and found the count almost always short.
That leaves the question a careful report must answer anyway: *given the restraints one does have,
which reported numbers are consequences of the data, and which are consequences of the prior?*

This file answers it exactly, and the answer is clean.  Fix a conformational library of size `m`,
`k` measured observables `g 1, ..., g k`, and an interior target `p` (every conformation populated
with weight at least `d > 0`).  Call a linear functional `f` -- a secondary structure content, a
contact frequency, a population of a substate, a radius of gyration -- *determined* if every
ensemble matching the data has the same average of `f` as the truth.  Then:

* `determined_of_mem` -- every `f` in `measured g`, the span of the constant function together
  with the measured observables, is determined.  (Normalisation is a restraint too, and it is the
  reason the constant appears.)
* `not_determined_of_notMem` -- and *nothing else is*.  If `f` lies outside that span, then some
  genuine ensemble reproduces every measurement and reports a different value of `f`.
* `determined_iff_mem` -- so, against an interior target, `f` is verifiable **iff**
  `f ∈ span {1, g 1, ..., g k}`.  Identifiability is not a matter of degree, of regularisation, or
  of how good the fit is: it is membership in a subspace fixed by the experiment list alone.
* `identifiable_dim` -- and that subspace has dimension at most `k + 1`.  An experiment set of
  size `k` supports at most `k + 1` independent verifiable numbers, whatever the library, whatever
  the sampling, and whatever the fitting method.  Everything else reported is prior.

The construction behind the negative half is the dual one: an `f` outside the span is separated
from it by a linear functional, and a linear functional on population space *is* a signed
population direction; being null on the span it is invisible to the data, and being interior the
target can be moved along it (`RequestProject.Restraints.exists_perturbation`) without leaving the
simplex.  So the failure of identifiability is never marginal -- the reported value of a
non-identifiable `f` can be moved by a finite amount at no cost in fit.

The practical protocol this proves correct: publish `(m, k)`, publish the observable list, and
quote as results only functionals exhibited as combinations of the constant and the measured
observables -- with the combination given.  A number so exhibited is a theorem about the data.  A
number not so exhibited is a property of the reference ensemble, and Part III already showed the
reference is where such numbers come from.
-/
import Mathlib
import RequestProject.Restraints

set_option autoImplicit false

namespace IDR

namespace Identify

open Finset IDR.Restraint

variable {m k : ℕ}

/-- The observables the experiment actually reports: the constant function -- normalisation, which
is always known -- followed by the `k` measured observables. -/
noncomputable def obsFam (g : Fin k → Fin m → ℝ) : Fin (k + 1) → (Fin m → ℝ) :=
  Fin.cons (fun _ => 1) g

/-- The measured span: all linear combinations of normalisation and the measured observables. -/
noncomputable def measured (g : Fin k → Fin m → ℝ) : Submodule ℝ (Fin m → ℝ) :=
  Submodule.span ℝ (Set.range (obsFam g))

/-- A functional is *determined* by the data at the target `p` when every ensemble reproducing the
measured averages reports the same average for it. -/
def Determined (g : Fin k → Fin m → ℝ) (p f : Fin m → ℝ) : Prop :=
  ∀ q : Fin m → ℝ, IsEns q → (∀ j, obs (g j) q = obs (g j) p) → obs f q = obs f p

/-! ## Averaging is linear in the observable -/

lemma obs_zero (q : Fin m → ℝ) : obs 0 q = 0 := by simp [obs]

lemma obs_add (f h q : Fin m → ℝ) : obs (f + h) q = obs f q + obs h q := by
  simp only [obs, Pi.add_apply, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

lemma obs_smul (c : ℝ) (f q : Fin m → ℝ) : obs (c • f) q = c * obs f q := by
  simp only [obs, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => by ring

lemma obs_one (q : Fin m → ℝ) : obs (fun _ => 1) q = ∑ i, q i := by
  simp [obs]

lemma measured_mem_const (g : Fin k → Fin m → ℝ) : (fun _ => (1:ℝ)) ∈ measured g :=
  Submodule.subset_span ⟨0, by simp [obsFam]⟩

lemma measured_mem_obs (g : Fin k → Fin m → ℝ) (j : Fin k) : g j ∈ measured g :=
  Submodule.subset_span ⟨j.succ, by simp [obsFam]⟩

/-! ## The positive half -/

/-- **Everything in the measured span is verifiable.**  If `f` is a linear combination of
normalisation and the measured observables, then every ensemble matching the data reports the same
average of `f`. -/
theorem determined_of_mem (g : Fin k → Fin m → ℝ) {p : Fin m → ℝ} (hp1 : ∑ i, p i = 1)
    {f : Fin m → ℝ} (hf : f ∈ measured g) : Determined g p f := by
  intro q hq hdata
  induction hf using Submodule.span_induction with
  | mem x hx =>
      obtain ⟨j, rfl⟩ := hx
      refine Fin.cases ?_ ?_ j
      · simpa [obsFam, obs_one] using hq.2.trans hp1.symm
      · intro i
        simpa [obsFam] using hdata i
  | zero => simp [obs_zero]
  | add x y _ _ ihx ihy => rw [obs_add, obs_add, ihx, ihy]
  | smul c x _ ih => rw [obs_smul, obs_smul, ih]

/-! ## The negative half -/

/-- A linear functional on population space is a signed population direction. -/
lemma exists_vector_of_dual (phi : (Fin m → ℝ) →ₗ[ℝ] ℝ) :
    ∃ v : Fin m → ℝ, ∀ x : Fin m → ℝ, phi x = obs x v := by
  refine ⟨fun i => phi (Pi.single i 1), fun x => ?_⟩
  have hx : x = ∑ i, Pi.single i (x i) := (Finset.univ_sum_single x).symm
  have hsingle : ∀ (i : Fin m) (a : ℝ),
      (Pi.single i a : Fin m → ℝ) = a • (Pi.single i (1:ℝ) : Fin m → ℝ) := by
    intro i a; funext j; by_cases h : j = i <;> simp [Pi.single_apply, h]
  calc phi x = phi (∑ i, Pi.single i (x i)) := by rw [← hx]
  _ = ∑ i, phi (Pi.single i (x i)) := by rw [map_sum]
  _ = ∑ i, x i * phi (Pi.single i 1) := by
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [hsingle i (x i), map_smul, smul_eq_mul]
  _ = obs x (fun i => phi (Pi.single i 1)) := rfl

/-- **Nothing outside the measured span is verifiable.**  Against an interior target, a functional
outside `span {1, g 1, ..., g k}` takes a different value on some genuine ensemble reproducing
every measurement. -/
theorem not_determined_of_notMem (g : Fin k → Fin m → ℝ)
    {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1)
    {f : Fin m → ℝ} (hf : f ∉ measured g) : ¬ Determined g p f := by
  obtain ⟨phi, hphif, hker⟩ := Submodule.exists_le_ker_of_notMem hf
  obtain ⟨v, hv⟩ := exists_vector_of_dual phi
  have hvf : obs f v ≠ 0 := by rw [← hv]; exact hphif
  have hv0 : v ≠ 0 := by
    intro h
    apply hvf
    simp [obs, h]
  have hvsum : ∑ i, v i = 0 := by
    have : phi (fun _ => (1:ℝ)) = 0 := hker (measured_mem_const g)
    rw [hv] at this
    simpa [obs] using this
  have hvg : ∀ j, obs (g j) v = 0 := by
    intro j
    have : phi (g j) = 0 := hker (measured_mem_obs g j)
    rw [hv] at this
    exact this
  obtain ⟨c, hc, hens, -⟩ := exists_perturbation hd hp hp1 hv0 hvsum
  intro hdet
  have hdata : ∀ j, obs (g j) (fun i => p i + c * v i) = obs (g j) p := by
    intro j; rw [obs_add_smul, hvg j, mul_zero, add_zero]
  have := hdet _ hens hdata
  rw [obs_add_smul] at this
  have hzero : c * obs f v = 0 := by linarith
  rcases mul_eq_zero.1 hzero with h | h
  · exact absurd h hc.ne'
  · exact hvf h

/-- **The identifiability criterion.**  Against an interior target, a linear functional of the
ensemble is determined by the data **iff** it lies in the span of normalisation and the measured
observables. -/
theorem determined_iff_mem (g : Fin k → Fin m → ℝ)
    {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1)
    (f : Fin m → ℝ) : Determined g p f ↔ f ∈ measured g := by
  refine ⟨fun hdet => ?_, fun hmem => determined_of_mem g hp1 hmem⟩
  by_contra hf
  exact not_determined_of_notMem g hd hp hp1 hf hdet

/-- **`k` experiments support at most `k + 1` independent verifiable numbers.**  The space of
identifiable functionals has dimension at most `k + 1`, whatever the library size, the sampling
protocol or the fitting method. -/
theorem identifiable_dim (g : Fin k → Fin m → ℝ) :
    Module.finrank ℝ (measured g) ≤ k + 1 := by
  have h := finrank_range_le_card (R := ℝ) (obsFam g)
  simpa [measured] using h

/-- The average of the indicator of a conformation is its population. -/
lemma obs_single (i : Fin m) (q : Fin m → ℝ) : obs (Pi.single i (1:ℝ)) q = q i := by
  simp [obs, Pi.single_apply, Finset.sum_ite_eq']

/-- **Below threshold, some reported population is not a consequence of the data.**  If
`k + 1 < m`, at least one single-conformation population fails to be determined: no amount of
care in the fit makes that number a theorem about the experiment. -/
theorem exists_population_not_determined (hk : k + 1 < m) (g : Fin k → Fin m → ℝ)
    {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1) :
    ∃ i : Fin m, ¬ Determined g p (Pi.single i (1:ℝ)) := by
  by_contra hcon
  push_neg at hcon
  have hmem : ∀ i : Fin m, (Pi.single i (1:ℝ) : Fin m → ℝ) ∈ measured g := fun i =>
    (determined_iff_mem g hd hp hp1 _).1 (hcon i)
  have htop : (⊤ : Submodule ℝ (Fin m → ℝ)) ≤ measured g := by
    rw [← (Pi.basisFun ℝ (Fin m)).span_eq]
    refine Submodule.span_le.2 ?_
    rintro x ⟨i, rfl⟩
    simpa [Pi.basisFun_apply] using hmem i
  have heq : measured g = ⊤ := top_le_iff.1 htop
  have hdim : Module.finrank ℝ (measured g) = m := by
    rw [heq, finrank_top]
    simp
  have := identifiable_dim g
  rw [hdim] at this
  omega

end Identify

end IDR
