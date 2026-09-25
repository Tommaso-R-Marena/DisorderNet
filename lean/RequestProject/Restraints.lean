/-
# Part LXIX  How many experiments does an ensemble cost?

Every ensemble description of a disordered region is fitted against *restraints*: a set of
experimental averages -- SAXS intensities at a list of scattering angles, PRE rates, RDCs,
chemical shifts, FRET efficiencies, each of them the ensemble average of a known back-calculated
observable.  The fitted object is a population vector `p` over a conformational library of size
`m`; the data are `k` numbers `⟨g j, p⟩`.

The question this file answers exactly is the one a designer must answer before choosing the
library: *how large may `k` be allowed to be, relative to `m`, for the data to pin the ensemble
down at all?*  The answer is a counting law, and it is brutal.

* `exists_null_direction` -- `k` observables plus normalisation are `k + 1` linear functionals on
  an `m`-dimensional space, so as soon as `k + 1 < m` they annihilate a nonzero direction.
* `restraints_insufficient` -- the design statement.  If every conformation of the target carries
  population at least `d` (an interior ensemble: the generic situation for a disordered region,
  where nothing is excluded), and `k + 1 < m`, then there is a *bona fide* ensemble `q` --
  nonnegative, normalised -- reproducing **every** one of the `k` measured averages exactly, at
  population distance `‖q - p‖₁ ≥ 2 d` from the truth.  Not a near-copy: on a uniform target the
  guaranteed separation is `2/m`, exactly the weight of two whole conformations.
* `uniform_restraints_insufficient` -- the uniform instance, with `d = 1/m`.
* `experiments_needed` -- the contrapositive, and the number to quote: an experiment set that
  determines an interior target to population accuracy better than `2 d` must contain at least
  `m - 1` independent restraints.  With `d = 1/m`: at least `m - 1` restraints for a library of
  `m` conformations, i.e. one restraint per conformation.
* `restraints_exp_entropy` -- and since a library realising conformational entropy `H` has
  `m = exp H` members (Part III), the restraint count needed grows *exponentially* in the
  conformational entropy.  Tens of restraints against a library of thousands is not
  under-determination at the margin; it is under-determination in almost every direction.
* `agree_except_one`, `indicator_restraints_determine` -- the bound is sharp: `m - 1` restraints
  do suffice, namely the populations of all but one conformation.  So `m - 1` is exactly the
  threshold, not an artefact of the argument.
* `feasible_convex`, `feasible_antitone` -- the geometry behind the law: the set of ensembles
  consistent with a data set is convex, and shrinks as restraints are added.  There is no
  "lucky" restraint set below the threshold, because the deficient direction is a *subspace*.

The reading for practice: the restraint count does not fix the ensemble; it fixes an affine
slice of the simplex whose width is set by the missing dimensions.  Everything reported inside
that slice -- and, by Part III's maximum-entropy analysis, the point picked inside it -- comes
from the prior, not from the experiment.  Reporting an IDR ensemble therefore requires reporting
`m` and `k` together, and quoting only functionals constant on the slice.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace Restraint

open Finset

variable {m k : ℕ}

/-- A population vector over a conformational library of size `m`. -/
def IsEns (p : Fin m → ℝ) : Prop := (∀ i, 0 ≤ p i) ∧ ∑ i, p i = 1

/-- The ensemble average of a back-calculated observable. -/
def obs (g p : Fin m → ℝ) : ℝ := ∑ i, g i * p i

/-- Population distance (ℓ¹, twice total variation). -/
def ell1 (p q : Fin m → ℝ) : ℝ := ∑ i, |p i - q i|

/-- The uniform ensemble on the library. -/
noncomputable def unif (m : ℕ) : Fin m → ℝ := fun _ => (m : ℝ)⁻¹

lemma isEns_unif (hm : 0 < m) : IsEns (unif m) := by
  refine ⟨fun _ => by unfold unif; positivity, ?_⟩
  have : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hm.ne'
  simp [unif, Finset.sum_const, Finset.card_univ]
  field_simp

/-! ## The counting argument -/

/-- The linear map recording normalisation together with the `k` measured averages. -/
noncomputable def restMap (g : Fin k → Fin m → ℝ) :
    (Fin m → ℝ) →ₗ[ℝ] (ℝ × (Fin k → ℝ)) where
  toFun v := (∑ i, v i, fun j => ∑ i, g j i * v i)
  map_add' v w := by
    refine Prod.ext ?_ ?_
    · simp [Finset.sum_add_distrib]
    · funext j; simp [mul_add, Finset.sum_add_distrib]
  map_smul' c v := by
    refine Prod.ext ?_ ?_
    · simp [Finset.mul_sum]
    · funext j
      simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply, Prod.smul_snd, Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => by ring

/-- **Fewer restraints than conformations leave a blind direction.**  `k` observables together
with normalisation are `k + 1` linear conditions on an `m`-dimensional population space, so when
`k + 1 < m` some nonzero signed population change is invisible to all of them. -/
theorem exists_null_direction (hk : k + 1 < m) (g : Fin k → Fin m → ℝ) :
    ∃ v : Fin m → ℝ, v ≠ 0 ∧ (∑ i, v i = 0) ∧ ∀ j, ∑ i, g j i * v i = 0 := by
  have hlt : Module.finrank ℝ (ℝ × (Fin k → ℝ)) < Module.finrank ℝ (Fin m → ℝ) := by
    simp [Module.finrank_prod]
    omega
  obtain ⟨v, hv, hv0⟩ :=
    Submodule.exists_mem_ne_zero_of_ne_bot (LinearMap.ker_ne_bot_of_finrank_lt (f := restMap g) hlt)
  have hker := (LinearMap.mem_ker).1 hv
  refine ⟨v, hv0, ?_, fun j => ?_⟩
  · simpa [restMap] using congrArg Prod.fst hker
  · simpa [restMap] using congrArg (fun z => z.2 j) hker

/-- A signed vector summing to zero has ℓ¹ norm at least twice any of its entries. -/
lemma two_mul_abs_le_sum_abs (v : Fin m → ℝ) (h0 : ∑ i, v i = 0) (i0 : Fin m) :
    2 * |v i0| ≤ ∑ i, |v i| := by
  have hsplit : |v i0| + ∑ i ∈ univ.erase i0, |v i| = ∑ i, |v i| :=
    Finset.add_sum_erase univ (fun i => |v i|) (mem_univ i0)
  have hrest : ∑ i ∈ univ.erase i0, v i = -v i0 := by
    have := Finset.add_sum_erase (univ : Finset (Fin m)) (fun i => v i) (mem_univ i0)
    linarith [this, h0]
  have : |v i0| ≤ ∑ i ∈ univ.erase i0, |v i| := by
    calc |v i0| = |∑ i ∈ univ.erase i0, v i| := by rw [hrest, abs_neg]
    _ ≤ ∑ i ∈ univ.erase i0, |v i| := Finset.abs_sum_le_sum_abs _ _
  linarith [hsplit, this]

/-! ## The perturbation construction -/

/-- Averaging is affine in the ensemble. -/
lemma obs_add_smul (w p v : Fin m → ℝ) (c : ℝ) :
    obs w (fun i => p i + c * v i) = obs w p + c * obs w v := by
  simp only [obs, Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- **An interior ensemble can be moved macroscopically along any null direction.**  If the target
populates every conformation with weight at least `d > 0`, then along any nonzero direction `v`
summing to zero there is a scale `c > 0` for which `p + c v` is still a genuine ensemble and sits
at population distance at least `2 d` from `p`.  This is the engine behind every
under-determination statement below: the size of the move is set by how far the target is from
the boundary of the simplex, not by the direction. -/
theorem exists_perturbation {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d) (hp : ∀ i, d ≤ p i)
    (hp1 : ∑ i, p i = 1) {v : Fin m → ℝ} (hv0 : v ≠ 0) (hvsum : ∑ i, v i = 0) :
    ∃ c : ℝ, 0 < c ∧ IsEns (fun i => p i + c * v i) ∧
      2 * d ≤ ell1 (fun i => p i + c * v i) p := by
  obtain ⟨j0, hj0⟩ : ∃ j, v j ≠ 0 := Function.ne_iff.1 hv0
  have hne : (univ : Finset (Fin m)).Nonempty := ⟨j0, mem_univ _⟩
  obtain ⟨i0, -, hi0⟩ := Finset.exists_max_image univ (fun i => |v i|) hne
  have hM : 0 < |v i0| := lt_of_lt_of_le (abs_pos.2 hj0) (hi0 j0 (mem_univ j0))
  set c : ℝ := d / |v i0| with hc
  have hcpos : 0 < c := div_pos hd hM
  have hbound : ∀ i, |c * v i| ≤ d := by
    intro i
    rw [abs_mul, abs_of_pos hcpos, hc, div_mul_eq_mul_div, div_le_iff₀ hM]
    exact (mul_le_mul_of_nonneg_left (hi0 i (mem_univ i)) hd.le)
  refine ⟨c, hcpos, ⟨fun i => ?_, ?_⟩, ?_⟩
  · have h1 : -d ≤ c * v i := (abs_le.1 (hbound i)).1
    have := hp i
    linarith
  · rw [Finset.sum_add_distrib, hp1, ← Finset.mul_sum, hvsum, mul_zero, add_zero]
  · have hell : ell1 (fun i => p i + c * v i) p = c * ∑ i, |v i| := by
      simp only [ell1, add_sub_cancel_left, Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => by rw [abs_mul, abs_of_pos hcpos]
    rw [hell]
    have hstep : c * (2 * |v i0|) ≤ c * ∑ i, |v i| :=
      mul_le_mul_of_nonneg_left (two_mul_abs_le_sum_abs v hvsum i0) hcpos.le
    have hcv0 : c * |v i0| = d := by rw [hc]; field_simp
    have hcv : c * (2 * |v i0|) = 2 * d := by rw [← hcv0]; ring
    linarith

/-! ## The restraint-counting law -/

/-- **Restraint counting.**  Let the target `p` be an interior ensemble, every conformation of the
library carrying population at least `d > 0`.  If the number `k` of measured averages satisfies
`k + 1 < m`, then there is a genuine ensemble `q` -- nonnegative and normalised -- which
reproduces every measured average exactly and yet differs from the truth by at least `2 d` in
population distance.  Below the threshold the data do not merely leave the ensemble noisy; they
leave whole conformations' worth of population free. -/
theorem restraints_insufficient (hk : k + 1 < m) (g : Fin k → Fin m → ℝ)
    {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1) :
    ∃ q : Fin m → ℝ, IsEns q ∧ (∀ j, obs (g j) q = obs (g j) p) ∧ 2 * d ≤ ell1 q p := by
  obtain ⟨v, hv0, hvsum, hvg⟩ := exists_null_direction hk g
  obtain ⟨c, -, hens, hfar⟩ := exists_perturbation hd hp hp1 hv0 hvsum
  refine ⟨_, hens, fun j => ?_, hfar⟩
  rw [obs_add_smul]
  simp [obs, hvg j]

/-- The uniform instance: with fewer than `m - 1` restraints, an ensemble at population distance
`2/m` from uniform -- the weight of two whole conformations -- reproduces every measurement. -/
theorem uniform_restraints_insufficient (hk : k + 1 < m) (g : Fin k → Fin m → ℝ) :
    ∃ q : Fin m → ℝ, IsEns q ∧ (∀ j, obs (g j) q = obs (g j) (unif m)) ∧
      2 / (m : ℝ) ≤ ell1 q (unif m) := by
  have hm : 0 < m := by omega
  have hmR : (0:ℝ) < m := by exact_mod_cast hm
  obtain ⟨q, hq, hdata, hfar⟩ :=
    restraints_insufficient hk g (p := unif m) (d := (m:ℝ)⁻¹) (by positivity)
      (fun _ => le_refl _) (isEns_unif hm).2
  refine ⟨q, hq, hdata, ?_⟩
  rw [div_eq_mul_inv]
  exact hfar

/-- **The number of experiments an ensemble costs.**  If a restraint set determines an interior
target to population accuracy better than `2 d`, it must contain at least `m - 1` restraints. -/
theorem experiments_needed (g : Fin k → Fin m → ℝ)
    {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1)
    (hdet : ∀ q : Fin m → ℝ, IsEns q → (∀ j, obs (g j) q = obs (g j) p) → ell1 q p < 2 * d) :
    m ≤ k + 1 := by
  by_contra h
  push_neg at h
  obtain ⟨q, hq, hdata, hfar⟩ := restraints_insufficient h g hd hp hp1
  exact absurd (hdet q hq hdata) (not_lt.2 hfar)

/-- **The restraint count is exponential in the conformational entropy.**  A library realising
conformational entropy `H = log m` has `m = exp H` members, so a restraint set that determines the
uniform ensemble at all carries at least `exp H - 1` restraints. -/
theorem restraints_exp_entropy (hm : 0 < m) (g : Fin k → Fin m → ℝ)
    (hdet : ∀ q : Fin m → ℝ, IsEns q → (∀ j, obs (g j) q = obs (g j) (unif m)) →
      ell1 q (unif m) < 2 / (m:ℝ)) :
    Real.exp (Real.log m) ≤ (k : ℝ) + 1 := by
  have hmR : (0:ℝ) < m := by exact_mod_cast hm
  have hstep : m ≤ k + 1 := by
    refine experiments_needed g (p := unif m) (d := (m:ℝ)⁻¹) (by positivity)
      (fun _ => le_refl _) (isEns_unif hm).2 ?_
    intro q hq hdata
    have h2 : ell1 q (unif m) < 2 / (m:ℝ) := hdet q hq hdata
    rw [← div_eq_mul_inv]
    exact h2
  rw [Real.exp_log hmR]
  exact_mod_cast hstep

/-- **Cross-validation cannot certify an ensemble below threshold.**  Split the experiments into
`k` used for fitting and `l` held out for validation.  If `k + l + 1 < m` there is a genuine
ensemble reproducing the fitting data *and* the held-out data exactly, while sitting at population
distance at least `2 d` from the truth.  Agreement with held-out restraints is therefore evidence
about the ensemble only in so far as the *total* restraint count crosses the threshold: a
validation set does not buy accuracy that the counting law forbids. -/
theorem cross_validation_cannot_certify {l : ℕ} (hk : k + l + 1 < m)
    (g : Fin k → Fin m → ℝ) (h : Fin l → Fin m → ℝ)
    {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1) :
    ∃ q : Fin m → ℝ, IsEns q ∧ (∀ j, obs (g j) q = obs (g j) p) ∧
      (∀ j, obs (h j) q = obs (h j) p) ∧ 2 * d ≤ ell1 q p := by
  obtain ⟨q, hq, hdata, hfar⟩ :=
    restraints_insufficient (k := k + l) hk (Fin.append g h) hd hp hp1
  refine ⟨q, hq, fun j => ?_, fun j => ?_, hfar⟩
  · have := hdata (Fin.castAdd l j)
    simpa [Fin.append_left] using this
  · have := hdata (Fin.natAdd k j)
    simpa [Fin.append_right] using this

/-! ## Sharpness: `m - 1` restraints do suffice -/

/-- Two normalised vectors agreeing off a single index agree everywhere. -/
theorem agree_except_one {p q : Fin m → ℝ} (hp : ∑ i, p i = 1) (hq : ∑ i, q i = 1)
    (i0 : Fin m) (h : ∀ i, i ≠ i0 → p i = q i) : p = q := by
  have hsp : p i0 + ∑ i ∈ univ.erase i0, p i = 1 := by
    rw [Finset.add_sum_erase univ (fun i => p i) (mem_univ i0)]; exact hp
  have hsq : q i0 + ∑ i ∈ univ.erase i0, q i = 1 := by
    rw [Finset.add_sum_erase univ (fun i => q i) (mem_univ i0)]; exact hq
  have hrest : ∑ i ∈ univ.erase i0, p i = ∑ i ∈ univ.erase i0, q i :=
    Finset.sum_congr rfl fun i hi => h i (Finset.ne_of_mem_erase hi)
  have h0 : p i0 = q i0 := by rw [hrest] at hsp; linarith
  funext i
  by_cases hi : i = i0
  · rw [hi]; exact h0
  · exact h i hi

/-- **The threshold is exactly `m - 1`.**  The `m - 1` indicator observables of the conformations
other than `i0` -- i.e. reading off all but one population -- determine the ensemble. -/
theorem indicator_restraints_determine {p q : Fin m → ℝ} (hp : ∑ i, p i = 1) (hq : ∑ i, q i = 1)
    (i0 : Fin m)
    (h : ∀ i, i ≠ i0 → obs (fun x => if x = i then 1 else 0) p
      = obs (fun x => if x = i then 1 else 0) q) : p = q := by
  refine agree_except_one hp hq i0 fun i hi => ?_
  have := h i hi
  simpa [obs] using this

/-! ## The geometry behind the law -/

/-- The set of ensembles consistent with a data set is convex: mixing two fits is a fit. -/
theorem feasible_convex (g : Fin k → Fin m → ℝ) (data : Fin k → ℝ)
    {q r : Fin m → ℝ} (hq : IsEns q) (hr : IsEns r)
    (hqd : ∀ j, obs (g j) q = data j) (hrd : ∀ j, obs (g j) r = data j)
    {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    IsEns (fun i => t * q i + (1 - t) * r i) ∧
      ∀ j, obs (g j) (fun i => t * q i + (1 - t) * r i) = data j := by
  refine ⟨⟨fun i => by nlinarith [hq.1 i, hr.1 i], ?_⟩, fun j => ?_⟩
  · rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hq.2, hr.2]; ring
  · have : ∑ i, g j i * (t * q i + (1 - t) * r i)
        = t * (∑ i, g j i * q i) + (1 - t) * (∑ i, g j i * r i) := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    simp only [obs] at hqd hrd ⊢
    rw [this, hqd j, hrd j]; ring

/-- Adding restraints can only shrink the feasible set: any ensemble fitting the enlarged data set
fits the original one. -/
theorem feasible_antitone (g : Fin k → Fin m → ℝ) (h : Fin 1 → Fin m → ℝ)
    (q : Fin m → ℝ) (data : Fin k → ℝ) (extra : Fin 1 → ℝ)
    (hfit : ∀ j : Fin (k + 1),
      obs (Fin.append g h j) q = Fin.append data extra j) :
    ∀ j, obs (g j) q = data j := by
  intro j
  have := hfit (Fin.castAdd 1 j)
  simpa [Fin.append_left] using this

end Restraint

end IDR
