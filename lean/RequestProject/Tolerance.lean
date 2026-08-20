/-
# Part LXXI  The precision floor: what finite experimental accuracy costs, at any restraint count

Part LXIX counted restraints assuming the data are matched *exactly*.  Real restraints are matched
to a tolerance: a chi-squared target, an error bar, a statistical uncertainty from finitely many
frames or finitely many photons.  This part asks what that tolerance costs, and the answer is a
law that no amount of extra experiment repairs.

Fix a library of size `m`, observables `g 1, ..., g k` with entries bounded by `G` -- every
back-calculated observable is bounded, since a finite library has a largest and a smallest value --
and an interior target `p`, every conformation carrying weight at least `d`.  Let the data be
reproduced only to tolerance `eps`.

* `exists_pair_perturbation` -- the construction: moving weight `c` from conformation `b` to
  conformation `a` changes the `j`-th measured average by exactly `c (g j a - g j b)` and changes
  the ensemble by `2 c` in population distance.  Only the *difference* of the observable between
  the two conformations is visible.
* `pair_degeneracy` -- hence the structural degeneracy statement, the exact counterpart of the
  sequence degeneracy of Part LXVII: if two conformations give **equal** values of every measured
  observable, then their relative population is completely undetermined -- the data are matched
  exactly while the ensemble moves by `2 d`.  Not resolved poorly: not resolved at all.
* `precision_floor` -- and the quantitative version, which is the point of this part.  Whatever
  the number `k` of restraints, whatever the library, whatever the fitting method, an ensemble
  matched to tolerance `eps` against observables bounded by `G` is determined only to population
  distance `min (2 d) (eps / G)`.  The floor does not contain `k`.  Adding experiments cannot
  push it down; only reducing `eps` -- more frames, more photons, better calibration -- can.
* `tolerance_ceiling` -- the matching upper bound, so the `eps`-scaling is the truth and not an
  artefact of the construction: if the `m - 1` populations themselves are measured to tolerance
  `eps`, two consistent ensembles differ by at most `2 (m - 1) eps`.

The two bounds bracket the resolution of an ensemble measurement between `eps / G` and
`2 (m - 1) eps`.  Both are linear in the tolerance and neither improves with the number of
restraints, which is the practical content: *an ensemble of a disordered region is a measurement
of finite resolution, and its resolution is set by the accuracy of the data and the structural
contrast of the observables, not by the size of the restraint list*.  Reporting populations to a
precision finer than `eps / G` reports the prior; reporting the relative population of two
conformations that the observables cannot separate reports nothing at all.
-/
import Mathlib
import RequestProject.Restraints

set_option autoImplicit false

namespace IDR

namespace Tolerance

open Finset IDR.Restraint

variable {m k : ℕ}

/-- The population-transfer direction: take weight from `b` and give it to `a`. -/
noncomputable def transfer (a b : Fin m) : Fin m → ℝ :=
  fun i => (if i = a then (1:ℝ) else 0) - (if i = b then (1:ℝ) else 0)

lemma transfer_sum (a b : Fin m) : ∑ i, transfer a b i = 0 := by
  simp [transfer, Finset.sum_sub_distrib]

lemma obs_transfer (w : Fin m → ℝ) (a b : Fin m) : obs w (transfer a b) = w a - w b := by
  simp [obs, transfer, mul_sub, Finset.sum_sub_distrib]

/-- **Moving weight between two conformations.**  Transferring population `c` from `b` to `a`
keeps the ensemble legitimate (provided `c` does not exceed the interior margin), moves it by at
least `2 c` in population distance, and changes the `j`-th measured average by exactly
`c (g j a - g j b)`. -/
theorem exists_pair_perturbation (g : Fin k → Fin m → ℝ)
    {p : Fin m → ℝ} {d c : ℝ} (hd : 0 < d) (hc : 0 < c) (hcd : c ≤ d)
    (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1) (a b : Fin m) (hab : a ≠ b) :
    ∃ q : Fin m → ℝ, IsEns q ∧ (∀ j, obs (g j) q - obs (g j) p = c * (g j a - g j b)) ∧
      2 * c ≤ ell1 q p := by
  set v : Fin m → ℝ := transfer a b with hv
  refine ⟨fun i => p i + c * v i, ⟨fun i => ?_, ?_⟩, fun j => ?_, ?_⟩
  · show (0:ℝ) ≤ p i + c * v i
    have hpi := hp i
    by_cases h1 : i = a
    · have hva : v i = 1 := by simp [hv, transfer, h1, hab]
      rw [hva]; nlinarith
    · by_cases h2 : i = b
      · have hvb : v i = -1 := by simp [hv, transfer, h2, Ne.symm hab]
        rw [hvb]; nlinarith
      · have hv0 : v i = 0 := by simp [hv, transfer, h1, h2]
        rw [hv0]; nlinarith
  · rw [Finset.sum_add_distrib, hp1, ← Finset.mul_sum, hv, transfer_sum, mul_zero, add_zero]
  · rw [obs_add_smul, hv, obs_transfer]; ring
  · have hell : ell1 (fun i => p i + c * v i) p = ∑ i, |c * v i| := by
      simp only [ell1, add_sub_cancel_left]
    have hsum0 : ∑ i, c * v i = 0 := by rw [← Finset.mul_sum, hv, transfer_sum, mul_zero]
    have hkey := two_mul_abs_le_sum_abs (fun i => c * v i) hsum0 a
    have hva : v a = 1 := by simp [hv, transfer, hab]
    rw [hell]
    rw [hva, mul_one, abs_of_pos hc] at hkey
    exact hkey

/-- **Structural degeneracy.**  If two conformations give the same value of every measured
observable, their relative population is not determined at all: an ensemble at population distance
`2 d` from the truth reproduces every measurement exactly.  This is the conformational counterpart
of the sequence degeneracy of Part LXVII, and it is independent of the number of restraints. -/
theorem pair_degeneracy (g : Fin k → Fin m → ℝ)
    {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1)
    {a b : Fin m} (hab : a ≠ b) (hdeg : ∀ j, g j a = g j b) :
    ∃ q : Fin m → ℝ, IsEns q ∧ (∀ j, obs (g j) q = obs (g j) p) ∧ 2 * d ≤ ell1 q p := by
  obtain ⟨q, hq, hdata, hfar⟩ :=
    exists_pair_perturbation g hd hd (le_refl d) hp hp1 a b hab
  refine ⟨q, hq, fun j => ?_, hfar⟩
  have := hdata j
  rw [hdeg j, sub_self, mul_zero] at this
  linarith

/-- **The precision floor.**  Whatever the number of restraints, an ensemble matched to tolerance
`eps` against observables bounded by `G` is pinned down only to population distance
`min (2 d) (eps / G)`: there is a genuine ensemble that far away whose predicted averages all lie
within tolerance.  The bound contains no `k`: more experiments do not sharpen it. -/
theorem precision_floor (g : Fin k → Fin m → ℝ) {G eps : ℝ} (hG : 0 < G) (heps : 0 < eps)
    (hgb : ∀ j i, |g j i| ≤ G)
    {p : Fin m → ℝ} {d : ℝ} (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1)
    {a b : Fin m} (hab : a ≠ b) :
    ∃ q : Fin m → ℝ, IsEns q ∧ (∀ j, |obs (g j) q - obs (g j) p| ≤ eps) ∧
      min (2 * d) (eps / G) ≤ ell1 q p := by
  set c : ℝ := min d (eps / (2 * G)) with hc
  have hcpos : 0 < c := lt_min hd (by positivity)
  obtain ⟨q, hq, hdata, hfar⟩ := exists_pair_perturbation g hd hcpos (min_le_left _ _) hp hp1 a b hab
  refine ⟨q, hq, fun j => ?_, ?_⟩
  · rw [hdata j, abs_mul, abs_of_pos hcpos]
    have hdiff : |g j a - g j b| ≤ 2 * G := by
      have := hgb j a; have := hgb j b
      calc |g j a - g j b| ≤ |g j a| + |g j b| := abs_sub _ _
      _ ≤ 2 * G := by linarith
    have hcG : c ≤ eps / (2 * G) := min_le_right _ _
    calc c * |g j a - g j b| ≤ (eps / (2 * G)) * (2 * G) := by
          apply mul_le_mul hcG hdiff (abs_nonneg _) (by positivity)
    _ = eps := by field_simp
  · refine le_trans ?_ hfar
    rcases le_total d (eps / (2 * G)) with h | h
    · have : c = d := min_eq_left h
      rw [this]
      exact min_le_left _ _
    · have hce : c = eps / (2 * G) := min_eq_right h
      have : 2 * c = eps / G := by rw [hce]; field_simp
      rw [this]
      exact min_le_right _ _

/-- **The matching ceiling.**  If the populations themselves are measured to tolerance `eps`, two
ensembles consistent with the data differ by at most `2 (m - 1) eps` in population distance.  So
the resolution of an ensemble measurement is linear in the tolerance from both sides, and in
neither direction does it depend on the number of restraints. -/
theorem tolerance_ceiling {p q : Fin m → ℝ} {eps : ℝ}
    (hp1 : ∑ i, p i = 1) (hq1 : ∑ i, q i = 1) (i0 : Fin m)
    (h : ∀ i, i ≠ i0 → |q i - p i| ≤ eps) :
    ell1 q p ≤ 2 * (m - 1 : ℕ) * eps := by
  have hcard : (univ.erase i0).card = m - 1 := by
    rw [Finset.card_erase_of_mem (mem_univ i0), Finset.card_univ, Fintype.card_fin]
  have hrest : ∑ i ∈ univ.erase i0, |q i - p i| ≤ (m - 1 : ℕ) * eps := by
    calc ∑ i ∈ univ.erase i0, |q i - p i|
        ≤ ∑ _i ∈ univ.erase i0, eps :=
          Finset.sum_le_sum fun i hi => h i (Finset.ne_of_mem_erase hi)
    _ = (m - 1 : ℕ) * eps := by rw [Finset.sum_const, hcard, nsmul_eq_mul]
  have hzero : q i0 - p i0 = -∑ i ∈ univ.erase i0, (q i - p i) := by
    have h1 := Finset.add_sum_erase (univ : Finset (Fin m)) (fun i => q i - p i) (mem_univ i0)
    have h2 : ∑ i, (q i - p i) = 0 := by
      rw [Finset.sum_sub_distrib, hp1, hq1, sub_self]
    linarith [h1, h2]
  have hi0 : |q i0 - p i0| ≤ (m - 1 : ℕ) * eps := by
    rw [hzero, abs_neg]
    exact le_trans (Finset.abs_sum_le_sum_abs _ _) hrest
  have hsplit : ell1 q p = |q i0 - p i0| + ∑ i ∈ univ.erase i0, |q i - p i| := by
    rw [ell1, ← Finset.add_sum_erase univ (fun i => |q i - p i|) (mem_univ i0)]
  rw [hsplit]
  linarith

end Tolerance

end IDR
