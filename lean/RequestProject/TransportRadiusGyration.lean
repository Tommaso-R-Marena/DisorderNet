/-
# Which experimental observables an RMSD-level error bound actually certifies

The transport distance of `RequestProject.TransportGeometry` is only as useful as the
observables it controls, and `RequestProject.TransportDuality` shows that control is
governed by one number per observable: its Lipschitz constant against the structural
metric.  This file computes those constants for the two observables that dominate the
experimental study of intrinsically disordered regions, on honest Cartesian conformations
of an `m`-residue chain with the (fixed-frame) RMSD as structural metric.

* `gyr_lipschitz` -- the **radius of gyration is `1`-Lipschitz in RMSD**.  An ensemble
  model certified to `ε` ångström in transport distance predicts the SAXS radius of
  gyration to `ε` ångström, whatever the two ensembles look like.  The bound survives
  optimisation over translations (`gyr_shift`), so it is a statement about shape, not
  about the frame.
* `resDist_lipschitz` -- a **single inter-residue distance is only `√(2m)`-Lipschitz**,
  and `resDist_lipschitz_sharp` shows that constant is attained.  The same `ε` therefore
  certifies a FRET pair distance only to `√(2m)·ε`: with `m = 100` residues, a `0.5 Å`
  ensemble is worth `7 Å` on the FRET observable.  Global size observables are
  intrinsically far more robust than local ones, and this is a theorem, not a heuristic.
* `rg_gap_le_transportCost`, `worked_saxs_certificate` -- the certificate in the direction
  an experimentalist uses it: a `5 Å` disagreement between a model's mean radius of
  gyration and the measured one proves the model ensemble is at least `5 Å` from the truth
  in transport distance, with no further assumption.
* `fret_bound_from_transport` -- and the same error budget applied to the FRET observable,
  with the `√(2m)` penalty made explicit.

`rmsd_isMetric` records that the fixed-frame RMSD really is a metric, which is what licenses
all of the above via the results of the two preceding files.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Transport
import RequestProject.TransportGeometry
import RequestProject.TransportDuality

namespace IDR

open Finset
open scoped Classical

/-- A Cartesian conformation of an `m`-residue chain: three coordinates per residue. -/
abbrev Struct (m : ℕ) := EuclideanSpace ℝ (Fin m × Fin 3)

variable {m : ℕ}

/-! ## The structural metric -/

/-- Root-mean-square deviation in a fixed frame: the Euclidean distance between two
conformations, normalised per residue. -/
noncomputable def rmsd (x y : Struct m) : ℝ := ‖x - y‖ / Real.sqrt m

lemma norm_eq_sqrt_sum (x : Struct m) : ‖x‖ = Real.sqrt (∑ p, (x p) ^ 2) := by
  rw [EuclideanSpace.norm_eq]
  simp [sq_abs]

lemma rmsd_nonneg (x y : Struct m) : 0 ≤ rmsd x y :=
  div_nonneg (norm_nonneg _) (Real.sqrt_nonneg _)

lemma rmsd_self (x : Struct m) : rmsd x x = 0 := by simp [rmsd]

lemma rmsd_comm (x y : Struct m) : rmsd x y = rmsd y x := by
  rw [rmsd, rmsd, norm_sub_rev]

lemma rmsd_triangle (x y z : Struct m) : rmsd x z ≤ rmsd x y + rmsd y z := by
  rw [rmsd, rmsd, rmsd, ← add_div]
  refine div_le_div_of_nonneg_right ?_ ?_ |>.trans_eq rfl
  · exact norm_sub_le_norm_sub_add_norm_sub x y z
  · exact Real.sqrt_nonneg _

lemma rmsd_eq_zero (hm : 0 < m) {x y : Struct m} (h : rmsd x y = 0) : x = y := by
  have hs : Real.sqrt (m : ℝ) ≠ 0 :=
    ne_of_gt (Real.sqrt_pos.2 (by exact_mod_cast hm))
  have hz : ‖x - y‖ = 0 := by
    rw [rmsd, div_eq_zero_iff] at h
    rcases h with h | h
    · exact h
    · exact absurd h hs
  exact sub_eq_zero.1 (norm_eq_zero.1 hz)

/-- The fixed-frame RMSD is a metric on conformation space, so all the transport results
apply to it. -/
theorem rmsd_isMetric (hm : 0 < m) :
    (∀ x y : Struct m, 0 ≤ rmsd x y) ∧ (∀ x : Struct m, rmsd x x = 0) ∧
    (∀ x y : Struct m, rmsd x y = rmsd y x) ∧
    (∀ x y z : Struct m, rmsd x z ≤ rmsd x y + rmsd y z) ∧
    (∀ x y : Struct m, rmsd x y = 0 → x = y) :=
  ⟨rmsd_nonneg, rmsd_self, rmsd_comm, rmsd_triangle, fun _ _ h => rmsd_eq_zero hm h⟩

/-! ## The radius of gyration is `1`-Lipschitz -/

/-- The conformation with its centre of mass removed. -/
noncomputable def centerVec (x : Struct m) : Struct m :=
  WithLp.toLp 2 (fun p => x p - (m : ℝ)⁻¹ * ∑ k, x (k, p.2))

@[simp] lemma centerVec_apply (x : Struct m) (p : Fin m × Fin 3) :
    centerVec x p = x p - (m : ℝ)⁻¹ * ∑ k, x (k, p.2) := rfl

/-- The radius of gyration: the root-mean-square distance of the residues from their
centre of mass. -/
noncomputable def gyr (x : Struct m) : ℝ := ‖centerVec x‖ / Real.sqrt m

lemma centerVec_sub (x y : Struct m) : centerVec (x - y) = centerVec x - centerVec y := by
  ext p
  simp [Finset.sum_sub_distrib, mul_sub]
  ring

/-- Centring can only shrink: the mean is the minimiser of the sum of squared deviations. -/
lemma sum_sub_mean_sq_le (f : Fin m → ℝ) :
    ∑ k, (f k - (m : ℝ)⁻¹ * ∑ j, f j) ^ 2 ≤ ∑ k, (f k) ^ 2 := by
  rcases Nat.eq_zero_or_pos m with hm | hm
  · subst hm; simp
  have hmpos : (0 : ℝ) < m := by exact_mod_cast hm
  set s := ∑ j, f j with hs
  have hexp : ∑ k, (f k - (m : ℝ)⁻¹ * s) ^ 2
      = ∑ k, (f k) ^ 2 - 2 * ((m : ℝ)⁻¹ * s) * s + m * ((m : ℝ)⁻¹ * s) ^ 2 := by
    have : ∀ k, (f k - (m : ℝ)⁻¹ * s) ^ 2
        = (f k) ^ 2 - 2 * ((m : ℝ)⁻¹ * s) * f k + ((m : ℝ)⁻¹ * s) ^ 2 := fun k => by ring
    rw [Finset.sum_congr rfl fun k _ => this k]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← hs]
    simp [mul_comm]
  rw [hexp]
  have : m * ((m : ℝ)⁻¹ * s) ^ 2 = (m : ℝ)⁻¹ * s ^ 2 := by
    field_simp
  rw [this]
  have h2 : 2 * ((m : ℝ)⁻¹ * s) * s = 2 * ((m : ℝ)⁻¹ * s ^ 2) := by ring
  rw [h2]
  have : 0 ≤ (m : ℝ)⁻¹ * s ^ 2 := mul_nonneg (by positivity) (sq_nonneg _)
  linarith

lemma norm_centerVec_le (x : Struct m) : ‖centerVec x‖ ≤ ‖x‖ := by
  rw [norm_eq_sqrt_sum, norm_eq_sqrt_sum]
  refine Real.sqrt_le_sqrt ?_
  have hsplit : ∀ y : Struct m, ∑ p, (y p) ^ 2 = ∑ a : Fin 3, ∑ k : Fin m, (y (k, a)) ^ 2 := by
    intro y
    rw [Fintype.sum_prod_type]
    exact Finset.sum_comm
  rw [hsplit, hsplit]
  refine Finset.sum_le_sum fun a _ => ?_
  simpa using sum_sub_mean_sq_le (fun k => x (k, a))

/-- **The radius of gyration is `1`-Lipschitz with respect to RMSD.**  Two conformations
that differ by `ε` ångström RMSD differ by at most `ε` ångström in radius of gyration. -/
theorem gyr_lipschitz (x y : Struct m) : |gyr x - gyr y| ≤ 1 * rmsd x y := by
  rw [one_mul, gyr, gyr, rmsd, div_sub_div_same, abs_div,
    abs_of_nonneg (Real.sqrt_nonneg (m : ℝ))]
  refine div_le_div_of_nonneg_right ?_ (Real.sqrt_nonneg _) |>.trans_eq rfl
  calc |‖centerVec x‖ - ‖centerVec y‖| ≤ ‖centerVec x - centerVec y‖ := abs_norm_sub_norm_le _ _
    _ = ‖centerVec (x - y)‖ := by rw [centerVec_sub]
    _ ≤ ‖x - y‖ := norm_centerVec_le _

/-- Translating a whole conformation leaves the radius of gyration unchanged, so the bound
above is a statement about shape: it survives optimising the RMSD over translations. -/
lemma gyr_shift (x : Struct m) (t : Fin 3 → ℝ) :
    gyr (WithLp.toLp 2 (fun p : Fin m × Fin 3 => x p + t p.2)) = gyr x := by
  have hc : centerVec (WithLp.toLp 2 (fun p : Fin m × Fin 3 => x p + t p.2)) = centerVec x := by
    ext p
    rcases Nat.eq_zero_or_pos m with hm | hm
    · exact absurd p.1.2 (by simp [hm])
    have hmne : (m : ℝ) ≠ 0 := by positivity
    simp only [centerVec_apply]
    rw [Finset.sum_add_distrib]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
    ring
  rw [gyr, gyr, hc]

/-! ## A single inter-residue distance is only `√(2m)`-Lipschitz -/

/-- The distance between two labelled residues -- the quantity a FRET experiment reports. -/
noncomputable def resDist (x : Struct m) (k l : Fin m) : ℝ :=
  ‖(WithLp.toLp 2 (fun a : Fin 3 => x (k, a)) - WithLp.toLp 2 (fun a : Fin 3 => x (l, a)) :
    EuclideanSpace ℝ (Fin 3))‖

lemma resDist_eq_sqrt (x : Struct m) (k l : Fin m) :
    resDist x k l = Real.sqrt (∑ a : Fin 3, (x (k, a) - x (l, a)) ^ 2) := by
  rw [resDist, EuclideanSpace.norm_eq]
  simp [sq_abs]

/-- The coordinates of a single residue are dominated by the whole conformation. -/
lemma sum_res_le (d : Struct m) (k : Fin m) : ∑ a : Fin 3, (d (k, a)) ^ 2 ≤ ∑ p, (d p) ^ 2 := by
  rw [Fintype.sum_prod_type]
  exact Finset.single_le_sum (f := fun k' : Fin m => ∑ a : Fin 3, (d (k', a)) ^ 2)
    (fun k' _ => Finset.sum_nonneg fun a _ => sq_nonneg _) (mem_univ k)

/-- Two distinct residues together are dominated by the whole conformation. -/
lemma sum_res_pair_le (d : Struct m) {k l : Fin m} (hkl : k ≠ l) :
    (∑ a : Fin 3, (d (k, a)) ^ 2) + ∑ a : Fin 3, (d (l, a)) ^ 2 ≤ ∑ p, (d p) ^ 2 := by
  rw [Fintype.sum_prod_type]
  have hsub : ({k, l} : Finset (Fin m)) ⊆ univ := subset_univ _
  have hpair : ∑ k' ∈ ({k, l} : Finset (Fin m)), ∑ a : Fin 3, (d (k', a)) ^ 2
      = (∑ a : Fin 3, (d (k, a)) ^ 2) + ∑ a : Fin 3, (d (l, a)) ^ 2 :=
    Finset.sum_pair hkl
  rw [← hpair]
  exact Finset.sum_le_sum_of_subset_of_nonneg hsub
    (fun k' _ _ => Finset.sum_nonneg fun a _ => sq_nonneg _)

/-- **A single inter-residue distance is `√(2m)`-Lipschitz in RMSD.**  The Lipschitz
constant grows with chain length: an RMSD-level guarantee says much less about one FRET
pair than about the global size. -/
theorem resDist_lipschitz {k l : Fin m} (hkl : k ≠ l) (x y : Struct m) :
    |resDist x k l - resDist y k l| ≤ Real.sqrt (2 * m) * rmsd x y := by
  rcases Nat.eq_zero_or_pos m with hm | hm
  · subst hm; exact absurd k.2 (by simp)
  have hmpos : (0 : ℝ) < m := by exact_mod_cast hm
  set u : EuclideanSpace ℝ (Fin 3) :=
    WithLp.toLp 2 (fun a : Fin 3 => x (k, a)) - WithLp.toLp 2 (fun a : Fin 3 => x (l, a)) with hu
  set v : EuclideanSpace ℝ (Fin 3) :=
    WithLp.toLp 2 (fun a : Fin 3 => y (k, a)) - WithLp.toLp 2 (fun a : Fin 3 => y (l, a)) with hv
  have hstep : |resDist x k l - resDist y k l| ≤ ‖u - v‖ := abs_norm_sub_norm_le _ _
  have hsq : ‖u - v‖ ^ 2 ≤ 2 * ‖x - y‖ ^ 2 := by
    have hnorm : ‖u - v‖ ^ 2 = ∑ a : Fin 3, ((x (k, a) - y (k, a)) - (x (l, a) - y (l, a))) ^ 2 := by
      rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
      refine Finset.sum_congr rfl fun a _ => ?_
      simp [hu, hv, sq_abs]
      ring_nf
    have hxy : ‖x - y‖ ^ 2 = ∑ p, ((x - y) p) ^ 2 := by
      rw [norm_eq_sqrt_sum, Real.sq_sqrt (by positivity)]
    have hbound : ∑ a : Fin 3, ((x (k, a) - y (k, a)) - (x (l, a) - y (l, a))) ^ 2
        ≤ 2 * ((∑ a : Fin 3, ((x - y) (k, a)) ^ 2) + ∑ a : Fin 3, ((x - y) (l, a)) ^ 2) := by
      rw [mul_add, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      refine Finset.sum_le_sum fun a _ => ?_
      have h1 : (x - y) (k, a) = x (k, a) - y (k, a) := by simp
      have h2 : (x - y) (l, a) = x (l, a) - y (l, a) := by simp
      rw [h1, h2]
      nlinarith [sq_nonneg ((x (k, a) - y (k, a)) + (x (l, a) - y (l, a)))]
    rw [hnorm, hxy]
    refine hbound.trans ?_
    have hpair := sum_res_pair_le (x - y) hkl
    linarith
  have hnn : (0 : ℝ) ≤ Real.sqrt 2 * ‖x - y‖ := by positivity
  have hle : ‖u - v‖ ≤ Real.sqrt 2 * ‖x - y‖ := by
    have hsq2 : ‖u - v‖ ^ 2 ≤ (Real.sqrt 2 * ‖x - y‖) ^ 2 := by
      rw [mul_pow, Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2)]
      exact hsq
    calc ‖u - v‖ = Real.sqrt (‖u - v‖ ^ 2) := (Real.sqrt_sq (norm_nonneg _)).symm
      _ ≤ Real.sqrt ((Real.sqrt 2 * ‖x - y‖) ^ 2) := Real.sqrt_le_sqrt hsq2
      _ = Real.sqrt 2 * ‖x - y‖ := Real.sqrt_sq hnn
  have hfinal : Real.sqrt 2 * ‖x - y‖ = Real.sqrt (2 * m) * rmsd x y := by
    rw [rmsd, Real.sqrt_mul (by norm_num : (0:ℝ) ≤ 2)]
    have hs : Real.sqrt (m : ℝ) ≠ 0 := ne_of_gt (Real.sqrt_pos.2 hmpos)
    field_simp
  exact hstep.trans (hle.trans_eq hfinal)

/-! ## The certificates -/

/-- **The SAXS certificate.**  The gap between a model's mean radius of gyration and the
true one is a lower bound, in ångström, on the transport distance between the ensembles. -/
theorem rg_gap_le_transportCost (E F : Ens (Struct m)) :
    |E.expect gyr - F.expect gyr| ≤ transportCost rmsd E F := by
  have := expect_diff_le_of_lipschitz (c := rmsd (m := m)) rmsd_nonneg
    (L := 1) (f := gyr) gyr_lipschitz E F
  simpa using this

/-- A worked case: a 100-residue disordered region whose measured mean radius of gyration
is `27 Å` while the candidate ensemble predicts `22 Å`.  The candidate is at least `5 Å`
away from the truth in transport distance -- irrespective of how either ensemble is
shaped, and irrespective of how the model was built. -/
theorem worked_saxs_certificate {E F : Ens (Struct 100)}
    (hE : E.expect gyr = 22) (hF : F.expect gyr = 27) :
    5 ≤ transportCost rmsd E F := by
  have h := rg_gap_le_transportCost (m := 100) E F
  rw [hE, hF] at h
  norm_num at h
  exact h

/-- **The FRET side of the same budget.**  An ensemble certified to `ε` in transport
distance pins a single labelled-pair distance only to `√(2m)·ε`. -/
theorem fret_bound_from_transport {k l : Fin m} (hkl : k ≠ l) (E F : Ens (Struct m)) :
    |E.expect (fun x => resDist x k l) - F.expect (fun x => resDist x k l)|
      ≤ Real.sqrt (2 * m) * transportCost rmsd E F :=
  expect_diff_le_of_lipschitz (c := rmsd (m := m)) rmsd_nonneg
    (L := Real.sqrt (2 * m)) (f := fun x => resDist x k l) (resDist_lipschitz hkl) E F

/-- The conformation in which every residue sits at the origin. -/
noncomputable def zeroStruct (m : ℕ) : Struct m := 0

/-- The conformation in which the two labelled residues are displaced by `±t` along the
first axis and every other residue sits at the origin. -/
noncomputable def twoLabelStruct (k l : Fin m) (t : ℝ) : Struct m :=
  WithLp.toLp 2 (fun p : Fin m × Fin 3 =>
    if p.1 = k then (if p.2 = 0 then t else 0)
    else if p.1 = l then (if p.2 = 0 then -t else 0) else 0)

lemma norm_twoLabelStruct {k l : Fin m} (hkl : k ≠ l) (t : ℝ) :
    ‖twoLabelStruct k l t‖ = |t| * Real.sqrt 2 := by
  rw [norm_eq_sqrt_sum]
  have hsum : ∑ p, (twoLabelStruct k l t p) ^ 2 = t ^ 2 * 2 := by
    rw [Fintype.sum_prod_type]
    have hterm : ∀ k' : Fin m, ∑ a : Fin 3, (twoLabelStruct k l t (k', a)) ^ 2
        = (if k' = k then t ^ 2 else 0) + (if k' = l then t ^ 2 else 0) := by
      intro k'
      by_cases h1 : k' = k
      · subst h1
        simp [twoLabelStruct, hkl]
      · by_cases h2 : k' = l
        · subst h2
          simp [twoLabelStruct, h1]
        · simp [twoLabelStruct, h1, h2]
    rw [Finset.sum_congr rfl fun k' _ => hterm k', Finset.sum_add_distrib]
    simp [Finset.sum_ite_eq']
    ring
  rw [hsum, Real.sqrt_mul (sq_nonneg t), Real.sqrt_sq_eq_abs]

lemma resDist_zeroStruct (k l : Fin m) : resDist (zeroStruct m) k l = 0 := by
  rw [resDist_eq_sqrt]
  simp [zeroStruct]

lemma resDist_twoLabelStruct {k l : Fin m} (hkl : k ≠ l) {t : ℝ} (ht : 0 ≤ t) :
    resDist (twoLabelStruct k l t) k l = 2 * t := by
  rw [resDist_eq_sqrt]
  have : ∀ a : Fin 3, (twoLabelStruct k l t (k, a) - twoLabelStruct k l t (l, a)) ^ 2
      = if a = 0 then (2 * t) ^ 2 else 0 := by
    intro a
    by_cases ha : a = 0
    · subst ha; simp [twoLabelStruct, Ne.symm hkl]; ring
    · simp [twoLabelStruct, Ne.symm hkl, ha]
  rw [Finset.sum_congr rfl fun a _ => this a]
  simp [Finset.sum_ite_eq']
  exact Real.sqrt_sq (by linarith)

/-- **The `√(2m)` penalty is real.**  Two single-structure ensembles differing only in the
positions of the two labelled residues attain the Lipschitz constant exactly: the labelled
pair distance changes by `√(2m)` times the RMSD, which is also their transport distance.
So an RMSD-level guarantee genuinely says `√(2m)` times less about one FRET pair than about
the radius of gyration. -/
theorem resDist_lipschitz_sharp (hm : 0 < m) {k l : Fin m} (hkl : k ≠ l) {t : ℝ} (ht : 0 < t) :
    ∃ x y : Struct m,
      0 < rmsd x y ∧
      transportCost rmsd (Ens.dirac x) (Ens.dirac y) = rmsd x y ∧
      |resDist x k l - resDist y k l| = Real.sqrt (2 * m) * rmsd x y := by
  have hmpos : (0 : ℝ) < m := by exact_mod_cast hm
  have hsm : (0 : ℝ) < Real.sqrt (m : ℝ) := Real.sqrt_pos.2 hmpos
  have hsub : zeroStruct m - twoLabelStruct k l t = -twoLabelStruct k l t := by
    simp [zeroStruct]
  have hrm : rmsd (zeroStruct m) (twoLabelStruct k l t) = t * Real.sqrt 2 / Real.sqrt m := by
    rw [rmsd, hsub, norm_neg, norm_twoLabelStruct hkl, abs_of_pos ht]
  refine ⟨zeroStruct m, twoLabelStruct k l t, ?_, transportCost_dirac _ _ _, ?_⟩
  · rw [hrm]
    have h2 : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
    positivity
  · rw [hrm, resDist_zeroStruct, resDist_twoLabelStruct hkl ht.le, zero_sub, abs_neg,
      abs_of_pos (by linarith : (0 : ℝ) < 2 * t), Real.sqrt_mul (by norm_num : (0:ℝ) ≤ 2)]
    have h2 : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
    field_simp
    nlinarith [h2, hsm]

end IDR
