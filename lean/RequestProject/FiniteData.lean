/-
# No finite set of measurements determines a conformational ensemble

The counterexamples of `RequestProject.Statistics` are specific: coordinates, distograms,
marginals, two moments.  This file proves the general statement they are instances of.

`not_separates_of_finite`: for **any** finite family `f 0, ..., f (m-1)` of observables
there exist two conformational ensembles with identical averages of all of them, which are
nevertheless observationally different.  The proof is a dimension count: matching `m`
averages plus normalisation is `m + 1` linear conditions on the weights, so on `m + 2`
conformations there is always a direction in weight space along which the data do not
change.

Consequences (`finite_statistics_predictor_fails`, `finite_experiment_underdetermined`):
any predictor -- of any architecture, trained on any amount of data -- whose output
depends on the target ensemble only through finitely many statistics is wrong on some
ensemble; and ensemble refinement against a finite set of experimental restraints (SAXS
curves, NMR chemical shifts, PRE or FRET averages) never determines the ensemble.  Note
that this is *not* a defect of the ensemble picture, which is exact
(`solvable_by_conditional_ensembles`); it is a limit on what finitely many measurements,
or finitely many fitted statistics, can pin down.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics

set_option maxHeartbeats 1000000

namespace IDR

open Finset
open scoped Classical

/-- A dimension count: `m + 1` homogeneous linear conditions on `m + 2` unknowns have a
nonzero solution. -/
theorem exists_nonzero_of_underdetermined {m : ℕ} (g : Fin (m + 1) → Fin (m + 2) → ℝ) :
    ∃ v : Fin (m + 2) → ℝ, v ≠ 0 ∧ ∀ i, ∑ j, g i j * v j = 0 := by
  set M : Matrix (Fin (m + 1)) (Fin (m + 2)) ℝ := Matrix.of g with hM
  have hni : ¬ Function.Injective (Matrix.mulVecLin M) := by
    intro hinj
    have h := LinearMap.finrank_le_finrank_of_injective hinj
    simp at h
  rw [Function.not_injective_iff] at hni
  obtain ⟨a, b, hab, hne⟩ := hni
  refine ⟨a - b, sub_ne_zero.2 hne, fun i => ?_⟩
  have h0 : Matrix.mulVecLin M (a - b) = 0 := by rw [map_sub, hab, sub_self]
  have hi := congrFun h0 i
  simp only [Matrix.mulVecLin_apply, Matrix.mulVec, dotProduct, hM, Matrix.of_apply,
    Pi.zero_apply, Pi.sub_apply] at hi
  rw [← hi]
  exact Finset.sum_congr rfl fun j _ => by simp [Pi.sub_apply, mul_sub]

/-- The `m + 2` distinct one-dimensional conformations `0, 1, ..., m + 1`. -/
noncomputable def ladder (m : ℕ) : Fin (m + 2) → Conf 1 := fun j => fun _ => (j : ℝ)

lemma ladder_injective (m : ℕ) : Function.Injective (ladder m) := by
  intro a b hab
  have h1 : ((a : ℕ) : ℝ) = ((b : ℕ) : ℝ) := congrFun hab 0
  have h2 : (a : ℕ) = (b : ℕ) := by exact_mod_cast h1
  exact Fin.ext h2

/-- The ensemble supported on the `m + 2` conformations of `ladder m` with weights `u`. -/
noncomputable def ladderEns (m : ℕ) (u : Fin (m + 2) → ℝ) (hu : ∀ j, 0 ≤ u j)
    (hs : ∑ j, u j = 1) : Ens (Conf 1) where
  card := m + 2
  pt := ladder m
  w := u
  w_nonneg := hu
  w_sum := hs

lemma ladderEns_expect {m : ℕ} {u : Fin (m + 2) → ℝ} (hu : ∀ j, 0 ≤ u j)
    (hs : ∑ j, u j = 1) (g : Conf 1 → ℝ) :
    (ladderEns m u hu hs).expect g = ∑ j, u j * g (ladder m j) := by
  simp [ladderEns, Ens.expect]

lemma ladderEns_prob {m : ℕ} {u : Fin (m + 2) → ℝ} (hu : ∀ j, 0 ≤ u j)
    (hs : ∑ j, u j = 1) (j : Fin (m + 2)) :
    (ladderEns m u hu hs).prob (ladder m j) = u j := by
  have h := Ens.prob_pt_of_injective (ladderEns m u hu hs) (ladder_injective m) j
  exact h

/-- **No finite family of observables determines the ensemble.**  Whatever finitely many
averages a model is fitted to, two observationally different ensembles produce exactly
those averages. -/
theorem exists_two_ensembles_matching {m : ℕ} (f : Fin m → (Conf 1 → ℝ)) :
    ∃ E F : Ens (Conf 1), (∀ i, E.expect (f i) = F.expect (f i)) ∧ ¬ E.Same F := by
  classical
  set N : ℝ := (m : ℝ) + 2 with hN
  have hNpos : 0 < N := by positivity
  have hNcast : ((m + 2 : ℕ) : ℝ) = N := by rw [hN]; push_cast; ring
  obtain ⟨v, hv0, hv⟩ := exists_nonzero_of_underdetermined
    (Fin.cons (fun _ => (1 : ℝ)) (fun i j => f i (ladder m j)))
  have hsum : ∑ j, v j = 0 := by
    have h := hv 0
    simpa using h
  have hstat : ∀ i : Fin m, ∑ j, f i (ladder m j) * v j = 0 := by
    intro i
    have h := hv i.succ
    simpa using h
  obtain ⟨j₀, hj₀⟩ : ∃ j, v j ≠ 0 := by
    by_contra hcon
    push_neg at hcon
    exact hv0 (funext hcon)
  set s : ℝ := ∑ j, |v j| with hs
  have hspos : 0 < s :=
    Finset.sum_pos' (fun j _ => abs_nonneg _) ⟨j₀, Finset.mem_univ j₀, abs_pos.2 hj₀⟩
  set e : ℝ := 1 / (N * s) with he
  have hepos : 0 < e := by positivity
  have hes : e * s = 1 / N := by
    rw [he]; field_simp
  have hbound : ∀ j, e * |v j| ≤ 1 / N := by
    intro j
    have h1 : |v j| ≤ s :=
      Finset.single_le_sum (f := fun j => |v j|) (fun j _ => abs_nonneg _) (Finset.mem_univ j)
    have := mul_le_mul_of_nonneg_left h1 hepos.le
    linarith [hes]
  -- the uniform ensemble
  have hu1 : ∀ _j : Fin (m + 2), (0 : ℝ) ≤ 1 / N := fun _ => by positivity
  have hs1 : ∑ _j : Fin (m + 2), (1 / N : ℝ) = 1 := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, hNcast]
    field_simp
  -- the perturbed ensemble
  have hu2 : ∀ j : Fin (m + 2), (0 : ℝ) ≤ 1 / N + e * v j := by
    intro j
    have h1 : -(e * |v j|) ≤ e * v j := by
      have : -|v j| ≤ v j := neg_abs_le (v j)
      nlinarith [hepos]
    have h2 := hbound j
    linarith
  have hs2 : ∑ j : Fin (m + 2), (1 / N + e * v j) = 1 := by
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, hsum, hs1]
    ring
  refine ⟨ladderEns m _ hu1 hs1, ladderEns m _ hu2 hs2, fun i => ?_, ?_⟩
  · rw [ladderEns_expect, ladderEns_expect]
    have hsplit : ∀ j : Fin (m + 2),
        (1 / N + e * v j) * f i (ladder m j)
          = (1 / N) * f i (ladder m j) + e * (f i (ladder m j) * v j) := fun j => by ring
    rw [Finset.sum_congr rfl (fun j _ => hsplit j), Finset.sum_add_distrib,
      show (∑ j, e * (f i (ladder m j) * v j)) = e * ∑ j, f i (ladder m j) * v j from
        (Finset.mul_sum _ _ _).symm, hstat i]
    ring
  · intro hsame
    have h := Ens.prob_eq_of_same hsame (ladder m j₀)
    rw [ladderEns_prob, ladderEns_prob] at h
    have hzero : e * v j₀ = 0 := by linarith
    rcases mul_eq_zero.1 hzero with h' | h'
    · exact absurd h' (ne_of_gt hepos)
    · exact hj₀ h'

/-- The family of statistics `f 0, ..., f (m-1)` does not separate ensembles. -/
theorem not_separates_of_finite {m : ℕ} (f : Fin m → (Conf 1 → ℝ)) :
    ¬ Separates {g : Conf 1 → ℝ | ∃ i, g = f i} := by
  obtain ⟨E, F, hEF, hne⟩ := exists_two_ensembles_matching f
  refine not_separates_iff.2 ⟨E, F, ?_, hne⟩
  rintro _ ⟨i, rfl⟩
  exact hEF i

/-- **Any predictor that only sees finitely many statistics of its target is wrong on some
target.**  More data of the same finitely many kinds does not help; the missing
information is not in them. -/
theorem finite_statistics_predictor_fails {m : ℕ} (f : Fin m → (Conf 1 → ℝ))
    (A : Ens (Conf 1) → Ens (Conf 1))
    (hA : ∀ E F : Ens (Conf 1), AgreeOn {g : Conf 1 → ℝ | ∃ i, g = f i} E F → A E = A F) :
    ∃ E : Ens (Conf 1), ¬ (A E).Same E :=
  exists_failure_of_not_separating A hA (not_separates_of_finite f)

/-- **Ensemble refinement against finitely many experimental averages is
underdetermined.**  Two different ensembles fit the data exactly equally well. -/
theorem finite_experiment_underdetermined {m : ℕ} (f : Fin m → (Conf 1 → ℝ)) :
    ∃ E F : Ens (Conf 1), (∀ i, E.expect (f i) = F.expect (f i)) ∧ ¬ E.Same F :=
  exists_two_ensembles_matching f

end IDR
