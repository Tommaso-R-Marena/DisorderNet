/-
# Part CXLI  A disordered region is an ensemble: the titration reads the distance *distribution*

Parts CXXXVIII–CXL calibrated an internal-distance profile `R(d)` — one distance per sequence
separation — culminating in Part CXL's non-parametric identification theorem.  That is still one
structure per separation, and it is the wrong object for an intrinsically disordered region.  A
disordered region at a fixed separation `d` does not have *a* distance; it has a distribution of
distances over its conformational ensemble, and the screened Coulomb reading of a charge
correlation averages the kernel over that distribution:

  `ensCurve S r p κ = ∑_{k ∈ S} p k · e^{−κ·r k} / (r k)`,

with `r k` the distances realised by the ensemble and `p k` their weights.  Because the kernel is
not linear in the distance, this is *not* the kernel of any single distance, and the question is
what a complete titration sees.

* **The distribution is identified, weight by weight.**  `ens_weights_identifiable` — over a fixed
  finite set of candidate distances, all distinct and positive, the weights are determined by the
  titration curve on any half-line of ionic strengths.  A salt titration therefore measures the
  distance *distribution* of the ensemble at a separation, not merely a mean or an apparent
  distance.  `distance_distribution_identifiable` removes the candidate set: two finitely supported
  distributions of positive distances agreeing at every ionic strength have the same weight at
  every distance, so the support is identified along with the weights.

* **No single conformation can imitate an ensemble.**  `two_conformer_not_single_distance` — if a
  region realises two distinct distances with positive weights, then for *every* candidate single
  distance `R` there is an ionic strength at which the single-distance model and the ensemble
  disagree.  A single-structure model of a disordered region is falsifiable, and a complete
  titration falsifies it.

* **In particular, fitting the mean is not enough.**  `mean_distance_model_falsified` — the
  single-distance model placed at the ensemble's mean distance is separated from the ensemble by
  some ionic strength.  What the experiment reports is not the mean of the distance distribution;
  quoting `⟨R⟩` as the model's prediction is a statement the data can refute.

* **Tools.**  `two_exp_indep`, `three_exp_indep` — two and three real exponentials with distinct
  rates are independent as functions of the ionic strength on a half-line; specialisations of the
  independence lemma of Part CXXXVIII, stated in the form the arguments here need.

Design consequence, closing Parts CXXXIV–CXLI: the object a model of a charged disordered region
should be calibrated against is a *distribution of internal distances per sequence separation*.
Its weights are identified by a complete titration with an independently measured correlation
profile; its mean is not a sufficient summary; and a model that reports one structure, or one
distance per separation, makes a prediction that a titration can and generally will refute.
-/
import Mathlib
import RequestProject.DistanceProfile

set_option autoImplicit false

namespace IDR
namespace DistanceEnsemble

open Finset

/-! ## 1. Independence of two and three exponentials -/

/-- Two real exponentials with distinct rates are independent on a half-line of conditions. -/
theorem two_exp_indep {A B x y kappa0 : ℝ} (hxy : x ≠ y)
    (h : ∀ kappa, kappa0 ≤ kappa →
      A * Real.exp (-(kappa * x)) + B * Real.exp (-(kappa * y)) = 0) :
    A = 0 ∧ B = 0 := by
  classical
  set rate : ℕ → ℝ := fun j => if j = 0 then x else y with hrate
  set amp : ℕ → ℝ := fun j => if j = 0 then A else B with hamp
  have hinj : Set.InjOn rate (({0, 1} : Finset ℕ)) := by
    intro i hi j hj hij
    simp only [Finset.coe_insert, Finset.coe_singleton, Set.mem_insert_iff,
      Set.mem_singleton_iff] at hi hj
    rcases hi with rfl | rfl <;> rcases hj with rfl | rfl <;>
      simp_all
  have hzero : ∀ kappa, kappa0 ≤ kappa →
      DistanceLaw.expSum ({0, 1} : Finset ℕ) amp rate kappa = 0 := by
    intro k hk
    rw [DistanceLaw.expSum]
    simpa [hamp, hrate] using h k hk
  have h0 := DistanceLaw.expSum_eq_zero hinj hzero 0 (by decide)
  have h1 := DistanceLaw.expSum_eq_zero hinj hzero 1 (by decide)
  simp only [hamp] at h0 h1
  norm_num at h0 h1
  exact ⟨h0, h1⟩

/-- Three real exponentials with pairwise distinct rates are independent on a half-line of
conditions. -/
theorem three_exp_indep {A B C x y z kappa0 : ℝ} (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
    (h : ∀ kappa, kappa0 ≤ kappa →
      A * Real.exp (-(kappa * x)) + B * Real.exp (-(kappa * y))
        + C * Real.exp (-(kappa * z)) = 0) :
    A = 0 ∧ B = 0 ∧ C = 0 := by
  classical
  set rate : ℕ → ℝ := fun j => if j = 0 then x else if j = 1 then y else z with hrate
  set amp : ℕ → ℝ := fun j => if j = 0 then A else if j = 1 then B else C with hamp
  have hinj : Set.InjOn rate (({0, 1, 2} : Finset ℕ)) := by
    intro i hi j hj hij
    simp only [Finset.coe_insert, Finset.coe_singleton, Set.mem_insert_iff,
      Set.mem_singleton_iff] at hi hj
    rcases hi with rfl | rfl | rfl <;> rcases hj with rfl | rfl | rfl <;>
      simp_all
  have hzero : ∀ kappa, kappa0 ≤ kappa →
      DistanceLaw.expSum ({0, 1, 2} : Finset ℕ) amp rate kappa = 0 := by
    intro k hk
    rw [DistanceLaw.expSum]
    simpa [hamp, hrate, add_assoc] using h k hk
  have h0 := DistanceLaw.expSum_eq_zero hinj hzero 0 (by decide)
  have h1 := DistanceLaw.expSum_eq_zero hinj hzero 1 (by decide)
  have h2 := DistanceLaw.expSum_eq_zero hinj hzero 2 (by decide)
  simp only [hamp] at h0 h1 h2
  norm_num at h0 h1 h2
  exact ⟨h0, h1, h2⟩

/-! ## 2. The titration curve of a distance ensemble -/

/-- The screened reading of a charge correlation at one sequence separation, averaged over the
conformational ensemble: the ensemble realises the distances `r k`, `k ∈ S`, with weights `p k`. -/
noncomputable def ensCurve (S : Finset ℕ) (r p : ℕ → ℝ) (kappa : ℝ) : ℝ :=
  ∑ k ∈ S, p k * (Real.exp (-(kappa * r k)) / r k)

/-- **The distance distribution is identified.**  Over a fixed finite set of candidate distances,
all positive and distinct, a complete titration determines the weight of every candidate: two
ensembles agreeing at every ionic strength on a half-line are the same ensemble. -/
theorem ens_weights_identifiable {S : Finset ℕ} {r p p' : ℕ → ℝ} {kappa0 : ℝ}
    (hinj : Set.InjOn r S) (hpos : ∀ k ∈ S, 0 < r k)
    (h : ∀ kappa, kappa0 ≤ kappa → ensCurve S r p kappa = ensCurve S r p' kappa) :
    ∀ k ∈ S, p k = p' k := by
  have hzero : ∀ kappa, kappa0 ≤ kappa →
      DistanceLaw.expSum S (fun k => (p k - p' k) / r k) r kappa = 0 := by
    intro k hk
    have hcurve := h k hk
    rw [DistanceLaw.expSum]
    rw [ensCurve, ensCurve, ← sub_eq_zero, ← Finset.sum_sub_distrib] at hcurve
    rw [← hcurve]
    exact Finset.sum_congr rfl fun d _ => by ring
  intro k hk
  have hamp := DistanceLaw.expSum_eq_zero hinj hzero k hk
  have hrk : r k ≠ 0 := ne_of_gt (hpos k hk)
  rcases div_eq_zero_iff.1 hamp with h1 | h2
  · linarith [sub_eq_zero.1 h1]
  · exact absurd h2 hrk

/-! ## 3. No single conformation imitates an ensemble -/

/-- **A single distance cannot reproduce a two-conformer ensemble.**  If a sequence separation
realises two distinct distances with positive weights, then for every candidate single distance
`R > 0` there is an ionic strength at which the single-distance model and the ensemble disagree.
Conformational heterogeneity is therefore falsifiable, and a complete titration falsifies any
single-structure model of the region. -/
theorem two_conformer_not_single_distance {r1 r2 p1 p2 R kappa0 : ℝ}
    (hr1 : 0 < r1) (hr2 : 0 < r2) (hR : 0 < R) (hne : r1 ≠ r2)
    (hp1 : 0 < p1) (hp2 : 0 < p2) :
    ∃ kappa, kappa0 ≤ kappa ∧
      p1 * (Real.exp (-(kappa * r1)) / r1) + p2 * (Real.exp (-(kappa * r2)) / r2)
        ≠ Real.exp (-(kappa * R)) / R := by
  by_contra hcon
  push_neg at hcon
  by_cases hR1 : R = r1
  · -- the single distance coincides with the first conformer's distance
    subst hR1
    have h : ∀ kappa, kappa0 ≤ kappa →
        (p1 / R - 1 / R) * Real.exp (-(kappa * R))
          + (p2 / r2) * Real.exp (-(kappa * r2)) = 0 := by
      intro k hk
      have := hcon k hk
      field_simp at this ⊢
      nlinarith [this]
    have := (two_exp_indep (by simpa using hne) h).2
    have hp2ne : p2 / r2 ≠ 0 := by positivity
    exact hp2ne this
  · by_cases hR2 : R = r2
    · subst hR2
      have h : ∀ kappa, kappa0 ≤ kappa →
          (p1 / r1) * Real.exp (-(kappa * r1))
            + (p2 / R - 1 / R) * Real.exp (-(kappa * R)) = 0 := by
        intro k hk
        have := hcon k hk
        field_simp at this ⊢
        nlinarith [this]
      have := (two_exp_indep hne h).1
      have hp1ne : p1 / r1 ≠ 0 := by positivity
      exact hp1ne this
    · -- all three distances are distinct
      have h : ∀ kappa, kappa0 ≤ kappa →
          (p1 / r1) * Real.exp (-(kappa * r1)) + (p2 / r2) * Real.exp (-(kappa * r2))
            + (-(1 / R)) * Real.exp (-(kappa * R)) = 0 := by
        intro k hk
        have := hcon k hk
        field_simp at this ⊢
        nlinarith [this]
      have := (three_exp_indep hne (Ne.symm hR1) (Ne.symm hR2) h).1
      have hp1ne : p1 / r1 ≠ 0 := by positivity
      exact hp1ne this

/-- **Fitting the mean distance is refuted too.**  The single-distance model placed at the mean
distance of a two-conformer ensemble is separated from the ensemble by some ionic strength: the
mean of the distance distribution is not what the experiment reports. -/
theorem mean_distance_model_falsified {r1 r2 w kappa0 : ℝ}
    (hr1 : 0 < r1) (hr2 : 0 < r2) (hne : r1 ≠ r2) (hw0 : 0 < w) (hw1 : w < 1) :
    ∃ kappa, kappa0 ≤ kappa ∧
      w * (Real.exp (-(kappa * r1)) / r1) + (1 - w) * (Real.exp (-(kappa * r2)) / r2)
        ≠ Real.exp (-(kappa * (w * r1 + (1 - w) * r2))) / (w * r1 + (1 - w) * r2) := by
  have hw2 : 0 < 1 - w := by linarith
  have hmean : 0 < w * r1 + (1 - w) * r2 := by positivity
  exact two_conformer_not_single_distance hr1 hr2 hmean hne hw0 hw2

/-! ## 3½. The support of the distribution need not be known either -/

/-- The weight function of a distance distribution, extended by zero off its support. -/
noncomputable def ext (T : Finset ℝ) (w : ℝ → ℝ) : ℝ → ℝ := fun t => if t ∈ T then w t else 0

/-- The titration reading of a distance distribution given as weights `w` on a finite set `T` of
distances — no index set, and no assumption that two candidate distributions share a support. -/
noncomputable def measCurve (T : Finset ℝ) (w : ℝ → ℝ) (kappa : ℝ) : ℝ :=
  ∑ t ∈ T, w t * (Real.exp (-(kappa * t)) / t)

/-- Exponentials indexed by their own (distinct, by construction) rates are independent: if a
finite sum `∑_{t ∈ T} A t · e^{−κt}` over a finite set of real rates vanishes at every condition on
a half-line, every amplitude vanishes. -/
theorem realRate_amplitudes_zero {T : Finset ℝ} {A : ℝ → ℝ} {kappa0 : ℝ}
    (h : ∀ kappa, kappa0 ≤ kappa → ∑ t ∈ T, A t * Real.exp (-(kappa * t)) = 0) :
    ∀ t ∈ T, A t = 0 := by
  classical
  set n : ℕ := T.card with hn
  set g : Fin n ≃ { x // x ∈ T } := T.equivFin.symm with hg
  set rate : ℕ → ℝ := fun j => if hj : j < n then ((g ⟨j, hj⟩ : { x // x ∈ T }) : ℝ) else 0
    with hrate
  set amp : ℕ → ℝ := fun j => if hj : j < n then A ((g ⟨j, hj⟩ : { x // x ∈ T }) : ℝ) else 0
    with hamp
  -- the sum over `T` is the sum over `range n` after reindexing
  have hreindex : ∀ (F : ℝ → ℝ), ∑ t ∈ T, F t
      = ∑ j ∈ Finset.range n, (if hj : j < n then F ((g ⟨j, hj⟩ : { x // x ∈ T }) : ℝ) else 0) := by
    intro F
    have e1 : ∑ t ∈ T, F t = ∑ i : Fin n, F ((g i : { x // x ∈ T }) : ℝ) := by
      rw [← Finset.sum_coe_sort T F, ← Equiv.sum_comp g (fun x : { x // x ∈ T } => F (x : ℝ))]
    have e2 : ∑ i : Fin n, F ((g i : { x // x ∈ T }) : ℝ)
        = ∑ j ∈ Finset.range n,
            (if hj : j < n then F ((g ⟨j, hj⟩ : { x // x ∈ T }) : ℝ) else 0) := by
      rw [← Fin.sum_univ_eq_sum_range (fun j => if hj : j < n then
        F ((g ⟨j, hj⟩ : { x // x ∈ T }) : ℝ) else 0) n]
      exact Finset.sum_congr rfl fun i _ => by simp
    rw [e1, e2]
  have hinj : Set.InjOn rate (Finset.range n) := by
    intro i hi j hj hij
    simp only [Finset.coe_range, Set.mem_Iio] at hi hj
    simp only [hrate, dif_pos hi, dif_pos hj] at hij
    have : (g ⟨i, hi⟩ : { x // x ∈ T }) = g ⟨j, hj⟩ := Subtype.ext hij
    have := g.injective this
    simpa using this
  have hzero : ∀ kappa, kappa0 ≤ kappa →
      DistanceLaw.expSum (Finset.range n) amp rate kappa = 0 := by
    intro k hk
    rw [DistanceLaw.expSum]
    have := h k hk
    rw [hreindex (fun t => A t * Real.exp (-(k * t)))] at this
    rw [← this]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [Finset.mem_range] at hj
    simp [hamp, hrate, dif_pos hj]
  have hall := DistanceLaw.expSum_eq_zero hinj hzero
  intro t ht
  obtain ⟨i, hi⟩ : ∃ i : Fin n, ((g i : { x // x ∈ T }) : ℝ) = t := by
    refine ⟨g.symm ⟨t, ht⟩, ?_⟩
    simp
  have := hall i.1 (Finset.mem_range.2 i.2)
  simp only [hamp, dif_pos i.2] at this
  rwa [show ((g ⟨i.1, i.2⟩ : { x // x ∈ T }) : ℝ) = t by simpa using hi] at this

/-- **The distance distribution is identified, support and all.**  Two finitely supported
distributions of positive internal distances whose titration curves agree at every ionic strength
on a half-line have the same weight at every distance — in particular the same support.  The
experiment does not need to be told which distances the ensemble realises. -/
theorem distance_distribution_identifiable {T T' : Finset ℝ} {w w' : ℝ → ℝ} {kappa0 : ℝ}
    (hT : ∀ t ∈ T, 0 < t) (hT' : ∀ t ∈ T', 0 < t)
    (h : ∀ kappa, kappa0 ≤ kappa → measCurve T w kappa = measCurve T' w' kappa) :
    ∀ t : ℝ, ext T w t = ext T' w' t := by
  classical
  set U : Finset ℝ := T ∪ T' with hU
  have hUpos : ∀ t ∈ U, 0 < t := by
    intro t htU
    rcases Finset.mem_union.1 htU with ht | ht
    · exact hT t ht
    · exact hT' t ht
  have hcurve : ∀ (S : Finset ℝ) (v : ℝ → ℝ), S ⊆ U → ∀ kappa,
      measCurve S v kappa = ∑ t ∈ U, ext S v t * (Real.exp (-(kappa * t)) / t) := by
    intro S v hSU kappa
    rw [measCurve]
    have hsub : ∑ t ∈ S, ext S v t * (Real.exp (-(kappa * t)) / t)
        = ∑ t ∈ U, ext S v t * (Real.exp (-(kappa * t)) / t) := by
      refine Finset.sum_subset hSU ?_
      intro x _ hx
      simp [ext, hx]
    rw [← hsub]
    exact Finset.sum_congr rfl fun t htS => by simp [ext, htS]
  have hamp : ∀ kappa, kappa0 ≤ kappa →
      ∑ t ∈ U, ((ext T w t - ext T' w' t) / t) * Real.exp (-(kappa * t)) = 0 := by
    intro k hk
    have h1 := hcurve T w Finset.subset_union_left k
    have h2 := hcurve T' w' Finset.subset_union_right k
    have h3 := h k hk
    rw [h1, h2] at h3
    have h4 : ∑ t ∈ U, (ext T w t * (Real.exp (-(k * t)) / t)
        - ext T' w' t * (Real.exp (-(k * t)) / t)) = 0 := by
      rw [Finset.sum_sub_distrib, h3, sub_self]
    rw [← h4]
    exact Finset.sum_congr rfl fun t _ => by ring
  have hzero := realRate_amplitudes_zero hamp
  intro t
  by_cases htU : t ∈ U
  · have ht0 : t ≠ 0 := ne_of_gt (hUpos t htU)
    have := hzero t htU
    rcases div_eq_zero_iff.1 this with h1 | h2
    · linarith [sub_eq_zero.1 h1]
    · exact absurd h2 ht0
  · have h1 : t ∉ T := fun hc => htU (Finset.mem_union_left _ hc)
    have h2 : t ∉ T' := fun hc => htU (Finset.mem_union_right _ hc)
    simp [ext, h1, h2]

/-! ## 4. The law -/

/-- **The ensemble law.**  At a fixed sequence separation, a complete salt titration determines
the weight of every candidate internal distance — the distance *distribution*, not a mean or an
apparent distance; and no single distance reproduces a genuinely heterogeneous ensemble, the mean
distance included; and the identification needs no prior knowledge of which distances the ensemble
realises.  A model of a disordered region must therefore predict, and be calibrated against, a
distribution of distances per separation. -/
theorem distance_ensemble_law {S : Finset ℕ} {r : ℕ → ℝ} {kappa0 : ℝ}
    (hinj : Set.InjOn r S) (hpos : ∀ k ∈ S, 0 < r k) :
    (∀ p p' : ℕ → ℝ,
        (∀ kappa, kappa0 ≤ kappa → ensCurve S r p kappa = ensCurve S r p' kappa) →
        ∀ k ∈ S, p k = p' k) ∧
    (∀ r1 r2 p1 p2 R : ℝ, 0 < r1 → 0 < r2 → 0 < R → r1 ≠ r2 → 0 < p1 → 0 < p2 →
        ∃ kappa, kappa0 ≤ kappa ∧
          p1 * (Real.exp (-(kappa * r1)) / r1) + p2 * (Real.exp (-(kappa * r2)) / r2)
            ≠ Real.exp (-(kappa * R)) / R) ∧
    (∀ (T T' : Finset ℝ) (w w' : ℝ → ℝ), (∀ t ∈ T, 0 < t) → (∀ t ∈ T', 0 < t) →
        (∀ kappa, kappa0 ≤ kappa → measCurve T w kappa = measCurve T' w' kappa) →
        ∀ t : ℝ, ext T w t = ext T' w' t) :=
  ⟨fun _ _ h => ens_weights_identifiable hinj hpos h,
   fun _ _ _ _ _ hr1 hr2 hR hne hp1 hp2 =>
     two_conformer_not_single_distance hr1 hr2 hR hne hp1 hp2,
   fun _ _ _ _ hT hT' h => distance_distribution_identifiable hT hT' h⟩

end DistanceEnsemble
end IDR
