/-
# Coarse-graining and mixing cannot manufacture structural accuracy

Two structural laws for the transport distance of `RequestProject.TransportGeometry`, both
about operations that ensemble modelling performs constantly.

* **Coarse-graining.**  Models are compared with experiment through descriptors: a radius
  of gyration, a contact map, a secondary-structure propensity, a Cα-only trace.  Each is a
  map `h : X → Y`, and comparison happens between the push-forwards `E.map h`.
  `transportCost_map_le` is the data-processing inequality: if `h` is `L`-Lipschitz then
  the descriptor-space error is at most `L` times the structural error, so *no descriptor
  can make a model look worse than it is*, and (contrapositively, `transportCost_ge_of_map`)
  a discrepancy seen in any descriptor is a certificate against the full ensemble --
  divided by `L`, and never more.  `transportCost_map_eq_of_isometry` says nothing is lost
  when the descriptor is isometric: relabelling conformations changes nothing.
* **Mixing.**  Disorder models are built by combining sub-ensembles -- states of a
  conformational equilibrium, replicas, clusters of a trajectory.  `transportCost_mix_le`
  is joint convexity: matching sub-ensembles pairwise and averaging the errors is always at
  least as good as the true error of the mixture.  Errors of components therefore never
  amplify when the components are combined, and a per-state error budget is a valid budget
  for the whole model.

Both are proved by exhibiting explicit transport plans, so both come with the matching
plan, not merely with the inequality.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.CoarseGraining
import RequestProject.ModelNature
import RequestProject.Transport
import RequestProject.TransportGeometry

namespace IDR

open Finset
open scoped Classical

variable {X Y : Type*}

/-! ## Coarse-graining: the data-processing inequality -/

/-- A transport plan between two ensembles is also a transport plan between their
push-forwards under any descriptor: the index sets and weights are unchanged. -/
lemma isCoupling_map {E F : Ens X} (h : X → Y) {g : Fin E.card → Fin F.card → ℝ}
    (hg : IsCoupling E F g) :
    IsCoupling (E.map h) (F.map h) (fun i j => g i j) := hg

/-- **Data processing.**  A descriptor that is `L`-Lipschitz for the structural metric can
only shrink the transport distance by a factor `L`: coarse comparison never overstates the
structural error of an ensemble model. -/
theorem transportCost_map_le {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) {d : Y → Y → ℝ}
    (hd : ∀ y y', 0 ≤ d y y') {L : ℝ} {h : X → Y} (hL : ∀ x x', d (h x) (h x') ≤ L * c x x')
    (E F : Ens X) :
    transportCost d (E.map h) (F.map h) ≤ L * transportCost c E F := by
  obtain ⟨g, hg, hgc⟩ := exists_optimal_coupling hc E F
  refine (transportCost_le_of_coupling hd (isCoupling_map h hg)).trans ?_
  rw [hgc, planCost, planCost, Finset.mul_sum]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun j _ => ?_
  calc g i j * d ((E.map h).pt i) ((F.map h).pt j)
      = g i j * d (h (E.pt i)) (h (F.pt j)) := rfl
    _ ≤ g i j * (L * c (E.pt i) (F.pt j)) :=
        mul_le_mul_of_nonneg_left (hL _ _) (hg.nonneg i j)
    _ = L * (g i j * c (E.pt i) (F.pt j)) := by ring

/-- Read backwards: a discrepancy detected in a coarse descriptor certifies structural
error in the full conformational ensemble. -/
theorem transportCost_ge_of_map {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) {d : Y → Y → ℝ}
    (hd : ∀ y y', 0 ≤ d y y') {L : ℝ} (hLpos : 0 < L) {h : X → Y}
    (hL : ∀ x x', d (h x) (h x') ≤ L * c x x') (E F : Ens X) :
    transportCost d (E.map h) (F.map h) / L ≤ transportCost c E F :=
  (div_le_iff₀ hLpos).2 <| by
    rw [mul_comm]
    exact transportCost_map_le hc hd hL E F

/-- An isometric descriptor loses nothing: the transport distance is unchanged. -/
theorem transportCost_map_eq_of_isometry {c : X → X → ℝ} {d : Y → Y → ℝ} {h : X → Y}
    (hiso : ∀ x x', d (h x) (h x') = c x x') (E F : Ens X) :
    transportCost d (E.map h) (F.map h) = transportCost c E F := by
  have hset : {r : ℝ | ∃ gam : Fin (E.map h).card → Fin (F.map h).card → ℝ,
      IsCoupling (E.map h) (F.map h) gam ∧ r = planCost (E.map h) (F.map h) d gam}
      = {r : ℝ | ∃ gam : Fin E.card → Fin F.card → ℝ,
        IsCoupling E F gam ∧ r = planCost E F c gam} := by
    ext r
    have hcost : ∀ gam : Fin E.card → Fin F.card → ℝ,
        planCost (E.map h) (F.map h) d gam = planCost E F c gam := by
      intro gam
      simp only [planCost, Ens.map]
      exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by rw [hiso]
    constructor
    · rintro ⟨gam, hgam, rfl⟩
      exact ⟨gam, hgam, (hcost gam).symm ▸ rfl⟩
    · rintro ⟨gam, hgam, rfl⟩
      exact ⟨gam, hgam, (hcost gam).symm⟩
  rw [transportCost, transportCost, hset]

/-! ## Mixing: joint convexity -/

/-- The block-diagonal plan: transport each component of the mixture to the corresponding
component of the other mixture. -/
noncomputable def blockPlan {E₁ E₂ F₁ F₂ : Ens X} (t : ℝ)
    (g₁ : Fin E₁.card → Fin F₁.card → ℝ) (g₂ : Fin E₂.card → Fin F₂.card → ℝ) :
    Fin (E₁.card + E₂.card) → Fin (F₁.card + F₂.card) → ℝ :=
  fun i j =>
    Fin.addCases
      (fun i₁ => Fin.addCases (fun j₁ => t * g₁ i₁ j₁) (fun _ => (0 : ℝ)) j)
      (fun i₂ => Fin.addCases (fun _ => (0 : ℝ)) (fun j₂ => (1 - t) * g₂ i₂ j₂) j) i

lemma blockPlan_isCoupling {E₁ E₂ F₁ F₂ : Ens X} {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1)
    {g₁ : Fin E₁.card → Fin F₁.card → ℝ} {g₂ : Fin E₂.card → Fin F₂.card → ℝ}
    (h₁ : IsCoupling E₁ F₁ g₁) (h₂ : IsCoupling E₂ F₂ g₂) :
    IsCoupling (Ens.mix E₁ E₂ t ht0 ht1) (Ens.mix F₁ F₂ t ht0 ht1) (blockPlan t g₁ g₂) := by
  refine ⟨fun i j => ?_, fun i => ?_, fun j => ?_⟩
  · induction i using Fin.addCases with
    | left i₁ =>
        induction j using Fin.addCases with
        | left j₁ => simpa [blockPlan] using mul_nonneg ht0 (h₁.nonneg i₁ j₁)
        | right j₂ => simp [blockPlan]
    | right i₂ =>
        induction j using Fin.addCases with
        | left j₁ => simp [blockPlan]
        | right j₂ =>
            simpa [blockPlan] using mul_nonneg (by linarith) (h₂.nonneg i₂ j₂)
  · induction i using Fin.addCases with
    | left i₁ =>
        simp only [Ens.mix]
        rw [Fin.sum_univ_add]
        simp only [blockPlan, Fin.addCases_left, Fin.addCases_right]
        rw [← Finset.mul_sum, h₁.row i₁]
        simp
    | right i₂ =>
        simp only [Ens.mix]
        rw [Fin.sum_univ_add]
        simp only [blockPlan, Fin.addCases_left, Fin.addCases_right]
        rw [← Finset.mul_sum, h₂.row i₂]
        simp
  · induction j using Fin.addCases with
    | left j₁ =>
        simp only [Ens.mix]
        rw [Fin.sum_univ_add]
        simp only [blockPlan, Fin.addCases_left, Fin.addCases_right]
        rw [← Finset.mul_sum, h₁.col j₁]
        simp
    | right j₂ =>
        simp only [Ens.mix]
        rw [Fin.sum_univ_add]
        simp only [blockPlan, Fin.addCases_left, Fin.addCases_right]
        rw [← Finset.mul_sum, h₂.col j₂]
        simp

lemma planCost_blockPlan {c : X → X → ℝ} {E₁ E₂ F₁ F₂ : Ens X} {t : ℝ} (ht0 : 0 ≤ t)
    (ht1 : t ≤ 1) (g₁ : Fin E₁.card → Fin F₁.card → ℝ) (g₂ : Fin E₂.card → Fin F₂.card → ℝ) :
    planCost (Ens.mix E₁ E₂ t ht0 ht1) (Ens.mix F₁ F₂ t ht0 ht1) c (blockPlan t g₁ g₂)
      = t * planCost E₁ F₁ c g₁ + (1 - t) * planCost E₂ F₂ c g₂ := by
  simp only [planCost, Ens.mix]
  rw [Fin.sum_univ_add]
  congr 1
  · rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i₁ _ => ?_
    rw [Fin.sum_univ_add]
    simp only [blockPlan, Fin.addCases_left, Fin.addCases_right, Fin.append_left,
      Fin.append_right]
    rw [Finset.mul_sum]
    simp [mul_assoc]
  · rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i₂ _ => ?_
    rw [Fin.sum_univ_add]
    simp only [blockPlan, Fin.addCases_left, Fin.addCases_right, Fin.append_left,
      Fin.append_right]
    rw [Finset.mul_sum]
    simp [mul_assoc]

/-- **Joint convexity of the structural error.**  If each sub-ensemble of a model is within
`ε_i` of the corresponding true sub-ensemble, the mixture is within the weighted average of
the `ε_i`.  Combining states, clusters or replicas therefore never amplifies structural
error, and a per-state error budget is a valid budget for the assembled model. -/
theorem transportCost_mix_le {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) {E₁ E₂ F₁ F₂ : Ens X}
    {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    transportCost c (Ens.mix E₁ E₂ t ht0 ht1) (Ens.mix F₁ F₂ t ht0 ht1)
      ≤ t * transportCost c E₁ F₁ + (1 - t) * transportCost c E₂ F₂ := by
  obtain ⟨g₁, h₁, hc₁⟩ := exists_optimal_coupling hc E₁ F₁
  obtain ⟨g₂, h₂, hc₂⟩ := exists_optimal_coupling hc E₂ F₂
  rw [hc₁, hc₂, ← planCost_blockPlan ht0 ht1 g₁ g₂]
  exact transportCost_le_of_coupling hc (blockPlan_isCoupling ht0 ht1 h₁ h₂)

end IDR
