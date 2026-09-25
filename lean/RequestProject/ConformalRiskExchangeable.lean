/-
# Conformal risk control for exchangeable data: the guarantee in expectation

`ConformalRisk.lean` proves the *deterministic* core of conformal risk control: for monotone
losses bounded by `B` on a finite threshold grid,

  `∑_j L j (t̂_j) ≤ α·(n+1)`     (`crc_validity`)

where `t̂_j` is the threshold calibrated from the `n` items other than `j`.  That is the
combinatorial content; the statement a paper actually quotes is the probabilistic one:

  `𝔼[ L_test(t̂) ] ≤ α`

for *exchangeable* losses, with `t̂` calibrated on the calibration items only.  This file proves
that statement, `crc_expected_risk_le`, from the deterministic one.

The model is the honest one: the law of the whole loss table is a probability measure `μ` on
`LossTable n m = Fin (n+1) → Fin (m+1) → ℝ`, and *exchangeability* (`Exchangeable`) is invariance
of `μ` under relabelling the `n+1` items.  No other assumption on the data-generating process is
made: the guarantee is distribution-free.

The proof is exactly the informal one, made precise.

* `calib_relabel` — relabelling the items permutes the calibrated thresholds:
  `t̂_j(L ∘ π) = t̂_{π j}(L)`, because the leave-one-out sums are permuted.  Hence
  `realisedLoss_relabel`: the realised loss of item `j` in the relabelled table is the realised
  loss of item `π j` in the original.
* `measurable_realisedLoss` — the realised loss is a measurable function of the table.  The
  calibrated threshold takes finitely many values and each level set is a finite Boolean
  combination of the half-spaces `{L : (∑_{i ≠ j} L i t) + B ≤ α(n+1)}`, so it is measurable
  (`measurable_calib`).
* `integral_realisedLoss_eq` — exchangeability plus the two previous facts give that every item
  has the *same* expected realised loss.
* `crc_expected_risk_le` — averaging the deterministic inequality over the `n+1` items then gives
  `𝔼[L_test(t̂)] ≤ α`.

The companion negative result — that a *group-level* guarantee of this kind does not give a
per-element one, so that a panel can report `1 − α` per protein and deliver arbitrarily little per
residue — is `grouped_validity_underdetermines_element_coverage` in `ConformalRisk.lean`.
-/
import Mathlib
import RequestProject.ConformalRisk

set_option autoImplicit false

namespace IDR.ConformalRisk

open Finset MeasureTheory

/-! ## Loss tables and exchangeability -/

section Exchangeable

variable {n m : ℕ}

/-- A loss table: the loss of each of the `n+1` items at each grid point. -/
abbrev LossTable (n m : ℕ) : Type := Fin (n + 1) → Fin (m + 1) → ℝ

/-- Relabelling the items of a loss table. -/
def relabel (π : Equiv.Perm (Fin (n + 1))) (L : LossTable n m) : LossTable n m := fun i => L (π i)

/-- **Exchangeability**: the law of the loss table is invariant under relabelling the items.  This
is the only assumption conformal risk control makes about the data. -/
def Exchangeable (μ : Measure (LossTable n m)) : Prop :=
  ∀ π : Equiv.Perm (Fin (n + 1)), μ.map (relabel π) = μ

theorem measurable_relabel (π : Equiv.Perm (Fin (n + 1))) :
    Measurable (relabel π : LossTable n m → LossTable n m) := by
  refine measurable_pi_lambda _ (fun i => ?_)
  exact measurable_pi_apply (π i)

/-- The leave-one-out sums of a relabelled table are those of the original table, with the
held-out index moved by the permutation. -/
theorem sum_erase_relabel (π : Equiv.Perm (Fin (n + 1))) (L : LossTable n m)
    (j : Fin (n + 1)) (t : Fin (m + 1)) :
    (∑ i ∈ univ.erase j, relabel π L i t) = ∑ i ∈ univ.erase (π j), L i t := by
  classical
  have h1 : (∑ i ∈ univ.erase j, relabel π L i t)
      = (∑ i, L (π i) t) - L (π j) t := by
    rw [Finset.sum_erase_eq_sub (Finset.mem_univ j)]
    rfl
  have h2 : (∑ i ∈ univ.erase (π j), L i t) = (∑ i, L i t) - L (π j) t :=
    Finset.sum_erase_eq_sub (Finset.mem_univ (π j))
  have h3 : (∑ i, L (π i) t) = ∑ i, L i t := Equiv.sum_comp π (fun i => L i t)
  rw [h1, h2, h3]

theorem crcSet_relabel (α B : ℝ) (π : Equiv.Perm (Fin (n + 1))) (L : LossTable n m)
    (j : Fin (n + 1)) : crcSet α B (relabel π L) j = crcSet α B L (π j) := by
  classical
  ext t
  simp only [mem_crcSet, sum_erase_relabel]

/-- Relabelling the items permutes the calibrated thresholds. -/
theorem calib_relabel (α B : ℝ) (π : Equiv.Perm (Fin (n + 1))) (L : LossTable n m)
    (j : Fin (n + 1)) : calib α B (relabel π L) j = calib α B L (π j) := by
  classical
  unfold calib
  rw [crcSet_relabel]

/-- The loss actually realised by item `j` when the threshold is calibrated on the other `n`
items. -/
noncomputable def realisedLoss (α B : ℝ) (j : Fin (n + 1)) (L : LossTable n m) : ℝ :=
  L j (calib α B L j)

theorem realisedLoss_relabel (α B : ℝ) (π : Equiv.Perm (Fin (n + 1))) (L : LossTable n m)
    (j : Fin (n + 1)) : realisedLoss α B j (relabel π L) = realisedLoss α B (π j) L := by
  unfold realisedLoss
  rw [calib_relabel]
  rfl

/-! ## Measurability -/

/-- The set of loss tables for which grid point `t` is already safe for the leave-`j`-out risk. -/
def safeSet (α B : ℝ) (j : Fin (n + 1)) (t : Fin (m + 1)) : Set (LossTable n m) :=
  {L | (∑ i ∈ univ.erase j, L i t) + B ≤ α * (n + 1)}

theorem mem_safeSet {α B : ℝ} {j : Fin (n + 1)} {t : Fin (m + 1)} {L : LossTable n m} :
    L ∈ safeSet α B j t ↔ t ∈ crcSet α B L j := by
  classical
  rw [mem_crcSet]
  rfl

/-- The half-space in which grid point `t` is already safe for the leave-`j`-out risk. -/
theorem measurableSet_safe (α B : ℝ) (j : Fin (n + 1)) (t : Fin (m + 1)) :
    MeasurableSet (safeSet α B j t) := by
  classical
  have hmeas : Measurable fun L : LossTable n m => (∑ i ∈ univ.erase j, L i t) + B := by
    refine Measurable.add_const ?_ B
    exact Finset.measurable_sum _ (fun i _ => (measurable_pi_apply i).eval)
  exact measurableSet_le hmeas measurable_const

/-- When the calibrated threshold equals a given grid point: either that point is the first safe
one, or nothing is safe and it is the last grid point. -/
theorem calib_eq_iff (α B : ℝ) (j : Fin (n + 1)) (L : LossTable n m) (t : Fin (m + 1)) :
    calib α B L j = t ↔
      ((t ∈ crcSet α B L j ∧ ∀ t' < t, t' ∉ crcSet α B L j)
        ∨ (t = Fin.last m ∧ ∀ t', t' ∉ crcSet α B L j)) := by
  classical
  by_cases hne : (crcSet α B L j).Nonempty
  · have hcal : calib α B L j = (crcSet α B L j).min' hne := by
      unfold calib; rw [dif_pos hne]
    obtain ⟨t₀, ht₀⟩ := hne
    constructor
    · intro h
      left
      refine ⟨by rw [← h, hcal]; exact Finset.min'_mem _ _, ?_⟩
      intro t' ht' hmem
      have : (crcSet α B L j).min' ⟨t₀, ht₀⟩ ≤ t' := Finset.min'_le _ _ hmem
      rw [← hcal, h] at this
      exact absurd ht' (not_lt.mpr this)
    · rintro (⟨hmem, hmin⟩ | ⟨-, hall⟩)
      · rw [hcal]
        refine le_antisymm (Finset.min'_le _ _ hmem) ?_
        by_contra hlt
        push_neg at hlt
        exact hmin _ hlt (Finset.min'_mem _ _)
      · exact absurd ht₀ (hall t₀)
  · have hcal : calib α B L j = Fin.last m := by
      unfold calib; rw [dif_neg hne]
    have hall : ∀ t', t' ∉ crcSet α B L j := by
      intro t' ht'
      exact hne ⟨t', ht'⟩
    constructor
    · intro h; exact Or.inr ⟨by rw [← h, hcal], hall⟩
    · rintro (⟨hmem, -⟩ | ⟨htl, -⟩)
      · exact absurd hmem (hall t)
      · rw [hcal, htl]

/-- The calibrated threshold is a measurable function of the loss table: it takes finitely many
values, and each level set is a finite Boolean combination of half-spaces. -/
theorem measurable_calib (α B : ℝ) (j : Fin (n + 1)) :
    Measurable fun L : LossTable n m => calib α B L j := by
  classical
  refine measurable_to_countable' (fun t => ?_)
  have hset : (fun L : LossTable n m => calib α B L j) ⁻¹' {t}
      = ((safeSet α B j t ∩ ⋂ t' : {t' : Fin (m + 1) // t' < t}, (safeSet α B j t'.1)ᶜ)
        ∪ (if t = Fin.last m then ⋂ t' : Fin (m + 1), (safeSet α B j t')ᶜ else ∅)) := by
    ext L
    have hcal := calib_eq_iff α B j L t
    constructor
    · intro hmem
      have h := hcal.mp hmem
      rcases h with ⟨hmem', hmin⟩ | ⟨htl, hall⟩
      · refine Or.inl ⟨mem_safeSet.mpr hmem', ?_⟩
        exact Set.mem_iInter.mpr (fun t' => fun hs => hmin t'.1 t'.2 (mem_safeSet.mp hs))
      · refine Or.inr ?_
        rw [if_pos htl]
        exact Set.mem_iInter.mpr (fun t' => fun hs => hall t' (mem_safeSet.mp hs))
    · intro hmem
      refine hcal.mpr ?_
      rcases hmem with ⟨hs, hint⟩ | hmem
      · refine Or.inl ⟨mem_safeSet.mp hs, ?_⟩
        intro t' ht' hmem'
        exact (Set.mem_iInter.mp hint ⟨t', ht'⟩) (mem_safeSet.mpr hmem')
      · by_cases htl : t = Fin.last m
        · rw [if_pos htl] at hmem
          exact Or.inr ⟨htl, fun t' hmem' =>
            (Set.mem_iInter.mp hmem t') (mem_safeSet.mpr hmem')⟩
        · rw [if_neg htl] at hmem
          exact absurd hmem (Set.notMem_empty L)
  rw [hset]
  refine MeasurableSet.union ?_ ?_
  · refine MeasurableSet.inter (measurableSet_safe α B j t) ?_
    exact MeasurableSet.iInter (fun t' => (measurableSet_safe α B j t'.1).compl)
  · split
    · exact MeasurableSet.iInter (fun t' => (measurableSet_safe α B j t').compl)
    · exact MeasurableSet.empty

theorem measurable_realisedLoss (α B : ℝ) (j : Fin (n + 1)) :
    Measurable (realisedLoss α B j : LossTable n m → ℝ) := by
  classical
  have hsum : (realisedLoss α B j : LossTable n m → ℝ)
      = fun L => ∑ t : Fin (m + 1), if calib α B L j = t then L j t else 0 := by
    funext L
    simp [realisedLoss, Finset.sum_ite_eq]
  rw [hsum]
  refine Finset.measurable_sum _ (fun t _ => ?_)
  refine Measurable.ite ?_ ((measurable_pi_apply j).eval) measurable_const
  exact (measurable_calib α B j) (measurableSet_singleton t)

/-! ## The guarantee -/

variable {α B : ℝ} {μ : Measure (LossTable n m)}

theorem integrable_realisedLoss [IsProbabilityMeasure μ] (j : Fin (n + 1))
    (hbdd : ∀ᵐ L ∂μ, ∀ i t, |L i t| ≤ B) :
    Integrable (realisedLoss α B j) μ := by
  refine Integrable.mono' (integrable_const B) (measurable_realisedLoss α B j).aestronglyMeasurable
    ?_
  filter_upwards [hbdd] with L hL
  simpa [realisedLoss, Real.norm_eq_abs] using hL j (calib α B L j)

/-- Under exchangeability every item has the same expected realised loss. -/
theorem integral_realisedLoss_eq [IsProbabilityMeasure μ] (hexch : Exchangeable μ)
    (j j' : Fin (n + 1)) :
    ∫ L, realisedLoss α B j L ∂μ = ∫ L, realisedLoss α B j' L ∂μ := by
  classical
  set π : Equiv.Perm (Fin (n + 1)) := Equiv.swap j j' with hπ
  have hmap : μ.map (relabel π) = μ := hexch π
  have h1 : ∫ L, realisedLoss α B j L ∂μ = ∫ L, realisedLoss α B j L ∂(μ.map (relabel π)) := by
    rw [hmap]
  have h2 : ∫ L, realisedLoss α B j L ∂(μ.map (relabel π))
      = ∫ L, realisedLoss α B j (relabel π L) ∂μ :=
    integral_map (measurable_relabel π).aemeasurable
      (measurable_realisedLoss α B j).aestronglyMeasurable
  have h3 : ∀ L : LossTable n m, realisedLoss α B j (relabel π L) = realisedLoss α B j' L := by
    intro L
    rw [realisedLoss_relabel]
    simp [hπ]
  rw [h1, h2]
  exact integral_congr_ae (Filter.Eventually.of_forall h3)

/-- **Conformal risk control is valid for exchangeable data.**

If the loss table of the `n+1` items is exchangeable, each item's loss is non-increasing in the
threshold and bounded by `B` in absolute value, and the last grid point is safe, then the expected
loss of the held-out (test) item at the threshold calibrated on the other `n` items is at most the
target `α`.  Nothing is assumed about the data-generating distribution beyond exchangeability. -/
theorem crc_expected_risk_le [IsProbabilityMeasure μ] (hexch : Exchangeable μ)
    (hbdd : ∀ᵐ L ∂μ, ∀ i t, |L i t| ≤ B)
    (hanti : ∀ᵐ L ∂μ, ∀ i, Antitone (L i))
    (hlast : ∀ᵐ L ∂μ, ∀ i, L i (Fin.last m) ≤ α) :
    ∫ L, realisedLoss α B (Fin.last n) L ∂μ ≤ α := by
  classical
  have hint : ∀ j : Fin (n + 1), Integrable (realisedLoss α B j) μ := fun j =>
    integrable_realisedLoss j hbdd
  have hsum_int : Integrable (fun L => ∑ j, realisedLoss α B j L) μ :=
    integrable_finset_sum _ (fun j _ => hint j)
  have hae : ∀ᵐ L ∂μ, (∑ j, realisedLoss α B j L) ≤ α * (n + 1) := by
    filter_upwards [hbdd, hanti, hlast] with L hb ha hl
    exact crc_validity α B L ha (fun i t => (abs_le.mp (hb i t)).2) hl
  have hbound : ∫ L, (∑ j, realisedLoss α B j L) ∂μ ≤ α * (n + 1) := by
    calc ∫ L, (∑ j, realisedLoss α B j L) ∂μ ≤ ∫ _L, α * (n + 1) ∂μ :=
          integral_mono_ae hsum_int (integrable_const _) hae
      _ = α * (n + 1) := by simp
  have hsplit : ∫ L, (∑ j, realisedLoss α B j L) ∂μ
      = ∑ j, ∫ L, realisedLoss α B j L ∂μ :=
    integral_finset_sum _ (fun j _ => hint j)
  have hconst : ∑ j : Fin (n + 1), ∫ L, realisedLoss α B j L ∂μ
      = (n + 1) * ∫ L, realisedLoss α B (Fin.last n) L ∂μ := by
    rw [Finset.sum_congr rfl (fun j _ =>
      integral_realisedLoss_eq (α := α) (B := B) hexch j (Fin.last n))]
    simp [Finset.sum_const]
  rw [hsplit, hconst] at hbound
  have hpos : (0:ℝ) < (n + 1) := by positivity
  nlinarith [hbound]

/-- A sanity check that the hypotheses of `crc_expected_risk_le` are satisfiable and the
conclusion is not vacuous: the point mass at the identically zero loss table is exchangeable,
bounded, antitone and safe at the last grid point, and the guarantee holds for it. -/
theorem crc_expected_risk_le_dirac_zero {α : ℝ} (hα : 0 ≤ α) :
    Exchangeable (Measure.dirac (0 : LossTable n m)) ∧
      ∫ L, realisedLoss α 1 (Fin.last n) L ∂(Measure.dirac (0 : LossTable n m)) ≤ α := by
  classical
  have hexch : Exchangeable (Measure.dirac (0 : LossTable n m)) := by
    intro π
    rw [Measure.map_dirac (measurable_relabel π)]
    rfl
  have hae0 : ∀ᵐ (L : LossTable n m) ∂(Measure.dirac (0 : LossTable n m)), L = 0 := by
    rw [MeasureTheory.ae_dirac_eq]
    exact Filter.eventually_pure.mpr rfl
  refine ⟨hexch, ?_⟩
  refine crc_expected_risk_le hexch ?_ ?_ ?_
  · filter_upwards [hae0] with L hL
    intro i t
    simp [hL]
  · filter_upwards [hae0] with L hL
    intro i
    subst hL
    exact fun a b _ => le_refl (0:ℝ)
  · filter_upwards [hae0] with L hL
    intro i
    simpa [hL] using hα

end Exchangeable

end IDR.ConformalRisk
