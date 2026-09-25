/-
# Part LXXXIII  Ion mobility: the arrival-time distribution is the datum

`RequestProject.IonMobility` treats an ion-mobility measurement the way the earlier parts treat
every other experiment on a disordered region: the trace is a *mixture*, one peak per conformer
family, and what a fit reports is the first two moments of that mixture.

`IDR.ion_mobility_laws` bundles five statements.

1. *Total variance.*  The measured squared width is the mean instrumental variance plus the
   conformational spread of the drift times, exactly; so the spread is identified precisely as the
   excess width over the instrument function, and the measured width never falls below it.
2. *One structure predicts the instrument.*  A single-conformation model predicts the instrumental
   width and nothing else, so any excess width whatsoever refutes it -- the ion-mobility analogue
   of a sub-stoichiometric crosslink yield.
3. *The centroid selects nothing.*  It is bracketed by the conformer drift times, and conversely
   every intermediate arrival time is reproduced exactly by a compact/extended two-state mixture
   with both weights strictly positive.
4. *Nor do the centroid and the width together.*  Two explicitly different ensembles share both.
5. *Resolving power.*  Two families are separated at criterion `k` iff `k(t₁+t₂) ≤ R|t₁−t₂|`, so a
   two-percent cross-section difference needs `R ≥ 101`; and `k` resolved peaks force at least `k`
   conformers in any model reproducing them.

The design conclusion is the standard one of this development, in a new instrument: an
ion-mobility measurement of a disordered region is a distribution, and a model of that region must
predict a mean *and* an excess width, not the single cross section that a single structure has.
-/
import Mathlib
import RequestProject.IonMobility

set_option autoImplicit false

namespace IDR

open Finset IDR.IonMob

/-- **The ion-mobility laws.**

1. the measured squared width splits exactly into instrumental variance plus conformational
   spread, the spread is the excess width, and the width is at least instrumental;
2. a single-conformation model predicts the instrumental width, so any excess refutes it;
3. the centroid is bracketed by the conformer drift times, and every intermediate value is hit by
   a two-state mixture with both weights in `(0,1)`;
4. mean and width together still do not determine the ensemble;
5. the resolving-power criterion, the two-percent instance, and the peak-counting bound. -/
theorem ion_mobility_laws {m : ℕ} :
    (∀ w t sig : Fin m → ℝ, (∀ j, 0 ≤ w j) → ∑ j, w j = 1 →
        width2 w t sig = instr w sig + spread w t ∧
        spread w t = width2 w t sig - instr w sig ∧
        instr w sig ≤ width2 w t sig) ∧
    (∀ t sig : Fin 1 → ℝ, width2 (fun _ => (1 : ℝ)) t sig = sig 0 ^ 2 ∧
        ∀ measured : ℝ, sig 0 ^ 2 < measured →
          width2 (fun _ => (1 : ℝ)) t sig ≠ measured) ∧
    ((∀ w t : Fin m → ℝ, (∀ j, 0 ≤ w j) → ∑ j, w j = 1 → ∀ lo hi : ℝ,
        (∀ j, lo ≤ t j) → (∀ j, t j ≤ hi) → lo ≤ meanTime w t ∧ meanTime w t ≤ hi) ∧
      (∀ a b x : ℝ, a < x → x < b →
        ∃ p : ℝ, 0 < p ∧ p < 1 ∧ meanTime ![p, 1 - p] ![a, b] = x)) ∧
    (let t : Fin 5 → ℝ := ![0, 1, 2, 3, 4]
     let wA : Fin 5 → ℝ := ![0, 1/2, 0, 1/2, 0]
     let wB : Fin 5 → ℝ := ![1/8, 0, 3/4, 0, 1/8]
     (∑ j, wA j = 1) ∧ (∑ j, wB j = 1) ∧ wA ≠ wB ∧
       meanTime wA t = meanTime wB t ∧ spread wA t = spread wB t) ∧
    ((∀ R k t₁ t₂ : ℝ, 0 < R → (resolvedAt R k t₁ t₂ ↔ k * (t₁ + t₂) ≤ R * |t₁ - t₂|)) ∧
      (∀ R : ℝ, 0 < R → (resolvedAt R 1 1 (51 / 50) ↔ 101 ≤ R)) ∧
      (∀ (k : ℕ) (t : Fin m → ℝ) (peaks : Fin k → ℝ), Function.Injective peaks →
        (∀ i, ∃ j, t j = peaks i) → k ≤ m)) :=
  ⟨fun w t sig hwpos hw =>
      ⟨width2_eq_instr_add_spread w t sig hw, spread_eq_width2_sub_instr w t sig hw,
        width2_ge_instr w t sig hwpos hw⟩,
    fun t sig => ⟨width2_single t sig, fun _ h => excess_width_refutes_single t sig _ h⟩,
    ⟨fun w t hwpos hw _ _ hlo hhi =>
        ⟨min_le_meanTime w t hwpos hw hlo, meanTime_le_max w t hwpos hw hhi⟩,
      fun _ _ _ hxa hxb => mixture_hits_intermediate hxa hxb⟩,
    moment_ambiguity,
    ⟨fun _ _ _ _ hR => resolved_iff hR, fun _ hR => resolved_two_percent hR,
      fun _ t peaks hinj hmem => card_ge_of_distinct_peaks t peaks hinj hmem⟩⟩

end IDR
