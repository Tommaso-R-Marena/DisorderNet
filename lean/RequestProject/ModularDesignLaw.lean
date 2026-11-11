/-
# Modular ensembles IV: the design law

One statement collecting what the previous three files prove about building a full-length
disordered ensemble out of fragment models.

`modular_design_law` — for any ensemble of a chain cut into (segment, seam, segment):

1. the glued model reproduces **both** fragment panels exactly, so no measurement confined
   to one fragment can ever reveal that it is wrong;
2. it is exactly right precisely when the two segments are conditionally independent given
   the seam;
3. its relative entropy from the truth is exactly the conditional mutual information across
   the seam, which is nonnegative and vanishes only in that case;
4. **no** modular model — no strictly positive ensemble conditionally independent across the
   seam, however it was fitted — has smaller relative entropy;
5. a cut whose seam information is below `eps²/2` gives population accuracy `eps`.

`modular_design_law_is_not_vacuous` — the price is not a technicality: for the long-range
contact of `RequestProject.ModularCoupling` the glued model reproduces both fragment panels
exactly, is at the maximal population distance `1` from the truth, halves the measured
end-to-end contact probability, and every modular model is at least `log 2` from the truth.
-/
import Mathlib
import RequestProject.ModularGluing
import RequestProject.ModularOptimality
import RequestProject.ModularCoupling

namespace RequestProject.Modular

open Finset IDR.Pinsker

/-- **The modular design law for ensembles of disordered chains.** -/
theorem modular_design_law {X Y Z : Type*} [Fintype X] [Fintype Y] [Fintype Z]
    [Nonempty X] [Nonempty Y] [Nonempty Z] {p : X → Y → Z → ℝ}
    (hp : ∀ x y z, 0 < p x y z) (hs : ∑ x, ∑ y, ∑ z, p x y z = 1) :
    -- 1. the modular model reproduces the data it was built from
    ((∀ x y, ∑ z, glue p x y z = margXY p x y) ∧ (∀ y z, ∑ x, glue p x y z = margYZ p y z)) ∧
    -- 2. exact precisely under conditional independence across the seam
    ((∀ x y z, glue p x y z = p x y z) ↔ CondIndep p) ∧
    -- 3. the price of modularity is the seam information
    (klG (fun t : X × Y × Z => p t.1 t.2.1 t.2.2)
        (fun t : X × Y × Z => glue p t.1 t.2.1 t.2.2) = cmi p
      ∧ 0 ≤ cmi p ∧ (cmi p = 0 ↔ CondIndep p)) ∧
    -- 4. no modular model at all does better
    (∀ q : X → Y → Z → ℝ, (∀ x y z, 0 < q x y z) → (∑ x, ∑ y, ∑ z, q x y z = 1) →
      CondIndep q → cmi p ≤ ∑ x, ∑ y, ∑ z, p x y z * Real.log (p x y z / q x y z)) ∧
    -- 5. the design rule for choosing where to cut
    (∀ eps : ℝ, 0 ≤ eps → cmi p ≤ eps ^ 2 / 2 →
      ∑ t : X × Y × Z, |p t.1 t.2.1 t.2.2 - glue p t.1 t.2.1 t.2.2| ≤ eps) := by
  have hp' : ∀ x y z, 0 ≤ p x y z := fun x y z => le_of_lt (hp x y z)
  refine ⟨⟨fun x y => glue_margXY hp' x y, fun y z => glue_margYZ hp' y z⟩,
    glue_eq_self_iff_condIndep hp',
    ⟨klG_glue_eq_cmi hp', cmi_nonneg hp' hs, cmi_eq_zero_iff_condIndep hp' hs⟩,
    fun q hq hqs hci => no_modular_model_beats_seam_information hp' hq hs hqs hci,
    fun eps heps hcut => seam_design_rule hp hs heps hcut⟩

/-- **The law has teeth.**  A long-range contact between the two termini, with the seam in a
single state, is invisible to both fragment measurements yet defeats every modular model. -/
theorem modular_design_law_is_not_vacuous :
    (∀ x y, margXY longRange x y = 1/2) ∧ (∀ y z, margYZ longRange y z = 1/2) ∧
    (∀ x y z, glue longRange x y z = 1/4) ∧
    (∑ x, ∑ y, ∑ z, |longRange x y z - glue longRange x y z| = 1) ∧
    (contactProb longRange = 1 ∧ contactProb (glue longRange) = 1/2) ∧
    cmi longRange = Real.log 2 ∧
    (∀ q : Bool → Unit → Bool → ℝ, (∀ x y z, 0 < q x y z) →
      (∑ x, ∑ y, ∑ z, q x y z = 1) → CondIndep q →
      Real.log 2 ≤ ∑ x, ∑ y, ∑ z, longRange x y z * Real.log (longRange x y z / q x y z)) :=
  ⟨longRange_margXY, longRange_margYZ, longRange_glue, longRange_l1, longRange_contact,
    longRange_cmi, fun _ hq hqs hci => longRange_no_modular_model_is_close hq hqs hci⟩

end RequestProject.Modular
