/-
# Part XVII capstone: the sample and the proteoform

Two further gaps between what is measured and what is meant, both routine in practice and
both fatal to a model that ignores them.

`sample_design_laws` bundles two clauses.

1. **A reported ensemble belongs to a concentration.**  Solving mass action for a
   monomer--dimer equilibrium (`SelfAssociation.mass_action`) gives the monomer fraction
   `2/(1 + sqrt(1 + 8Kc))`, which is a genuine fraction (`monoFrac_pos`, `monoFrac_le_one`),
   equals one only at infinite dilution (`monoFrac_zero`), decreases *strictly* with total
   concentration at every concentration (`monoFrac_strictAnti`) and vanishes at high
   concentration (`monoFrac_tendsto_zero`).  Any observable that distinguishes monomer from
   dimer therefore drifts strictly with concentration
   (`measured_observable_depends_on_concentration`, `apparentObs_strictAnti`), and only the
   infinite-dilution value is a property of the molecule (`apparentObs_at_zero_eq_monomer`).
   A model trained on published ensembles inherits their concentrations unless these are
   modelled or extrapolated away.
2. **Single-site modification data do not add up.**  Two modifications tilt the ensemble
   jointly; their interaction free energy `Multisite.coupling` vanishes when one of them does
   not discriminate between conformations (`coupling_eq_zero_of_constant`), but for an explicit
   two-conformation region in which both favour the same conformation it is strictly positive
   (`coupling_pos_of_correlated`), so the joint effect is not the sum of the single effects
   (`multisite_effects_not_additive`).  For a multiply modifiable region the couplings are
   separate parameters -- and by Part XVI each of them is simultaneously the reciprocal
   coupling that governs the partner's affinity.
-/
import Mathlib
import RequestProject.SelfAssociation
import RequestProject.Multisite

namespace IDR

/-- **The sample design laws for a model of an intrinsically disordered region.**  Each
clause is an instance of a theorem proved in Part XVII. -/
theorem sample_design_laws :
    -- (1) the measured ensemble is a property of the sample, not of the molecule
    ((∀ K c : ℝ, 0 ≤ K → 0 ≤ c →
        SelfAssociation.monoConc K c + 2 * K * SelfAssociation.monoConc K c ^ 2 = c) ∧
      (∀ K c : ℝ, 0 ≤ K → 0 ≤ c →
        0 < SelfAssociation.monoFrac K c ∧ SelfAssociation.monoFrac K c ≤ 1) ∧
      (∀ K : ℝ, 0 < K → StrictAntiOn (SelfAssociation.monoFrac K) (Set.Ici 0)) ∧
      (∀ K : ℝ, 0 < K →
        Filter.Tendsto (SelfAssociation.monoFrac K) Filter.atTop (nhds 0)) ∧
      (∀ K xm xd : ℝ, 0 < K → xm ≠ xd → ∀ a b : ℝ, 0 ≤ a → 0 ≤ b → a < b →
        SelfAssociation.apparentObs K a xm xd ≠ SelfAssociation.apparentObs K b xm xd) ∧
      (∀ K xm xd : ℝ, SelfAssociation.apparentObs K 0 xm xd = xm)) ∧
    -- (2) modifications interact: single-site data do not determine the multiply modified state
    ((∀ (n : ℕ) (q A B : Fin n → ℝ), 0 < n → (∀ j, 0 < q j) → ∀ c : ℝ, (∀ j, B j = c) →
        ∀ lam mu : ℝ, Multisite.coupling q A B lam mu = 0) ∧
      0 < Multisite.coupling Multisite.twoQ Multisite.indic Multisite.indic 1 1 ∧
      (∃ (q A B : Fin 2 → ℝ) (lam mu : ℝ),
        (∀ j, 0 < q j) ∧
        Multisite.logPart q A B lam mu + Multisite.logPart q A B 0 0
          ≠ Multisite.logPart q A B lam 0 + Multisite.logPart q A B 0 mu)) := by
  refine ⟨⟨fun K c hK hc => SelfAssociation.mass_action hK hc,
      fun K c hK hc => ⟨SelfAssociation.monoFrac_pos hK hc, SelfAssociation.monoFrac_le_one hK hc⟩,
      fun K hK => SelfAssociation.monoFrac_strictAnti hK,
      fun K hK => SelfAssociation.monoFrac_tendsto_zero hK,
      fun K xm xd hK hne a b ha hb hab =>
        SelfAssociation.measured_observable_depends_on_concentration hK hne ha hb hab,
      fun K xm xd => SelfAssociation.apparentObs_at_zero_eq_monomer K xm xd⟩,
    ⟨fun n q A B hn hq c hconst lam mu => Multisite.coupling_eq_zero_of_constant hn hq hconst lam mu,
      Multisite.coupling_pos_of_correlated,
      Multisite.multisite_effects_not_additive⟩⟩

end IDR
