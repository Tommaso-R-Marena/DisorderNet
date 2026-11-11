/-
# Part XVI capstone: the coupling design laws

Part XV situated the region -- in the cell, under force, in time.  Part XVI turns to the
thing a disordered region is *for*: coupling.  Two exact statements, one differential and
one geometric, constrain any model that is meant to predict function.

`coupling_design_laws` bundles two clauses.

1. **Linkage is reciprocal, and there is only one coupling constant.**  In the doubly
   tilted ensemble both partial responses are covariances in the same ensemble, so
   `∂⟨A⟩/∂mu = ∂⟨B⟩/∂lam` (`Linkage.linkage_reciprocity`); the coupling vanishes exactly
   when the partner does not discriminate between conformations
   (`Linkage.no_linkage_of_uniform_affinity`) and is strictly positive as soon as it does
   (`Linkage.linkage_of_comonotone`).  In finite form, on the folding/binding cycle, the
   ligand's stabilisation of the folded state *is* the folded state's enhancement of
   binding (`Linkage.thermodynamic_box`,
   `Linkage.folding_stabilization_eq_binding_enhancement`), independently of the intrinsic
   stability and affinity, and with favourable coupling the folded fraction strictly rises
   on saturation (`Linkage.apo_folded_fraction_lt_holo`).  So a model may not predict a
   conformational shift and an affinity change separately: they are the same parameter, and
   fitting one determines the other.
2. **The linker is part of the binding site.**  For the ideal three-dimensional tether the
   contact probability obeys the exact recursion `Tether.ret1_succ`, decreases strictly with
   length (`Tether.contactProb_strictAnti`) and obeys the quantitative two-sided `N^{-3/2}` law
   (`Tether.le_contactProb`, `Tether.contactProb_le`).  Hence effective concentration, and with
   it the apparent affinity of an otherwise identical motif, is strictly decreasing in length
   (`Tether.avidity_not_a_property_of_the_motif`): an affinity measured in one construct is
   not a property of the motif and does not transfer, and a model that predicts binding by a
   disordered region must model the disordered part.

Together with the earlier capstones: the object to be predicted is a situated conditional
distribution; its couplings to partners are single reciprocal numbers, not independent
fitting handles; and the disordered linker enters the prediction of function
quantitatively, through statistics that the model must get right.
-/
import Mathlib
import RequestProject.Linkage
import RequestProject.Tether

namespace IDR

/-- **The coupling design laws for a model of an intrinsically disordered region.**  Each
clause is an instance of a theorem proved in Part XVI. -/
theorem coupling_design_laws :
    -- (1) linkage: one reciprocal coupling constant, differential and finite forms
    ((∀ (n : ℕ) (q A B : Fin n → ℝ), 0 < n → (∀ j, 0 < q j) → ∀ lam mu : ℝ,
        deriv (fun m => Linkage.mean2 q A B A lam m) mu
          = deriv (fun l => Linkage.mean2 q A B B l mu) lam) ∧
      (∀ (n : ℕ) (q A B : Fin n → ℝ), 0 < n → (∀ j, 0 < q j) → ∀ c : ℝ, (∀ j, B j = c) →
        ∀ lam mu : ℝ, Linkage.cov2 q A B A B lam mu = 0) ∧
      (∀ (n : ℕ) (q A B : Fin n → ℝ), 0 < n → (∀ j, 0 < q j) → Crowding.Comonotone A B →
        ∀ i₀ j₀ : Fin n, 0 < (A i₀ - A j₀) * (B i₀ - B j₀) →
        ∀ lam mu : ℝ, 0 < Linkage.cov2 q A B A B lam mu) ∧
      (∀ eF eL w : ℝ,
        Linkage.boxP eF eL w 3 / Linkage.boxP eF eL w 2
            = Real.exp (-w) * (Linkage.boxP eF eL w 1 / Linkage.boxP eF eL w 0) ∧
          Linkage.boxP eF eL w 3 / Linkage.boxP eF eL w 1
            = Real.exp (-w) * (Linkage.boxP eF eL w 2 / Linkage.boxP eF eL w 0)) ∧
      (∀ eF eL w : ℝ, w < 0 →
        Linkage.boxP eF eL w 1 / (Linkage.boxP eF eL w 0 + Linkage.boxP eF eL w 1)
          < Linkage.boxP eF eL w 3 / (Linkage.boxP eF eL w 2 + Linkage.boxP eF eL w 3))) ∧
    -- (2) the tether: contact statistics set the affinity
    ((∀ m : ℕ, Tether.ret1 (m + 1) = Tether.ret1 m * ((2 * m + 1) / (2 * m + 2))) ∧
      StrictAnti Tether.contactProb ∧
      (∀ m : ℕ, 1 / Real.sqrt (4 * (m : ℝ) + 1) ^ 3 ≤ Tether.contactProb m
        ∧ Tether.contactProb m ≤ 1 / Real.sqrt (3 * (m : ℝ) + 1) ^ 3) ∧
      (∀ (Kintr vol : ℝ), 0 < Kintr → 0 < vol → ∀ m₁ m₂ : ℕ, m₁ < m₂ →
        Tether.apparentK Kintr vol m₂ < Tether.apparentK Kintr vol m₁)) := by
  refine ⟨⟨fun n q A B hn hq lam mu => Linkage.linkage_reciprocity hn hq lam mu,
      fun n q A B hn hq c hconst lam mu => Linkage.no_linkage_of_uniform_affinity hn hq hconst lam mu,
      fun n q A B hn hq hc i₀ j₀ hsep lam mu => Linkage.linkage_of_comonotone hn hq hc hsep lam mu,
      fun eF eL w => Linkage.folding_stabilization_eq_binding_enhancement eF eL w,
      fun eF eL w hw => Linkage.apo_folded_fraction_lt_holo hw⟩,
    ⟨fun m => Tether.ret1_succ m, Tether.contactProb_strictAnti,
      fun m => ⟨Tether.le_contactProb m, Tether.contactProb_le m⟩,
      fun Kintr vol hK hvol m₁ m₂ h => Tether.avidity_not_a_property_of_the_motif hK hvol h⟩⟩

end IDR
