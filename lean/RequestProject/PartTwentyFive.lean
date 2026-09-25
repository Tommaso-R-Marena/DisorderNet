/-
# Part XXV  What the class-I model still assumes, and what each assumption costs

Part XXIV builds the molecular model: continuous space, a class-I additive force field,
implicit solvent, and the Boltzmann-Gibbs measure.  Four idealisations remain inside it,
and this part removes each one and prices it exactly rather than leaving it in a list of
caveats.

* `RequestProject.ManyBody`         -- pairwise additivity of the solvent-averaged energy;
* `RequestProject.Protonation`      -- fixed partial charges (pH and charge regulation);
* `RequestProject.BrokenErgodicity` -- the identification of a finite run with the
                                       equilibrium ensemble;
* `RequestProject.Constraints`      -- rigid constraints on fast degrees of freedom
                                       (the Fixman factor).

`IDR.residual_assumption_laws` bundles the four statements.  In each case the conclusion is
the same in form: the idealisation is not a small error to be absorbed by refitting, it is a
*representability* or *identification* failure with an exact, computable size, and a model
that keeps the idealisation must say so.
-/
import Mathlib
import RequestProject.ManyBody
import RequestProject.Protonation
import RequestProject.BrokenErgodicity
import RequestProject.Constraints

namespace IDR

open ManyBody Protonation BrokenErgodicity Constraints

/-- **The design laws of the residual assumptions.**

1. *Pairwise additivity is a representability assumption.*  A cooperative three-body
   potential of mean force -- the generic shape of hydrophobic and desolvation cooperativity
   -- is not a sum of pair terms, no matter how the pair terms are fitted, and any additive
   surrogate errs by at least a quarter of the mixed second difference somewhere on the
   quadruple of configurations that exposes it.
2. *Fixed charges are a one-pH model.*  The mean charge of a titratable region is strictly
   decreasing in pH, and no constant reproduces it, so pH belongs in the context alongside
   temperature, ionic strength and partner concentration.  The linkage cycle closes exactly:
   a predicted pKa shift on binding *is* a predicted pH dependence of the affinity.
3. *A finite run is not the equilibrium ensemble.*  The two-state kinetics of a slow
   isomerisation is solved exactly; for every horizon and tolerance there is a barrier for
   which the run -- and its whole-trajectory average -- still reports essentially the initial
   population rather than the Boltzmann one.
4. *Rigid constraints change the distribution.*  Constraining a stiff coordinate does not
   give the stiff limit of the unconstrained ensemble: the relative weights differ by exactly
   the Fixman factor `sqrt (w q2 / w q1)`, for every stiffness, and agree only where the
   stiffness is constant. -/
theorem residual_assumption_laws :
    -- 1  pairwise additivity cannot represent cooperativity
    (¬ IsPairwiseAdditive cooperative ∧
      (∀ (W : PMF3) (a a' b b' c : ℝ), IsPairwiseAdditive W → mixed W a a' b b' c = 0) ∧
      (∀ (W V : PMF3) (a a' b b' c : ℝ), IsPairwiseAdditive V →
        |mixed W a a' b b' c| / 4 ≤
          max (max |W a' b' c - V a' b' c| |W a' b c - V a' b c|)
            (max |W a b' c - V a b' c| |W a b c - V a b c|))) ∧
    -- 2  pH is a coordinate of the model
    ((∀ pKa pH pH' : ℝ, pH < pH' →
        protonatedFraction pKa pH' < protonatedFraction pKa pH) ∧
      (∀ pKa q : ℝ, ∃ pH : ℝ, acidCharge pKa pH ≠ q) ∧
      (∀ G : Bool → Bool → ℝ,
        (G true true - G false true) - (G true false - G false false)
          = (G true true - G true false) - (G false true - G false false))) ∧
    -- 3  a finite run of a slow degree of freedom is not the Boltzmann ensemble
    ((∀ (k1 k2 p0 : ℝ), 0 < k1 + k2 →
        Filter.Tendsto (popB k1 k2 p0) Filter.atTop (nhds (peq k1 k2))) ∧
      (∀ T eps : ℝ, 0 < T → 0 < eps → ∃ k1 k2 : ℝ, 0 < k1 ∧ 0 < k2 ∧ peq k1 k2 = 1/2 ∧
        (1/2 - eps) ≤ |popB k1 k2 1 T - peq k1 k2|) ∧
      (∀ (k1 k2 p0 T delta : ℝ), 0 < k1 + k2 → 0 < T → (k1 + k2) * T ≤ delta →
        (1 - delta) * |p0 - peq k1 k2|
          ≤ |(∫ t in (0:ℝ)..T, popB k1 k2 p0 t) / T - peq k1 k2|)) ∧
    -- 4  rigid constraints differ from the stiff limit by the Fixman factor
    ((∀ (beta eps : ℝ) (V w : ℝ → ℝ) (q1 q2 : ℝ), 0 < beta → 0 < eps → 0 < w q1 → 0 < w q2 →
        softMarginal beta V w eps q1 / softMarginal beta V w eps q2
          = (rigidMarginal beta V q1 / rigidMarginal beta V q2)
            * Real.sqrt (w q2 / w q1)) ∧
      (∀ (beta eps : ℝ) (V w : ℝ → ℝ) (q1 q2 : ℝ), 0 < beta → 0 < eps → 0 < w q1 → 0 < w q2 →
        (softMarginal beta V w eps q1 / softMarginal beta V w eps q2
            = rigidMarginal beta V q1 / rigidMarginal beta V q2 ↔ w q1 = w q2)) ∧
      (∃ (beta eps : ℝ) (V w : ℝ → ℝ) (q1 q2 : ℝ), 0 < beta ∧ 0 < eps ∧ 0 < w q1 ∧ 0 < w q2 ∧
        softMarginal beta V w eps q1 / softMarginal beta V w eps q2
          ≠ rigidMarginal beta V q1 / rigidMarginal beta V q2)) :=
  ⟨⟨not_additive_cooperative,
      fun _ _ _ _ _ _ hW => mixed_eq_zero_of_additive hW _ _ _ _ _,
      fun _ _ _ _ _ _ _ hV => additive_error_lower_bound hV _ _ _ _ _⟩,
    ⟨fun _ _ _ h => protonatedFraction_strictAnti_pH h,
      fun pKa q => no_fixed_charge_model pKa q,
      fun G => linkage_cycle G⟩,
    ⟨fun _ _ p0 hk => popB_tendsto_equilibrium hk p0,
      fun _ _ hT heps => finite_run_not_boltzmann hT heps,
      fun _ _ p0 _ _ hk hT hd => timeAverage_stuck hk p0 hT hd⟩,
    ⟨fun _ _ V w _ _ hbeta heps hw1 hw2 => soft_ratio V w hbeta heps hw1 hw2,
      fun _ _ V w _ _ hbeta heps hw1 hw2 => soft_eq_rigid_iff V w hbeta heps hw1 hw2,
      rigid_ne_soft⟩⟩

end IDR
