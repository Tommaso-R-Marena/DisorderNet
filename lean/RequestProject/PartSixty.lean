/-
# Part LX  Explicit solvent: the potential of mean force is not a force field

Part XXIV puts the solvent in through two terms, and Part XXV named what stays outside: the
structure of the solvent itself.  `RequestProject.Pmf` supplies it, exactly.

`IDR.pmf_laws` bundles four statements:

1. *Integrating out the solvent is exact.*  The solute marginal of the joint Boltzmann measure is
   the Boltzmann measure of the solute energy plus the potential of mean force, for every finite
   solvent, every interaction and every temperature.
2. *The result is not pairwise.*  In the smallest explicit-solvent model -- one water molecule,
   two states, binding stabilised additively by each solute in contact -- the three-body
   inclusion--exclusion residue is `(1/b)·log(250/243)`, in closed form.  The interaction is
   exactly pairwise; the free energy is not, because a logarithm of a sum is not a sum.
3. *And its sign is anti-cooperative*: the residue is strictly positive, the third solute gains
   less than the second.
4. *The result is not even a potential.*  The solvation contribution of a single solute takes
   different values at two temperatures, so it is a free energy carrying an entropy and cannot be
   tabulated once and transferred.

For a disordered region -- which is, by construction, mostly surface -- this is where the physics
of collapse, of the temperature dependence of the radius of gyration, and of hydrophobic
cooperativity lives; and it is exactly the part that a pairwise, temperature-independent
solvation term cannot carry.
-/
import Mathlib
import RequestProject.Pmf

set_option autoImplicit false

namespace IDR

open IDR.Pmf

/-- **The explicit-solvent laws.**

1. the potential of mean force reproduces the solute marginal exactly;
2. its three-body residue in the minimal hydration model, in closed form;
3. that residue is strictly positive;
4. and the potential of mean force is temperature dependent. -/
theorem pmf_laws :
    (∀ (X S : Type) (_ : Fintype S) (_ : Nonempty S) (b : ℝ), b ≠ 0 →
        ∀ (Uint : X → S → ℝ) (Usol : X → ℝ) (x : X),
          (∑ s, Real.exp (-b * (Usol x + Uint x s)))
            = Real.exp (-b * (Usol x + pmf b Uint x))) ∧
    (∀ b : ℝ, b ≠ 0 →
        solventThreeBody b (Real.log 2 / b) = (1 / b) * Real.log (250 / 243)) ∧
    (∀ b : ℝ, 0 < b → 0 < solventThreeBody b (Real.log 2 / b)) ∧
    solvationShift (Real.log 2) ≠ solvationShift (2 * Real.log 2) := by
  refine ⟨fun X S _ _ b hb Uint Usol x => marginal_eq hb Uint Usol x,
    fun b hb => solventThreeBody_eq hb,
    fun b hb => solventThreeBody_pos hb,
    pmf_temperature_dependent⟩

end IDR
