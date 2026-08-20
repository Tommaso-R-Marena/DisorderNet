/-
# Part LX.1  Explicit solvent: the potential of mean force is not a force field

Part XXIV puts the solvent in through two terms -- a Generalized Born electrostatic term and a
surface-area term -- and Part XXV prices four of the remaining idealisations.  The item that was
named and left outside is the *structure* of the solvent: what is actually lost when the water
molecules are integrated out rather than simulated.  This file supplies it.

The construction is the exact one: a finite solvent state space, an arbitrary solute--solvent
interaction, and the potential of mean force

  `pmf b Uint x = -(1/b)·log ∑_s exp(-b·Uint x s)`.

* `marginal_eq` -- **the potential of mean force is exact.**  The solute marginal of the joint
  Boltzmann measure is the Boltzmann measure of `Usol + pmf`, identically, for every interaction
  and every temperature.  Integrating out the solvent is not an approximation; the question is
  only what the resulting object looks like.
* `solventZ_hydration`, `solventThreeBody_eq` -- **and it looks nothing like a force field.**  For
  the smallest nontrivial model -- one water molecule with a bound and a free state, binding
  stabilised additively by each solute in contact -- the solute--solute interaction acquires a
  three-body term whose inclusion--exclusion residue is computed in closed form,
  `(1/b)·log(250/243)`, at the coupling `b·eps = log 2`.  The *interaction* is exactly pairwise
  by construction; the free energy is not, because a logarithm of a sum is not a sum.
* `solventThreeBody_pos` -- the residue is strictly positive: solvent-mediated three-body forces
  here are *anti*-cooperative, the third solute gains less than the second.  This is the sign of
  the effect that makes hydrophobic association of a disordered chain non-additive.
* `pmf_temperature_dependent` -- **and it is not even a potential.**  The solvation contribution
  of a single solute differs between two temperatures: the potential of mean force is a free
  energy, carries an entropy, and cannot be tabulated once and reused.  A fixed implicit-solvent
  term is therefore not a temperature-transferable approximation to explicit water even in
  principle.

For a disordered region -- which is, by construction, mostly surface -- this is the term that
carries the physics: the collapse transition, the temperature dependence of the radius of
gyration, and the cooperativity of hydrophobic contacts all live in the part of the solvent free
energy that no pairwise, temperature-independent solvation term can represent.
-/
import Mathlib

set_option autoImplicit false

namespace IDR
namespace Pmf

open Finset

/-! ## The exact construction -/

variable {X S : Type*} [Fintype S]

/-- The solvent partition function at solute configuration `x`. -/
noncomputable def solventZ (b : ℝ) (Uint : X → S → ℝ) (x : X) : ℝ :=
  ∑ s, Real.exp (-b * Uint x s)

lemma solventZ_pos [Nonempty S] (b : ℝ) (Uint : X → S → ℝ) (x : X) :
    0 < solventZ b Uint x := by
  unfold solventZ
  apply Finset.sum_pos
  · intro s _
    exact Real.exp_pos _
  · exact Finset.univ_nonempty

/-- The potential of mean force: the solvent free energy at fixed solute configuration. -/
noncomputable def pmf (b : ℝ) (Uint : X → S → ℝ) (x : X) : ℝ :=
  -(1 / b) * Real.log (solventZ b Uint x)

/-- **The potential of mean force is exact.**  The solute marginal of the joint Boltzmann weight
is the Boltzmann weight of the solute energy plus the potential of mean force. -/
theorem marginal_eq [Nonempty S] {b : ℝ} (hb : b ≠ 0) (Uint : X → S → ℝ) (Usol : X → ℝ)
    (x : X) :
    (∑ s, Real.exp (-b * (Usol x + Uint x s)))
      = Real.exp (-b * (Usol x + pmf b Uint x)) := by
  have hZ : 0 < solventZ b Uint x := solventZ_pos b Uint x
  have hsplit : (∑ s, Real.exp (-b * (Usol x + Uint x s)))
      = Real.exp (-b * Usol x) * solventZ b Uint x := by
    unfold solventZ
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun s _ => ?_
    rw [← Real.exp_add]
    congr 1
    ring
  rw [hsplit]
  have hexp : Real.exp (-b * pmf b Uint x) = solventZ b Uint x := by
    unfold pmf
    have : -b * (-(1 / b) * Real.log (solventZ b Uint x)) = Real.log (solventZ b Uint x) := by
      field_simp
    rw [this, Real.exp_log hZ]
  rw [show -b * (Usol x + pmf b Uint x) = -b * Usol x + -b * pmf b Uint x by ring,
    Real.exp_add, hexp]

/-! ## The smallest explicit-solvent model, solved exactly -/

/-- One water molecule with two states: bound (state `0`), stabilised by `eps` for each solute in
contact, and free (state `1`).  The solute--solvent interaction is *exactly additive*. -/
noncomputable def hydration (eps : ℝ) (A : Finset (Fin 3)) (s : Fin 2) : ℝ :=
  if s = 0 then -eps * A.card else 0

lemma solventZ_hydration (b eps : ℝ) (A : Finset (Fin 3)) :
    solventZ b (hydration eps) A = Real.exp (b * eps * A.card) + 1 := by
  unfold solventZ hydration
  rw [Fin.sum_univ_two]
  simp only [if_neg (by decide : ¬((1 : Fin 2) = 0)), if_true]
  rw [show -b * (-eps * (A.card : ℝ)) = b * eps * A.card from by ring,
    show -b * (0:ℝ) = 0 from by ring, Real.exp_zero]

lemma pmf_hydration (b eps : ℝ) (A : Finset (Fin 3)) :
    pmf b (hydration eps) A = -(1 / b) * Real.log (Real.exp (b * eps * A.card) + 1) := by
  unfold pmf
  rw [solventZ_hydration]

/-- The inclusion--exclusion residue of the solvent-mediated interaction between the three
solutes: what is left after the best possible one- and two-body terms. -/
noncomputable def solventThreeBody (b eps : ℝ) : ℝ :=
  pmf b (hydration eps) {0, 1, 2}
    - (pmf b (hydration eps) {0, 1} + pmf b (hydration eps) {0, 2}
        + pmf b (hydration eps) {1, 2})
    + (pmf b (hydration eps) {0} + pmf b (hydration eps) {1} + pmf b (hydration eps) {2})
    - pmf b (hydration eps) ∅

/-- **The solvent-mediated three-body term, in closed form.**  At the coupling `b·eps = log 2`
the residue is exactly `(1/b)·log(250/243)`. -/
theorem solventThreeBody_eq {b : ℝ} (hb : b ≠ 0) :
    solventThreeBody b (Real.log 2 / b) = (1 / b) * Real.log (250 / 243) := by
  have hpow : ∀ n : ℕ, Real.exp (b * (Real.log 2 / b) * n) = 2 ^ n := by
    intro n
    have hn : b * (Real.log 2 / b) * n = n * Real.log 2 := by
      field_simp
    rw [hn, Real.exp_nat_mul, Real.exp_log (by norm_num)]
  have hc3 : ({0, 1, 2} : Finset (Fin 3)).card = 3 := by decide
  have hc01 : ({0, 1} : Finset (Fin 3)).card = 2 := by decide
  have hc02 : ({0, 2} : Finset (Fin 3)).card = 2 := by decide
  have hc12 : ({1, 2} : Finset (Fin 3)).card = 2 := by decide
  have hc0 : ({0} : Finset (Fin 3)).card = 1 := by decide
  have hc1 : ({1} : Finset (Fin 3)).card = 1 := by decide
  have hc2 : ({2} : Finset (Fin 3)).card = 1 := by decide
  have hce : (∅ : Finset (Fin 3)).card = 0 := by decide
  unfold solventThreeBody
  rw [pmf_hydration, pmf_hydration, pmf_hydration, pmf_hydration, pmf_hydration, pmf_hydration,
    pmf_hydration, pmf_hydration, hc3, hc01, hc02, hc12, hc0, hc1, hc2, hce]
  rw [hpow 3, hpow 2, hpow 1, hpow 0]
  norm_num
  -- goal is a linear identity between logarithms of 9, 5, 3, 2 and 250/243
  have h9 : Real.log 9 = 2 * Real.log 3 := by
    rw [show (9:ℝ) = 3 ^ 2 by norm_num, Real.log_pow]
    ring
  have h250 : Real.log (250 / 243) = Real.log 2 + 3 * Real.log 5 - 5 * Real.log 3 := by
    rw [Real.log_div (by norm_num) (by norm_num),
      show (250:ℝ) = 2 * 5 ^ 3 by norm_num, show (243:ℝ) = 3 ^ 5 by norm_num,
      Real.log_mul (by norm_num) (by norm_num), Real.log_pow, Real.log_pow]
    ring
  rw [h9, h250]
  field_simp
  ring

/-- The residue is strictly positive: the solvent-mediated three-body force is
anti-cooperative. -/
theorem solventThreeBody_pos {b : ℝ} (hb : 0 < b) :
    0 < solventThreeBody b (Real.log 2 / b) := by
  rw [solventThreeBody_eq hb.ne']
  have hlog : 0 < Real.log (250 / 243) := Real.log_pos (by norm_num)
  positivity

/-! ## The potential of mean force is a free energy, not a potential -/

/-- The solvation contribution of a single solute, at inverse temperature `b`, for unit
binding energy. -/
noncomputable def solvationShift (b : ℝ) : ℝ :=
  pmf b (hydration 1) {0} - pmf b (hydration 1) ∅

/-- **The potential of mean force is temperature dependent.**  The solvation contribution of a
single solute is not the same number at two different temperatures, so it is a free energy and
not a transferable potential. -/
theorem pmf_temperature_dependent :
    solvationShift (Real.log 2) ≠ solvationShift (2 * Real.log 2) := by
  have hl2 : (0:ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  have hc0 : ({0} : Finset (Fin 3)).card = 1 := by decide
  have hce : (∅ : Finset (Fin 3)).card = 0 := by decide
  have he1 : Real.exp (Real.log 2 * 1 * (1:ℕ)) = 2 := by
    norm_num
    exact Real.exp_log (by norm_num)
  have he2 : Real.exp (2 * Real.log 2 * 1 * (1:ℕ)) = 4 := by
    have : 2 * Real.log 2 * 1 * ((1:ℕ):ℝ) = Real.log 2 + Real.log 2 := by
      norm_num
      ring
    rw [this, Real.exp_add, Real.exp_log (by norm_num)]
    norm_num
  have hz : ∀ b : ℝ, Real.exp (b * 1 * ((0:ℕ):ℝ)) = 1 := by
    intro b
    norm_num
  have hA : solvationShift (Real.log 2)
      = -(1 / Real.log 2) * (Real.log 3 - Real.log 2) := by
    unfold solvationShift
    rw [pmf_hydration, pmf_hydration, hc0, hce, he1, hz]
    norm_num
    field_simp
    ring
  have hB : solvationShift (2 * Real.log 2)
      = -(1 / (2 * Real.log 2)) * (Real.log 5 - Real.log 2) := by
    unfold solvationShift
    rw [pmf_hydration, pmf_hydration, hc0, hce, he2, hz]
    norm_num
    field_simp
    ring
  rw [hA, hB]
  intro hcon
  have hne : Real.log 2 ≠ 0 := hl2.ne'
  have hkey : 2 * (Real.log 3 - Real.log 2) = Real.log 5 - Real.log 2 := by
    field_simp at hcon
    linarith
  have h92 : Real.log (9 / 2) = 2 * Real.log 3 - Real.log 2 := by
    rw [Real.log_div (by norm_num) (by norm_num), show (9:ℝ) = 3 ^ 2 by norm_num, Real.log_pow]
    ring
  have hfive : Real.log (9 / 2) = Real.log 5 := by
    rw [h92]
    linarith
  have := Real.log_injOn_pos (by norm_num : (9:ℝ)/2 ∈ Set.Ioi (0:ℝ))
    (by norm_num : (5:ℝ) ∈ Set.Ioi (0:ℝ)) hfive
  norm_num at this

end Pmf
end IDR
