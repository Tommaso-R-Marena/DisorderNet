/-
# Part X.3  Transient secondary structure: the helix--coil chain

The single most characteristic experimental fact about an intrinsically disordered region is
*fractional, cooperative* secondary structure: a residue is 30% helical, its neighbour 25%,
and the two are correlated.  The classical exactly-solvable model of that physics is the
one-dimensional Ising (Zimm--Bragg) chain, in which each residue carries a helix/coil
variable, a field `h` biases it towards helix and a nearest-neighbour coupling `J` makes
helices cooperative.

This file builds the model from its configuration sum -- no transfer matrix is *assumed* --
and proves the facts a model of an IDR has to respect.

* `weight`, `Zhead` -- the Boltzmann weight of a helix/coil configuration and the partition
  function of a chain whose first residue is in a prescribed state.
* `Zhead_succ` -- **the transfer-matrix identity**, derived: the configuration sum satisfies
  `Z_{n+1}(b) = Σ_c e^{h σ_b + J σ_b σ_c} Z_n(c)`.  This is what makes the model solvable and
  what makes a residue-level model with nearest-neighbour couplings tractable at all.
* `Zhead_pos` -- every state of every finite chain has strictly positive weight.
* `helixFraction_pos`, `helixFraction_lt_one` -- **the punchline for disorder**: at any finite
  helix propensity and any finite temperature the helical population of a residue is strictly
  between 0 and 1.  No single structure -- helix or coil -- is the answer; the answer is a
  number in `(0,1)`, i.e. a distribution.  A model of an IDR must therefore emit populations.
* `helix_cooperativity` -- with a positive coupling, neighbouring residues are *positively
  correlated* and the joint distribution does not factorise.  A per-residue (independent)
  propensity model is provably not enough.
-/
import Mathlib

namespace IDR

open Finset

namespace HelixCoil

/-- The helix/coil variable of a residue as a spin: `+1` helical, `-1` coil. -/
def spin (b : Bool) : ℝ := if b then 1 else -1

lemma spin_sq (b : Bool) : spin b * spin b = 1 := by
  cases b <;> norm_num [spin]

/-- The Boltzmann weight of a helix/coil configuration of a chain, read left to right:
each residue contributes `e^{h σ}` and each bond `e^{J σσ'}`. -/
noncomputable def weight (J h : ℝ) : List Bool → ℝ
  | [] => 1
  | [b] => Real.exp (h * spin b)
  | b :: c :: t => Real.exp (h * spin b + J * spin b * spin c) * weight J h (c :: t)

lemma weight_pos (J h : ℝ) : ∀ l : List Bool, 0 < weight J h l
  | [] => by norm_num [weight]
  | [_] => by rw [weight]; exact Real.exp_pos _
  | b :: c :: t => by
      rw [weight]
      exact mul_pos (Real.exp_pos _) (weight_pos J h (c :: t))

/-- The partition function of a chain of `n+1` residues whose first residue is in state `b`,
as a sum over all configurations of the remaining `n` residues. -/
noncomputable def Zhead (J h : ℝ) (n : ℕ) (b : Bool) : ℝ :=
  ∑ s : Fin n → Bool, weight J h (b :: List.ofFn s)

/-- The transfer matrix element `T_{bc} = e^{h σ_b + J σ_b σ_c}`. -/
noncomputable def transfer (J h : ℝ) (b c : Bool) : ℝ :=
  Real.exp (h * spin b + J * spin b * spin c)

lemma Zhead_zero (J h : ℝ) (b : Bool) : Zhead J h 0 b = Real.exp (h * spin b) := by
  simp [Zhead, weight]

/-- **The transfer-matrix identity, derived from the configuration sum.** -/
theorem Zhead_succ (J h : ℝ) (n : ℕ) (b : Bool) :
    Zhead J h (n + 1) b = ∑ c : Bool, transfer J h b c * Zhead J h n c := by
  unfold Zhead
  have hsplit : (∑ s : Fin (n + 1) → Bool, weight J h (b :: List.ofFn s))
      = ∑ p : Bool × (Fin n → Bool), weight J h (b :: p.1 :: List.ofFn p.2) :=
    Fintype.sum_equiv (Fin.consEquiv (fun _ : Fin (n + 1) => Bool)).symm _ _ (fun s => by
      rw [List.ofFn_succ]
      simp [Fin.consEquiv]
      rfl)
  rw [hsplit, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [weight, transfer]

lemma Zhead_pos (J h : ℝ) (n : ℕ) (b : Bool) : 0 < Zhead J h n b :=
  Finset.sum_pos (fun _ _ => weight_pos J h _) Finset.univ_nonempty

/-- The helical population of the first residue of a chain of `n+1` residues. -/
noncomputable def helixFraction (J h : ℝ) (n : ℕ) : ℝ :=
  Zhead J h n true / (Zhead J h n true + Zhead J h n false)

/-- **Transient helicity is strictly positive.** -/
theorem helixFraction_pos (J h : ℝ) (n : ℕ) : 0 < helixFraction J h n := by
  have h1 := Zhead_pos J h n true
  have h2 := Zhead_pos J h n false
  unfold helixFraction
  positivity

/-- **Transient helicity is strictly below one.**  At every finite propensity and every
finite temperature a residue of the chain is *partly* helical: the correct description of the
residue is a population, not a structure. -/
theorem helixFraction_lt_one (J h : ℝ) (n : ℕ) : helixFraction J h n < 1 := by
  have h1 := Zhead_pos J h n true
  have h2 := Zhead_pos J h n false
  unfold helixFraction
  rw [div_lt_one (by linarith)]
  linarith

/-! ## Cooperativity: neighbouring residues do not factorise -/

/-- The two-residue partition function at zero field. -/
noncomputable def Z2 (J : ℝ) : ℝ := ∑ b : Bool, ∑ c : Bool, Real.exp (J * spin b * spin c)

lemma Z2_eq (J : ℝ) : Z2 J = 2 * Real.exp J + 2 * Real.exp (-J) := by
  simp [Z2, spin]
  ring

lemma Z2_pos (J : ℝ) : 0 < Z2 J := by
  rw [Z2_eq]
  have := Real.exp_pos J
  have := Real.exp_pos (-J)
  linarith

/-- The mean spin of the first of two neighbouring residues, at zero field. -/
noncomputable def meanSpin (J : ℝ) : ℝ :=
  (∑ b : Bool, ∑ c : Bool, spin b * Real.exp (J * spin b * spin c)) / Z2 J

/-- The two-residue spin correlation, at zero field. -/
noncomputable def pairSpin (J : ℝ) : ℝ :=
  (∑ b : Bool, ∑ c : Bool, spin b * spin c * Real.exp (J * spin b * spin c)) / Z2 J

lemma meanSpin_eq_zero (J : ℝ) : meanSpin J = 0 := by
  unfold meanSpin
  have : (∑ b : Bool, ∑ c : Bool, spin b * Real.exp (J * spin b * spin c)) = 0 := by
    simp [spin]
    ring
  rw [this, zero_div]

/-- **Cooperativity is real and is not a per-residue effect.**  With a positive
nearest-neighbour coupling the covariance of two neighbouring helix/coil variables is
strictly positive, while each has mean zero: the joint distribution does not factorise into
independent residue propensities. -/
theorem helix_cooperativity {J : ℝ} (hJ : 0 < J) :
    0 < pairSpin J - meanSpin J * meanSpin J := by
  rw [meanSpin_eq_zero]
  have hnum : (∑ b : Bool, ∑ c : Bool, spin b * spin c * Real.exp (J * spin b * spin c))
      = 2 * Real.exp J - 2 * Real.exp (-J) := by
    simp [spin]
    ring
  have hlt : Real.exp (-J) < Real.exp J := Real.exp_lt_exp.mpr (by linarith)
  have : 0 < pairSpin J := by
    unfold pairSpin
    rw [hnum]
    apply div_pos (by linarith) (Z2_pos J)
  linarith

end HelixCoil

end IDR
