/-
# Part XXX  The entropy price of ordering: affinity is a free-energy difference of ensembles

Part XVI proved the *reciprocity* between conformational populations and affinity; Part X proved
that a binding constant is not a function of a mean structure with error bars.  What neither
gives is the accounting: how much free energy a disordered region pays for being disordered when
it binds.  This file computes it exactly, in the standard finite-conformer partition-function
formalism, and derives the three consequences that a binding model must respect.

Setting: a free-state ensemble with normalised populations `p` over `m` conformers, a
binding-competent subset `S`, and a per-conformer interaction energy `-eps k` gained on binding.
The bound-state conformational sum runs over `S` only, so the binding free energy is
`ΔG = -(1/β) log Σ_{k∈S} p_k e^{β eps_k}` (`deltaG`).

* `deltaG_eq_selection` -- **the conformational-selection decomposition.**  With a uniform
  interaction `e` over the competent set, `ΔG = -e + (1/β) log (1/P_S)`: the intrinsic
  interaction, plus a penalty that is the free energy of the population restriction.
* `selection_penalty_pos` -- that penalty is strictly positive whenever the competent set is not
  the whole ensemble, and it is `kT log(1/P_S)` -- at 300 K, 1.4 kcal/mol per decade of
  population.  The affinity of a disordered motif is not a property of the bound structure.
* `penalty_uniform_eq_log_card` -- for a uniform free ensemble over `m` conformers binding
  through one of them, the penalty is exactly `kT log m`, i.e. `kT` times the conformational
  entropy of the free state.  Two models with the same predicted bound structure and different
  free-state breadth predict different affinities, and the gap is the entropy they disagree on.
* `deltaG_antitone_subset` -- **fuzziness pays it back.**  Enlarging the set of conformers the
  complex tolerates can only strengthen binding.  A fuzzy complex is not a defect of the model;
  it is the thermodynamically favoured arrangement whenever the interaction survives it.
* `deltaG_le_neg_mean` -- **averaging interaction energies underestimates binding.**  Jensen for
  `exp`: the true `ΔG` is at most the population average of `-eps`, so scoring a designed binder
  by the mean interaction energy over an ensemble is systematically conservative.
* `deltaG_le_of_conformer` -- **the minority-report bound.**  One competent conformer of
  population `p_k` and interaction `eps_k` alone guarantees `ΔG ≤ -eps_k + kT log (1/p_k)`,
  whatever the rest of the ensemble does: a 1% conformer with a 10 kcal/mol interface binds at
  least as well as -7.3 kcal/mol at 300 K.
-/
import Mathlib

set_option autoImplicit false

namespace Selection

open Finset

variable {m : ℕ}

/-- Bound-state conformational sum: only the competent conformers `S` contribute. -/
noncomputable def Zbound (beta : ℝ) (p eps : Fin m → ℝ) (S : Finset (Fin m)) : ℝ :=
  ∑ k ∈ S, p k * Real.exp (beta * eps k)

/-- Binding free energy of the region, in energy units. -/
noncomputable def deltaG (beta : ℝ) (p eps : Fin m → ℝ) (S : Finset (Fin m)) : ℝ :=
  -(1 / beta) * Real.log (Zbound beta p eps S)

/-- Population of the binding-competent subset in the free state. -/
noncomputable def popOf (p : Fin m → ℝ) (S : Finset (Fin m)) : ℝ := ∑ k ∈ S, p k

lemma Zbound_pos {beta : ℝ} {p eps : Fin m → ℝ} {S : Finset (Fin m)} (hp : ∀ k, 0 ≤ p k)
    {k0 : Fin m} (hk0 : k0 ∈ S) (hpk : 0 < p k0) : 0 < Zbound beta p eps S := by
  refine lt_of_lt_of_le (mul_pos hpk (Real.exp_pos (beta * eps k0))) ?_
  exact Finset.single_le_sum
    (f := fun k => p k * Real.exp (beta * eps k))
    (fun k _ => mul_nonneg (hp k) (Real.exp_pos _).le) hk0

/-- **The conformational-selection decomposition.**  With a uniform interaction energy `e` over
the competent set, the binding free energy is the interaction plus the free energy of the
population restriction. -/
theorem deltaG_eq_selection {beta e : ℝ} (hbeta : 0 < beta) {p eps : Fin m → ℝ}
    {S : Finset (Fin m)} (hp : ∀ k, 0 ≤ p k) (heps : ∀ k ∈ S, eps k = e)
    {k0 : Fin m} (hk0 : k0 ∈ S) (hpk : 0 < p k0) :
    deltaG beta p eps S = -e + (1 / beta) * Real.log (1 / popOf p S) := by
  have hZ : Zbound beta p eps S = Real.exp (beta * e) * popOf p S := by
    unfold Zbound popOf
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k hk => ?_
    rw [heps k hk]
    ring
  have hPpos : 0 < popOf p S := by
    refine lt_of_lt_of_le hpk (Finset.single_le_sum (f := p) (fun k _ => hp k) hk0)
  rw [deltaG, hZ, Real.log_mul (Real.exp_ne_zero _) (ne_of_gt hPpos), Real.log_exp,
    Real.log_div one_ne_zero (ne_of_gt hPpos), Real.log_one]
  field_simp
  ring

/-- **The penalty is strictly positive.**  If the competent conformers are not all of the
ensemble, ordering costs `kT log (1/P_S) > 0`. -/
theorem selection_penalty_pos {beta : ℝ} (hbeta : 0 < beta) {P : ℝ} (hP0 : 0 < P) (hP1 : P < 1) :
    0 < (1 / beta) * Real.log (1 / P) := by
  have hlog : 0 < Real.log (1 / P) := Real.log_pos (by rw [lt_div_iff₀ hP0]; linarith)
  exact mul_pos (one_div_pos.mpr hbeta) hlog

/-- **The penalty is the conformational entropy.**  A uniform free ensemble over `m` conformers
that binds through exactly one of them pays `kT log m`, which is `kT` times the Gibbs--Shannon
entropy of the free state. -/
theorem penalty_uniform_eq_log_card (beta : ℝ) {m : ℕ} (hm : 0 < m) (k0 : Fin m) :
    deltaG beta (fun _ => (1 : ℝ) / m) (fun _ => (0 : ℝ)) {k0}
      = (1 / beta) * Real.log m := by
  have hmR : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  have hZ : Zbound beta (fun _ => (1 : ℝ) / m) (fun _ => (0 : ℝ)) {k0} = 1 / m := by
    simp [Zbound]
  rw [deltaG, hZ, Real.log_div one_ne_zero (ne_of_gt hmR), Real.log_one]
  ring

/-- **Fuzziness pays the entropy back.**  If the complex tolerates a larger set of conformers,
the binding free energy can only go down. -/
theorem deltaG_antitone_subset {beta : ℝ} (hbeta : 0 < beta) {p eps : Fin m → ℝ}
    {S T : Finset (Fin m)} (hST : S ⊆ T) (hp : ∀ k, 0 ≤ p k)
    {k0 : Fin m} (hk0 : k0 ∈ S) (hpk : 0 < p k0) :
    deltaG beta p eps T ≤ deltaG beta p eps S := by
  have hSpos : 0 < Zbound beta p eps S := Zbound_pos hp hk0 hpk
  have hmono : Zbound beta p eps S ≤ Zbound beta p eps T :=
    Finset.sum_le_sum_of_subset_of_nonneg hST
      (fun k _ _ => mul_nonneg (hp k) (Real.exp_pos _).le)
  have hlog : Real.log (Zbound beta p eps S) ≤ Real.log (Zbound beta p eps T) :=
    Real.log_le_log hSpos hmono
  have hcoef : 0 < 1 / beta := by positivity
  rw [deltaG, deltaG]
  nlinarith [hlog, hcoef]

/-- **Averaging interaction energies underestimates binding.**  Jensen for `exp`: the exact
binding free energy of the ensemble is at most the population average of the per-conformer
interaction free energies. -/
theorem deltaG_le_neg_mean {beta : ℝ} (hbeta : 0 < beta) {p eps : Fin m → ℝ}
    (hp : ∀ k, 0 ≤ p k) (hsum : ∑ k, p k = 1) :
    deltaG beta p eps Finset.univ ≤ -∑ k, p k * eps k := by
  have hjensen : Real.exp (∑ k, p k * (beta * eps k))
      ≤ ∑ k, p k * Real.exp (beta * eps k) := by
    have := (convexOn_exp).map_centerMass_le (t := Finset.univ) (w := p)
      (p := fun k => beta * eps k) (fun k _ => hp k) (by rw [hsum]; norm_num)
      (fun k _ => Set.mem_univ _)
    simpa [Finset.centerMass, hsum, smul_eq_mul, Function.comp] using this
  have hZpos : 0 < Zbound beta p eps Finset.univ := by
    obtain ⟨k0, hk0⟩ : ∃ k, 0 < p k := by
      by_contra hcon
      push_neg at hcon
      have : ∑ k, p k ≤ 0 := Finset.sum_nonpos fun k _ => hcon k
      rw [hsum] at this; linarith
    exact Zbound_pos hp (Finset.mem_univ k0) hk0
  have hlog : ∑ k, p k * (beta * eps k) ≤ Real.log (Zbound beta p eps Finset.univ) := by
    have := Real.log_le_log (Real.exp_pos _) hjensen
    rwa [Real.log_exp] at this
  have hlin : ∑ k, p k * (beta * eps k) = beta * ∑ k, p k * eps k := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => by ring
  rw [hlin] at hlog
  have hcoef : (0 : ℝ) < 1 / beta := one_div_pos.mpr hbeta
  have h2 := mul_le_mul_of_nonneg_left hlog hcoef.le
  have h3 : (1 / beta) * (beta * ∑ k, p k * eps k) = ∑ k, p k * eps k := by
    field_simp
  rw [h3] at h2
  rw [deltaG]
  linarith

/-- **The minority-report bound.**  A single competent conformer of population `p k` and
interaction `eps k` already guarantees an affinity, whatever the rest of the ensemble does. -/
theorem deltaG_le_of_conformer {beta : ℝ} (hbeta : 0 < beta) {p eps : Fin m → ℝ}
    {S : Finset (Fin m)} (hp : ∀ i, 0 ≤ p i) {k : Fin m} (hk : k ∈ S) (hpk : 0 < p k) :
    deltaG beta p eps S ≤ -eps k + (1 / beta) * Real.log (1 / p k) := by
  have hterm : p k * Real.exp (beta * eps k) ≤ Zbound beta p eps S :=
    Finset.single_le_sum (f := fun i => p i * Real.exp (beta * eps i))
      (fun i _ => mul_nonneg (hp i) (Real.exp_pos _).le) hk
  have hpos : 0 < p k * Real.exp (beta * eps k) := mul_pos hpk (Real.exp_pos _)
  have hlog : Real.log (p k) + beta * eps k ≤ Real.log (Zbound beta p eps S) := by
    have h := Real.log_le_log hpos hterm
    rwa [Real.log_mul (ne_of_gt hpk) (Real.exp_ne_zero _), Real.log_exp] at h
  have hcoef : (0 : ℝ) < 1 / beta := one_div_pos.mpr hbeta
  have h2 := mul_le_mul_of_nonneg_left hlog hcoef.le
  have h3 : (1 / beta) * (Real.log (p k) + beta * eps k)
      = (1 / beta) * Real.log (p k) + eps k := by
    field_simp
  rw [h3] at h2
  rw [deltaG, Real.log_div one_ne_zero (ne_of_gt hpk), Real.log_one]
  linarith

end Selection
