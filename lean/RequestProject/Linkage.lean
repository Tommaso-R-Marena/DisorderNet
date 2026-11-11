/-
# Part XVI  Linkage: coupled folding and binding, and the reciprocity it obeys

Disordered regions do their work by *coupling*: a ligand, a phosphate, a partner surface
shifts the conformational populations, and reciprocally the conformational populations set
the affinity.  This file proves that the two effects are one number, in two independent
forms -- a differential one (Wyman's linkage relation, as an equality of two derivatives)
and a finite one (the thermodynamic box).

Setting: a conformational library carrying a reference ensemble `q` and two couplings, a
structural field `A` (say, a helix-stabilising co-solute, or the folded-state indicator) and
a chemical potential `B` (the ligand).  The populations at coupling `(lam, mu)` are
`q · exp (lam·A + mu·B)`, normalised (`part2`, `mean2`).

* `hasDerivAt_mean2_mu`, `hasDerivAt_mean2_lam` -- both partial responses are covariances
  in the *same* ensemble.
* `linkage_reciprocity` -- **Wyman's linkage relation**: `∂⟨A⟩/∂mu = ∂⟨B⟩/∂lam`.  The amount
  by which the ligand orders the region equals the amount by which ordering the region
  loads the ligand: a model cannot fit one without committing to the other.
* `no_linkage_of_uniform_affinity` and `linkage_of_comonotone` -- the coupling vanishes
  exactly when the ligand does not discriminate between conformations, and is strictly
  positive as soon as it does (with the populated conformations ordering both ways alike).
* `thermodynamic_box` / `folding_stabilization_eq_binding_enhancement` -- the finite form on
  the four-state cycle `U, F, UL, FL`: the ligand's stabilisation of the folded state and
  the folded state's enhancement of binding are the *same* factor `exp (-w)`, independent of
  the intrinsic stability and the intrinsic affinity.  Consequently a model that predicts a
  binding constant and a model that predicts a folded population are the same model, and
  reporting one while leaving the other free is not an option.
* `apo_folded_fraction_lt_holo` -- a worked consequence: with favourable coupling the folded
  fraction strictly increases on saturation with ligand, by exactly the box factor.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.FreeEnergy
import RequestProject.Response
import RequestProject.Crowding

namespace IDR

open Finset
open scoped Classical

namespace Linkage

variable {n : ℕ}

/-! ## The two-parameter tilted family -/

/-- Partition function of the doubly tilted family. -/
noncomputable def part2 (q A B : Fin n → ℝ) (lam mu : ℝ) : ℝ :=
  ∑ j, q j * Real.exp (lam * A j + mu * B j)

/-- Unnormalised average of `f` in the doubly tilted family. -/
noncomputable def unAvg2 (q A B f : Fin n → ℝ) (lam mu : ℝ) : ℝ :=
  ∑ j, q j * f j * Real.exp (lam * A j + mu * B j)

/-- The populations of the doubly tilted family. -/
noncomputable def pop2 (q A B : Fin n → ℝ) (lam mu : ℝ) : Fin n → ℝ :=
  fun j => q j * Real.exp (lam * A j + mu * B j) / part2 q A B lam mu

/-- The average of `f` at coupling `(lam, mu)`. -/
noncomputable def mean2 (q A B f : Fin n → ℝ) (lam mu : ℝ) : ℝ :=
  unAvg2 q A B f lam mu / part2 q A B lam mu

/-- The covariance of `f` and `g` at coupling `(lam, mu)`. -/
noncomputable def cov2 (q A B f g : Fin n → ℝ) (lam mu : ℝ) : ℝ :=
  mean2 q A B (fun j => f j * g j) lam mu - mean2 q A B f lam mu * mean2 q A B g lam mu

variable {q A B f g : Fin n → ℝ}

lemma part2_pos (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam mu : ℝ) : 0 < part2 q A B lam mu :=
  Finset.sum_pos (fun j _ => mul_pos (hq j) (Real.exp_pos _)) ⟨⟨0, hn⟩, Finset.mem_univ _⟩

lemma pop2_pos (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam mu : ℝ) (j : Fin n) :
    0 < pop2 q A B lam mu j :=
  div_pos (mul_pos (hq j) (Real.exp_pos _)) (part2_pos hn hq lam mu)

lemma pop2_sum_one (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam mu : ℝ) :
    ∑ j, pop2 q A B lam mu j = 1 := by
  simp only [pop2, part2, ← Finset.sum_div]
  exact div_self (part2_pos (A := A) (B := B) hn hq lam mu).ne'

lemma mean2_eq_sum (lam mu : ℝ) :
    mean2 q A B f lam mu = ∑ j, pop2 q A B lam mu j * f j := by
  simp only [mean2, unAvg2, pop2, Finset.sum_div]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- The covariance of the tilted family is the weighted covariance of its populations. -/
lemma cov2_eq_weighted (lam mu : ℝ) :
    cov2 q A B f g lam mu
      = (∑ j, pop2 q A B lam mu j * (f j * g j))
        - (∑ j, pop2 q A B lam mu j * f j) * (∑ j, pop2 q A B lam mu j * g j) := by
  simp only [cov2, mean2_eq_sum]

/-- Covariance is symmetric -- the fact from which the whole of linkage follows. -/
lemma cov2_comm (lam mu : ℝ) : cov2 q A B f g lam mu = cov2 q A B g f lam mu := by
  simp only [cov2]
  rw [mul_comm (mean2 q A B f lam mu)]
  congr 2
  funext j
  ring

/-! ## Partial responses -/

lemma hasDerivAt_part2_mu (lam mu : ℝ) :
    HasDerivAt (fun m => part2 q A B lam m) (unAvg2 q A B B lam mu) mu := by
  have h : ∀ j : Fin n,
      HasDerivAt (fun m : ℝ => q j * Real.exp (lam * A j + m * B j))
        (q j * B j * Real.exp (lam * A j + mu * B j)) mu := by
    intro j
    have h1 : HasDerivAt (fun m : ℝ => lam * A j + m * B j) (B j) mu := by
      simpa using ((hasDerivAt_id mu).mul_const (B j)).const_add (lam * A j)
    have h2 := (h1.exp).const_mul (q j)
    convert h2 using 1
    ring
  exact HasDerivAt.fun_sum (fun j (_ : j ∈ (Finset.univ : Finset (Fin n))) => h j)

lemma hasDerivAt_unAvg2_mu (lam mu : ℝ) :
    HasDerivAt (fun m => unAvg2 q A B f lam m)
      (unAvg2 q A B (fun j => f j * B j) lam mu) mu := by
  have h : ∀ j : Fin n,
      HasDerivAt (fun m : ℝ => q j * f j * Real.exp (lam * A j + m * B j))
        (q j * (f j * B j) * Real.exp (lam * A j + mu * B j)) mu := by
    intro j
    have h1 : HasDerivAt (fun m : ℝ => lam * A j + m * B j) (B j) mu := by
      simpa using ((hasDerivAt_id mu).mul_const (B j)).const_add (lam * A j)
    have h2 := (h1.exp).const_mul (q j * f j)
    convert h2 using 1
    ring
  exact HasDerivAt.fun_sum (fun j (_ : j ∈ (Finset.univ : Finset (Fin n))) => h j)

lemma hasDerivAt_part2_lam (lam mu : ℝ) :
    HasDerivAt (fun l => part2 q A B l mu) (unAvg2 q A B A lam mu) lam := by
  have h : ∀ j : Fin n,
      HasDerivAt (fun l : ℝ => q j * Real.exp (l * A j + mu * B j))
        (q j * A j * Real.exp (lam * A j + mu * B j)) lam := by
    intro j
    have h1 : HasDerivAt (fun l : ℝ => l * A j + mu * B j) (A j) lam := by
      simpa using ((hasDerivAt_id lam).mul_const (A j)).add_const (mu * B j)
    have h2 := (h1.exp).const_mul (q j)
    convert h2 using 1
    ring
  exact HasDerivAt.fun_sum (fun j (_ : j ∈ (Finset.univ : Finset (Fin n))) => h j)

lemma hasDerivAt_unAvg2_lam (lam mu : ℝ) :
    HasDerivAt (fun l => unAvg2 q A B f l mu)
      (unAvg2 q A B (fun j => f j * A j) lam mu) lam := by
  have h : ∀ j : Fin n,
      HasDerivAt (fun l : ℝ => q j * f j * Real.exp (l * A j + mu * B j))
        (q j * (f j * A j) * Real.exp (lam * A j + mu * B j)) lam := by
    intro j
    have h1 : HasDerivAt (fun l : ℝ => l * A j + mu * B j) (A j) lam := by
      simpa using ((hasDerivAt_id lam).mul_const (A j)).add_const (mu * B j)
    have h2 := (h1.exp).const_mul (q j * f j)
    convert h2 using 1
    ring
  exact HasDerivAt.fun_sum (fun j (_ : j ∈ (Finset.univ : Finset (Fin n))) => h j)

/-- The response of any average to the chemical potential is its covariance with the
binding observable. -/
theorem hasDerivAt_mean2_mu (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam mu : ℝ) :
    HasDerivAt (fun m => mean2 q A B f lam m) (cov2 q A B f B lam mu) mu := by
  have hZ := hasDerivAt_part2_mu (q := q) (A := A) (B := B) lam mu
  have hN := hasDerivAt_unAvg2_mu (q := q) (A := A) (B := B) (f := f) lam mu
  have hpos := part2_pos (A := A) (B := B) hn hq lam mu
  have hdiv := hN.div hZ hpos.ne'
  have hval : (unAvg2 q A B (fun j => f j * B j) lam mu * part2 q A B lam mu
      - unAvg2 q A B f lam mu * unAvg2 q A B B lam mu) / part2 q A B lam mu ^ 2
      = cov2 q A B f B lam mu := by
    simp only [cov2, mean2]
    field_simp
  rw [← hval]
  exact hdiv

/-- The response of any average to the structural field is its covariance with the field. -/
theorem hasDerivAt_mean2_lam (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam mu : ℝ) :
    HasDerivAt (fun l => mean2 q A B f l mu) (cov2 q A B f A lam mu) lam := by
  have hZ := hasDerivAt_part2_lam (q := q) (A := A) (B := B) lam mu
  have hN := hasDerivAt_unAvg2_lam (q := q) (A := A) (B := B) (f := f) lam mu
  have hpos := part2_pos (A := A) (B := B) hn hq lam mu
  have hdiv := hN.div hZ hpos.ne'
  have hval : (unAvg2 q A B (fun j => f j * A j) lam mu * part2 q A B lam mu
      - unAvg2 q A B f lam mu * unAvg2 q A B A lam mu) / part2 q A B lam mu ^ 2
      = cov2 q A B f A lam mu := by
    simp only [cov2, mean2]
    field_simp
  rw [← hval]
  exact hdiv

/-- **Wyman's linkage relation.**  The derivative of the structural average with respect to
the ligand's chemical potential equals the derivative of the ligand's average occupancy
with respect to the structural field.  A model of a disordered region that fits how a
partner reshapes its ensemble has thereby committed itself to how the ensemble sets the
partner's affinity, and vice versa: the two numbers are one. -/
theorem linkage_reciprocity (hn : 0 < n) (hq : ∀ j, 0 < q j) (lam mu : ℝ) :
    deriv (fun m => mean2 q A B A lam m) mu = deriv (fun l => mean2 q A B B l mu) lam := by
  rw [(hasDerivAt_mean2_mu (f := A) hn hq lam mu).deriv,
    (hasDerivAt_mean2_lam (f := B) hn hq lam mu).deriv]
  exact cov2_comm lam mu

/-- If the ligand binds every conformation equally well, it shifts nothing: no linkage. -/
theorem no_linkage_of_uniform_affinity (hn : 0 < n) (hq : ∀ j, 0 < q j) {c : ℝ}
    (hconst : ∀ j, B j = c) (lam mu : ℝ) : cov2 q A B A B lam mu = 0 := by
  have hone := pop2_sum_one (A := A) (B := B) hn hq lam mu
  rw [cov2_eq_weighted]
  simp only [hconst]
  rw [← Finset.sum_mul, hone, one_mul]
  have : ∀ j : Fin n, pop2 q A B lam mu j * (A j * c) = (pop2 q A B lam mu j * A j) * c := by
    intro j; ring
  rw [Finset.sum_congr rfl fun j (_ : j ∈ Finset.univ) => this j, ← Finset.sum_mul]
  ring

/-- And if it discriminates -- two conformations differing in both the structural
coordinate and the affinity, ordered alike -- the linkage is strictly positive: the ligand
*must* reshape the ensemble, and the ensemble *must* set the affinity. -/
theorem linkage_of_comonotone (hn : 0 < n) (hq : ∀ j, 0 < q j)
    (hc : Crowding.Comonotone A B) {i₀ j₀ : Fin n}
    (hsep : 0 < (A i₀ - A j₀) * (B i₀ - B j₀)) (lam mu : ℝ) :
    0 < cov2 q A B A B lam mu := by
  rw [cov2_eq_weighted]
  exact Crowding.cov_pos_of_comonotone (pop2_sum_one hn hq lam mu)
    (fun j => (pop2_pos hn hq lam mu j).le) hc (pop2_pos hn hq lam mu i₀)
    (pop2_pos hn hq lam mu j₀) hsep

/-! ## The finite form: the thermodynamic box

Four states: unfolded apo `U`, folded apo `F`, unfolded holo `UL`, folded holo `FL`, with
energies `0`, `eF`, `eL`, `eF + eL + w`.  The single number `w` is the coupling free energy;
everything else is intrinsic stability and intrinsic affinity. -/

/-- The energies of the four-state folding/binding cycle. -/
noncomputable def boxE (eF eL w : ℝ) : Fin 4 → ℝ := ![0, eF, eL, eF + eL + w]

/-- Populations of the cycle at unit inverse temperature. -/
noncomputable def boxP (eF eL w : ℝ) : Fin 4 → ℝ := FreeEnergy.boltz 1 (boxE eF eL w)

lemma boxP_pos (eF eL w : ℝ) (j : Fin 4) : 0 < boxP eF eL w j :=
  FreeEnergy.boltz_pos (by norm_num) 1 _ j

/-- **The thermodynamic box.**  The cross-product of the four populations is `exp (-w)`,
independently of the intrinsic stability `eF` and the intrinsic affinity `eL`. -/
theorem thermodynamic_box (eF eL w : ℝ) :
    boxP eF eL w 3 * boxP eF eL w 0
      = Real.exp (-w) * (boxP eF eL w 1 * boxP eF eL w 2) := by
  have hZ : (0:ℝ) < FreeEnergy.part 1 (boxE eF eL w) := FreeEnergy.part_pos (by norm_num) _ _
  have key : Real.exp (-1 * (eF + eL + w)) * Real.exp (-1 * 0)
      = Real.exp (-w) * (Real.exp (-1 * eF) * Real.exp (-1 * eL)) := by
    rw [← Real.exp_add, ← Real.exp_add, ← Real.exp_add]
    ring_nf
  simp only [boxP, FreeEnergy.boltz, boxE, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, Matrix.cons_val_two, Matrix.cons_val_three, Matrix.tail_cons,
    div_mul_div_comm]
  rw [key]
  ring

/-- **Stabilisation equals enhancement.**  The factor by which the ligand shifts the
folding equilibrium (`FL/UL` against `F/U`) is exactly the factor by which folding shifts
the binding equilibrium, namely `exp (-w)`.  There is one coupling constant, and a model
must report it once. -/
theorem folding_stabilization_eq_binding_enhancement (eF eL w : ℝ) :
    boxP eF eL w 3 / boxP eF eL w 2 = Real.exp (-w) * (boxP eF eL w 1 / boxP eF eL w 0)
      ∧ boxP eF eL w 3 / boxP eF eL w 1 = Real.exp (-w) * (boxP eF eL w 2 / boxP eF eL w 0) := by
  have h := thermodynamic_box eF eL w
  have h0 := (boxP_pos eF eL w 0).ne'
  have h1 := (boxP_pos eF eL w 1).ne'
  have h2 := (boxP_pos eF eL w 2).ne'
  constructor
  · field_simp
    linarith [h]
  · field_simp
    linarith [h]

/-- The apo folded fraction `F/(U+F)` and the holo folded fraction `FL/(UL+FL)`: with
favourable coupling (`w < 0`) the ligand strictly folds the region, and by exactly the box
factor. -/
theorem apo_folded_fraction_lt_holo {eF eL w : ℝ} (hw : w < 0) :
    boxP eF eL w 1 / (boxP eF eL w 0 + boxP eF eL w 1)
      < boxP eF eL w 3 / (boxP eF eL w 2 + boxP eF eL w 3) := by
  have h0 := boxP_pos eF eL w 0
  have h1 := boxP_pos eF eL w 1
  have h2 := boxP_pos eF eL w 2
  have h3 := boxP_pos eF eL w 3
  have hbox := thermodynamic_box eF eL w
  have hexp : (1:ℝ) < Real.exp (-w) := by
    rw [← Real.exp_zero]
    exact Real.exp_lt_exp.mpr (by linarith)
  rw [div_lt_div_iff₀ (by linarith) (by linarith)]
  nlinarith [mul_pos h1 h2, mul_pos h0 h3]

end Linkage

end IDR
