/-
# Part XXVI.2  Tractability: an energy function is not yet an ensemble

Everything the earlier parts require of a model -- that it emit a distribution, conditioned
on context, with enough capacity to carry the populated conformations -- is a requirement on
what the model *denotes*.  It says nothing about whether the distribution can be *evaluated*.
A residue-level model of a disordered region of `n+1` residues with `k` rotamer states per
residue denotes a distribution on `k^(n+1)` conformations (`card_chain_states`); writing an
energy function down does not by itself produce a single population, because every population
is a ratio of sums with exponentially many terms.

This file isolates the structural property that makes the sums computable, in the general
`k`-state setting (`RequestProject.HelixCoil` solves the two-state Ising chain by a recursion;
here the closed form is proved for arbitrary local state spaces).

* `Z_eq_vecMul_pow` -- **the transfer-matrix theorem**: for any nearest-neighbour weight the
  configuration sum over `k^(n+1)` chains equals `v · Mⁿ · u`, i.e. `n` matrix
  multiplications.  Exponentially many terms, polynomially many operations.
* `Z_pos`, `chainProb_sum_one`, `chainProb_pos` -- the normalized nearest-neighbour Gibbs
  measure is a genuine ensemble, and `Z_boltzmann` states the identity in the form it is used:
  the partition function of a nearest-neighbour energy at inverse temperature `beta`.

Tractability is bought, and the price is stated exactly:

* `chainProb_pairFactored` -- a three-residue nearest-neighbour model makes the two ends
  conditionally independent given the middle residue (`pairFactored_cross`);
* `contactDist_not_pairFactored` -- and an explicit ensemble with a *long-range contact*
  between the two ends (`contactDist_ends_correlated`) admits no nearest-neighbour
  factorization at all, for any transfer matrix and any boundary conditions.

So the design conclusion is a two-sided one.  A model of a disordered region must be
factorized to be usable at all; but a purely local (nearest-neighbour) factorization is a
genuine restriction on the ensembles it can express, and transient long-range contacts --
the characteristic signature of a disordered region that is not a random coil -- live exactly
outside it.  The factorization has to include the contacts one intends to predict.
-/
import Mathlib

namespace IDR

open Finset

namespace Transfer

variable {k : ℕ}

/-! ### The configuration sum of a nearest-neighbour chain -/

/-- The nearest-neighbour weight of a chain configuration `x` of `n+1` residues, each in one
of `k` local states, under the local weight matrix `M`. -/
noncomputable def chainWeight {n : ℕ} (M : Matrix (Fin k) (Fin k) ℝ)
    (x : Fin (n + 1) → Fin k) : ℝ :=
  ∏ i : Fin n, M (x i.castSucc) (x i.succ)

/-- The configuration sum (partition function) of the chain, with boundary weights `v` on the
first residue and `u` on the last. -/
noncomputable def Z {n : ℕ} (v : Fin k → ℝ) (M : Matrix (Fin k) (Fin k) ℝ) (u : Fin k → ℝ) :
    ℝ :=
  ∑ x : Fin (n + 1) → Fin k, v (x 0) * chainWeight M x * u (x (Fin.last n))

/-- The number of chain configurations is exponential in the number of residues. -/
lemma card_chain_states (n : ℕ) : Fintype.card (Fin (n + 1) → Fin k) = k ^ (n + 1) := by
  simp

lemma cons_chain {n : ℕ} (M : Matrix (Fin k) (Fin k) ℝ) (a : Fin k) (y : Fin (n + 1) → Fin k) :
    chainWeight M (Fin.cons a y : Fin (n + 2) → Fin k) = M a (y 0) * chainWeight M y := by
  rw [chainWeight, chainWeight, Fin.prod_univ_succ]
  congr 1

lemma cons_last {n : ℕ} (a : Fin k) (y : Fin (n + 1) → Fin k) :
    (Fin.cons a y : Fin (n + 2) → Fin k) (Fin.last (n + 1)) = y (Fin.last n) := by
  rw [← Fin.succ_last]
  simp

lemma vecMul_apply_sum (v : Fin k → ℝ) (M : Matrix (Fin k) (Fin k) ℝ) (b : Fin k) :
    Matrix.vecMul v M b = ∑ a, v a * M a b := by
  simp [Matrix.vecMul, dotProduct]

/-- **The transfer-matrix theorem.**  The sum over all `k^(n+1)` chain configurations of a
nearest-neighbour weight is the vector-matrix-vector product `v · Mⁿ · u`: it is computed by
`n` matrix multiplications, i.e. in time linear in the length of the region, even though the
sum it evaluates has exponentially many terms. -/
theorem Z_eq_vecMul_pow (M : Matrix (Fin k) (Fin k) ℝ) (u : Fin k → ℝ) :
    ∀ (n : ℕ) (v : Fin k → ℝ),
      (Z (n := n) v M u) = ∑ b, (Matrix.vecMul v (M ^ n)) b * u b := by
  intro n
  induction n with
  | zero =>
      intro v
      simp only [Z, chainWeight, Finset.univ_eq_empty, Finset.prod_empty, pow_zero,
        Matrix.vecMul_one, mul_one]
      refine (Fintype.sum_equiv (Equiv.funUnique (Fin 1) (Fin k)).symm _ _ ?_).symm
      intro a
      simp [Fin.last]
  | succ n ih =>
      intro v
      have hsplit : ∑ p : Fin k × (Fin (n + 1) → Fin k),
          (fun x : Fin (n + 2) → Fin k =>
            v (x 0) * chainWeight M x * u (x (Fin.last (n + 1)))) (Fin.cons p.1 p.2)
          = ∑ x : Fin (n + 2) → Fin k,
              v (x 0) * chainWeight M x * u (x (Fin.last (n + 1))) :=
        Fintype.sum_equiv (Fin.consEquiv (fun _ : Fin (n + 2) => Fin k)) _ _ (fun _ => rfl)
      rw [Z, ← hsplit, Fintype.sum_prod_type_right]
      have hstep : ∀ y : Fin (n + 1) → Fin k,
          ∑ a : Fin k,
            v ((Fin.cons a y : Fin (n + 2) → Fin k) 0)
              * chainWeight M (Fin.cons a y : Fin (n + 2) → Fin k)
              * u ((Fin.cons a y : Fin (n + 2) → Fin k) (Fin.last (n + 1)))
            = (Matrix.vecMul v M) (y 0) * chainWeight M y * u (y (Fin.last n)) := by
        intro y
        simp only [cons_chain, cons_last, Fin.cons_zero]
        rw [vecMul_apply_sum, Finset.sum_mul, Finset.sum_mul]
        exact Finset.sum_congr rfl fun a _ => by ring
      rw [Finset.sum_congr rfl fun y (_ : y ∈ Finset.univ) => hstep y]
      have := ih (Matrix.vecMul v M)
      rw [Z] at this
      rw [this, pow_succ', Matrix.vecMul_vecMul]

/-! ### The nearest-neighbour Gibbs ensemble -/

lemma chainWeight_pos {n : ℕ} {M : Matrix (Fin k) (Fin k) ℝ} (hM : ∀ a b, 0 < M a b)
    (x : Fin (n + 1) → Fin k) : 0 < chainWeight M x :=
  Finset.prod_pos fun _ _ => hM _ _

/-- With strictly positive local weights and boundary conditions and at least one local
state, the configuration sum is strictly positive: the Gibbs ensemble exists. -/
lemma Z_pos {n : ℕ} [NeZero k] {v u : Fin k → ℝ} {M : Matrix (Fin k) (Fin k) ℝ}
    (hv : ∀ a, 0 < v a) (hM : ∀ a b, 0 < M a b) (hu : ∀ a, 0 < u a) :
    0 < Z (n := n) v M u := by
  refine Finset.sum_pos (fun x _ => ?_) ⟨fun _ => (0 : Fin k), Finset.mem_univ _⟩
  exact mul_pos (mul_pos (hv _) (chainWeight_pos hM x)) (hu _)

/-- The normalized nearest-neighbour Gibbs measure on chain configurations. -/
noncomputable def chainProb {n : ℕ} (v : Fin k → ℝ) (M : Matrix (Fin k) (Fin k) ℝ)
    (u : Fin k → ℝ) (x : Fin (n + 1) → Fin k) : ℝ :=
  v (x 0) * chainWeight M x * u (x (Fin.last n)) / Z (n := n) v M u

lemma chainProb_pos {n : ℕ} [NeZero k] {v u : Fin k → ℝ} {M : Matrix (Fin k) (Fin k) ℝ}
    (hv : ∀ a, 0 < v a) (hM : ∀ a b, 0 < M a b) (hu : ∀ a, 0 < u a)
    (x : Fin (n + 1) → Fin k) : 0 < chainProb v M u x :=
  div_pos (mul_pos (mul_pos (hv _) (chainWeight_pos hM x)) (hu _)) (Z_pos hv hM hu)

/-- The nearest-neighbour Gibbs measure is a probability distribution on the `k^(n+1)`
conformations. -/
theorem chainProb_sum_one {n : ℕ} [NeZero k] {v u : Fin k → ℝ} {M : Matrix (Fin k) (Fin k) ℝ}
    (hv : ∀ a, 0 < v a) (hM : ∀ a b, 0 < M a b) (hu : ∀ a, 0 < u a) :
    ∑ x : Fin (n + 1) → Fin k, chainProb v M u x = 1 := by
  simp only [chainProb]
  rw [← Finset.sum_div]
  exact div_self (Z_pos (n := n) hv hM hu).ne'

/-- The transfer matrix of a nearest-neighbour energy at inverse temperature `beta`. -/
noncomputable def boltzmannTransfer (beta : ℝ) (E : Fin k → Fin k → ℝ) :
    Matrix (Fin k) (Fin k) ℝ :=
  Matrix.of fun a b => Real.exp (-beta * E a b)

lemma boltzmannTransfer_pos (beta : ℝ) (E : Fin k → Fin k → ℝ) (a b : Fin k) :
    0 < boltzmannTransfer beta E a b := Real.exp_pos _

/-- **The partition function of a nearest-neighbour chain energy is a matrix product.**  The
Boltzmann sum `∑_x exp(-beta ∑_i E(x_i, x_{i+1}))` over `k^(n+1)` conformations equals
`1 · (e^{-beta E})ⁿ · 1`. -/
theorem Z_boltzmann (n : ℕ) (beta : ℝ) (E : Fin k → Fin k → ℝ) :
    ∑ x : Fin (n + 1) → Fin k,
        Real.exp (-beta * ∑ i : Fin n, E (x i.castSucc) (x i.succ))
      = ∑ b, (Matrix.vecMul (fun _ => (1:ℝ)) ((boltzmannTransfer beta E) ^ n)) b := by
  have hZ := Z_eq_vecMul_pow (boltzmannTransfer beta E) (fun _ => (1:ℝ)) n (fun _ => (1:ℝ))
  rw [Z] at hZ
  have hleft : ∀ x : Fin (n + 1) → Fin k,
      Real.exp (-beta * ∑ i : Fin n, E (x i.castSucc) (x i.succ))
        = (fun _ => (1:ℝ)) (x 0) * chainWeight (boltzmannTransfer beta E) x
            * (fun _ => (1:ℝ)) (x (Fin.last n)) := by
    intro x
    simp only [one_mul, mul_one, chainWeight, boltzmannTransfer, Matrix.of_apply]
    rw [Finset.mul_sum, Real.exp_sum]
  rw [Finset.sum_congr rfl fun x (_ : x ∈ Finset.univ) => hleft x, hZ]
  simp

/-! ### What a nearest-neighbour factorization cannot express -/

/-- A distribution on three-residue configurations is *pair-factored* when it is a product of
a weight on the first bond and a weight on the second: the general shape of a
nearest-neighbour model, with the boundary conditions absorbed into the bond weights. -/
def PairFactored (P : (Fin 3 → Fin 2) → ℝ) : Prop :=
  ∃ f g : Fin 2 → Fin 2 → ℝ, ∀ x, P x = f (x 0) (x 1) * g (x 1) (x 2)

/-- Every nearest-neighbour Gibbs measure on three residues is pair-factored. -/
theorem chainProb_pairFactored (v u : Fin 2 → ℝ) (M : Matrix (Fin 2) (Fin 2) ℝ) :
    PairFactored (fun x => chainProb (n := 2) v M u x) := by
  refine ⟨fun a b => v a * M a b / Z (n := 2) v M u, fun b c => M b c * u c, fun x => ?_⟩
  have hlast : (Fin.last 2 : Fin 3) = 2 := rfl
  have hw : chainWeight (n := 2) M x = M (x 0) (x 1) * M (x 1) (x 2) := by
    rw [chainWeight, Fin.prod_univ_two]
    rfl
  simp only [chainProb]
  rw [hw, hlast]
  field_simp

/-- **Conditional independence of the ends.**  In a pair-factored model the two end residues
are independent given the middle one: the cross products of the four configurations that
share a middle state agree. -/
theorem pairFactored_cross {P : (Fin 3 → Fin 2) → ℝ} (h : PairFactored P)
    (a a' b c c' : Fin 2) :
    P ![a, b, c] * P ![a', b, c'] = P ![a, b, c'] * P ![a', b, c] := by
  obtain ⟨f, g, hfg⟩ := h
  simp only [hfg, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two]
  simp only [Matrix.head_cons, Matrix.tail_cons]
  ring

/-- An ensemble of three residues with a **long-range contact**: the two ends are pinned to
the same state, while the middle residue is free.  This is the caricature of a transient
tertiary contact in an otherwise disordered region. -/
noncomputable def contactDist : (Fin 3 → Fin 2) → ℝ :=
  fun x => if x 0 = x 2 then (1:ℝ)/4 else 0

lemma contactDist_nonneg (x : Fin 3 → Fin 2) : 0 ≤ contactDist x := by
  rw [contactDist]
  split <;> norm_num

/-- It is a probability distribution. -/
theorem contactDist_sum_one : ∑ x : Fin 3 → Fin 2, contactDist x = 1 := by
  have hcard : (Finset.univ.filter (fun x : Fin 3 → Fin 2 => x 0 = x 2)).card = 4 := by decide
  simp only [contactDist]
  rw [Finset.sum_ite, Finset.sum_const, Finset.sum_const_zero, hcard]
  norm_num

/-- Sums of the contact ensemble over events are counts of configurations. -/
lemma contactDist_event (S : (Fin 3 → Fin 2) → Prop) [DecidablePred S] :
    ∑ x ∈ Finset.univ.filter S, contactDist x
      = ((Finset.univ.filter (fun x : Fin 3 → Fin 2 => S x ∧ x 0 = x 2)).card : ℝ) / 4 := by
  simp only [contactDist]
  rw [Finset.sum_ite, Finset.sum_const, Finset.sum_const_zero, Finset.filter_filter,
    nsmul_eq_mul, add_zero]
  ring

/-- The two ends really are correlated: the joint population of the doubly-occupied contact
state is `1/2`, while the product of the two marginals is `1/4`. -/
theorem contactDist_ends_correlated :
    (∑ x ∈ Finset.univ.filter (fun x : Fin 3 → Fin 2 => x 0 = 1 ∧ x 2 = 1), contactDist x)
      ≠ (∑ x ∈ Finset.univ.filter (fun x : Fin 3 → Fin 2 => x 0 = 1), contactDist x)
        * (∑ x ∈ Finset.univ.filter (fun x : Fin 3 → Fin 2 => x 2 = 1), contactDist x) := by
  have c1 : (Finset.univ.filter
      (fun x : Fin 3 → Fin 2 => (x 0 = 1 ∧ x 2 = 1) ∧ x 0 = x 2)).card = 2 := by decide
  have c2 : (Finset.univ.filter
      (fun x : Fin 3 → Fin 2 => x 0 = 1 ∧ x 0 = x 2)).card = 2 := by decide
  have c3 : (Finset.univ.filter
      (fun x : Fin 3 → Fin 2 => x 2 = 1 ∧ x 0 = x 2)).card = 2 := by decide
  rw [contactDist_event, contactDist_event, contactDist_event, c1, c2, c3]
  norm_num

/-- **A long-range contact has no nearest-neighbour factorization.**  No transfer matrix and
no boundary conditions -- indeed no pair of bond weights whatsoever -- reproduce the contact
ensemble. -/
theorem contactDist_not_pairFactored : ¬ PairFactored contactDist := by
  intro h
  have hcross := pairFactored_cross h 0 1 0 0 1
  have h00 : contactDist ![0, 0, 0] = 1/4 := by
    simp [contactDist, Matrix.cons_val_two, Matrix.tail_cons]
  have h11 : contactDist ![1, 0, 1] = 1/4 := by
    simp [contactDist, Matrix.cons_val_two, Matrix.tail_cons]
  have h01 : contactDist ![0, 0, 1] = 0 := by
    simp [contactDist, Matrix.cons_val_two, Matrix.tail_cons]
  rw [h00, h11, h01] at hcross
  norm_num at hcross

/-- **Corollary: a three-residue nearest-neighbour Gibbs model cannot express the contact.**
For every transfer matrix and all boundary weights, the model's distribution differs from the
contact ensemble at some configuration. -/
theorem chainProb_ne_contactDist (v u : Fin 2 → ℝ) (M : Matrix (Fin 2) (Fin 2) ℝ) :
    ∃ x : Fin 3 → Fin 2, chainProb (n := 2) v M u x ≠ contactDist x := by
  by_contra hcon
  push_neg at hcon
  have : PairFactored contactDist := by
    obtain ⟨f, g, hfg⟩ := chainProb_pairFactored v u M
    exact ⟨f, g, fun x => by rw [← hcon x]; exact hfg x⟩
  exact contactDist_not_pairFactored this

end Transfer

end IDR
