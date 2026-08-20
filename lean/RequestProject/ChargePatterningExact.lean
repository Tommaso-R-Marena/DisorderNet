/-
# Part CXX  Charge patterning: an exact extremal law for a disordered polyampholyte

Almost every disordered region is a **polyampholyte**: it carries positive and negative
charges, usually in near-equal numbers, and the *order* in which those charges appear along
the sequence -- the patterning, not the composition -- is what experiment finds to control
the dimensions of the ensemble.  Two sequences with identical amino-acid composition, one
with its charges well mixed and one with its charges segregated into blocks, behave like
different molecules.  This file proves that statement exactly, for the ideal chain of
`RequestProject.Chain`, with the **mean squared dipole moment** as the observable.

The development has three parts.

## 1.  Two exact algebraic identities

* `neutral_pair_identity` -- for *any* neutral charge assignment `q` and *any* conformation
  `r` whatsoever, `∑_{i<j} q i q j (r i - r j)^2 = -(∑ q k r k)^2`.  The pairwise
  squared-distance coupling of a neutral sequence is minus the squared dipole, exactly,
  with no averaging and no approximation.
* `neutral_lin_kernel` -- for a neutral `q`, `∑_{i<j} q i q j (j - i) = -∑_k (pre q k)^2`,
  where `pre q k` is the charge of the first `k` residues.  The linear-kernel charge
  decoration is minus the sum of squared prefix charges: the "cut" identity.

## 2.  The ideal chain

`mean_dipole_sq` combines them: averaged over the `2^N` conformations of the freely jointed
chain of `RequestProject.Chain`,

    ⟨dipole²⟩ = b² ∑_{k ≤ N} (prefix charge of the first k residues)².

The mean squared dipole of a disordered polyampholyte is a *purely sequence-level* quantity:
the conformational average has been performed exactly.

## 3.  The extremal patterns

Over all neutral `±1` sequences of `m = 2t` residues the sum of squared prefix charges is
bracketed exactly, and both ends are attained:

* `sum_pre_sq_ge` / `sum_pre_sq_alt` -- it is at least `m/2`, with equality for the
  perfectly alternating sequence `+-+-+-...`;
* `sum_pre_sq_le` / `sum_pre_sq_blk` -- it is at most `∑_k min(k, m-k)^2`, with equality for
  the diblock `++...+--...-`.

Hence `dipole_bracket`, and the headline `patterning_amplification`:

    t² · ⟨dipole²⟩(alternating)  ≤  3 · ⟨dipole²⟩(diblock),

i.e. segregating the charges of a fixed composition into two blocks amplifies the mean
squared dipole by a factor growing like the *square* of the chain length.  Composition is
held fixed throughout -- only the order of the residues changes.  This is the exact,
machine-checked form of the experimental rule that charge patterning, and not charge
content, sets the dimensions of a disordered region.
-/
import Mathlib
import RequestProject.Chain

namespace IDR
namespace Charge

open Finset
open scoped Classical

/-! ## 1. Two exact algebraic identities -/

/-- The charge of the first `k` residues. -/
def pre (q : ℕ → ℝ) (k : ℕ) : ℝ := ∑ i ∈ range k, q i

lemma pre_succ (q : ℕ → ℝ) (k : ℕ) : pre q (k + 1) = pre q k + q k :=
  Finset.sum_range_succ _ _

/-- The pairwise squared-distance coupling of a charge assignment, in full generality. -/
lemma pair_identity (q r : ℕ → ℝ) (m : ℕ) :
    2 * ∑ j ∈ range m, ∑ i ∈ range j, q i * q j * (r i - r j) ^ 2
      = 2 * (∑ k ∈ range m, q k * (r k) ^ 2) * pre q m
        - 2 * (∑ k ∈ range m, q k * r k) ^ 2 := by
  induction m with
  | zero => simp [pre]
  | succ m ih =>
      have hexp : ∑ i ∈ range m, q i * q m * (r i - r m) ^ 2
          = q m * (∑ k ∈ range m, q k * (r k) ^ 2)
            - 2 * (q m * r m) * (∑ k ∈ range m, q k * r k)
            + q m * (r m) ^ 2 * pre q m := by
        simp only [pre, Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun i _ => by ring
      rw [Finset.sum_range_succ (f := fun j => ∑ i ∈ range j, q i * q j * (r i - r j) ^ 2),
        Finset.sum_range_succ (f := fun k => q k * (r k) ^ 2),
        Finset.sum_range_succ (f := fun k => q k * r k), pre_succ, hexp]
      linear_combination ih

/-- **The dipole identity.**  For a neutral sequence the pairwise squared-distance coupling
is exactly minus the squared dipole moment -- for every conformation, with no averaging. -/
theorem neutral_pair_identity (q r : ℕ → ℝ) (m : ℕ) (hQ : pre q m = 0) :
    ∑ j ∈ range m, ∑ i ∈ range j, q i * q j * (r i - r j) ^ 2
      = -(∑ k ∈ range m, q k * r k) ^ 2 := by
  have h := pair_identity q r m
  rw [hQ] at h
  linarith

lemma sum_pre (q : ℕ → ℝ) (m : ℕ) :
    ∑ k ∈ range (m + 1), pre q k = ∑ i ∈ range m, q i * ((m : ℝ) - i) := by
  induction m with
  | zero => simp [pre]
  | succ m ih =>
      have h1 : ∑ i ∈ range (m + 1), q i * (((m + 1 : ℕ) : ℝ) - i)
          = ∑ i ∈ range (m + 1), q i * ((m : ℝ) - i) + ∑ i ∈ range (m + 1), q i := by
        rw [← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun i _ => by push_cast; ring
      have h2 : ∑ i ∈ range (m + 1), q i * ((m : ℝ) - i) = ∑ i ∈ range m, q i * ((m : ℝ) - i) := by
        rw [Finset.sum_range_succ]
        simp
      rw [Finset.sum_range_succ (f := fun k => pre q k), ih, h1, h2, pre]

/-- The cut identity, in full generality: the linear-kernel charge decoration in terms of
the prefix charges. -/
lemma lin_kernel_identity (q : ℕ → ℝ) (m : ℕ) :
    ∑ j ∈ range m, ∑ i ∈ range j, q i * q j * ((j : ℝ) - i)
      = ∑ k ∈ range m, pre q k * (pre q m - pre q k) := by
  induction m with
  | zero => simp
  | succ m ih =>
      have hS : pre q (m + 1) = pre q m + q m := pre_succ q m
      have hshift : ∑ k ∈ range m, pre q k * (pre q m + q m - pre q k)
          = ∑ k ∈ range m, pre q k * (pre q m - pre q k) + q m * ∑ k ∈ range m, pre q k := by
        rw [Finset.mul_sum, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun k _ => by ring
      have hnew : ∑ i ∈ range m, q i * q m * ((m : ℝ) - i)
          = q m * ∑ k ∈ range (m + 1), pre q k := by
        rw [sum_pre, Finset.mul_sum]
        exact Finset.sum_congr rfl fun i _ => by ring
      rw [Finset.sum_range_succ (f := fun j => ∑ i ∈ range j, q i * q j * ((j : ℝ) - i)),
        hS, Finset.sum_range_succ (f := fun k => pre q k * (pre q m + q m - pre q k)),
        hshift, hnew, ih, Finset.sum_range_succ (f := fun k => pre q k)]
      ring

/-- **The cut identity for a neutral sequence.**  The linear-kernel charge decoration is
minus the sum of squared prefix charges. -/
theorem neutral_lin_kernel (q : ℕ → ℝ) (m : ℕ) (hQ : pre q m = 0) :
    ∑ j ∈ range m, ∑ i ∈ range j, q i * q j * ((j : ℝ) - i)
      = -∑ k ∈ range m, (pre q k) ^ 2 := by
  rw [lin_kernel_identity, hQ, ← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun k _ => by ring

/-! ## 2. The ideal chain: the conformational average -/

variable {N : ℕ}

/-- The position of residue `k` along the chain axis, in the conformation `s` of the freely
jointed chain of `N` bonds.  Residue `0` sits at the origin. -/
noncomputable def pos (b : ℝ) (s : Chain N) (k : ℕ) : ℝ :=
  ∑ i ∈ univ.filter (fun i : Fin N => (i : ℕ) < k), bondVec b s i

/-- The mean square of any partial sum of bond vectors: bonds are uncorrelated. -/
lemma sum_sq_over_chain (b : ℝ) (A : Finset (Fin N)) :
    ∑ s : Chain N, (∑ i ∈ A, bondVec b s i) ^ 2 = (A.card : ℝ) * 2 ^ N * b ^ 2 := by
  have hexp : ∀ s : Chain N, (∑ i ∈ A, bondVec b s i) ^ 2
      = ∑ i ∈ A, ∑ j ∈ A, bondVec b s i * bondVec b s j := by
    intro s; rw [sq, Finset.sum_mul_sum]
  rw [Finset.sum_congr rfl (fun s _ => hexp s), Finset.sum_comm]
  have hinner : ∀ i ∈ A, ∑ s : Chain N, ∑ j ∈ A, bondVec b s i * bondVec b s j
      = (2 : ℝ) ^ N * b ^ 2 := by
    intro i hi
    rw [Finset.sum_comm]
    have hj : ∀ j ∈ A, ∑ s : Chain N, bondVec b s i * bondVec b s j
        = if j = i then (2 : ℝ) ^ N * b ^ 2 else 0 := by
      intro j _
      by_cases h : j = i
      · subst h; rw [if_pos rfl]; exact sum_bondVec_diag b j
      · rw [if_neg h]; exact sum_bondVec_cross b (Ne.symm h)
    rw [Finset.sum_congr rfl hj, Finset.sum_ite_eq' A i, if_pos hi]
  rw [Finset.sum_congr rfl hinner, Finset.sum_const, nsmul_eq_mul]
  ring

lemma pos_sub (b : ℝ) (s : Chain N) {k j : ℕ} (h : k ≤ j) :
    pos b s j - pos b s k
      = ∑ i ∈ univ.filter (fun i : Fin N => k ≤ (i : ℕ) ∧ (i : ℕ) < j), bondVec b s i := by
  have hsplit : (univ.filter (fun i : Fin N => (i : ℕ) < j))
      = (univ.filter (fun i : Fin N => (i : ℕ) < k))
        ∪ (univ.filter (fun i : Fin N => k ≤ (i : ℕ) ∧ (i : ℕ) < j)) := by
    ext i; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union]; omega
  have hdisj : Disjoint (univ.filter (fun i : Fin N => (i : ℕ) < k))
      (univ.filter (fun i : Fin N => k ≤ (i : ℕ) ∧ (i : ℕ) < j)) := by
    rw [Finset.disjoint_left]
    intro a ha hb
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
    omega
  rw [pos, pos, hsplit, Finset.sum_union hdisj]
  ring

lemma card_seg {k j : ℕ} (hj : j ≤ N) :
    (univ.filter (fun i : Fin N => k ≤ (i : ℕ) ∧ (i : ℕ) < j)).card = j - k := by
  have himg : (univ.filter (fun i : Fin N => k ≤ (i : ℕ) ∧ (i : ℕ) < j)).image Fin.val
      = Finset.Ico k j := by
    ext x
    simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Ico]
    constructor
    · rintro ⟨i, ⟨h1, h2⟩, rfl⟩; exact ⟨h1, h2⟩
    · rintro ⟨h1, h2⟩
      exact ⟨⟨x, by omega⟩, ⟨h1, h2⟩, rfl⟩
  have hcard := Finset.card_image_of_injective
    (univ.filter (fun i : Fin N => k ≤ (i : ℕ) ∧ (i : ℕ) < j)) Fin.val_injective
  rw [himg, Nat.card_Ico] at hcard
  omega

/-- **The random-walk law for internal distances.**  The mean squared distance between
residues `k` and `j` is `|j - k| b²`. -/
lemma sum_pos_sub_sq (b : ℝ) {k j : ℕ} (hkj : k ≤ j) (hj : j ≤ N) :
    ∑ s : Chain N, (pos b s j - pos b s k) ^ 2 = ((j - k : ℕ) : ℝ) * 2 ^ N * b ^ 2 := by
  rw [Finset.sum_congr rfl (fun s (_ : s ∈ (univ : Finset (Chain N))) => by
    rw [pos_sub b s hkj]), sum_sq_over_chain, card_seg hj]

/-- The dipole moment of the charge assignment `q` in the conformation `s`. -/
noncomputable def dipole (q : ℕ → ℝ) (b : ℝ) (s : Chain N) : ℝ :=
  ∑ k ∈ range (N + 1), q k * pos b s k

/-- **The conformational average is exact.**  For a neutral charge assignment the mean
squared dipole moment of the ideal chain is `b²` times the sum of the squared prefix
charges: a pure sequence functional. -/
theorem mean_dipole_sq (q : ℕ → ℝ) (b : ℝ) (hQ : pre q (N + 1) = 0) :
    (chainEns N).expect (fun s => (dipole q b s) ^ 2)
      = b ^ 2 * ∑ k ∈ range (N + 1), (pre q k) ^ 2 := by
  have key : ∀ s : Chain N, (dipole q b s) ^ 2
      = -∑ j ∈ range (N + 1), ∑ i ∈ range j, q i * q j * (pos b s i - pos b s j) ^ 2 := by
    intro s
    rw [neutral_pair_identity q (pos b s) (N + 1) hQ, dipole]
    ring
  have hswap : ∑ s : Chain N,
        ∑ j ∈ range (N + 1), ∑ i ∈ range j, q i * q j * (pos b s i - pos b s j) ^ 2
      = ∑ j ∈ range (N + 1), ∑ i ∈ range j, q i * q j * (((j - i : ℕ) : ℝ) * 2 ^ N * b ^ 2) := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i hi => ?_
    have hij : i ≤ j := le_of_lt (Finset.mem_range.1 hi)
    have hjN : j ≤ N := by have := Finset.mem_range.1 hj; omega
    rw [← Finset.mul_sum, ← sum_pos_sub_sq b hij hjN]
    exact congrArg _ (Finset.sum_congr rfl fun s _ => by ring)
  have hlin : ∑ j ∈ range (N + 1), ∑ i ∈ range j, q i * q j * (((j - i : ℕ) : ℝ) * 2 ^ N * b ^ 2)
      = 2 ^ N * b ^ 2 * (-∑ k ∈ range (N + 1), (pre q k) ^ 2) := by
    rw [← neutral_lin_kernel q (N + 1) hQ, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i hi => ?_
    have hij : i ≤ j := le_of_lt (Finset.mem_range.1 hi)
    have hcast : ((j - i : ℕ) : ℝ) = (j : ℝ) - i := by
      have h := Nat.cast_sub (R := ℝ) hij
      simpa using h
    rw [hcast]; ring
  have hsum : ∑ s : Chain N, (dipole q b s) ^ 2
      = 2 ^ N * (b ^ 2 * ∑ k ∈ range (N + 1), (pre q k) ^ 2) := by
    rw [Finset.sum_congr rfl (fun s (_ : s ∈ (univ : Finset (Chain N))) => key s),
      Finset.sum_neg_distrib, hswap, hlin]
    ring
  rw [chainEns_expect, hsum]
  have h2 : (2 : ℝ) ^ N ≠ 0 := by positivity
  field_simp

/-! ## 3. Extremal charge patterns -/

/-- The prefix charge of an integer-valued charge sequence. -/
def preZ (q : ℕ → ℤ) (k : ℕ) : ℤ := ∑ i ∈ range k, q i

lemma preZ_succ (q : ℕ → ℤ) (k : ℕ) : preZ q (k + 1) = preZ q k + q k :=
  Finset.sum_range_succ _ _

/-- A `±1` charge assignment on the first `m` residues. -/
def IsCharge (m : ℕ) (q : ℕ → ℤ) : Prop := ∀ k, k < m → q k = 1 ∨ q k = -1

lemma pre_cast (q : ℕ → ℤ) (k : ℕ) : pre (fun i => (q i : ℝ)) k = (preZ q k : ℝ) := by
  simp [pre, preZ]

/-- Prefix charges have the parity of their index. -/
lemma preZ_parity {m : ℕ} {q : ℕ → ℤ} (h : IsCharge m q) (k : ℕ) (hk : k ≤ m) :
    Even (preZ q k - k) := by
  revert hk
  induction k with
  | zero => intro _; simp [preZ]
  | succ n ih =>
      intro hk
      obtain ⟨c, hc⟩ := ih (by omega)
      rcases h n (by omega) with h2 | h2
      · exact ⟨c, by rw [preZ_succ, h2]; push_cast; linarith⟩
      · exact ⟨c - 1, by rw [preZ_succ, h2]; push_cast; linarith⟩

lemma preZ_ne_zero_of_odd {m : ℕ} {q : ℕ → ℤ} (h : IsCharge m q) {k : ℕ} (hk : k ≤ m)
    (hodd : ¬ Even k) : preZ q k ≠ 0 := by
  intro h0
  have hp := preZ_parity h k hk
  rw [h0, zero_sub] at hp
  have : Even ((k : ℤ)) := by simpa using hp.neg
  exact hodd (by exact_mod_cast this)

lemma card_odd_range (m : ℕ) : ((range m).filter (fun k => ¬ Even k)).card = m / 2 := by
  induction m with
  | zero => simp
  | succ n ih =>
      rw [Finset.range_add_one, Finset.filter_insert]
      by_cases h : ¬ Even n
      · rw [if_pos h, Finset.card_insert_of_notMem (by simp), ih]
        have : n % 2 = 1 := Nat.odd_iff.1 (Nat.not_even_iff_odd.1 h)
        omega
      · rw [if_neg h, ih]
        have : n % 2 = 0 := Nat.even_iff.1 (not_not.1 h)
        omega

/-- **The mixing bound.**  For every `±1` charge sequence the sum of squared prefix charges
is at least `m/2`: odd prefixes can never be neutral. -/
theorem sum_pre_sq_ge {m : ℕ} {q : ℕ → ℤ} (h : IsCharge m q) :
    ((m / 2 : ℕ) : ℤ) ≤ ∑ k ∈ range m, (preZ q k) ^ 2 := by
  have hstep : ∀ k ∈ (range m).filter (fun k => ¬ Even k), (1 : ℤ) ≤ (preZ q k) ^ 2 := by
    intro k hk
    simp only [Finset.mem_filter, Finset.mem_range] at hk
    have hne : preZ q k ≠ 0 := preZ_ne_zero_of_odd h (le_of_lt hk.1) hk.2
    have h1 : 1 ≤ |preZ q k| := Int.one_le_abs hne
    nlinarith [abs_nonneg (preZ q k), sq_abs (preZ q k)]
  calc ((m / 2 : ℕ) : ℤ)
      = ∑ _k ∈ (range m).filter (fun k => ¬ Even k), (1 : ℤ) := by
        rw [Finset.sum_const, card_odd_range]; simp
    _ ≤ ∑ k ∈ (range m).filter (fun k => ¬ Even k), (preZ q k) ^ 2 :=
        Finset.sum_le_sum hstep
    _ ≤ ∑ k ∈ range m, (preZ q k) ^ 2 :=
        Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
          (fun k _ _ => sq_nonneg _)

lemma preZ_abs_le_left {m : ℕ} {q : ℕ → ℤ} (h : IsCharge m q) {k : ℕ} (hk : k ≤ m) :
    |preZ q k| ≤ (k : ℤ) := by
  have h1 : |preZ q k| ≤ ∑ i ∈ range k, |q i| := Finset.abs_sum_le_sum_abs _ _
  have h2 : ∑ i ∈ range k, |q i| ≤ ∑ _i ∈ range k, (1 : ℤ) := by
    refine Finset.sum_le_sum fun i hi => ?_
    rcases h i (lt_of_lt_of_le (Finset.mem_range.1 hi) hk) with hq | hq <;> rw [hq] <;> simp
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at h2
  omega

lemma preZ_abs_le_right {m : ℕ} {q : ℕ → ℤ} (h : IsCharge m q) (hneutral : preZ q m = 0)
    {k : ℕ} (hk : k ≤ m) : |preZ q k| ≤ ((m - k : ℕ) : ℤ) := by
  have hsplit : ∑ i ∈ Finset.Ico 0 k, q i + ∑ i ∈ Finset.Ico k m, q i = ∑ i ∈ Finset.Ico 0 m, q i :=
    Finset.sum_Ico_consecutive q (Nat.zero_le k) hk
  rw [← Finset.range_eq_Ico] at hsplit
  have hpk : preZ q k + ∑ i ∈ Finset.Ico k m, q i = 0 := by
    rw [preZ] at *
    rw [hsplit]
    exact hneutral
  have habs : |∑ i ∈ Finset.Ico k m, q i| ≤ ∑ i ∈ Finset.Ico k m, |q i| :=
    Finset.abs_sum_le_sum_abs _ _
  have hb : ∑ i ∈ Finset.Ico k m, |q i| ≤ ∑ _i ∈ Finset.Ico k m, (1 : ℤ) := by
    refine Finset.sum_le_sum fun i hi => ?_
    have := Finset.mem_Ico.1 hi
    rcases h i this.2 with hq | hq <;> rw [hq] <;> simp
  simp only [Finset.sum_const, Nat.card_Ico, nsmul_eq_mul, mul_one] at hb
  have : preZ q k = -∑ i ∈ Finset.Ico k m, q i := by omega
  rw [this, abs_neg]
  omega

/-- **The segregation bound.**  Every prefix charge is bounded by the distance to the nearer
end of the chain. -/
theorem sum_pre_sq_le {m : ℕ} {q : ℕ → ℤ} (h : IsCharge m q) (hneutral : preZ q m = 0) :
    ∑ k ∈ range m, (preZ q k) ^ 2 ≤ ∑ k ∈ range m, ((min k (m - k) : ℕ) : ℤ) ^ 2 := by
  refine Finset.sum_le_sum fun k hk => ?_
  have hkm : k ≤ m := le_of_lt (Finset.mem_range.1 hk)
  have h1 : |preZ q k| ≤ (k : ℤ) := preZ_abs_le_left h hkm
  have h2 : |preZ q k| ≤ ((m - k : ℕ) : ℤ) := preZ_abs_le_right h hneutral hkm
  have hmin : |preZ q k| ≤ ((min k (m - k) : ℕ) : ℤ) := by
    rcases le_total k (m - k) with hc | hc
    · rw [Nat.min_eq_left hc]; exact h1
    · rw [Nat.min_eq_right hc]; exact h2
  have habs : (preZ q k) ^ 2 = |preZ q k| ^ 2 := (sq_abs _).symm
  rw [habs]
  exact pow_le_pow_left₀ (abs_nonneg _) hmin 2

/-! ### The alternating and diblock sequences -/

/-- The perfectly mixed sequence `+-+-+-...`. -/
def alt (k : ℕ) : ℤ := if Even k then 1 else -1

/-- The diblock sequence: `t` positive charges followed by `t` negative charges. -/
def blk (t : ℕ) (k : ℕ) : ℤ := if k < t then 1 else -1

lemma alt_isCharge (m : ℕ) : IsCharge m alt := by
  intro k _
  unfold alt
  split
  · exact Or.inl rfl
  · exact Or.inr rfl

lemma blk_isCharge (m t : ℕ) : IsCharge m (blk t) := by
  intro k _
  unfold blk
  split
  · exact Or.inl rfl
  · exact Or.inr rfl

lemma preZ_alt (k : ℕ) : preZ alt k = if Even k then 0 else 1 := by
  induction k with
  | zero => simp [preZ]
  | succ n ih =>
      rw [preZ_succ, ih]
      unfold alt
      by_cases h : Even n
      · simp [h, Nat.even_add_one]
      · simp [h, Nat.even_add_one]

lemma preZ_blk (t : ℕ) : ∀ k, k ≤ 2 * t → preZ (blk t) k = ((min k (2 * t - k) : ℕ) : ℤ) := by
  intro k
  induction k with
  | zero => intro _; simp [preZ]
  | succ n ih =>
      intro hk
      rw [preZ_succ, ih (by omega)]
      unfold blk
      by_cases h : n < t
      · rw [if_pos h]; push_cast; omega
      · rw [if_neg h]; push_cast; omega

lemma alt_neutral {m : ℕ} (hm : Even m) : preZ alt m = 0 := by
  rw [preZ_alt, if_pos hm]

lemma blk_neutral (t : ℕ) : preZ (blk t) (2 * t) = 0 := by
  rw [preZ_blk t (2 * t) le_rfl]
  simp

/-- The alternating sequence attains the mixing bound exactly. -/
theorem sum_pre_sq_alt (m : ℕ) : ∑ k ∈ range m, (preZ alt k) ^ 2 = ((m / 2 : ℕ) : ℤ) := by
  have hterm : ∀ k ∈ range m, (preZ alt k) ^ 2 = if ¬ Even k then (1 : ℤ) else 0 := by
    intro k _
    rw [preZ_alt]
    by_cases h : Even k <;> simp [h]
  rw [Finset.sum_congr rfl hterm, Finset.sum_ite, Finset.sum_const, Finset.sum_const,
    card_odd_range]
  simp

/-- The diblock sequence attains the segregation bound exactly. -/
theorem sum_pre_sq_blk (t : ℕ) :
    ∑ k ∈ range (2 * t), (preZ (blk t) k) ^ 2
      = ∑ k ∈ range (2 * t), ((min k (2 * t - k) : ℕ) : ℤ) ^ 2 := by
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [preZ_blk t k (le_of_lt (Finset.mem_range.1 hk))]

lemma six_sum_sq (n : ℕ) : 6 * ∑ k ∈ range (n + 1), (k : ℤ) ^ 2 = n * (n + 1) * (2 * n + 1) := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_range_succ, mul_add, ih]
      push_cast
      ring

/-- **The diblock is cubically decorated.**  Its sum of squared prefix charges exceeds
`t³/3`. -/
theorem sum_pre_sq_blk_ge (t : ℕ) :
    (t : ℤ) ^ 3 ≤ 3 * ∑ k ∈ range (2 * t), (preZ (blk t) k) ^ 2 := by
  rcases Nat.eq_zero_or_pos t with ht | ht
  · subst ht; simp
  rw [sum_pre_sq_blk]
  have hsub : range (t + 1) ⊆ range (2 * t) := by
    intro x hx
    simp only [Finset.mem_range] at hx ⊢
    omega
  have hsmall : ∀ k ∈ range (t + 1), (k : ℤ) ^ 2 = ((min k (2 * t - k) : ℕ) : ℤ) ^ 2 := by
    intro k hk
    have hkt : k ≤ t := by have := Finset.mem_range.1 hk; omega
    have hmin : min k (2 * t - k) = k := by omega
    rw [hmin]
  have hle : ∑ k ∈ range (t + 1), (k : ℤ) ^ 2
      ≤ ∑ k ∈ range (2 * t), ((min k (2 * t - k) : ℕ) : ℤ) ^ 2 := by
    calc ∑ k ∈ range (t + 1), (k : ℤ) ^ 2
        = ∑ k ∈ range (t + 1), ((min k (2 * t - k) : ℕ) : ℤ) ^ 2 :=
          Finset.sum_congr rfl hsmall
      _ ≤ ∑ k ∈ range (2 * t), ((min k (2 * t - k) : ℕ) : ℤ) ^ 2 :=
          Finset.sum_le_sum_of_subset_of_nonneg hsub (fun k _ _ => sq_nonneg _)
  have hsix := six_sum_sq t
  have htpos : (1 : ℤ) ≤ (t : ℤ) := by exact_mod_cast ht
  nlinarith [hle, hsix, htpos]

/-! ### The physical conclusion -/

/-- **The exact bracket.**  For every neutral `±1` sequence on the `2t` residues of an ideal
chain, the mean squared dipole moment lies between the alternating and the diblock values. -/
theorem dipole_bracket {N t : ℕ} (hm : N + 1 = 2 * t) (q : ℕ → ℤ)
    (h : IsCharge (N + 1) q) (hneutral : preZ q (N + 1) = 0) (b : ℝ) :
    b ^ 2 * (t : ℝ) ≤ (chainEns N).expect (fun s => (dipole (fun k => (q k : ℝ)) b s) ^ 2)
      ∧ (chainEns N).expect (fun s => (dipole (fun k => (q k : ℝ)) b s) ^ 2)
        ≤ b ^ 2 * ((∑ k ∈ range (N + 1), ((min k (N + 1 - k) : ℕ) : ℤ) ^ 2 : ℤ) : ℝ) := by
  have hQ : pre (fun k => (q k : ℝ)) (N + 1) = 0 := by
    rw [pre_cast, hneutral]; simp
  have hmean := mean_dipole_sq (N := N) (fun k => (q k : ℝ)) b hQ
  have hcast : ∑ k ∈ range (N + 1), (pre (fun i => (q i : ℝ)) k) ^ 2
      = ((∑ k ∈ range (N + 1), (preZ q k) ^ 2 : ℤ) : ℝ) := by
    push_cast
    exact Finset.sum_congr rfl fun k _ => by rw [pre_cast]
  rw [hcast] at hmean
  have hlow : ((N + 1) / 2 : ℕ) = t := by omega
  have h1 : ((t : ℤ) : ℝ) ≤ ((∑ k ∈ range (N + 1), (preZ q k) ^ 2 : ℤ) : ℝ) := by
    have := sum_pre_sq_ge h
    rw [hlow] at this
    exact_mod_cast this
  have h2 : ((∑ k ∈ range (N + 1), (preZ q k) ^ 2 : ℤ) : ℝ)
      ≤ ((∑ k ∈ range (N + 1), ((min k (N + 1 - k) : ℕ) : ℤ) ^ 2 : ℤ) : ℝ) := by
    exact_mod_cast sum_pre_sq_le h hneutral
  have hb : (0 : ℝ) ≤ b ^ 2 := sq_nonneg b
  constructor
  · rw [hmean]
    have : (t : ℝ) = ((t : ℤ) : ℝ) := by push_cast; ring
    rw [this]
    exact mul_le_mul_of_nonneg_left h1 hb
  · rw [hmean]
    exact mul_le_mul_of_nonneg_left h2 hb

/-- The mean squared dipole of the perfectly mixed sequence: exactly `t b²` on `2t`
residues -- it grows only *linearly* with the length of the region. -/
theorem mean_dipole_sq_alt {N t : ℕ} (hm : N + 1 = 2 * t) (b : ℝ) :
    (chainEns N).expect (fun s => (dipole (fun k => (alt k : ℝ)) b s) ^ 2) = b ^ 2 * (t : ℝ) := by
  have heven : Even (N + 1) := ⟨t, by omega⟩
  have hQ : pre (fun k => (alt k : ℝ)) (N + 1) = 0 := by
    rw [pre_cast, alt_neutral heven]; simp
  have hmean := mean_dipole_sq (N := N) (fun k => (alt k : ℝ)) b hQ
  have hcast : ∑ k ∈ range (N + 1), (pre (fun i => (alt i : ℝ)) k) ^ 2
      = ((∑ k ∈ range (N + 1), (preZ alt k) ^ 2 : ℤ) : ℝ) := by
    push_cast
    exact Finset.sum_congr rfl fun k _ => by rw [pre_cast]
  rw [hcast, sum_pre_sq_alt] at hmean
  have hlow : ((N + 1) / 2 : ℕ) = t := by omega
  rw [hlow] at hmean
  rw [hmean]
  push_cast
  ring

/-- The mean squared dipole of the diblock: at least `t³ b²/3` on `2t` residues -- it grows
like the *cube* of the length of the region. -/
theorem mean_dipole_sq_blk_ge {N t : ℕ} (hm : N + 1 = 2 * t) (b : ℝ) :
    b ^ 2 * (t : ℝ) ^ 3 / 3
      ≤ (chainEns N).expect (fun s => (dipole (fun k => (blk t k : ℝ)) b s) ^ 2) := by
  have hQ : pre (fun k => (blk t k : ℝ)) (N + 1) = 0 := by
    rw [pre_cast, hm, blk_neutral t]; simp
  have hmean := mean_dipole_sq (N := N) (fun k => (blk t k : ℝ)) b hQ
  have hcast : ∑ k ∈ range (N + 1), (pre (fun i => (blk t i : ℝ)) k) ^ 2
      = ((∑ k ∈ range (N + 1), (preZ (blk t) k) ^ 2 : ℤ) : ℝ) := by
    push_cast
    exact Finset.sum_congr rfl fun k _ => by rw [pre_cast]
  rw [hcast] at hmean
  have hge : ((t : ℤ) ^ 3 : ℤ) ≤ 3 * ∑ k ∈ range (N + 1), (preZ (blk t) k) ^ 2 := by
    rw [hm]; exact sum_pre_sq_blk_ge t
  have hgeR : ((t : ℝ)) ^ 3 ≤ 3 * ((∑ k ∈ range (N + 1), (preZ (blk t) k) ^ 2 : ℤ) : ℝ) := by
    exact_mod_cast hge
  have hb : (0 : ℝ) ≤ b ^ 2 := sq_nonneg b
  rw [hmean]
  nlinarith [hgeR, hb]

/-- **Charge patterning amplifies the dipole quadratically in the chain length.**  At fixed
composition -- exactly `t` positive and `t` negative charges -- segregating the charges into
two blocks multiplies the mean squared dipole moment of the disordered region by a factor
that grows like `t²`.  Sequence order, not sequence content, sets the electrostatic size of
the ensemble. -/
theorem patterning_amplification {N t : ℕ} (hm : N + 1 = 2 * t) (b : ℝ) :
    (t : ℝ) ^ 2 * (chainEns N).expect (fun s => (dipole (fun k => (alt k : ℝ)) b s) ^ 2)
      ≤ 3 * (chainEns N).expect (fun s => (dipole (fun k => (blk t k : ℝ)) b s) ^ 2) := by
  have h1 := mean_dipole_sq_alt hm b
  have h2 := mean_dipole_sq_blk_ge hm b
  rw [h1]
  nlinarith [h2, sq_nonneg b]

end Charge
end IDR
