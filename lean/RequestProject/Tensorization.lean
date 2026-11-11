/-
# Part XXVI.1  Tensorization: the price of a mismatched reference grows with the chain

`RequestProject.Reweighting` prices one act of importance reweighting: `N` frames of a
reference ensemble `q` are worth `N / (1 + chiSq p q)` frames of the target `p`.  That
statement is about a single random variable.  A disordered region is not a single random
variable: it is a chain of many residues, and the reference ensemble a force field provides
is (to first approximation) a *product* of per-residue laws.

This file computes what happens to the price when the same small per-residue mismatch is
repeated along the chain.  The answer is exact and multiplicative:

* `sum_sq_div_prodDist` -- the second moment of the importance weight factorizes over sites;
* `chiSq_prodDist` -- `1 + χ²` of a product is the **product** of the per-site `1 + χ²`;
* `chiSq_prodDist_iid`, `essFrac_prodDist_iid` -- for `n` identical sites with per-site
  mismatch `c > 0` the effective sample size fraction is exactly `(1 + c)^(-n)`, so the
  number of frames needed is `(1 + c)^n`: **exponential in the length of the region**.
* `kl_prodDist` -- relative entropy, by contrast, is *additive*: `n · K` nats.  Combined with
  `Reweight.essFrac_le_exp_neg_kl` this is the same exponential seen from the entropy side.

Two further lemmas make the product model itself usable rather than only expensive:
`prodDist_sum_one` (it is a probability distribution) and `sum_prodDist_local` (a one-site
observable is averaged by an `O(n k)` computation, not by a sum over `k^n` conformations).

The moral for model design: a reference ensemble that is wrong by a fixed amount *per
residue* cannot be repaired by reweighting on any realistic budget.  Reweighting is a local
correction; a systematically mis-specified chain model must be refitted, not reweighted.
-/
import Mathlib
import RequestProject.Reweighting

namespace IDR

open Finset
open scoped Classical

namespace Tensor

/-! ### Chi-squared divergence on an arbitrary finite state space -/

variable {α : Type*} [Fintype α]

/-- The chi-squared divergence of `p` from `q` on an arbitrary finite state space. -/
noncomputable def chiSqT (p q : α → ℝ) : ℝ := ∑ x, (p x - q x) ^ 2 / q x

lemma chiSqT_nonneg {p q : α → ℝ} (hq : ∀ x, 0 < q x) : 0 ≤ chiSqT p q :=
  Finset.sum_nonneg fun x _ => div_nonneg (sq_nonneg _) (hq x).le

/-- `∑ p²/q = 1 + χ²` on any finite state space. -/
lemma sum_sq_div_gen {p q : α → ℝ} (hq : ∀ x, 0 < q x) (hps : ∑ x, p x = 1)
    (hqs : ∑ x, q x = 1) :
    ∑ x, p x ^ 2 / q x = 1 + chiSqT p q := by
  have key : ∀ x : α, p x ^ 2 / q x = (p x - q x) ^ 2 / q x + (2 * p x - q x) := by
    intro x
    have h := (hq x).ne'
    field_simp
    ring
  rw [Finset.sum_congr rfl fun x (_ : x ∈ Finset.univ) => key x, Finset.sum_add_distrib,
    ← chiSqT, Finset.sum_sub_distrib, ← Finset.mul_sum, hps, hqs]
  ring

/-- The generic chi-squared divergence agrees with the `Fin m`-indexed one of
`RequestProject.Reweighting` under any relabelling of the state space. -/
lemma chiSqT_eq_chiSq {m : ℕ} (e : α ≃ Fin m) (p q : α → ℝ) :
    chiSqT p q = Reweight.chiSq (fun j => p (e.symm j)) (fun j => q (e.symm j)) := by
  rw [chiSqT, Reweight.chiSq]
  exact Fintype.sum_equiv e _ _ (fun x => by rw [Equiv.symm_apply_apply])

/-- Kish's effective sample size fraction on an arbitrary finite state space. -/
noncomputable def essFracT (p q : α → ℝ) : ℝ :=
  (∑ x, q x * (p x / q x)) ^ 2 / (∑ x, q x * (p x / q x) ^ 2)

/-- The effective sample size fraction is exactly `1 / (1 + χ²)`. -/
theorem essFracT_eq {p q : α → ℝ} (hq : ∀ x, 0 < q x) (hps : ∑ x, p x = 1)
    (hqs : ∑ x, q x = 1) :
    essFracT p q = 1 / (1 + chiSqT p q) := by
  have h1 : ∑ x, q x * (p x / q x) = 1 := by
    rw [← hps]
    exact Finset.sum_congr rfl fun x _ => by
      rw [mul_comm, div_mul_cancel₀ _ (hq x).ne']
  have h2 : ∑ x, q x * (p x / q x) ^ 2 = 1 + chiSqT p q := by
    rw [← sum_sq_div_gen hq hps hqs]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [div_pow]
    field_simp
  rw [essFracT, h1, h2]
  norm_num

/-! ### Product (independent-site) models -/

variable {n k : ℕ}

/-- The product distribution built from `n` per-site laws on `k` rotamer states: the law of
a chain whose residues are statistically independent. -/
noncomputable def prodDist (p : Fin n → Fin k → ℝ) : (Fin n → Fin k) → ℝ :=
  fun x => ∏ i, p i (x i)

@[simp] lemma prodDist_apply (p : Fin n → Fin k → ℝ) (x : Fin n → Fin k) :
    prodDist p x = ∏ i, p i (x i) := rfl

/-- The state space of a chain of `n` residues with `k` rotamers each has `k ^ n` points:
an explicit table of conformations is impossible for any realistic region. -/
lemma card_chain_states : Fintype.card (Fin n → Fin k) = k ^ n := by
  simp

/-- A product of per-site probability vectors is a probability distribution. -/
lemma prodDist_sum_one {p : Fin n → Fin k → ℝ} (h : ∀ i, ∑ s, p i s = 1) :
    ∑ x : Fin n → Fin k, prodDist p x = 1 := by
  have hprod := (Finset.prod_univ_sum (fun _ : Fin n => (Finset.univ : Finset (Fin k))) p).symm
  simp only [Fintype.piFinset_univ] at hprod
  rw [show (∑ x : Fin n → Fin k, prodDist p x) = ∑ x : Fin n → Fin k, ∏ i, p i (x i) from rfl,
    hprod]
  simp [h]

lemma prodDist_pos {p : Fin n → Fin k → ℝ} (h : ∀ i s, 0 < p i s) (x : Fin n → Fin k) :
    0 < prodDist p x :=
  Finset.prod_pos fun i _ => h i (x i)

/-- **A one-site observable of a product model is computed site-locally.**  Averaging `g`
applied to residue `i` costs one sum of `k` terms, not a sum over `k ^ n` conformations. -/
theorem sum_prodDist_local {p : Fin n → Fin k → ℝ} (h : ∀ i, ∑ s, p i s = 1)
    (i : Fin n) (g : Fin k → ℝ) :
    ∑ x : Fin n → Fin k, prodDist p x * g (x i) = ∑ s, p i s * g s := by
  classical
  set h' : Fin n → Fin k → ℝ := fun j s => if j = i then p j s * g s else p j s with hh'
  have hterm : ∀ x : Fin n → Fin k, prodDist p x * g (x i) = ∏ j, h' j (x j) := by
    intro x
    have : ∏ j, h' j (x j)
        = h' i (x i) * ∏ j ∈ Finset.univ.erase i, h' j (x j) := by
      rw [← Finset.prod_erase_mul _ _ (Finset.mem_univ i)]
      ring
    rw [this]
    have he : ∀ j ∈ Finset.univ.erase i, h' j (x j) = p j (x j) := by
      intro j hj
      simp [hh', Finset.ne_of_mem_erase hj]
    rw [Finset.prod_congr rfl he, prodDist_apply,
      ← Finset.prod_erase_mul _ _ (Finset.mem_univ i)]
    simp [hh']
    ring
  have hprod := (Finset.prod_univ_sum (fun _ : Fin n => (Finset.univ : Finset (Fin k))) h').symm
  simp only [Fintype.piFinset_univ] at hprod
  rw [Finset.sum_congr rfl fun x (_ : x ∈ Finset.univ) => hterm x, hprod]
  have hone : ∀ j, j ≠ i → ∑ s, h' j s = 1 := by
    intro j hj
    simp [hh', hj, h j]
  rw [← Finset.prod_erase_mul _ _ (Finset.mem_univ i)]
  rw [Finset.prod_congr rfl (fun j hj => hone j (Finset.ne_of_mem_erase hj))]
  simp [hh']

/-! ### Tensorization of the reweighting cost -/

/-- The second moment of the importance weight factorizes over sites. -/
theorem sum_sq_div_prodDist (p q : Fin n → Fin k → ℝ) :
    ∑ x : Fin n → Fin k, (prodDist p x) ^ 2 / (prodDist q x)
      = ∏ i, ∑ s, (p i s) ^ 2 / (q i s) := by
  have hterm : ∀ x : Fin n → Fin k,
      (prodDist p x) ^ 2 / (prodDist q x) = ∏ i, (p i (x i)) ^ 2 / (q i (x i)) := by
    intro x
    rw [prodDist_apply, prodDist_apply, ← Finset.prod_pow, ← Finset.prod_div_distrib]
  have hprod := (Finset.prod_univ_sum (fun _ : Fin n => (Finset.univ : Finset (Fin k)))
    (fun i s => (p i s) ^ 2 / (q i s))).symm
  simp only [Fintype.piFinset_univ] at hprod
  rw [Finset.sum_congr rfl fun x (_ : x ∈ Finset.univ) => hterm x, ← hprod]

/-- **Tensorization of the chi-squared divergence.**  For product models the reweighting
cost multiplies over sites:  `1 + χ²(P‖Q) = ∏ᵢ (1 + χ²(pᵢ‖qᵢ))`. -/
theorem chiSq_prodDist {p q : Fin n → Fin k → ℝ} (hq : ∀ i s, 0 < q i s)
    (hps : ∀ i, ∑ s, p i s = 1) (hqs : ∀ i, ∑ s, q i s = 1) :
    1 + chiSqT (prodDist p) (prodDist q) = ∏ i, (1 + chiSqT (p i) (q i)) := by
  rw [← sum_sq_div_gen (fun x => prodDist_pos hq x) (prodDist_sum_one hps)
      (prodDist_sum_one hqs),
    sum_sq_div_prodDist p q]
  exact Finset.prod_congr rfl fun i _ => sum_sq_div_gen (hq i) (hps i) (hqs i)

/-- **The same mismatch at every residue costs `(1 + c)^n`.**  With identical per-site laws
`p₀ ≠ q₀` of per-site chi-squared `c`, the chain-level cost is `1 + χ² = (1 + c)^n`. -/
theorem chiSq_prodDist_iid {p₀ q₀ : Fin k → ℝ} (hq : ∀ s, 0 < q₀ s) (hps : ∑ s, p₀ s = 1)
    (hqs : ∑ s, q₀ s = 1) (n : ℕ) :
    1 + chiSqT (prodDist (fun _ : Fin n => p₀)) (prodDist (fun _ : Fin n => q₀))
      = (1 + chiSqT p₀ q₀) ^ n := by
  rw [chiSq_prodDist (fun _ s => hq s) (fun _ => hps) (fun _ => hqs)]
  simp

/-- **The effective sample size decays exponentially in the length of the region.** -/
theorem essFrac_prodDist_iid {p₀ q₀ : Fin k → ℝ} (hq : ∀ s, 0 < q₀ s) (hps : ∑ s, p₀ s = 1)
    (hqs : ∑ s, q₀ s = 1) (n : ℕ) :
    essFracT (prodDist (fun _ : Fin n => p₀)) (prodDist (fun _ : Fin n => q₀))
      = 1 / (1 + chiSqT p₀ q₀) ^ n := by
  rw [essFracT_eq (fun x => prodDist_pos (fun _ s => hq s) x)
      (prodDist_sum_one (fun _ => hps)) (prodDist_sum_one (fun _ => hqs)),
    chiSq_prodDist_iid hq hps hqs n]

/-- **How many frames a chain-length mismatch costs.**  To keep `neff` effective frames of
the target after reweighting a product reference with per-residue mismatch `c > 0`, a
simulation of at least `neff · (1 + c)^n` frames is required. -/
theorem frames_needed_prodDist {p₀ q₀ : Fin k → ℝ} (hq : ∀ s, 0 < q₀ s) (hps : ∑ s, p₀ s = 1)
    (hqs : ∑ s, q₀ s = 1) (n : ℕ) {N neff : ℝ}
    (h : neff ≤ N * essFracT (prodDist (fun _ : Fin n => p₀)) (prodDist (fun _ : Fin n => q₀))) :
    neff * (1 + chiSqT p₀ q₀) ^ n ≤ N := by
  have hc : 0 ≤ chiSqT p₀ q₀ := chiSqT_nonneg hq
  have hpos : (0:ℝ) < (1 + chiSqT p₀ q₀) ^ n := by positivity
  rw [essFrac_prodDist_iid hq hps hqs n] at h
  have := mul_le_mul_of_nonneg_right h hpos.le
  rw [mul_assoc, one_div, inv_mul_cancel₀ hpos.ne', mul_one] at this
  exact this

/-! ### Relative entropy is additive -/

/-- Relative entropy on an arbitrary finite state space. -/
noncomputable def klT (p q : α → ℝ) : ℝ := ∑ x, p x * Real.log (p x / q x)

/-- **Relative entropy is additive over sites.**  A per-residue mismatch of `K` nats is a
chain-level mismatch of `n · K` nats. -/
theorem kl_prodDist {p q : Fin n → Fin k → ℝ} (hp : ∀ i s, 0 < p i s) (hq : ∀ i s, 0 < q i s)
    (hps : ∀ i, ∑ s, p i s = 1) :
    klT (prodDist p) (prodDist q) = ∑ i, klT (p i) (q i) := by
  have hlog : ∀ x : Fin n → Fin k,
      prodDist p x * Real.log (prodDist p x / prodDist q x)
        = ∑ i, prodDist p x * Real.log (p i (x i) / q i (x i)) := by
    intro x
    rw [← Finset.mul_sum]
    congr 1
    rw [prodDist_apply, prodDist_apply, ← Finset.prod_div_distrib,
      Real.log_prod (fun i _ => (div_pos (hp i (x i)) (hq i (x i))).ne')]
  rw [klT, Finset.sum_congr rfl fun x (_ : x ∈ Finset.univ) => hlog x, Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => ?_
  exact sum_prodDist_local hps i (fun s => Real.log (p i s / q i s))

end Tensor

end IDR
