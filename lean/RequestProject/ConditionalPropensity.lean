/-
# Conditional propensities: folding as a chain of conditional probabilities

A residue's contribution to folding is not a property of the residue: it is a property of the
residue *given its neighbours*.  Hydrophobic–polar patterning, the helix propensity of a pair,
beta-branching next to a proline — all of these are statements about conditional probabilities
`P(x_{i+1} | x_i)`, not about single-residue frequencies.  A model built on single-residue
composition therefore describes a strictly larger, strictly less structured ensemble than the
real one.

This file makes that exact, in nats.

* `IDR.CondSeq.H_kernel` — the **chain rule**: for a joint law `P(a,b) = p(a)·T(a,b)`,
  `H(P) = H(p) + ∑_a p(a)·H(T(a,·))`.
* `IDR.CondSeq.H_le_marginals`, `mutualInfo_nonneg` — **subadditivity**: a joint law never has
  more entropy than its marginals separately, the deficit being the mutual information
  `I = H(marg₁) + H(marg₂) − H(P) ≥ 0`.
* `IDR.CondSeq.condH_le_single`, `condH_eq_sub_mutualInfo` — **conditioning reduces entropy, by
  exactly `I`**: the entropy of the next residue *given* the previous one is `H(p) − I`.
* `IDR.CondSeq.chain`, `blockH_eq`, `block_entropy_deficit` — the length-`n+1` Markov source has
  block entropy `H(p) + n·(H(p) − I)`, so it falls short of the composition-matched independent
  model by **exactly `n·I` nats**: the pair model describes `exp(−n·I)` as many sequences.
* `IDR.CondSeq.hp_mutualInfo`, `hp_quarter_mutualInfo_pos` — a two-letter
  hydrophobic/polar chain with switch probability `e` has `I = log 2 − h₂(e)`; at `e = 1/4`
  (a mildly blocky HP pattern) `I = (3/4)·log 3 − log 2 ≈ 0.131` nats per residue, so over a
  100-residue region the conditional model is smaller than the composition model by a factor
  `e^{13}`.

The design consequence: the sequence-entropy ceiling of `RequestProject.SequenceEntropyLimit`
must be evaluated with the **conditional** entropy rate.  Using single-residue entropy
overestimates the ensemble by `exp(n·I)` and therefore systematically over-predicts disorder.
-/
import Mathlib
import RequestProject.SequenceEntropyCore

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR

namespace CondSeq

open Finset
open SeqEnt

/-! ## Joint laws, marginals and mutual information -/

section Pair

variable {A B : Type*} [Fintype A] [Fintype B]

/-- The marginal of a joint law on the first coordinate. -/
def marg1 (P : A × B → ℝ) : A → ℝ := fun a => ∑ b, P (a, b)

/-- The marginal of a joint law on the second coordinate. -/
def marg2 (P : A × B → ℝ) : B → ℝ := fun b => ∑ a, P (a, b)

/-- The mutual information of a joint law: the entropy a factorised model would assign minus the
entropy the joint law actually has. -/
noncomputable def mutualInfo (P : A × B → ℝ) : ℝ := H (marg1 P) + H (marg2 P) - H P

/-- **The chain rule for entropy.**  A conditional model `P(a,b) = p(a)·T(a,b)` has entropy equal
to the entropy of the first letter plus the mean entropy of the conditional law of the second. -/
theorem H_kernel {p : A → ℝ} {T : A → B → ℝ} (hp : ∀ a, 0 < p a)
    (hT : ∀ a b, 0 < T a b) (hTs : ∀ a, ∑ b, T a b = 1) :
    H (fun z : A × B => p z.1 * T z.1 z.2) = H p + ∑ a, p a * H (T a) := by
  have hterm : ∀ a : A, ∑ b, -(p a * T a b * Real.log (p a * T a b))
      = -(p a * Real.log (p a)) + p a * H (T a) := by
    intro a
    have hlog : ∀ b, Real.log (p a * T a b) = Real.log (p a) + Real.log (T a b) := fun b =>
      Real.log_mul (ne_of_gt (hp a)) (ne_of_gt (hT a b))
    calc ∑ b, -(p a * T a b * Real.log (p a * T a b))
        = ∑ b, ((-(p a * Real.log (p a))) * T a b + p a * -(T a b * Real.log (T a b))) := by
          refine Finset.sum_congr rfl fun b _ => ?_
          rw [hlog b]; ring
      _ = (-(p a * Real.log (p a))) * (∑ b, T a b) + p a * ∑ b, -(T a b * Real.log (T a b)) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
      _ = -(p a * Real.log (p a)) + p a * H (T a) := by rw [hTs a]; simp [H]
  calc H (fun z : A × B => p z.1 * T z.1 z.2)
      = ∑ a, ∑ b, -(p a * T a b * Real.log (p a * T a b)) := by
        rw [H, Fintype.sum_prod_type]
    _ = ∑ a, (-(p a * Real.log (p a)) + p a * H (T a)) := Finset.sum_congr rfl fun a _ => hterm a
    _ = H p + ∑ a, p a * H (T a) := by rw [Finset.sum_add_distrib, H]

/-- The entropy of a product law is the sum of the entropies: an independent-site model has no
conditional structure to exploit. -/
theorem H_prod {p : A → ℝ} {r : B → ℝ} (hp : ∀ a, 0 < p a) (hr : ∀ b, 0 < r b)
    (hrs : ∑ b, r b = 1) (hps : ∑ a, p a = 1) :
    H (fun z : A × B => p z.1 * r z.2) = H p + H r := by
  have := H_kernel (T := fun (_ : A) (b : B) => r b) hp (fun _ b => hr b) (fun _ => hrs)
  rw [this, ← Finset.sum_mul, hps, one_mul]

/-- **Subadditivity of entropy.**  A joint law has at most the entropy of its marginals taken
separately. -/
theorem H_le_marginals {P : A × B → ℝ} (hP : ∀ z, 0 ≤ P z) (hsum : ∑ z, P z = 1)
    (h1 : ∀ a, 0 < marg1 P a) (h2 : ∀ b, 0 < marg2 P b) :
    H P ≤ H (marg1 P) + H (marg2 P) := by
  have hu : ∀ z : A × B, 0 ≤ marg1 P z.1 * marg2 P z.2 := fun z =>
    le_of_lt (mul_pos (h1 z.1) (h2 z.2))
  have hm1 : ∑ a, marg1 P a = 1 := by
    simp only [marg1]; rw [← Fintype.sum_prod_type]; exact hsum
  have hm2 : ∑ b, marg2 P b = 1 := by
    simp only [marg2]; rw [← Fintype.sum_prod_type_right]; exact hsum
  have hus : ∑ z : A × B, marg1 P z.1 * marg2 P z.2 ≤ 1 := by
    rw [Fintype.sum_prod_type]
    have hsplit : ∑ a, ∑ b, marg1 P a * marg2 P b = ∑ a, marg1 P a * ∑ b, marg2 P b :=
      Finset.sum_congr rfl fun a _ => (Finset.mul_sum _ _ _).symm
    rw [hsplit, hm2]
    simp [hm1]
  have hgibbs := gibbs hP hsum hu hus (fun z _ => ne_of_gt (mul_pos (h1 z.1) (h2 z.2)))
  refine le_trans hgibbs (le_of_eq ?_)
  have hlog : ∀ z : A × B, -(P z * Real.log (marg1 P z.1 * marg2 P z.2))
      = -(P z * Real.log (marg1 P z.1)) + -(P z * Real.log (marg2 P z.2)) := by
    intro z
    rw [Real.log_mul (ne_of_gt (h1 z.1)) (ne_of_gt (h2 z.2))]
    ring
  calc ∑ z : A × B, -(P z * Real.log (marg1 P z.1 * marg2 P z.2))
      = (∑ z : A × B, -(P z * Real.log (marg1 P z.1)))
          + ∑ z : A × B, -(P z * Real.log (marg2 P z.2)) := by
        rw [← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun z _ => hlog z
    _ = H (marg1 P) + H (marg2 P) := by
        congr 1
        · rw [Fintype.sum_prod_type, H]
          refine Finset.sum_congr rfl fun a _ => ?_
          calc ∑ y, -(P (a, y) * Real.log (marg1 P a))
              = -((∑ y, P (a, y)) * Real.log (marg1 P a)) := by
                rw [Finset.sum_mul, ← Finset.sum_neg_distrib]
            _ = -(marg1 P a * Real.log (marg1 P a)) := rfl
        · rw [Fintype.sum_prod_type_right, H]
          refine Finset.sum_congr rfl fun b _ => ?_
          calc ∑ x, -(P (x, b) * Real.log (marg2 P b))
              = -((∑ x, P (x, b)) * Real.log (marg2 P b)) := by
                rw [Finset.sum_mul, ← Finset.sum_neg_distrib]
            _ = -(marg2 P b * Real.log (marg2 P b)) := rfl

/-- **Mutual information is nonnegative**: conditional structure can only remove entropy. -/
theorem mutualInfo_nonneg {P : A × B → ℝ} (hP : ∀ z, 0 ≤ P z) (hsum : ∑ z, P z = 1)
    (h1 : ∀ a, 0 < marg1 P a) (h2 : ∀ b, 0 < marg2 P b) : 0 ≤ mutualInfo P := by
  have := H_le_marginals hP hsum h1 h2
  unfold mutualInfo
  linarith

end Pair

/-! ## A stationary pair model of a sequence -/

section Markov

variable {A : Type*} [Fintype A]

/-- The dimer law of a pair model: first residue from `p`, second from the conditional law `T`. -/
noncomputable def dimer (p : A → ℝ) (T : A → A → ℝ) : A × A → ℝ := fun z => p z.1 * T z.1 z.2

/-- The conditional (per-residue) entropy of the pair model. -/
noncomputable def condH (p : A → ℝ) (T : A → A → ℝ) : ℝ := ∑ a, p a * H (T a)

lemma marg1_dimer {p : A → ℝ} {T : A → A → ℝ} (hTs : ∀ a, ∑ b, T a b = 1) :
    marg1 (dimer p T) = p := by
  funext a
  rw [marg1]
  simp only [dimer]
  rw [← Finset.mul_sum, hTs a, mul_one]

lemma marg2_dimer {p : A → ℝ} {T : A → A → ℝ} (hstat : ∀ b, ∑ a, p a * T a b = p b) :
    marg2 (dimer p T) = p := by
  funext b
  rw [marg2]
  exact hstat b

lemma sum_dimer {p : A → ℝ} {T : A → A → ℝ} (hps : ∑ a, p a = 1) (hTs : ∀ a, ∑ b, T a b = 1) :
    ∑ z : A × A, dimer p T z = 1 := by
  rw [Fintype.sum_prod_type]
  calc ∑ a, ∑ b, dimer p T (a, b) = ∑ a, p a := by
        refine Finset.sum_congr rfl fun a _ => ?_
        simp only [dimer]
        rw [← Finset.mul_sum, hTs a, mul_one]
    _ = 1 := hps

/-- **The conditional entropy is the dimer entropy minus the composition entropy.** -/
theorem condH_eq_dimer_sub {p : A → ℝ} {T : A → A → ℝ} (hp : ∀ a, 0 < p a)
    (hT : ∀ a b, 0 < T a b) (hTs : ∀ a, ∑ b, T a b = 1) :
    condH p T = H (dimer p T) - H p := by
  have hk := H_kernel hp hT hTs
  have hd : H (dimer p T) = H (fun z : A × A => p z.1 * T z.1 z.2) := rfl
  have hc : condH p T = ∑ a, p a * H (T a) := rfl
  rw [hd, hc]
  linarith [hk]

/-- **Conditioning reduces the per-residue entropy by exactly the mutual information of adjacent
residues.**  This is the information-theoretic content of "folding is driven by conditional
propensities": the pair statistics remove `I` nats per residue from the description. -/
theorem condH_eq_sub_mutualInfo {p : A → ℝ} {T : A → A → ℝ} (hp : ∀ a, 0 < p a)
    (hT : ∀ a b, 0 < T a b) (hTs : ∀ a, ∑ b, T a b = 1)
    (hstat : ∀ b, ∑ a, p a * T a b = p b) :
    condH p T = H p - mutualInfo (dimer p T) := by
  have hd := condH_eq_dimer_sub hp hT hTs
  unfold mutualInfo
  rw [marg1_dimer hTs, marg2_dimer hstat]
  linarith

/-- **Conditional entropy never exceeds single-residue entropy.** -/
theorem condH_le_single {p : A → ℝ} {T : A → A → ℝ} (hp : ∀ a, 0 < p a)
    (hT : ∀ a b, 0 < T a b) (hTs : ∀ a, ∑ b, T a b = 1) (hps : ∑ a, p a = 1)
    (hstat : ∀ b, ∑ a, p a * T a b = p b) :
    condH p T ≤ H p := by
  have hI : 0 ≤ mutualInfo (dimer p T) := by
    refine mutualInfo_nonneg (fun z => le_of_lt (mul_pos (hp z.1) (hT z.1 z.2)))
      (sum_dimer hps hTs) ?_ ?_
    · rw [marg1_dimer hTs]; exact hp
    · rw [marg2_dimer hstat]; exact hp
  rw [condH_eq_sub_mutualInfo hp hT hTs hstat]
  linarith

/-! ## Blocks: the length-`n+1` Markov source -/

/-- Splitting a word into its prefix and its last letter. -/
def snocE (A : Type*) (n : ℕ) : ((Fin n → A) × A) ≃ (Fin (n + 1) → A) where
  toFun z := Fin.snoc z.1 z.2
  invFun x := (Fin.init x, x (Fin.last n))
  left_inv := by intro z; simp [Fin.init_snoc]
  right_inv := by intro x; simp [Fin.snoc_init_self]

/-- The law of a length-`n+1` word under the stationary pair model. -/
noncomputable def chain (p : A → ℝ) (T : A → A → ℝ) : (n : ℕ) → (Fin (n + 1) → A) → ℝ
  | 0 => fun x => p (x 0)
  | (n + 1) => fun x =>
      chain p T n (Fin.init x) * T ((Fin.init x) (Fin.last n)) (x (Fin.last (n + 1)))

omit [Fintype A] in
lemma chain_pos {p : A → ℝ} {T : A → A → ℝ} (hp : ∀ a, 0 < p a) (hT : ∀ a b, 0 < T a b) :
    ∀ (n : ℕ) (x : Fin (n + 1) → A), 0 < chain p T n x := by
  intro n
  induction n with
  | zero => intro x; exact hp _
  | succ n ih => intro x; exact mul_pos (ih _) (hT _ _)

/-- **The last letter of a word is distributed as `p`** — stationarity, propagated to every
length.  Stated against an arbitrary test function, which is what the induction needs. -/
lemma chain_last_marginal {p : A → ℝ} {T : A → A → ℝ} (hstat : ∀ b, ∑ a, p a * T a b = p b)
    (n : ℕ) (f : A → ℝ) :
    ∑ x : Fin (n + 1) → A, chain p T n x * f (x (Fin.last n)) = ∑ a, p a * f a := by
  induction n generalizing f with
  | zero =>
      rw [← Equiv.sum_comp (Equiv.funUnique (Fin 1) A).symm
        (fun x : Fin 1 → A => chain p T 0 x * f (x (Fin.last 0)))]
      refine Finset.sum_congr rfl fun a _ => ?_
      simp [chain]
  | succ n ih =>
      rw [← Equiv.sum_comp (snocE A (n + 1))
        (fun x : Fin (n + 2) → A => chain p T (n + 1) x * f (x (Fin.last (n + 1))))]
      rw [Fintype.sum_prod_type]
      have step : ∀ y : Fin (n + 1) → A,
          ∑ b, (fun x : Fin (n + 2) → A => chain p T (n + 1) x * f (x (Fin.last (n + 1))))
              (snocE A (n + 1) (y, b))
            = chain p T n y * ∑ b, T (y (Fin.last n)) b * f b := by
        intro y
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun b _ => ?_
        simp only [snocE, Equiv.coe_fn_mk, chain, Fin.init_snoc, Fin.snoc_last]
        ring
      rw [Finset.sum_congr rfl (fun y _ => step y), ih (fun a => ∑ b, T a b * f b)]
      have key : ∀ b, ∑ a, p a * (T a b * f b) = p b * f b := by
        intro b
        calc ∑ a, p a * (T a b * f b) = (∑ a, p a * T a b) * f b := by
              rw [Finset.sum_mul]
              exact Finset.sum_congr rfl fun a _ => by ring
          _ = p b * f b := by rw [hstat b]
      simp_rw [Finset.mul_sum]
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun b _ => key b

lemma chain_sum_one {p : A → ℝ} {T : A → A → ℝ} (hps : ∑ a, p a = 1)
    (hstat : ∀ b, ∑ a, p a * T a b = p b) (n : ℕ) :
    ∑ x : Fin (n + 1) → A, chain p T n x = 1 := by
  have := chain_last_marginal hstat n (fun _ => 1)
  simpa [hps] using this

/-- The block entropy of the pair model at length `n+1`. -/
noncomputable def blockH (p : A → ℝ) (T : A → A → ℝ) (n : ℕ) : ℝ := H (chain p T n)

/-- **Each further residue costs exactly one conditional entropy.** -/
theorem blockH_succ {p : A → ℝ} {T : A → A → ℝ} (hp : ∀ a, 0 < p a) (hT : ∀ a b, 0 < T a b)
    (hTs : ∀ a, ∑ b, T a b = 1) (hstat : ∀ b, ∑ a, p a * T a b = p b) (n : ℕ) :
    blockH p T (n + 1) = blockH p T n + condH p T := by
  have hfact : (fun z : (Fin (n + 1) → A) × A => chain p T (n + 1) (snocE A (n + 1) z))
      = fun z : (Fin (n + 1) → A) × A => chain p T n z.1 * T (z.1 (Fin.last n)) z.2 := by
    funext z
    simp only [snocE, Equiv.coe_fn_mk, chain, Fin.init_snoc, Fin.snoc_last]
  have hinv : blockH p T (n + 1)
      = H (fun z : (Fin (n + 1) → A) × A => chain p T n z.1 * T (z.1 (Fin.last n)) z.2) := by
    rw [blockH, H, ← Equiv.sum_comp (snocE A (n + 1))
      (fun x : Fin (n + 2) → A => -(chain p T (n + 1) x * Real.log (chain p T (n + 1) x)))]
    rw [H]
    exact Finset.sum_congr rfl fun z _ => by rw [← congrFun hfact z]
  rw [hinv, H_kernel (p := chain p T n) (T := fun y b => T (y (Fin.last n)) b)
    (fun y => chain_pos hp hT n y) (fun y b => hT _ b) (fun y => hTs _)]
  have : ∑ y : Fin (n + 1) → A, chain p T n y * H (fun b => T (y (Fin.last n)) b) = condH p T :=
    chain_last_marginal hstat n (fun a => H (T a))
  rw [blockH, this]

/-- **The block entropy of a pair model.**  Length `n+1` carries `H(p) + n·H(next | previous)`
nats: the first residue costs its full composition entropy, every later residue only its
conditional entropy. -/
theorem blockH_eq {p : A → ℝ} {T : A → A → ℝ} (hp : ∀ a, 0 < p a) (hT : ∀ a b, 0 < T a b)
    (hTs : ∀ a, ∑ b, T a b = 1) (hstat : ∀ b, ∑ a, p a * T a b = p b) (n : ℕ) :
    blockH p T n = H p + n * condH p T := by
  induction n with
  | zero =>
      have : blockH p T 0 = H p := by
        rw [blockH, H, H, ← Equiv.sum_comp (Equiv.funUnique (Fin 1) A).symm
          (fun x : Fin 1 → A => -(chain p T 0 x * Real.log (chain p T 0 x)))]
        exact Finset.sum_congr rfl fun a _ => by simp [chain]
      simpa using this
  | succ n ih =>
      rw [blockH_succ hp hT hTs hstat n, ih]
      push_cast
      ring

/-- **What a composition-only model gets wrong.**  The independent model matched to the same
single-residue composition assigns `(n+1)·H(p)` nats to a length-`n+1` region; the pair model
assigns `n·I` fewer.  Equivalently the pair model's ensemble is smaller by a factor `e^{n·I}`. -/
theorem block_entropy_deficit {p : A → ℝ} {T : A → A → ℝ} (hp : ∀ a, 0 < p a)
    (hT : ∀ a b, 0 < T a b) (hTs : ∀ a, ∑ b, T a b = 1)
    (hstat : ∀ b, ∑ a, p a * T a b = p b) (n : ℕ) :
    ((n : ℝ) + 1) * H p - blockH p T n = n * mutualInfo (dimer p T) := by
  rw [blockH_eq hp hT hTs hstat n, condH_eq_sub_mutualInfo hp hT hTs hstat]
  ring

/-- The pair model never assigns more entropy than the composition model. -/
theorem blockH_le {p : A → ℝ} {T : A → A → ℝ} (hp : ∀ a, 0 < p a) (hT : ∀ a b, 0 < T a b)
    (hTs : ∀ a, ∑ b, T a b = 1) (hps : ∑ a, p a = 1)
    (hstat : ∀ b, ∑ a, p a * T a b = p b) (n : ℕ) :
    blockH p T n ≤ ((n : ℝ) + 1) * H p := by
  have hle : condH p T ≤ H p := condH_le_single hp hT hTs hps hstat
  have hn : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  rw [blockH_eq hp hT hTs hstat n]
  nlinarith

end Markov

/-! ## A hydrophobic/polar chain with a pair propensity -/

section HP

/-- A two-letter (hydrophobic / polar) chain: the next residue repeats the current one with
probability `1-e`.  `e < 1/2` is blocky patterning, the hallmark of a foldable HP sequence. -/
noncomputable def hpT (e : ℝ) : Bool → Bool → ℝ := fun a b => if a = b then 1 - e else e

/-- The uniform composition on the two letters. -/
noncomputable def hpP : Bool → ℝ := fun _ => 1 / 2

lemma hpT_pos {e : ℝ} (h0 : 0 < e) (h1 : e < 1) (a b : Bool) : 0 < hpT e a b := by
  unfold hpT
  by_cases h : a = b <;> simp [h] <;> linarith

lemma hpT_sum (e : ℝ) (a : Bool) : ∑ b, hpT e a b = 1 := by
  cases a <;> simp [hpT]

lemma hpP_sum : ∑ a, hpP a = 1 := by simp [hpP]

lemma hpP_pos (a : Bool) : 0 < hpP a := by unfold hpP; norm_num

lemma hp_stationary (e : ℝ) (b : Bool) : ∑ a, hpP a * hpT e a b = hpP b := by
  cases b <;> simp [hpP, hpT] <;> ring

lemma H_hpP : H hpP = Real.log 2 := by
  simp [H, hpP]

lemma H_hpT (e : ℝ) (a : Bool) : H (hpT e a) = h₂ e := by
  cases a <;> simp [H, hpT, h₂] <;> ring

/-- **The pair propensity of an HP chain, in nats.**  Adjacent residues of the two-letter chain
with switch probability `e` share `log 2 − h₂(e)` nats of mutual information. -/
theorem hp_mutualInfo {e : ℝ} (h0 : 0 < e) (h1 : e < 1) :
    mutualInfo (dimer hpP (hpT e)) = Real.log 2 - h₂ e := by
  have hd : H (dimer hpP (hpT e)) = H hpP + condH hpP (hpT e) := by
    rw [condH_eq_dimer_sub hpP_pos (hpT_pos h0 h1) (hpT_sum e)]; ring
  have hc : condH hpP (hpT e) = h₂ e := by
    rw [condH]
    simp only [H_hpT]
    rw [← Finset.sum_mul, hpP_sum, one_mul]
  unfold mutualInfo
  rw [marg1_dimer (hpT_sum e), marg2_dimer (hp_stationary e), hd, hc, H_hpP]
  ring

/-- At a switch probability of `1/4` the pair information is `(3/4)·log 3 − log 2 > 0`: the
conditional model is strictly smaller than the composition model. -/
theorem hp_quarter_mutualInfo :
    mutualInfo (dimer hpP (hpT (1/4))) = (3/4) * Real.log 3 - Real.log 2 := by
  rw [hp_mutualInfo (by norm_num) (by norm_num)]
  have h4 : Real.log (1/4 : ℝ) = -(2 * Real.log 2) := by
    rw [show (1/4 : ℝ) = (2 : ℝ)⁻¹ ^ 2 by norm_num, Real.log_pow, Real.log_inv]
    ring
  have h34 : Real.log (3/4 : ℝ) = Real.log 3 - 2 * Real.log 2 := by
    rw [Real.log_div (by norm_num) (by norm_num),
      show (4 : ℝ) = 2 ^ 2 by norm_num, Real.log_pow]
    ring
  unfold h₂
  rw [show (1 : ℝ) - 1/4 = 3/4 by norm_num, h4, h34]
  ring

theorem hp_quarter_mutualInfo_pos : 0 < mutualInfo (dimer hpP (hpT (1/4))) := by
  rw [hp_quarter_mutualInfo]
  have h : Real.log 16 < Real.log 27 := Real.log_lt_log (by norm_num) (by norm_num)
  have h16 : Real.log 16 = 4 * Real.log 2 := by
    rw [show (16 : ℝ) = 2 ^ 4 by norm_num, Real.log_pow]; ring
  have h27 : Real.log 27 = 3 * Real.log 3 := by
    rw [show (27 : ℝ) = 3 ^ 3 by norm_num, Real.log_pow]; ring
  rw [h16, h27] at h
  linarith

/-- **The realism statement.**  For a 100-residue HP region with blocky patterning, the pair
model assigns `100·((3/4)log 3 − log 2) ≈ 13` nats less than the composition-matched independent
model: the conditional description is smaller by a factor of about `e^{13}`, and it is the
conditional description that has to be searched. -/
theorem hp_block_deficit (n : ℕ) :
    ((n : ℝ) + 1) * H hpP - blockH hpP (hpT (1/4)) n = n * ((3/4) * Real.log 3 - Real.log 2) := by
  rw [block_entropy_deficit hpP_pos (hpT_pos (by norm_num) (by norm_num)) (hpT_sum _)
    (hp_stationary _) n, hp_quarter_mutualInfo]

end HP

end CondSeq

end IDR
