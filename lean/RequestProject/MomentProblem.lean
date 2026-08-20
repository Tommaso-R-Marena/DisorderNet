/-
# Part LXXXIV  The moment problem: what a finite list of averages can and cannot fix

Almost every experiment in this development returns a *moment* of some conformational observable:
SAXS returns the second moment of the distance distribution, FRET a sixth moment, ion mobility a
first and a second, an NMR average a first.  A refinement protocol is then handed a finite list of
numbers and asked for the distribution behind them.  This file settles what that list determines.

* `mom` — the `p`-th moment of an observable `x` under weights `w`.
* `alt_choose_pow_sum` — the classical vanishing of an alternating binomial sum against a power of
  degree below `n`; the `n`-th forward difference of a polynomial of degree `< n` is zero.
* `wEven`, `wOdd`, `moment_indeterminacy` — **no finite list of moments determines the ensemble.**
  For every `k` there are two ensembles on the `k+2` points `0,1,…,k+1` with *disjoint supports* —
  they share no conformation whatsoever — whose moments of every order `p ≤ k` agree exactly.  So a
  protocol restrained by `k` moments cannot be said to have measured the distribution, however
  small its residuals: an ensemble sharing no member with the truth fits the same data equally
  well.
* `weights_eq_of_moments_eq` — **but the support is exactly the missing information.**  Once the
  candidate conformations are fixed and distinct, `k+1` moments determine their populations
  uniquely, by Lagrange interpolation.  Structural prior knowledge is not a convenience here; it is
  what converts moment data into an ensemble.
* `markov_bound`, `chebyshev_bound` — **and what moments do certify without any prior is a
  bound.**  The population of conformations with `x ≥ a` is at most `⟨x⟩/a`, and the population
  deviating from the mean by `a` or more is at most `Var/a²`.  These are the statements a moment
  measurement supports outright, and they are the honest form in which to report one.
* `markov_sharp` — and they are the best such statements: the first bound is attained by an
  explicit two-conformation ensemble, so no sharper population claim follows from a mean.

Design consequence: a model of a disordered region fitted to `k` averages should report the
population bounds those averages certify, plus the conformational support it assumed — because the
support, not the data, is what made the answer unique.
-/
import Mathlib

namespace IDR

open Finset

namespace Moments

variable {N : ℕ}

/-! ## Moments -/

/-- The `p`-th moment of the observable `x` under the ensemble weights `w`. -/
def mom (w x : Fin N → ℝ) (p : ℕ) : ℝ := ∑ j, w j * x j ^ p

/-- The zeroth moment is the total weight. -/
theorem mom_zero (w x : Fin N → ℝ) : mom w x 0 = ∑ j, w j := by
  simp [mom]

/-! ## Two ensembles, no shared conformation, the same first `k` moments -/

/-- **Vanishing alternating binomial sum.**  For `m < n` the `n`-th forward difference of the
polynomial `r ↦ r ^ m` is zero, which is exactly `∑_i (-1)^i C(n,i) i^m = 0`. -/
theorem alt_choose_pow_sum (n m : ℕ) (h : m < n) :
    ∑ i ∈ range (n + 1), (-1 : ℝ) ^ i * (n.choose i) * (i : ℝ) ^ m = 0 := by
  have key := fwdDiff_iter_eq_sum_shift (M := ℝ) (G := ℝ) 1 (fun r => r ^ m) n 0
  rw [fwdDiff_iter_pow_eq_zero_of_lt h] at key
  simp only [Pi.zero_apply, zero_add, nsmul_eq_mul, mul_one, zsmul_eq_mul] at key
  push_cast at key
  have h2 : ∀ i ∈ range (n + 1), (-1 : ℝ) ^ (n - i) * (n.choose i) * (i : ℝ) ^ m
      = (-1 : ℝ) ^ n * ((-1 : ℝ) ^ i * (n.choose i) * (i : ℝ) ^ m) := by
    intro i hi
    have hin : i ≤ n := Nat.lt_succ_iff.1 (mem_range.1 hi)
    have hsq : ((-1 : ℝ) ^ i) * ((-1 : ℝ) ^ i) = 1 := by
      rw [← pow_add, ← two_mul, pow_mul]; norm_num
    have hsplit : (-1 : ℝ) ^ (n - i) * (-1 : ℝ) ^ i = (-1 : ℝ) ^ n := by
      rw [← pow_add, Nat.sub_add_cancel hin]
    have hrw : (-1 : ℝ) ^ (n - i) = (-1 : ℝ) ^ n * (-1 : ℝ) ^ i := by
      calc (-1 : ℝ) ^ (n - i) = (-1 : ℝ) ^ (n - i) * ((-1 : ℝ) ^ i * (-1 : ℝ) ^ i) := by
            rw [hsq, mul_one]
        _ = ((-1 : ℝ) ^ (n - i) * (-1 : ℝ) ^ i) * (-1 : ℝ) ^ i := by ring
        _ = (-1 : ℝ) ^ n * (-1 : ℝ) ^ i := by rw [hsplit]
    rw [hrw]; ring
  rw [Finset.sum_congr rfl h2, ← Finset.mul_sum] at key
  have hne : ((-1 : ℝ) ^ n) ≠ 0 := by
    intro hc
    have habs := abs_eq_zero.2 hc
    rw [abs_pow, abs_neg, abs_one, one_pow] at habs
    exact one_ne_zero habs
  exact (mul_eq_zero.1 key.symm).resolve_left hne

/-- The alternating sum of the binomial coefficients of `k+1` vanishes. -/
theorem alt_choose_sum (k : ℕ) :
    ∑ i ∈ range (k + 2), (-1 : ℝ) ^ i * ((k + 1).choose i) = 0 := by
  have h := alt_choose_pow_sum (k + 1) 0 (Nat.succ_pos k)
  simpa using h

/-- Half of the binomial coefficients of `k+1`, at the even indices, sum to `2^k`. -/
theorem sum_even_choose (k : ℕ) :
    ∑ i ∈ range (k + 2), (if Even i then (((k + 1).choose i : ℝ)) else 0) = 2 ^ k := by
  have hsum : ∑ i ∈ range (k + 2), (((k + 1).choose i : ℝ)) = 2 ^ (k + 1) := by
    exact_mod_cast congrArg (Nat.cast : ℕ → ℝ) (Nat.sum_range_choose (k + 1))
  have hterm : ∀ i ∈ range (k + 2), (if Even i then (((k + 1).choose i : ℝ)) else 0)
      = (((k + 1).choose i : ℝ) + (-1 : ℝ) ^ i * ((k + 1).choose i)) / 2 := by
    intro i _
    rcases Nat.even_or_odd i with he | ho
    · rw [if_pos he, he.neg_one_pow]; ring
    · rw [if_neg (Nat.not_even_iff_odd.2 ho), ho.neg_one_pow]; ring
  rw [Finset.sum_congr rfl hterm, ← Finset.sum_div, Finset.sum_add_distrib, hsum,
    alt_choose_sum k]
  ring

/-- The other half, at the odd indices, sums to `2^k` as well. -/
theorem sum_odd_choose (k : ℕ) :
    ∑ i ∈ range (k + 2), (if Even i then 0 else (((k + 1).choose i : ℝ))) = 2 ^ k := by
  have hsum : ∑ i ∈ range (k + 2), (((k + 1).choose i : ℝ)) = 2 ^ (k + 1) := by
    exact_mod_cast congrArg (Nat.cast : ℕ → ℝ) (Nat.sum_range_choose (k + 1))
  have hsplit : ∀ i ∈ range (k + 2),
      (((k + 1).choose i : ℝ))
        = (if Even i then (((k + 1).choose i : ℝ)) else 0)
          + (if Even i then 0 else (((k + 1).choose i : ℝ))) := by
    intro i _
    by_cases he : Even i <;> simp [he]
  rw [Finset.sum_congr rfl hsplit, Finset.sum_add_distrib, sum_even_choose k] at hsum
  have : (2 : ℝ) ^ (k + 1) = 2 ^ k + 2 ^ k := by ring
  linarith [hsum, this]

/-! ### The construction, indexed by natural numbers -/

/-- Observable value of the `i`-th candidate conformation: `0, 1, …`. -/
def ptsN (i : ℕ) : ℝ := (i : ℝ)

/-- Weight of the `i`-th conformation in the even-supported ensemble. -/
noncomputable def wEvenN (k i : ℕ) : ℝ :=
  if Even i then (((k + 1).choose i : ℝ)) / 2 ^ k else 0

/-- Weight of the `i`-th conformation in the odd-supported ensemble. -/
noncomputable def wOddN (k i : ℕ) : ℝ :=
  if Even i then 0 else (((k + 1).choose i : ℝ)) / 2 ^ k

/-- The `k+2` candidate conformations of the construction, with observable values `0,1,…,k+1`. -/
def pts (k : ℕ) : Fin (k + 2) → ℝ := fun i => ptsN (i : ℕ)

/-- The ensemble supported on the even-numbered conformations. -/
noncomputable def wEven (k : ℕ) : Fin (k + 2) → ℝ := fun i => wEvenN k (i : ℕ)

/-- The ensemble supported on the odd-numbered conformations. -/
noncomputable def wOdd (k : ℕ) : Fin (k + 2) → ℝ := fun i => wOddN k (i : ℕ)

theorem pts_injective (k : ℕ) : Function.Injective (pts k) := by
  intro a b hab
  have h : ((a : ℕ) : ℝ) = ((b : ℕ) : ℝ) := hab
  exact Fin.ext (by exact_mod_cast h)

theorem wEven_nonneg (k : ℕ) (i : Fin (k + 2)) : 0 ≤ wEven k i := by
  unfold wEven wEvenN; split <;> positivity

theorem wOdd_nonneg (k : ℕ) (i : Fin (k + 2)) : 0 ≤ wOdd k i := by
  unfold wOdd wOddN; split <;> positivity

/-- The two ensembles share no conformation: wherever one puts weight, the other puts none. -/
theorem supports_disjoint (k : ℕ) (i : Fin (k + 2)) : wEven k i = 0 ∨ wOdd k i = 0 := by
  unfold wEven wOdd wEvenN wOddN
  by_cases he : Even (i : ℕ)
  · right; simp [he]
  · left; simp [he]

theorem wEven_sum (k : ℕ) : ∑ i, wEven k i = 1 := by
  rw [show (∑ i, wEven k i) = ∑ i : Fin (k + 2), wEvenN k (i : ℕ) from rfl,
    Fin.sum_univ_eq_sum_range (wEvenN k) (k + 2)]
  have h : ∀ i ∈ range (k + 2), wEvenN k i
      = (if Even i then (((k + 1).choose i : ℝ)) else 0) / 2 ^ k := by
    intro i _
    unfold wEvenN
    by_cases he : Even i <;> simp [he]
  rw [Finset.sum_congr rfl h, ← Finset.sum_div, sum_even_choose k]
  field_simp

theorem wOdd_sum (k : ℕ) : ∑ i, wOdd k i = 1 := by
  rw [show (∑ i, wOdd k i) = ∑ i : Fin (k + 2), wOddN k (i : ℕ) from rfl,
    Fin.sum_univ_eq_sum_range (wOddN k) (k + 2)]
  have h : ∀ i ∈ range (k + 2), wOddN k i
      = (if Even i then 0 else (((k + 1).choose i : ℝ))) / 2 ^ k := by
    intro i _
    unfold wOddN
    by_cases he : Even i <;> simp [he]
  rw [Finset.sum_congr rfl h, ← Finset.sum_div, sum_odd_choose k]
  field_simp

theorem wEven_ne_wOdd (k : ℕ) : wEven k ≠ wOdd k := by
  intro h
  have h0 : wEven k (0 : Fin (k + 2)) = wOdd k (0 : Fin (k + 2)) := by rw [h]
  simp [wEven, wOdd, wEvenN, wOddN] at h0

/-- **The two ensembles agree on every moment of order at most `k`.** -/
theorem moments_agree (k : ℕ) {p : ℕ} (hp : p ≤ k) :
    mom (wEven k) (pts k) p = mom (wOdd k) (pts k) p := by
  have hkey := alt_choose_pow_sum (k + 1) p (Nat.lt_succ_of_le hp)
  have hdiff : mom (wEven k) (pts k) p - mom (wOdd k) (pts k) p
      = (∑ i ∈ range (k + 2), (-1 : ℝ) ^ i * ((k + 1).choose i) * (i : ℝ) ^ p) / 2 ^ k := by
    rw [mom, mom, ← Finset.sum_sub_distrib]
    rw [show (∑ i : Fin (k + 2), (wEven k i * pts k i ^ p - wOdd k i * pts k i ^ p))
        = ∑ i : Fin (k + 2),
            (fun j : ℕ => wEvenN k j * ptsN j ^ p - wOddN k j * ptsN j ^ p) (i : ℕ) from rfl,
      Fin.sum_univ_eq_sum_range (fun j : ℕ => wEvenN k j * ptsN j ^ p - wOddN k j * ptsN j ^ p),
      Finset.sum_div]
    refine Finset.sum_congr rfl fun i _ => ?_
    unfold wEvenN wOddN ptsN
    by_cases he : Even i
    · rw [if_pos he, if_pos he, he.neg_one_pow]; ring
    · rw [if_neg he, if_neg he, (Nat.not_even_iff_odd.1 he).neg_one_pow]; ring
  have hz : mom (wEven k) (pts k) p - mom (wOdd k) (pts k) p = 0 := by
    rw [hdiff, hkey]; simp
  linarith

/-- **No finite list of moments determines the ensemble.**  For every `k` there are two ensembles
over the same `k+2` distinct conformations, with nonnegative weights summing to one and *no
conformation in common*, whose moments of every order up to `k` coincide. -/
theorem moment_indeterminacy (k : ℕ) :
    ∃ (x wA wB : Fin (k + 2) → ℝ),
      Function.Injective x ∧
      (∀ j, 0 ≤ wA j) ∧ (∀ j, 0 ≤ wB j) ∧
      (∑ j, wA j = 1) ∧ (∑ j, wB j = 1) ∧
      (∀ j, wA j = 0 ∨ wB j = 0) ∧ wA ≠ wB ∧
      ∀ p ≤ k, mom wA x p = mom wB x p :=
  ⟨pts k, wEven k, wOdd k, pts_injective k, wEven_nonneg k, wOdd_nonneg k, wEven_sum k,
    wOdd_sum k, supports_disjoint k, wEven_ne_wOdd k, fun _ hp => moments_agree k hp⟩

/-- **The smallest instance.**  At `k = 1` the construction is the familiar pair: half the
population at `0` and half at `2`, against all of it at `1`.  The two share no conformation and
have the same mean; they are told apart only by the second moment. -/
theorem two_state_instance :
    mom (wEven 1) (pts 1) 1 = mom (wOdd 1) (pts 1) 1 ∧
    mom (wEven 1) (pts 1) 2 ≠ mom (wOdd 1) (pts 1) 2 := by
  refine ⟨by simp [mom, wEven, wOdd, wEvenN, wOddN, pts, ptsN, Fin.sum_univ_three], ?_⟩
  simp [mom, wEven, wOdd, wEvenN, wOddN, pts, ptsN, Fin.sum_univ_three]
  norm_num

/-! ## With the support fixed, the moments do determine the populations -/

/-- **Known support plus `k+1` moments is identifiability.**  If two weightings of the *same* `k+1`
distinct conformations reproduce the same moments of orders `0,…,k`, they are the same weighting.
The support is what turns moment data into a unique ensemble. -/
theorem weights_eq_of_moments_eq {k : ℕ} (x u v : Fin (k + 1) → ℝ) (hx : Function.Injective x)
    (h : ∀ p ≤ k, mom u x p = mom v x p) : u = v := by
  have hpoly : ∀ P : Polynomial ℝ, P.natDegree ≤ k →
      ∑ j, u j * P.eval (x j) = ∑ j, v j * P.eval (x j) := by
    intro P hP
    have hev : ∀ j : Fin (k + 1), P.eval (x j) = ∑ p ∈ range (k + 1), P.coeff p * x j ^ p :=
      fun j => Polynomial.eval_eq_sum_range' (Nat.lt_succ_of_le hP) (x j)
    calc ∑ j, u j * P.eval (x j)
        = ∑ p ∈ range (k + 1), P.coeff p * ∑ j, u j * x j ^ p := by
          simp only [hev, Finset.mul_sum]
          rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring
      _ = ∑ p ∈ range (k + 1), P.coeff p * ∑ j, v j * x j ^ p :=
          Finset.sum_congr rfl fun p hp => by
            have := h p (Nat.lt_succ_iff.1 (mem_range.1 hp))
            rw [mom, mom] at this
            rw [this]
      _ = ∑ j, v j * P.eval (x j) := by
          simp only [hev, Finset.mul_sum]
          rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring
  funext i
  have hinj : Set.InjOn x ↑(Finset.univ : Finset (Fin (k + 1))) := fun a _ b _ hab => hx hab
  have hdeg : (Lagrange.basis Finset.univ x i).natDegree ≤ k := by
    rw [Lagrange.natDegree_basis hinj (Finset.mem_univ i)]; simp
  have hmain := hpoly _ hdeg
  have hu : ∑ j, u j * (Lagrange.basis Finset.univ x i).eval (x j) = u i := by
    rw [Finset.sum_eq_single i]
    · rw [Lagrange.eval_basis_self hinj (Finset.mem_univ i), mul_one]
    · intro b _ hb
      rw [Lagrange.eval_basis_of_ne (Ne.symm hb) (Finset.mem_univ b), mul_zero]
    · intro hc; exact absurd (Finset.mem_univ i) hc
  have hv : ∑ j, v j * (Lagrange.basis Finset.univ x i).eval (x j) = v i := by
    rw [Finset.sum_eq_single i]
    · rw [Lagrange.eval_basis_self hinj (Finset.mem_univ i), mul_one]
    · intro b _ hb
      rw [Lagrange.eval_basis_of_ne (Ne.symm hb) (Finset.mem_univ b), mul_zero]
    · intro hc; exact absurd (Finset.mem_univ i) hc
  rw [← hu, ← hv, hmain]

/-! ## What moments certify with no prior at all -/

/-- The population of conformations whose observable is at least `a`. -/
noncomputable def pop (w x : Fin N → ℝ) (a : ℝ) : ℝ := ∑ j, if a ≤ x j then w j else 0

/-- **Markov's inequality for an ensemble.**  A first moment bounds the population of the tail:
at most `⟨x⟩/a` of the ensemble can have `x ≥ a`. -/
theorem markov_bound (w x : Fin N → ℝ) (hw : ∀ j, 0 ≤ w j) (hx : ∀ j, 0 ≤ x j)
    {a : ℝ} (ha : 0 < a) : pop w x a ≤ mom w x 1 / a := by
  have hstep : a * pop w x a ≤ mom w x 1 := by
    rw [pop, Finset.mul_sum, mom]
    refine Finset.sum_le_sum fun j _ => ?_
    by_cases hc : a ≤ x j
    · rw [if_pos hc]
      nlinarith [mul_le_mul_of_nonneg_left hc (hw j)]
    · rw [if_neg hc, mul_zero]
      nlinarith [mul_nonneg (hw j) (hx j)]
  rw [le_div_iff₀ ha]
  nlinarith [hstep]

/-- The population of conformations deviating from `mu` by at least `a`. -/
noncomputable def popDev (w x : Fin N → ℝ) (mu a : ℝ) : ℝ :=
  ∑ j, if a ≤ |x j - mu| then w j else 0

/-- **Chebyshev's inequality for an ensemble.**  A second central moment bounds the population far
from the mean: at most `Var/a²` of the ensemble can deviate from `mu` by `a` or more. -/
theorem chebyshev_bound (w x : Fin N → ℝ) (hw : ∀ j, 0 ≤ w j) {mu a : ℝ} (ha : 0 < a) :
    popDev w x mu a ≤ (∑ j, w j * (x j - mu) ^ 2) / a ^ 2 := by
  have hstep : a ^ 2 * popDev w x mu a ≤ ∑ j, w j * (x j - mu) ^ 2 := by
    rw [popDev, Finset.mul_sum]
    refine Finset.sum_le_sum fun j _ => ?_
    by_cases hc : a ≤ |x j - mu|
    · rw [if_pos hc]
      have hsq : a ^ 2 ≤ (x j - mu) ^ 2 := by
        have h1 : a * a ≤ |x j - mu| * |x j - mu| := mul_self_le_mul_self ha.le hc
        rw [← sq_abs (x j - mu)]
        nlinarith [h1]
      nlinarith [mul_le_mul_of_nonneg_left hsq (hw j)]
    · rw [if_neg hc, mul_zero]
      nlinarith [mul_nonneg (hw j) (sq_nonneg (x j - mu))]
  rw [le_div_iff₀ (by positivity)]
  nlinarith [hstep]

/-- **Markov's bound is attained.**  For every mean `m` in `[0, a]` there is an ensemble of two
conformations, at `0` and at `a`, with that mean and with exactly the fraction `m/a` of its
population at or beyond `a`.  The certified population bound is therefore the best statement a
first moment supports; nothing sharper can be claimed. -/
theorem markov_sharp {a m : ℝ} (ha : 0 < a) (hm0 : 0 ≤ m) (hma : m ≤ a) :
    ∃ w x : Fin 2 → ℝ, (∀ j, 0 ≤ w j) ∧ (∀ j, 0 ≤ x j) ∧ (∑ j, w j = 1) ∧
      mom w x 1 = m ∧ pop w x a = m / a := by
  refine ⟨![1 - m / a, m / a], ![0, a], ?_, ?_, ?_, ?_, ?_⟩
  · intro j
    fin_cases j
    · simp
      rw [div_le_one ha]; exact hma
    · simp; positivity
  · intro j; fin_cases j <;> simp [ha.le]
  · simp [Fin.sum_univ_two]
  · simp [mom, Fin.sum_univ_two]
    field_simp
  · simp [pop, Fin.sum_univ_two, not_le.2 ha]

end Moments

end IDR
