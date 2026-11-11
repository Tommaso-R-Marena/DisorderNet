/-
# Part LXXIV  How steep can a multivalent binding curve be?

An intrinsically disordered region binds its partners multivalently: several short motifs on the
same chain engage several sites on the partner, and the measured quantity is a titration curve.
Its steepness -- the Hill slope -- is routinely read as evidence of cooperativity, of coupled
folding, of a switch.  This part fixes exactly what a Hill slope can and cannot say, for an
arbitrary equilibrium model with `n` sites.

The general object is the *binding polynomial* `Z(x) = sum_{j<=n} a j x^j` with nonnegative
weights `a j` and ligand activity `x`; the occupancy `j` carries probability proportional to
`a j x^j`.  Everything below holds for this whole class, i.e. for **every** equilibrium binding
model on `n` sites, with couplings of arbitrary order.

* `hasDerivAt_meanOcc` -- the fluctuation--response identity `d<N>/d ln x = Var(N)`.
* `meanOcc_nondecreasing_deriv` -- hence every binding curve of every such model is
  nondecreasing in activity: a nonmonotone titration is not an equilibrium binding effect.
* `hasDerivAt_logit` -- the Hill slope, defined as `d ln(theta/(1-theta))/d ln x` for the
  fractional saturation `theta = <N>/n`, equals `n Var / (<N>(n - <N>))`.
* `varOcc_le`, `hill_le_valence` -- **the valence bound**: `Var <= <N>(n - <N>)`, so the Hill
  slope never exceeds the valence `n`, at any activity, under any coupling scheme.
* `hill_eq_valence_iff` -- and the bound is attained **only** all-or-none: the slope equals `n`
  at an activity exactly when every partially bound state has zero weight.  A measured slope near
  the valence is a statement about intermediate populations, not about affinities.
* `hill_allOrNone` -- the all-or-none polynomial attains it, so the bound is sharp.
* `hillInd_le_one`, `hillInd_identical` -- **the independence bound**: if the sites bind
  independently, so that `Z` factorises as `prod_s (1 + k_s x)`, the Hill slope is at most `1`,
  with `1` attained exactly by identical sites.  A measured slope above `1` falsifies
  independence; a slope below `1` needs only heterogeneous affinities and no negative
  cooperativity at all.

The two bounds bracket what a titration on a disordered region can establish: `1 < n_H <= n` is
the entire content of steepness, `n_H = n` forces an all-or-none ensemble, and `n_H < 1` is
explained by site heterogeneity -- which a disordered chain with distinct motifs has for free.
-/
import Mathlib
import RequestProject.Valence

set_option autoImplicit false

namespace IDR

namespace BindPoly

open Finset

/-! ## The binding polynomial and its occupancy statistics -/

/-- The `k`-th unnormalised occupancy moment of a binding polynomial with weights `a` on `n`
sites: `sum_j j^k a j x^j`.  For `k = 0` this is the binding polynomial itself. -/
def mom (n : ℕ) (a : ℕ → ℝ) (k : ℕ) (x : ℝ) : ℝ :=
  ∑ j ∈ range (n + 1), (j : ℝ) ^ k * (a j * x ^ j)

/-- The binding polynomial (grand partition function) `Z(x) = sum_j a j x^j`. -/
def part (n : ℕ) (a : ℕ → ℝ) (x : ℝ) : ℝ := ∑ j ∈ range (n + 1), a j * x ^ j

theorem part_eq_mom (n : ℕ) (a : ℕ → ℝ) : part n a = mom n a 0 := by
  funext x; unfold part mom; simp

/-- The mean number of bound sites at activity `x`. -/
noncomputable def meanOcc (n : ℕ) (a : ℕ → ℝ) (x : ℝ) : ℝ := mom n a 1 x / part n a x

/-- The variance of the number of bound sites at activity `x`. -/
noncomputable def varOcc (n : ℕ) (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  mom n a 2 x / part n a x - (meanOcc n a x) ^ 2

/-- The Hill slope, expressed through occupancy fluctuations:
`n Var(N) / (<N> (n - <N>))`.  `hasDerivAt_logit` proves this is the logarithmic derivative
`d ln(theta/(1-theta))/d ln x` of the fractional saturation. -/
noncomputable def hill (n : ℕ) (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  (n : ℝ) * varOcc n a x / (meanOcc n a x * ((n : ℝ) - meanOcc n a x))

section Basic

variable {n : ℕ} {a : ℕ → ℝ} {x : ℝ}

theorem mom_nonneg (ha : ∀ j, 0 ≤ a j) (hx : 0 ≤ x) (k : ℕ) : 0 ≤ mom n a k x :=
  Finset.sum_nonneg fun j _ =>
    mul_nonneg (pow_nonneg (Nat.cast_nonneg j) k) (mul_nonneg (ha j) (pow_nonneg hx j))

/-- The binding polynomial is positive as soon as one accessible state has positive weight. -/
theorem part_pos (ha : ∀ j, 0 ≤ a j) (hx : 0 < x) {j₀ : ℕ} (hj₀ : j₀ < n + 1) (hpos : 0 < a j₀) :
    0 < part n a x :=
  Finset.sum_pos' (fun j _ => mul_nonneg (ha j) (pow_nonneg hx.le j))
    ⟨j₀, Finset.mem_range.mpr hj₀, mul_pos hpos (pow_pos hx j₀)⟩

/-- Differentiating a moment raises its order: `d/dx (sum_j j^k a_j x^j) = (1/x) sum_j j^(k+1)
a_j x^j`. -/
theorem hasDerivAt_mom (n : ℕ) (a : ℕ → ℝ) (k : ℕ) (hx : x ≠ 0) :
    HasDerivAt (mom n a k) (mom n a (k + 1) x / x) x := by
  have h : HasDerivAt (fun y : ℝ => ∑ j ∈ range (n + 1), (j : ℝ) ^ k * (a j * y ^ j))
      (∑ j ∈ range (n + 1), (j : ℝ) ^ k * (a j * ((j : ℝ) * x ^ (j - 1)))) x :=
    HasDerivAt.fun_sum (fun (j : ℕ) _ =>
      HasDerivAt.const_mul ((j : ℝ) ^ k) (HasDerivAt.const_mul (a j) (hasDerivAt_pow j x)))
  have heq : ∑ j ∈ range (n + 1), (j : ℝ) ^ k * (a j * ((j : ℝ) * x ^ (j - 1)))
      = mom n a (k + 1) x / x := by
    unfold mom
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun j _ => ?_
    rcases Nat.eq_zero_or_pos j with hj | hj
    · subst hj; simp
    · have hxj : x ^ j = x * x ^ (j - 1) := by
        conv_lhs => rw [show j = 1 + (j - 1) by omega]
        rw [pow_add, pow_one]
      rw [hxj]
      field_simp
      ring
  exact heq ▸ h

theorem hasDerivAt_part (n : ℕ) (a : ℕ → ℝ) (hx : x ≠ 0) :
    HasDerivAt (part n a) (mom n a 1 x / x) x := by
  rw [part_eq_mom]
  exact hasDerivAt_mom n a 0 hx

/-- The mean occupancy is the standard thermodynamic occupancy `x d log Z/dx`. -/
theorem meanOcc_eq_occupancy (n : ℕ) (a : ℕ → ℝ) (hx : x ≠ 0) :
    Valence.occupancy (part n a) x = meanOcc n a x := by
  rw [Valence.occupancy, (hasDerivAt_part n a hx).deriv, meanOcc]
  field_simp


/-- **Fluctuation--response.**  `d<N>/d ln x = Var(N)`: the slope of the binding curve is the
occupancy variance. -/
theorem hasDerivAt_meanOcc (n : ℕ) (a : ℕ → ℝ) (hx : x ≠ 0) (hZ : part n a x ≠ 0) :
    HasDerivAt (meanOcc n a) (varOcc n a x / x) x := by
  have h := (hasDerivAt_mom n a 1 hx).div (hasDerivAt_part n a hx) hZ
  have heq : (mom n a 2 x / x * part n a x - mom n a 1 x * (mom n a 1 x / x))
      / part n a x ^ 2 = varOcc n a x / x := by
    rw [varOcc, meanOcc]
    field_simp
  exact heq ▸ h

/-- Cauchy--Schwarz for the occupancy distribution: the variance is nonnegative. -/
theorem varOcc_nonneg (ha : ∀ j, 0 ≤ a j) (hx : 0 < x) (hZ : 0 < part n a x) :
    0 ≤ varOcc n a x := by
  have hCS : (mom n a 1 x) ^ 2 ≤ part n a x * mom n a 2 x := by
    have h := Finset.sum_sq_le_sum_mul_sum_of_sq_eq_mul (range (n + 1))
      (r := fun j => (j : ℝ) * (a j * x ^ j)) (f := fun j => a j * x ^ j)
      (g := fun j => (j : ℝ) ^ 2 * (a j * x ^ j))
      (fun j _ => mul_nonneg (ha j) (pow_nonneg hx.le j))
      (fun j _ => mul_nonneg (sq_nonneg _) (mul_nonneg (ha j) (pow_nonneg hx.le j)))
      (fun j _ => by ring)
    have h1 : mom n a 1 x = ∑ j ∈ range (n + 1), (j : ℝ) * (a j * x ^ j) := by
      unfold mom; simp
    rw [h1]
    exact h
  rw [varOcc, meanOcc, div_pow, sub_nonneg, div_le_div_iff₀ (by positivity) hZ]
  nlinarith [hCS, hZ]

/-- Every binding curve of every equilibrium model on `n` sites is nondecreasing in activity. -/
theorem meanOcc_nondecreasing_deriv (n : ℕ) (a : ℕ → ℝ) (ha : ∀ j, 0 ≤ a j) (hx : 0 < x)
    (hZ : 0 < part n a x) : 0 ≤ deriv (meanOcc n a) x := by
  rw [(hasDerivAt_meanOcc n a hx.ne' hZ.ne').deriv]
  exact div_nonneg (varOcc_nonneg ha hx hZ) hx.le

theorem meanOcc_nonneg (ha : ∀ j, 0 ≤ a j) (hx : 0 < x) (hZ : 0 < part n a x) :
    0 ≤ meanOcc n a x :=
  div_nonneg (mom_nonneg ha hx.le 1) hZ.le

/-- The mean occupancy never exceeds the valence. -/
theorem meanOcc_le_valence (ha : ∀ j, 0 ≤ a j) (hx : 0 < x) (hZ : 0 < part n a x) :
    meanOcc n a x ≤ n := by
  have h : mom n a 1 x ≤ (n : ℝ) * part n a x := by
    unfold mom part
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun j hj => ?_
    have hjn : (j : ℝ) ≤ n := by
      have := Finset.mem_range.mp hj
      exact_mod_cast Nat.lt_succ_iff.mp this
    have hterm : 0 ≤ a j * x ^ j := mul_nonneg (ha j) (pow_nonneg hx.le j)
    have hkey := mul_nonneg (sub_nonneg.mpr hjn) hterm
    nlinarith [hkey]
  rw [meanOcc, div_le_iff₀ hZ]
  linarith

/-- **The valence bound on fluctuations.**  `Var(N) <= <N>(n - <N>)` for every binding
polynomial: occupancy fluctuations never exceed those of a two-point distribution on the empty
and the fully bound state. -/
theorem varOcc_le (ha : ∀ j, 0 ≤ a j) (hx : 0 < x) (hZ : 0 < part n a x) :
    varOcc n a x ≤ meanOcc n a x * ((n : ℝ) - meanOcc n a x) := by
  have h : mom n a 2 x ≤ (n : ℝ) * mom n a 1 x := by
    unfold mom
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun j hj => ?_
    have hjn : (j : ℝ) ≤ n := by
      have := Finset.mem_range.mp hj
      exact_mod_cast Nat.lt_succ_iff.mp this
    have hterm : 0 ≤ a j * x ^ j := mul_nonneg (ha j) (pow_nonneg hx.le j)
    have hj0 : (0 : ℝ) ≤ (j : ℝ) := Nat.cast_nonneg j
    have hkey := mul_nonneg (mul_nonneg (sub_nonneg.mpr hjn) hj0) hterm
    nlinarith [hkey]
  have hdiv : mom n a 2 x / part n a x ≤ (n : ℝ) * meanOcc n a x := by
    rw [meanOcc, div_le_iff₀ hZ]
    have hh : (n : ℝ) * (mom n a 1 x / part n a x) * part n a x = (n : ℝ) * mom n a 1 x := by
      field_simp
    rw [hh]
    exact h
  rw [varOcc]
  nlinarith [hdiv]

/-- **The Hill slope never exceeds the valence**, at any activity, for any coupling scheme. -/
theorem hill_le_valence (ha : ∀ j, 0 ≤ a j) (hx : 0 < x) (hZ : 0 < part n a x)
    (h0 : 0 < meanOcc n a x) (hn : meanOcc n a x < n) : hill n a x ≤ n := by
  have hden : 0 < meanOcc n a x * ((n : ℝ) - meanOcc n a x) := mul_pos h0 (by linarith)
  rw [hill, div_le_iff₀ hden]
  have := varOcc_le ha hx hZ
  nlinarith [Nat.cast_nonneg (α := ℝ) n]

/-- The fluctuation bound saturates exactly when the occupancy distribution is supported on the
empty and the fully bound state. -/
theorem mom_two_eq_iff (ha : ∀ j, 0 ≤ a j) (hx : 0 < x) :
    mom n a 2 x = (n : ℝ) * mom n a 1 x ↔ ∀ j, 0 < j → j < n → a j = 0 := by
  have hsum : (n : ℝ) * mom n a 1 x - mom n a 2 x
      = ∑ j ∈ range (n + 1), ((n : ℝ) - (j : ℝ)) * (j : ℝ) * (a j * x ^ j) := by
    unfold mom
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  have hnn : ∀ j ∈ range (n + 1), 0 ≤ ((n : ℝ) - (j : ℝ)) * (j : ℝ) * (a j * x ^ j) := by
    intro j hj
    have hjn : (j : ℝ) ≤ n := by
      have := Finset.mem_range.mp hj
      exact_mod_cast Nat.lt_succ_iff.mp this
    exact mul_nonneg (mul_nonneg (by linarith) (Nat.cast_nonneg j))
      (mul_nonneg (ha j) (pow_nonneg hx.le j))
  constructor
  · intro h
    have hzero : ∑ j ∈ range (n + 1), ((n : ℝ) - (j : ℝ)) * (j : ℝ) * (a j * x ^ j) = 0 := by
      rw [← hsum, h]; ring
    intro j hj0 hjn
    have hmem : j ∈ range (n + 1) := Finset.mem_range.mpr (by omega)
    have hterm := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp hzero j hmem
    have hjpos : (0 : ℝ) < (j : ℝ) := by exact_mod_cast hj0
    have hlt : (0 : ℝ) < (n : ℝ) - (j : ℝ) := by
      have : (j : ℝ) < n := by exact_mod_cast hjn
      linarith
    have hxpow : (0 : ℝ) < x ^ j := pow_pos hx j
    rcases (ha j).lt_or_eq with hpos | heq
    · exact absurd hterm (mul_pos (mul_pos hlt hjpos) (mul_pos hpos hxpow)).ne'
    · exact heq.symm
  · intro h
    have hz : ∑ j ∈ range (n + 1), ((n : ℝ) - (j : ℝ)) * (j : ℝ) * (a j * x ^ j) = 0 := by
      refine Finset.sum_eq_zero fun j hj => ?_
      rcases Nat.eq_zero_or_pos j with hj0 | hj0
      · subst hj0; ring
      · rcases lt_or_ge j n with hjn | hjn
        · rw [h j hj0 hjn]; ring
        · have hjeq : (j : ℝ) = n := by
            have h1 := Nat.lt_succ_iff.mp (Finset.mem_range.mp hj)
            have : j = n := le_antisymm h1 hjn
            exact_mod_cast this
          rw [hjeq]; ring
    rw [← hsum] at hz
    linarith

/-- **Equality forces an all-or-none ensemble.**  The Hill slope equals the valence at an
activity exactly when every partially bound state carries zero weight there. -/
theorem hill_eq_valence_iff (ha : ∀ j, 0 ≤ a j) (hx : 0 < x) (hZ : 0 < part n a x)
    (h0 : 0 < meanOcc n a x) (hn : meanOcc n a x < n) :
    hill n a x = n ↔ ∀ j, 0 < j → j < n → a j = 0 := by
  have hnpos : (0 : ℝ) < n := lt_trans h0 hn
  have hden : 0 < meanOcc n a x * ((n : ℝ) - meanOcc n a x) := mul_pos h0 (by linarith)
  have hkey : hill n a x = n ↔ mom n a 2 x = (n : ℝ) * mom n a 1 x := by
    rw [hill, div_eq_iff hden.ne', varOcc, meanOcc]
    constructor
    · intro h
      field_simp at h
      nlinarith [h, hZ, hnpos]
    · intro h
      rw [h]
      field_simp
  rw [hkey, mom_two_eq_iff ha hx]

/-- **The Hill slope is the logarithmic slope of the saturation curve.**  With
`theta = <N>/n`, `d ln(theta/(1-theta))/d ln x = hill`. -/
theorem hasDerivAt_logit (n : ℕ) (a : ℕ → ℝ) (hx : 0 < x) (hZ : part n a x ≠ 0)
    (h0 : meanOcc n a x ≠ 0) (hn : (n : ℝ) - meanOcc n a x ≠ 0) :
    HasDerivAt (fun y => Real.log (meanOcc n a y) - Real.log ((n : ℝ) - meanOcc n a y))
      (hill n a x / x) x := by
  have hm := hasDerivAt_meanOcc n a hx.ne' hZ
  have h1 : HasDerivAt (fun y => Real.log (meanOcc n a y))
      ((varOcc n a x / x) / meanOcc n a x) x := hm.log h0
  have hsub : HasDerivAt (fun y => (n : ℝ) - meanOcc n a y) (0 - varOcc n a x / x) x :=
    (hasDerivAt_const x ((n : ℝ))).sub hm
  have h2 : HasDerivAt (fun y => Real.log ((n : ℝ) - meanOcc n a y))
      ((0 - varOcc n a x / x) / ((n : ℝ) - meanOcc n a x)) x := hsub.log hn
  have h := h1.sub h2
  have heq : (varOcc n a x / x) / meanOcc n a x
      - (0 - varOcc n a x / x) / ((n : ℝ) - meanOcc n a x) = hill n a x / x := by
    rw [hill]
    field_simp
    ring
  exact heq ▸ h

/-! ## The bound is attained: the all-or-none polynomial -/

/-- The all-or-none binding polynomial `1 + x^n`: only the empty and the fully bound state are
populated. -/
def allOrNone (n : ℕ) : ℕ → ℝ := fun j => if j = 0 then 1 else if j = n then 1 else 0

theorem allOrNone_nonneg (n : ℕ) : ∀ j, 0 ≤ allOrNone n j := by
  intro j
  unfold allOrNone
  split
  · norm_num
  · split <;> norm_num

theorem part_allOrNone {n : ℕ} (hn : 0 < n) {x : ℝ} : part n (allOrNone n) x = 1 + x ^ n := by
  have hterm : ∀ j ∈ range (n + 1), allOrNone n j * x ^ j
      = (if j = 0 then (1 : ℝ) else 0) + (if j = n then x ^ n else 0) := by
    intro j _
    unfold allOrNone
    rcases eq_or_ne j 0 with hj0 | hj0
    · subst hj0; simp [hn.ne]
    · rcases eq_or_ne j n with hjn | hjn
      · subst hjn; simp [hj0]
      · simp [hj0, hjn]
  rw [part, Finset.sum_congr rfl hterm, Finset.sum_add_distrib]
  rw [Finset.sum_ite_eq' (range (n + 1)) 0 (fun _ => (1 : ℝ)),
    Finset.sum_ite_eq' (range (n + 1)) n (fun _ => x ^ n)]
  simp

theorem mom1_allOrNone {n : ℕ} (hn : 0 < n) {x : ℝ} :
    mom n (allOrNone n) 1 x = (n : ℝ) * x ^ n := by
  rw [mom, Finset.sum_eq_single n]
  · simp [allOrNone, hn.ne']
  · intro j _ hjn
    rcases Nat.eq_zero_or_pos j with hj0 | hj0
    · subst hj0; simp
    · simp [allOrNone, hjn, hj0.ne']
  · intro h
    exact absurd (Finset.mem_range.mpr (Nat.lt_succ_self n)) h

theorem meanOcc_allOrNone {n : ℕ} (hn : 0 < n) {x : ℝ} :
    meanOcc n (allOrNone n) x = (n : ℝ) * x ^ n / (1 + x ^ n) := by
  rw [meanOcc, mom1_allOrNone hn, part_allOrNone hn]

/-- **Sharpness.**  The all-or-none model has Hill slope exactly `n` at every activity. -/
theorem hill_allOrNone {n : ℕ} (hn : 0 < n) {x : ℝ} (hx : 0 < x) :
    hill n (allOrNone n) x = n := by
  have hxn : 0 < x ^ n := pow_pos hx n
  have hZ : 0 < part n (allOrNone n) x := by rw [part_allOrNone hn]; linarith
  have hmean : meanOcc n (allOrNone n) x = (n : ℝ) * x ^ n / (1 + x ^ n) := meanOcc_allOrNone hn
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  have h0 : 0 < meanOcc n (allOrNone n) x := by
    rw [hmean]; positivity
  have hlt : meanOcc n (allOrNone n) x < n := by
    rw [hmean, div_lt_iff₀ (by linarith)]
    nlinarith
  rw [hill_eq_valence_iff (allOrNone_nonneg n) hx hZ h0 hlt]
  intro j hj0 hjn
  unfold allOrNone
  simp [hj0.ne', hjn.ne]

end Basic

/-! ## Independent sites: the Hill slope cannot exceed one -/

/-- The occupancy probability of site `s` at activity `x` when the sites bind independently with
association constants `k s`. -/
noncomputable def pOcc (k : ℕ → ℝ) (x : ℝ) (s : ℕ) : ℝ := k s * x / (1 + k s * x)

/-- The product binding polynomial of `m` independent sites. -/
noncomputable def prodPart (m : ℕ) (k : ℕ → ℝ) (x : ℝ) : ℝ := ∏ s ∈ range m, (1 + k s * x)

/-- Mean occupancy of `m` independent sites. -/
noncomputable def meanInd (m : ℕ) (k : ℕ → ℝ) (x : ℝ) : ℝ := ∑ s ∈ range m, pOcc k x s

/-- Occupancy variance of `m` independent sites: a sum of Bernoulli variances. -/
noncomputable def varInd (m : ℕ) (k : ℕ → ℝ) (x : ℝ) : ℝ :=
  ∑ s ∈ range m, pOcc k x s * (1 - pOcc k x s)

/-- The Hill slope of `m` independent sites. -/
noncomputable def hillInd (m : ℕ) (k : ℕ → ℝ) (x : ℝ) : ℝ :=
  (m : ℝ) * varInd m k x / (meanInd m k x * ((m : ℝ) - meanInd m k x))

section Independent

variable {m : ℕ} {k : ℕ → ℝ} {x : ℝ}

theorem one_add_pos (hk : ∀ s, 0 < k s) (hx : 0 < x) (s : ℕ) : 0 < 1 + k s * x := by
  have := mul_pos (hk s) hx; linarith

/-- Derivative of a single site factor. -/
theorem hasDerivAt_site (c y : ℝ) : HasDerivAt (fun z : ℝ => 1 + c * z) c y := by
  have h : HasDerivAt (fun z : ℝ => 1 + c * z) (0 + c * 1) y :=
    (hasDerivAt_const y (1 : ℝ)).add (HasDerivAt.const_mul c (hasDerivAt_id y))
  rw [show (0 + c * 1 : ℝ) = c by ring] at h
  exact h

theorem pOcc_pos (hk : ∀ s, 0 < k s) (hx : 0 < x) (s : ℕ) : 0 < pOcc k x s :=
  div_pos (mul_pos (hk s) hx) (one_add_pos hk hx s)

theorem pOcc_lt_one (hk : ∀ s, 0 < k s) (hx : 0 < x) (s : ℕ) : pOcc k x s < 1 := by
  rw [pOcc, div_lt_one (one_add_pos hk hx s)]
  linarith

/-- The log-derivative of a product of site factors is the sum of the site log-derivatives. -/
theorem hasDerivAt_prodPart (hk : ∀ s, 0 < k s) (hx : 0 < x) (m : ℕ) :
    HasDerivAt (prodPart m k) (prodPart m k x * ∑ s ∈ range m, k s / (1 + k s * x)) x := by
  induction m with
  | zero => simpa [prodPart] using hasDerivAt_const x (1 : ℝ)
  | succ p ih =>
      have hp : HasDerivAt (fun y : ℝ => 1 + k p * y) (k p) x := hasDerivAt_site (k p) x
      have h : HasDerivAt (fun y : ℝ => prodPart p k y * (1 + k p * y))
          ((prodPart p k x * ∑ s ∈ range p, k s / (1 + k s * x)) * (1 + k p * x)
            + prodPart p k x * k p) x := ih.mul hp
      have hfun : (fun y : ℝ => prodPart p k y * (1 + k p * y)) = prodPart (p + 1) k := by
        funext y; simp [prodPart, Finset.prod_range_succ]
      rw [hfun] at h
      have hne : (1 + k p * x) ≠ 0 := (one_add_pos hk hx p).ne'
      have hstep : prodPart (p + 1) k x = prodPart p k x * (1 + k p * x) := by
        simp [prodPart, Finset.prod_range_succ]
      have hcancel : prodPart p k x * (1 + k p * x) * (k p / (1 + k p * x))
          = prodPart p k x * k p := by field_simp
      have heq : (prodPart p k x * ∑ s ∈ range p, k s / (1 + k s * x)) * (1 + k p * x)
            + prodPart p k x * k p
          = prodPart (p + 1) k x * ∑ s ∈ range (p + 1), k s / (1 + k s * x) := by
        rw [hstep, Finset.sum_range_succ]
        linear_combination -hcancel
      exact heq ▸ h

/-- For independent sites the thermodynamic occupancy is the sum of the site occupancies. -/
theorem occupancy_prodPart (hk : ∀ s, 0 < k s) (hx : 0 < x) (m : ℕ) :
    Valence.occupancy (prodPart m k) x = meanInd m k x := by
  have hpos : 0 < prodPart m k x := Finset.prod_pos fun s _ => one_add_pos hk hx s
  rw [Valence.occupancy, (hasDerivAt_prodPart hk hx m).deriv]
  have hsimp : x * (prodPart m k x * ∑ s ∈ range m, k s / (1 + k s * x)) / prodPart m k x
      = x * ∑ s ∈ range m, k s / (1 + k s * x) := by
    field_simp
  rw [hsimp, meanInd, Finset.mul_sum]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [pOcc]
  ring

/-- Fluctuation--response for independent sites: `d<N>/d ln x` is the sum of the Bernoulli
variances of the individual sites. -/
theorem hasDerivAt_meanInd (hk : ∀ s, 0 < k s) (hx : 0 < x) (m : ℕ) :
    HasDerivAt (meanInd m k) (varInd m k x / x) x := by
  have h : HasDerivAt (fun y : ℝ => ∑ s ∈ range m, pOcc k y s)
      (∑ s ∈ range m, k s / (1 + k s * x) ^ 2) x := by
    refine HasDerivAt.fun_sum fun (s : ℕ) _ => ?_
    have hden : HasDerivAt (fun y : ℝ => 1 + k s * y) (k s) x := hasDerivAt_site (k s) x
    have hnum : HasDerivAt (fun y : ℝ => k s * y) (k s) x := by
      have h0 : HasDerivAt (fun y : ℝ => k s * y) (k s * 1) x :=
        HasDerivAt.const_mul (k s) (hasDerivAt_id x)
      rw [show (k s * 1 : ℝ) = k s by ring] at h0
      exact h0
    have hne : (1 + k s * x) ≠ 0 := (one_add_pos hk hx s).ne'
    have hdiv := hnum.div hden hne
    have heq : (k s * (1 + k s * x) - k s * x * k s) / (1 + k s * x) ^ 2
        = k s / (1 + k s * x) ^ 2 := by
      field_simp
      ring
    exact heq ▸ hdiv
  have heq2 : ∑ s ∈ range m, k s / (1 + k s * x) ^ 2 = varInd m k x / x := by
    rw [varInd, Finset.sum_div]
    refine Finset.sum_congr rfl fun s _ => ?_
    have hne : (1 + k s * x) ≠ 0 := (one_add_pos hk hx s).ne'
    rw [pOcc]
    field_simp
    ring
  exact heq2 ▸ h

theorem meanInd_pos (hk : ∀ s, 0 < k s) (hx : 0 < x) (hm : 0 < m) : 0 < meanInd m k x :=
  Finset.sum_pos (fun s _ => pOcc_pos hk hx s) (by simp [Finset.nonempty_range_iff, hm.ne'])

theorem meanInd_lt (hk : ∀ s, 0 < k s) (hx : 0 < x) (hm : 0 < m) : meanInd m k x < m := by
  have h : meanInd m k x < ∑ _s ∈ range m, (1 : ℝ) :=
    Finset.sum_lt_sum_of_nonempty (by simp [Finset.nonempty_range_iff, hm.ne'])
      (fun s _ => pOcc_lt_one hk hx s)
  simpa using h

/-- **Independent sites cannot be steep.**  With `Z = prod_s (1 + k_s x)` the Hill slope is at
most one at every activity, whatever the individual affinities: steepness beyond one is
impossible without coupling between the sites. -/
theorem hillInd_le_one (hk : ∀ s, 0 < k s) (hx : 0 < x) (hm : 0 < m) : hillInd m k x ≤ 1 := by
  have hmean := meanInd_pos hk hx hm
  have hlt := meanInd_lt hk hx hm
  have hden : 0 < meanInd m k x * ((m : ℝ) - meanInd m k x) := mul_pos hmean (by linarith)
  rw [hillInd, div_le_one hden]
  have hCS : (∑ s ∈ range m, pOcc k x s) ^ 2 ≤ (m : ℝ) * ∑ s ∈ range m, pOcc k x s ^ 2 := by
    have h := sq_sum_le_card_mul_sum_sq (s := range m) (f := fun s => pOcc k x s)
    simpa using h
  have hvar : varInd m k x
      = ∑ s ∈ range m, pOcc k x s - ∑ s ∈ range m, pOcc k x s ^ 2 := by
    rw [varInd, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun s _ => by ring
  rw [hvar, meanInd]
  nlinarith [hCS]

/-- Identical independent sites attain the bound: the Hill slope is exactly one, the Langmuir
isotherm.  A measured slope below one therefore needs no negative cooperativity, only
heterogeneous affinities. -/
theorem hillInd_identical {c : ℝ} (hc : 0 < c) (hx : 0 < x) (hm : 0 < m) :
    hillInd m (fun _ => c) x = 1 := by
  have hcx : (0 : ℝ) < 1 + c * x := by nlinarith
  set p : ℝ := c * x / (1 + c * x) with hp
  have hppos : 0 < p := div_pos (mul_pos hc hx) hcx
  have hplt : p < 1 := by rw [hp, div_lt_one hcx]; linarith
  have hmean : meanInd m (fun _ => c) x = (m : ℝ) * p := by
    rw [meanInd]; simp [pOcc, ← hp]
  have hvar : varInd m (fun _ => c) x = (m : ℝ) * (p * (1 - p)) := by
    rw [varInd]; simp [pOcc, ← hp]
  have hmpos : (0 : ℝ) < m := by exact_mod_cast hm
  have hden : (m : ℝ) * p * ((m : ℝ) - (m : ℝ) * p) = (m : ℝ) ^ 2 * (p * (1 - p)) := by ring
  rw [hillInd, hmean, hvar, hden,
    div_eq_one_iff_eq (mul_pos (pow_pos hmpos 2)
      (mul_pos hppos (by linarith : (0 : ℝ) < 1 - p))).ne']
  ring

end Independent

end BindPoly

end IDR
