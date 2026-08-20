/-
# Part CIII  Fragmentation, secondary nucleation, and the fibril length distribution

Parts XXXVII and LVI treat aggregation through the total fibril mass: the unsaturated equation
`M'' = κ²M` and the saturated logistic form.  The assumptions list recorded exactly what that
leaves out — "fibril fragmentation beyond its lumped contribution to `κ`, secondary nucleation as
a separate channel, and any distribution over fibril lengths: both parts track the total fibril
mass only".  This file supplies all three.

**The length distribution.**  A fibril population is a multiset of lengths, and fragmentation is
the move that replaces one fibril of length `a + b` by two of lengths `a` and `b`.

* `Splits.mass_eq`, `Splits.card_eq` — a split conserves total mass exactly and raises the fibril
  number by exactly one.  Iterating (`SplitsMany.mass_eq`, `SplitsMany.card_le`) gives the same
  statement for any fragmentation history.
* `mean_lt_of_splits` — hence the mean length strictly decreases at every split, and
  `mean_after` computes it: mass over the grown number.  Fragmentation cannot change the
  thioflavin signal at the moment it happens; what it changes is the number of growing ends,
  which is why it feeds back into the *rate*.
* `mass_blind_to_length` — **and the mass observable cannot see any of it.**  One fibril of length
  four and two of length two have the same mass, so no mass-reading experiment distinguishes
  them, at any time; the length distribution is a genuinely separate observable (it is what an
  electron micrograph or a light-scattering molar mass reports).

**The two channels.**  Fragmentation and secondary nucleation both make new ends in proportion to
existing fibril mass, so both enter the early-time curve through the same lumped constant
`κ² = 2 k₊ (k₂·m + k₋)` — secondary nucleation carrying a factor of monomer concentration `m`,
fragmentation not.

* `channels_unidentifiable` — **at a single monomer concentration they are not separable**: an
  explicit pair of parameter sets, one purely fragmentation-driven and one with a secondary
  nucleation channel, produce the identical mass curve at that concentration, while differing at
  another.  A fit to one thioflavin trace that reports a secondary nucleation rate is reporting a
  number the data does not contain.
* `channels_identified` — **and two concentrations suffice**: `κ²` is affine in `m`, so its slope
  and intercept recover `k₂` and `k₋` separately.  This is the concentration-dependence experiment,
  and the theorem says it is not merely good practice but exactly what identifiability requires.

**The shape.**  `sec_dominates` — a curve driven by either secondary channel eventually exceeds
*every* primary-only curve, whose mass is exactly quadratic in time.  The two mechanisms are
therefore distinguishable by the growth-phase shape even when the lag times agree, which is the
positive counterpart of the unidentifiability above: what the curve determines is the shape class,
not the individual rate constants.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR.FibrilLength

/-! ## The fibril length distribution -/

/-- A fibril population: the multiset of the lengths of the fibrils present. -/
abbrev Population := Multiset ℕ

/-- Total fibril mass: the sum of the lengths. -/
def mass (s : Population) : ℕ := s.sum

/-- Fibril number. -/
def number (s : Population) : ℕ := Multiset.card s

/-- One fragmentation event: a fibril of length `a + b` breaks into fibrils of lengths `a` and
`b`, both nonempty. -/
def Splits (s s' : Population) : Prop :=
  ∃ a b t, 1 ≤ a ∧ 1 ≤ b ∧ s = (a + b) ::ₘ t ∧ s' = a ::ₘ b ::ₘ t

/-- **Fragmentation conserves mass exactly.** -/
theorem Splits.mass_eq {s s' : Population} (h : Splits s s') : mass s' = mass s := by
  obtain ⟨a, b, t, _, _, hs, hs'⟩ := h
  subst hs; subst hs'
  simp [mass, Multiset.sum_cons, Nat.add_assoc]

/-- **Fragmentation raises the fibril number by exactly one.** -/
theorem Splits.card_eq {s s' : Population} (h : Splits s s') : number s' = number s + 1 := by
  obtain ⟨a, b, t, _, _, hs, hs'⟩ := h
  subst hs; subst hs'
  simp [number]

/-- A fragmentation history. -/
def SplitsMany : Population → Population → Prop := Relation.ReflTransGen Splits

theorem SplitsMany.mass_eq {s s' : Population} (h : SplitsMany s s') : mass s' = mass s := by
  induction h with
  | refl => rfl
  | tail _ hstep ih => rw [hstep.mass_eq, ih]

theorem SplitsMany.card_le {s s' : Population} (h : SplitsMany s s') : number s ≤ number s' := by
  induction h with
  | refl => exact le_rfl
  | tail _ hstep ih => rw [hstep.card_eq]; omega

/-- Mean fibril length. -/
noncomputable def meanLength (s : Population) : ℝ := (mass s : ℝ) / (number s : ℝ)

/-- After a split, the mean length is the same mass divided by one more fibril. -/
theorem mean_after {s s' : Population} (h : Splits s s') :
    meanLength s' = (mass s : ℝ) / ((number s : ℝ) + 1) := by
  unfold meanLength
  rw [h.mass_eq, h.card_eq]
  push_cast
  ring_nf

/-- **A split strictly lowers the mean fibril length** (given at least one fibril of positive
length before it). -/
theorem mean_lt_of_splits {s s' : Population} (h : Splits s s') (hm : 0 < mass s)
    (hn : 0 < number s) : meanLength s' < meanLength s := by
  have hm' : (0 : ℝ) < (mass s : ℝ) := by exact_mod_cast hm
  have hn' : (0 : ℝ) < (number s : ℝ) := by exact_mod_cast hn
  rw [mean_after h]
  unfold meanLength
  rw [div_lt_div_iff₀ (by linarith) hn']
  nlinarith

/-- **The mass observable is blind to the length distribution.**  A single fibril of length four
and two fibrils of length two carry the same mass and are indistinguishable to any experiment that
reads mass, yet they are different populations with different fibril numbers. -/
theorem mass_blind_to_length :
    mass {4} = mass {2, 2} ∧ number {4} ≠ number ({2, 2} : Population) := by
  constructor
  · simp [mass]
  · simp [number]

/-! ## The two secondary channels -/

/-- The lumped early-time growth constant squared, with elongation rate `kp`, secondary
nucleation rate `k2` (proportional to monomer concentration `m`) and fragmentation rate `km`. -/
def kappaSq (kp k2 km m : ℝ) : ℝ := 2 * kp * (k2 * m + km)

/-- **A single monomer concentration does not separate fragmentation from secondary
nucleation.**  Pure fragmentation with `k₋ = 2` and a mixed mechanism with `k₂ = 1, k₋ = 1` give
exactly the same growth constant at `m = 1`, and different ones at `m = 2`. -/
theorem channels_unidentifiable :
    kappaSq 1 0 2 1 = kappaSq 1 1 1 1 ∧ kappaSq 1 0 2 2 ≠ kappaSq 1 1 1 2 := by
  constructor
  · norm_num [kappaSq]
  · norm_num [kappaSq]

/-- **Two monomer concentrations determine both channels.**  The growth constant is affine in the
monomer concentration, so its values at two concentrations recover the secondary nucleation rate
and the fragmentation rate separately. -/
theorem channels_identified {kp k2 km k2' km' m₁ m₂ : ℝ} (hkp : 0 < kp) (hm : m₁ ≠ m₂)
    (h1 : kappaSq kp k2 km m₁ = kappaSq kp k2' km' m₁)
    (h2 : kappaSq kp k2 km m₂ = kappaSq kp k2' km' m₂) :
    k2 = k2' ∧ km = km' := by
  unfold kappaSq at h1 h2
  have e1 : k2 * m₁ + km = k2' * m₁ + km' := by
    have := mul_left_cancel₀ (by positivity : (2 * kp) ≠ 0) h1
    linarith [this]
  have e2 : k2 * m₂ + km = k2' * m₂ + km' := by
    have := mul_left_cancel₀ (by positivity : (2 * kp) ≠ 0) h2
    linarith [this]
  have hk2 : k2 = k2' := by
    have hd : (k2 - k2') * (m₁ - m₂) = 0 := by nlinarith [e1, e2]
    rcases mul_eq_zero.mp hd with h | h
    · linarith
    · exact absurd (by linarith : m₁ = m₂) hm
  refine ⟨hk2, ?_⟩
  rw [hk2] at e1
  linarith

/-! ## The shape of the growth phase -/

/-- Fibril mass with primary nucleation only: `P' = kn` constant, `M' = 2k₊P`, so the mass is
exactly quadratic in time. -/
noncomputable def massPrimary (a t : ℝ) : ℝ := a * t ^ 2

/-- Fibril mass with a secondary channel: the solution of `M'' = κ²M`, `M 0 = 0`, `M' 0 = v`. -/
noncomputable def massSecondary (v kappa t : ℝ) : ℝ := v / kappa * Real.sinh (kappa * t)

lemma cube_le_exp {x : ℝ} (hx : 0 ≤ x) : x ^ 3 / 6 ≤ Real.exp x := by
  have h := Real.sum_le_exp_of_nonneg hx 4
  have : x ^ 3 / 6 ≤ ∑ i ∈ Finset.range 4, x ^ i / (Nat.factorial i) := by
    simp [Finset.sum_range_succ, Nat.factorial]
    nlinarith [sq_nonneg x, pow_nonneg hx 2]
  linarith

/-- **A curve with a secondary channel eventually exceeds every primary-only curve.**  The
primary-only mass is quadratic in time; the secondary one is not bounded by any quadratic, so the
growth-phase shape distinguishes the mechanisms even when other features agree. -/
theorem sec_dominates {a v kappa : ℝ} (hv : 0 < v) (hk : 0 < kappa) :
    ∃ T : ℝ, 0 < T ∧ ∀ t, T ≤ t → massPrimary a t < massSecondary v kappa t := by
  refine ⟨max 1 (24 * (a + v / kappa) / (v * kappa ^ 2)), lt_of_lt_of_le one_pos (le_max_left _ _),
    ?_⟩
  intro t ht
  have ht1 : (1 : ℝ) ≤ t := le_trans (le_max_left _ _) ht
  have ht0 : (0 : ℝ) < t := lt_of_lt_of_le one_pos ht1
  have htT : 24 * (a + v / kappa) / (v * kappa ^ 2) ≤ t := le_trans (le_max_right _ _) ht
  have hvk : 0 < v * kappa ^ 2 := by positivity
  have hkey : 24 * (a + v / kappa) ≤ t * (v * kappa ^ 2) := by
    rw [div_le_iff₀ hvk] at htT
    linarith
  -- lower bound the sinh by a cubic
  have hx : 0 ≤ kappa * t := by positivity
  have hexp : (kappa * t) ^ 3 / 6 ≤ Real.exp (kappa * t) := cube_le_exp hx
  have hem : Real.exp (-(kappa * t)) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    linarith
  have hsinh : (kappa * t) ^ 3 / 12 - 1 / 2 ≤ Real.sinh (kappa * t) := by
    rw [Real.sinh_eq]
    linarith
  have hvk' : 0 < v / kappa := by positivity
  have hlow : v / kappa * ((kappa * t) ^ 3 / 12 - 1 / 2) ≤ massSecondary v kappa t := by
    unfold massSecondary
    exact mul_le_mul_of_nonneg_left hsinh hvk'.le
  have hcube : v / kappa * ((kappa * t) ^ 3 / 12) = (v * kappa ^ 2 / 12) * t ^ 3 := by
    field_simp
  have hquad : a * t ^ 2 + v / kappa * (1 / 2) < (v * kappa ^ 2 / 12) * t ^ 3 := by
    have h2 : (a + v / kappa) * t ^ 2 ≤ (v * kappa ^ 2 / 24) * t ^ 3 := by
      have := mul_le_mul_of_nonneg_right hkey (by positivity : (0:ℝ) ≤ t ^ 2 / 24)
      calc (a + v / kappa) * t ^ 2 = 24 * (a + v / kappa) * (t ^ 2 / 24) := by ring
        _ ≤ t * (v * kappa ^ 2) * (t ^ 2 / 24) := this
        _ = (v * kappa ^ 2 / 24) * t ^ 3 := by ring
    have h3 : 0 < (v * kappa ^ 2 / 24) * t ^ 3 := by positivity
    have h4 : v / kappa * (1 / 2) ≤ v / kappa * t ^ 2 := by
      have : (1 : ℝ) ≤ t ^ 2 := by nlinarith
      nlinarith [hvk'.le]
    nlinarith
  unfold massPrimary
  have : v / kappa * ((kappa * t) ^ 3 / 12 - 1 / 2)
      = (v * kappa ^ 2 / 12) * t ^ 3 - v / kappa * (1 / 2) := by
    rw [← hcube]; ring
  linarith [hlow, hquad, this ▸ hlow]

end IDR.FibrilLength
