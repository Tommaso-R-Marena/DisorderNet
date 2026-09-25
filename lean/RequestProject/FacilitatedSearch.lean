import Mathlib

/-!
# Part CXLIX — Facilitated diffusion: what a disordered tail buys in a target search

Many DNA-binding proteins find their site far faster than three-dimensional diffusion allows,
and the accepted explanation is that they alternate between three-dimensional excursions and
one-dimensional sliding along the DNA — the sliding being mediated, in a large fraction of
cases, by a positively charged disordered tail that keeps a weak nonspecific grip on the
backbone.  A model of a disordered region that claims to explain such a tail is claiming
something quantitative about that alternation, and this file says exactly what.

Write `tau1` for the mean duration of a sliding excursion, `tau3` for the mean duration of a
three-dimensional hop, `D1` for the one-dimensional diffusion coefficient and `L` for the
number of sites to be searched.  One sliding excursion scans about `2√(D1 tau1)` sites, so the
mean search time is

    searchTime L D1 tau3 tau1 = L (tau1 + tau3) / (2 √(D1 tau1)).

We prove:

* `searchTime_ge_opt` — the search time is bounded below by `L √(tau3/D1)`, and
  `searchTime_at_opt` shows the bound is attained at `tau1 = tau3`.  **The optimal search
  spends equal time sliding and hopping** — a statement about the tail's grip, not about the
  folded domain.
* `searchTime_gt_opt_of_ne` — the optimum is strict: any other partition of the time is
  strictly slower.
* `speedup_eq` — at the optimum the search is faster than a pure three-dimensional search by
  exactly `√(D1 tau3)`, the number of sites scanned per excursion.
* `optimum_unidentifiable` — and the sharp caveat: the optimal search time depends on `D1`
  and `tau3` only through their ratio.  Multiplying both by the same factor leaves every
  measured association rate unchanged, so a rate measurement alone cannot say how fast the
  tail slides; it can only say how far it slides per hop.
-/

noncomputable section

namespace RequestProject.FacilitatedSearch

open Real

/-- Mean time to find one target among `L` sites, alternating sliding excursions of mean
duration `tau1` with three-dimensional hops of mean duration `tau3`. -/
def searchTime (L D1 tau3 tau1 : ℝ) : ℝ := L * (tau1 + tau3) / (2 * Real.sqrt (D1 * tau1))

/-- The best achievable mean search time. -/
def optimalTime (L D1 tau3 : ℝ) : ℝ := L * Real.sqrt (tau3 / D1)

/-- The number of sites scanned in one sliding excursion of the optimal duration. -/
def slidingLength (D1 tau3 : ℝ) : ℝ := Real.sqrt (D1 * tau3)

/-- **The optimal partition is the equal one.**  Sliding for exactly as long as hopping
attains the lower bound. -/
theorem searchTime_at_opt {L D1 tau3 : ℝ} (hD : 0 < D1) (h3 : 0 < tau3) :
    searchTime L D1 tau3 tau3 = optimalTime L D1 tau3 := by
  obtain ⟨w, hw0, rfl⟩ : ∃ w, 0 < w ∧ D1 = w ^ 2 :=
    ⟨Real.sqrt D1, Real.sqrt_pos.2 hD, (Real.sq_sqrt hD.le).symm⟩
  obtain ⟨v, hv0, rfl⟩ : ∃ v, 0 < v ∧ tau3 = v ^ 2 :=
    ⟨Real.sqrt tau3, Real.sqrt_pos.2 h3, (Real.sq_sqrt h3.le).symm⟩
  unfold searchTime optimalTime
  rw [show w ^ 2 * v ^ 2 = (w * v) ^ 2 by ring, Real.sqrt_sq (by positivity),
    show v ^ 2 / w ^ 2 = (v / w) ^ 2 by ring, Real.sqrt_sq (by positivity)]
  field_simp
  ring

/-- **No partition beats the equal one.** -/
theorem searchTime_ge_opt {L D1 tau3 tau1 : ℝ} (hL : 0 ≤ L) (hD : 0 < D1) (h3 : 0 < tau3)
    (h1 : 0 < tau1) :
    optimalTime L D1 tau3 ≤ searchTime L D1 tau3 tau1 := by
  obtain ⟨w, hw0, rfl⟩ : ∃ w, 0 < w ∧ D1 = w ^ 2 :=
    ⟨Real.sqrt D1, Real.sqrt_pos.2 hD, (Real.sq_sqrt hD.le).symm⟩
  obtain ⟨v, hv0, rfl⟩ : ∃ v, 0 < v ∧ tau3 = v ^ 2 :=
    ⟨Real.sqrt tau3, Real.sqrt_pos.2 h3, (Real.sq_sqrt h3.le).symm⟩
  obtain ⟨u, hu0, rfl⟩ : ∃ u, 0 < u ∧ tau1 = u ^ 2 :=
    ⟨Real.sqrt tau1, Real.sqrt_pos.2 h1, (Real.sq_sqrt h1.le).symm⟩
  unfold searchTime optimalTime
  rw [show w ^ 2 * u ^ 2 = (w * u) ^ 2 by ring, Real.sqrt_sq (by positivity),
    show v ^ 2 / w ^ 2 = (v / w) ^ 2 by ring, Real.sqrt_sq (by positivity),
    ← mul_div_assoc, div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_nonneg (mul_nonneg hL hw0.le) (sq_nonneg (u - v)), hw0.le, hu0.le, hv0.le]

/-- The optimum is strict: an unequal partition is strictly slower. -/
theorem searchTime_gt_opt_of_ne {L D1 tau3 tau1 : ℝ} (hL : 0 < L) (hD : 0 < D1)
    (h3 : 0 < tau3) (h1 : 0 < tau1) (hne : tau1 ≠ tau3) :
    optimalTime L D1 tau3 < searchTime L D1 tau3 tau1 := by
  obtain ⟨w, hw0, rfl⟩ : ∃ w, 0 < w ∧ D1 = w ^ 2 :=
    ⟨Real.sqrt D1, Real.sqrt_pos.2 hD, (Real.sq_sqrt hD.le).symm⟩
  obtain ⟨v, hv0, rfl⟩ : ∃ v, 0 < v ∧ tau3 = v ^ 2 :=
    ⟨Real.sqrt tau3, Real.sqrt_pos.2 h3, (Real.sq_sqrt h3.le).symm⟩
  obtain ⟨u, hu0, rfl⟩ : ∃ u, 0 < u ∧ tau1 = u ^ 2 :=
    ⟨Real.sqrt tau1, Real.sqrt_pos.2 h1, (Real.sq_sqrt h1.le).symm⟩
  have huv : u ≠ v := by
    intro hcon
    exact hne (by rw [hcon])
  have hsq : 0 < (u - v) ^ 2 := by
    have hne0 : u - v ≠ 0 := sub_ne_zero.2 huv
    positivity
  unfold searchTime optimalTime
  rw [show w ^ 2 * u ^ 2 = (w * u) ^ 2 by ring, Real.sqrt_sq (by positivity),
    show v ^ 2 / w ^ 2 = (v / w) ^ 2 by ring, Real.sqrt_sq (by positivity),
    ← mul_div_assoc, div_lt_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_pos (mul_pos hL hw0) hsq, hw0.le, hu0.le, hv0.le]

/-- **The speedup is the sliding length.**  A pure three-dimensional search visits the `L`
sites one hop at a time; the optimal alternating search is faster by exactly the number of
sites scanned per excursion. -/
theorem speedup_eq {L D1 tau3 : ℝ} (hL : 0 < L) (hD : 0 < D1) (h3 : 0 < tau3) :
    (L * tau3) / optimalTime L D1 tau3 = slidingLength D1 tau3 := by
  obtain ⟨w, hw0, rfl⟩ : ∃ w, 0 < w ∧ D1 = w ^ 2 :=
    ⟨Real.sqrt D1, Real.sqrt_pos.2 hD, (Real.sq_sqrt hD.le).symm⟩
  obtain ⟨v, hv0, rfl⟩ : ∃ v, 0 < v ∧ tau3 = v ^ 2 :=
    ⟨Real.sqrt tau3, Real.sqrt_pos.2 h3, (Real.sq_sqrt h3.le).symm⟩
  unfold optimalTime slidingLength
  rw [show w ^ 2 * v ^ 2 = (w * v) ^ 2 by ring, Real.sqrt_sq (by positivity),
    show v ^ 2 / w ^ 2 = (v / w) ^ 2 by ring, Real.sqrt_sq (by positivity)]
  field_simp

/-- **The rate measures the distance, not the speed.**  Scaling the sliding diffusion
coefficient and the hop time by the same factor leaves the optimal search time — and hence
every measured association rate — exactly unchanged. -/
theorem optimum_unidentifiable {L D1 tau3 k : ℝ} (hk : 0 < k) :
    optimalTime L (k * D1) (k * tau3) = optimalTime L D1 tau3 := by
  unfold optimalTime
  rw [mul_div_mul_left _ _ hk.ne']

/-- **The facilitated-search design law.**  For a disordered tail that alternates sliding
with hopping:

1. no partition of the time beats the equal one, and the equal one attains `L √(tau3/D1)`;
2. every unequal partition is strictly slower;
3. the speedup over a pure three-dimensional search is exactly the sliding length
   `√(D1 tau3)`;
4. and the optimum depends on the sliding coefficient and the hop time only through their
   ratio, so an association-rate measurement alone cannot separate them. -/
theorem facilitated_search_law {L D1 tau3 : ℝ} (hL : 0 < L) (hD : 0 < D1) (h3 : 0 < tau3) :
    searchTime L D1 tau3 tau3 = optimalTime L D1 tau3 ∧
    (∀ tau1 : ℝ, 0 < tau1 → optimalTime L D1 tau3 ≤ searchTime L D1 tau3 tau1) ∧
    (∀ tau1 : ℝ, 0 < tau1 → tau1 ≠ tau3 →
      optimalTime L D1 tau3 < searchTime L D1 tau3 tau1) ∧
    (L * tau3) / optimalTime L D1 tau3 = slidingLength D1 tau3 ∧
    (∀ k : ℝ, 0 < k → optimalTime L (k * D1) (k * tau3) = optimalTime L D1 tau3) :=
  ⟨searchTime_at_opt hD h3,
    fun _ h1 => searchTime_ge_opt hL.le hD h3 h1,
    fun _ h1 hne => searchTime_gt_opt_of_ne hL hD h3 h1 hne,
    speedup_eq hL hD h3,
    fun _ hk => optimum_unidentifiable hk⟩

end RequestProject.FacilitatedSearch
