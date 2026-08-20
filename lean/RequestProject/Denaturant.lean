/-
# Part XXXV  Titration curves: the cooperativity is fitted, not measured

A disordered region is routinely characterised by a chemical denaturation curve: a signal is
recorded against urea or guanidinium concentration and fitted with the two-state
linear-extrapolation model, which returns a midpoint and an `m`-value quoted as the
cooperativity of the transition.  For a region that has no folded state the transition is not
two-state at all — the chain expands gradually — and this file asks what the fit then reports.

The two-state fraction is the logistic curve `frac dG m RT x = sigmoid((m x − ΔG)/RT)`
(`sigmoid`, `frac`).  Everything below is exact.

* `frac_midpoint`, `frac_hasDerivAt`, `frac_deriv_midpoint` -- the curve passes through `1/2` at
  `x = ΔG/m` with slope exactly `m/(4RT)`.  So `m` *is* four `RT` times the measured midpoint
  slope (`m_eq_four_RT_slope`) — a restatement of the slope, not an independent quantity.
* `fit_matches_any_curve` -- consequently *every* signal that crosses the midpoint with a
  positive slope is matched there, in value and in slope, by a two-state model.  A fit that
  reproduces the midpoint and the steepness of the transition is no evidence at all that there
  are two states.
* `sigmoid_le_tangent`, `sigmoid_ge_tangent`, `sigmoid_tangent_cubic` -- and the agreement is not
  merely first order: the logistic curve deviates from its own midpoint tangent by at most
  `|u|³/48` in the reduced variable `u = m(x − x½)/RT`, and always on one side.  Translating
  (`frac_close_to_linear`), a strictly *linear*, non-cooperative expansion of a disordered chain
  is reproduced by a two-state model with `m = 4RT·s` to within `(4s|x − x½|)³/48`.  A cubic
  discrepancy is what has to be resolved, over the whole titration and against experimental
  noise, before a fitted `m`-value can be called a cooperativity.
* `frac_mem_Ioo` -- the one qualitative difference is saturation: the two-state curve is confined
  to `(0,1)`, whereas a gradual expansion is not.  The information that distinguishes them
  therefore lives in the *ends* of the titration, exactly where the baselines are fitted.

Design consequence, the same as for the scaling exponent of Part XXXI: a model of a disordered
region should be compared with the titration curve itself, forward-modelled, and not with a
fitted `ΔG` and `m`-value, which are a repackaging of one point and one slope.
-/
import Mathlib

set_option autoImplicit false

namespace Denat

/-! ### Two inequalities for `tanh` -/

private lemma sinh_le_self_mul_cosh {y : ℝ} (hy : 0 ≤ y) : Real.sinh y ≤ y * Real.cosh y := by
  set f : ℝ → ℝ := fun t => t * Real.cosh t - Real.sinh t with hf
  have hderiv : ∀ t : ℝ, HasDerivAt f (t * Real.sinh t) t := by
    intro t
    have h1 : HasDerivAt (fun t : ℝ => t * Real.cosh t) (1 * Real.cosh t + t * Real.sinh t) t :=
      (hasDerivAt_id t).mul (Real.hasDerivAt_cosh t)
    simpa using h1.sub (Real.hasDerivAt_sinh t)
  have hmono : MonotoneOn f (Set.Ici 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
    · exact Continuous.continuousOn (by fun_prop)
    · intro t _
      exact (hderiv t).differentiableAt.differentiableWithinAt
    · intro t ht
      simp only [interior_Ici, Set.mem_Ioi] at ht
      rw [(hderiv t).deriv]
      exact mul_nonneg ht.le (Real.sinh_nonneg_iff.mpr ht.le)
  have hle := hmono Set.self_mem_Ici (Set.mem_Ici.mpr hy) hy
  simp only [hf] at hle
  norm_num at hle
  linarith

/-- `tanh y ≤ y` for `y ≥ 0`. -/
lemma tanh_le_self {y : ℝ} (hy : 0 ≤ y) : Real.tanh y ≤ y := by
  rw [Real.tanh_eq_sinh_div_cosh, div_le_iff₀ (Real.cosh_pos y)]
  exact sinh_le_self_mul_cosh hy

lemma hasDerivAt_tanh (y : ℝ) : HasDerivAt Real.tanh (1 / (Real.cosh y) ^ 2) y := by
  have h := (Real.hasDerivAt_sinh y).div (Real.hasDerivAt_cosh y) (Real.cosh_pos y).ne'
  have hfun : Real.tanh = fun x => Real.sinh x / Real.cosh x := by
    funext x; exact Real.tanh_eq_sinh_div_cosh x
  have hid : (Real.cosh y * Real.cosh y - Real.sinh y * Real.sinh y) / Real.cosh y ^ 2
      = 1 / Real.cosh y ^ 2 := by
    have h2 := Real.cosh_sq_sub_sinh_sq y
    field_simp
    nlinarith [h2]
  rw [hfun]
  simpa [hid] using h

/-- `y − y³/3 ≤ tanh y` for `y ≥ 0`: the cubic Taylor bound, proved from `tanh y ≤ y`. -/
lemma tanh_ge_sub_cube {y : ℝ} (hy : 0 ≤ y) : y - y ^ 3 / 3 ≤ Real.tanh y := by
  set g : ℝ → ℝ := fun t => Real.tanh t - t + t ^ 3 / 3 with hg
  have hderiv : ∀ t : ℝ, HasDerivAt g (1 / (Real.cosh t) ^ 2 - 1 + t ^ 2) t := by
    intro t
    have h1 := (hasDerivAt_tanh t).sub (hasDerivAt_id t)
    have h2 : HasDerivAt (fun s : ℝ => s ^ 3 / 3) (t ^ 2) t := by
      have := (hasDerivAt_pow 3 t).div_const 3
      simpa using this
    simpa using h1.add h2
  have hmono : MonotoneOn g (Set.Ici 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
    · have hcont : Continuous g := by
        have htanh : Continuous Real.tanh := by
          have : Real.tanh = fun x => Real.sinh x / Real.cosh x := by
            funext x; exact Real.tanh_eq_sinh_div_cosh x
          rw [this]
          exact Real.continuous_sinh.div Real.continuous_cosh (fun x => (Real.cosh_pos x).ne')
        rw [hg]
        exact (htanh.sub continuous_id).add (by fun_prop)
      exact hcont.continuousOn
    · intro t _
      exact (hderiv t).differentiableAt.differentiableWithinAt
    · intro t ht
      simp only [interior_Ici, Set.mem_Ioi] at ht
      rw [(hderiv t).deriv]
      have htanh : Real.tanh t ≤ t := tanh_le_self ht.le
      have htanh0 : 0 ≤ Real.tanh t := by
        rw [Real.tanh_eq_sinh_div_cosh]
        exact div_nonneg (Real.sinh_nonneg_iff.mpr ht.le) (Real.cosh_pos t).le
      have hsq : 1 / (Real.cosh t) ^ 2 = 1 - (Real.tanh t) ^ 2 := by
        have h2 := Real.cosh_sq_sub_sinh_sq t
        rw [Real.tanh_eq_sinh_div_cosh, div_pow]
        field_simp
        nlinarith [h2]
      rw [hsq]
      nlinarith
  have hle := hmono Set.self_mem_Ici (Set.mem_Ici.mpr hy) hy
  simp only [hg] at hle
  norm_num at hle
  linarith

/-! ### The two-state (linear-extrapolation) titration curve -/

/-- The logistic curve. -/
noncomputable def sigmoid (u : ℝ) : ℝ := 1 / (1 + Real.exp (-u))

/-- The two-state fraction at denaturant concentration `x`, with unfolding free energy `dG`,
`m`-value `m` and thermal energy `RT`. -/
noncomputable def frac (dG m RT x : ℝ) : ℝ := sigmoid ((m * x - dG) / RT)

lemma sigmoid_eq_tanh (u : ℝ) : sigmoid u = 1 / 2 + (1 / 2) * Real.tanh (u / 2) := by
  rw [sigmoid, Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have h1 : Real.exp (-u) = Real.exp (-(u / 2)) * Real.exp (-(u / 2)) := by
    rw [← Real.exp_add]; ring_nf
  have h2 : Real.exp (u / 2) * Real.exp (-(u / 2)) = 1 := by
    rw [← Real.exp_add]; simp
  have hp : (0 : ℝ) < Real.exp (u / 2) := Real.exp_pos _
  have hn : (0 : ℝ) < Real.exp (-(u / 2)) := Real.exp_pos _
  rw [h1]
  field_simp
  nlinarith [h2, hp, hn]

@[simp] lemma sigmoid_zero : sigmoid 0 = 1 / 2 := by
  simp [sigmoid]
  norm_num

/-- The two-state curve is confined to `(0,1)`: saturation is its only qualitative feature. -/
lemma frac_mem_Ioo (dG m RT x : ℝ) : frac dG m RT x ∈ Set.Ioo (0 : ℝ) 1 := by
  have hpos : (0 : ℝ) < Real.exp (-((m * x - dG) / RT)) := Real.exp_pos _
  constructor
  · exact div_pos one_pos (by linarith)
  · rw [frac, sigmoid, div_lt_one (by linarith)]
    linarith

lemma hasDerivAt_sigmoid (u : ℝ) :
    HasDerivAt sigmoid (sigmoid u * (1 - sigmoid u)) u := by
  have hpos : (0 : ℝ) < 1 + Real.exp (-u) := by
    have := Real.exp_pos (-u); linarith
  have hden : HasDerivAt (fun t : ℝ => 1 + Real.exp (-t)) (-Real.exp (-u)) u := by
    have h1 : HasDerivAt (fun t : ℝ => Real.exp (-t)) (-Real.exp (-u)) u := by
      have := (Real.hasDerivAt_exp (-u)).comp u ((hasDerivAt_id u).neg)
      simpa using this
    simpa using h1.const_add 1
  have h := hden.inv (ne_of_gt hpos)
  have hval : sigmoid u * (1 - sigmoid u)
      = -(-Real.exp (-u)) / (1 + Real.exp (-u)) ^ 2 := by
    rw [sigmoid]
    field_simp
    ring
  have hfun : sigmoid = fun t : ℝ => (1 + Real.exp (-t))⁻¹ := by
    funext t; rw [sigmoid, one_div]
  have hfinal : HasDerivAt sigmoid (-(-Real.exp (-u)) / (1 + Real.exp (-u)) ^ 2) u := by
    rw [hfun]
    simpa using h
  rw [hval]
  exact hfinal

/-- The curve crosses `1/2` at `x = ΔG/m`. -/
theorem frac_midpoint {dG m RT : ℝ} (hm : m ≠ 0) : frac dG m RT (dG / m) = 1 / 2 := by
  have h1 : m * (dG / m) = dG := by field_simp
  have h2 : (m * (dG / m) - dG) / RT = 0 := by rw [h1, sub_self, zero_div]
  rw [frac, h2, sigmoid_zero]

theorem frac_hasDerivAt {dG m RT : ℝ} (x : ℝ) :
    HasDerivAt (frac dG m RT)
      ((m / RT) * (sigmoid ((m * x - dG) / RT) * (1 - sigmoid ((m * x - dG) / RT)))) x := by
  have hinner : HasDerivAt (fun t : ℝ => (m * t - dG) / RT) (m / RT) x := by
    have h1 : HasDerivAt (fun t : ℝ => m * t - dG) m x := by
      simpa using ((hasDerivAt_id x).const_mul m).sub_const dG
    simpa [div_eq_mul_inv, mul_comm, mul_assoc] using h1.div_const RT
  have := (hasDerivAt_sigmoid ((m * x - dG) / RT)).comp x hinner
  simpa [frac, mul_comm] using this

/-- **The `m`-value is four `RT` times the midpoint slope.** -/
theorem frac_deriv_midpoint {dG m RT : ℝ} (hm : m ≠ 0) (hRT : RT ≠ 0) :
    HasDerivAt (frac dG m RT) (m / (4 * RT)) (dG / m) := by
  have h1 : m * (dG / m) = dG := by field_simp
  have hzero : (m * (dG / m) - dG) / RT = 0 := by rw [h1, sub_self, zero_div]
  have h := frac_hasDerivAt (dG := dG) (m := m) (RT := RT) (dG / m)
  rw [hzero, sigmoid_zero] at h
  have hval : (m / RT) * ((1 / 2 : ℝ) * (1 - 1 / 2)) = m / (4 * RT) := by
    field_simp
    ring
  rwa [hval] at h

/-- Restated: a measured midpoint slope `s` *is* the reported `m`-value, divided by `4RT`. -/
theorem m_eq_four_RT_slope {dG m RT s : ℝ} (hm : m ≠ 0) (hRT : RT ≠ 0)
    (h : HasDerivAt (frac dG m RT) s (dG / m)) : m = 4 * RT * s := by
  have h2 := frac_deriv_midpoint (dG := dG) (m := m) hm hRT
  have := h.unique h2
  field_simp at this
  linarith

/-- **Every transition with a positive midpoint slope is matched by a two-state fit.**  Given a
midpoint `x₀` and a slope `s > 0`, there is a two-state model passing through `1/2` at `x₀` with
exactly that slope, and its `m`-value is `4RT·s`. -/
theorem fit_matches_any_curve {RT s x0 : ℝ} (hRT : 0 < RT) (hs : 0 < s) :
    ∃ dG m : ℝ, m = 4 * RT * s ∧ dG = m * x0 ∧ frac dG m RT x0 = 1 / 2 ∧
      HasDerivAt (frac dG m RT) s x0 := by
  obtain ⟨m, hm_def⟩ : ∃ m : ℝ, m = 4 * RT * s := ⟨_, rfl⟩
  have hmpos : 0 < m := by rw [hm_def]; positivity
  obtain ⟨dG, hdG⟩ : ∃ dG : ℝ, dG = m * x0 := ⟨_, rfl⟩
  have hx : dG / m = x0 := by rw [hdG]; field_simp
  refine ⟨dG, m, hm_def, hdG, ?_, ?_⟩
  · rw [← hx]
    exact frac_midpoint hmpos.ne'
  · rw [← hx]
    have h := frac_deriv_midpoint (dG := dG) (m := m) hmpos.ne' hRT.ne'
    have hval : m / (4 * RT) = s := by rw [hm_def]; field_simp
    rwa [hval] at h

/-! ### How closely the two-state curve tracks a straight line -/

/-- The logistic curve lies below its midpoint tangent to the right of the midpoint. -/
theorem sigmoid_le_tangent {u : ℝ} (hu : 0 ≤ u) : sigmoid u ≤ 1 / 2 + u / 4 := by
  have h := tanh_le_self (y := u / 2) (by linarith)
  rw [sigmoid_eq_tanh]
  linarith

/-- And above it to the left. -/
theorem sigmoid_ge_tangent {u : ℝ} (hu : u ≤ 0) : 1 / 2 + u / 4 ≤ sigmoid u := by
  have h := tanh_le_self (y := -u / 2) (by linarith)
  have hodd : Real.tanh (u / 2) = -Real.tanh (-u / 2) := by
    rw [show (-u / 2 : ℝ) = -(u / 2) by ring, Real.tanh_neg]
    ring
  rw [sigmoid_eq_tanh, hodd]
  linarith

/-- `|tanh y − y| ≤ |y|³/3`, for every real `y`. -/
lemma abs_tanh_sub_le (y : ℝ) : |Real.tanh y - y| ≤ |y| ^ 3 / 3 := by
  have hpos : ∀ z : ℝ, 0 ≤ z → |Real.tanh z - z| ≤ |z| ^ 3 / 3 := by
    intro z hz
    have h1 := tanh_le_self hz
    have h2 := tanh_ge_sub_cube hz
    rw [abs_of_nonpos (by linarith), abs_of_nonneg hz]
    linarith
  rcases le_or_gt 0 y with hy | hy
  · exact hpos y hy
  · have key := hpos (-y) (by linarith)
    rw [Real.tanh_neg, show -Real.tanh y - -y = -(Real.tanh y - y) by ring, abs_neg,
      abs_neg] at key
    exact key

/-- **The logistic curve agrees with its midpoint tangent to third order.** -/
theorem sigmoid_tangent_cubic (u : ℝ) : |sigmoid u - (1 / 2 + u / 4)| ≤ |u| ^ 3 / 48 := by
  have h := abs_tanh_sub_le (u / 2)
  have hu2 : |u / 2| = |u| / 2 := by rw [abs_div]; norm_num
  rw [hu2] at h
  have hcube : (|u| / 2) ^ 3 = |u| ^ 3 / 8 := by ring
  rw [hcube] at h
  have heq : sigmoid u - (1 / 2 + u / 4) = (1 / 2) * (Real.tanh (u / 2) - u / 2) := by
    rw [sigmoid_eq_tanh]; ring
  rw [heq, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2)]
  linarith

/-- **A gradual, non-cooperative expansion is reproduced by a two-state fit to cubic order.**
With the `m`-value set to `4RT·s`, the two-state curve differs from the straight line of slope
`s` through the midpoint by at most `(4 s |x − x₀|)³/48`. -/
theorem frac_close_to_linear {RT s x0 x : ℝ} (hRT : 0 < RT) (hs : 0 < s) :
    |frac ((4 * RT * s) * x0) (4 * RT * s) RT x - (1 / 2 + s * (x - x0))|
      ≤ |4 * s * (x - x0)| ^ 3 / 48 := by
  have hRTne : RT ≠ 0 := ne_of_gt hRT
  have harg : ((4 * RT * s) * x - (4 * RT * s) * x0) / RT = 4 * s * (x - x0) := by
    field_simp
  have hlin : (1 : ℝ) / 2 + s * (x - x0) = 1 / 2 + (4 * s * (x - x0)) / 4 := by ring
  rw [frac, harg, hlin]
  exact sigmoid_tangent_cubic _

end Denat
