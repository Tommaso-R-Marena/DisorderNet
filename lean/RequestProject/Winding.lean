/-
# Topological entanglement of a disordered chain: the winding budget

Every axis developed so far in this project treats a conformational ensemble through *metric*
data -- distances, radii, contacts, populations.  A disordered region in a cell is also subject
to constraints of a completely different kind: **topological** ones.  An intrinsically
disordered region threads through pores, wraps around partner helices and DNA, and in a
condensate it entangles with its neighbours; the amount of wrapping cannot be changed by any
motion that keeps the chain intact and keeps it out of the excluded volume of the object it is
wrapped around.  Entanglement is therefore a property of the ensemble that no amount of local
energy refinement can adjust, and a model that gets it wrong is wrong in a way that no
reweighting can repair.

This file builds the exact arithmetic of that constraint in the simplest setting in which it is
already sharp: a chain projected onto the plane transverse to a straight axis (a partner helix,
a pore, a filament), with the axis at the origin.

* `turn` / `totalTurn` / `winding` -- the discrete turning angle of a step as seen from the
  axis, its sum along the chain, and the winding number in turns.
* `abs_turn_le` -- **one step can only wrap so far**: a step of length at most `b` taken at
  distance at least `R` from the axis turns the chain by at most `π b / (2 R)` radians.  Proved
  from the chord bound `2 R |sin (θ/2)| ≤ b` (`two_mul_abs_sin_half_le`) and Jordan's
  inequality.
* `abs_winding_le` -- **the winding budget**: a chain of `n` bonds of length at most `b` that
  never comes closer than `R` to the axis has `|winding| ≤ n b / (4 R)` turns.
* `length_demand_of_winding` -- the same inequality read as a demand on the model: `k` turns of
  wrapping at exclusion radius `R` require at least `4 k R / b` residues.  Excluded volume and
  connectivity alone -- no force field, no sampling -- fix the number of residues needed to
  produce an observed amount of threading.
* `winding_int_of_closed` -- for a closed loop the winding number is an integer.
* `winding_invariant_of_deformation` -- **topological protection**: along any continuous
  deformation of a closed chain that keeps every residue outside the exclusion radius and keeps
  every bond shorter than `2 R`, the winding number never changes.  Unthreading is not a
  question of energy; within these constraints it is impossible.

Companion file `RequestProject.Threading` puts the budget to work on ensembles and data.
-/
import Mathlib

namespace RequestProject.Winding

open Finset Complex

/-- The turning angle of the step `z → w` as seen from the axis at the origin: the argument of
the ratio, which is the signed angle swept at the axis.  For steps shorter than twice the
exclusion radius this is the unique angle in `(-π, π)` (`abs_turn_lt_pi`). -/
noncomputable def turn (z w : ℂ) : ℝ := Complex.arg (w / z)

/-- The total angle swept at the axis by the first `n` bonds of the chain `p`. -/
noncomputable def totalTurn (p : ℕ → ℂ) (n : ℕ) : ℝ :=
  ∑ i ∈ range n, turn (p i) (p (i + 1))

/-- The winding number of the first `n` bonds of the chain, measured in whole turns. -/
noncomputable def winding (p : ℕ → ℂ) (n : ℕ) : ℝ := totalTurn p n / (2 * Real.pi)

theorem totalTurn_zero (p : ℕ → ℂ) : totalTurn p 0 = 0 := by simp [totalTurn]

theorem totalTurn_succ (p : ℕ → ℂ) (n : ℕ) :
    totalTurn p (n + 1) = totalTurn p n + turn (p n) (p (n + 1)) := by
  simp [totalTurn, Finset.sum_range_succ]

/-! ### The chord bound -/

/-- **Chord bound.**  If two consecutive residues are both at distance at least `R` from the
axis and are at most `b` apart, the angle `θ` they subtend at the axis satisfies
`2 R |sin (θ/2)| ≤ b`.  This is the exact geometric content of "a short step taken far from the
axis cannot wrap much"; everything else in this file is a consequence. -/
theorem two_mul_abs_sin_half_le {R b : ℝ} {z w : ℂ} (hR : 0 < R) (hz : R ≤ ‖z‖) (hw : R ≤ ‖w‖)
    (hb : ‖w - z‖ ≤ b) : 2 * R * |Real.sin (turn z w / 2)| ≤ b := by
  have hz0 : z ≠ 0 := by intro h; rw [h] at hz; simp at hz; linarith
  have hw0 : w ≠ 0 := by intro h; rw [h] at hw; simp at hw; linarith
  set u : ℂ := w / z with hu
  have hu0 : u ≠ 0 := div_ne_zero hw0 hz0
  set θ := turn z w with hθ
  have hcos : Real.cos θ = u.re / ‖u‖ := Complex.cos_arg hu0
  have hnu : ‖u‖ = ‖w‖ / ‖z‖ := by rw [hu, norm_div]
  have hznz : (0 : ℝ) < ‖z‖ := by positivity
  have hwz : w = u * z := by rw [hu]; field_simp
  have hexp : ‖w - z‖ ^ 2 = ‖z‖ ^ 2 * (1 + ‖u‖ ^ 2 - 2 * u.re) := by
    have h0 : w - z = -(z * ((1 : ℂ) - u)) := by rw [hwz]; ring
    rw [h0]
    have h1 : ‖(1 : ℂ) - u‖ ^ 2 = 1 + ‖u‖ ^ 2 - 2 * u.re := by
      simp [Complex.sq_norm, Complex.normSq_apply, Complex.sub_re, Complex.sub_im]; ring
    rw [norm_neg, norm_mul, mul_pow, h1]
  have hure : u.re = ‖u‖ * Real.cos θ := by rw [hcos]; field_simp
  have hkey : ‖w - z‖ ^ 2 = ‖z‖ ^ 2 + ‖w‖ ^ 2 - 2 * ‖z‖ * ‖w‖ * Real.cos θ := by
    rw [hexp, hure, hnu]; field_simp
  have hhalf : Real.cos θ = 1 - 2 * Real.sin (θ / 2) ^ 2 := by
    have h := Real.cos_two_mul' (θ / 2)
    have h2 := Real.sin_sq_add_cos_sq (θ / 2)
    have h3 : 2 * (θ / 2) = θ := by ring
    rw [h3] at h; nlinarith
  have hprod : R ^ 2 ≤ ‖z‖ * ‖w‖ := by nlinarith [norm_nonneg z, norm_nonneg w]
  have hsin : 4 * R ^ 2 * Real.sin (θ / 2) ^ 2 ≤ ‖w - z‖ ^ 2 := by
    rw [hkey, hhalf]
    nlinarith [sq_nonneg (‖z‖ - ‖w‖), sq_nonneg (Real.sin (θ / 2))]
  have hb0 : 0 ≤ b := le_trans (norm_nonneg _) hb
  nlinarith [abs_nonneg (Real.sin (θ / 2)), sq_abs (Real.sin (θ / 2)), norm_nonneg (w - z),
    sq_nonneg (2 * R * |Real.sin (θ / 2)| - b), sq_nonneg (‖w - z‖ - b)]

/-- **How far one step can wrap.**  A bond of length at most `b`, both of whose ends are at
distance at least `R` from the axis, sweeps at most `π b / (2 R)` radians at the axis. -/
theorem abs_turn_le {R b : ℝ} {z w : ℂ} (hR : 0 < R) (hz : R ≤ ‖z‖) (hw : R ≤ ‖w‖)
    (hb : ‖w - z‖ ≤ b) : |turn z w| ≤ Real.pi * b / (2 * R) := by
  have hchord := two_mul_abs_sin_half_le hR hz hw hb
  set θ := turn z w with hθ
  have habsθ : |θ| ≤ Real.pi := Complex.abs_arg_le_pi _
  have hpi := Real.pi_pos
  have hjordan : 2 / Real.pi * (|θ| / 2) ≤ Real.sin (|θ| / 2) :=
    Real.mul_le_sin (by positivity) (by linarith [abs_nonneg θ])
  have hsinabs : Real.sin (|θ| / 2) = |Real.sin (θ / 2)| := by
    rcases abs_cases θ with ⟨h1, _⟩ | ⟨h1, _⟩
    · rw [h1, abs_of_nonneg]
      exact Real.sin_nonneg_of_nonneg_of_le_pi (by linarith) (by linarith)
    · have hneg : Real.sin (θ / 2) ≤ 0 := by
        have h4 : 0 ≤ Real.sin (-(θ / 2)) :=
          Real.sin_nonneg_of_nonneg_of_le_pi (by linarith) (by linarith)
        rw [Real.sin_neg] at h4; linarith
      rw [h1, abs_of_nonpos hneg, show -θ / 2 = -(θ / 2) by ring, Real.sin_neg]
  rw [hsinabs] at hjordan
  have h6 : (2 * R) * (2 / Real.pi * (|θ| / 2)) ≤ (2 * R) * |Real.sin (θ / 2)| :=
    mul_le_mul_of_nonneg_left hjordan (by positivity)
  have h7 : (2 * R * |θ|) / Real.pi ≤ b := by
    have h8 : (2 * R * |θ|) / Real.pi = (2 * R) * (2 / Real.pi * (|θ| / 2)) := by field_simp
    rw [h8]; linarith
  rw [div_le_iff₀ hpi] at h7
  rw [le_div_iff₀ (by positivity : (0 : ℝ) < 2 * R)]
  linarith

/-- With bonds strictly shorter than the diameter `2 R` of the excluded region, no single step
can reverse the direction seen from the axis: the turning angle stays strictly inside
`(-π, π)`.  This is what makes the winding number well defined and continuous. -/
theorem abs_turn_lt_pi {R b : ℝ} {z w : ℂ} (hR : 0 < R) (hz : R ≤ ‖z‖) (hw : R ≤ ‖w‖)
    (hb : ‖w - z‖ ≤ b) (hbR : b < 2 * R) : |turn z w| < Real.pi := by
  have hchord := two_mul_abs_sin_half_le hR hz hw hb
  set θ := turn z w with hθ
  have habsθ : |θ| ≤ Real.pi := Complex.abs_arg_le_pi _
  rcases lt_or_eq_of_le habsθ with h | h
  · exact h
  · exfalso
    have h1 : |Real.sin (θ / 2)| = 1 := by
      rcases abs_cases θ with ⟨he, _⟩ | ⟨he, _⟩
      · have : θ = Real.pi := by rw [← he, h]
        rw [this]; simp [Real.sin_pi_div_two]
      · have : θ = -Real.pi := by
          have := he.symm.trans h
          linarith
        rw [this, show -Real.pi / 2 = -(Real.pi / 2) by ring, Real.sin_neg]
        simp [Real.sin_pi_div_two]
    rw [h1] at hchord
    linarith

/-! ### The winding budget -/

/-- **The winding budget, in radians.**  A chain of `n` bonds, each of length at most `b`, all of
whose residues stay at distance at least `R` from the axis, sweeps at most `n π b / (2 R)`
radians in total. -/
theorem abs_totalTurn_le {R b : ℝ} {p : ℕ → ℂ} {n : ℕ} (hR : 0 < R)
    (hfar : ∀ i, R ≤ ‖p i‖) (hstep : ∀ i, ‖p (i + 1) - p i‖ ≤ b) :
    |totalTurn p n| ≤ n * (Real.pi * b / (2 * R)) := by
  calc |totalTurn p n| ≤ ∑ i ∈ range n, |turn (p i) (p (i + 1))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i ∈ range n, Real.pi * b / (2 * R) :=
        Finset.sum_le_sum fun i _ => abs_turn_le hR (hfar i) (hfar (i + 1)) (hstep i)
    _ = n * (Real.pi * b / (2 * R)) := by simp [mul_comm]

/-- **The winding budget, in turns.**  `|winding| ≤ n b / (4 R)`: connectivity and excluded
volume alone cap how many times a disordered region of `n` residues can wrap an object of
exclusion radius `R`. -/
theorem abs_winding_le {R b : ℝ} {p : ℕ → ℂ} {n : ℕ} (hR : 0 < R)
    (hfar : ∀ i, R ≤ ‖p i‖) (hstep : ∀ i, ‖p (i + 1) - p i‖ ≤ b) :
    |winding p n| ≤ n * b / (4 * R) := by
  have hpi := Real.pi_pos
  have h := abs_totalTurn_le hR hfar hstep (n := n)
  rw [winding, abs_div, abs_of_pos (by positivity : (0 : ℝ) < 2 * Real.pi),
    div_le_iff₀ (by positivity)]
  have : (n : ℝ) * (Real.pi * b / (2 * R)) = n * b / (4 * R) * (2 * Real.pi) := by
    field_simp; ring
  linarith [h, this ▸ h]

/-- **Length demand.**  Read backwards, the budget says how big a model has to be: producing `k`
turns of wrapping around an object of exclusion radius `R` with bonds of length at most `b`
requires at least `4 k R / b` bonds.  A proposed ensemble with fewer residues than this does
not contain the threading the data report, whatever its energy function. -/
theorem length_demand_of_winding {R b k : ℝ} {p : ℕ → ℂ} {n : ℕ} (hR : 0 < R) (hb : 0 < b)
    (hfar : ∀ i, R ≤ ‖p i‖) (hstep : ∀ i, ‖p (i + 1) - p i‖ ≤ b)
    (hk : k ≤ |winding p n|) : 4 * k * R / b ≤ n := by
  have h := abs_winding_le hR hfar hstep (n := n)
  have hkb : k ≤ n * b / (4 * R) := le_trans hk h
  rw [div_le_iff₀ hb]
  rw [le_div_iff₀ (by positivity : (0 : ℝ) < 4 * R)] at hkb
  linarith

/-! ### Integrality and topological protection -/

private theorem arg_sum_coe (f : ℕ → ℂ) (hf : ∀ i, f i ≠ 0) (n : ℕ) :
    ((∑ i ∈ range n, Complex.arg (f i) : ℝ) : Real.Angle)
      = ((Complex.arg (∏ i ∈ range n, f i) : ℝ) : Real.Angle) := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_range_succ, Finset.prod_range_succ]
      have hp : (∏ i ∈ range n, f i) ≠ 0 := Finset.prod_ne_zero_iff.2 fun i _ => hf i
      rw [Real.Angle.coe_add, ih, Complex.arg_mul_coe_angle hp (hf n)]

private theorem prod_ratio_telescope (p : ℕ → ℂ) (hp : ∀ i, p i ≠ 0) (n : ℕ) :
    (∏ i ∈ range n, (p (i + 1) / p i)) = p n / p 0 := by
  induction n with
  | zero => exact (div_self (hp 0)).symm
  | succ n ih => rw [Finset.prod_range_succ, ih]; field_simp [hp n]

/-- **A closed chain winds a whole number of times.**  If the chain returns to its starting
point without ever touching the axis, its total turn is an integer multiple of `2π`. -/
theorem totalTurn_closed {p : ℕ → ℂ} {n : ℕ} (hp : ∀ i, p i ≠ 0) (hclosed : p n = p 0) :
    ∃ k : ℤ, totalTurn p n = k * (2 * Real.pi) := by
  have h1 := arg_sum_coe (fun i => p (i + 1) / p i) (fun i => div_ne_zero (hp _) (hp _)) n
  rw [prod_ratio_telescope p hp n, hclosed, div_self (hp 0)] at h1
  simp only [Complex.arg_one] at h1
  have h2 : ((totalTurn p n : ℝ) : Real.Angle) = 0 := by
    rw [totalTurn]; rw [show (fun i => turn (p i) (p (i+1))) = fun i => Complex.arg (p (i+1) / p i)
      from rfl] at *
    rw [h1]; rfl
  obtain ⟨k, hk⟩ := Real.Angle.coe_eq_zero_iff.1 h2
  exact ⟨k, by rw [← hk]; simp [zsmul_eq_mul]⟩

/-- The winding number of a closed chain is an integer. -/
theorem winding_int_of_closed {p : ℕ → ℂ} {n : ℕ} (hp : ∀ i, p i ≠ 0) (hclosed : p n = p 0) :
    ∃ k : ℤ, winding p n = k := by
  obtain ⟨k, hk⟩ := totalTurn_closed hp hclosed
  refine ⟨k, ?_⟩
  rw [winding, hk]
  field_simp

private theorem continuous_turn_of_constraints {R b : ℝ} (hR : 0 < R) (hbR : b < 2 * R)
    {za zb : ℝ → ℂ} (hza : Continuous za) (hzb : Continuous zb)
    (hfar : ∀ t, R ≤ ‖za t‖) (hfar' : ∀ t, R ≤ ‖zb t‖) (hstep : ∀ t, ‖zb t - za t‖ ≤ b) :
    Continuous fun t => turn (za t) (zb t) := by
  have hza0 : ∀ t, za t ≠ 0 := by
    intro t h; have hi := hfar t; rw [h] at hi; simp at hi; linarith
  have hzb0 : ∀ t, zb t ≠ 0 := by
    intro t h; have hi := hfar' t; rw [h] at hi; simp at hi; linarith
  have hq : Continuous fun t => zb t / za t := hzb.div hza hza0
  refine continuous_iff_continuousAt.2 fun t => ?_
  have hslit : (zb t / za t) ∈ Complex.slitPlane := by
    have hlt := abs_turn_lt_pi hR (hfar t) (hfar' t) (hstep t) hbR
    have hne : (zb t / za t) ≠ 0 := div_ne_zero (hzb0 t) (hza0 t)
    rw [Complex.mem_slitPlane_iff]
    by_contra hcon
    push_neg at hcon
    obtain ⟨hre, him⟩ := hcon
    rcases lt_or_eq_of_le hre with hre' | hre'
    · have harg : Complex.arg (zb t / za t) = Real.pi := Complex.arg_eq_pi_iff.2 ⟨hre', him⟩
      rw [turn, harg, abs_of_pos Real.pi_pos] at hlt
      exact lt_irrefl _ hlt
    · exact hne (Complex.ext (by simpa using hre') (by simpa using him))
  simp only [turn]
  exact ContinuousAt.comp (g := Complex.arg) (f := fun t => zb t / za t) (x := t)
    (Complex.continuousAt_arg hslit) hq.continuousAt

private theorem continuous_winding_of_constraints {R b : ℝ} (hR : 0 < R) (hbR : b < 2 * R)
    {n : ℕ} (P : ℝ → ℕ → ℂ) (hcont : ∀ i, Continuous fun t => P t i)
    (hfar : ∀ t i, R ≤ ‖P t i‖) (hstep : ∀ t i, ‖P t (i + 1) - P t i‖ ≤ b) :
    Continuous fun t => winding (P t) n := by
  have hsum : Continuous fun t => totalTurn (P t) n := by
    rw [show (fun t => totalTurn (P t) n)
        = fun t => ∑ i ∈ range n, turn (P t i) (P t (i + 1)) from rfl]
    exact continuous_finset_sum _ fun i _ =>
      continuous_turn_of_constraints hR hbR (hcont i) (hcont (i + 1))
        (fun t => hfar t i) (fun t => hfar t (i + 1)) (fun t => hstep t i)
  exact hsum.div_const _

/-- A continuous real function that only takes integer values is constant: the intermediate
value theorem forbids it from jumping.  This is the analytic core of topological protection. -/
private theorem eq_of_continuous_int_valued {f : ℝ → ℝ} (hf : Continuous f)
    (hint : ∀ t, ∃ k : ℤ, f t = k) (s u : ℝ) : f s = f u := by
  obtain ⟨k, hk⟩ := hint s
  obtain ⟨l, hl⟩ := hint u
  by_contra hne
  have hkl : k ≠ l := fun h => hne (by rw [hk, hl, h])
  have hgap : min (f s) (f u) + 1 ≤ max (f s) (f u) := by
    rw [hk, hl]
    rcases lt_or_gt_of_ne hkl with h | h
    · rw [min_eq_left (by exact_mod_cast h.le), max_eq_right (by exact_mod_cast h.le)]
      exact_mod_cast Int.add_one_le_of_lt h
    · rw [min_eq_right (by exact_mod_cast h.le), max_eq_left (by exact_mod_cast h.le)]
      exact_mod_cast Int.add_one_le_of_lt h
  set c := min (f s) (f u) + 1 / 2 with hc
  have hmem : c ∈ Set.uIcc (f s) (f u) := by
    rw [Set.mem_uIcc]
    rcases le_total (f s) (f u) with h | h
    · left
      rw [min_eq_left h] at hc
      rw [min_eq_left h, max_eq_right h] at hgap
      constructor <;> linarith
    · right
      rw [min_eq_right h] at hc
      rw [min_eq_right h, max_eq_left h] at hgap
      constructor <;> linarith
  obtain ⟨t, _, hval⟩ := intermediate_value_uIcc (f := f) hf.continuousOn hmem
  obtain ⟨m, hm⟩ := hint t
  rw [hm, hc, hk, hl, ← Int.cast_min] at hval
  push_cast at hval
  have h2 : (2 * m : ℤ) = 2 * (min k l) + 1 := by
    have h3 : ((2 * m : ℤ) : ℝ) = ((2 * (min k l) + 1 : ℤ) : ℝ) := by push_cast; linarith
    exact_mod_cast h3
  omega

/-- **Topological protection of threading.**  Let a closed disordered loop be deformed
continuously in time, in such a way that at every instant every residue stays at distance at
least `R` from the axis and every bond is shorter than `2 R` -- that is, the chain never breaks
and never passes through the object it is wrapped around.  Then the winding number is the same
at every instant.  The amount of threading in a conformation is not a soft, energy-tunable
quantity: within excluded volume and connectivity it cannot change at all, so an ensemble model
that samples the wrong topological sector cannot be repaired by reweighting, only by a move
that violates the constraints. -/
theorem winding_invariant_of_deformation {R b : ℝ} (hR : 0 < R) (hbR : b < 2 * R) {n : ℕ}
    (P : ℝ → ℕ → ℂ) (hcont : ∀ i, Continuous fun t => P t i)
    (hfar : ∀ t i, R ≤ ‖P t i‖) (hstep : ∀ t i, ‖P t (i + 1) - P t i‖ ≤ b)
    (hclosed : ∀ t, P t n = P t 0) (s u : ℝ) :
    winding (P s) n = winding (P u) n := by
  refine eq_of_continuous_int_valued
    (continuous_winding_of_constraints hR hbR P hcont hfar hstep) (fun t => ?_) s u
  refine winding_int_of_closed (fun i => ?_) (hclosed t)
  intro h
  have hi := hfar t i
  rw [h] at hi
  simp at hi
  linarith

/-- **The polygon bound.**  Independently of any excluded volume, a chain of `n` bonds cannot
wind more than `n / 2` times around a point it never touches: each bond sweeps less than a half
turn, because the turning angle of a step is read in `(-π, π]`.  Excluded volume sharpens this
to `n b / (4 R)` (`abs_winding_le`); the two together are the whole kinematic budget. -/
theorem abs_winding_le_half (p : ℕ → ℂ) (n : ℕ) : |winding p n| ≤ n / 2 := by
  have hpi := Real.pi_pos
  have h : |totalTurn p n| ≤ n * Real.pi := by
    calc |totalTurn p n| ≤ ∑ i ∈ range n, |turn (p i) (p (i + 1))| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i ∈ range n, Real.pi :=
          Finset.sum_le_sum fun i _ => Complex.abs_arg_le_pi _
      _ = n * Real.pi := by simp [mul_comm]
  rw [winding, abs_div, abs_of_pos (by positivity : (0 : ℝ) < 2 * Real.pi),
    div_le_iff₀ (by positivity)]
  nlinarith

/-- **Unthreading costs a violation.**  If a closed chain has different winding numbers at two
times of a continuous deformation, then at some instant some residue has entered the excluded
region or some bond has been stretched past `2 R`.  This is the contrapositive of
`winding_invariant_of_deformation`, and it is the statement a simulation has to answer for: a
trajectory that changes topology has passed a chain through an object, or through itself. -/
theorem unthreading_requires_violation {R b : ℝ} (hR : 0 < R) (hbR : b < 2 * R) {n : ℕ}
    (P : ℝ → ℕ → ℂ) (hcont : ∀ i, Continuous fun t => P t i)
    (hclosed : ∀ t, P t n = P t 0) {s u : ℝ} (hne : winding (P s) n ≠ winding (P u) n) :
    ∃ t i, ‖P t i‖ < R ∨ b < ‖P t (i + 1) - P t i‖ := by
  by_contra hcon
  push_neg at hcon
  exact hne (winding_invariant_of_deformation hR hbR P hcont
    (fun t i => (hcon t i).1) (fun t i => (hcon t i).2) hclosed s u)

end RequestProject.Winding
