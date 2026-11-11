/-
# Threading of a disordered region: what an ensemble can and cannot contain

`RequestProject.Winding` established the geometry of a single conformation wrapped around an
axis: the winding budget `|winding| ≤ n b / (4 R)`, and the topological protection of the
winding number under any deformation that respects connectivity and excluded volume.  This file
carries that geometry over to *ensembles* and to *data*, in the same style as the rest of the
project: what a measurement of threading forces a model to contain, and what it can refute.

* `meanAbsWinding`, `threadedFraction` -- the two ensemble observables: the mean amount of
  wrapping, and the population of the threaded sector at level `k`.
* `meanAbsWinding_le_budget` and `threadedFraction_le_budget` -- the budget survives averaging,
  and, through a Markov step, caps the *population* that can be threaded: at level `k` no
  ensemble of `n`-residue chains with bond length `b` and exclusion radius `R` puts more than
  `n b / (4 R k)` of its weight in the threaded sector.  A measured threaded population above
  that ceiling is refuted without a model (`no_ensemble_of_excess_threading`).
* `length_demand_of_threaded_fraction` -- any nonzero threaded population forces
  `n ≥ 4 k R / b` residues: threading data set a hard lower bound on the length of the
  disordered region involved.
* `reweighting_cannot_create_threading` -- **the sharpest modelling statement here.**  If the
  conformations a model samples all sit in the unthreaded sector, then *no* reweighting of that
  model -- no maximum-entropy correction, no Bayesian ensemble refinement, no force-field
  rescaling -- produces any threading at all: the discrepancy with a measured threaded
  population is exactly that population, for every weight vector.  Threading is a property of
  the support of the ensemble, and by `Winding.winding_invariant_of_deformation` the support
  cannot be moved into the threaded sector by any dynamics that respects excluded volume.  A
  model must be *built* in the right topological sector.
* `sector_population_invariant` -- the ensemble form of topological protection: sector
  populations are constant under admissible dynamics.  A simulation with soft-core potentials,
  which lets chains pass through each other, silently equilibrates a quantity that in reality is
  frozen.
* `wrap` and `wrap_winding`, `wrap_admissible` -- an explicit realisation: a regular helical
  wrap with `n` bonds realises exactly `k` turns whenever `n ≥ 2π k R / b`.  With the budget
  this brackets the true residue demand for `k` turns between `4 k R / b` and `2π k R / b`
  (`wrap_threshold_window`): the design law is tight to within a factor `π / 2`.
* `two_state_meanWinding` -- and a warning in the same key as the rest of the project: any mean
  amount of threading between `0` and `k` is reproduced by a two-state mixture of one fully
  wrapped and one unwrapped conformation.  A measured mean reports a population, not a
  structure.
-/
import Mathlib
import RequestProject.Winding

namespace RequestProject.Threading

open Finset Complex RequestProject.Winding

open scoped Classical

/-- An ensemble of `m` conformations of the projected chain, with weights `w`. -/
noncomputable def meanWinding {m : ℕ} (w : Fin m → ℝ) (X : Fin m → ℕ → ℂ) (n : ℕ) : ℝ :=
  ∑ a, w a * winding (X a) n

/-- The mean amount of wrapping in an ensemble, irrespective of handedness. -/
noncomputable def meanAbsWinding {m : ℕ} (w : Fin m → ℝ) (X : Fin m → ℕ → ℂ) (n : ℕ) : ℝ :=
  ∑ a, w a * |winding (X a) n|

/-- The population of the ensemble that is threaded at level `k`: the total weight of the
conformations that wind at least `k` times around the axis. -/
noncomputable def threadedFraction {m : ℕ} (w : Fin m → ℝ) (X : Fin m → ℕ → ℂ) (n : ℕ)
    (k : ℝ) : ℝ :=
  ∑ a ∈ Finset.univ.filter (fun a => k ≤ |winding (X a) n|), w a

variable {m : ℕ} {w : Fin m → ℝ} {X : Fin m → ℕ → ℂ} {R b : ℝ} {n : ℕ}

/-- An ensemble is *admissible* for exclusion radius `R` and bond length `b` when every
conformation keeps every residue outside the excluded region and every bond within `b`. -/
def Admissible (X : Fin m → ℕ → ℂ) (R b : ℝ) : Prop :=
  (∀ a i, R ≤ ‖X a i‖) ∧ (∀ a i, ‖X a (i + 1) - X a i‖ ≤ b)

/-- **The winding budget survives ensemble averaging.**  Whatever the ensemble, its mean amount
of wrapping obeys the single-conformation cap. -/
theorem meanAbsWinding_le_budget (hR : 0 < R) (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a = 1)
    (hX : Admissible X R b) : meanAbsWinding w X n ≤ n * b / (4 * R) := by
  obtain ⟨hfar, hstep⟩ := hX
  calc meanAbsWinding w X n ≤ ∑ _a : Fin m, w _a * (n * b / (4 * R)) :=
        Finset.sum_le_sum fun a _ =>
          mul_le_mul_of_nonneg_left (abs_winding_le hR (hfar a) (hstep a)) (hw a)
    _ = n * b / (4 * R) := by rw [← Finset.sum_mul, hsum, one_mul]

/-- Markov's inequality for threading: the threaded population at level `k` is at most the mean
amount of wrapping divided by `k`. -/
theorem threadedFraction_le_meanAbsWinding (hw : ∀ a, 0 ≤ w a) {k : ℝ} :
    k * threadedFraction w X n k ≤ meanAbsWinding w X n := by
  have hsub : (Finset.univ.filter (fun a => k ≤ |winding (X a) n|)) ⊆ Finset.univ :=
    Finset.filter_subset _ _
  calc k * threadedFraction w X n k
      = ∑ a ∈ Finset.univ.filter (fun a => k ≤ |winding (X a) n|), w a * k := by
        rw [threadedFraction, Finset.mul_sum]
        exact Finset.sum_congr rfl fun a _ => by ring
    _ ≤ ∑ a ∈ Finset.univ.filter (fun a => k ≤ |winding (X a) n|), w a * |winding (X a) n| := by
        refine Finset.sum_le_sum fun a ha => ?_
        exact mul_le_mul_of_nonneg_left (Finset.mem_filter.1 ha).2 (hw a)
    _ ≤ ∑ a : Fin m, w a * |winding (X a) n| :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub
          (fun a _ _ => mul_nonneg (hw a) (abs_nonneg _))
    _ = meanAbsWinding w X n := rfl

/-- **Population ceiling for threading.**  No ensemble of `n`-residue chains with bond length at
most `b` and exclusion radius `R` can have more than `n b / (4 R k)` of its weight wound `k` or
more times around the axis. -/
theorem threadedFraction_le_budget (hR : 0 < R) (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a = 1)
    (hX : Admissible X R b) {k : ℝ} (hk : 0 < k) :
    threadedFraction w X n k ≤ n * b / (4 * R * k) := by
  have h1 := threadedFraction_le_meanAbsWinding (X := X) (w := w) (n := n) (k := k) hw
  have h2 := meanAbsWinding_le_budget (X := X) (w := w) (n := n) hR hw hsum hX
  rw [le_div_iff₀ (by positivity : (0 : ℝ) < 4 * R * k)]
  have h3 : k * threadedFraction w X n k ≤ n * b / (4 * R) := le_trans h1 h2
  have h4 : (n : ℝ) * b / (4 * R) * (4 * R) = n * b := by field_simp
  nlinarith [mul_le_mul_of_nonneg_left h3 (le_of_lt (by positivity : (0 : ℝ) < 4 * R))]

/-- **Threading data set a length demand.**  If even one conformation of the ensemble is wound
`k` times, the region must have at least `4 k R / b` bonds. -/
theorem length_demand_of_conformation (hR : 0 < R) (hb : 0 < b) (hX : Admissible X R b)
    {k : ℝ} {a : Fin m} (hka : k ≤ |winding (X a) n|) : 4 * k * R / b ≤ n :=
  length_demand_of_winding hR hb (hX.1 a) (hX.2 a) hka

/-- The same demand from a population: a nonzero threaded population at level `k` forces at
least `4 k R / b` bonds. -/
theorem length_demand_of_threaded_fraction (hR : 0 < R) (hb : 0 < b)
    (hX : Admissible X R b) {k : ℝ} (hpos : 0 < threadedFraction w X n k) :
    4 * k * R / b ≤ n := by
  by_contra hcon
  push_neg at hcon
  have hempty : (Finset.univ.filter (fun a => k ≤ |winding (X a) n|)) = ∅ := by
    refine Finset.eq_empty_of_forall_notMem fun a ha => ?_
    have hka := (Finset.mem_filter.1 ha).2
    exact absurd (length_demand_of_conformation (X := X) hR hb hX hka) (not_le.2 hcon)
  rw [threadedFraction, hempty, Finset.sum_empty] at hpos
  exact lt_irrefl _ hpos

/-- **Model-free refutation of a threading measurement.**  If the reported threaded population
`phi` at level `k`, less its error bar `e`, is above the ceiling that excluded volume and
connectivity allow, then no ensemble of admissible conformations reproduces the measurement
within its error bars.  Either the region is longer, or the exclusion radius smaller, than
reported. -/
theorem no_ensemble_of_excess_threading (hR : 0 < R) (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a = 1)
    (hX : Admissible X R b) {k phi e : ℝ} (hk : 0 < k)
    (hexcess : (n : ℝ) * b / (4 * R * k) < phi - e) :
    e < |threadedFraction w X n k - phi| := by
  have h := threadedFraction_le_budget (X := X) (w := w) (n := n) hR hw hsum hX hk
  have hlt : threadedFraction w X n k - phi < -e := by linarith
  calc e = -(-e) := by ring
    _ < -(threadedFraction w X n k - phi) := by linarith
    _ ≤ |threadedFraction w X n k - phi| := neg_le_abs _

/-- **Reweighting cannot create threading.**  If every conformation the model samples is
unthreaded, then for every weight vector whatsoever the model's threaded population at any
positive level is exactly zero.  Ensemble refinement adjusts weights, so it can never repair a
model that was built in the wrong topological sector; and, by
`Winding.winding_invariant_of_deformation`, no dynamics respecting excluded volume moves a
conformation into the right one either. -/
theorem reweighting_cannot_create_threading {k : ℝ} (hk : 0 < k)
    (hzero : ∀ a, winding (X a) n = 0) : threadedFraction w X n k = 0 := by
  have hempty : (Finset.univ.filter (fun a => k ≤ |winding (X a) n|)) = ∅ := by
    refine Finset.eq_empty_of_forall_notMem fun a ha => ?_
    have hka := (Finset.mem_filter.1 ha).2
    rw [hzero a, abs_zero] at hka
    exact absurd hka (not_le.2 hk)
  rw [threadedFraction, hempty, Finset.sum_empty]

/-- The gap that no reweighting closes: against a measured threaded population `phi`, an
ensemble supported in the unthreaded sector is off by exactly `phi`, for every weight vector. -/
theorem unthreaded_model_gap {k phi : ℝ} (hk : 0 < k) (hzero : ∀ a, winding (X a) n = 0) :
    |threadedFraction w X n k - phi| = |phi| := by
  rw [reweighting_cannot_create_threading (w := w) hk hzero, zero_sub, abs_neg]

/-- **Ensemble topological protection.**  If every conformation of the ensemble is deformed
continuously in time while keeping every residue outside the exclusion radius and every bond
shorter than `2 R`, then the population of every topological sector is constant.  Sector
populations are not thermodynamic observables of the force field; a model that relaxes them --
as any soft-core potential permitting chain crossing does -- is describing a different physical
system. -/
theorem sector_population_invariant {R b : ℝ} (hR : 0 < R) (hbR : b < 2 * R)
    (P : ℝ → Fin m → ℕ → ℂ) (hcont : ∀ a i, Continuous fun t => P t a i)
    (hfar : ∀ t a i, R ≤ ‖P t a i‖) (hstep : ∀ t a i, ‖P t a (i + 1) - P t a i‖ ≤ b)
    (hclosed : ∀ t a, P t a n = P t a 0) (k : ℝ) (s u : ℝ) :
    threadedFraction w (P s) n k = threadedFraction w (P u) n k := by
  have hpt : ∀ a, winding (P s a) n = winding (P u a) n := fun a =>
    winding_invariant_of_deformation hR hbR (fun t => P t a) (fun i => hcont a i)
      (fun t i => hfar t a i) (fun t i => hstep t a i) (fun t => hclosed t a) s u
  unfold threadedFraction
  refine Finset.sum_congr (Finset.filter_congr fun a _ => ?_) fun _ _ => rfl
  rw [hpt a]

/-! ### An explicit wrap, and the tightness of the design law -/

/-- A regular wrap: `n` equal bonds on a circle of radius `R` about the axis, each turning the
chain by `theta`. -/
noncomputable def wrap (R theta : ℝ) : ℕ → ℂ :=
  fun i => (R : ℂ) * Complex.exp ((theta * i : ℝ) * Complex.I)

theorem wrap_norm {R theta : ℝ} (hR : 0 < R) (i : ℕ) : ‖wrap R theta i‖ = R := by
  rw [wrap, norm_mul, Complex.norm_exp_ofReal_mul_I, Complex.norm_real]
  simp [abs_of_pos hR]

theorem wrap_ratio {R theta : ℝ} (hR : 0 < R) (i : ℕ) :
    wrap R theta (i + 1) / wrap R theta i = Complex.exp ((theta : ℝ) * Complex.I) := by
  have hR0 : (R : ℂ) ≠ 0 := by exact_mod_cast hR.ne'
  simp only [wrap]
  push_cast
  rw [mul_div_mul_left _ _ hR0, ← Complex.exp_sub]
  ring_nf

private theorem norm_exp_sub_one_le (theta : ℝ) :
    ‖Complex.exp ((theta : ℝ) * Complex.I) - 1‖ ≤ |theta| := by
  have hsq : ‖Complex.exp ((theta : ℝ) * Complex.I) - 1‖ ^ 2 = 2 - 2 * Real.cos theta := by
    rw [Complex.exp_mul_I, Complex.sq_norm, Complex.normSq_apply]
    simp [Complex.cos_ofReal_re, Complex.sin_ofReal_re]
    nlinarith [Real.sin_sq_add_cos_sq theta]
  have hhalf : Real.cos theta = 1 - 2 * Real.sin (theta / 2) ^ 2 := by
    have h := Real.cos_two_mul' (theta / 2)
    have h2 := Real.sin_sq_add_cos_sq (theta / 2)
    have h3 : 2 * (theta / 2) = theta := by ring
    rw [h3] at h; nlinarith
  have hs : |Real.sin (theta / 2)| ≤ |theta / 2| := Real.abs_sin_le_abs
  have hle : ‖Complex.exp ((theta : ℝ) * Complex.I) - 1‖ ^ 2 ≤ theta ^ 2 := by
    rw [hsq, hhalf]
    nlinarith [abs_nonneg (Real.sin (theta / 2)), sq_abs (Real.sin (theta / 2)),
      abs_nonneg (theta / 2), sq_abs (theta / 2)]
  nlinarith [norm_nonneg (Complex.exp ((theta : ℝ) * Complex.I) - 1), abs_nonneg theta,
    sq_abs theta]

theorem wrap_step_le {R theta : ℝ} (hR : 0 < R) (i : ℕ) :
    ‖wrap R theta (i + 1) - wrap R theta i‖ ≤ R * |theta| := by
  have h : wrap R theta (i + 1) - wrap R theta i
      = wrap R theta i * (Complex.exp ((theta : ℝ) * Complex.I) - 1) := by
    simp only [wrap]
    have hadd : ((theta * (i + 1 : ℕ) : ℝ) : ℂ) * Complex.I
        = ((theta * i : ℝ) : ℂ) * Complex.I + ((theta : ℝ) : ℂ) * Complex.I := by
      push_cast; ring
    rw [hadd, Complex.exp_add]
    ring
  rw [h, norm_mul, wrap_norm hR]
  exact mul_le_mul_of_nonneg_left (norm_exp_sub_one_le theta) hR.le

theorem wrap_turn {R theta : ℝ} (hR : 0 < R) (hlt : |theta| < Real.pi) (i : ℕ) :
    turn (wrap R theta i) (wrap R theta (i + 1)) = theta := by
  rw [turn, wrap_ratio hR, Complex.exp_mul_I]
  refine Complex.arg_cos_add_sin_mul_I ?_
  constructor
  · have := abs_lt.1 hlt; linarith [this.1]
  · exact le_of_lt (abs_lt.1 hlt).2

theorem wrap_totalTurn {R theta : ℝ} (hR : 0 < R) (hlt : |theta| < Real.pi) (n : ℕ) :
    totalTurn (wrap R theta) n = n * theta := by
  rw [totalTurn]
  rw [Finset.sum_congr rfl fun i _ => wrap_turn hR hlt i]
  simp [mul_comm]

theorem wrap_winding {R theta : ℝ} (hR : 0 < R) (hlt : |theta| < Real.pi) (n : ℕ) :
    winding (wrap R theta) n = n * theta / (2 * Real.pi) := by
  rw [winding, wrap_totalTurn hR hlt]

/-- The regular wrap closes up after `n` bonds when the turn per bond is `2π k / n`. -/
theorem wrap_closed {R : ℝ} {k n : ℕ} (hn : 0 < n) :
    wrap R (2 * Real.pi * k / n) n = wrap R (2 * Real.pi * k / n) 0 := by
  have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hn.ne'
  have hr : 2 * Real.pi * (k : ℝ) / n * n = (k : ℝ) * (2 * Real.pi) := by
    rw [div_mul_cancel₀ _ hn0]; ring
  simp only [wrap]
  rw [hr]
  push_cast
  rw [show ((k : ℂ) * (2 * (Real.pi : ℂ))) * Complex.I
      = ((k : ℤ) : ℂ) * (2 * (Real.pi : ℂ) * Complex.I) by push_cast; ring]
  rw [Complex.exp_int_mul_two_pi_mul_I]
  simp

/-- **Achievability.**  With `n ≥ 2π k R / b` bonds of length at most `b` on a circle of radius
`R`, a chain wraps the axis exactly `k` times without ever entering the excluded region.  Read
with `Winding.length_demand_of_winding`, the residue demand for `k` turns is bracketed:
at least `4 k R / b`, and at most `2π k R / b`. -/
theorem wrap_admissible {R b : ℝ} {k n : ℕ} (hR : 0 < R) (hb : 0 < b) (hbR : b < 2 * R)
    (hk : 0 < k) (hn : 2 * Real.pi * k * R / b ≤ n) :
    (∀ i, R ≤ ‖wrap R (2 * Real.pi * k / n) i‖) ∧
      (∀ i, ‖wrap R (2 * Real.pi * k / n) (i + 1) - wrap R (2 * Real.pi * k / n) i‖ ≤ b) ∧
      winding (wrap R (2 * Real.pi * k / n)) n = k := by
  have hpi := Real.pi_pos
  have hkR : (0 : ℝ) < k := by exact_mod_cast hk
  have hnpos : (0 : ℝ) < n := by
    have : 0 < 2 * Real.pi * k * R / b := by positivity
    linarith
  have hn0 : (n : ℝ) ≠ 0 := ne_of_gt hnpos
  set theta : ℝ := 2 * Real.pi * k / n with htheta
  have hthetapos : 0 < theta := by rw [htheta]; positivity
  -- `theta ≤ b / R` from the assumed number of bonds
  have hthetaR : theta * R ≤ b := by
    rw [htheta, div_mul_eq_mul_div, div_le_iff₀ hnpos]
    rw [div_le_iff₀ hb] at hn
    linarith
  have hthetalt : |theta| < Real.pi := by
    rw [abs_of_pos hthetapos]
    have h1 : theta * R ≤ b := hthetaR
    have h2 : theta * R < 2 * R := lt_of_le_of_lt h1 hbR
    nlinarith [Real.pi_gt_three]
  refine ⟨fun i => le_of_eq (wrap_norm hR i).symm, fun i => ?_, ?_⟩
  · refine le_trans (wrap_step_le hR i) ?_
    rw [abs_of_pos hthetapos]
    nlinarith
  · rw [wrap_winding hR hthetalt, htheta]
    field_simp

/-- **The design law is tight to within a factor `π / 2`.**  For `k` turns at exclusion radius
`R` with bonds of length `b`: no chain of `n` bonds achieves `k` turns unless `n ≥ 4 k R / b`,
and as soon as `n ≥ 2π k R / b` an explicit chain of `n` bonds achieves exactly `k` turns. -/
theorem wrap_threshold_window {R b : ℝ} {k n : ℕ} (hR : 0 < R) (hb : 0 < b) (hbR : b < 2 * R)
    (hk : 0 < k) (hn : 2 * Real.pi * k * R / b ≤ n) :
    (∀ p : ℕ → ℂ, (∀ i, R ≤ ‖p i‖) → (∀ i, ‖p (i + 1) - p i‖ ≤ b) →
        (k : ℝ) ≤ |winding p n| → 4 * k * R / b ≤ n) ∧
      ∃ p : ℕ → ℂ, (∀ i, R ≤ ‖p i‖) ∧ (∀ i, ‖p (i + 1) - p i‖ ≤ b) ∧ winding p n = k := by
  refine ⟨fun p hfar hstep hge => length_demand_of_winding hR hb hfar hstep hge, ?_⟩
  obtain ⟨hfar, hstep, hwind⟩ := wrap_admissible hR hb hbR hk hn
  exact ⟨wrap R (2 * Real.pi * k / n), hfar, hstep, hwind⟩

/-- **A measured mean tells you a population, not a structure.**  Mixing a fully wrapped
conformation with an unwrapped one reproduces every mean winding between `0` and the wrapped
value, with no change to any single structure in the ensemble. -/
theorem two_state_meanWinding {P Q : ℕ → ℂ} {n : ℕ} {t : ℝ} :
    meanWinding ![1 - t, t] ![Q, P] n = (1 - t) * winding Q n + t * winding P n := by
  simp [meanWinding, Fin.sum_univ_two]

/-- Every mean threading level between `0` and `k` is realised by such a mixture. -/
theorem meanWinding_realises {R b : ℝ} {k n : ℕ} (hR : 0 < R) (hb : 0 < b) (hbR : b < 2 * R)
    (hk : 0 < k) (hn : 2 * Real.pi * k * R / b ≤ n) {mu : ℝ} (hmu : 0 ≤ mu) (hmuk : mu ≤ k) :
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
      meanWinding ![1 - t, t] ![wrap R 0, wrap R (2 * Real.pi * k / n)] n = mu := by
  have hkR : (0 : ℝ) < k := by exact_mod_cast hk
  obtain ⟨_, _, hwind⟩ := wrap_admissible hR hb hbR hk hn
  refine ⟨mu / k, by positivity, by rw [div_le_one hkR]; exact hmuk, ?_⟩
  rw [two_state_meanWinding, hwind]
  have h0 : winding (wrap R 0) n = 0 := by
    rw [wrap_winding hR (by simpa using Real.pi_pos)]
    simp
  rw [h0]
  field_simp
  ring

end RequestProject.Threading
