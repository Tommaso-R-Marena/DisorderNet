import Mathlib

/-!
# Part CXLIV — Chaperones: an ATP cycle is not a stronger binder

A model that is meant to capture an intrinsically disordered region *inside a cell* has to
face the fact that the cell is not at equilibrium.  Hsp70, Hsp90, trigger factor and the
small heat-shock proteins all hold, release and remodel disordered clients while hydrolysing
ATP, and the whole point of the hydrolysis is that it buys something that binding alone
cannot buy.  This file makes that statement exact.

The client is described by the smallest kinetic scheme in which the question can even be
asked: a three-state cycle

* state `0` = free, disordered client (`U`),
* state `1` = chaperone-bound client (`C`),
* state `2` = folded / compact client (`F`),

with six positive rate constants, `f0 : U → C`, `f1 : C → F`, `f2 : F → U` running one way
round the cycle and `b0 : U → F`, `b1 : C → U`, `b2 : F → C` running the other way.  We

* solve the stationary state exactly (Kirchhoff's matrix–tree weights, `Theorem
  stationary_balance`),
* show that the stationary probability current is the *same* across all three edges and
  equals `(f0 f1 f2 - b0 b1 b2)/Z` (`current_edge01`, `current_edge12`, `current_edge20`),
  so that the Kolmogorov cycle condition `f0 f1 f2 = b0 b1 b2` is exactly the condition for
  equilibrium (`current_eq_zero_iff`),
* prove the **no-free-lunch theorem for equilibrium chaperones**: if the cycle condition
  holds, the folded : disordered ratio is `b0 / f2`, i.e. the ratio the client would have with
  no chaperone at all, *no matter how tightly or how fast the chaperone binds*
  (`equilibrium_ratio_chaperone_independent`),
* prove that a driven cycle has no such ceiling: the folded : disordered ratio can be pushed
  past any target by increasing the drive (`driven_ratio_unbounded`), while a driven cycle
  that is run backwards depletes the folded state past any target (`driven_ratio_to_zero`),
* prove that the entropy production `J · log A` is nonnegative and vanishes exactly at
  equilibrium (`entropy_production_nonneg`, `entropy_production_eq_zero_iff`).

The design consequence is the one every quantitative model of a disordered proteome has to
respect: a chaperone term added to a free-energy function — any term at all — is a term that
cannot move the folded fraction away from `b0/f2`.  Capturing chaperone action requires the
*rates*, and requires the cycle to be driven.
-/

noncomputable section

namespace RequestProject.ChaperoneCycle

open Real

/-- Six positive rate constants of the three-state client cycle. -/
structure Rates where
  /-- rate of the transition `U → C` (chaperone capture). -/
  f0 : ℝ
  /-- rate of the transition `C → F` (chaperone-assisted release into the folded state). -/
  f1 : ℝ
  /-- rate of the transition `F → U` (unfolding). -/
  f2 : ℝ
  /-- rate of the transition `U → F` (spontaneous folding). -/
  b0 : ℝ
  /-- rate of the transition `C → U` (unproductive release). -/
  b1 : ℝ
  /-- rate of the transition `F → C` (capture of the folded client). -/
  b2 : ℝ
  f0_pos : 0 < f0
  f1_pos : 0 < f1
  f2_pos : 0 < f2
  b0_pos : 0 < b0
  b1_pos : 0 < b1
  b2_pos : 0 < b2

namespace Rates

variable (r : Rates)

/-- Unnormalised stationary weight of the free disordered state `U`. -/
def w0 : ℝ := r.b1 * r.b2 + r.b1 * r.f2 + r.f1 * r.f2

/-- Unnormalised stationary weight of the chaperone-bound state `C`. -/
def w1 : ℝ := r.b2 * r.b0 + r.b2 * r.f0 + r.f2 * r.f0

/-- Unnormalised stationary weight of the folded state `F`. -/
def w2 : ℝ := r.b0 * r.b1 + r.b0 * r.f1 + r.f0 * r.f1

/-- The partition sum of the stationary weights. -/
def Z : ℝ := r.w0 + r.w1 + r.w2

lemma w0_pos : 0 < r.w0 := by
  have := r.b1_pos; have := r.b2_pos; have := r.f1_pos; have := r.f2_pos
  unfold w0; positivity

lemma w1_pos : 0 < r.w1 := by
  have := r.b0_pos; have := r.b2_pos; have := r.f0_pos; have := r.f2_pos
  unfold w1; positivity

lemma w2_pos : 0 < r.w2 := by
  have := r.b0_pos; have := r.b1_pos; have := r.f0_pos; have := r.f1_pos
  unfold w2; positivity

lemma Z_pos : 0 < r.Z := by
  have := r.w0_pos; have := r.w1_pos; have := r.w2_pos
  unfold Z; linarith

/-- Stationary probability of state `i`. -/
def prob0 : ℝ := r.w0 / r.Z
/-- Stationary probability of the chaperone-bound state. -/
def prob1 : ℝ := r.w1 / r.Z
/-- Stationary probability of the folded state. -/
def prob2 : ℝ := r.w2 / r.Z

lemma prob_sum_one : r.prob0 + r.prob1 + r.prob2 = 1 := by
  unfold prob0 prob1 prob2
  rw [← add_div, ← add_div]
  exact div_self r.Z_pos.ne'

lemma prob0_pos : 0 < r.prob0 := div_pos r.w0_pos r.Z_pos
lemma prob1_pos : 0 < r.prob1 := div_pos r.w1_pos r.Z_pos
lemma prob2_pos : 0 < r.prob2 := div_pos r.w2_pos r.Z_pos

/-! ### The stationary state -/

/-- **Stationarity.** The weights `w0, w1, w2` solve the master equation: for each state the
total inflow equals the total outflow. -/
theorem stationary_balance :
    r.b1 * r.w1 + r.f2 * r.w2 = (r.f0 + r.b0) * r.w0 ∧
    r.f0 * r.w0 + r.b2 * r.w2 = (r.f1 + r.b1) * r.w1 ∧
    r.f1 * r.w1 + r.b0 * r.w0 = (r.f2 + r.b2) * r.w2 := by
  refine ⟨?_, ?_, ?_⟩ <;> unfold w0 w1 w2 <;> ring

/-- The same statement for the normalised probabilities. -/
theorem stationary_balance_prob :
    r.b1 * r.prob1 + r.f2 * r.prob2 = (r.f0 + r.b0) * r.prob0 ∧
    r.f0 * r.prob0 + r.b2 * r.prob2 = (r.f1 + r.b1) * r.prob1 ∧
    r.f1 * r.prob1 + r.b0 * r.prob0 = (r.f2 + r.b2) * r.prob2 := by
  obtain ⟨h0, h1, h2⟩ := r.stationary_balance
  have hZ : r.Z ≠ 0 := r.Z_pos.ne'
  unfold prob0 prob1 prob2
  refine ⟨?_, ?_, ?_⟩ <;> field_simp <;> linarith

/-- The stationary folded : disordered ratio in terms of the unnormalised weights. -/
lemma prob_ratio : r.prob2 / r.prob0 = r.w2 / r.w0 := by
  have hZ : r.Z ≠ 0 := r.Z_pos.ne'
  have hw0 : r.w0 ≠ 0 := r.w0_pos.ne'
  unfold prob0 prob2
  field_simp

/-! ### The cycle current -/

/-- The thermodynamic drive of the cycle: the ratio of the two directed rate products.
`A = 1` is the Kolmogorov cycle condition; `log A` is the free energy dissipated per turn,
in units of `k_B T`. -/
def affinity : ℝ := (r.f0 * r.f1 * r.f2) / (r.b0 * r.b1 * r.b2)

/-- The stationary probability current around the cycle. -/
def current : ℝ := (r.f0 * r.f1 * r.f2 - r.b0 * r.b1 * r.b2) / r.Z

/-- **Kirchhoff's current law.** The net flux across the edge `U ↔ C` is the cycle
current. -/
theorem current_edge01 : r.f0 * r.prob0 - r.b1 * r.prob1 = r.current := by
  unfold prob0 prob1 current w0 w1
  field_simp
  ring

/-- The net flux across the edge `C ↔ F` is the same cycle current. -/
theorem current_edge12 : r.f1 * r.prob1 - r.b2 * r.prob2 = r.current := by
  unfold prob1 prob2 current w1 w2
  field_simp
  ring

/-- The net flux across the edge `F ↔ U` is the same cycle current. -/
theorem current_edge20 : r.f2 * r.prob2 - r.b0 * r.prob0 = r.current := by
  unfold prob2 prob0 current w2 w0
  field_simp
  ring

/-- **The Kolmogorov criterion.** The steady state carries no current exactly when the two
directed rate products around the cycle agree. -/
theorem current_eq_zero_iff :
    r.current = 0 ↔ r.f0 * r.f1 * r.f2 = r.b0 * r.b1 * r.b2 := by
  unfold current
  rw [div_eq_zero_iff]
  constructor
  · rintro (h | h)
    · linarith
    · exact absurd h r.Z_pos.ne'
  · intro h; exact Or.inl (by linarith)

/-- Equilibrium is the same thing as unit affinity. -/
theorem affinity_eq_one_iff :
    r.affinity = 1 ↔ r.f0 * r.f1 * r.f2 = r.b0 * r.b1 * r.b2 := by
  have hb : 0 < r.b0 * r.b1 * r.b2 := by
    have := r.b0_pos; have := r.b1_pos; have := r.b2_pos; positivity
  unfold affinity
  rw [div_eq_one_iff_eq hb.ne']

/-! ### The no-free-lunch theorem for equilibrium chaperones -/

/-- **An equilibrium chaperone cannot change the folded fraction.**  If the cycle condition
holds, the stationary folded : disordered ratio equals `b0 / f2`, the ratio set by the
spontaneous folding and unfolding rates alone.  The chaperone rates `f0, f1, b1, b2` have
dropped out entirely. -/
theorem equilibrium_ratio_chaperone_independent
    (h : r.f0 * r.f1 * r.f2 = r.b0 * r.b1 * r.b2) :
    r.prob2 / r.prob0 = r.b0 / r.f2 := by
  have hkey : r.b0 * r.w0 = r.f2 * r.w2 := by
    unfold w0 w2; nlinarith [h]
  rw [r.prob_ratio, div_eq_div_iff r.w0_pos.ne' r.f2_pos.ne']
  linarith [hkey]

/-- Restated as detailed balance across the direct folding edge. -/
theorem detailed_balance_of_cycle_condition
    (h : r.f0 * r.f1 * r.f2 = r.b0 * r.b1 * r.b2) :
    r.b0 * r.prob0 = r.f2 * r.prob2 ∧
    r.f0 * r.prob0 = r.b1 * r.prob1 ∧
    r.f1 * r.prob1 = r.b2 * r.prob2 := by
  have hz : r.current = 0 := (r.current_eq_zero_iff).2 h
  have h1 := r.current_edge01
  have h2 := r.current_edge12
  have h3 := r.current_edge20
  rw [hz] at h1 h2 h3
  exact ⟨by linarith, by linarith, by linarith⟩

/-- Two chaperone cycles that agree on the spontaneous folding and unfolding rates and both
satisfy the cycle condition have the same folded : disordered ratio, however different their
chaperone arms are. -/
theorem equilibrium_ratio_unique (r' : Rates)
    (h : r.f0 * r.f1 * r.f2 = r.b0 * r.b1 * r.b2)
    (h' : r'.f0 * r'.f1 * r'.f2 = r'.b0 * r'.b1 * r'.b2)
    (hb : r'.b0 = r.b0) (hf : r'.f2 = r.f2) :
    r'.prob2 / r'.prob0 = r.prob2 / r.prob0 := by
  rw [r.equilibrium_ratio_chaperone_independent h,
    r'.equilibrium_ratio_chaperone_independent h', hb, hf]

/-! ### A driven cycle has no ceiling -/

/-- Replace the capture rate `f0` by `t`, keeping every other rate. -/
def withCapture (t : ℝ) (ht : 0 < t) : Rates :=
  { r with f0 := t, f0_pos := ht }

lemma withCapture_ratio (t : ℝ) (ht : 0 < t) :
    (r.withCapture t ht).prob2 / (r.withCapture t ht).prob0
      = (r.b0 * r.b1 + r.b0 * r.f1 + t * r.f1) /
        (r.b1 * r.b2 + r.b1 * r.f2 + r.f1 * r.f2) := by
  rw [Rates.prob_ratio]
  rfl

/-- **A driven cycle can hold the client folded arbitrarily far past its equilibrium
ratio.**  For any target `R`, increasing the chaperone capture rate alone drives the
stationary folded : disordered ratio past `R`. -/
theorem driven_ratio_unbounded (R : ℝ) :
    ∃ t : ℝ, ∃ ht : 0 < t, R < (r.withCapture t ht).prob2 / (r.withCapture t ht).prob0 := by
  set D : ℝ := r.b1 * r.b2 + r.b1 * r.f2 + r.f1 * r.f2 with hD
  have hDpos : 0 < D := r.w0_pos
  have hf1 : 0 < r.f1 := r.f1_pos
  refine ⟨(|R| * D + 1) / r.f1, by positivity, ?_⟩
  rw [r.withCapture_ratio, ← hD]
  rw [lt_div_iff₀ hDpos]
  have hnum : (|R| * D + 1) / r.f1 * r.f1 = |R| * D + 1 := by field_simp
  have hpos : 0 < r.b0 * r.b1 + r.b0 * r.f1 := by
    have := r.b0_pos; have := r.b1_pos; positivity
  have hRle : R ≤ |R| := le_abs_self R
  nlinarith [hnum, hpos, hDpos, mul_le_mul_of_nonneg_right hRle hDpos.le]

/-- Replace the rate `b2` at which the chaperone recaptures the folded client by `t`,
keeping every other rate. -/
def withRecapture (t : ℝ) (ht : 0 < t) : Rates :=
  { r with b2 := t, b2_pos := ht }

lemma withRecapture_ratio (t : ℝ) (ht : 0 < t) :
    (r.withRecapture t ht).prob2 / (r.withRecapture t ht).prob0
      = (r.b0 * r.b1 + r.b0 * r.f1 + r.f0 * r.f1) /
        (r.b1 * t + r.b1 * r.f2 + r.f1 * r.f2) := by
  rw [Rates.prob_ratio]
  rfl

/-- **Running the same cycle the other way empties the folded state.**  For any positive
target `eps`, increasing the rate at which the chaperone recaptures the folded client alone
drives the stationary folded : disordered ratio below `eps`: an ATP-driven machine can
unfold a client that would fold on its own. -/
theorem driven_ratio_to_zero {eps : ℝ} (heps : 0 < eps) :
    ∃ t : ℝ, ∃ ht : 0 < t,
      (r.withRecapture t ht).prob2 / (r.withRecapture t ht).prob0 < eps := by
  have hb0 := r.b0_pos
  have hb1 := r.b1_pos
  have hf0 := r.f0_pos
  have hf1 := r.f1_pos
  have hf2 := r.f2_pos
  set K : ℝ := r.b0 * r.b1 + r.b0 * r.f1 + r.f0 * r.f1 with hK
  have hKpos : 0 < K := by rw [hK]; positivity
  refine ⟨K / (eps * r.b1) + 1, by positivity, ?_⟩
  set t : ℝ := K / (eps * r.b1) + 1 with ht
  have htpos : 0 < t := by rw [ht]; positivity
  have hden : 0 < r.b1 * t + r.b1 * r.f2 + r.f1 * r.f2 := by positivity
  rw [r.withRecapture_ratio, ← hK, div_lt_iff₀ hden]
  have hkey : eps * (r.b1 * t) = K + eps * r.b1 := by
    rw [ht]; field_simp
  nlinarith [hkey, mul_pos hb1 hf2, mul_pos hf1 hf2, heps]

/-! ### Entropy production -/

/-- The entropy production rate of the cycle, in units of `k_B` per unit time: the current
times the affinity. -/
def entropyProduction : ℝ := r.current * Real.log r.affinity

lemma affinity_pos : 0 < r.affinity := by
  have := r.f0_pos; have := r.f1_pos; have := r.f2_pos
  have := r.b0_pos; have := r.b1_pos; have := r.b2_pos
  unfold affinity; positivity

lemma current_pos_iff : 0 < r.current ↔ 1 < r.affinity := by
  have hb : 0 < r.b0 * r.b1 * r.b2 := by
    have := r.b0_pos; have := r.b1_pos; have := r.b2_pos; positivity
  unfold current affinity
  rw [div_pos_iff, one_lt_div hb]
  constructor
  · rintro (⟨h, _⟩ | ⟨_, h⟩)
    · linarith
    · exact absurd h (not_lt.2 r.Z_pos.le)
  · intro h; exact Or.inl ⟨by linarith, r.Z_pos⟩

lemma current_neg_iff : r.current < 0 ↔ r.affinity < 1 := by
  have hb : 0 < r.b0 * r.b1 * r.b2 := by
    have := r.b0_pos; have := r.b1_pos; have := r.b2_pos; positivity
  unfold current affinity
  rw [div_neg_iff, div_lt_one hb]
  constructor
  · rintro (⟨_, h⟩ | ⟨h, _⟩)
    · exact absurd h (not_lt.2 r.Z_pos.le)
    · linarith
  · intro h; exact Or.inr ⟨by linarith, r.Z_pos⟩

/-- **The second law for the cycle.** The entropy production is nonnegative. -/
theorem entropy_production_nonneg : 0 ≤ r.entropyProduction := by
  unfold entropyProduction
  rcases lt_trichotomy r.affinity 1 with h | h | h
  · have hJ : r.current < 0 := r.current_neg_iff.2 h
    have hlog : Real.log r.affinity < 0 := Real.log_neg r.affinity_pos h
    exact le_of_lt (mul_pos_of_neg_of_neg hJ hlog)
  · rw [h, Real.log_one, mul_zero]
  · have hJ : 0 < r.current := r.current_pos_iff.2 h
    have hlog : 0 < Real.log r.affinity := Real.log_pos h
    exact le_of_lt (mul_pos hJ hlog)

/-- The entropy production vanishes exactly at equilibrium. -/
theorem entropy_production_eq_zero_iff :
    r.entropyProduction = 0 ↔ r.f0 * r.f1 * r.f2 = r.b0 * r.b1 * r.b2 := by
  constructor
  · intro h
    by_contra hne
    have hA : r.affinity ≠ 1 := fun hA => hne (r.affinity_eq_one_iff.1 hA)
    rcases lt_or_gt_of_ne hA with hlt | hgt
    · have hJ : r.current < 0 := r.current_neg_iff.2 hlt
      have hlog : Real.log r.affinity < 0 := Real.log_neg r.affinity_pos hlt
      have := mul_pos_of_neg_of_neg hJ hlog
      rw [entropyProduction] at h
      linarith
    · have hJ : 0 < r.current := r.current_pos_iff.2 hgt
      have hlog : 0 < Real.log r.affinity := Real.log_pos hgt
      have := mul_pos hJ hlog
      rw [entropyProduction] at h
      linarith
  · intro h
    have hA : r.affinity = 1 := r.affinity_eq_one_iff.2 h
    unfold entropyProduction
    rw [hA, Real.log_one, mul_zero]

end Rates

/-- **The chaperone design law.**  For every three-state client cycle:

1. the stationary state is the explicit matrix-tree state, and the net probability flux is
   the same across all three edges;
2. that flux vanishes exactly when the Kolmogorov cycle condition holds, and then the
   stationary folded : disordered ratio is `b0 / f2` — the value it takes with no chaperone
   at all, whatever the chaperone's affinity and turnover;
3. away from that condition there is no ceiling and no floor: adjusting a single chaperone
   rate drives the ratio past any target in either direction;
4. and the price is an entropy production that is nonnegative, and zero exactly at
   equilibrium.

So a model of a disordered proteome that represents chaperones by a term in a free-energy
function is provably unable to reproduce chaperone action; the cycle, and its drive, have to
be in the model. -/
theorem chaperone_design_law (r : Rates) :
    (r.f0 * r.prob0 - r.b1 * r.prob1 = r.current ∧
      r.f1 * r.prob1 - r.b2 * r.prob2 = r.current ∧
      r.f2 * r.prob2 - r.b0 * r.prob0 = r.current) ∧
    (r.current = 0 ↔ r.f0 * r.f1 * r.f2 = r.b0 * r.b1 * r.b2) ∧
    (r.f0 * r.f1 * r.f2 = r.b0 * r.b1 * r.b2 → r.prob2 / r.prob0 = r.b0 / r.f2) ∧
    (∀ R : ℝ, ∃ t : ℝ, ∃ ht : 0 < t,
      R < (r.withCapture t ht).prob2 / (r.withCapture t ht).prob0) ∧
    (∀ eps : ℝ, 0 < eps → ∃ t : ℝ, ∃ ht : 0 < t,
      (r.withRecapture t ht).prob2 / (r.withRecapture t ht).prob0 < eps) ∧
    (0 ≤ r.entropyProduction ∧
      (r.entropyProduction = 0 ↔ r.f0 * r.f1 * r.f2 = r.b0 * r.b1 * r.b2)) :=
  ⟨⟨r.current_edge01, r.current_edge12, r.current_edge20⟩,
    r.current_eq_zero_iff,
    r.equilibrium_ratio_chaperone_independent,
    r.driven_ratio_unbounded,
    fun _ heps => r.driven_ratio_to_zero heps,
    ⟨r.entropy_production_nonneg, r.entropy_production_eq_zero_iff⟩⟩

end RequestProject.ChaperoneCycle
