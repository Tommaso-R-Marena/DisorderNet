/-
# Modular ensembles I: gluing two segments at a seam

A full-length disordered protein is almost never modelled in one piece.  The practice is
modular: sample (or measure) one segment, sample the next, and join them through the piece
of chain they share.  This file asks what that operation *is*, and what it costs.

Formally a conformation of the chain is a triple `(x, y, z)`: the state `x` of the first
segment, the state `y` of the seam (the stretch both fragments contain), and the state `z`
of the second segment.  A measurement on the first fragment sees only the pair `(x, y)`,
one on the second fragment only `(y, z)`.  The *glued* ensemble `glue p` is the distribution
built from those two panels by joining them through the seam,
`ĝ(x,y,z) = p(x,y)·p(y,z)/p(y)`.

* `glue_margXY`, `glue_margYZ` — the glued ensemble reproduces **exactly** the data it was
  built from: both fragment panels, hence every observable of either fragment alone.
* `condIndep_glue` — every glued ensemble is conditionally independent across the seam, and
  `glue_eq_self_iff_condIndep`: gluing is exact precisely for those ensembles.
* `cmi` — the conditional mutual information across the seam, and `klG_glue_eq_cmi`: the
  relative entropy of the truth from its glued model is exactly that quantity.  The price of
  modularity is the information the two ends share once the seam is known.
* `cmi_nonneg`, `cmi_eq_zero_iff_condIndep` — that price is nonnegative and vanishes only in
  the conditionally independent case.
* `modular_l1_le_sqrt_cmi` and `seam_design_rule` — through Pinsker's inequality the seam
  information converts into the operational `ℓ¹` error, giving the design rule: to build a
  full-length ensemble modularly to accuracy `eps`, cut the chain where the conditional
  mutual information across the cut is below `eps²/2`.
-/
import Mathlib
import RequestProject.Pinsker

namespace RequestProject.Modular

open Finset IDR.Pinsker

/-! ## Two analytic facts about `a·log(a/g)` -/

/-- Gibbs' pointwise inequality. -/
lemma mul_log_div_ge {a g : ℝ} (ha : 0 < a) (hg : 0 < g) : a - g ≤ a * Real.log (a / g) := by
  have h := Real.log_le_sub_one_of_pos (show (0:ℝ) < g / a by positivity)
  have hlog : Real.log (a / g) = -Real.log (g / a) := by
    rw [← Real.log_inv, inv_div]
  have := mul_le_mul_of_nonneg_left h (le_of_lt ha)
  rw [hlog]
  have hga : a * (g / a) = g := by field_simp
  nlinarith [this, hga]

/-- Equality in Gibbs' pointwise inequality forces the two arguments to agree. -/
lemma eq_of_mul_log_div_eq {a g : ℝ} (ha : 0 < a) (hg : 0 < g)
    (h : a * Real.log (a / g) = a - g) : g = a := by
  by_contra hne
  have hne' : g / a ≠ 1 := by
    intro hcon
    exact hne (by field_simp at hcon; linarith [hcon])
  have hstrict : Real.log (g / a) < g / a - 1 :=
    Real.log_lt_sub_one_of_pos (by positivity) hne'
  have hlog : Real.log (a / g) = -Real.log (g / a) := by
    rw [← Real.log_inv, inv_div]
  have hmul := mul_lt_mul_of_pos_left hstrict ha
  rw [hlog] at h
  have hga : a * (g / a - 1) = g - a := by field_simp
  nlinarith [hmul, hga, h]

variable {X Y Z : Type*} [Fintype X] [Fintype Y] [Fintype Z]

/-! ## Segment panels -/

/-- The panel a measurement on the first fragment returns: the joint law of the first
segment and the seam. -/
def margXY (p : X → Y → Z → ℝ) (x : X) (y : Y) : ℝ := ∑ z, p x y z

/-- The panel a measurement on the second fragment returns. -/
def margYZ (p : X → Y → Z → ℝ) (y : Y) (z : Z) : ℝ := ∑ x, p x y z

/-- The law of the seam itself, the only thing the two fragments share. -/
def margY (p : X → Y → Z → ℝ) (y : Y) : ℝ := ∑ x, ∑ z, p x y z

variable {p : X → Y → Z → ℝ}

omit [Fintype Y] in
lemma margY_eq_sum_margXY (p : X → Y → Z → ℝ) (y : Y) :
    margY p y = ∑ x, margXY p x y := rfl

omit [Fintype Y] in
lemma margY_eq_sum_margYZ (p : X → Y → Z → ℝ) (y : Y) :
    margY p y = ∑ z, margYZ p y z := Finset.sum_comm

omit [Fintype X] [Fintype Y] in
lemma margXY_nonneg (hp : ∀ x y z, 0 ≤ p x y z) (x : X) (y : Y) : 0 ≤ margXY p x y :=
  Finset.sum_nonneg fun _ _ => hp _ _ _

omit [Fintype Y] [Fintype Z] in
lemma margYZ_nonneg (hp : ∀ x y z, 0 ≤ p x y z) (y : Y) (z : Z) : 0 ≤ margYZ p y z :=
  Finset.sum_nonneg fun _ _ => hp _ _ _

omit [Fintype Y] in
lemma margY_nonneg (hp : ∀ x y z, 0 ≤ p x y z) (y : Y) : 0 ≤ margY p y :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => hp _ _ _

omit [Fintype Y] in
lemma margXY_le_margY (hp : ∀ x y z, 0 ≤ p x y z) (x : X) (y : Y) :
    margXY p x y ≤ margY p y := by
  rw [margY_eq_sum_margXY]
  exact Finset.single_le_sum (f := fun x => margXY p x y)
    (fun i _ => margXY_nonneg hp i y) (Finset.mem_univ x)

omit [Fintype Y] in
lemma margYZ_le_margY (hp : ∀ x y z, 0 ≤ p x y z) (y : Y) (z : Z) :
    margYZ p y z ≤ margY p y := by
  rw [margY_eq_sum_margYZ]
  exact Finset.single_le_sum (f := fun z => margYZ p y z)
    (fun i _ => margYZ_nonneg hp y i) (Finset.mem_univ z)

omit [Fintype X] [Fintype Y] in
lemma le_margXY (hp : ∀ x y z, 0 ≤ p x y z) (x : X) (y : Y) (z : Z) :
    p x y z ≤ margXY p x y :=
  Finset.single_le_sum (f := fun z => p x y z) (fun _ _ => hp _ _ _) (Finset.mem_univ z)

omit [Fintype Y] [Fintype Z] in
lemma le_margYZ (hp : ∀ x y z, 0 ≤ p x y z) (x : X) (y : Y) (z : Z) :
    p x y z ≤ margYZ p y z :=
  Finset.single_le_sum (f := fun x => p x y z) (fun _ _ => hp _ _ _) (Finset.mem_univ x)

omit [Fintype Y] in
lemma margXY_eq_zero_of_margY_eq_zero (hp : ∀ x y z, 0 ≤ p x y z) {y : Y} (h : margY p y = 0)
    (x : X) : margXY p x y = 0 :=
  le_antisymm (h ▸ margXY_le_margY hp x y) (margXY_nonneg hp x y)

omit [Fintype Y] in
lemma margYZ_eq_zero_of_margY_eq_zero (hp : ∀ x y z, 0 ≤ p x y z) {y : Y} (h : margY p y = 0)
    (z : Z) : margYZ p y z = 0 :=
  le_antisymm (h ▸ margYZ_le_margY hp y z) (margYZ_nonneg hp y z)

omit [Fintype Y] in
lemma eq_zero_of_margY_eq_zero (hp : ∀ x y z, 0 ≤ p x y z) {y : Y} (h : margY p y = 0)
    (x : X) (z : Z) : p x y z = 0 :=
  le_antisymm (by
      calc p x y z ≤ margXY p x y := le_margXY hp x y z
        _ = 0 := margXY_eq_zero_of_margY_eq_zero hp h x)
    (hp _ _ _)

omit [Fintype Y] in
/-- If a conformation is populated then so are the two fragment panels and the seam. -/
lemma pos_of_pos (hp : ∀ x y z, 0 ≤ p x y z) {x : X} {y : Y} {z : Z} (h : 0 < p x y z) :
    0 < margXY p x y ∧ 0 < margYZ p y z ∧ 0 < margY p y :=
  ⟨lt_of_lt_of_le h (le_margXY hp x y z), lt_of_lt_of_le h (le_margYZ hp x y z),
    lt_of_lt_of_le h (le_trans (le_margXY hp x y z) (margXY_le_margY hp x y))⟩

/-! ## The glued ensemble -/

/-- The modular model: the two fragment panels joined through the seam. -/
noncomputable def glue (p : X → Y → Z → ℝ) (x : X) (y : Y) (z : Z) : ℝ :=
  margXY p x y * margYZ p y z / margY p y

omit [Fintype Y] in
lemma glue_nonneg (hp : ∀ x y z, 0 ≤ p x y z) (x : X) (y : Y) (z : Z) :
    0 ≤ glue p x y z :=
  div_nonneg (mul_nonneg (margXY_nonneg hp x y) (margYZ_nonneg hp y z)) (margY_nonneg hp y)

omit [Fintype Y] in
lemma glue_pos (hp : ∀ x y z, 0 ≤ p x y z) {x : X} {y : Y} {z : Z} (h : 0 < p x y z) :
    0 < glue p x y z := by
  obtain ⟨hxy, hyz, hy⟩ := pos_of_pos hp h
  rw [glue]; positivity

omit [Fintype Y] in
/-- The glued model reproduces the first fragment's panel exactly. -/
lemma glue_margXY (hp : ∀ x y z, 0 ≤ p x y z) (x : X) (y : Y) :
    ∑ z, glue p x y z = margXY p x y := by
  have hrw : ∑ z, glue p x y z = margXY p x y * (∑ z, margYZ p y z) / margY p y := by
    rw [Finset.mul_sum, Finset.sum_div]
    exact Finset.sum_congr rfl fun z _ => rfl
  rw [hrw, ← margY_eq_sum_margYZ]
  rcases eq_or_lt_of_le (margY_nonneg hp y) with h | h
  · rw [margXY_eq_zero_of_margY_eq_zero hp h.symm x, ← h]
    simp
  · field_simp

omit [Fintype Y] in
/-- The glued model reproduces the second fragment's panel exactly. -/
lemma glue_margYZ (hp : ∀ x y z, 0 ≤ p x y z) (y : Y) (z : Z) :
    ∑ x, glue p x y z = margYZ p y z := by
  have hrw : ∑ x, glue p x y z = (∑ x, margXY p x y) * margYZ p y z / margY p y := by
    rw [Finset.sum_mul, Finset.sum_div]
    exact Finset.sum_congr rfl fun x _ => rfl
  rw [hrw, ← margY_eq_sum_margXY]
  rcases eq_or_lt_of_le (margY_nonneg hp y) with h | h
  · rw [margYZ_eq_zero_of_margY_eq_zero hp h.symm z, ← h]
    simp
  · field_simp

omit [Fintype Y] in
/-- The glued model has the same seam law. -/
lemma glue_margY (hp : ∀ x y z, 0 ≤ p x y z) (y : Y) : margY (glue p) y = margY p y := by
  rw [margY_eq_sum_margXY, Finset.sum_congr rfl fun x (_ : x ∈ Finset.univ) =>
    (show margXY (glue p) x y = margXY p x y from glue_margXY hp x y), ← margY_eq_sum_margXY]

/-- The glued model is a probability distribution. -/
lemma sum_glue (hp : ∀ x y z, 0 ≤ p x y z) (hs : ∑ x, ∑ y, ∑ z, p x y z = 1) :
    ∑ x, ∑ y, ∑ z, glue p x y z = 1 := by
  have hstep : ∀ x : X, ∑ y, ∑ z, glue p x y z = ∑ y, margXY p x y :=
    fun x => Finset.sum_congr rfl fun y _ => glue_margXY hp x y
  rw [Finset.sum_congr rfl fun x _ => hstep x]
  rw [← hs]
  exact Finset.sum_congr rfl fun x _ => rfl

/-- Conditional independence across the seam: knowing the seam state decouples the two
segments. -/
def CondIndep (p : X → Y → Z → ℝ) : Prop :=
  ∀ x y z, p x y z * margY p y = margXY p x y * margYZ p y z

omit [Fintype Y] in
/-- Every glued ensemble is conditionally independent across the seam: this is exactly the
structure a modular construction can express. -/
lemma condIndep_glue (hp : ∀ x y z, 0 ≤ p x y z) : CondIndep (glue p) := by
  intro x y z
  rw [show margXY (glue p) x y = margXY p x y from glue_margXY hp x y,
    show margYZ (glue p) y z = margYZ p y z from glue_margYZ hp y z, glue_margY hp y, glue]
  rcases eq_or_lt_of_le (margY_nonneg hp y) with h | h
  · rw [margXY_eq_zero_of_margY_eq_zero hp h.symm x, ← h]
    simp
  · field_simp

omit [Fintype Y] in
/-- **Gluing is exact precisely for conditionally independent ensembles.** -/
theorem glue_eq_self_iff_condIndep (hp : ∀ x y z, 0 ≤ p x y z) :
    (∀ x y z, glue p x y z = p x y z) ↔ CondIndep p := by
  constructor
  · intro h x y z
    rcases eq_or_lt_of_le (margY_nonneg hp y) with hy | hy
    · rw [margXY_eq_zero_of_margY_eq_zero hp hy.symm x, ← hy]
      simp
    · have hthis := h x y z
      rw [glue] at hthis
      field_simp at hthis
      linarith [hthis]
  · intro h x y z
    rcases eq_or_lt_of_le (margY_nonneg hp y) with hy | hy
    · rw [glue, margXY_eq_zero_of_margY_eq_zero hp hy.symm x,
        eq_zero_of_margY_eq_zero hp hy.symm x z]
      simp
    · have hthis := h x y z
      rw [glue, ← hthis]
      field_simp

/-! ## The price of modularity -/

/-- The conditional mutual information across the seam: the information the two segments
share once the seam state is known. -/
noncomputable def cmi (p : X → Y → Z → ℝ) : ℝ :=
  ∑ x, ∑ y, ∑ z, p x y z * Real.log (p x y z * margY p y / (margXY p x y * margYZ p y z))

/-- Termwise, the seam information is the log-likelihood ratio of truth to glued model. -/
lemma cmi_eq_sum_log_ratio (hp : ∀ x y z, 0 ≤ p x y z) :
    cmi p = ∑ x, ∑ y, ∑ z, p x y z * Real.log (p x y z / glue p x y z) := by
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun y _ =>
    Finset.sum_congr rfl fun z _ => ?_
  rcases eq_or_lt_of_le (hp x y z) with h | h
  · simp [← h]
  · obtain ⟨hxy, hyz, hy⟩ := pos_of_pos hp h
    congr 1
    rw [glue]
    field_simp

/-- **The price of modularity is exactly the seam information.**  The relative entropy of
the true ensemble from its glued model equals the conditional mutual information across the
seam. -/
theorem klG_glue_eq_cmi (hp : ∀ x y z, 0 ≤ p x y z) :
    klG (fun t : X × Y × Z => p t.1 t.2.1 t.2.2)
      (fun t : X × Y × Z => glue p t.1 t.2.1 t.2.2) = cmi p := by
  rw [klG, cmi_eq_sum_log_ratio hp, Fintype.sum_prod_type]
  exact Finset.sum_congr rfl fun x _ => by rw [Fintype.sum_prod_type]

omit [Fintype Y] in
/-- The pointwise slack in Gibbs' inequality for the glued model. -/
lemma glue_slack_nonneg (hp : ∀ x y z, 0 ≤ p x y z) (x : X) (y : Y) (z : Z) :
    0 ≤ p x y z * Real.log (p x y z / glue p x y z) - (p x y z - glue p x y z) := by
  rcases eq_or_lt_of_le (hp x y z) with h | h
  · simp [← h, glue_nonneg hp x y z]
  · have := mul_log_div_ge h (glue_pos hp h)
    linarith

/-- The seam information is nonnegative. -/
theorem cmi_nonneg (hp : ∀ x y z, 0 ≤ p x y z) (hs : ∑ x, ∑ y, ∑ z, p x y z = 1) :
    0 ≤ cmi p := by
  have hsum : ∑ x, ∑ y, ∑ z, (p x y z - glue p x y z)
      ≤ ∑ x, ∑ y, ∑ z, p x y z * Real.log (p x y z / glue p x y z) :=
    Finset.sum_le_sum fun x _ => Finset.sum_le_sum fun y _ => Finset.sum_le_sum fun z _ => by
      linarith [glue_slack_nonneg hp x y z]
  have hsplit : ∑ x, ∑ y, ∑ z, (p x y z - glue p x y z) = 0 := by
    have hd : ∑ x, ∑ y, ∑ z, (p x y z - glue p x y z)
        = (∑ x, ∑ y, ∑ z, p x y z) - ∑ x, ∑ y, ∑ z, glue p x y z := by
      simp [Finset.sum_sub_distrib]
    rw [hd, hs, sum_glue hp hs, sub_self]
  rw [cmi_eq_sum_log_ratio hp]
  linarith [hsum, hsplit]

/-- The seam information vanishes exactly when the ensemble is conditionally independent —
that is, exactly when a modular construction is not merely cheap but correct. -/
theorem cmi_eq_zero_iff_condIndep (hp : ∀ x y z, 0 ≤ p x y z)
    (hs : ∑ x, ∑ y, ∑ z, p x y z = 1) :
    cmi p = 0 ↔ CondIndep p := by
  constructor
  · intro hz
    refine (glue_eq_self_iff_condIndep hp).1 fun x y z => ?_
    -- the total slack is zero, so every term's slack is zero
    have hslack : ∑ x, ∑ y, ∑ z,
        (p x y z * Real.log (p x y z / glue p x y z) - (p x y z - glue p x y z)) = 0 := by
      have h1 : ∑ x, ∑ y, ∑ z,
          (p x y z * Real.log (p x y z / glue p x y z) - (p x y z - glue p x y z))
          = (∑ x, ∑ y, ∑ z, p x y z * Real.log (p x y z / glue p x y z))
            - ((∑ x, ∑ y, ∑ z, p x y z) - ∑ x, ∑ y, ∑ z, glue p x y z) := by
        simp [Finset.sum_sub_distrib]
      rw [h1, ← cmi_eq_sum_log_ratio hp, hz, hs, sum_glue hp hs]
      ring
    have hzero : ∀ x y z,
        p x y z * Real.log (p x y z / glue p x y z) - (p x y z - glue p x y z) = 0 := by
      intro x y z
      have h2 := (Finset.sum_eq_zero_iff_of_nonneg
        (fun x (_ : x ∈ (Finset.univ : Finset X)) => Finset.sum_nonneg fun y _ =>
          Finset.sum_nonneg fun z _ => glue_slack_nonneg hp x y z)).1 hslack x (Finset.mem_univ x)
      have h3 := (Finset.sum_eq_zero_iff_of_nonneg
        (fun y (_ : y ∈ (Finset.univ : Finset Y)) => Finset.sum_nonneg fun z _ =>
          glue_slack_nonneg hp x y z)).1 h2 y (Finset.mem_univ y)
      exact (Finset.sum_eq_zero_iff_of_nonneg
        (fun z (_ : z ∈ (Finset.univ : Finset Z)) =>
          glue_slack_nonneg hp x y z)).1 h3 z (Finset.mem_univ z)
    rcases eq_or_lt_of_le (hp x y z) with h | h
    · have hz0 := hzero x y z
      rw [← h] at hz0
      simp only [zero_mul, zero_sub, neg_neg] at hz0
      rw [hz0]
      exact h
    · exact eq_of_mul_log_div_eq h (glue_pos hp h) (by linarith [hzero x y z])
  · intro h
    have hg : ∀ x y z, glue p x y z = p x y z := (glue_eq_self_iff_condIndep hp).2 h
    rw [cmi_eq_sum_log_ratio hp]
    refine Finset.sum_eq_zero fun x _ => Finset.sum_eq_zero fun y _ =>
      Finset.sum_eq_zero fun z _ => ?_
    rcases eq_or_lt_of_le (hp x y z) with hzz | hzz
    · simp [← hzz]
    · rw [hg x y z, div_self (ne_of_gt hzz)]
      simp

/-! ## From seam information to structural error -/

/-- **Pinsker for the seam.**  The population-space `ℓ¹` error of the modular model is at
most the square root of twice the conditional mutual information across the seam. -/
theorem modular_l1_le_sqrt_cmi (hp : ∀ x y z, 0 < p x y z)
    (hs : ∑ x, ∑ y, ∑ z, p x y z = 1) :
    ∑ t : X × Y × Z, |p t.1 t.2.1 t.2.2 - glue p t.1 t.2.1 t.2.2| ≤ Real.sqrt (2 * cmi p) := by
  have hp' : ∀ x y z, 0 ≤ p x y z := fun x y z => le_of_lt (hp x y z)
  have hsum1 : ∑ t : X × Y × Z, p t.1 t.2.1 t.2.2 = 1 := by
    rw [Fintype.sum_prod_type, ← hs]
    exact Finset.sum_congr rfl fun x _ => by rw [Fintype.sum_prod_type]
  have hsum2 : ∑ t : X × Y × Z, glue p t.1 t.2.1 t.2.2 = 1 := by
    rw [Fintype.sum_prod_type, ← sum_glue hp' hs]
    exact Finset.sum_congr rfl fun x _ => by rw [Fintype.sum_prod_type]
  have := ell1_le_sqrt_two_klG (p := fun t : X × Y × Z => p t.1 t.2.1 t.2.2)
    (q := fun t : X × Y × Z => glue p t.1 t.2.1 t.2.2)
    (fun t => hp' _ _ _) (fun t => glue_pos hp' (hp t.1 t.2.1 t.2.2)) hsum1 hsum2
  rwa [klG_glue_eq_cmi hp'] at this

/-- **The seam design rule.**  To build a full-length ensemble modularly to population
accuracy `eps`, cut the chain where the conditional mutual information across the cut is
below `eps²/2`. -/
theorem seam_design_rule (hp : ∀ x y z, 0 < p x y z)
    (hs : ∑ x, ∑ y, ∑ z, p x y z = 1) {eps : ℝ} (heps : 0 ≤ eps)
    (hcut : cmi p ≤ eps ^ 2 / 2) :
    ∑ t : X × Y × Z, |p t.1 t.2.1 t.2.2 - glue p t.1 t.2.1 t.2.2| ≤ eps := by
  have h1 := modular_l1_le_sqrt_cmi hp hs
  have h2 : Real.sqrt (2 * cmi p) ≤ eps := by
    have : 2 * cmi p ≤ eps ^ 2 := by linarith
    calc Real.sqrt (2 * cmi p) ≤ Real.sqrt (eps ^ 2) := Real.sqrt_le_sqrt this
      _ = eps := by rw [Real.sqrt_sq heps]
  linarith

/-! ## Fragment data can never test the modular assumption -/

omit [Fintype Y] in
/-- The seam law of a glued ensemble is unchanged, and its own glued model is itself. -/
lemma glue_glue (hp : ∀ x y z, 0 ≤ p x y z) (x : X) (y : Y) (z : Z) :
    glue (glue p) x y z = glue p x y z := by
  have h1 : margXY (glue p) x y = margXY p x y := glue_margXY hp x y
  have h2 : margYZ (glue p) y z = margYZ p y z := glue_margYZ hp y z
  rw [glue, h1, h2, glue_margY hp y, ← glue]

/-- **Fragment measurements can never falsify modularity.**  Whatever the truth is, the
glued ensemble reproduces both fragment panels exactly and has seam information zero — so
there is always an exactly modular ensemble consistent with every fragment measurement.
The conditional independence a modular pipeline assumes is testable only by an observable
that spans the cut. -/
theorem fragment_panels_never_falsify_modularity (hp : ∀ x y z, 0 ≤ p x y z)
    (hs : ∑ x, ∑ y, ∑ z, p x y z = 1) :
    (∀ x y, margXY (glue p) x y = margXY p x y) ∧
      (∀ y z, margYZ (glue p) y z = margYZ p y z) ∧ cmi (glue p) = 0 :=
  ⟨fun x y => glue_margXY hp x y, fun y z => glue_margYZ hp y z,
    (cmi_eq_zero_iff_condIndep (glue_nonneg hp) (sum_glue hp hs)).2 (condIndep_glue hp)⟩

end RequestProject.Modular
