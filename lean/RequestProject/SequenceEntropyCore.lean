/-
# The information-theoretic toolkit for the sequence-entropy question

This file is the measure-free, physics-free half of the answer to the question *"is there a
maximum information entropy of an amino-acid sequence beyond which a region must be
disordered?"*.  It collects, with proofs, the four inequalities that the physical argument in
`RequestProject.SequenceEntropyLimit` consumes.

Everything is stated for a weight vector `p` on a finite type, with `H p = -∑ p log p` in **nats**.

* `gibbs` -- the Gibbs (log-sum) inequality against an arbitrary sub-probability `u`.  All the
  upper bounds below are one choice of `u`.
* `H_le_log_card` / `H_uniform` -- **the ceiling**: an ensemble supported on a set `F` has
  entropy at most `log |F|`, and the uniform ensemble on `F` attains it.  So `log |F|` is
  *exactly* the maximum entropy compatible with staying inside `F`.
* `H_le_split` -- **the robust ceiling**: an ensemble that leaves `F` with probability `eps`
  obeys `H ≤ (1-eps) log |F| + eps log |Fᶜ| + h₂(eps)`; `H_le_split_of_card_le` and
  `escape_prob_ge` turn this into a lower bound on the escape probability, which is the form
  the physics uses.
* `H_ge_neg_log_max` -- the min-entropy bound `H ≥ -log (max p)`.
* `H_ge_minority` -- `H ≥ δ log 2` where `δ` is the total weight off the modal letter.  Together
  with the previous line this is what converts "low compositional entropy" into "almost all
  residues are the same letter".
* `ballCard_mul_pow_le` / `log_ballCard_le` / `log_ballCard_le_qaryEntropy` -- the Hamming
  sphere-packing count: the number of sequences within `r` mutations of a given one is at most
  `exp (N · h_q(r/N))`, `h_q` the `q`-ary entropy function.  This is what makes the ceiling
  `log |F|` an explicit number once the physics bounds `F` by a union of Hamming balls.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace SeqEnt

open Finset

/-! ## Shannon entropy on a finite type -/

/-- Shannon entropy in nats of a weight vector on a finite type. -/
noncomputable def H {α : Type*} [Fintype α] (p : α → ℝ) : ℝ := ∑ a, -(p a * Real.log (p a))

/-- The binary entropy function, in nats. -/
noncomputable def h₂ (x : ℝ) : ℝ := -(x * Real.log x) - (1 - x) * Real.log (1 - x)

/-- The `q`-ary entropy function, in nats: `h_q(x) = h₂(x) + x·log (q-1)`. -/
noncomputable def qaryEnt (q : ℕ) (x : ℝ) : ℝ := h₂ x + x * Real.log (q - 1 : ℝ)

lemma h₂_nonneg {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1) : 0 ≤ h₂ x := by
  have hx : -(x * Real.log x) ≥ 0 := by
    rcases eq_or_lt_of_le h0 with h | h
    · simp [← h]
    · have : Real.log x ≤ 0 := Real.log_nonpos h0 h1
      nlinarith
  have hy : -((1 - x) * Real.log (1 - x)) ≥ 0 := by
    rcases eq_or_lt_of_le (by linarith : (0:ℝ) ≤ 1 - x) with h | h
    · simp [← h]
    · have : Real.log (1 - x) ≤ 0 := Real.log_nonpos (by linarith) (by linarith)
      nlinarith
  simp only [h₂]
  linarith

lemma h₂_le_log_two {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1) : h₂ x ≤ Real.log 2 := by
  -- `-x log x - (1-x) log (1-x) ≤ log 2` from `log t ≥ 1 - 1/t`, i.e. `-t log t ≤ t (1/t - 1)`…
  -- we use the Gibbs inequality in its two-point form directly.
  have key : ∀ y : ℝ, 0 ≤ y → y ≤ 1 → -(y * Real.log y) ≤ y * Real.log 2 + (1/2 - y) := by
    intro y hy0 hy1
    rcases eq_or_lt_of_le hy0 with h | h
    · simp [← h]
    · have hlog : Real.log ((1/2) / y) ≤ (1/2) / y - 1 :=
        Real.log_le_sub_one_of_pos (by positivity)
      have hdiv : Real.log ((1/2) / y) = Real.log (1/2) - Real.log y :=
        Real.log_div (by norm_num) (ne_of_gt h)
      have h12 : Real.log (1/2) = -Real.log 2 := by
        rw [one_div, Real.log_inv]
      have : y * (Real.log (1/2) - Real.log y) ≤ y * ((1/2) / y - 1) := by
        apply mul_le_mul_of_nonneg_left _ hy0
        rw [← hdiv]; exact hlog
      rw [h12] at this
      have hy : y * ((1/2)/y - 1) = 1/2 - y := by field_simp
      rw [hy] at this
      nlinarith
  have h1' := key x h0 h1
  have h2' := key (1 - x) (by linarith) (by linarith)
  simp only [h₂]
  nlinarith [h1', h2']

/-- **The Gibbs inequality.**  For a probability vector `p` and any sub-probability `u` whose
support contains that of `p`, the entropy of `p` is at most the cross entropy `-∑ p log u`. -/
theorem gibbs {α : Type*} [Fintype α] {p u : α → ℝ} (hp : ∀ a, 0 ≤ p a) (hps : ∑ a, p a = 1)
    (hu : ∀ a, 0 ≤ u a) (hus : ∑ a, u a ≤ 1) (hsupp : ∀ a, p a ≠ 0 → u a ≠ 0) :
    H p ≤ ∑ a, -(p a * Real.log (u a)) := by
  have key : ∀ a, -(p a * Real.log (p a)) + p a * Real.log (u a) ≤ u a - p a := by
    intro a
    rcases eq_or_lt_of_le (hp a) with h0 | hpos
    · simp [← h0, hu a]
    · have hune : u a ≠ 0 := hsupp a (ne_of_gt hpos)
      have hupos : 0 < u a := lt_of_le_of_ne (hu a) (Ne.symm hune)
      have hlog : Real.log (u a / p a) ≤ u a / p a - 1 :=
        Real.log_le_sub_one_of_pos (by positivity)
      have hdiv : Real.log (u a / p a) = Real.log (u a) - Real.log (p a) :=
        Real.log_div hune (ne_of_gt hpos)
      have hmul : p a * (Real.log (u a) - Real.log (p a)) ≤ p a * (u a / p a - 1) := by
        apply mul_le_mul_of_nonneg_left _ (hp a)
        rw [← hdiv]; exact hlog
      have : p a * (u a / p a - 1) = u a - p a := by field_simp
      rw [this] at hmul
      linarith
  have hsum : ∑ a, (-(p a * Real.log (p a)) + p a * Real.log (u a)) ≤ ∑ a, (u a - p a) :=
    Finset.sum_le_sum fun a _ => key a
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, hps] at hsum
  have : H p + ∑ a, p a * Real.log (u a) ≤ ∑ a, u a - 1 := by
    simpa [H] using hsum
  have h2 : ∑ a, -(p a * Real.log (u a)) = -∑ a, p a * Real.log (u a) := by
    simp [Finset.sum_neg_distrib]
  rw [h2]
  linarith

/-! ## The ceiling: entropy of an ensemble confined to a set -/

/-- **The entropy ceiling.**  An ensemble supported on a finite set `F` has entropy at most
`log |F|`. -/
theorem H_le_log_card {α : Type*} [Fintype α] [DecidableEq α] {p : α → ℝ} {F : Finset α}
    (hp : ∀ a, 0 ≤ p a) (hps : ∑ a, p a = 1) (hsupp : ∀ a, a ∉ F → p a = 0)
    (hF : F.Nonempty) : H p ≤ Real.log F.card := by
  classical
  have hcard : (0 : ℝ) < F.card := by exact_mod_cast Finset.card_pos.2 hF
  set u : α → ℝ := fun a => if a ∈ F then ((F.card : ℝ))⁻¹ else 0 with hu_def
  have hu : ∀ a, 0 ≤ u a := by
    intro a
    by_cases h : a ∈ F
    · simp only [hu_def, if_pos h]
      positivity
    · simp [hu_def, h]
  have hus : ∑ a, u a ≤ 1 := by
    have hval : ∑ a, u a = ∑ _a ∈ F, ((F.card : ℝ))⁻¹ := by
      rw [hu_def, Finset.sum_ite_mem]
      simp
    rw [hval, Finset.sum_const, nsmul_eq_mul, mul_inv_cancel₀ (ne_of_gt hcard)]
  have hsupp' : ∀ a, p a ≠ 0 → u a ≠ 0 := by
    intro a ha
    by_cases h : a ∈ F
    · simp only [hu_def, if_pos h]
      positivity
    · exact absurd (hsupp a h) ha
  have hg := gibbs hp hps hu hus hsupp'
  have hcross : ∑ a, -(p a * Real.log (u a)) = Real.log F.card := by
    have hterm : ∀ a ∈ (Finset.univ : Finset α),
        -(p a * Real.log (u a)) = p a * Real.log F.card := by
      intro a _
      by_cases h : a ∈ F
      · simp only [hu_def, if_pos h]
        rw [Real.log_inv]
        ring
      · rw [hsupp a h]; simp
    rw [Finset.sum_congr rfl hterm, ← Finset.sum_mul, hps, one_mul]
  rw [hcross] at hg
  exact hg

/-- The uniform ensemble on `F` attains the ceiling: its entropy is exactly `log |F|`. -/
theorem H_uniform {α : Type*} [Fintype α] [DecidableEq α] {F : Finset α} (hF : F.Nonempty) :
    H (fun a => if a ∈ F then ((F.card : ℝ))⁻¹ else 0) = Real.log F.card := by
  classical
  have hcard : (0 : ℝ) < F.card := by exact_mod_cast Finset.card_pos.2 hF
  have hterm : ∀ a ∈ (Finset.univ : Finset α),
      -((if a ∈ F then ((F.card : ℝ))⁻¹ else 0) *
        Real.log (if a ∈ F then ((F.card : ℝ))⁻¹ else 0))
        = if a ∈ F then ((F.card : ℝ))⁻¹ * Real.log F.card else 0 := by
    intro a _
    by_cases h : a ∈ F
    · simp only [if_pos h, Real.log_inv]; ring
    · simp [h]
  rw [H, Finset.sum_congr rfl hterm, Finset.sum_ite_mem, Finset.univ_inter, Finset.sum_const,
    nsmul_eq_mul]
  field_simp

/-- **The robust ceiling.**  If the ensemble `p` puts probability `eps` outside `F`, then
`H p ≤ (1-eps)·log |F| + eps·log |Fᶜ| + h₂(eps)`. -/
theorem H_le_split {α : Type*} [Fintype α] [DecidableEq α] {p : α → ℝ} {F : Finset α} {eps : ℝ}
    (hp : ∀ a, 0 ≤ p a) (hps : ∑ a, p a = 1) (hF : F.Nonempty) (hFc : Fᶜ.Nonempty)
    (heps : eps = ∑ a ∈ Fᶜ, p a) (h0 : 0 < eps) (h1 : eps < 1) :
    H p ≤ (1 - eps) * Real.log F.card + eps * Real.log Fᶜ.card + h₂ eps := by
  classical
  have hcard : (0 : ℝ) < F.card := by exact_mod_cast Finset.card_pos.2 hF
  have hcardc : (0 : ℝ) < Fᶜ.card := by exact_mod_cast Finset.card_pos.2 hFc
  set u : α → ℝ := fun a => if a ∈ F then (1 - eps) / (F.card : ℝ) else eps / (Fᶜ.card : ℝ)
    with hu_def
  have hu : ∀ a, 0 ≤ u a := by
    intro a
    by_cases h : a ∈ F
    · simp only [hu_def, if_pos h]
      have : (0:ℝ) ≤ 1 - eps := by linarith
      positivity
    · simp only [hu_def, if_neg h]
      positivity
  have hsplit : ∑ a, u a = 1 := by
    have h1' : ∑ a, u a = ∑ a ∈ F, u a + ∑ a ∈ Fᶜ, u a := by
      rw [← Finset.sum_add_sum_compl F u]
    have h2' : ∑ a ∈ F, u a = (1 - eps) := by
      have hcongr : ∀ a ∈ F, u a = (1 - eps) / (F.card : ℝ) := by
        intro a ha; simp only [hu_def, if_pos ha]
      rw [Finset.sum_congr rfl hcongr, Finset.sum_const, nsmul_eq_mul]
      field_simp
    have h3' : ∑ a ∈ Fᶜ, u a = eps := by
      have hcongr : ∀ a ∈ Fᶜ, u a = eps / (Fᶜ.card : ℝ) := by
        intro a ha
        have hnot : a ∉ F := Finset.mem_compl.1 ha
        simp only [hu_def, if_neg hnot]
      rw [Finset.sum_congr rfl hcongr, Finset.sum_const, nsmul_eq_mul]
      field_simp
    rw [h1', h2', h3']; ring
  have hsupp' : ∀ a, p a ≠ 0 → u a ≠ 0 := by
    intro a _
    by_cases h : a ∈ F
    · simp only [hu_def, if_pos h]
      have : (0:ℝ) < 1 - eps := by linarith
      positivity
    · simp only [hu_def, if_neg h]
      positivity
  have hg := gibbs hp hps hu (le_of_eq hsplit) hsupp'
  have hpF : ∑ a ∈ F, p a = 1 - eps := by
    have := Finset.sum_add_sum_compl F p
    rw [hps] at this
    rw [heps]; linarith [this]
  have hcross : ∑ a, -(p a * Real.log (u a))
      = (1 - eps) * (Real.log F.card - Real.log (1 - eps))
        + eps * (Real.log Fᶜ.card - Real.log eps) := by
    have hsum : ∑ a, -(p a * Real.log (u a))
        = ∑ a ∈ F, -(p a * Real.log (u a)) + ∑ a ∈ Fᶜ, -(p a * Real.log (u a)) := by
      rw [← Finset.sum_add_sum_compl F fun a => -(p a * Real.log (u a))]
    have hA : ∑ a ∈ F, -(p a * Real.log (u a))
        = (1 - eps) * (Real.log F.card - Real.log (1 - eps)) := by
      have : ∀ a ∈ F, -(p a * Real.log (u a))
          = p a * (Real.log F.card - Real.log (1 - eps)) := by
        intro a ha
        simp only [hu_def, if_pos ha]
        rw [Real.log_div (by linarith) (ne_of_gt hcard)]
        ring
      rw [Finset.sum_congr rfl this, ← Finset.sum_mul, hpF]
    have hB : ∑ a ∈ Fᶜ, -(p a * Real.log (u a))
        = eps * (Real.log Fᶜ.card - Real.log eps) := by
      have : ∀ a ∈ Fᶜ, -(p a * Real.log (u a))
          = p a * (Real.log Fᶜ.card - Real.log eps) := by
        intro a ha
        have hnot : a ∉ F := Finset.mem_compl.1 ha
        simp only [hu_def, if_neg hnot]
        rw [Real.log_div (ne_of_gt h0) (ne_of_gt hcardc)]
        ring
      rw [Finset.sum_congr rfl this, ← Finset.sum_mul, ← heps]
    rw [hsum, hA, hB]
  rw [hcross] at hg
  have : (1 - eps) * (Real.log F.card - Real.log (1 - eps))
        + eps * (Real.log Fᶜ.card - Real.log eps)
      = (1 - eps) * Real.log F.card + eps * Real.log Fᶜ.card + h₂ eps := by
    simp only [h₂]; ring
  linarith [hg, this.le, this.ge]

/-- The form the physics uses: with `|F| ≥ 1` and `|Fᶜ| ≤ C`, an ensemble whose escape
probability is `eps` has `H p ≤ log |F| + eps·log C + log 2`. -/
theorem H_le_split_of_card_le {α : Type*} [Fintype α] [DecidableEq α] {p : α → ℝ} {F : Finset α}
    {eps C : ℝ} (hp : ∀ a, 0 ≤ p a) (hps : ∑ a, p a = 1) (hF : F.Nonempty) (hFc : Fᶜ.Nonempty)
    (heps : eps = ∑ a ∈ Fᶜ, p a) (h0 : 0 < eps) (h1 : eps < 1)
    (hC : Real.log Fᶜ.card ≤ C) (hFcard : 1 ≤ F.card) :
    H p ≤ Real.log F.card + eps * C + Real.log 2 := by
  have hsplit := H_le_split hp hps hF hFc heps h0 h1
  have hlogF : 0 ≤ Real.log F.card := Real.log_nonneg (by exact_mod_cast hFcard)
  have hh : h₂ eps ≤ Real.log 2 := h₂_le_log_two h0.le h1.le
  nlinarith [hsplit, hh, hlogF, hC, h0.le]

/-- **The escape bound**, the contrapositive of the ceiling: an ensemble of sequences whose
entropy exceeds `log |F| + log 2` must place probability at least
`(H p - log |F| - log 2)/C` outside `F`. -/
theorem escape_prob_ge {α : Type*} [Fintype α] [DecidableEq α] {p : α → ℝ} {F : Finset α}
    {eps C : ℝ} (hp : ∀ a, 0 ≤ p a) (hps : ∑ a, p a = 1) (hF : F.Nonempty) (hFc : Fᶜ.Nonempty)
    (heps : eps = ∑ a ∈ Fᶜ, p a) (h0 : 0 < eps) (h1 : eps < 1)
    (hC : Real.log Fᶜ.card ≤ C) (hFcard : 1 ≤ F.card) (hC0 : 0 < C) :
    (H p - Real.log F.card - Real.log 2) / C ≤ eps := by
  have h := H_le_split_of_card_le hp hps hF hFc heps h0 h1 hC hFcard
  rw [div_le_iff₀ hC0]
  nlinarith [h]

/-! ## Lower bounds: what a *small* entropy forces -/

lemma H_nonneg {α : Type*} [Fintype α] {p : α → ℝ} (hp : ∀ a, 0 ≤ p a) (hp1 : ∀ a, p a ≤ 1) :
    0 ≤ H p := by
  refine Finset.sum_nonneg fun a _ => ?_
  rcases eq_or_lt_of_le (hp a) with h | h
  · simp [← h]
  · have : Real.log (p a) ≤ 0 := Real.log_nonpos (hp a) (hp1 a)
    nlinarith

/-- **The min-entropy bound.**  `H p ≥ -log (max_a p a)`. -/
theorem H_ge_neg_log_max {α : Type*} [Fintype α] {p : α → ℝ} {m : ℝ} (hp : ∀ a, 0 ≤ p a)
    (hps : ∑ a, p a = 1) (hm : ∀ a, p a ≤ m) : -Real.log m ≤ H p := by
  have key : ∀ a ∈ (Finset.univ : Finset α), p a * (-Real.log m) ≤ -(p a * Real.log (p a)) := by
    intro a _
    rcases eq_or_lt_of_le (hp a) with h0 | hpos
    · simp [← h0]
    · have : Real.log (p a) ≤ Real.log m := Real.log_le_log hpos (hm a)
      nlinarith
  have := Finset.sum_le_sum key
  rw [← Finset.sum_mul, hps, one_mul] at this
  exact this

/-- **The minority bound.**  If the weight off the letter `c` is `δ ≤ 1/2`, then
`H p ≥ δ·log 2`.  A sequence of very low compositional entropy is one in which almost every
residue is the same letter. -/
theorem H_ge_minority {α : Type*} [Fintype α] [DecidableEq α] {p : α → ℝ} {c : α} {del : ℝ}
    (hp : ∀ a, 0 ≤ p a) (hps : ∑ a, p a = 1) (hdel : del = ∑ a ∈ Finset.univ.erase c, p a)
    (hhalf : del ≤ 1 / 2) : del * Real.log 2 ≤ H p := by
  classical
  have hp1 : ∀ a, p a ≤ 1 := by
    intro a
    have : p a ≤ ∑ b, p b :=
      Finset.single_le_sum (f := p) (fun b _ => hp b) (Finset.mem_univ a)
    linarith [hps ▸ this]
  by_cases hd0 : 0 < del
  case neg =>
    have hle0 : del * Real.log 2 ≤ 0 := by
      have hl2 : (0:ℝ) ≤ Real.log 2 := Real.log_nonneg (by norm_num)
      have : del ≤ 0 := by linarith [not_lt.1 hd0]
      nlinarith
    linarith [H_nonneg hp hp1]
  case pos =>
    -- each off-`c` weight is at most `del ≤ 1/2`, so `-p log p ≥ p log 2`
    have hle : ∀ a ∈ Finset.univ.erase c, p a ≤ del := by
      intro a ha
      rw [hdel]
      exact Finset.single_le_sum (f := p) (fun b _ => hp b) ha
    have key : ∀ a ∈ Finset.univ.erase c, p a * Real.log 2 ≤ -(p a * Real.log (p a)) := by
      intro a ha
      rcases eq_or_lt_of_le (hp a) with h0 | hpos
      · simp [← h0]
      · have h1 : Real.log (p a) ≤ Real.log del := Real.log_le_log hpos (hle a ha)
        have h2 : Real.log del ≤ Real.log (1/2) := Real.log_le_log hd0 hhalf
        have h3 : Real.log (1/2) = -Real.log 2 := by rw [one_div, Real.log_inv]
        nlinarith
    have hsum : ∑ a ∈ Finset.univ.erase c, p a * Real.log 2
        ≤ ∑ a ∈ Finset.univ.erase c, -(p a * Real.log (p a)) := Finset.sum_le_sum key
    rw [← Finset.sum_mul, ← hdel] at hsum
    have hrest : ∑ a ∈ Finset.univ.erase c, -(p a * Real.log (p a)) ≤ H p := by
      have hsub : Finset.univ.erase c ⊆ (Finset.univ : Finset α) := Finset.erase_subset _ _
      refine Finset.sum_le_sum_of_subset_of_nonneg hsub ?_
      intro a _ _
      rcases eq_or_lt_of_le (hp a) with h0 | hpos
      · simp [← h0]
      · have : Real.log (p a) ≤ 0 := Real.log_nonpos (hp a) (hp1 a)
        nlinarith
    linarith

/-! ## Hamming ball volumes: the sphere-packing count -/

/-- The number of `q`-ary strings of length `N` within Hamming distance `r` of a fixed one. -/
def ballCard (N q r : ℕ) : ℕ := ∑ k ∈ Finset.range (r + 1), N.choose k * (q - 1) ^ k

/-- **The Chernoff form of the volume bound.**  For every `0 < t ≤ 1`,
`|B(r)| · t^r ≤ (1 + (q-1)t)^N`. -/
theorem ballCard_mul_pow_le (N r : ℕ) {q : ℕ} (hq : 1 ≤ q) {t : ℝ} (ht0 : 0 < t) (ht1 : t ≤ 1) :
    (ballCard N q r : ℝ) * t ^ r ≤ (1 + ((q : ℝ) - 1) * t) ^ N := by
  classical
  have hQ : ((q - 1 : ℕ) : ℝ) = (q : ℝ) - 1 := by
    simpa using (Nat.cast_sub hq : ((q - 1 : ℕ) : ℝ) = (q : ℝ) - (1 : ℕ))
  set Q : ℝ := (q : ℝ) - 1 with hQdef
  have hQ0 : 0 ≤ Q := by
    have : (1 : ℝ) ≤ (q : ℝ) := by exact_mod_cast hq
    simp [hQdef]; linarith
  -- step 1: `t^r ≤ t^k` for `k ≤ r`
  have step1 : (ballCard N q r : ℝ) * t ^ r
      ≤ ∑ k ∈ Finset.range (r + 1), (N.choose k : ℝ) * (Q * t) ^ k := by
    rw [ballCard]
    push_cast [hQ]
    rw [Finset.sum_mul]
    refine Finset.sum_le_sum fun k hk => ?_
    have hkr : k ≤ r := Nat.lt_succ_iff.1 (Finset.mem_range.1 hk)
    have htk : t ^ r ≤ t ^ k := pow_le_pow_of_le_one ht0.le ht1 hkr
    have hpos : (0 : ℝ) ≤ (N.choose k : ℝ) * Q ^ k := by positivity
    calc (N.choose k : ℝ) * Q ^ k * t ^ r ≤ (N.choose k : ℝ) * Q ^ k * t ^ k :=
          mul_le_mul_of_nonneg_left htk hpos
      _ = (N.choose k : ℝ) * (Q * t) ^ k := by rw [mul_pow]; ring
  -- step 2: extend/truncate the range to `N`
  have step2 : ∑ k ∈ Finset.range (r + 1), (N.choose k : ℝ) * (Q * t) ^ k
      ≤ ∑ k ∈ Finset.range (N + 1), (N.choose k : ℝ) * (Q * t) ^ k := by
    rcases le_total r N with h | h
    · refine Finset.sum_le_sum_of_subset_of_nonneg
        (Finset.range_subset_range.2 (show r + 1 ≤ N + 1 by omega)) fun k _ _ => ?_
      have : (0 : ℝ) ≤ (Q * t) ^ k := by positivity
      positivity
    · refine le_of_eq (Finset.sum_subset
        (Finset.range_subset_range.2 (show N + 1 ≤ r + 1 by omega)) ?_).symm
      intro k _ hk
      have : N.choose k = 0 := Nat.choose_eq_zero_of_lt (by
        have := Finset.mem_range.not.1 hk; omega)
      simp [this]
  -- step 3: the binomial theorem
  have step3 : ∑ k ∈ Finset.range (N + 1), (N.choose k : ℝ) * (Q * t) ^ k = (1 + Q * t) ^ N := by
    rw [add_comm (1 : ℝ) (Q * t), add_pow]
    refine Finset.sum_congr rfl fun k _ => ?_
    ring
  linarith [step1, step2, step3.le, step3.ge]

/-- The logarithmic form of the volume bound. -/
theorem log_ballCard_le (N r : ℕ) {q : ℕ} (hq : 1 ≤ q) {t : ℝ} (ht0 : 0 < t) (ht1 : t ≤ 1)
    (hball : 1 ≤ ballCard N q r) :
    Real.log (ballCard N q r) ≤ N * Real.log (1 + ((q : ℝ) - 1) * t) - r * Real.log t := by
  have h := ballCard_mul_pow_le N r hq ht0 ht1
  have hb : (1 : ℝ) ≤ (ballCard N q r : ℝ) := by exact_mod_cast hball
  have hQ0 : (0 : ℝ) ≤ (q : ℝ) - 1 := by
    have : (1 : ℝ) ≤ (q : ℝ) := by exact_mod_cast hq
    linarith
  have hbase : (0 : ℝ) < 1 + ((q : ℝ) - 1) * t := by positivity
  have hlhs : (0 : ℝ) < (ballCard N q r : ℝ) * t ^ r := by positivity
  have hlog := Real.log_le_log hlhs h
  rw [Real.log_mul (by positivity) (by positivity), Real.log_pow, Real.log_pow] at hlog
  linarith

/-- **The sphere-packing count in entropy form.**  The number of `q`-ary strings within `r`
mutations of a fixed one is at most `exp (N · h_q(r/N))`. -/
theorem log_ballCard_le_qaryEntropy {N r q : ℕ} (hq : 2 ≤ q) (hN : 0 < N) (hr : 0 < r)
    (hrN : r < N) (hfrac : (r : ℝ) / N ≤ 1 - 1 / q) (hball : 1 ≤ ballCard N q r) :
    Real.log (ballCard N q r) ≤ N * qaryEnt q ((r : ℝ) / N) := by
  set p : ℝ := (r : ℝ) / N with hpdef
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hq0 : (0 : ℝ) < (q : ℝ) - 1 := by
    have : (2 : ℝ) ≤ (q : ℝ) := by exact_mod_cast hq
    linarith
  have hp0 : 0 < p := by
    have : (0 : ℝ) < r := by exact_mod_cast hr
    positivity
  have hp1 : p < 1 := by
    rw [hpdef, div_lt_one hN0]
    exact_mod_cast hrN
  set t : ℝ := p / ((1 - p) * ((q : ℝ) - 1)) with htdef
  have ht0 : 0 < t := by
    have : (0 : ℝ) < (1 - p) * ((q : ℝ) - 1) := by nlinarith
    positivity
  have hqpos : (0 : ℝ) < (q : ℝ) := by linarith
  have ht1 : t ≤ 1 := by
    rw [htdef, div_le_one (by nlinarith)]
    have hpq : p * (q : ℝ) ≤ (q : ℝ) - 1 := by
      have hmul : p * (q : ℝ) ≤ (1 - 1 / (q : ℝ)) * (q : ℝ) :=
        mul_le_mul_of_nonneg_right hfrac hqpos.le
      have hid : (1 - 1 / (q : ℝ)) * (q : ℝ) = (q : ℝ) - 1 := by field_simp
      linarith
    nlinarith
  have hne : (1 : ℝ) - p ≠ 0 := by linarith
  have hq1 : (q : ℝ) - 1 ≠ 0 := ne_of_gt hq0
  have hbase : 1 + ((q : ℝ) - 1) * t = 1 / (1 - p) := by
    rw [htdef]
    field_simp
    ring
  have hlog := log_ballCard_le N r (by omega) ht0 ht1 hball
  rw [hbase] at hlog
  have hrp : (r : ℝ) = p * N := by rw [hpdef]; field_simp
  have hlogt : Real.log t = Real.log p - Real.log (1 - p) - Real.log ((q : ℝ) - 1) := by
    rw [htdef, Real.log_div (ne_of_gt hp0)
        (ne_of_gt (show (0:ℝ) < (1 - p) * ((q : ℝ) - 1) by nlinarith)),
      Real.log_mul hne hq1]
    ring
  have hlog1 : Real.log (1 / (1 - p)) = -Real.log (1 - p) := by
    rw [one_div, Real.log_inv]
  rw [hlog1, hlogt, hrp] at hlog
  have : (N : ℝ) * -Real.log (1 - p)
      - p * N * (Real.log p - Real.log (1 - p) - Real.log ((q : ℝ) - 1))
      = N * qaryEnt q p := by
    simp only [qaryEnt, h₂]
    ring
  linarith [hlog, this.le, this.ge]

lemma h₂_symm (x : ℝ) : h₂ (1 - x) = h₂ x := by
  simp only [h₂]
  ring_nf

lemma one_le_ballCard (N q r : ℕ) : 1 ≤ ballCard N q r := by
  have : N.choose 0 * (q - 1) ^ 0 ≤ ballCard N q r := by
    refine Finset.single_le_sum (f := fun k => N.choose k * (q - 1) ^ k)
      (fun k _ => Nat.zero_le _) (Finset.mem_range.2 (Nat.succ_pos r))
  simpa using this

lemma choose_le_ballCard {N r : ℕ} : N.choose r ≤ ballCard N 2 r := by
  have hmem : r ∈ Finset.range (r + 1) := Finset.mem_range.2 (Nat.lt_succ_self r)
  have := Finset.single_le_sum (f := fun k => N.choose k * (2 - 1) ^ k)
    (fun k _ => Nat.zero_le _) hmem
  simpa [ballCard] using this

/-- **The binomial coefficient bound.**  `log C(N,r) ≤ N·h₂(r/N)`, the entropy form of the
classical estimate, valid on the whole range `0 < r < N`. -/
theorem log_choose_le {N r : ℕ} (hN : 0 < N) (hr : 0 < r) (hrN : r < N) :
    Real.log (N.choose r) ≤ (N : ℝ) * h₂ ((r : ℝ) / N) := by
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hqe : ∀ x : ℝ, qaryEnt 2 x = h₂ x := by
    intro x
    simp only [qaryEnt]
    norm_num
  rcases le_or_gt (2 * r) N with hhalf | hhalf
  · have hfrac : (r : ℝ) / N ≤ 1 - 1 / (2 : ℕ) := by
      rw [div_le_iff₀ hN0]
      have : (2 : ℝ) * r ≤ N := by exact_mod_cast hhalf
      push_cast
      linarith
    have hb := log_ballCard_le_qaryEntropy (N := N) (r := r) (q := 2) (by norm_num) hN hr hrN
      hfrac (one_le_ballCard N 2 r)
    have hc : (N.choose r : ℝ) ≤ (ballCard N 2 r : ℝ) := by
      exact_mod_cast choose_le_ballCard
    have hpos : (0 : ℝ) < (N.choose r : ℝ) := by
      have : 0 < N.choose r := Nat.choose_pos (le_of_lt hrN)
      exact_mod_cast this
    have := Real.log_le_log hpos hc
    rw [hqe] at hb
    linarith
  · -- reflect: `C(N,r) = C(N,N-r)` and `h₂` is symmetric
    set r' : ℕ := N - r with hr'
    have hr'0 : 0 < r' := by omega
    have hr'N : r' < N := by omega
    have hhalf' : 2 * r' ≤ N := by omega
    have hfrac : (r' : ℝ) / N ≤ 1 - 1 / (2 : ℕ) := by
      rw [div_le_iff₀ hN0]
      have : (2 : ℝ) * r' ≤ N := by exact_mod_cast hhalf'
      push_cast
      linarith
    have hb := log_ballCard_le_qaryEntropy (N := N) (r := r') (q := 2) (by norm_num) hN hr'0 hr'N
      hfrac (one_le_ballCard N 2 r')
    have hsym : N.choose r = N.choose r' := by
      rw [hr']
      exact (Nat.choose_symm (le_of_lt hrN)).symm
    have hc : (N.choose r : ℝ) ≤ (ballCard N 2 r' : ℝ) := by
      rw [hsym]
      exact_mod_cast choose_le_ballCard
    have hpos : (0 : ℝ) < (N.choose r : ℝ) := by
      have : 0 < N.choose r := Nat.choose_pos (le_of_lt hrN)
      exact_mod_cast this
    have hlog := Real.log_le_log hpos hc
    rw [hqe] at hb
    have hfr : (r' : ℝ) / N = 1 - (r : ℝ) / N := by
      rw [hr']
      have : ((N - r : ℕ) : ℝ) = (N : ℝ) - r := by
        simpa using (Nat.cast_sub (le_of_lt hrN) : ((N - r : ℕ) : ℝ) = (N : ℝ) - (r : ℕ))
      rw [this]
      field_simp
    rw [hfr, h₂_symm] at hb
    linarith

end SeqEnt

end IDR
