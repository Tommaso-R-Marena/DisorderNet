/-
# `k`-mer entropy: why single-residue complexity is the wrong screen

`ATATATATAT…` has the maximal possible single-residue entropy for a two-letter alphabet — one
bit per residue — and no complexity whatever.  Every low-complexity screen phrased on residue
*composition* passes it; every biologist rejects it on sight.  The discriminating statistic is
the `k`-mer (block) entropy, and for a repetitive tract it collapses like `log(period)/k`.

This file develops the empirical `k`-mer entropy of a single sequence and proves the collapse.

* `IDR.Kmer.kmerFreq`, `kmerEnt` — the empirical distribution of the `m` windows of width `k`
  of a sequence, and its Shannon entropy in nats.  `kmerFreq_sum_one` says it is a probability
  distribution; `kmerEnt_le_log_distinct` is the ceiling `H_k ≤ log (#distinct k-mers)`.
* `IDR.Kmer.kmerEnt_le_log_period` — **the repeat theorem**: a sequence of period `P` has at
  most `P` distinct `k`-mers for every `k`, hence `H_k ≤ log P` however long `k` is, hence a
  `k`-mer entropy *rate* `H_k/k ≤ log P / k` that tends to zero.
* `IDR.Kmer.alt_single_entropy` and `IDR.Kmer.alt_kmerEnt_le_log_two` — the two facts side by
  side for `ATATAT…`: single-residue entropy exactly `log 2`, the maximum for two letters, and
  `k`-mer entropy at most `log 2` for every `k`, so a rate of `log 2/k`.
  `IDR.Kmer.alt_rate_eventually_small` turns this into: the repeat's `k`-mer rate is below any
  positive threshold once `k` is large enough, while its composition entropy stays maximal.
* `IDR.Kmer.composition_blind` — the sharp statement that no composition-based statistic can do
  this job: `ATAT` and `AABB` have *identical* single-residue distributions (hence identical
  entropy, identical everything a composition model can see) while their 2-mer entropies are
  `log 2` and `log 4`.

Design consequence for the disorder model: the low-complexity criterion of
`RequestProject.SequenceEntropyLimit` — a floor on `H₁` — is necessary but not sufficient.  A
sequence must be screened on its `k`-mer entropy rate; repeats are invisible to `H₁`.
-/
import Mathlib
import RequestProject.SequenceEntropyCore

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR

namespace Kmer

open Finset
open SeqEnt

/-! ## Empirical `k`-mer statistics of one sequence -/

variable {A : Type*} [Fintype A] [DecidableEq A]

/-- The window of width `k` starting at position `i`. -/
def window (s : ℕ → A) (k i : ℕ) : Fin k → A := fun t => s (i + t)

/-- How many of the first `m` windows of width `k` equal the word `w`. -/
def kmerCount (s : ℕ → A) (k m : ℕ) (w : Fin k → A) : ℕ :=
  ((Finset.range m).filter fun i => window s k i = w).card

/-- The empirical `k`-mer distribution of a sequence, over its first `m` windows. -/
noncomputable def kmerFreq (s : ℕ → A) (k m : ℕ) : (Fin k → A) → ℝ :=
  fun w => (kmerCount s k m w : ℝ) / m

/-- The empirical `k`-mer entropy, in nats. -/
noncomputable def kmerEnt (s : ℕ → A) (k m : ℕ) : ℝ := H (kmerFreq s k m)

omit [Fintype A] in
lemma kmerFreq_nonneg (s : ℕ → A) (k m : ℕ) (w : Fin k → A) : 0 ≤ kmerFreq s k m w := by
  unfold kmerFreq; positivity

lemma sum_kmerCount (s : ℕ → A) (k m : ℕ) : ∑ w : Fin k → A, kmerCount s k m w = m := by
  classical
  have := Finset.card_eq_sum_card_fiberwise
    (f := fun i => window s k i) (s := Finset.range m)
    (t := (Finset.univ : Finset (Fin k → A))) (fun i _ => Finset.mem_univ _)
  simpa [kmerCount] using this.symm

lemma kmerFreq_sum_one (s : ℕ → A) (k : ℕ) {m : ℕ} (hm : 0 < m) :
    ∑ w : Fin k → A, kmerFreq s k m w = 1 := by
  have hm0 : (m : ℝ) ≠ 0 := by positivity
  unfold kmerFreq
  rw [← Finset.sum_div]
  rw [show ∑ w : Fin k → A, (kmerCount s k m w : ℝ) = ((∑ w : Fin k → A, kmerCount s k m w : ℕ) : ℝ)
    by push_cast; rfl, sum_kmerCount]
  field_simp

/-- The set of `k`-mers a sequence actually displays in its first `m` windows. -/
noncomputable def kmerSupport (s : ℕ → A) (k m : ℕ) : Finset (Fin k → A) :=
  (Finset.range m).image (fun i => window s k i)

omit [Fintype A] in
lemma kmerFreq_eq_zero_of_notMem {s : ℕ → A} {k m : ℕ} {w : Fin k → A}
    (hw : w ∉ kmerSupport s k m) : kmerFreq s k m w = 0 := by
  classical
  have hcount : kmerCount s k m w = 0 := by
    rw [kmerCount, Finset.card_eq_zero]
    by_contra hne
    obtain ⟨i, hi⟩ := Finset.nonempty_iff_ne_empty.2 hne
    rw [Finset.mem_filter] at hi
    exact hw (Finset.mem_image.2 ⟨i, hi.1, hi.2⟩)
  unfold kmerFreq
  rw [hcount]
  simp

omit [Fintype A] in
lemma kmerSupport_nonempty {s : ℕ → A} {k m : ℕ} (hm : 0 < m) :
    (kmerSupport s k m).Nonempty :=
  ⟨window s k 0, Finset.mem_image.2 ⟨0, Finset.mem_range.2 hm, rfl⟩⟩

/-- **The `k`-mer entropy is capped by the number of distinct `k`-mers.** -/
theorem kmerEnt_le_log_distinct (s : ℕ → A) (k : ℕ) {m : ℕ} (hm : 0 < m) :
    kmerEnt s k m ≤ Real.log (kmerSupport s k m).card :=
  H_le_log_card (kmerFreq_nonneg s k m) (kmerFreq_sum_one s k hm)
    (fun _ hw => kmerFreq_eq_zero_of_notMem hw) (kmerSupport_nonempty hm)

/-! ## Repeats: a period-`P` sequence has at most `P` distinct `k`-mers -/

/-- A sequence of period `P`. -/
def Periodic (s : ℕ → A) (P : ℕ) : Prop := ∀ i, s (i + P) = s i

omit [Fintype A] [DecidableEq A] in
lemma periodic_add_mul {s : ℕ → A} {P : ℕ} (hper : Periodic s P) (i n : ℕ) :
    s (i + P * n) = s i := by
  induction n with
  | zero => simp
  | succ n ih =>
      have hrw : i + P * (n + 1) = (i + P * n) + P := by ring
      rw [hrw, hper, ih]

omit [Fintype A] [DecidableEq A] in
lemma window_mod {s : ℕ → A} {P : ℕ} (hper : Periodic s P) (k i : ℕ) :
    window s k i = window s k (i % P) := by
  funext t
  have hsplit : i + (t : ℕ) = (i % P + t) + P * (i / P) := by
    have := Nat.div_add_mod i P
    omega
  show s (i + t) = s (i % P + t)
  rw [hsplit, periodic_add_mul hper]

omit [Fintype A] in
/-- A periodic sequence displays no more `k`-mers than its period. -/
theorem card_kmerSupport_le_period {s : ℕ → A} {P : ℕ} (hper : Periodic s P) (hP : 0 < P)
    (k m : ℕ) : (kmerSupport s k m).card ≤ P := by
  classical
  have hsub : kmerSupport s k m ⊆ (Finset.range P).image (fun i => window s k i) := by
    intro w hw
    obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hw
    exact Finset.mem_image.2 ⟨i % P, Finset.mem_range.2 (Nat.mod_lt _ hP),
      (window_mod hper k i).symm⟩
  calc (kmerSupport s k m).card ≤ ((Finset.range P).image (fun i => window s k i)).card :=
        Finset.card_le_card hsub
    _ ≤ (Finset.range P).card := Finset.card_image_le
    _ = P := Finset.card_range P

/-- **The repeat theorem.**  A sequence of period `P` has `k`-mer entropy at most `log P` for
*every* window width `k`: its `k`-mer entropy rate `H_k/k` decays like `log P / k`, however
high its single-residue entropy is. -/
theorem kmerEnt_le_log_period {s : ℕ → A} {P : ℕ} (hper : Periodic s P) (hP : 0 < P)
    (k : ℕ) {m : ℕ} (hm : 0 < m) : kmerEnt s k m ≤ Real.log P := by
  refine le_trans (kmerEnt_le_log_distinct s k hm) ?_
  have hcard := card_kmerSupport_le_period hper hP k m
  have hpos : (0 : ℝ) < (kmerSupport s k m).card := by
    exact_mod_cast Finset.card_pos.2 (kmerSupport_nonempty (s := s) (k := k) hm)
  exact Real.log_le_log hpos (by exact_mod_cast hcard)

/-- The rate form. -/
theorem kmer_rate_le {s : ℕ → A} {P : ℕ} (hper : Periodic s P) (hP : 0 < P)
    {k m : ℕ} (hk : 0 < k) (hm : 0 < m) :
    kmerEnt s k m / k ≤ Real.log P / k := by
  have hk' : (0 : ℝ) < k := by exact_mod_cast hk
  exact (div_le_div_iff_of_pos_right hk').mpr (kmerEnt_le_log_period hper hP k hm)

/-! ## `ATATAT…`: maximal composition entropy, vanishing `k`-mer rate -/

/-- The alternating two-letter sequence. -/
def alt : ℕ → Bool := fun i => decide (i % 2 = 0)

lemma alt_periodic : Periodic alt 2 := by
  intro i
  simp [alt, Nat.add_mod_right]

lemma alt_count (b : Bool) (r : ℕ) :
    ((Finset.range (2 * r)).filter fun i => alt i = b).card = r := by
  induction r with
  | zero => simp
  | succ r ih =>
      have h : 2 * (r + 1) = (2 * r + 1) + 1 := by ring
      rw [h, Finset.range_add_one, Finset.filter_insert, Finset.range_add_one,
        Finset.filter_insert]
      have h1 : alt (2 * r) = true := by simp [alt, Nat.mul_mod_right]
      have h2 : alt (2 * r + 1) = false := by simp [alt, Nat.add_mod, Nat.mul_mod_right]
      cases b with
      | true =>
          rw [if_neg (by simp [h2]), if_pos (by simp [h1]),
            Finset.card_insert_of_notMem (by simp)]
          omega
      | false =>
          rw [if_pos (by simp [h2]), if_neg (by simp [h1]),
            Finset.card_insert_of_notMem (by simp)]
          omega

lemma alt_kmerFreq_one {r : ℕ} (hr : 0 < r) :
    kmerFreq alt 1 (2 * r) = fun w => if w ∈ (Finset.univ : Finset (Fin 1 → Bool))
      then (((Finset.univ : Finset (Fin 1 → Bool)).card : ℝ))⁻¹ else 0 := by
  have hcard : (Finset.univ : Finset (Fin 1 → Bool)).card = 2 := by decide
  funext w
  have hcount : kmerCount alt 1 (2 * r) w = r := by
    have hfil : ((Finset.range (2 * r)).filter fun i => window alt 1 i = w)
        = ((Finset.range (2 * r)).filter fun i => alt i = w 0) := by
      refine Finset.filter_congr fun i _ => ?_
      constructor
      · intro h; rw [← h]; simp [window]
      · intro h
        funext t
        have ht : t = 0 := Subsingleton.elim _ _
        subst ht
        simpa [window] using h
    rw [kmerCount, hfil, alt_count (w 0) r]
  have hrpos : (0 : ℝ) < r := by exact_mod_cast hr
  rw [kmerFreq, hcount, hcard]
  simp only [Finset.mem_univ, if_true]
  push_cast
  field_simp

/-- **The composition entropy of a perfect repeat is maximal.**  `ATATAT…` has single-residue
entropy exactly `log 2`, the largest value a two-letter alphabet allows. -/
theorem alt_single_entropy {r : ℕ} (hr : 0 < r) : kmerEnt alt 1 (2 * r) = Real.log 2 := by
  have hcard : (Finset.univ : Finset (Fin 1 → Bool)).card = 2 := by decide
  rw [kmerEnt, alt_kmerFreq_one hr, H_uniform (F := (Finset.univ : Finset (Fin 1 → Bool)))
    ⟨fun _ => true, Finset.mem_univ _⟩, hcard]
  norm_num

/-- **…and its `k`-mer entropy is stuck at `log 2` for every `k`.** -/
theorem alt_kmerEnt_le_log_two (k : ℕ) {m : ℕ} (hm : 0 < m) :
    kmerEnt alt k m ≤ Real.log 2 := by
  have := kmerEnt_le_log_period alt_periodic (by norm_num) k hm
  simpa using this

/-- **The screen that works.**  However small a per-residue complexity threshold `eps > 0` is,
the repeat's `k`-mer entropy rate falls below it once `k > log 2 / eps` — while its
single-residue entropy stays at the maximum `log 2`.  Composition and `k`-mer complexity are
genuinely different measurements. -/
theorem alt_rate_eventually_small {eps : ℝ} (heps : 0 < eps) :
    ∃ k0 : ℕ, 0 < k0 ∧ ∀ k ≥ k0, ∀ m : ℕ, 0 < m → kmerEnt alt k m / k < eps := by
  obtain ⟨k0, hk0⟩ := exists_nat_gt (Real.log 2 / eps)
  refine ⟨max k0 1, lt_of_lt_of_le Nat.zero_lt_one (le_max_right _ _), fun k hk m hm => ?_⟩
  have hk1 : 1 ≤ k := le_trans (le_max_right k0 1) hk
  have hkk0 : (k0 : ℝ) ≤ (k : ℝ) := by
    exact_mod_cast le_trans (le_max_left k0 1) hk
  have hkpos : (0 : ℝ) < k := by exact_mod_cast hk1
  have hlt : Real.log 2 / eps < (k : ℝ) := lt_of_lt_of_le hk0 hkk0
  have hbound : kmerEnt alt k m ≤ Real.log 2 := alt_kmerEnt_le_log_two k hm
  have hlog : Real.log 2 < eps * k := by
    rw [div_lt_iff₀ heps] at hlt
    linarith
  calc kmerEnt alt k m / k ≤ Real.log 2 / k := by
        exact (div_le_div_iff_of_pos_right hkpos).mpr hbound
    _ < eps := by rw [div_lt_iff₀ hkpos]; linarith

/-! ## No composition statistic can see the difference -/

/-- A period-4 block sequence, `AABBAABB…`. -/
def blk : ℕ → Bool := fun i => decide (i % 4 < 2)

/-- **The single-residue distributions of `ATAT` and `AABB` are identical.**  Every statistic
that a composition model computes — entropy included — is therefore the same for the two. -/
theorem composition_blind_one : kmerFreq alt 1 4 = kmerFreq blk 1 4 := by
  funext w
  have hc : kmerCount alt 1 4 w = kmerCount blk 1 4 w := by revert w; decide
  rw [kmerFreq, kmerFreq, hc]

/-- **But their 2-mer entropies differ by `log 2`.**  `ATAT` shows two dimers, `AABB` shows
four: `H₂ = log 2` against `H₂ = log 4`.  A model that scores sequences on composition alone
cannot represent this difference, which is exactly the difference between a repeat and a
non-repetitive tract. -/
theorem composition_blind_two :
    kmerEnt alt 2 4 = Real.log 2 ∧ kmerEnt blk 2 4 = Real.log 4 := by
  constructor
  · -- `ATAT` displays exactly the two dimers `AB` and `BA`, each twice
    set F : Finset (Fin 2 → Bool) :=
      (Finset.univ : Finset (Fin 2 → Bool)).filter (fun w => w 0 ≠ w 1) with hF
    have hFcard : F.card = 2 := by rw [hF]; decide
    have hFne : F.Nonempty := by rw [← Finset.card_pos, hFcard]; norm_num
    have hfreq : kmerFreq alt 2 4 = fun w => if w ∈ F then ((F.card : ℝ))⁻¹ else 0 := by
      funext w
      have hcount : kmerCount alt 2 4 w = if w ∈ F then 2 else 0 := by
        rw [hF]; revert w; decide
      rw [kmerFreq, hcount, hFcard]
      by_cases hw : w ∈ F
      · simp only [hw, if_true]; norm_num
      · simp [hw]
    rw [kmerEnt, hfreq, H_uniform hFne, hFcard]
    norm_num
  · -- `AABB` displays all four dimers, once each
    have hfreq : kmerFreq blk 2 4 = fun w => if w ∈ (Finset.univ : Finset (Fin 2 → Bool))
        then (((Finset.univ : Finset (Fin 2 → Bool)).card : ℝ))⁻¹ else 0 := by
      have hcard : (Finset.univ : Finset (Fin 2 → Bool)).card = 4 := by decide
      funext w
      have hcount : kmerCount blk 2 4 w = 1 := by revert w; decide
      rw [kmerFreq, hcount, hcard]
      simp
    have hcard : (Finset.univ : Finset (Fin 2 → Bool)).card = 4 := by decide
    rw [kmerEnt, hfreq, H_uniform (F := (Finset.univ : Finset (Fin 2 → Bool)))
      ⟨fun _ => true, Finset.mem_univ _⟩, hcard]
    norm_num

end Kmer

end IDR
