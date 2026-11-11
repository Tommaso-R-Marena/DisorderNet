/-
# The imbalance-corrected pair bound: a closed form with no balance hypothesis

`RequestProject.DiscordantPairs` proves the quadratic gain of the pairwise protocol under a
*balance* hypothesis: as many residues both labellings call disordered as both call ordered
(`BalancedClasses`).  On real annotation the two agreement classes are never exactly equal, so the
published bound `ν_pair ≤ 2 ε²` has hypotheses that a given benchmark may simply fail.

This file removes the hypothesis.  Write

* `a = |T ∩ L|` — residues both labellings call disordered,
* `e = |(T ∪ L)ᶜ|` — residues both call ordered,
* `d = |T \ L|`, `u = |L \ T|` — the two flip classes, `ν = d + u = noise T L`,
* `ε = ν / n` — the residue noise rate, `n = |α|`.

and define the **imbalance factor**

    κ = (a + e)² / (4 a e)   (`kappa`),

which is `1` exactly when the classes balance (`kappa_eq_one_iff_balanced`) and larger otherwise
(`one_le_kappa`).  Then, with no hypothesis beyond both agreement classes being nonempty,

* `nuPair_le_imbalanced` — **`ν_pair ≤ κ · ε² / (1 − ε)²`.**

The proof is the identity `ν_pair = du/(ae + du)`, the AM–GM step `4du ≤ ν²` already used for the
balanced bound, and the substitution `a + e = n(1 − ε)` coming from `card_split`.

The published bound becomes a corollary rather than a separate result: whenever
`κ ≤ 2(1 − ε)²` — in particular for balanced classes at a noise rate at most `1/4`, where `κ = 1`
and `2(1 − ε)² ≥ 9/8` — one recovers `ν_pair ≤ 2 ε²` (`nuPair_le_two_eps_sq_of_kappa`,
`nuPair_le_two_eps_sq_of_balanced'`).  Unlike the balanced statement, the corrected bound is a
closed form the benchmark can always evaluate: `a`, `e` and `ν` are all counted from the data.
-/
import Mathlib
import RequestProject.DiscordantPairs

set_option autoImplicit false

namespace IDR
namespace Discordant

open Finset
open IDR.LabelNoise

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The **imbalance factor** of an annotation: `κ = (a + e)² / (4 a e)`, where `a = |T ∩ L|` and
`e = |(T ∪ L)ᶜ|` are the two agreement classes.  It is `1` exactly when they balance. -/
noncomputable def kappa (T L : Finset α) : ℝ :=
  (((T ∩ L).card : ℝ) + ((T ∪ L)ᶜ).card) ^ 2 / (4 * ((T ∩ L).card : ℝ) * ((T ∪ L)ᶜ).card)

/-- The residue noise rate `ε = ν / n`. -/
noncomputable def epsRate (T L : Finset α) : ℝ := (noise T L : ℝ) / Fintype.card α

/-- The imbalance factor is at least one. -/
lemma one_le_kappa {T L : Finset α} (ha : 0 < (T ∩ L).card) (he : 0 < ((T ∪ L)ᶜ).card) :
    1 ≤ kappa T L := by
  have haR : (0:ℝ) < ((T ∩ L).card : ℝ) := by exact_mod_cast ha
  have heR : (0:ℝ) < (((T ∪ L)ᶜ).card : ℝ) := by exact_mod_cast he
  have hsq := sq_nonneg (((T ∩ L).card : ℝ) - (((T ∪ L)ᶜ).card : ℝ))
  rw [kappa, le_div_iff₀ (by positivity)]
  nlinarith

/-- The imbalance factor is one exactly for balanced classes. -/
lemma kappa_eq_one_iff_balanced {T L : Finset α} (ha : 0 < (T ∩ L).card)
    (he : 0 < ((T ∪ L)ᶜ).card) :
    kappa T L = 1 ↔ BalancedClasses T L := by
  have haR : (0:ℝ) < ((T ∩ L).card : ℝ) := by exact_mod_cast ha
  have heR : (0:ℝ) < (((T ∪ L)ᶜ).card : ℝ) := by exact_mod_cast he
  have hsq := sq_nonneg (((T ∩ L).card : ℝ) - (((T ∪ L)ᶜ).card : ℝ))
  rw [kappa, div_eq_one_iff_eq (by positivity)]
  unfold BalancedClasses
  constructor
  · intro h
    have hR : ((T ∩ L).card : ℝ) = (((T ∪ L)ᶜ).card : ℝ) := by nlinarith
    exact_mod_cast hR
  · intro h
    have hR : ((T ∩ L).card : ℝ) = (((T ∪ L)ᶜ).card : ℝ) := by exact_mod_cast h
    rw [hR]; ring

/-- Balanced classes have imbalance factor one. -/
lemma kappa_of_balanced {T L : Finset α} (ha : 0 < (T ∩ L).card) (he : 0 < ((T ∪ L)ᶜ).card)
    (hbal : BalancedClasses T L) : kappa T L = 1 :=
  (kappa_eq_one_iff_balanced ha he).2 hbal

/-- The arithmetic behind the corrected bound, on real numbers: with `v = d + u` the noise,
`N = a + e + v` the total, and both agreement classes positive,

    2du / (2(ae + du)) ≤ ((a+e)²/(4ae)) · (v/N)² / (1 − v/N)². -/
private lemma imbalanced_real {a e d u v N : ℝ} (ha : 0 < a) (he : 0 < e) (hd : 0 ≤ d)
    (hu : 0 ≤ u) (hv : v = d + u) (hN : N = a + e + v) :
    2 * (d * u) / (2 * (a * e + d * u)) ≤ (a + e) ^ 2 / (4 * a * e) * (v / N) ^ 2 /
      (1 - v / N) ^ 2 := by
  have hae : 0 < a + e := by linarith
  have hvnn : 0 ≤ v := by rw [hv]; linarith
  have hNpos : 0 < N := by rw [hN]; linarith
  have hone : 1 - v / N = (a + e) / N := by
    rw [hN]; field_simp; ring
  have hrhs : (a + e) ^ 2 / (4 * a * e) * (v / N) ^ 2 / (1 - v / N) ^ 2 = v ^ 2 / (4 * (a * e)) := by
    rw [hone]
    field_simp
  have hlhs : 2 * (d * u) / (2 * (a * e + d * u)) = d * u / (a * e + d * u) := by
    rw [mul_div_mul_left _ _ (by norm_num : (2:ℝ) ≠ 0)]
  have h4du : 4 * (d * u) ≤ v ^ 2 := by
    rw [hv]; nlinarith [sq_nonneg (d - u)]
  rw [hlhs, hrhs, div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_nonneg hd hu, mul_pos ha he, mul_nonneg (mul_nonneg hd hu) (mul_nonneg hd hu)]

/-- **The imbalance-corrected bound.**  With both agreement classes nonempty — and *no* balance
hypothesis, and no bound on the noise rate — the pair-level error rate obeys

    ν_pair ≤ κ · ε² / (1 − ε)²,

with `κ = (a + e)²/(4ae)` the imbalance factor and `ε = ν/n` the residue noise rate.  For balanced
classes `κ = 1` and the factor disappears. -/
theorem nuPair_le_imbalanced {T L : Finset α} (ha : 0 < (T ∩ L).card)
    (he : 0 < ((T ∪ L)ᶜ).card) :
    nuPair T L ≤ kappa T L * epsRate T L ^ 2 / (1 - epsRate T L) ^ 2 := by
  classical
  have hsplit : Fintype.card α = (T ∩ L).card + ((T ∪ L)ᶜ).card + noise T L := card_split T L
  have hvdu : noise T L = (T \ L).card + (L \ T).card := noise_eq_flip_add T L
  have hcard : ((comparablePairs T L).card : ℝ)
      = 2 * (((T ∩ L).card : ℝ) * ((T ∪ L)ᶜ).card + ((T \ L).card : ℝ) * ((L \ T).card : ℝ)) := by
    rw [card_comparablePairs]; push_cast; ring
  have hdisc : ((discordantPairs T L : ℕ) : ℝ)
      = 2 * (((T \ L).card : ℝ) * ((L \ T).card : ℝ)) := by
    rw [discordant_eq_flip_product]; push_cast; ring
  rw [nuPair, kappa, epsRate, hcard, hdisc]
  refine imbalanced_real (by exact_mod_cast ha) (by exact_mod_cast he) (by positivity)
    (by positivity) ?_ ?_
  · exact_mod_cast hvdu
  · exact_mod_cast hsplit

/-- Both agreement classes nonempty forces `ε < 1`; indeed `1 − ε = (a + e)/n`. -/
lemma one_sub_epsRate_eq {T L : Finset α} (ha : 0 < (T ∩ L).card) (he : 0 < ((T ∪ L)ᶜ).card) :
    1 - epsRate T L = (((T ∩ L).card : ℝ) + ((T ∪ L)ᶜ).card) / Fintype.card α := by
  have hsplit : Fintype.card α = (T ∩ L).card + ((T ∪ L)ᶜ).card + noise T L := card_split T L
  have hnpos : 0 < Fintype.card α := by omega
  have hnR : (0:ℝ) < (Fintype.card α : ℝ) := by exact_mod_cast hnpos
  have hvR : ((noise T L : ℕ) : ℝ)
      = (Fintype.card α : ℝ) - (((T ∩ L).card : ℝ) + ((T ∪ L)ᶜ).card) := by
    have : (Fintype.card α : ℝ) = ((T ∩ L).card : ℝ) + ((T ∪ L)ᶜ).card + (noise T L : ℕ) := by
      exact_mod_cast hsplit
    linarith
  rw [epsRate, hvR]
  field_simp
  ring

lemma epsRate_lt_one {T L : Finset α} (ha : 0 < (T ∩ L).card) (he : 0 < ((T ∪ L)ᶜ).card) :
    epsRate T L < 1 := by
  have hsplit : Fintype.card α = (T ∩ L).card + ((T ∪ L)ᶜ).card + noise T L := card_split T L
  have hnpos : 0 < Fintype.card α := by omega
  have hnR : (0:ℝ) < (Fintype.card α : ℝ) := by exact_mod_cast hnpos
  have haR : (0:ℝ) < ((T ∩ L).card : ℝ) := by exact_mod_cast ha
  have heR : (0:ℝ) < (((T ∪ L)ᶜ).card : ℝ) := by exact_mod_cast he
  have h := one_sub_epsRate_eq ha he
  have : 0 < 1 - epsRate T L := by rw [h]; positivity
  linarith

/-- **The published bound is a corollary.**  Whenever the imbalance factor is at most
`2 (1 − ε)²` — for balanced classes and a noise rate at most `1/4` it is `1 ≤ 9/8` — the corrected
bound gives back `ν_pair ≤ 2 ε²`. -/
theorem nuPair_le_two_eps_sq_of_kappa {T L : Finset α} (ha : 0 < (T ∩ L).card)
    (he : 0 < ((T ∪ L)ᶜ).card) (hk : kappa T L ≤ 2 * (1 - epsRate T L) ^ 2) :
    nuPair T L ≤ 2 * epsRate T L ^ 2 := by
  have hlt : epsRate T L < 1 := epsRate_lt_one ha he
  have hne : (0:ℝ) < 1 - epsRate T L := by linarith
  refine (nuPair_le_imbalanced ha he).trans ?_
  rw [div_le_iff₀ (by positivity)]
  nlinarith [sq_nonneg (epsRate T L), pow_pos hne 2]

/-- The balanced case, derived from the corrected bound: at a noise rate of at most `1/4` and with
balanced classes, `ν_pair ≤ 2 ε²`. -/
theorem nuPair_le_two_eps_sq_of_balanced' {T L : Finset α} (ha : 0 < (T ∩ L).card)
    (he : 0 < ((T ∪ L)ᶜ).card) (hbal : BalancedClasses T L)
    (h4 : 4 * noise T L ≤ Fintype.card α) :
    nuPair T L ≤ 2 * epsRate T L ^ 2 := by
  refine nuPair_le_two_eps_sq_of_kappa ha he ?_
  rw [kappa_of_balanced ha he hbal]
  have hsplit : Fintype.card α = (T ∩ L).card + ((T ∪ L)ᶜ).card + noise T L := card_split T L
  have hnpos : 0 < Fintype.card α := by omega
  have hnR : (0:ℝ) < (Fintype.card α : ℝ) := by exact_mod_cast hnpos
  have h4R : 4 * ((noise T L : ℕ) : ℝ) ≤ (Fintype.card α : ℝ) := by exact_mod_cast h4
  have hle : epsRate T L ≤ 1 / 4 := by
    rw [epsRate, div_le_div_iff₀ hnR (by norm_num)]
    linarith
  have hnn : 0 ≤ epsRate T L := by
    rw [epsRate]; positivity
  nlinarith

/-! ## The corrected bound on the worked instance -/

/-- On the sixteen-residue instance of `RequestProject.DiscordantPairs` the classes are balanced,
so the imbalance factor is one and the corrected bound reads `1/50 ≤ (1/8)²/(7/8)²`. -/
theorem imbalanced_example :
    kappa exampleTruth exampleAnnot = 1 ∧
      epsRate exampleTruth exampleAnnot = 1 / 8 ∧
      nuPair exampleTruth exampleAnnot
        ≤ kappa exampleTruth exampleAnnot * epsRate exampleTruth exampleAnnot ^ 2 /
            (1 - epsRate exampleTruth exampleAnnot) ^ 2 := by
  have ha : 0 < (exampleTruth ∩ exampleAnnot).card := by decide
  have he : 0 < ((exampleTruth ∪ exampleAnnot)ᶜ).card := by decide
  refine ⟨kappa_of_balanced ha he (by unfold BalancedClasses; decide), ?_,
    nuPair_le_imbalanced ha he⟩
  rw [epsRate, show noise exampleTruth exampleAnnot = 2 from exampleCounts.1]
  norm_num

end Discordant
end IDR
