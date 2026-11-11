/-
# Part CXLVIII  The certificates under measurement error, and how tight they can be at once

Part CXLVI turned a measured average of a disordered region into sharp population certificates.
Two things stand between such a statement and an experiment.  First, no average is measured
exactly, and the contour bound `B` used in the certificate is itself an estimate.  Second,
sharpness was proved *one threshold at a time*: an ensemble attaining the bound at a chosen
distance need not attain it anywhere else, and it matters whether a single ensemble could saturate
the whole family of certificates simultaneously — if one could, the certificates would jointly say
nothing more than each says alone.

* `certificate_under_mean_error` — **error bars propagate linearly and in the safe direction.**
  If the measured average is within `δ` of the truth, the certified population is only reduced by
  `δ/(B − a)`.  The certificate degrades continuously in the measurement error, so a one-number
  experiment with an error bar still proves a positive expanded population whenever the measured
  mean exceeds `a + δ`.

* `certificate_conservative_in_contour` — **overestimating the contour length is safe.**  The
  certificate computed with any bound `B'` at least as large as a true bound `B` remains valid, so
  a conservative estimate of the maximum extension can never produce a false claim: the residual
  modelling input of Part CXLVI is one-sided.

* `equality_forces_two_point_support` — **an ensemble that saturates the certificate at a threshold
  is pinned to two conformations.**  If equality holds at `a`, every populated conformation sits
  either exactly at `a` or exactly at the contour bound.

* `equality_at_two_thresholds_forces_extended` — **and it cannot saturate at two thresholds unless
  it is fully extended.**  Saturation at two distinct thresholds forces every conformation onto the
  contour bound, hence a mean equal to `B`.  Consequently (`strict_slack_at_some_threshold`) any
  region that is not fully extended has strict slack at all but at most one threshold: the extremal
  ensemble is threshold-specific, and reading the certificate at several thresholds at once is
  strictly more informative than reading it at any one of them.

Design consequence, closing Parts CXLVI–CXLVIII: report the certified population as a function of
threshold, with the measurement error folded in as `δ/(B − a)` and the contour bound taken
generously.  The resulting band is valid, degrades gracefully, and — because no single ensemble
saturates it at two thresholds — is strictly stronger than any of its individual values.
-/
import Mathlib
import RequestProject.PopulationCertificate

set_option autoImplicit false

namespace IDR
namespace RobustCertificate

open Finset IDR.PopulationCertificate

variable {N : ℕ}

/-- **The certificate under measurement error.**  If the reported average `muHat` is within `δ` of
the ensemble average, the certified population loses only `δ/(B − a)`. -/
theorem certificate_under_mean_error {w x : Fin N → ℝ} {B a muHat delta : ℝ}
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) (hx : ∀ j, x j ≤ B) (hab : a < B)
    (herr : |muHat - wmean w x| ≤ delta) :
    (muHat - delta - a) / (B - a) ≤ popGt w x a := by
  have hBa : (0:ℝ) < B - a := by linarith
  have h1 := reverse_markov_lower hw hsum hx hab
  have h2 : muHat - delta ≤ wmean w x := by
    have := abs_le.mp herr
    linarith [this.2]
  refine le_trans ?_ h1
  apply div_le_div_of_nonneg_right ?_ hBa.le
  linarith

/-- **Overestimating the maximum extension is safe.**  A certificate computed with a contour bound
`B'` at least as large as a valid bound remains valid. -/
theorem certificate_conservative_in_contour {w x : Fin N → ℝ} {B B' a : ℝ}
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) (hx : ∀ j, x j ≤ B) (hBB : B ≤ B') (hab : a < B) :
    (wmean w x - a) / (B' - a) ≤ popGt w x a :=
  reverse_markov_lower hw hsum (fun j => le_trans (hx j) hBB) (lt_of_lt_of_le hab hBB)

/-- The certificate weakens monotonically as the assumed contour bound grows. -/
theorem certificate_monotone_in_contour {mu B B' a : ℝ} (hab : a < B) (hBB : B ≤ B')
    (hmu : a ≤ mu) :
    (mu - a) / (B' - a) ≤ (mu - a) / (B - a) := by
  have hBa : (0:ℝ) < B - a := by linarith
  have hBa' : (0:ℝ) < B' - a := by linarith
  apply div_le_div_of_nonneg_left (by linarith) hBa
  linarith

/-- **Saturation pins the ensemble to two conformations.**  If the certificate holds with equality
at the threshold `a`, then every populated conformation lies either exactly at `a` or exactly at
the contour bound `B`. -/
theorem equality_forces_two_point_support {w x : Fin N → ℝ} {B a : ℝ}
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) (hx : ∀ j, x j ≤ B) (hab : a < B)
    (heq : popGt w x a = (wmean w x - a) / (B - a)) :
    ∀ j, 0 < w j → x j = a ∨ x j = B := by
  have hBa : (0:ℝ) < B - a := by linarith
  -- the pointwise inequality behind the certificate
  have hpt : ∀ j ∈ (Finset.univ : Finset (Fin N)),
      w j * x j ≤ a * w j + (B - a) * (if a < x j then w j else 0) := by
    intro j _
    by_cases h : a < x j
    · rw [if_pos h]
      nlinarith [hw j, hx j]
    · rw [if_neg h]
      push_neg at h
      nlinarith [hw j]
  -- under equality the two sums agree
  have hsums : ∑ j, w j * x j
      = ∑ j, (a * w j + (B - a) * (if a < x j then w j else 0)) := by
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hsum,
      show (∑ j, if a < x j then w j else 0) = popGt w x a from rfl, heq]
    rw [show (∑ j, w j * x j) = wmean w x from rfl]
    field_simp
    ring
  have hall := (Finset.sum_eq_sum_iff_of_le hpt).mp hsums
  intro j hwj
  have hj := hall j (Finset.mem_univ j)
  by_cases h : a < x j
  · right
    rw [if_pos h] at hj
    have : w j * (x j - B) = 0 := by linarith [hj]
    rcases mul_eq_zero.mp this with h1 | h2
    · exact absurd h1 (ne_of_gt hwj)
    · linarith
  · left
    rw [if_neg h] at hj
    have : w j * (x j - a) = 0 := by linarith [hj]
    rcases mul_eq_zero.mp this with h1 | h2
    · exact absurd h1 (ne_of_gt hwj)
    · linarith

/-- **The extremal ensemble is threshold-specific.**  A strictly populated ensemble that saturates
the certificate at two distinct thresholds below the contour bound has every conformation at the
contour bound, hence mean exactly `B`: it is the fully extended chain. -/
theorem equality_at_two_thresholds_forces_extended {w x : Fin N → ℝ} {B a1 a2 : ℝ}
    (hw : ∀ j, 0 < w j) (hsum : ∑ j, w j = 1) (hx : ∀ j, x j ≤ B) (h12 : a1 < a2) (h2B : a2 < B)
    (heq1 : popGt w x a1 = (wmean w x - a1) / (B - a1))
    (heq2 : popGt w x a2 = (wmean w x - a2) / (B - a2)) :
    (∀ j, x j = B) ∧ wmean w x = B := by
  have hw' : ∀ j, 0 ≤ w j := fun j => (hw j).le
  have h1 := equality_forces_two_point_support hw' hsum hx (lt_trans h12 h2B) heq1
  have h2 := equality_forces_two_point_support hw' hsum hx h2B heq2
  have hxB : ∀ j, x j = B := by
    intro j
    rcases h1 j (hw j) with e1 | e1
    · rcases h2 j (hw j) with e2 | e2
      · exact absurd (e1 ▸ e2 : a1 = a2) (ne_of_lt h12)
      · exact e2
    · exact e1
  refine ⟨hxB, ?_⟩
  rw [wmean]
  have : ∀ j ∈ (Finset.univ : Finset (Fin N)), w j * x j = B * w j := fun j _ => by
    rw [hxB j]; ring
  rw [Finset.sum_congr rfl this, ← Finset.mul_sum, hsum, mul_one]

/-- **Strict slack away from the fully extended chain.**  A strictly populated ensemble whose mean
falls short of the contour bound cannot saturate the certificate at two thresholds: at least one of
any two distinct thresholds carries a strict inequality, so reading the certificate at several
thresholds is strictly more informative than reading it at one. -/
theorem strict_slack_at_some_threshold {w x : Fin N → ℝ} {B a1 a2 : ℝ}
    (hw : ∀ j, 0 < w j) (hsum : ∑ j, w j = 1) (hx : ∀ j, x j ≤ B) (h12 : a1 < a2) (h2B : a2 < B)
    (hmean : wmean w x < B) :
    (wmean w x - a1) / (B - a1) < popGt w x a1 ∨ (wmean w x - a2) / (B - a2) < popGt w x a2 := by
  by_contra hc
  push_neg at hc
  obtain ⟨hc1, hc2⟩ := hc
  have hw' : ∀ j, 0 ≤ w j := fun j => (hw j).le
  have hge1 := reverse_markov_lower hw' hsum hx (lt_trans h12 h2B)
  have hge2 := reverse_markov_lower hw' hsum hx h2B
  have heq1 : popGt w x a1 = (wmean w x - a1) / (B - a1) := le_antisymm hc1 hge1
  have heq2 : popGt w x a2 = (wmean w x - a2) / (B - a2) := le_antisymm hc2 hge2
  have := (equality_at_two_thresholds_forces_extended hw hsum hx h12 h2B heq1 heq2).2
  linarith

end RobustCertificate
end IDR
