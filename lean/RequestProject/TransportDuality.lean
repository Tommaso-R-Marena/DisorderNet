/-
# What a measured observable proves about a candidate ensemble, in ångströms

`RequestProject.TransportGeometry` makes the structural transport cost a genuine distance
between conformational ensembles.  This file turns that distance into an *experimental*
quantity, in both directions.

* `expect_diff_le_of_lipschitz` (**weak Kantorovich duality**, the accuracy-transfer law):
  if an observable `f` is `L`-Lipschitz with respect to the structural metric -- radius of
  gyration, end-to-end distance, a contact fraction, a FRET efficiency -- then a model
  whose ensemble is within `ε` of the truth in transport distance predicts the average of
  `f` to within `L·ε`.  One geometric error bound controls *every* Lipschitz observable at
  once, which is exactly what a "model of the disordered region" is expected to deliver.
* `transportCost_ge_of_observable` (the **falsification certificate**): read backwards, a
  measured discrepancy of `Δ` in the average of an `L`-Lipschitz observable proves that the
  candidate ensemble is at least `Δ/L` away from the truth in transport distance -- a
  model-free lower bound on model error, stated in structural units, with no assumption
  whatsoever about the shape of either ensemble.
* `transportCost_pairEns` computes the transport distance between two two-state ensembles
  exactly (`|p - q| · c u v`), and `certificate_is_sharp` uses it to show that the
  certificate above is *attained*: there is nothing conservative about it.

The mathematical content is the easy half of Kantorovich duality plus an exactly solvable
case that shows the easy half is already sharp.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Transport
import RequestProject.TransportGeometry

namespace IDR

open Finset
open scoped Classical

variable {X : Type*}

/-! ## Weak duality: transport distance controls every Lipschitz observable -/

/-- **Accuracy transfer.**  If the observable `f` varies by at most `L` times the structural
distance, then two ensembles at transport distance `ε` disagree on the average of `f` by at
most `L·ε`.  A single geometric error budget therefore certifies every Lipschitz observable
simultaneously. -/
theorem expect_diff_le_of_lipschitz {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) {L : ℝ} {f : X → ℝ} (hf : ∀ x y, |f x - f y| ≤ L * c x y) (E F : Ens X) :
    |E.expect f - F.expect f| ≤ L * transportCost c E F := by
  obtain ⟨g, hg, hgc⟩ := exists_optimal_coupling hc E F
  have h1 : E.expect f = ∑ i, ∑ j, g i j * f (E.pt i) := by
    simp only [Ens.expect]
    exact Finset.sum_congr rfl fun i _ => by rw [← Finset.sum_mul, hg.row i]
  have h2 : F.expect f = ∑ i, ∑ j, g i j * f (F.pt j) := by
    simp only [Ens.expect]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun j _ => by rw [← Finset.sum_mul, hg.col j]
  have hdiff : E.expect f - F.expect f = ∑ i, ∑ j, g i j * (f (E.pt i) - f (F.pt j)) := by
    rw [h1, h2, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [hdiff, hgc]
  calc |∑ i, ∑ j, g i j * (f (E.pt i) - f (F.pt j))|
      ≤ ∑ i, |∑ j, g i j * (f (E.pt i) - f (F.pt j))| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, ∑ j, |g i j * (f (E.pt i) - f (F.pt j))| :=
        Finset.sum_le_sum fun i _ => Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, ∑ j, g i j * (L * c (E.pt i) (F.pt j)) := by
        refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
        rw [abs_mul, abs_of_nonneg (hg.nonneg i j)]
        exact mul_le_mul_of_nonneg_left (hf _ _) (hg.nonneg i j)
    _ = L * planCost E F c g := by
        rw [planCost, Finset.mul_sum]
        exact Finset.sum_congr rfl fun i _ => by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun j _ => by ring

/-- **The falsification certificate.**  A measured discrepancy in the ensemble average of an
`L`-Lipschitz observable is a lower bound, in structural units, on the transport distance
between the candidate ensemble and the true one.  No modelling assumption enters. -/
theorem transportCost_ge_of_observable {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) {L : ℝ}
    (hL : 0 < L) {f : X → ℝ} (hf : ∀ x y, |f x - f y| ≤ L * c x y) (E F : Ens X) :
    |E.expect f - F.expect f| / L ≤ transportCost c E F :=
  (div_le_iff₀ hL).2 <| by
    rw [mul_comm]
    exact expect_diff_le_of_lipschitz hc hf E F

/-- The distance to a fixed reference structure is `1`-Lipschitz -- the dual test function
that makes the certificate sharp. -/
lemma lipschitz_dist_to {c : X → X → ℝ} (hsymm : ∀ x y, c x y = c y x)
    (htri : ∀ x y z, c x z ≤ c x y + c y z) (u : X) (x y : X) :
    |c u x - c u y| ≤ 1 * c x y := by
  rw [one_mul, abs_sub_le_iff]
  constructor
  · have := htri u y x
    rw [hsymm y x] at this
    linarith
  · have := htri u x y
    linarith

/-! ## An exactly solvable case: two-state ensembles -/

/-- The two-state ensemble: conformation `u` with probability `1 - p` and conformation `v`
with probability `p`.  The minimal caricature of a disordered region that interconverts
between a compact and an expanded form. -/
noncomputable def pairEns (u v : X) (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) : Ens X where
  card := 2
  pt := ![u, v]
  w := ![1 - p, p]
  w_nonneg := by
    intro j
    fin_cases j
    · simpa using hp1
    · simpa using hp0
  w_sum := by simp

@[simp] lemma pairEns_expect (u v : X) (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (f : X → ℝ) :
    (pairEns u v p hp0 hp1).expect f = (1 - p) * f u + p * f v := by
  simp [Ens.expect, pairEns, Fin.sum_univ_two]

lemma transportCost_pairEns_le {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) (hcd : ∀ x, c x x = 0)
    (u v : X) {p q : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hpq : p ≤ q) :
    transportCost c (pairEns u v p hp0 hp1) (pairEns u v q hq0 hq1) ≤ (q - p) * c u v := by
  have hcoup : IsCoupling (pairEns u v p hp0 hp1) (pairEns u v q hq0 hq1)
      (fun i j => ![![1 - q, q - p], ![0, p]] i j) := by
    refine ⟨fun i j => ?_, fun i => ?_, fun j => ?_⟩
    · fin_cases i <;> fin_cases j <;> simp <;> linarith
    · fin_cases i <;> simp only [pairEns] <;> simp [Fin.sum_univ_two]
    · fin_cases j <;> simp only [pairEns] <;> simp [Fin.sum_univ_two]
  refine (transportCost_le_of_coupling hc hcoup).trans_eq ?_
  simp [planCost, Fin.sum_univ_two, pairEns, hcd]

/-- **The transport distance between two two-state ensembles is exactly the population
error times the structural separation.**  Getting the compact/expanded populations wrong by
`|p - q|` costs `|p - q|` times the structural distance between the two states -- no more
and no less. -/
theorem transportCost_pairEns {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) (hcd : ∀ x, c x x = 0)
    (hsymm : ∀ x y, c x y = c y x) (htri : ∀ x y z, c x z ≤ c x y + c y z)
    (u v : X) {p q : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) :
    transportCost c (pairEns u v p hp0 hp1) (pairEns u v q hq0 hq1) = |p - q| * c u v := by
  set E := pairEns u v p hp0 hp1 with hE
  set F := pairEns u v q hq0 hq1 with hF
  have hlower : |p - q| * c u v ≤ transportCost c E F := by
    have hlip := lipschitz_dist_to hsymm htri u
    have := expect_diff_le_of_lipschitz hc hlip E F
    rw [one_mul] at this
    refine le_trans (le_of_eq ?_) this
    rw [hE, hF, pairEns_expect, pairEns_expect, hcd u]
    have : (1 - p) * 0 + p * c u v - ((1 - q) * 0 + q * c u v) = (p - q) * c u v := by ring
    rw [this, abs_mul, abs_of_nonneg (hc u v)]
  refine le_antisymm ?_ hlower
  rcases le_total p q with h | h
  · rw [abs_of_nonpos (by linarith), neg_sub]
    exact transportCost_pairEns_le hc hcd u v hp0 hp1 hq0 hq1 h
  · rw [abs_of_nonneg (by linarith), transportCost_comm hc hsymm]
    exact transportCost_pairEns_le hc hcd u v hq0 hq1 hp0 hp1 h

/-- **The certificate is sharp.**  For a model that gets the two populations of a two-state
disordered region wrong, the lower bound proved by the single measured observable
"distance from the compact state" equals the true transport distance exactly: the
experiment leaves no slack. -/
theorem certificate_is_sharp {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y) (hcd : ∀ x, c x x = 0)
    (hsymm : ∀ x y, c x y = c y x) (htri : ∀ x y z, c x z ≤ c x y + c y z)
    (u v : X) {p q : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) :
    |(pairEns u v p hp0 hp1).expect (c u) - (pairEns u v q hq0 hq1).expect (c u)| / 1
      = transportCost c (pairEns u v p hp0 hp1) (pairEns u v q hq0 hq1) := by
  rw [transportCost_pairEns hc hcd hsymm htri, pairEns_expect, pairEns_expect, hcd u, div_one]
  have : (1 - p) * 0 + p * c u v - ((1 - q) * 0 + q * c u v) = (p - q) * c u v := by ring
  rw [this, abs_mul, abs_of_nonneg (hc u v)]

end IDR
