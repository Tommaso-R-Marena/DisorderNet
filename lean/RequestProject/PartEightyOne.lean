/-
# Part LXXXI  Ensemble reweighting as convex duality

`RequestProject.Dual` studies the objective that is actually minimised when an ensemble of a
disordered region is refined against experimental restraints:
`dual q f d lam = log Z(lam) - <lam, d>`, whose minimisers are the multipliers of the
maximum-entropy tilt of Part XLIII.

`IDR.reweighting_duality_laws` bundles five statements.

1. *Convexity.*  The log-partition function is convex in the multipliers, and so is the dual
   objective.  Ensemble refinement is a convex problem: no spurious local minima, whatever the
   conformational pool.
2. *The gap is a relative entropy.*  If the tilt at `lam` matches the data then for every other
   multiplier vector `mu`, `dual mu - dual lam = KL(tilt lam || tilt mu)` exactly.  The
   suboptimality of a candidate fit is measured in the same information units as the refinement
   objective, which makes it a reportable certificate rather than a residual.
3. *Hence global optimality*: a matching multiplier vector minimises the objective globally.
4. *The ensemble is unique even when the multipliers are not.*  Two multiplier vectors that both
   match the data give the identical reweighted ensemble -- non-identifiability of `lam` (which is
   generic, since restraints are usually dependent on the pool) is not non-identifiability of the
   fit.
5. *Feasibility is exactly boundedness.*  If some ensemble on the pool reproduces the data, the
   objective never drops below `log c`, `c` the smallest prior weight; and if a direction in
   restraint space separates the data from every conformation of the pool by a positive margin,
   the objective is unbounded below.  A refinement whose multipliers diverge is therefore not a
   numerical pathology to be regularised away: it is a proof that the restraints are inconsistent
   with the conformational pool.
-/
import Mathlib
import RequestProject.Dual

set_option autoImplicit false

namespace IDR

open Finset IDR.MaxEnt

/-- **The duality laws of ensemble reweighting.**

1. the log-partition function and the dual objective are convex in the multipliers;
2. against a matching multiplier vector the duality gap is exactly a relative entropy;
3. a matching multiplier vector is a global minimiser;
4. two matching multiplier vectors give the same reweighted ensemble;
5. feasible data bound the objective below, and data separated from the pool make it unbounded
   below. -/
theorem reweighting_duality_laws {n r : ℕ} :
    (∀ q : Fin n → ℝ, (∀ j, 0 < q j) → 0 < n → ∀ (f : Fin r → Fin n → ℝ) (d lam mu : Fin r → ℝ)
        (t : ℝ), 0 ≤ t → t ≤ 1 →
        Real.log (partition q (fun a => t * lam a + (1 - t) * mu a) f)
            ≤ t * Real.log (partition q lam f) + (1 - t) * Real.log (partition q mu f) ∧
          dual q f d (fun a => t * lam a + (1 - t) * mu a)
            ≤ t * dual q f d lam + (1 - t) * dual q f d mu) ∧
    (∀ q : Fin n → ℝ, (∀ j, 0 < q j) → 0 < n → ∀ (f : Fin r → Fin n → ℝ) (d lam : Fin r → ℝ),
        Matches (tilt q lam f) f d → ∀ mu : Fin r → ℝ,
          dual q f d mu - dual q f d lam = klDiv (tilt q lam f) (tilt q mu f)) ∧
    (∀ q : Fin n → ℝ, (∀ j, 0 < q j) → 0 < n → ∀ (f : Fin r → Fin n → ℝ) (d lam : Fin r → ℝ),
        Matches (tilt q lam f) f d → ∀ mu : Fin r → ℝ, dual q f d lam ≤ dual q f d mu) ∧
    (∀ q : Fin n → ℝ, (∀ j, 0 < q j) → 0 < n → ∀ (f : Fin r → Fin n → ℝ) (d lam mu : Fin r → ℝ),
        Matches (tilt q lam f) f d → Matches (tilt q mu f) f d → tilt q lam f = tilt q mu f) ∧
    ((∀ (q : Fin n → ℝ), 0 < n → ∀ c : ℝ, 0 < c → (∀ j, c ≤ q j) →
        ∀ (f : Fin r → Fin n → ℝ) (d : Fin r → ℝ) (p : Fin n → ℝ), (∀ j, 0 ≤ p j) →
          ∑ j, p j = 1 → Matches p f d → ∀ lam : Fin r → ℝ, Real.log c ≤ dual q f d lam) ∧
      (∀ q : Fin n → ℝ, (∀ j, 0 < q j) → ∑ j, q j = 1 →
        ∀ (f : Fin r → Fin n → ℝ) (d u : Fin r → ℝ) (eps : ℝ), 0 < eps →
          (∀ j, ∑ a, u a * f a j ≤ (∑ a, u a * d a) - eps) →
          ∀ b : ℝ, ∃ lam : Fin r → ℝ, dual q f d lam < b)) :=
  ⟨fun _ hq hn f d lam mu _ ht0 ht1 =>
      ⟨logPartition_convex hq hn f lam mu ht0 ht1, dual_convex hq hn f d lam mu ht0 ht1⟩,
    fun _ hq hn _ _ _ hmatch mu => dual_gap hq hn hmatch mu,
    fun _ hq hn _ _ _ hmatch mu => dual_min_of_matches hq hn hmatch mu,
    fun _ hq hn _ _ _ _ hlam hmu => tilt_eq_of_both_match hq hn hlam hmu,
    ⟨fun _ hn _ hc hqc _ _ _ hp hps hmatch lam => dual_ge_of_feasible hn hc hqc hp hps hmatch lam,
      fun _ hq hqs _ _ _ _ heps hsep b => dual_unbounded_of_separated hq hqs heps hsep b⟩⟩

end IDR
