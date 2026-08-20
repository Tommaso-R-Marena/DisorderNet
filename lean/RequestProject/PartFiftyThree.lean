/-
# Part LIII  The orientation factor: what a FRET efficiency actually measures

Part XLVII treated the donor--acceptor distance distribution of a finite weighted ensemble and
the ideal transfer efficiency `R₀⁶/(R₀⁶+r⁶)`, and listed the orientation factor `κ²` among the
things it did not model.  `RequestProject.Orientation` removes that idealisation, in the only
way that settles the matter: by proving the exact range of `κ²`, by exhibiting a finite
orientational model whose mean is the textbook `2/3`, and by computing what the `κ² = 2/3`
substitution then costs.

`IDR.orientation_laws` bundles five statements:

1. *The exact range* `0 ≤ κ² ≤ 4` for unit dipoles and a unit separation direction, proved from
   the Gram determinant, not assumed.
2. *The textbook average is right*: donor and acceptor uniform over the six axis directions
   give `⟨κ²⟩ = 2/3` exactly.
3. *And it is still biased*: for that same model the mean transfer efficiency is `1/5` where
   the `κ² = 2/3` formula returns `2/5`.  Getting the average orientation factor exactly right
   does not get the average efficiency even approximately right, because the efficiency is not
   linear in `κ²`.
4. *In distance units*: the inferred `r⁶` is `8/3` when the true value is `1`.
5. *And the residual bracket is fixed by the range, not by the data*: at any efficiency the
   distance inferred with `κ² = 4` exceeds the one inferred with `κ² = 2/3` by a factor `6` in
   `r⁶`.

Statement 5 is the practical one: for a disordered region, where dye rotation is neither free
nor frozen and the linkers are short, a reported single-molecule distance without an
orientational model is a number whose systematic uncertainty is set by `[0, 4]` and not by the
photon count.
-/
import Mathlib
import RequestProject.Orientation

set_option autoImplicit false

namespace IDR

open IDR.Orientation

/-- **The orientation factor, exactly.**

1. the range `0 ≤ κ² ≤ 4`;
2. the finite orientational model with `⟨κ²⟩ = 2/3`;
3. the efficiency bias `1/5` against `2/5`;
4. the distance bias `8/3` against `1`;
5. the `κ²`-bracket, a factor `6` in `r⁶`. -/
theorem orientation_laws :
    (∀ d a r : Fin 3 → ℝ, IsUnitVec d → IsUnitVec a → IsUnitVec r →
        0 ≤ kappa d a r ^ 2 ∧ kappa d a r ^ 2 ≤ 4) ∧
    ((∀ i : Fin 6, IsUnitVec (axisVec i)) ∧ IsUnitVec ez ∧ meanKappaSq = 2 / 3) ∧
    (meanOrientEff 1 1 = 1 / 5 ∧ orientEff 1 (2 / 3) 1 = 2 / 5 ∧
      meanOrientEff 1 1 ≠ orientEff 1 (2 / 3) 1) ∧
    (apparentSixth 1 (2 / 3) (meanOrientEff 1 1) = 8 / 3 ∧
      ∀ c k r : ℝ, 0 < c → 0 < k → apparentSixth c k (orientEff c k r) = r ^ 6) ∧
    (∀ c E : ℝ, apparentSixth c 4 E = 6 * apparentSixth c (2 / 3) E) :=
  ⟨fun d a r hd ha hr => ⟨kappaSq_nonneg d a r, kappaSq_le_four hd ha hr⟩,
    ⟨axisVec_unit, ez_unit, meanKappaSq_eq⟩,
    meanEff_ne_effMean,
    ⟨apparentSixth_of_meanEff, fun _ _ _ hc hk => apparentSixth_orientEff hc hk⟩,
    apparentSixth_ratio_six⟩

end IDR
