/-
# Part LXXIV  What a titration curve on a disordered region can establish

A disordered region binds multivalently, and the standard summary of a titration is the Hill
slope.  `RequestProject.BindingPolynomial` treats the general case: a binding polynomial
`Z(x) = sum_{j<=n} a j x^j` with nonnegative weights, i.e. an arbitrary equilibrium model of `n`
sites with couplings of any order.

`IDR.binding_polynomial_laws` bundles five statements.

1. *Fluctuation--response.*  `d<N>/d ln x = Var(N)`; in particular every binding curve of every
   such model is nondecreasing, so a nonmonotone titration cannot be an equilibrium binding
   effect at all.
2. *The Hill slope is a fluctuation ratio.*  `d ln(theta/(1-theta))/d ln x = n Var/(<N>(n-<N>))`.
3. *The valence bound.*  `Var <= <N>(n - <N>)`, hence the Hill slope never exceeds the valence.
4. *Equality is all-or-none, and is attained.*  The slope equals `n` at an activity exactly when
   every partially bound state carries zero weight there, and the all-or-none polynomial attains
   it at every activity.  A steep titration is a statement about intermediate populations.
5. *The independence bound.*  If the sites bind independently -- `Z = prod_s (1 + k_s x)` -- the
   slope is at most one at every activity whatever the affinities, with equality for identical
   sites.  Slopes below one need heterogeneity, not negative cooperativity; slopes above one
   falsify independence outright.

Combined with Part LXX, this is a rare positive: the Hill slope is one of the few numbers about a
disordered region that is both measurable and interpretable, because its two bounds are
model-free.  What it does not give is the ensemble: infinitely many weight vectors realise any
admissible slope, and Part LXIX prices the rest.
-/
import Mathlib
import RequestProject.BindingPolynomial

set_option autoImplicit false

namespace IDR

open IDR.BindPoly

/-- **The binding-polynomial laws.**

1. `d<N>/d ln x = Var(N)`, so binding curves are nondecreasing;
2. the Hill slope is the logarithmic derivative of the saturation ratio;
3. `Var <= <N>(n - <N>)` and hence the Hill slope is at most the valence;
4. equality holds exactly for all-or-none weights, and the all-or-none polynomial attains it;
5. independent sites have Hill slope at most one, attained by identical sites. -/
theorem binding_polynomial_laws :
    (∀ (n : ℕ) (a : ℕ → ℝ) (x : ℝ), x ≠ 0 → part n a x ≠ 0 →
        HasDerivAt (meanOcc n a) (varOcc n a x / x) x) ∧
    (∀ (n : ℕ) (a : ℕ → ℝ) (x : ℝ), (∀ j, 0 ≤ a j) → 0 < x → 0 < part n a x →
        0 ≤ deriv (meanOcc n a) x) ∧
    (∀ (n : ℕ) (a : ℕ → ℝ) (x : ℝ), 0 < x → part n a x ≠ 0 → meanOcc n a x ≠ 0 →
        (n : ℝ) - meanOcc n a x ≠ 0 →
        HasDerivAt (fun y => Real.log (meanOcc n a y) - Real.log ((n : ℝ) - meanOcc n a y))
          (hill n a x / x) x) ∧
    (∀ (n : ℕ) (a : ℕ → ℝ) (x : ℝ), (∀ j, 0 ≤ a j) → 0 < x → 0 < part n a x →
        varOcc n a x ≤ meanOcc n a x * ((n : ℝ) - meanOcc n a x)) ∧
    ((∀ (n : ℕ) (a : ℕ → ℝ) (x : ℝ), (∀ j, 0 ≤ a j) → 0 < x → 0 < part n a x →
        0 < meanOcc n a x → meanOcc n a x < n → hill n a x ≤ n) ∧
      (∀ (n : ℕ) (a : ℕ → ℝ) (x : ℝ), (∀ j, 0 ≤ a j) → 0 < x → 0 < part n a x →
        0 < meanOcc n a x → meanOcc n a x < n →
          (hill n a x = n ↔ ∀ j, 0 < j → j < n → a j = 0)) ∧
      (∀ (n : ℕ) (x : ℝ), 0 < n → 0 < x → hill n (allOrNone n) x = n)) ∧
    ((∀ (m : ℕ) (k : ℕ → ℝ) (x : ℝ), (∀ s, 0 < k s) → 0 < x → 0 < m → hillInd m k x ≤ 1) ∧
      (∀ (m : ℕ) (c x : ℝ), 0 < c → 0 < x → 0 < m → hillInd m (fun _ => c) x = 1)) := by
  refine ⟨fun n a x hx hZ => hasDerivAt_meanOcc n a hx hZ,
    fun n a x ha hx hZ => meanOcc_nondecreasing_deriv n a ha hx hZ,
    fun n a x hx hZ h0 hn => hasDerivAt_logit n a hx hZ h0 hn,
    fun _ _ _ ha hx hZ => varOcc_le ha hx hZ,
    ⟨fun _ _ _ ha hx hZ h0 hn => hill_le_valence ha hx hZ h0 hn,
      fun _ _ _ ha hx hZ h0 hn => hill_eq_valence_iff ha hx hZ h0 hn,
      fun _ _ hn hx => hill_allOrNone hn hx⟩,
    fun _ _ _ hk hx hm => hillInd_le_one hk hx hm,
    fun _ _ _ hc hx hm => hillInd_identical hc hx hm⟩

end IDR
