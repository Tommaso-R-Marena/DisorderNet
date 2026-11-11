/-
# Part LXII  The Markov state model that is actually estimated

Part IV.6 (`RequestProject.Kinetics`) settles the qualitative question about clustering
microstates into macrostates: under Dynkin's condition the coarse variable is Markov, and
generically it is not.  What it leaves open is the question a Markov state model of a
disordered region actually raises, because nobody verifies Dynkin's condition.  What is done
instead is to count transitions between clusters along an equilibrium trajectory at some lag
time and normalise, producing the stationary-weighted lumping `macroW`.
`RequestProject.MarkovStateModel` asks what *that* matrix is a statement about, and the answer
separates cleanly into three.

`IDR.markov_state_model_laws` bundles five statements:

1. *The estimator is consistent.*  When the clustering happens to be lumpable, the estimated
   matrix is the lumped matrix.
2. *The thermodynamics is always right.*  With no condition on the clustering whatever, the
   estimated matrix is stochastic, the coarse equilibrium populations are stationary for it,
   and detailed balance is inherited.  A Markov state model reproduces the equilibrium
   populations of its own clustering by construction -- so agreement there is no evidence that
   the clustering is any good.
3. *The kinetics is one-sided.*  The coarse mean, mean square and quadratic form of a macro
   observable `g` are exactly the microscopic ones of `g ∘ phi`, so every bound on the
   microscopic Rayleigh quotient is inherited.  Coarse-graining cannot invent a slow mode: a
   reported implied timescale is a lower bound on the truth.
4. *Chapman--Kolmogorov is a genuine test.*  Under lumpability the lag-two estimate is the
   square of the lag-one estimate.
5. *And it is generically failed.*  For an explicit three-state chain that is stochastic,
   reversible and uniformly populated -- no absorbing state, no vanishing population -- the
   lag-two estimate of the escape probability is `1/4` while the square of the lag-one estimate
   is `5/16`.  The matrix a Markov state model reports is a function of the lag.

The design reading for a disordered region, whose clusters are cuts through a continuum rather
than genuine metastable basins: a state-decomposition kinetic model may be quoted for
populations without qualification, must have its timescales quoted as lower bounds, and is not
even defined until the lag time is quoted with it.
-/
import Mathlib
import RequestProject.MarkovStateModel

set_option autoImplicit false

namespace IDR

open IDR.Kinetics IDR.MSM

/-- **The Markov state model laws.**

1. the weighted lumping is a consistent estimator of a lumpable reduction;
2. it always reproduces the coarse thermodynamics -- stochasticity, stationarity and detailed
   balance -- whatever the clustering;
3. its quadratic forms are microscopic quadratic forms, so it can only underestimate a
   relaxation time;
4. under lumpability it satisfies Chapman--Kolmogorov;
5. and on an explicit reversible chain with uniform equilibrium it does not: `1/4` at lag two
   against `5/16` for the square at lag one. -/
theorem markov_state_model_laws :
    (∀ (n p : ℕ) (P : Fin n → Fin n → ℝ) (phi : Fin n → Fin p) (Q : Fin p → Fin p → ℝ)
        (pi : Fin n → ℝ), Lumpable P phi Q → ∀ a b : Fin p, lump phi pi a ≠ 0 →
          macroW phi pi P a b = Q a b) ∧
    (∀ (n p : ℕ) (P : Fin n → Fin n → ℝ) (phi : Fin n → Fin p) (pi : Fin n → ℝ),
        IsStochastic P → DetailedBalance P pi → (∀ i, 0 ≤ pi i) →
        (∀ a : Fin p, lump phi pi a ≠ 0) →
          (∀ a b : Fin p, 0 ≤ macroW phi pi P a b) ∧
          (∀ a : Fin p, ∑ b, macroW phi pi P a b = 1) ∧
          (∀ b : Fin p, ∑ a, lump phi pi a * macroW phi pi P a b = lump phi pi b) ∧
          (∀ a b : Fin p, lump phi pi a * macroW phi pi P a b
            = lump phi pi b * macroW phi pi P b a)) ∧
    (∀ (n p : ℕ) (P : Fin n → Fin n → ℝ) (phi : Fin n → Fin p) (pi : Fin n → ℝ),
        (∀ a : Fin p, lump phi pi a ≠ 0) → ∀ lam : ℝ,
          (∀ f : Fin n → ℝ, mean pi f = 0 → form pi P f ≤ lam * nrm pi f) →
          ∀ g : Fin p → ℝ, mean (lump phi pi) g = 0 →
            form (lump phi pi) (macroW phi pi P) g ≤ lam * nrm (lump phi pi) g) ∧
    (∀ (n p : ℕ) (P : Fin n → Fin n → ℝ) (phi : Fin n → Fin p) (Q : Fin p → Fin p → ℝ)
        (pi : Fin n → ℝ), Lumpable P phi Q → (∀ a : Fin p, lump phi pi a ≠ 0) →
          ∀ a b : Fin p, macroW phi pi (matMul P P) a b
            = ∑ c, macroW phi pi P a c * macroW phi pi P c b) ∧
    (IsStochastic msmP ∧ DetailedBalance msmP msmPi ∧
      macroW badPhi msmPi (matMul msmP msmP) 0 1 = 1/4 ∧
      (∑ c, macroW badPhi msmPi msmP 0 c * macroW badPhi msmPi msmP c 1) = 5/16) := by
  refine ⟨fun n p P phi Q pi h a b ha => macroW_eq_of_lumpable h pi ha b,
    fun n p P phi pi hP hrev hpi hne =>
      ⟨fun a b => macroW_nonneg hP hpi a b, fun a => macroW_stochastic hP (hne a),
        fun b => macroW_stationary (stationary_of_detailedBalance hP hrev) hne b,
        fun a b => macroW_detailedBalance hrev hne a b⟩,
    fun n p P phi pi hne lam h g hg => macro_gap_bound hne lam h g hg,
    fun n p P phi Q pi h hne a b => chapman_kolmogorov_of_lumpable h pi hne a b,
    msmP_stochastic, msmP_detailedBalance, chapman_kolmogorov_fails.1,
    chapman_kolmogorov_fails.2⟩

end IDR
