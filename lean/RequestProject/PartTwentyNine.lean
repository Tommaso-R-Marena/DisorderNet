/-
# Part XXIX  What the simulation actually reports: box, cutoff, and error bars

Parts XX and XXVII priced the *dynamics* of generating an ensemble (finite timestep, finite
run length).  This part prices the three remaining systematic differences between the ensemble
a paper reports and the ensemble its model denotes.  All three are properties of the protocol,
not of the force field, and all three have a known sign or a known cost.

* `RequestProject.Box` -- **the periodic cell compacts the region.**  Reweighting any ensemble
  by a positive factor that decreases with the chain dimension can only decrease the mean
  dimension (a weighted Chebyshev inequality), and the image interaction of a periodic box is
  exactly such a factor.  Hence `⟨R⟩_box ≤ ⟨R⟩_∞` at every temperature, strictly in an explicit
  two-state instance, and `⟨R⟩` increases monotonically with box size.  A single box size
  cannot validate a reported dimension.
* `RequestProject.Cutoff` -- **a truncated potential has no landscape beyond the cutoff.**  Any
  two conformations whose pair distances all exceed `rc` receive exactly equal Boltzmann weight,
  while the untruncated model separates them; and the neglected energy of an `n`-site
  conformation with a `C/r` tail is bounded by `n²C/rc`, growing quadratically with the length
  of the region at fixed cutoff.
* `RequestProject.CorrSample` -- **frames are not samples.**  For an exponentially correlated
  series the variance of a trajectory average is exactly
  `σ²/N² · (N(1+ρ)/(1-ρ) - 2ρ(1-ρ^N)/(1-ρ)²)`; it equals `σ²/N` only at `ρ = 0`, equals `σ²`
  at `ρ = 1` however large `N` is, is never below `σ²/N`, and exceeds it by the statistical
  inefficiency `(1+ρ)/(1-ρ) ≈ 2τ/Δt`.  Error bars and sample-complexity bounds must be quoted
  in effective samples `N(1-ρ)/(1+ρ)`.

`IDR.protocol_laws` bundles the seven statements.
-/
import Mathlib
import RequestProject.PeriodicBox
import RequestProject.Cutoff
import RequestProject.CorrelatedSampling

set_option autoImplicit false

namespace IDR

/-- **The design laws of the simulation protocol.**

1. *A tilt decreasing in `R` decreases `⟨R⟩`* (weighted Chebyshev).
2. *The periodic box compacts*: with an image energy nondecreasing in the chain dimension,
   `⟨R⟩_box ≤ ⟨R⟩_∞`, strictly in an explicit two-state instance.
3. *The compaction is monotone in the box size.*
4. *A cutoff model cannot rank conformations beyond the cutoff*: they all receive the same
   Boltzmann weight, while the true potential separates them.
5. *The truncation error is quadratic in the number of sites*: `n²C/rc`.
6. *Correlated frames*: the exact variance of a trajectory average, equal to `σ²` at `ρ = 1`
   for every `N`, and never below the independent-sample value.
7. *The statistical inefficiency*: the variance is inflated by `(1+ρ)/(1-ρ)` up to `O(1/N)`. -/
theorem protocol_laws :
    -- 1  a tilt that decreases with `R` decreases `⟨R⟩`
    (∀ (m : ℕ) (w R f : Fin m → ℝ), (∀ k, 0 ≤ w k) → (∀ k, 0 < f k) → 0 < (∑ k, w k) →
        Box.AntitoneIn R f → Box.wmean (fun k => w k * f k) R ≤ Box.wmean w R) ∧
    -- 2  the periodic box compacts the ensemble, strictly in an explicit instance
    ((∀ (m : ℕ) (beta : ℝ), 0 ≤ beta → ∀ w g R : Fin m → ℝ, (∀ k, 0 ≤ w k) → 0 < (∑ k, w k) →
        (∀ i j, R i ≤ R j → g i ≤ g j) → Box.boxMean beta w g R ≤ Box.wmean w R) ∧
      (∀ beta : ℝ, 0 < beta →
        Box.boxMean beta (![1 / 2, 1 / 2] : Fin 2 → ℝ) (![0, 1] : Fin 2 → ℝ)
            (![1, 2] : Fin 2 → ℝ)
          < Box.wmean (![1 / 2, 1 / 2] : Fin 2 → ℝ) (![1, 2] : Fin 2 → ℝ))) ∧
    -- 3  and the artefact is monotone in the box size
    (∀ (m : ℕ) (beta : ℝ), 0 ≤ beta → ∀ w g g' R : Fin m → ℝ, (∀ k, 0 ≤ w k) → 0 < (∑ k, w k) →
        (∀ i j, R i ≤ R j → g i - g' i ≤ g j - g' j) →
        Box.boxMean beta w g R ≤ Box.boxMean beta w g' R) ∧
    -- 4  a cutoff model is blind beyond the cutoff, where the true model is not
    ((∀ (n : ℕ) (rc beta : ℝ) (u : ℝ → ℝ) (x y : Fin n → ℝ),
        (∀ i j : Fin n, i ≠ j → rc < dist (x i) (x j)) →
        (∀ i j : Fin n, i ≠ j → rc < dist (y i) (y j)) →
        Cutoff.weight beta (Cutoff.truncate rc u) x
          = Cutoff.weight beta (Cutoff.truncate rc u) y) ∧
      (Cutoff.pairEnergy (Cutoff.truncate 1 Cutoff.coulombAttraction) (![(0 : ℝ), 2] : Fin 2 → ℝ)
          = Cutoff.pairEnergy (Cutoff.truncate 1 Cutoff.coulombAttraction)
              (![(0 : ℝ), 3] : Fin 2 → ℝ) ∧
        Cutoff.pairEnergy Cutoff.coulombAttraction (![(0 : ℝ), 2] : Fin 2 → ℝ)
          ≠ Cutoff.pairEnergy Cutoff.coulombAttraction (![(0 : ℝ), 3] : Fin 2 → ℝ))) ∧
    -- 5  the neglected energy grows quadratically with the number of sites
    (∀ (n : ℕ) (rc C : ℝ), 0 < rc → 0 ≤ C → ∀ u : ℝ → ℝ, (∀ r : ℝ, rc < r → |u r| ≤ C / r) →
        ∀ x : Fin n → ℝ,
          |Cutoff.pairEnergy u x - Cutoff.pairEnergy (Cutoff.truncate rc u) x|
            ≤ (n : ℝ) ^ 2 * (C / rc)) ∧
    -- 6  the exact variance of a correlated trajectory average
    ((∀ (rho : ℝ), rho ≠ 1 → ∀ N : ℕ,
        CorrSample.corrSum rho N
          = N * (1 + rho) / (1 - rho) - 2 * rho * (1 - rho ^ N) / (1 - rho) ^ 2) ∧
      (∀ (sigma2 : ℝ) (N : ℕ), 0 < N → CorrSample.varMean sigma2 0 N = sigma2 / N) ∧
      (∀ (sigma2 : ℝ) (N : ℕ), 0 < N → CorrSample.varMean sigma2 1 N = sigma2) ∧
      (∀ (sigma2 rho : ℝ), 0 ≤ sigma2 → 0 ≤ rho → ∀ N : ℕ, 0 < N →
        sigma2 / N ≤ CorrSample.varMean sigma2 rho N)) ∧
    -- 7  the statistical inefficiency, and the frames it costs
    ((∀ (sigma2 rho : ℝ), 0 ≤ sigma2 → 0 ≤ rho → rho < 1 → ∀ N : ℕ, 0 < N →
        sigma2 / N * ((1 + rho) / (1 - rho) - 2 * rho / ((N : ℝ) * (1 - rho) ^ 2))
          ≤ CorrSample.varMean sigma2 rho N) ∧
      (∀ (sigma2 rho eps : ℝ), 0 ≤ sigma2 → 0 ≤ rho → rho < 1 → 0 < eps → ∀ N : ℕ, 0 < N →
        CorrSample.varMean sigma2 rho N ≤ eps →
          sigma2 / eps * ((1 + rho) / (1 - rho) - 2 * rho / (1 - rho) ^ 2) ≤ N)) :=
  ⟨fun _ _ _ _ hw hf hpos hanti => Box.wmean_tilt_le hw hf hpos hanti,
    ⟨fun _ _ hbeta _ _ _ hw hpos hg => Box.boxMean_le_freeMean hbeta hw hpos hg,
      fun _ hbeta => Box.boxMean_lt_freeMean_two hbeta⟩,
    fun _ _ hbeta _ _ _ _ hw hpos hdiff => Box.boxMean_mono_in_box hbeta hw hpos hdiff,
    ⟨fun _ _ _ u _ _ hx hy => Cutoff.cutoff_blind u hx hy,
      Cutoff.cutoff_loses_true_ranking⟩,
    fun _ _ _ hrc hC u hu x => Cutoff.truncation_error_le hrc hC u hu x,
    ⟨fun _ h N => CorrSample.corrSum_eq h N,
      fun sigma2 _ hN => CorrSample.varMean_eq_iid sigma2 hN,
      fun sigma2 _ hN => CorrSample.varMean_frozen sigma2 hN,
      fun _ _ hs hrho _ hN => CorrSample.varMean_ge_iid hs hrho hN⟩,
    ⟨fun _ _ hs hrho h1 _ hN => CorrSample.varMean_ge_inflated hs hrho h1 hN,
      fun _ _ _ hs hrho h1 heps _ hN hgoal =>
        CorrSample.frames_needed hs hrho h1 heps hN hgoal⟩⟩

end IDR
