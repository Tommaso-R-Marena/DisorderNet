/-
# Part LXIII  Pulling out of equilibrium: work, dissipation and the free energy

Part XV.2 treats force spectroscopy at equilibrium.  A real optical-tweezer or AFM experiment on
a disordered region is not at equilibrium: the trap moves at a finite speed, and what is recorded
is the *work* of each pull, from whose distribution a free energy is then extracted.
`RequestProject.WorkTheorem` formalises what that extraction is worth, from the one physical
hypothesis -- microscopic reversibility, `p γ = q(rev γ)·exp(b·(W γ − dF))`.

`IDR.work_theorem_laws` bundles six statements:

1. *The hypothesis is satisfiable*, and pins the free energy: for every reverse protocol and
   every work function there is a forward protocol obeying Crooks' relation.
2. *The exponential average is exact*: `⟨exp(−b·W)⟩ = exp(−b·dF)`, however fast and however
   dissipative the pull.  A nonequilibrium experiment does determine an equilibrium free energy.
3. *The dissipated work is a relative entropy*: `b·(⟨W⟩ − dF) = Σ p·log(p / q∘rev)`, exactly --
   irreversibility is the statistical distinguishability of the pull from its own time reverse.
4. *Hence the second law* `dF ≤ ⟨W⟩`, with equality exactly when the work is the same on every
   trajectory.  For a disordered chain, whose work distribution is broad because there is no
   folded state to hold it narrow, the mean work is an upper bound on the free energy and not an
   estimate of it.
5. *The forward and reverse work histograms cross exactly at `dF`* -- the one construction that
   returns the free energy with no model in between.
6. *But the exponential average is carried by trajectories that are not sampled*: for every `M`
   there is a work distribution whose rare branch has probability at most `exp(−M)`, contributes
   more than all the rest to the exponential average, and puts the free energy at `−M` or below,
   while the estimate from the typical branch alone is `0`.  And the direction of that error is a
   theorem: discarding any set of low-work trajectories can only raise the estimate.

The design reading: a pulling experiment on a disordered region reports an upper bound (`⟨W⟩`), a
lower-variance but sample-limited exponential average, and -- only if the reverse protocol is
measured too -- an unbiased crossing point.  A model fitted to the first is fitted to the
dissipation of the instrument as much as to the region.
-/
import Mathlib
import RequestProject.WorkTheorem

set_option autoImplicit false

namespace IDR

open IDR.Work

/-- **The nonequilibrium work laws.**

1. Crooks' relation is satisfiable;
2. the Jarzynski equality;
3. dissipation is exactly a relative entropy;
4. the second law, with equality iff the work is deterministic;
5. the work histograms cross at the free energy;
6. and the exponential average is dominated by unsampled trajectories, with a signed error. -/
theorem work_theorem_laws :
    (∀ (G : Type) (_ : Fintype G) (_ : Nonempty G) (rev : Equiv.Perm G) (q W : G → ℝ) (b : ℝ),
        b ≠ 0 → (∀ g, 0 < q g) →
          ∃ (p : G → ℝ) (dF : ℝ), (∀ g, 0 < p g) ∧ (∑ g, p g = 1) ∧ Crooks rev p q W b dF) ∧
    (∀ (G : Type) (_ : Fintype G) (rev : Equiv.Perm G) (p q W : G → ℝ) (b dF : ℝ),
        Crooks rev p q W b dF → (∑ g, q g = 1) →
          ∑ g, p g * Real.exp (-b * W g) = Real.exp (-b * dF)) ∧
    (∀ (G : Type) (_ : Fintype G) (rev : Equiv.Perm G) (p q W : G → ℝ) (b dF : ℝ),
        Crooks rev p q W b dF → (∀ g, 0 < q g) → (∑ g, p g = 1) →
          b * (meanWork p W - dF) = ∑ g, p g * Real.log (p g / q (rev g))) ∧
    (∀ (G : Type) (_ : Fintype G) (rev : Equiv.Perm G) (p q W : G → ℝ) (b dF : ℝ),
        0 < b → Crooks rev p q W b dF → (∀ g, 0 < p g) → (∀ g, 0 < q g) →
        (∑ g, p g = 1) → (∑ g, q g = 1) →
          dF ≤ meanWork p W ∧ ((∀ g, p g = q (rev g)) ↔ ∀ g, W g = dF)) ∧
    (∀ (G : Type) (_ : Fintype G) (rev : Equiv.Perm G) (p q W : G → ℝ) (b dF : ℝ),
        Crooks rev p q W b dF → (∀ g, W (rev g) = - W g) →
          (∀ w : ℝ, hist p W w = Real.exp (b * (w - dF)) * hist q W (-w)) ∧
          hist p W dF = hist q W (-dF)) ∧
    ((∀ M : ℝ, 0 < M → ∃ eps wRare : ℝ, 0 < eps ∧ eps ≤ Real.exp (-M) ∧
        (1 - eps) * Real.exp (-(0:ℝ)) ≤ eps * Real.exp (-wRare) ∧
        M ≤ Real.log (eps * Real.exp (-wRare) + (1 - eps) * Real.exp (-(0:ℝ)))) ∧
      (∀ (G : Type) (_ : Fintype G) (_ : DecidableEq G) (p W : G → ℝ) (b : ℝ) (S : Finset G),
        (∀ g, 0 ≤ p g) → (∑ g, p g = 1) → (∑ g ∈ S, p g < 1) → 0 < b →
        (∀ g ∈ S, ∀ h ∈ Sᶜ, W g < W h) →
          (∑ g ∈ Sᶜ, p g * Real.exp (-b * W g)) / (∑ g ∈ Sᶜ, p g)
            ≤ ∑ g, p g * Real.exp (-b * W g))) := by
  refine ⟨fun G _ _ rev q W b hb hq => crooks_satisfiable rev q W hb hq,
    fun G _ rev p q W b dF hC hq => jarzynski hC hq,
    fun G _ rev p q W b dF hC hq hp => dissipation_eq_relEntropy hC hq hp,
    fun G _ rev p q W b dF hb hC hp hq hps hqs =>
      ⟨second_law hb hC hp hq hps hqs, no_dissipation_iff_deterministic (ne_of_gt hb) hC hq⟩,
    fun G _ rev p q W b dF hC hW =>
      ⟨fun w => crooks_histogram hC hW w, crooks_crossing hC hW⟩,
    fun M hM => rare_trajectories_dominate M hM,
    fun G _ _ p W b S hp hps hS hb hlow => omitting_low_work_overestimates b S hp hps hS hb hlow⟩

end IDR
