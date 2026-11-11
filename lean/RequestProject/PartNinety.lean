/-
# Part XC  What the test costs on a real instrument

Capstone for `RequestProject.NoisyDetection` and `RequestProject.NoisyInstrument`.

Part LXXXIX turned the capacity law into a study design, but under one idealisation that no
laboratory satisfies: that an observation *reveals the conformation*, so that a single molecule
found in the set an under-capacity model omits sets that model's likelihood to exactly zero.
Real read-outs are binary reporters with errors — a contact, a distance window, a labelled pair,
an antibody, a crosslink — and under such a reporter nothing is ever refuted outright.  Part XC
redoes the calculation for that instrument, with finite-sample rigour and with no asymptotics
anywhere, and reports three things a design must know.

1. **The contrast collapses to `τ·J`** (`readRate_sub`).  A reporter of sensitivity `se` and
   specificity `sp` raises the positive rate above the disorder-free baseline by exactly
   `τ·(se + sp - 1)` — Youden's index times the omitted population, never more than `τ`.

2. **The price becomes quadratic** (`noisy_capacity_power`).  Where a state-resolving
   observation refuted an under-capacity model in `⌈log(1/α)/τ⌉` draws, a counting test on a
   noisy reporter needs `⌈1/(α·(τJ)²)⌉` molecules to hold *both* error probabilities at `α`, at
   every finite sample size and with no distributional approximation: the bound comes from the
   exact variance `n·q·(1-q)` of the read-out law and Chebyshev's inequality proved from it.
   On the illustrative record of `RequestProject.Instrument` this is `693` molecules against
   `14` — the cost of realism, computed rather than asserted.

3. **And there is something the sample size cannot buy** (`calibration_is_the_binding_constraint`).
   A system whose omitted set carries population `τ`, read by a well-calibrated reporter, and a
   system with *no* such population read by a reporter whose specificity is lower by `τ·J`,
   generate the **identical** law on data records.  Every decision rule, at every sample size,
   behaves identically on the two.  So the reporter's specificity must be known to better than
   `τ·J`; that is a requirement on the instrument, and no amount of counting substitutes for it.
   `RequestProject.NoisyInstrument` prices the calibration run that meets it.

The honest summary of Part XC is therefore not a stronger claim but a more expensive one: the
capacity prediction survives contact with a realistic read-out, and the price of testing it is
quadratic in a contrast that the read-out itself degrades, with a calibration floor underneath
that no statistics can lift.
-/
import Mathlib
import RequestProject.CapacityExact
import RequestProject.DetectionPower
import RequestProject.NoisyDetection
import RequestProject.NoisyInstrument
import RequestProject.DetectionLowerBound
import RequestProject.ReporterPhysics
import RequestProject.ProbePanel

set_option autoImplicit false

namespace IDR
namespace PartXC

open Finset Capacity Noisy

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- **The realistic power theorem for the capacity prediction.**

Let the truth be the ensemble with measured populations `P` on distinct conformations `g`, and
let `M` be any model with at most `k < m` components.  Then there is a set `A` of conformations
that `M` calls unoccupied and the truth occupies with probability at least `P.tail k`.  Score
molecules with a binary reporter that fires with probability `se` inside `A` and `1 - sp`
outside it.  If Youden's index of the reporter is positive, then after

  `⌈1 / (α · (P.tail k · (se + sp - 1))²)⌉`

scored molecules the midpoint counting test has *both* error probabilities at most `α`: it
fails to reject the disorder-free baseline rate with probability at most `α` when the omitted
population is really there, and rejects it wrongly with probability at most `α` when it is
not. -/
theorem noisy_capacity_power {m k : ℕ} (P : Profile m) {g : Fin m → X}
    (hg : Function.Injective g) {M : Ens X} (hM : M.card ≤ k) (hk : k < m)
    {se sp alpha : ℝ} (hse : se ≤ 1) (hsp0 : 0 ≤ sp) (hsp1 : sp ≤ 1)
    (hJ : 0 < youden se sp) (ha : 0 < alpha) {n : ℕ}
    (hn : samplesForReporter alpha (P.tail k) se sp ≤ n) :
    ∃ (A : Finset X) (tau : ℝ),
      (∀ x ∈ A, M.prob x = 0) ∧
      tau = ∑ x ∈ A, (target P g).prob x ∧
      P.tail k ≤ tau ∧
      ∑ s ∈ accepts n (n * (readRate 0 se sp + readRate tau se sp) / 2),
          recProb (readRate tau se sp) s ≤ alpha ∧
      ∑ s ∈ rejects n (n * (readRate 0 se sp + readRate tau se sp) / 2),
          recProb (readRate 0 se sp) s ≤ alpha := by
  classical
  obtain ⟨A, hA0, hAmass⟩ := missed_states_of_under_capacity P hg hM
  set tau := ∑ x ∈ A, (target P g).prob x with htau
  have htail_pos : 0 < P.tail k := P.tail_pos hk
  have htau_pos : 0 < tau := lt_of_lt_of_le htail_pos hAmass
  have htau_le : tau ≤ 1 := Power.mass_le_one _ A
  -- the sample size computed at the (smaller) tail contrast is enough at the true contrast
  have hcontrast : P.tail k * youden se sp ≤ tau * youden se sp := by
    exact mul_le_mul_of_nonneg_right hAmass hJ.le
  have hn' : samplesForReporter alpha tau se sp ≤ n := by
    refine le_trans ?_ hn
    exact samplesFor_antitone ha (mul_pos htail_pos hJ) hcontrast
  obtain ⟨h₁, h₂⟩ := reporter_power htau_pos htau_le hse hsp0 hsp1 hJ ha hn'
  exact ⟨A, tau, hA0, htau, hAmass, h₁, h₂⟩

/-- **Noise is never free.**  Since Youden's index of any reporter is at most one, the number of
molecules the realistic design demands is never smaller than the number the same calculation
would demand of a perfect reporter. -/
theorem noise_costs_samples {tau se sp alpha : ℝ} (htau : 0 < tau) (hse : se ≤ 1) (hsp : sp ≤ 1)
    (hJ : 0 < youden se sp) (ha : 0 < alpha) :
    samplesFor alpha tau ≤ samplesForReporter alpha tau se sp := by
  have h : tau * youden se sp ≤ tau := by
    nlinarith [youden_le_one hse hsp]
  exact samplesFor_antitone ha (mul_pos htau hJ) h

/-- **Calibration, not sample size, is the binding constraint.**  For every population `τ` and
every reporter, there is a specificity within `τ·J` of the nominal one under which a system with
*no* omitted population produces exactly the law of data that the real system produces — so
every decision rule, at every sample size, is exactly as likely to reject in the two worlds.
Reducing the reporter's calibration uncertainty below `τ·J` is therefore a precondition for the
experiment to have any power at all, and `RequestProject.NoisyInstrument` prices it. -/
theorem calibration_is_the_binding_constraint {tau se sp : ℝ} (htau : 0 ≤ tau)
    (hJ : 0 ≤ youden se sp) :
    ∃ sp' : ℝ, |sp' - sp| ≤ tau * youden se sp ∧
      readRate 0 se sp' = readRate tau se sp ∧
      ∀ (n : ℕ) (T : (Fin n → Bool) → Bool),
        ∑ s ∈ ruleSet T, recProb (readRate tau se sp) s
          = ∑ s ∈ ruleSet T, recProb (readRate 0 se sp') s :=
  no_power_of_uncalibrated htau hJ le_rfl

/-- **Part XC, in one statement.**  For an under-capacity model on a measured population profile
and a realistic binary reporter:

1. the contrast available to a counting experiment is exactly `τ·(se+sp-1)`;
2. `⌈1/(α·(τJ)²)⌉` scored molecules bound both error probabilities of the test by `α`, exactly
   and at finite sample size;
3. that number is never smaller than the perfect-reporter number; and
4. no sample size whatsoever separates the hypothesis from a calibration error of size `τ·J`,
   so the specificity must be known better than that. -/
theorem noisy_design_laws {m k : ℕ} (P : Profile m) {g : Fin m → X}
    (hg : Function.Injective g) {M : Ens X} (hM : M.card ≤ k) (hk : k < m)
    {se sp alpha : ℝ} (hse : se ≤ 1) (hsp0 : 0 ≤ sp) (hsp1 : sp ≤ 1)
    (hJ : 0 < youden se sp) (ha : 0 < alpha) {n : ℕ}
    (hn : samplesForReporter alpha (P.tail k) se sp ≤ n) :
    (∀ tau : ℝ, readRate tau se sp - readRate 0 se sp = tau * youden se sp) ∧
    (∃ (A : Finset X) (tau : ℝ),
      (∀ x ∈ A, M.prob x = 0) ∧
      tau = ∑ x ∈ A, (target P g).prob x ∧
      P.tail k ≤ tau ∧
      ∑ s ∈ accepts n (n * (readRate 0 se sp + readRate tau se sp) / 2),
          recProb (readRate tau se sp) s ≤ alpha ∧
      ∑ s ∈ rejects n (n * (readRate 0 se sp + readRate tau se sp) / 2),
          recProb (readRate 0 se sp) s ≤ alpha) ∧
    samplesFor alpha (P.tail k) ≤ samplesForReporter alpha (P.tail k) se sp ∧
    (∃ sp' : ℝ, |sp' - sp| ≤ P.tail k * youden se sp ∧
      readRate 0 se sp' = readRate (P.tail k) se sp ∧
      ∀ (j : ℕ) (T : (Fin j → Bool) → Bool),
        ∑ s ∈ ruleSet T, recProb (readRate (P.tail k) se sp) s
          = ∑ s ∈ ruleSet T, recProb (readRate 0 se sp') s) := by
  have htail_pos : 0 < P.tail k := P.tail_pos hk
  exact ⟨fun tau => readRate_sub tau se sp,
    noisy_capacity_power P hg hM hk hse hsp0 hsp1 hJ ha hn,
    noise_costs_samples htail_pos hse hsp1 hJ ha,
    calibration_is_the_binding_constraint htail_pos.le hJ.le⟩

/-- **The whole of Part XC in one statement.**  For a measured population profile, an
under-capacity model, and a physically realistic probe:

1. the contrast available to a counting experiment is the labelled fraction times the omitted
   population times Youden's index of the *population-weighted mean* sensitivity over the
   omitted states;
2. states the probe cannot see subtract from that contrast;
3. combining probes with an OR rule caps the index at the product of the specificities, so a
   panel of many imperfect probes has exponentially little contrast;
4. at the computed sample size the counting test holds both error probabilities at `alpha`; and
5. no analysis whatsoever — at any sample size, by any statistic — attains both errors `alpha`
   with fewer than `(1 - 2·alpha)/Δ` molecules. -/
theorem realistic_design_verdict {m k : ℕ} (P : Profile m) {g : Fin m → X}
    (hg : Function.Injective g) {M : Ens X} (hM : M.card ≤ k) (hk : k < m)
    {se sp alpha : ℝ} (hse : se ≤ 1) (hsp0 : 0 ≤ sp) (hsp1 : sp ≤ 1)
    (hJ : 0 < youden se sp) (ha : 0 < alpha) {n : ℕ}
    (hn : samplesForReporter alpha (P.tail k) se sp ≤ n)
    (E : Ens X) (A : Finset X) (p : X → ℝ) {d : ℝ} (hA : 0 < Reporter.mass E A)
    (hp1 : ∀ x, p x ≤ 1) {j : ℕ} (pse psp : Fin j → ℝ) (hpse : ∀ i, pse i ≤ 1) :
    (E.expect (Reporter.labelled d (Reporter.fireOf A p sp)) - d * (1 - sp)
        = Reporter.designContrast E A p sp d) ∧
    (∀ D : Finset X, D ⊆ A → (∀ x ∈ D, p x = 0) →
      E.expect (Reporter.fireOf A p sp) - (1 - sp)
        ≤ sp * (Reporter.mass E A - Reporter.mass E D)
          - (1 - sp) * Reporter.mass E D) ∧
    youden (Panel.orSe pse) (Panel.orSp psp) ≤ ∏ i, psp i ∧
    (∃ (A' : Finset X) (tau : ℝ),
      (∀ x ∈ A', M.prob x = 0) ∧
      tau = ∑ x ∈ A', (target P g).prob x ∧
      P.tail k ≤ tau ∧
      ∑ s ∈ accepts n (n * (readRate 0 se sp + readRate tau se sp) / 2),
          recProb (readRate tau se sp) s ≤ alpha ∧
      ∑ s ∈ rejects n (n * (readRate 0 se sp + readRate tau se sp) / 2),
          recProb (readRate 0 se sp) s ≤ alpha) ∧
    (∀ (q₀ q₁ : ℝ), 0 ≤ q₀ → q₀ ≤ 1 → 0 ≤ q₁ → q₁ ≤ 1 → q₀ < q₁ →
      ∀ (i : ℕ) (T : (Fin i → Bool) → Bool),
        ∑ s ∈ ruleSet T, recProb q₀ s ≤ alpha →
        ∑ s ∈ Lower.acceptSet T, recProb q₁ s ≤ alpha →
        1 - 2 * alpha ≤ i * (q₁ - q₀)) :=
  ⟨Reporter.contrast_labelled E A p sp d hA,
    fun _ hDA hdark => Reporter.contrast_le_of_dark E hDA hp1 hdark,
    Panel.youden_or_le_pow hpse,
    noisy_capacity_power P hg hM hk hse hsp0 hsp1 hJ ha hn,
    fun _ _ h₀0 h₀1 h₁0 h₁1 hlt _ T hI hII =>
      Lower.molecules_lower_bound h₀0 h₀1 h₁0 h₁1 hlt T hI hII⟩

end PartXC
end IDR
