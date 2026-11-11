/-
# Part XC.2  The design numbers for a real instrument

`RequestProject.Instrument` computes four numbers from a frozen study record — components,
observations, observables, contrast — under the idealisation that an observation *reveals the
conformation*, so that a single molecule in the omitted set refutes an under-capacity model
outright.  `RequestProject.NoisyDetection` removes that idealisation.  This file puts the two
together: a study record extended with the two numbers that describe the actual reporter (its
sensitivity and its specificity) now yields, in exact rational arithmetic,

* `reporterContrast` — the rate contrast `τ·(se + sp - 1)` that the counting experiment actually
  sees, which is smaller than `τ` for every imperfect reporter;
* `moleculesNeeded` — `⌈1/(α·(τ·J)²)⌉`, the number of molecules to observe, **quadratic** in the
  contrast where the idealised design paid only a logarithm;
* `specTolerance` — how accurately the reporter's specificity must be known before the
  experiment can distinguish a real population from a calibration error, and
* `calibrationMolecules` — how many calibration observations that accuracy itself costs.

Each is accompanied by a soundness theorem: `moleculesNeeded_sound` bounds both error
probabilities of the counting test at the computed sample size, and `specTolerance_sharp` shows
that at any looser calibration the two hypotheses generate identical data, so the number is not
conservatism but a threshold.

The illustrative run (`demoNoisy_report`) is arithmetic on stipulated populations, not a
measurement.  Its content is the comparison: on the same record, the same refutation that an
idealised state-resolving observation buys with fourteen draws costs several hundred molecules
once the read-out is a realistic binary reporter.
-/
import Mathlib
import RequestProject.Instrument
import RequestProject.NoisyDetection

set_option autoImplicit false

namespace IDR
namespace NoisyInstrument

open Instrument Noisy

/-- Ceilings commute with the embedding of the rationals in the reals, so the rational
arithmetic below computes exactly the real sample sizes of `RequestProject.NoisyDetection`. -/
lemma ceil_ratCast (x : ℚ) : ⌈(x : ℝ)⌉₊ = ⌈x⌉₊ := by
  refine le_antisymm ?_ ?_
  · rw [Nat.ceil_le]
    exact_mod_cast Nat.le_ceil x
  · rw [Nat.ceil_le]
    exact_mod_cast Nat.le_ceil (x : ℝ)

/-- A **study record for a real instrument**: the frozen record of
`RequestProject.Instrument`, extended by the measured characteristics of the binary reporter
that will actually be used to score the molecules. -/
structure NoisyStudy extends Study where
  /-- probability that the reporter fires on a molecule in the omitted set -/
  se : ℚ
  /-- probability that the reporter is silent on a molecule outside it -/
  sp : ℚ
  se_le_one : se ≤ 1
  sp_nonneg : 0 ≤ sp
  sp_le_one : sp ≤ 1
  /-- the reporter carries information: Youden's index is positive -/
  youden_pos : 0 < se + sp - 1

namespace NoisyStudy

variable (R : NoisyStudy)

/-- Youden's index of the reporter, `se + sp - 1`. -/
def youdenRat : ℚ := R.se + R.sp - 1

/-- The population the baseline omits, exactly, from the frozen record. -/
def tau : ℚ := R.tailAt R.baselineK

lemma tau_pos : 0 < R.tau := R.tailAt_pos R.baseline_lt

lemma tau_le_one : R.tau ≤ 1 := R.tailAt_le_one R.baselineK

/-- **The contrast a counting experiment actually sees**: `τ·J`, not `τ`. -/
def reporterContrast : ℚ := R.tau * R.youdenRat

lemma reporterContrast_pos : 0 < R.reporterContrast :=
  mul_pos R.tau_pos R.youden_pos

/-- A reporter never improves the contrast. -/
lemma reporterContrast_le_tau : R.reporterContrast ≤ R.tau := by
  have h1 : R.youdenRat ≤ 1 := by
    simp only [youdenRat]
    linarith [R.se_le_one, R.sp_le_one]
  calc R.reporterContrast = R.tau * R.youdenRat := rfl
    _ ≤ R.tau * 1 := by nlinarith [R.tau_pos]
    _ = R.tau := mul_one _

/-- **How many molecules the run must observe** once the read-out is a realistic reporter. -/
def moleculesNeeded : ℕ := ⌈1 / (R.alpha * R.reporterContrast ^ 2)⌉₊

/-- **How accurately the reporter's specificity must be known.**  Above this the calibration
confound of `RequestProject.NoisyDetection` makes the two hypotheses indistinguishable. -/
def specTolerance : ℚ := R.reporterContrast

/-- **How many calibration observations that accuracy costs**, at the same significance. -/
def calibrationMolecules : ℕ := ⌈1 / (4 * R.alpha * (R.specTolerance / 2) ^ 2)⌉₊

end NoisyStudy

/-! ## Soundness -/

open NoisyStudy

variable (R : NoisyStudy)

/-- The rational sample size is exactly the real one of `RequestProject.NoisyDetection`. -/
theorem moleculesNeeded_eq :
    R.moleculesNeeded
      = Noisy.samplesForReporter ((R.alpha : ℚ) : ℝ) ((R.tau : ℚ) : ℝ) ((R.se : ℚ) : ℝ)
          ((R.sp : ℚ) : ℝ) := by
  simp only [NoisyStudy.moleculesNeeded, Noisy.samplesForReporter, Noisy.samplesFor,
    Noisy.youden]
  rw [← ceil_ratCast (1 / (R.alpha * R.reporterContrast ^ 2))]
  congr 1
  simp only [NoisyStudy.reporterContrast, NoisyStudy.youdenRat]
  push_cast
  ring

/-- **The computed sample size does what it says.**  Observe `moleculesNeeded` molecules (or
more) and score each with the reporter.  Then the midpoint counting test rejects the
disorder-free baseline with probability at least `1 - α` when the omitted population really is
there, and rejects it wrongly with probability at most `α` when it is not. -/
theorem moleculesNeeded_sound {n : ℕ} (hn : R.moleculesNeeded ≤ n) :
    ∑ s ∈ Noisy.accepts n
        (n * (Noisy.readRate 0 ((R.se : ℚ) : ℝ) ((R.sp : ℚ) : ℝ)
          + Noisy.readRate ((R.tau : ℚ) : ℝ) ((R.se : ℚ) : ℝ) ((R.sp : ℚ) : ℝ)) / 2),
        Noisy.recProb (Noisy.readRate ((R.tau : ℚ) : ℝ) ((R.se : ℚ) : ℝ) ((R.sp : ℚ) : ℝ)) s
      ≤ ((R.alpha : ℚ) : ℝ) ∧
    ∑ s ∈ Noisy.rejects n
        (n * (Noisy.readRate 0 ((R.se : ℚ) : ℝ) ((R.sp : ℚ) : ℝ)
          + Noisy.readRate ((R.tau : ℚ) : ℝ) ((R.se : ℚ) : ℝ) ((R.sp : ℚ) : ℝ)) / 2),
        Noisy.recProb (Noisy.readRate 0 ((R.se : ℚ) : ℝ) ((R.sp : ℚ) : ℝ)) s
      ≤ ((R.alpha : ℚ) : ℝ) := by
  have ht0 : (0 : ℝ) < ((R.tau : ℚ) : ℝ) := by exact_mod_cast R.tau_pos
  have ht1 : ((R.tau : ℚ) : ℝ) ≤ 1 := by exact_mod_cast R.tau_le_one
  have hse : ((R.se : ℚ) : ℝ) ≤ 1 := by exact_mod_cast R.se_le_one
  have hsp0 : (0 : ℝ) ≤ ((R.sp : ℚ) : ℝ) := by exact_mod_cast R.sp_nonneg
  have hsp1 : ((R.sp : ℚ) : ℝ) ≤ 1 := by exact_mod_cast R.sp_le_one
  have hJ : 0 < Noisy.youden ((R.se : ℚ) : ℝ) ((R.sp : ℚ) : ℝ) := by
    simp only [Noisy.youden]
    have : (0 : ℝ) < ((R.se + R.sp - 1 : ℚ) : ℝ) := by exact_mod_cast R.youden_pos
    push_cast at this
    linarith
  have ha : (0 : ℝ) < ((R.alpha : ℚ) : ℝ) := by exact_mod_cast R.alpha_pos
  refine Noisy.reporter_power ht0 ht1 hse hsp0 hsp1 hJ ha ?_
  rw [← moleculesNeeded_eq R]
  exact hn

/-- **The calibration tolerance is a threshold, not conservatism.**  If the specificity is known
only to within `specTolerance` or worse, then a system with *no* omitted population, read by a
reporter whose specificity is admissible under that uncertainty, produces exactly the same law
on data — so every decision rule, at every sample size, behaves identically on the two. -/
theorem specTolerance_sharp {eta : ℚ} (hbad : R.specTolerance ≤ eta) :
    ∃ sp' : ℝ, |sp' - ((R.sp : ℚ) : ℝ)| ≤ ((eta : ℚ) : ℝ) ∧
      ∀ (n : ℕ) (T : (Fin n → Bool) → Bool),
        ∑ s ∈ Noisy.ruleSet T,
            Noisy.recProb (Noisy.readRate ((R.tau : ℚ) : ℝ) ((R.se : ℚ) : ℝ)
              ((R.sp : ℚ) : ℝ)) s
          = ∑ s ∈ Noisy.ruleSet T,
            Noisy.recProb (Noisy.readRate 0 ((R.se : ℚ) : ℝ) sp') s := by
  have ht : (0 : ℝ) ≤ ((R.tau : ℚ) : ℝ) := by exact_mod_cast R.tau_pos.le
  have hJ : 0 ≤ Noisy.youden ((R.se : ℚ) : ℝ) ((R.sp : ℚ) : ℝ) := by
    simp only [Noisy.youden]
    have : (0 : ℝ) < ((R.se + R.sp - 1 : ℚ) : ℝ) := by exact_mod_cast R.youden_pos
    push_cast at this
    linarith
  have hsmall : ((R.tau : ℚ) : ℝ) * Noisy.youden ((R.se : ℚ) : ℝ) ((R.sp : ℚ) : ℝ)
      ≤ ((eta : ℚ) : ℝ) := by
    have h : R.reporterContrast ≤ eta := hbad
    have : ((R.reporterContrast : ℚ) : ℝ) ≤ ((eta : ℚ) : ℝ) := by exact_mod_cast h
    simpa [NoisyStudy.reporterContrast, NoisyStudy.youdenRat, Noisy.youden] using this
  obtain ⟨sp', hsp', _, hlaw⟩ := Noisy.no_power_of_uncalibrated ht hJ hsmall
  exact ⟨sp', hsp', hlaw⟩

/-- **The calibration run is sufficient**: `calibrationMolecules` observations pin a reporter
rate to half the tolerance with confidence `1 - α`, hence certify the specificity well enough
for the design above. -/
theorem calibrationMolecules_sound {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) {n : ℕ}
    (hn : R.calibrationMolecules ≤ n) (hn0 : 0 < n) :
    ∑ s ∈ Noisy.estErrSet n q (((R.specTolerance / 2 : ℚ) : ℝ)), Noisy.recProb q s
      ≤ ((R.alpha : ℚ) : ℝ) := by
  have ha : (0 : ℝ) < ((R.alpha : ℚ) : ℝ) := by exact_mod_cast R.alpha_pos
  have heps : (0 : ℝ) < ((R.specTolerance / 2 : ℚ) : ℝ) := by
    have : (0 : ℚ) < R.specTolerance / 2 := by
      have := R.reporterContrast_pos
      simp only [NoisyStudy.specTolerance]
      linarith
    exact_mod_cast this
  refine Noisy.calibrationSamples_spec h0 h1 ha heps ?_ hn0
  have hcast : Noisy.calibrationSamples ((R.alpha : ℚ) : ℝ) (((R.specTolerance / 2 : ℚ) : ℝ))
      = R.calibrationMolecules := by
    simp only [Noisy.calibrationSamples, NoisyStudy.calibrationMolecules]
    rw [← ceil_ratCast (1 / (4 * R.alpha * (R.specTolerance / 2) ^ 2))]
    congr 1
    push_cast
    ring
  rw [hcast]
  exact hn

/-! ## The report -/

/-- The four numbers the realistic design adds to `IDR.Instrument.Report`. -/
structure NoisyReport where
  /-- the rate contrast the reporter delivers -/
  contrast : ℚ
  /-- molecules to observe -/
  molecules : ℕ
  /-- required accuracy on the reporter's specificity -/
  specTol : ℚ
  /-- molecules to spend calibrating the reporter -/
  calibration : ℕ
  deriving Repr, DecidableEq

/-- Compute the realistic design from the frozen record. -/
def noisyReport (R : NoisyStudy) : NoisyReport :=
  { contrast := R.reporterContrast
    molecules := R.moleculesNeeded
    specTol := R.specTolerance
    calibration := R.calibrationMolecules }

/-! ## The instrument running end to end

The populations are the stipulated illustration of `IDR.Prereg.panelA`, and the reporter
characteristics are stipulated too; nothing here is a measurement. -/

/-- The illustrative record of `IDR.Instrument.demoStudy`, scored with a good but imperfect
reporter: sensitivity `0.90`, specificity `0.95`. -/
def demoNoisy : NoisyStudy where
  toStudy := demoStudy
  se := 9/10
  sp := 19/20
  se_le_one := by norm_num
  sp_nonneg := by norm_num
  sp_le_one := by norm_num
  youden_pos := by norm_num

/-- The realistic run.  The omitted population is `1/5`, Youden's index is `0.85`, so the
contrast is `17/100`; the design then calls for `693` scored molecules, a specificity known to
`17/100`, and `693` calibration molecules — against the `14` state-resolving observations the
idealised design of `RequestProject.Instrument` asked for. -/
theorem demoNoisy_report :
    noisyReport demoNoisy =
      { contrast := 17/100, molecules := 693, specTol := 17/100, calibration := 693 } := by
  native_decide

end NoisyInstrument
end IDR
