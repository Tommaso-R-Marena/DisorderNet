/-
# Part LXXXIX.3  The instrument: four numbers a real study must fix before it starts

`RequestProject.Falsification` fixes *what* is scored.  This file fixes *how much of
everything the scoring needs*, and makes each quantity a computation on the independently
measured populations rather than a judgement.  A `Study` is a pre-registered record: the
measured state count and populations, the tolerance, the baseline's component count, the
significance level, and the precision of the reporter observable.  From it four numbers are
computed exactly, in rational arithmetic:

* `requiredComponents` — how many mixture components the threshold-respecting arm must have
  (`requiredComponents_eq_optimalK`, `requiredComponents_sound`, `requiredComponents_min`);
* `requiredSamples` — how many independent single-molecule observations the refutation of the
  baseline needs at significance `alpha` (`requiredSamples_sound`);
* `requiredObservables` — how many independent observables the measurement suite must carry
  for the scored populations to be a consequence of the data (`requiredObservables_sound`);
* `requiredContrast` — the dynamic range the reporter probe must have for the predicted
  discrepancy to clear the measurement precision (`requiredContrast_sound`).

Each is accompanied by a theorem tying the computed rational number to the corresponding
statement about real ensembles, so the arithmetic and the mathematics cannot drift apart.
`report` bundles the four, and `#eval` on the illustrative record shows the instrument
running end to end.

The record `demoStudy` uses the **stipulated illustrative** populations of
`RequestProject.Falsification`; they are not measurements, and no claim about any real system
is made here.  Replacing them with populations from an independent experiment — and nothing
else — turns this file into the analysis plan of a real study.
-/
import Mathlib
import RequestProject.CapacityExact
import RequestProject.DetectionPower
import RequestProject.Visibility
import RequestProject.Falsification

set_option autoImplicit false

namespace IDR
namespace Instrument

open Finset Capacity
open scoped Classical

/-! ## Rational tails -/

/-- Past the last state the rational tail vanishes. -/
lemma tailRat_self {m : ℕ} (q : Fin m → ℚ) : tailRat q m = 0 := by
  refine Finset.sum_eq_zero fun i _ => ?_
  have : ¬ (m ≤ (i : ℕ)) := Nat.not_le.mpr i.isLt
  simp [this]

lemma tailRat_nonneg {m : ℕ} {q : Fin m → ℚ} (hq : ∀ i, 0 < q i) (k : ℕ) :
    0 ≤ tailRat q k := by
  refine Finset.sum_nonneg fun i _ => ?_
  by_cases h : k ≤ (i : ℕ) <;> simp [h, (hq i).le]

lemma tailRat_le_one {m : ℕ} {q : Fin m → ℚ} (hq : ∀ i, 0 < q i) (hsum : ∑ i, q i = 1)
    (k : ℕ) : tailRat q k ≤ 1 := by
  rw [← hsum, tailRat]
  refine Finset.sum_le_sum fun i _ => ?_
  by_cases h : k ≤ (i : ℕ) <;> simp [h, (hq i).le]

lemma tailRat_pos {m k : ℕ} {q : Fin m → ℚ} (hq : ∀ i, 0 < q i) (hk : k < m) :
    0 < tailRat q k := by
  have hmem : (⟨k, hk⟩ : Fin m) ∈ (Finset.univ : Finset (Fin m)) := Finset.mem_univ _
  have hterm : 0 < (if k ≤ ((⟨k, hk⟩ : Fin m) : ℕ) then q ⟨k, hk⟩ else 0) := by
    simp [hq ⟨k, hk⟩]
  refine lt_of_lt_of_le hterm ?_
  refine Finset.single_le_sum (f := fun i : Fin m => if k ≤ (i : ℕ) then q i else 0)
    (fun i _ => ?_) hmem
  by_cases h : k ≤ (i : ℕ) <;> simp [h, (hq i).le]

/-! ## The pre-registered study record -/

/-- A **study record**: everything that must be frozen before the models are trained.  It
extends the scoring record of `RequestProject.Falsification` with the three quantities the
design of the run needs — the significance level, the precision of the reporter observable,
and the error bar on the reported populations — together with the standing assumption that
the baseline really is below the state count, which is when the theory makes a prediction at
all. -/
structure Study extends Prereg.SystemSpec where
  /-- the significance level of the refutation test -/
  alpha : ℚ
  alpha_pos : 0 < alpha
  /-- the smallest discrepancy the reporter observable can resolve -/
  sigma : ℚ
  /-- the `ℓ¹` error bar on the reported populations -/
  eta : ℚ
  eps_nonneg : 0 ≤ eps
  /-- the baseline has fewer components than the system has populated states -/
  baseline_lt : baselineK < m

namespace Study

variable (S : Study)

/-- The population outside the top `k` states, exactly. -/
def tailAt (k : ℕ) : ℚ := tailRat S.q k

lemma tailAt_pos {k : ℕ} (hk : k < S.m) : 0 < S.tailAt k := tailRat_pos S.qpos hk

lemma tailAt_le_one (k : ℕ) : S.tailAt k ≤ 1 := tailRat_le_one S.qpos S.qsum k

lemma floorAt_eq (k : ℕ) : S.floorAt k = 2 * S.tailAt k := rfl

/-! ### Number 1: how many components -/

lemma exists_ok_capacity : ∃ k : ℕ, S.floorAt k ≤ S.eps :=
  ⟨S.m, by simpa [Prereg.SystemSpec.floorAt, minErrRat, tailRat_self S.q] using S.eps_nonneg⟩

/-- **How many mixture components the threshold-respecting arm must have**: the least count
whose exact error floor is inside the pre-registered tolerance. -/
def requiredComponents : ℕ := Nat.find S.exists_ok_capacity

lemma requiredComponents_le : S.requiredComponents ≤ S.m :=
  Nat.find_le (by simpa [Prereg.SystemSpec.floorAt, minErrRat, tailRat_self S.q] using S.eps_nonneg)

lemma requiredComponents_floor : S.floorAt S.requiredComponents ≤ S.eps :=
  Nat.find_spec S.exists_ok_capacity

lemma requiredComponents_min' {k : ℕ} (hk : k < S.requiredComponents) : S.eps < S.floorAt k :=
  lt_of_not_ge (Nat.find_min S.exists_ok_capacity hk)

/-! ### Number 2: how many observations -/

lemma exists_ok_samples : ∃ n : ℕ, (1 - S.tailAt S.baselineK) ^ n ≤ S.alpha := by
  obtain ⟨n, hn⟩ := exists_pow_lt_of_lt_one (x := S.alpha)
    (y := 1 - S.tailAt S.baselineK) S.alpha_pos
    (by linarith [S.tailAt_pos S.baseline_lt])
  exact ⟨n, hn.le⟩

/-- **How many independent observations the refutation needs**: the least number of draws for
which the probability of never seeing a state the baseline omits falls below `alpha`. -/
def requiredSamples : ℕ := Nat.find S.exists_ok_samples

lemma requiredSamples_pow : (1 - S.tailAt S.baselineK) ^ S.requiredSamples ≤ S.alpha :=
  Nat.find_spec S.exists_ok_samples

/-! ### Numbers 3 and 4: how many observables, and how sharp -/

/-- **How many independent observables the measurement suite must carry** for the scored
populations to be determined by the data. -/
def requiredObservables : ℕ := S.m - 1

/-- **The dynamic range the reporter probe must have**: the resolvable discrepancy divided by
the population the baseline omits. -/
def requiredContrast : ℚ := S.sigma / S.tailAt S.baselineK

end Study

/-! ## Soundness: each computed number means what it says -/

/-- The real tail of the profile is the exact rational tail of the reported populations. -/
lemma tail_profile (S : Study) (k : ℕ) : S.profile.tail k = ((S.tailAt k : ℚ) : ℝ) := by
  classical
  simp only [Study.tailAt, tailRat, Profile.tail, Prereg.SystemSpec.profile, Profile.ofRat,
    Finset.sum_filter]
  push_cast
  refine Finset.sum_congr rfl fun i _ => ?_
  split_ifs <;> simp


variable {X : Type*} [Fintype X] [DecidableEq X]

/-- The computed component count is exactly the design rule of `RequestProject.CapacityExact`
evaluated on the measured populations. -/
theorem requiredComponents_eq_optimalK (S : Study) :
    S.requiredComponents = optimalK S.profile ((S.eps : ℚ) : ℝ) := by
  have hfloor : ∀ k : ℕ, minErr S.profile k = ((S.floorAt k : ℚ) : ℝ) := fun k =>
    minErr_ofRat S.q S.qpos S.qanti S.qsum k
  have hmem : ∀ k : ℕ, (minErr S.profile k ≤ ((S.eps : ℚ) : ℝ)) ↔ S.floorAt k ≤ S.eps := by
    intro k
    rw [hfloor k]
    exact_mod_cast Iff.rfl
  refine le_antisymm ?_ ?_
  · refine Nat.find_le ?_
    exact (hmem _).1 (optimalK_spec S.profile (by exact_mod_cast S.eps_nonneg))
  · refine Nat.sInf_le ?_
    exact (hmem _).2 S.requiredComponents_floor

/-- **The component count is sufficient**: an explicit model with that many components
attains the tolerance. -/
theorem requiredComponents_sound (S : Study) {g : Fin S.m → X}
    (hg : Function.Injective g) (hpos : 0 < S.requiredComponents) :
    Ens.ell1 (truncModel S.profile hpos S.requiredComponents_le g) (target S.profile g)
      ≤ ((S.eps : ℚ) : ℝ) := by
  have heq := minErr_eq S.profile hpos S.requiredComponents_le hg
  have hfloor : minErr S.profile S.requiredComponents
      = ((S.floorAt S.requiredComponents : ℚ) : ℝ) :=
    minErr_ofRat S.q S.qpos S.qanti S.qsum _
  have hle : ((S.floorAt S.requiredComponents : ℚ) : ℝ) ≤ ((S.eps : ℚ) : ℝ) := by
    exact_mod_cast S.requiredComponents_floor
  rw [heq, hfloor]
  exact hle

/-- **And nothing smaller is**: every model with fewer components misses the tolerance,
whatever its parameters. -/
theorem requiredComponents_min (S : Study) {k : ℕ} (hk : k < S.requiredComponents)
    {g : Fin S.m → X} (hg : Function.Injective g) {M : Ens X} (hM : M.card ≤ k) :
    ((S.eps : ℚ) : ℝ) < Ens.ell1 M (target S.profile g) := by
  have hfloor : minErr S.profile k ≤ Ens.ell1 M (target S.profile g) :=
    ell1_ge_two_tail S.profile hg hM
  have hval : minErr S.profile k = ((S.floorAt k : ℚ) : ℝ) :=
    minErr_ofRat S.q S.qpos S.qanti S.qsum k
  have hlt : ((S.eps : ℚ) : ℝ) < ((S.floorAt k : ℚ) : ℝ) := by
    exact_mod_cast S.requiredComponents_min' hk
  linarith [hval ▸ hfloor]

/-- **The sample size is sufficient.**  Run the real system for `requiredSamples` independent
observations.  Then for *every* model with at most `baselineK` components there is a set of
conformations the model calls unoccupied, the truth occupies it with probability at least the
measured tail, every observation inside it makes the model's likelihood exactly zero, and the
probability that the run produces no such observation is at most `alpha`. -/
theorem requiredSamples_sound (S : Study) {g : Fin S.m → X} (hg : Function.Injective g)
    {M : Ens X} (hM : M.card ≤ S.baselineK) {n : ℕ} (hn : S.requiredSamples ≤ n) :
    ∃ A : Finset X,
      (∀ x ∈ A, M.prob x = 0) ∧
      ((S.tailAt S.baselineK : ℚ) : ℝ) ≤ ∑ x ∈ A, (target S.profile g).prob x ∧
      (∀ s : Fin n → X, s ∉ Power.avoiding A n → Power.pathProb M s = 0) ∧
      Power.missProb (target S.profile g) A n ≤ ((S.alpha : ℚ) : ℝ) := by
  classical
  obtain ⟨A, hA0, hAmass⟩ := missed_states_of_under_capacity S.profile hg hM
  have htailR : S.profile.tail S.baselineK = ((S.tailAt S.baselineK : ℚ) : ℝ) :=
    tail_profile S S.baselineK
  have hmass : ((S.tailAt S.baselineK : ℚ) : ℝ) ≤ ∑ x ∈ A, (target S.profile g).prob x := by
    rw [← htailR]; exact hAmass
  refine ⟨A, hA0, hmass, fun s hs => Power.likelihood_zero_off_avoiding hA0 hs, ?_⟩
  -- the rational bound at the required sample size, transported to `n ≥ requiredSamples`
  have hbase0 : (0 : ℚ) ≤ 1 - S.tailAt S.baselineK := by
    linarith [S.tailAt_le_one S.baselineK]
  have hbase1 : (1 : ℚ) - S.tailAt S.baselineK ≤ 1 := by
    linarith [(S.tailAt_pos S.baseline_lt).le]
  have hmono : (1 - S.tailAt S.baselineK) ^ n
      ≤ (1 - S.tailAt S.baselineK) ^ S.requiredSamples :=
    pow_le_pow_of_le_one hbase0 hbase1 hn
  have hrat : (1 - S.tailAt S.baselineK) ^ n ≤ S.alpha :=
    le_trans hmono S.requiredSamples_pow
  have hreal : (1 - ((S.tailAt S.baselineK : ℚ) : ℝ)) ^ n ≤ ((S.alpha : ℚ) : ℝ) := by
    have : (((1 - S.tailAt S.baselineK) ^ n : ℚ) : ℝ) ≤ ((S.alpha : ℚ) : ℝ) := by
      exact_mod_cast hrat
    push_cast at this
    exact this
  refine le_trans (Power.missProb_le_of_mass_ge _ A hmass n) hreal

/-- **The observable count is necessary.**  If the measurement suite carries fewer than
`requiredObservables` independent observables, some state population reported by the fit is
not a consequence of the data — so the populations the test is scored against would come from
the prior, not the experiment. -/
theorem requiredObservables_sound {m k : ℕ} (g : Fin k → Fin m → ℝ) {p : Fin m → ℝ} {d : ℝ}
    (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1)
    (hall : ∀ i : Fin m, Identify.Determined g p (Pi.single i (1 : ℝ))) :
    m - 1 ≤ k :=
  Nat.sub_le_of_le_add (Visible.observables_needed g hd hp hp1 hall)

/-- **The contrast requirement is necessary.**  A probe with values in `[a, b]` that resolves
the predicted discrepancy at precision `sigma`, against a baseline sitting on its capacity
floor, must have dynamic range at least `requiredContrast`. -/
theorem requiredContrast_sound (S : Study) (M E : Ens X)
    (hfloor : Ens.ell1 M E ≤ 2 * S.profile.tail S.baselineK) {f : X → ℝ} {a b : ℝ}
    (hlb : ∀ x, a ≤ f x) (hub : ∀ x, f x ≤ b) (hab : a ≤ b)
    (hsee : ((S.sigma : ℚ) : ℝ) ≤ |M.expect f - E.expect f|) :
    ((S.requiredContrast : ℚ) : ℝ) ≤ b - a := by
  have htailR : S.profile.tail S.baselineK = ((S.tailAt S.baselineK : ℚ) : ℝ) :=
    tail_profile S S.baselineK
  have h := Visible.contrast_requirement S.profile M E hfloor hlb hub hab S.baseline_lt hsee
  have hcast : ((S.requiredContrast : ℚ) : ℝ)
      = ((S.sigma : ℚ) : ℝ) / S.profile.tail S.baselineK := by
    rw [htailR, Study.requiredContrast]; push_cast; ring
  rw [hcast]
  exact h

/-! ## The report -/

/-- The four numbers a study must fix in advance. -/
structure Report where
  /-- component count of the threshold-respecting arm -/
  components : ℕ
  /-- independent observations needed to refute the baseline at significance `alpha` -/
  samples : ℕ
  /-- independent observables the measurement suite must carry -/
  observables : ℕ
  /-- dynamic range the reporter probe must have -/
  contrast : ℚ
  /-- the exact error floor of the baseline -/
  baselineFloor : ℚ
  /-- whether the theory predicts the baseline must fail on this system -/
  baselineExcluded : Bool
  deriving Repr, DecidableEq

/-- Compute the whole design from the frozen record. -/
def report (S : Study) : Report :=
  { components := S.requiredComponents
    samples := S.requiredSamples
    observables := S.requiredObservables
    contrast := S.requiredContrast
    baselineFloor := S.floorAt S.baselineK
    baselineExcluded := decide (S.eps < S.floorAt S.baselineK) }

/-- The report's verdict on the baseline is the pre-registered admissibility condition of
`RequestProject.Falsification`. -/
theorem report_baselineExcluded (S : Study) :
    (report S).baselineExcluded = true ↔ S.BaselineExcluded := by
  simp [report, Prereg.SystemSpec.BaselineExcluded]

/-! ## The instrument running end to end

The populations are the stipulated illustration `IDR.Prereg.qA`, not a measurement. -/

/-- A filled-in study record on the illustrative five-state system: tolerance `0.10`,
baseline a fixed three-component mixture, significance `0.05`, reporter precision `0.02`. -/
def demoStudy : Study where
  toSystemSpec := Prereg.panelA
  alpha := 1/20
  alpha_pos := by norm_num
  sigma := 1/50
  eta := 1/100
  eps_nonneg := by norm_num [Prereg.panelA]
  baseline_lt := by norm_num [Prereg.panelA]

/-- The illustrative run: five components required, fourteen observations, four independent
observables, contrast `1/10`, baseline floor `2/5`, baseline excluded. -/
theorem demoStudy_report :
    report demoStudy =
      { components := 5, samples := 14, observables := 4, contrast := 1/10,
        baselineFloor := 2/5, baselineExcluded := true } := by
  native_decide

end Instrument
end IDR
