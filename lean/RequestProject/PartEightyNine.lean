/-
# Part LXXXIX  From a threshold to a study design: power, visibility, and the four numbers

Capstone for `RequestProject.DetectionPower`, `RequestProject.Visibility` and
`RequestProject.Instrument`.

Part LXXXV turned the capacity statement from a prohibition into an exact law: the best `ℓ¹`
error a `k`-component model can achieve on a target with measured populations
`w₀ ≥ ⋯ ≥ w_{m-1}` is exactly `2·(w_k + ⋯ + w_{m-1})`.  A law is still not a study.  A study
needs to know how many molecules to observe, what to observe them with, how sharply, and how
many independent observables the read-out must carry before the numbers it is scored against
are properties of the data rather than of the prior.  This part supplies those four
quantities, each as a computation on the independently measured populations, each with a
theorem tying the computed number to the behaviour of real ensembles.

1. **Power.**  `capacity_power_laws`: the probability that `n` independent observations of the
   real system never enter the set of states an under-capacity model omits is exactly
   `(1 - τ)ⁿ`; one observation inside that set drives the model's likelihood to exactly zero;
   `⌈log(1/α)/τ⌉` observations suffice to make failure-to-refute less likely than `α`; and,
   as an honesty clause, a shorter run misses the states most of the time even when the model
   is wrong, so non-observation at small `n` is not evidence.
2. **Visibility.**  `capacity_visibility_laws`: an observable with values in `[a,b]` moves by
   at most `(b-a)·τ` between the truth and a model on the capacity floor — which is why an
   under-capacity model can fit a low-contrast probe beautifully — while the indicator of the
   omitted states, range `1`, shows the entire discrepancy.  Hence the probe to build and the
   contrast `σ/τ` it must have.
3. **The design.**  `design_window`: for a frozen study record, the component count that is
   necessary and the one that is sufficient, the number of observations, the probe contrast,
   and the exclusion of the practitioner's baseline — all at once, all computable, all fixed
   before any model is trained.
4. **The panel.**  `panel_design_laws`: run an `n`-system panel with each record frozen at
   level `alpha/n` and the per-system sample sizes that level dictates, and the whole panel
   refutes its under-capacity baselines with probability at least `1 - alpha` — multiplicity
   paid for in advance, in observations, not in post-hoc corrections.
5. **Where to spend.**  `budget_laws`: on the maximally disordered target the accuracy `eps`
   costs exactly `⌈m(1 - eps/2)⌉` components — payable, with an explicit model, and with no
   data at all once the populated states are known — while the sampling route through any
   support-honest model costs `m(1 - eps)` observations of a state space exponential in the
   length of the region.  Buy capacity and state identification, not pool size.
6. **What it does not buy.**  `design_limits`: meeting the threshold never entails fitting
   well (`IDR.Capacity.at_capacity_not_sufficient`), and with fewer than `m-1` independent
   observables some scored population is not a consequence of the data.  The empirical content
   of the test lives exactly in the clause no proof supplies.

No number in this part is a measurement, and no claim is made about any real system.
-/
import RequestProject.DetectionPower
import RequestProject.Visibility
import RequestProject.Instrument
import RequestProject.PanelPower
import RequestProject.Budget

set_option autoImplicit false

namespace IDR

open Finset Capacity Instrument

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- **The power laws of the capacity test.**  Let a target populate `m` states with profile
`P`, let `k < m`, and let `M` be any model with at most `k` components.  Then there is a set
`A` of conformations with:

1. *the model says empty*: `M` gives every conformation of `A` population zero, while the
   truth gives `A` population at least `tail P k`;
2. *exact miss law*: the chance that `n` independent observations all avoid `A` is exactly
   `(1 - prob A)ⁿ`;
3. *one hit is fatal*: any observation sequence entering `A` has likelihood exactly `0` under
   the model — refutation, not a bad score;
4. *the pre-registered sample size works*: with `n ≥ ⌈log(1/α)/tail P k⌉` observations the
   probability of failing to refute is at most `α`, so the run refutes with probability at
   least `1 - α`;
5. *and short runs prove nothing*: for any `n`, the chance of seeing nothing is still at least
   `1 - n·prob A`, so an under-powered non-observation must not be reported as support. -/
theorem capacity_power_laws {m k n : ℕ} (P : Capacity.Profile m) {g : Fin m → X}
    (hg : Function.Injective g) {M : Ens X} (hM : M.card ≤ k) (hk : k < m)
    {alpha : ℝ} (ha : 0 < alpha) (hn : Power.samplesFor alpha (P.tail k) ≤ n) :
    ∃ A : Finset X,
      ((∀ x ∈ A, M.prob x = 0) ∧
        P.tail k ≤ ∑ x ∈ A, (target P g).prob x) ∧
      Power.missProb (target P g) A n
        = (1 - ∑ x ∈ A, (target P g).prob x) ^ n ∧
      (∀ s : Fin n → X, s ∉ Power.avoiding A n → Power.pathProb M s = 0) ∧
      (Power.missProb (target P g) A n ≤ alpha ∧
        1 - alpha ≤ Power.hitProb (target P g) A n) ∧
      1 - (n : ℝ) * (∑ x ∈ A, (target P g).prob x)
        ≤ Power.missProb (target P g) A n := by
  obtain ⟨A, hA0, hAmass, hlik, hmiss, hhit⟩ :=
    Power.detection_power P hg hM hk ha hn
  exact ⟨A, ⟨hA0, hAmass⟩, Power.missProb_eq _ A n, hlik, ⟨hmiss, hhit⟩,
    Power.missProb_ge_one_sub _ A n⟩

/-- **The visibility laws.**  For a target with profile `P` and any model `M` with at most `k`
components:

1. *the ceiling*: every observable with values in `[a,b]` differs between any two ensembles by
   at most `(b-a)/2` times their population `ℓ¹` distance — so on a model at the capacity
   floor, by at most `(b-a)·tail P k`.  A low-contrast probe cannot see the failure, however
   good the data;
2. *the contrast requirement*: consequently a probe that resolves the discrepancy at precision
   `sigma` must have dynamic range at least `sigma / tail P k`;
3. *the probe to build*: the indicator of the omitted states has range `1`, is predicted to be
   exactly `0` by the model, and is at least `tail P k` in truth — the full discrepancy, in a
   quantity an experiment can be designed to measure. -/
theorem capacity_visibility_laws {m k : ℕ} (P : Capacity.Profile m) {g : Fin m → X}
    (hg : Function.Injective g) {M : Ens X} (hM : M.card ≤ k) (hk : k < m) :
    (∀ (N E : Ens X) (f : X → ℝ) (a b : ℝ), (∀ x, a ≤ f x) → (∀ x, f x ≤ b) →
        |N.expect f - E.expect f| ≤ (b - a) / 2 * Ens.ell1 N E) ∧
    (∀ (E : Ens X) (f : X → ℝ) (a b sigma : ℝ), Ens.ell1 M E ≤ 2 * P.tail k →
        (∀ x, a ≤ f x) → (∀ x, f x ≤ b) → a ≤ b →
        sigma ≤ |M.expect f - E.expect f| → sigma / P.tail k ≤ b - a) ∧
    (∃ A : Finset X,
        M.expect (fun x => if x ∈ A then (1 : ℝ) else 0) = 0 ∧
        P.tail k ≤ (target P g).expect (fun x => if x ∈ A then (1 : ℝ) else 0)) := by
  refine ⟨fun N E f a b hlb hub => Visible.expect_gap_le_range N E hlb hub,
    fun E f a b sigma hfloor hlb hub hab hsee =>
      Visible.contrast_requirement P M E hfloor hlb hub hab hk hsee, ?_⟩
  obtain ⟨A, hM0, hmass, -⟩ := Visible.optimal_reporter P hg hM
  exact ⟨A, hM0, hmass⟩

/-- **The design window for a frozen study record.**  Fix a study `S`: the independently
measured state count and populations, the pre-registered tolerance, the practitioner's
baseline component count, the significance level and the reporter precision.  Then, before
any model is trained:

1. *how many components are needed*: no model with fewer than `S.requiredComponents`
   components reaches the tolerance, on any data, ever;
2. *and that many suffice*: an explicit model with exactly that many components attains it;
3. *how many observations are needed*: after `S.requiredSamples` independent observations, any
   model with the baseline's component count is refuted outright — likelihood exactly zero —
   except with probability at most `alpha`;
4. *how sharp the probe must be*: a reporter resolving the discrepancy at precision `sigma`
   must have dynamic range at least `S.requiredContrast`;
5. *and the baseline's fate is decided by arithmetic*: the report's verdict on the baseline is
   exactly the pre-registered exclusion condition.

Every quantity on the right-hand side is computed from the frozen record; nothing is fitted. -/
theorem design_window (S : Study) {g : Fin S.m → X} (hg : Function.Injective g)
    (hpos : 0 < S.requiredComponents) :
    (∀ (k : ℕ), k < S.requiredComponents → ∀ M : Ens X, M.card ≤ k →
        ((S.eps : ℚ) : ℝ) < Ens.ell1 M (target S.profile g)) ∧
    Ens.ell1 (truncModel S.profile hpos S.requiredComponents_le g) (target S.profile g)
      ≤ ((S.eps : ℚ) : ℝ) ∧
    (∀ (M : Ens X), M.card ≤ S.baselineK → ∀ n : ℕ, S.requiredSamples ≤ n →
        ∃ A : Finset X,
          (∀ x ∈ A, M.prob x = 0) ∧
          ((S.tailAt S.baselineK : ℚ) : ℝ) ≤ ∑ x ∈ A, (target S.profile g).prob x ∧
          (∀ s : Fin n → X, s ∉ Power.avoiding A n → Power.pathProb M s = 0) ∧
          Power.missProb (target S.profile g) A n ≤ ((S.alpha : ℚ) : ℝ)) ∧
    (∀ (M E : Ens X) (f : X → ℝ) (a b : ℝ),
        Ens.ell1 M E ≤ 2 * S.profile.tail S.baselineK →
        (∀ x, a ≤ f x) → (∀ x, f x ≤ b) → a ≤ b →
        ((S.sigma : ℚ) : ℝ) ≤ |M.expect f - E.expect f| →
        ((S.requiredContrast : ℚ) : ℝ) ≤ b - a) ∧
    ((report S).baselineExcluded = true ↔ S.BaselineExcluded) := by
  refine ⟨fun k hk M hM => requiredComponents_min S hk hg hM,
    requiredComponents_sound S hg hpos,
    fun M hM n hn => requiredSamples_sound S hg hM hn,
    fun M E f a b hfloor hlb hub hab hsee =>
      requiredContrast_sound S M E hfloor hlb hub hab hsee,
    report_baselineExcluded S⟩

/-- **What the design does not buy.**  Two limits, stated so they are never elided.

1. *Capacity is necessary, never sufficient*: for every profile there is a model with exactly
   `m` components — at or above any threshold the rule can ask for — that is maximally wrong.
   So "the threshold-respecting arm fits" is an empirical claim, not a corollary.
2. *The read-out must be rich enough*: if every state population reported by a fit is to be a
   consequence of the data, the measurement suite must carry at least `m - 1` independent
   observables.  Below that, the populations the test is scored against come from the prior. -/
theorem design_limits {m : ℕ} (P : Capacity.Profile m) :
    (∃ (g : Fin m → (Fin m ⊕ Fin m)) (M : Ens (Fin m ⊕ Fin m)),
        Function.Injective g ∧ M.card = m ∧ Ens.ell1 M (target P g) = 2) ∧
    (∀ (k : ℕ) (g : Fin k → Fin m → ℝ) (p : Fin m → ℝ) (d : ℝ), 0 < d → (∀ i, d ≤ p i) →
        ∑ i, p i = 1 → (∀ i : Fin m, Identify.Determined g p (Pi.single i (1 : ℝ))) →
        m - 1 ≤ k) := by
  refine ⟨at_capacity_not_sufficient P, ?_⟩
  intro k g p d hd hp hp1 hall
  exact requiredObservables_sound g hd hp hp1 hall

/-- **The panel laws.**  A study of `n` systems, each with its own independently measured tail
`tau i` outside the baseline's component count, each observed
`IDR.Power.samplesFor (alpha/n) (tau i)` times.  If the systems are independent, the
probability that every one of them refutes its under-capacity baseline is at least
`1 - alpha`: the Bonferroni split is paid for by the pre-registered sample sizes rather than
by a post-hoc correction, and the resulting per-system sizes are known before the run. -/
theorem panel_design_laws {n : ℕ} (hn : 0 < n) {alpha : ℝ} (ha : 0 < alpha)
    (halpha1 : alpha ≤ 1) (tau : Fin n → ℝ) (htau : ∀ i, 0 < tau i) (htau1 : ∀ i, tau i ≤ 1)
    (runs : Fin n → ℕ) (hruns : ∀ i, Power.samplesFor (alpha / (n : ℝ)) (tau i) ≤ runs i)
    (miss : Fin n → ℝ) (hmiss0 : ∀ i, 0 ≤ miss i)
    (hmiss : ∀ i, miss i ≤ (1 - tau i) ^ (runs i)) :
    1 - alpha ≤ ∏ i, (1 - miss i) :=
  Panel.panel_samples_suffice hn ha halpha1 tau htau htau1 runs hruns miss hmiss0 hmiss

/-- **The budget laws.**  On `m` equally populated states, at accuracy `eps`:

1. the design rule returns exactly `⌈m(1 - eps/2)⌉` components;
2. no model with fewer reaches `eps`, and an explicit model with that many does — with no data
   at all, once the populated states are known;
3. while a support-honest learner — any reweighting or weighted-frames model, which can only
   put weight on conformations it has seen — needs `m(1 - eps)` samples for the same accuracy.

Since `m` grows exponentially in the length of a disordered region, the third clause is the
expensive one: accuracy on broad ensembles is bought with capacity and state identification,
not with pool size. -/
theorem budget_laws {m : ℕ} (hm : 0 < m) {eps : ℝ} (heps : 0 ≤ eps) {g : Fin m → X}
    (hg : Function.Injective g) (hpos : 0 < ⌈(m : ℝ) * (1 - eps / 2)⌉₊) :
    optimalK (uniformProfile hm) eps = ⌈(m : ℝ) * (1 - eps / 2)⌉₊ ∧
    ((∀ (k : ℕ), k < ⌈(m : ℝ) * (1 - eps / 2)⌉₊ → ∀ M : Ens X, M.card ≤ k →
        eps < Ens.ell1 M (target (uniformProfile hm) g)) ∧
      (∃ hkm : ⌈(m : ℝ) * (1 - eps / 2)⌉₊ ≤ m,
          Ens.ell1 (truncModel (uniformProfile hm) hpos hkm g)
            (target (uniformProfile hm) g) ≤ eps) ∧
      (∀ (n : ℕ) (T : (Fin n → Fin m) → (Fin m → ℝ)), Learn.SupportHonest T →
          Learn.risk (Learn.unifW m) T ≤ eps → (m : ℝ) * (1 - eps) ≤ n)) :=
  ⟨Budget.optimalK_uniform hm heps, Budget.budget_window hm heps hg hpos⟩

end IDR
