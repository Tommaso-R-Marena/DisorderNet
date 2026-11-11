# Pre-registration: the capacity threshold for IDR ensemble models

**Status of this document.** This is a *protocol*, written before any data is touched, for
the one experiment that would turn the capacity threshold from a proved consistency
statement into a design rule with empirical standing. **No data has been analysed here and
no empirical claim is made anywhere in this repository.** Everything in the Lean development
is mathematics; everything in this document that concerns real systems is a commitment about
what will be measured and how the result will be scored, made in advance.

The document is deliberately written so that it can be executed by someone else, and so that
it can fail. Sections 1–3 state the prediction and where it comes from. Sections 4–8 fix the
systems, the models, the metric and the analysis. Section 9 states the outcomes that would
refute it. Section 10 states exactly what the proofs do and do not underwrite.

---

## 1. The quantitative claim being tested

Let a target intrinsically disordered region populate `M` conformational states with
populations `w₁ ≥ w₂ ≥ … ≥ w_M`, `Σ wᵢ = 1`, determined **independently of any model in this
study**. Let a model be any finite mixture with `K` components (a Gaussian mixture over a
structural coordinate, a `K`-cluster ensemble, a `K`-mode generative model — the architecture
is irrelevant to the claim). Write the discrepancy between model and target as the `ℓ¹`
distance between their state-population vectors.

> **Threshold law (proved).** The smallest `ℓ¹` error any `K`-component model can achieve is
> exactly
>
> ```
> E(K) = 2 · (w_{K+1} + w_{K+2} + … + w_M).
> ```
>
> No model with `K` components does better; an explicit one achieves it.

Lean: `IDR.Capacity.ell1_ge_two_tail` (floor), `IDR.Capacity.minErr_eq` (attained), bundled as
`IDR.capacity_threshold_laws` in `RequestProject/PartEightyFive.lean`.

Three consequences are what the experiment will test.

* **C1 — exclusion.** A `K`-component model with `2·Σ_{i>K} wᵢ > ε` cannot reach tolerance
  `ε`, whatever its parameters, training data, optimiser or compute budget. *This is a
  theorem, not a hypothesis.*
* **C2 — sufficiency in practice.** A model with `K = M` components, trained on the same data
  with the same budget, does reach `ε`. *This is not a theorem — it is the empirical claim at
  risk.* Formally, `IDR.Capacity.at_capacity_not_sufficient` shows an `M`-component model can
  be maximally wrong, so nothing in the mathematics forces C2 to hold.
* **C3 — curve shape.** The attainable error drops by exactly `2 w_K` when the `K`-th
  component is added and is exactly zero from `K = M` on: a kink at `K = M`, not smooth
  diminishing returns. Lean: `IDR.Capacity.minErr_step`, `IDR.Capacity.minErr_eq_zero_iff`,
  `IDR.Capacity.minErr_strictMono_below`.

## 2. What this discriminates against

The prediction is only worth running because a plausible competing account makes a different
prediction on the same measurements.

| Hypothesis | Prediction for fitted error vs. `K` |
|---|---|
| **H₁ (threshold law)** | Error tracks `E(K) = 2Σ_{i>K}wᵢ`: strictly decreasing up to `K = M`, last significant drop at exactly `K = M`, no significant improvement for `K > M`. |
| **H₀ (generic capacity intuition)** | Smooth diminishing returns, no distinguished `K`; improvements continue, at shrinking size, past `M`. |
| **H₂ (representation-limited)** | Error plateaus *before* `K = M`, because the component family cannot express the states regardless of count. |

H₁ vs. H₀ is the discriminating comparison, and the location of the last significant drop is
the discriminating statistic. It is fixed in advance by the independently measured `M`.

## 3. Why the proof matters here, and how much

The proof buys three things and no more:

1. `E(K)` has no off-by-one and no hidden assumption: the floor is over *all* `K`-component
   models, and it is attained, so it is an equality rather than a possibly-loose bound.
2. The floor survives the step from structures to state labels: state assignment is a
   push-forward and a push-forward cannot increase component count
   (`IDR.Capacity.stateLevel_floor`). So it is legitimate to score the comparison at the level
   of experimentally reported populations.
3. The floor survives the error bars on the reported populations: profiles differing by `η`
   in `ℓ¹` have floors differing by at most `2η` (`IDR.Capacity.minErr_perturb`), and the
   predicted baseline failure holds against every profile within the error bars once the
   tolerance is `2η` below the reported floor
   (`IDR.Prereg.baseline_must_fail_robust`).

It buys nothing about whether the test succeeds. That is the point of §9.

## 4. Systems: inclusion criteria and how `M` is fixed

The panel is chosen by criteria fixed here, before any system-specific number is entered.

**I1 — independent state decomposition.** The system must have a published decomposition of
its disordered region into a finite number of populated conformational states with reported
populations, obtained by a method that does not fit the model class under test. Acceptable
sources, in order of preference:

1. single-molecule FRET photon-trajectory analysis with a state count selected by an
   information criterion or by a hidden-Markov model comparison;
2. NMR relaxation-dispersion / chemical-exchange saturation transfer analyses reporting
   exchanging states and their populations;
3. deposited conformational ensembles with reported cluster populations (e.g. entries in the
   Protein Ensemble Database), clustered by the depositors' own published procedure.

**I2 — reported uncertainties.** Each population must come with an uncertainty; `η` is the
`ℓ¹` uncertainty of the population vector, computed by the pre-specified rule in §7.

**I3 — training data disjoint from the state determination.** The data used to train or
condition the models must not be the data from which `M` and `w` were derived.

**I4 — panel composition.** At least six systems; at least three with `M ≥ 5` (where the
three-component baseline is excluded) and at least two with `M ≤ 3` or a dominant state
(where it is *not* excluded, so the rule is tested for silence as well as for firing). The
worked illustrations `panelA`, `panelB`, `panelC` in `RequestProject/Falsification.lean` show
the arithmetic of a firing, a firing and a non-firing record respectively; their populations
are stipulated, not measured, and are replaced by real ones at execution.

**I5 — freeze.** For each system, `M`, `w`, `η`, the tolerance `ε` and the baseline component
count `K_base` are entered into a `SystemSpec` record and committed to version control
**before any model is trained**. Admissibility (`BaselineExcludedRobust`, i.e.
`ε + 2η < E(K_base)`) is then checked by computation, not by judgement.

## 5. Models compared

Both arms use the same training data, the same feature representation, the same optimiser,
the same number of random restarts, the same seeds, and the same wall-clock and step budget.

* **Baseline arm.** A fixed three-component mixture — the common practitioner default — with
  `K_base = 3` regardless of `M`. Where the study is run against a published tool with a fixed
  or default component count, that count is used instead and recorded in the record.
* **Threshold arm.** The same family with `K = optimalK(w, ε)`, the least component count with
  `E(K) ≤ ε` (`IDR.Capacity.optimalK`, minimal by `IDR.Capacity.optimalK_min`, sufficient by
  `IDR.Capacity.optimalK_spec`, never above `M` by `IDR.Capacity.optimalK_le`).
* **Curve arm.** The same family at every `K` from `1` to `M + 3`, for the C3 curve-shape test.

## 6. Read-out: from structures to populations

1. Reference structures: one representative per experimentally reported state, taken from the
   same source as the populations.
2. Assignment map `h`: each sampled structure is assigned to the nearest reference under the
   published structural metric (Cα RMSD after optimal superposition, or the order parameter
   used by the source), provided it is within `Δ/2` of that reference, where `Δ` is the
   minimum pairwise distance between references; otherwise it is assigned to a single extra
   "unassigned" bin.
3. Model population vector: the empirical frequencies of the assignments over `N` samples.
4. Error: `ℓ¹` distance between the model population vector and the reported populations, with
   any unassigned mass contributing in full.

`Δ`, the reference set and `N` are fixed at freeze time. `N` is chosen so that the
Monte-Carlo standard error of each `ℓ¹` estimate is below `ε/10`, computed from the
pre-specified bound `sqrt((M+1)/N)`; that value of `N` is recorded in the record.

## 7. Tolerance and uncertainty

* `ε` is set per system, before freezing, as `ε = 0.10` in `ℓ¹` on populations — i.e. the
  model's populations may be off by five percentage points in total variation. It is not tuned
  per system and not changed after any model is run.
* `η` is the `ℓ¹` half-width of the reported population uncertainties, `η = Σᵢ σᵢ`, with `σᵢ`
  the reported standard error of `wᵢ`; if the source reports asymmetric intervals, the larger
  half-width is used.
* A system enters the *firing* stratum only if `ε + 2η < E(K_base)` — the robust admissibility
  condition. Systems failing this are reported in the *silent* stratum and are used as
  controls, never dropped.

## 8. Analysis plan

* **Per-system criterion (pre-registered).** `Confirms` in `RequestProject/Falsification.lean`:
  the baseline arm's error exceeds `ε` **and** the threshold arm's error does not.
* **Primary outcome.** The proportion of firing-stratum systems that confirm.
* **Statistical comparison.** For each system, the baseline-vs-threshold error difference over
  the pre-registered seeds is tested with a two-sided paired test at family-wise `α = 0.05`,
  with Holm–Bonferroni correction across the firing stratum. The silent stratum is reported
  with the same tests but is not part of the primary outcome.
* **Effect size.** Reported as the achieved error difference and as the ratio of the observed
  baseline error to the predicted floor `E(K_base)`; a ratio near `1` is the signature that
  the failure is the predicted one rather than a training artefact.
* **C3 test.** For each system, the location of the last drop in fitted error exceeding
  `2 w_M / 2` is recorded and compared with `M`; H₁ predicts equality, H₀ predicts no
  distinguished location, H₂ predicts a smaller value.
* **Stratified reporting.** Results are reported separately by state count, by data modality
  (smFRET / NMR / deposited ensemble) and by model family, and no stratum is dropped.

## 9. What would refute the prediction

The test is genuinely refutable — formally so: `IDR.Prereg.confirms_refutable` exhibits
outcomes consistent with everything proved here on which the criterion is false, and
`IDR.Prereg.Panel.refutable` does the same at panel level.

* **R1 — the threshold arm fails.** A model at `K = optimalK` does not reach `ε` on a firing
  system. Then capacity is necessary but not practically sufficient, and the design rule is
  demoted from "build at least this large and you will fit" to "build at least this large or
  you certainly will not". This is the most likely failure and the most informative one: it
  localises the gap in optimisation, data or component family, not in the algebra.
* **R2 — the baseline arm succeeds.** The under-capacity model reaches `ε`. Since C1 is a
  theorem, this refutes one of the *bridge* assumptions rather than the mathematics, and the
  analysis then reports which: the state count `M` is wrong (states not resolved, or spurious),
  the populations are wrong beyond `η`, the assignment map does not respect the `Δ/2` rule, or
  the fitted "3-component" model is not in fact a 3-component mixture (e.g. tied components,
  or per-sample conditioning that smuggles in extra capacity). Each of these is checkable, and
  each is a finding.
* **R3 — the curve is the wrong shape.** The last significant drop occurs at `K ≠ M` across the
  panel. That refutes C3 and, with it, the claim that the measured state count is the quantity
  controlling model capacity.

A result in which the firing stratum confirms and the silent stratum does not show the same
gap is the only outcome that supports the rule. Confirmation in both strata would indicate a
confound (e.g. component count correlating with something else) and is reported as such.

## 10. What the Lean development does and does not underwrite

**Underwritten (machine-checked, no `sorry`, standard axioms only).**

* the exact floor `E(K)` and its attainment (`IDR.capacity_threshold_laws`);
* the exclusion of the baseline on any admissible record, including under population error
  bars (`IDR.Prereg.baseline_must_fail`, `IDR.Prereg.baseline_must_fail_robust`);
* legitimacy of scoring at state level (`IDR.Capacity.stateLevel_floor`);
* the specific missed subensemble that an under-capacity model claims is unoccupied
  (`IDR.Capacity.missed_states_of_under_capacity`);
* the minimality and sufficiency of the design rule `optimalK`;
* the refutability of the pre-registered criterion, per system and per panel;
* the arithmetic of every worked record, checked by computation rather than asserted.

**Not underwritten.**

* that any real IDR has a well-defined finite set of populated states — the discreteness is an
  idealisation, and the read-out procedure of §6 is where it is cashed in;
* that the published `M` and `w` for any system are correct;
* that a model at or above the threshold fits well in practice (C2) — this is the empirical
  claim, and `at_capacity_not_sufficient` shows it cannot be derived;
* any statement about how existing published tools perform. Nothing in this repository has
  been run against real data.

## 11. Deviations

Any deviation from this protocol during execution is to be recorded in this file, dated, with
the reason, below this line. No deviations recorded.
