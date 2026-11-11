# Second response: from a rule to a study design — and what still cannot be done here

The previous round conceded the critique's central point and moved the capacity statement from
a prohibition to an exact two-sided law. This round answers the part of the critique that was
left standing: *a threshold is not a test*. A test needs a sample size, a probe, a read-out
rich enough to score, a baseline that fails for the predicted reason, and a multiplicity plan
across a panel. Those are now formal objects, computed from the measured populations, and each
is tied by a theorem to the behaviour of real ensembles.

What has **not** changed: this repository still contains no measurements, and no claim is made
about any real protein, any published ensemble, or any published tool. §5 below states exactly
what is missing and why a proof assistant cannot supply it.

---

## 1. "The proof is table stakes. What buys you something is the test."

Agreed, and the response is to build the test's *instrument* rather than to prove more algebra.
A study is now a frozen record (`IDR.Instrument.Study`), and four numbers are computed from it
in exact rational arithmetic, each with a soundness theorem:

| number | what it fixes | theorem |
|---|---|---|
| `requiredComponents` | how many mixture components the threshold-respecting arm must have | `requiredComponents_sound` (sufficient), `requiredComponents_min` (necessary) |
| `requiredSamples` | how many independent single-molecule observations the refutation needs at level `alpha` | `requiredSamples_sound` |
| `requiredObservables` | how many independent observables the read-out must carry for the scored populations to be a consequence of the data | `requiredObservables_sound` |
| `requiredContrast` | the dynamic range the reporter probe must have to resolve the predicted discrepancy at precision `sigma` | `requiredContrast_sound` |

`IDR.Instrument.report` computes all four; `IDR.Instrument.demoStudy_report` runs it end to end
on the stipulated illustrative five-state record and returns

```
components 5, samples 14, observables 4, contrast 1/10, baselineFloor 2/5, baselineExcluded true
```

so the pipeline is executable, not a schema. Replace the populations with measurements from an
independent experiment and the same computation produces that study's design. Nothing else in
the file changes; nothing is fitted.

## 2. "Stated in advance, and genuinely falsifiable."

The sample size is the piece that was missing, because without it "we did not see the predicted
missed states" is uninterpretable. `RequestProject/DetectionPower.lean` supplies it exactly:

* the probability that `n` independent observations of the real system all avoid the states an
  under-capacity model omits is **exactly** `(1 - tau)^n` (`missProb_eq`) — not a bound;
* one observation inside those states makes the model's likelihood **exactly zero**
  (`likelihood_zero_of_hit`): the outcome is a refutation, not a worse fit statistic;
* `samplesFor alpha tau = ⌈log(1/alpha)/tau⌉` observations make failure-to-refute less likely
  than `alpha` (`samplesFor_spec`);
* and, as an anti-spin clause, with `n` draws the chance of seeing nothing is still at least
  `1 - n·tau` (`missProb_ge_one_sub`), so an under-powered run may not be reported as support
  for the baseline.

The number is a function of the *independently measured* tail population and the pre-registered
`alpha` only. It is computable before the models exist, which is what "stated in advance" has
to mean operationally.

## 3. "It has to beat or explain something a real baseline gets wrong."

The new content here is the explanation of *why the failure is usually invisible*, which is the
part that would otherwise sink the experiment.

* `expect_gap_le_range`: an observable with values in `[a, b]` differs between two ensembles by
  at most `(b-a)/2` times their population `ℓ¹` distance. So a model sitting exactly on the
  capacity floor moves such an observable by at most `(b-a)·tau` (`gap_le_of_at_floor`).
  A fixed three-component mixture that is provably `0.4` away in population space can therefore
  have an excellent χ² against a low-contrast global observable. This is a *mechanism* for a
  familiar and otherwise mysterious situation, and it is a theorem.
* `contrast_requirement`: hence the probe must satisfy `b - a ≥ sigma / tau` to expose the
  failure at measurement precision `sigma`. A number to check before choosing the probe.
* `optimal_reporter`: the indicator of the omitted states has range `1` and reports the whole
  discrepancy — the instruction is to build a reporter that fires on the tail states (a
  contact, a distance window, a labelled pair), not to average harder.
* `observables_needed`: with fewer than `m-1` independent observables, some population the test
  is scored against is not determined by the data at all — so a fit's own populations cannot be
  used as ground truth. That is a constraint on the *experiment*, not on the model.

Taken together these say where a practitioner's default demonstrably loses accuracy, and
simultaneously why the loss does not show up in the statistic they are looking at.

### A positive rule that cuts against current practice

`budget_laws` prices the two routes to accuracy on the same target. On `m` equally populated
states, accuracy `eps` costs exactly `ceil(m(1 - eps/2))` components — and that price is payable
by an explicit model with no data at all, once the populated states are known — while any
*support-honest* learner (a reweighting or weighted-frames model, which can only put weight on
conformations it has seen) needs `m(1 - eps)` observations of a state space that is exponential
in the length of the region. The instruction is prescriptive and it is not where reflexes point:
on broad ensembles, compute spent enlarging a conformational pool buys accuracy only linearly in
coverage, while compute spent identifying and representing the populated states buys it exactly.

## 4. "Effect size and generality set the ceiling."

`panel_design_laws` handles the panel and the multiplicity question at once: run `n`
independent systems, freeze each record at level `alpha/n`, take the sample size that level
dictates on each, and the probability that *every* system refutes its under-capacity baseline
is at least `1 - alpha`. Multiplicity is paid in observations planned in advance, not in a
post-hoc correction. Selectivity is already formal: the rule fires on broad ensembles and is
silent on systems with a dominant state (`IDR.Prereg.illustrativePanel_selective`), which is
what distinguishes a rule with content from one that always says "more capacity".

## 5. What is still missing, stated plainly

**No real data.** No number in this repository is a measurement. The illustrative populations
are stipulated; `demoStudy` is a worked example of the arithmetic, not a result about a
protein. Consequently:

* nothing here claims that any real IDR has any particular populated-state count;
* nothing here claims that any published tool is under-capacity;
* nothing here reports a fit, a baseline comparison, or an effect size on data.

**And this cannot be fixed from inside a proof assistant.** The remaining step is an experiment
plus an analysis run: freeze a record with populations from NMR/SAXS/single-molecule data
obtained independently of the models under test, compute the four numbers, run the baseline and
the threshold-respecting arm on the same data with the same budget, and score. `STUDY_DESIGN.md`
is that protocol, written so that someone else can execute it, with every quantity it needs
either measured or computed by `IDR.Instrument.report`.

**The honest summary.** What has been added is not evidence; it is the difference between a
threshold and a study one could actually run and lose. The theorems now say how many molecules
to watch, with what probe, at what contrast, through how rich a read-out, and what counts as a
refutation — and they say, equally formally, which clause of the outcome no proof can supply
(`design_limits`, `IDR.Prereg.confirms_iff_threshold_fits`). That clause is where the science
is, and it is still empty.

Files: `RequestProject/DetectionPower.lean`, `RequestProject/Visibility.lean`,
`RequestProject/Instrument.lean`, `RequestProject/PanelPower.lean`,
`RequestProject/PartEightyNine.lean`, `STUDY_DESIGN.md`.
