# Response to the critique: from a filter to a rule, and to a test that can fail

The critique had four claims. Taken together: what existed was one-sided (a filter, not a map),
what "go" content there was restated where the field already is, the quantitative thresholds had
never touched data, and the proof step is table stakes rather than the contribution. Below is what
each claim asked for and what has been added. Two of the four are now addressed inside the
repository; one is addressed as far as it can be without running an experiment; one is conceded.

---

## 1. "A filter, not a map. The no-go theorems shrink the search space; they don't point at the remaining space."

**Conceded as a description of what existed, and fixed.** The capacity statement is now two-sided
and exact rather than a prohibition.

For a target populating `m` states with populations `w₀ ≥ … ≥ w_{m-1}`, the smallest `ℓ¹` error a
`K`-component model can achieve is *exactly*

```
E(K) = 2 · (w_K + w_{K+1} + … + w_{m-1}).
```

* No `K`-component model does better — `IDR.Capacity.ell1_ge_two_tail`.
* An explicit one does exactly this — `IDR.Capacity.ell1_trunc_le_two_tail`; the equality is
  `IDR.Capacity.minErr_eq`.

So the statement is no longer "below `m` you fail". It is a value for every capacity, attained. From
it the design side follows directly:

* `IDR.Capacity.optimalK` — the least component count meeting a requested accuracy, proved
  sufficient (`optimalK_spec`), proved minimal (`optimalK_min`), proved never to exceed `m`
  (`optimalK_le`). That is a positive instruction: build this many components, and here is a
  construction that achieves the accuracy.
* `IDR.Capacity.minErr_step` — the exact improvement bought by the `(K+1)`-st component, `2 w_K`.

## 2. "The no-go results forbid strawmen; the 'stop spending time here' signal confirms a stopping point people already passed."

**Partly conceded, and answered by changing what the statement is about.** A prohibition on a pure
per-residue point predictor is indeed a prohibition on something nobody ships. The threshold law is
not of that kind, for three reasons that are now formal.

* It is quantitative and it fires selectively. On a system with populations
  `0.40, 0.25, 0.15, 0.12, 0.08`, a three-component model has an error floor of `0.40` in `ℓ¹`,
  four times a `0.10` tolerance (`IDR.Prereg.panelA_floor_three`). On a system with a dominant state
  — `0.90, 0.06, 0.02, 0.02` — the same three-component model is *not* excluded
  (`IDR.Prereg.panelC_baseline_not_excluded`). A rule that fires everywhere carries no information;
  this one is silent on exactly the systems where the practitioner's default is fine
  (`IDR.Prereg.illustrativePanel_selective`).
* It predicts a *shape*, and a different shape from the obvious alternative. Generic "capacity
  helps" intuition predicts smooth diminishing returns with no distinguished component count. The
  law predicts strict improvement below `K = m`, a drop of exactly `2 w_K` at each step, and exactly
  zero improvement from `K = m` on (`minErr_strictMono_below`, `minErr_step`, `minErr_eq_zero_iff`).
  The location of the last significant drop is a statistic whose predicted value is fixed by an
  independent measurement before any model is run.
* It predicts a *specific observable failure*, not just a distance. An under-capacity model assigns
  population zero to an identifiable set of conformations that really carries at least the tail
  population (`IDR.Capacity.missed_states_of_under_capacity`) — a named subensemble the model calls
  unoccupied and the experiment finds occupied.

* It quantifies the cost of the practitioner's fixed choice as a function of disorder breadth. On a
  uniform target with `m` states the attainable error at capacity `k` is exactly `2(m-k)/m`
  (`IDR.Capacity.minErr_uniform`), so for any fixed component count there are targets on which the
  best that count can do approaches the maximal error `2` (`IDR.Capacity.fixed_capacity_degrades`).
  "Use three components" is not a mild approximation on broad ensembles; it is one whose error
  tends to the trivial, at a stated rate.

What is *not* claimed: that any published tool is under-capacity, or that any real IDR has these
populations. That is an empirical claim and it is not made anywhere in this repository.

## 3. "None of it has touched real data. Pick the sharpest claim, map it onto one real system, see if it survives."

**Conceded, and unchanged in substance: this repository still contains no data analysis, and none
of the numbers in it are measurements.** What has been added is everything that can honestly be
built before the data run, and the honest statement of the gap.

* `PREREGISTRATION.md` — the protocol, written in advance and executable by someone else: the
  quantitative prediction, the competing hypothesis it discriminates against, the inclusion criteria
  fixing how the state count and populations are obtained independently of the models under test,
  the baseline (a fixed three-component mixture) and the threshold-respecting arm on the same data
  with the same budget, the state-assignment read-out, the tolerance, the Monte-Carlo sample size,
  the multiple-comparison correction and stratified reporting, and the outcomes that would refute
  the prediction.
* The criterion is a formal object, not prose: `IDR.Prereg.SystemSpec` (the frozen record),
  `IDR.Prereg.Confirms` (the success criterion, decidable), `IDR.Prereg.Consistent` (outcomes
  compatible with the proved theory), and their panel-level versions. Filling in a record and
  scoring the result is a computation, not a judgement call, and the record must be committed before
  the models are trained.
* Two theorems close the gap between the idealisation and a real measurement, which is where a
  prediction of this kind usually dies. `IDR.Capacity.stateLevel_floor`: assigning structures to
  reference states is a push-forward, a push-forward cannot increase component count, so the floor
  still holds after read-out. `IDR.Prereg.baseline_must_fail_robust`: if the tolerance sits `2η`
  below the reported floor, the predicted failure holds for *every* population profile within the
  reported error bars, so the prediction does not depend on the published populations being exact.

The honest position is the one the critique states: the theorem tells you what to test; the test is
the scientific content; the test has not been run here, and could not be run from inside a proof
assistant.

## 4. "The proof is table stakes, not the contribution."

**Agreed, and made explicit rather than papered over.** The development now states formally which
half of the experiment is at risk.

* Not at risk: `IDR.Prereg.baseline_must_fail` — on an admissible record, no model with the
  baseline's component count reaches the tolerance. This is a theorem. Its value is diagnostic: if
  the experiment reports a below-tolerance under-capacity model, the error is in the bridge (state
  count, populations, read-out, or the baseline not really having that many components), and
  `PREREGISTRATION.md` §9 lists which to inspect.
* At risk, provably: `IDR.Capacity.at_capacity_not_sufficient` exhibits an `m`-component model that
  is maximally wrong, so "at or above threshold" never entails a good fit;
  `IDR.Prereg.confirms_refutable` and `IDR.Prereg.Panel.refutable` exhibit outcomes consistent with
  every theorem here on which the pre-registered criterion is false. The test can fail.
* And the split is a theorem in its own right: `IDR.Prereg.confirms_iff_threshold_fits` — given
  admissibility, the criterion holds exactly when the threshold-respecting model meets its
  tolerance. All the empirical content is in that clause, and no proof can supply it.

---

## Where this leaves the claim

* **Delivered and machine-checked:** the exact attainable error at every component count and its
  attainment; the predicted improvement per component and the kink at the measured state count; the
  minimal sufficient component count as a design rule; the read-out bridge; robustness to population
  error bars; the identifiable missed subensemble; a pre-registered, decidable success criterion,
  proved refutable at system and panel level. No `sorry`, standard axioms only
  (`propext`, `Classical.choice`, `Quot.sound`).
* **Not delivered:** any contact with real data. The panel records are stipulated illustrations. No
  claim is made about any real system, any published ensemble, or any published tool.
* **The next step is not a proof.** It is §4–§8 of `PREREGISTRATION.md`, run once, with the record
  frozen first.

Files: `RequestProject/CapacityExact.lean`, `RequestProject/Falsification.lean`,
`RequestProject/PartEightyFive.lean`, `PREREGISTRATION.md`. Narrative: `PAPER.md` §6sss,
`ANSWER.md` Part LXXXV.
