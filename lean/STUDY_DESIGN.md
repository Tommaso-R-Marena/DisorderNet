# Study design addendum to `PREREGISTRATION.md`: the four numbers, and how to fill them in

`PREREGISTRATION.md` fixes *what is scored*. This addendum fixes *how much of everything the
scoring needs*, so that the run can be planned, powered and refuted rather than merely
described. Every quantity below is computed by `IDR.Instrument.report` from a frozen record;
nothing in it is fitted, and nothing in it is a measurement taken from this repository.

**Nothing in this file is data.** The worked numbers use the stipulated illustrative
populations of `RequestProject/Falsification.lean`. A real study replaces them, and only them.

---

## 1. The frozen record

`IDR.Instrument.Study` (`RequestProject/Instrument.lean`) requires:

| field | meaning | source |
|---|---|---|
| `m` | number of populated conformational states | independent measurement, fixed before modelling |
| `q` | their populations, decreasing, summing to 1 | same measurement |
| `eta` | `ℓ¹` error bar on `q` | same measurement |
| `eps` | the `ℓ¹` tolerance at which the comparison is scored | pre-registered choice |
| `baselineK` | component count of the practitioner's baseline (e.g. `3`) | the baseline's published specification |
| `alpha` | significance level of the refutation test | pre-registered choice |
| `sigma` | smallest discrepancy the reporter observable resolves | instrument specification |

Admissibility (`baseline_lt`) requires `baselineK < m`: the theory only predicts a failure when
the baseline has fewer components than the system has measured states. On systems where it does
not, the rule is silent, and that silence is part of the design
(`IDR.Prereg.panelC_baseline_not_excluded`).

## 2. The four numbers

Run `IDR.Instrument.report` on the record. It returns, in exact rational arithmetic:

1. **`components`** — the least component count whose exact error floor is inside `eps`
   (`requiredComponents_eq_optimalK`). Build the threshold-respecting arm with this many.
   Below it, *no* model reaches the tolerance (`requiredComponents_min`); at it, an explicit
   model does (`requiredComponents_sound`).
2. **`samples`** — the least number of independent observations for which the probability of
   never observing a state the baseline omits falls below `alpha` (`requiredSamples_sound`).
   This is the size of the single-molecule run. Below it, a negative result is uninformative
   (`IDR.Power.missProb_ge_one_sub`).
3. **`observables`** — `m - 1`: the number of independent observables the measurement suite
   must carry for the scored populations to be a consequence of the data
   (`requiredObservables_sound`). If the suite is poorer than this, the populations come from
   the prior, and the comparison is not a test of anything.
4. **`contrast`** — `sigma / tail`: the dynamic range the reporter probe must have across
   conformations (`requiredContrast_sound`). A probe below this cannot expose the failure even
   if the failure is total.

Plus `baselineFloor`, the baseline's exact attainable error, and `baselineExcluded`, the
decidable verdict of `PREREGISTRATION.md` §3.

Worked example on the illustrative record (`IDR.Instrument.demoStudy`, populations
`0.40, 0.25, 0.15, 0.12, 0.08`, `eps = 0.10`, `baselineK = 3`, `alpha = 0.05`, `sigma = 0.02`):

```
components 5   samples 14   observables 4   contrast 1/10   baselineFloor 2/5   excluded true
```

## 3. What to measure, and with what

The theory does not merely say "measure more". It names the probe:

* the discriminating observable is the **indicator of the omitted states** — a contact, a
  distance window, a labelled pair that is present in the tail states and absent in the top
  `baselineK` (`IDR.Visible.optimal_reporter`);
* a global, low-contrast observable is the wrong instrument: a model on the capacity floor
  moves it by at most `(b - a)·tail` (`IDR.Visible.gap_le_of_at_floor`), so a good fit to it is
  not evidence of adequate capacity;
* the read-out through which structures are assigned to reference states does not rescue an
  under-capacity model (`IDR.Capacity.stateLevel_floor`), so the comparison may be run at state
  level against reported populations.

## 3a. Where the budget should go

Before costing the run, note the price of each route to accuracy on a broad ensemble
(`IDR.budget_laws`): the component count `ceil(m(1 - eps/2))` is payable by construction once the
populated states are known, whereas a support-honest model — any reweighting or weighted-frames
fit that can only put weight on conformations it has sampled — needs `m(1 - eps)` samples of a
state space exponential in the length of the region. Budget accordingly: state identification and
capacity first, pool size last.

## 4. The run

1. Freeze the record (commit it) **before** training anything.
2. Compute the four numbers; record them in the same commit.
3. Collect `samples` independent observations of the real system with the reporter probe, in a
   suite carrying at least `observables` independent observables.
4. Train the baseline (`baselineK` components) and the threshold-respecting arm (`components`
   components) on the same data, with the same budget.
5. Score exactly as `PREREGISTRATION.md` §7 specifies, i.e. by `IDR.Prereg.Confirms`.

## 5. Panel and multiplicity

For a panel of `n` systems: freeze each record at level `alpha/n` and take that level's sample
size on each. Then the probability that every system refutes its under-capacity baseline is at
least `1 - alpha` (`IDR.panel_design_laws`), assuming the systems are independent preparations.
Multiplicity is thus paid in planned observations rather than in a post-hoc correction; report
stratified by system as `PREREGISTRATION.md` §8 requires.

## 6. Outcomes and what each one means

* **Baseline fails, threshold-respecting arm passes.** The prediction survives. Note that only
  the first half is entailed (`IDR.Prereg.baseline_must_fail`); the second half is the empirical
  content (`IDR.Prereg.confirms_iff_threshold_fits`).
* **Baseline passes.** Something in the bridge is wrong, because the failure is a theorem given
  the record: inspect the state count, the populations, the read-out, or whether the baseline
  really has `baselineK` components (`PREREGISTRATION.md` §9). The robust version
  (`IDR.Prereg.baseline_must_fail_robust`) tolerates population error bars of `eta`.
* **Both fail.** Capacity is necessary, never sufficient
  (`IDR.Capacity.at_capacity_not_sufficient`); the result refutes the pre-registered criterion
  and is reported as such.
* **Nothing observed in the omitted states with fewer than `samples` observations.** Not a
  result. See `IDR.Power.missProb_ge_one_sub`.

## 7. Status

The protocol above is executable by someone with access to a system whose populated-state count
and populations are known independently of the models under test. It has not been executed
here, and this repository contains no measurement of any kind.
