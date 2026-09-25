# Part XC — Pushing the test onto a real instrument

Everything before Part XC prices the capacity prediction under an idealisation that no
laboratory satisfies: that an observation *reveals which conformation a molecule is in*, so that
a single molecule found in the set an under-capacity model omits drives that model's likelihood
to exactly zero, and `⌈log(1/α)/τ⌉` observations settle the matter.

Real read-outs are not like that.  They are binary reporters with errors — a contact, a distance
window, a labelled pair, a crosslink, an antibody — attached to a fraction of the molecules,
firing with a probability that depends on the conformation, and calibrated only to some finite
accuracy.  Part XC redoes the calculation for that instrument.  Every statement below is
machine-checked in Lean, at finite sample size, with no asymptotics and no normal approximation.

## The files

| File | Content |
| --- | --- |
| `RequestProject/NoisyDetection.lean` | The read-out law on `Fin n → Bool`, its exact moments, Chebyshev, the counting test, sample size, calibration confound, calibration cost, pilot-based sizing |
| `RequestProject/ReporterPhysics.lean` | State-dependent firing probabilities, the weighted-mean sensitivity, dark states, labelling efficiency |
| `RequestProject/DetectionLowerBound.lean` | The converse: what no analysis whatsoever can do, and why reads are not molecules |
| `RequestProject/NoisyInstrument.lean` | The design numbers in exact rational arithmetic, with soundness theorems and a worked record |
| `RequestProject/ProbePanel.lean` | Combining probes: closed forms for OR and AND panels, when a panel helps, and when it destroys the contrast |
| `RequestProject/PartNinety.lean` | The capstone, tying all of it back to the capacity law |

## 1. The contrast collapses to `τ·J`

A reporter of sensitivity `se` and specificity `sp` raises the rate of positive reads above the
disorder-free baseline by exactly

    τ · (se + sp − 1)        (`IDR.Noisy.readRate_sub`)

— the omitted population times Youden's index, never more than `τ`.  Everything downstream is a
function of this one number.

## 2. The price becomes quadratic

The read-out law of `n` independent molecules has mean `n·q` and variance exactly `n·q·(1−q)`
(`sum_recProb_cnt`, `sum_recProb_var`), and Chebyshev's inequality follows from those two
identities (`chebyshev`).  The midpoint counting test then has *both* error probabilities at most
`1/(n·Δ²)` at every finite `n` (`test_error_truth`, `test_error_baseline`), so

    ⌈1 / (α · (τ·J)²)⌉ molecules        (`samplesFor`, `reporter_power`)

suffice.  Where a state-resolving observation paid a logarithm, a realistic reporter pays a
square.  On the illustrative record of `RequestProject.Instrument` — omitted population `1/5`,
`α = 0.05`, a good reporter with `se = 0.90`, `sp = 0.95` — the design calls for **693 molecules**
against the **14** state-resolving observations of the idealised calculation
(`IDR.NoisyInstrument.demoNoisy_report`; the populations are stipulated, not measured).

Because the contrast is never known before the study, the sample size must be computed from a
*lower confidence bound* on it.  That is sound (`design_from_pilot`) precisely because the sample
size is antitone in the contrast (`samplesFor_antitone`), so a conservative estimate can only
inflate the run.

## 3. The reporter is a physical object

A probe does not have one sensitivity; it has a firing probability per conformation.  The number
the design must use is then the **population-weighted mean** over the omitted states:

    contrast = τ · (se_eff + sp − 1),   se_eff = Σ_{x∈A} w_x p_x / τ     (`Reporter.contrast_eq`)

Two probes with the same nominal sensitivity but different state preferences give different
contrasts on the same target.  Worse, states the probe cannot see do not merely fail to
contribute: each unit of dark population contributes `−(1−sp)`.  Once the dark population
outweighs the bright one in the ratio `sp : (1−sp)` the contrast is **negative**
(`Reporter.contrast_neg_of_dark`): the count moves the wrong way and the test rejects a false
baseline *less* often than a true one.  A probe must therefore be validated for *coverage* of the
omitted states, not only for affinity.  Incomplete labelling multiplies the contrast by the
labelled fraction `d` (`Reporter.contrast_labelled`), so the molecule count scales as `1/d²`.

## 4. What no sample size can buy

The sharpest statement in Part XC is a limitation.  A system whose omitted set carries population
`τ`, read by a well-calibrated reporter, and a system with *no* such population, read by a
reporter whose specificity is lower by `τ·J`, generate the **identical** law on data records.
Every decision rule, at every sample size, is exactly as likely to reject in the two worlds
(`calibration_is_the_binding_constraint`).  Hence the reporter's specificity must be known to
better than `τ·J` — a requirement on the hardware.  `NoisyInstrument` prices the calibration run
that meets it (`calibrationMolecules_sound`), and shows the tolerance is a threshold rather than
conservatism (`specTolerance_sharp`).

## 5. A converse, so the numbers are about the experiment and not about one statistic

An upper bound invites the reply that a cleverer analysis would need far fewer molecules.  The
`ℓ¹` distance between the two `n`-fold data laws is at most `2·n·Δ` (`l1_dist_le`, proved by a
hybrid argument), so for **every** decision rule the two error probabilities sum to at least
`1 − n·Δ` (`test_error_sum_ge`).  Consequently a study claiming both errors at most `α < 1/2`
must have observed at least `(1 − 2α)/Δ` molecules, whatever statistic it used
(`molecules_lower_bound`).  The decidability window is therefore

    (1 − 2α)/Δ   ≤   molecules   ≤   ⌈1/(α·Δ²)⌉        (`detection_window`)

and both ends degrade with the reporter, through the same `Δ = τ·J`.

Finally, the bound counts *independent* molecules.  Reading one molecule `c` times gives a record
with `g·c` entries but no more information than the `g` molecules carried: any rule applied to it
still has errors summing to at least `1 − g·Δ` (`replicate_no_help`, a corollary of a data
processing statement).  Photon bins, frames and repeated interrogations are not sample size.

## 6. Combining probes

The natural response to a poor reporter is to use several.  Modelling independent probes by the
usual product formula, Youden's index of an OR panel (score positive if any probe fires) is
`∏ sp_i − ∏ (1 − se_i)`, and of an AND panel `∏ se_i − ∏ (1 − sp_i)` (`youden_or_eq`,
`youden_and_eq`); with one probe both reduce to `se + sp − 1`.  A second probe with no false
positives can only help (`youden_or_ge_of_clean`), and the exact criterion for a second probe to
*hurt* is `sp₁·(1 − sp₂) > (1 − se₁)·se₂` (`youden_or_lt_iff`) — the false positives it brings
against the true positives it adds.

The warning is asymptotic and it is sharp: an OR panel's index is at most `∏ sp_i` and an AND
panel's at most `∏ se_i` (`youden_or_le_pow`, `youden_and_le_pow`), so with `k` identical
imperfect probes both decay exponentially in `k`, and the molecule count `⌈1/(α(τJ)²)⌉` grows
exponentially with them.  Panels of many dirty probes are not a route to contrast; one clean
probe is.

## What is still not here

No number in this repository is a measurement.  The populations, the reporter characteristics and
the illustrative records are stipulated; `demoNoisy` is arithmetic on a worked example, not a
result about a protein.  What Part XC adds is not evidence but cost accounting: it says how many
molecules a real instrument needs, which average of the probe's response enters that number, what
happens when the probe is blind to part of the target, how well the instrument must be
calibrated before counting helps at all, and how few molecules can never be enough for anyone.
