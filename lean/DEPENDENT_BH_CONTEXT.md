# Part XCIV — The harmonic correction, proved and shown to be necessary

Files: `RequestProject/DependentBH.lean` (the theorem), `RequestProject/DependentBHSharp.lean`
(the matching instance).

## What was missing

Two earlier parts control the false discovery rate of a proteome-scale screen for disordered
regions, and each leaves the same hole.

* Part XCI (`FalseDiscovery.lean`) proves the Benjamini–Hochberg theorem in a **product**
  experiment — one independent coordinate per candidate region.
* Part XCIII (`DependentScreen.lean`) drops independence entirely, but changes the currency: each
  candidate must supply an **e-value**, not a p-value.

A real screen reports p-values, and its candidates share a calibration run, a plate and a batch,
so the dependence is neither absent nor sign-constrained. Both files recorded the gap in the same
words: *"under arbitrary dependence the Benjamini–Hochberg procedure needs the harmonic
correction, which is not proved here."* Part XCIV proves it, in the same finite,
measure-theory-free setting (`Law Ω` — an arbitrary probability distribution on a finite outcome
space, reused from Part XCIII).

## The assumption, and only it

`Superuniform P X` says `P(X ≤ t) ≤ t` for every `t ≥ 0`: the p-value of a **true null** is valid.
Nothing is assumed about the non-nulls, about the number of candidates, or — this is the point —
about the joint law. `superuniform_grid` verifies the condition for the exact p-value of a
discrete uniform read-out, so the hypotheses are satisfiable; `not_superuniform_of_const` shows
they have content.

## The argument

Write `R(ω)` for the reported list and `H₀` for the true nulls. The false discovery proportion is
the total weight `∑_{i ∈ H₀} 1{i ∈ R}/|R|` (`fdp_eq_sum_wgt`), so it suffices to bound, for one
null `i`, the expectation of `1{i ∈ R}/|R|`.

A step-up rule at level `q` is **self-consistent**: every reported candidate has `p_i ≤ |R|·q/m`
(`SelfConsistentP`; the BH list is of this shape, `bhList_selfConsistent`). The whole difficulty is
that `1/|R|` couples candidate `i` to all the others. The mechanism (`wgt_decomp`) removes the
coupling:

    1{i ∈ R}/|R| = Σ_{j=1}^{m} (1/j − 1/(j+1))·1{i ∈ R, |R| ≤ j} + (1/(m+1))·1{i ∈ R},

a telescoping identity on every single outcome. The events appearing on the right are *nested*,
not disjoint slices `{|R| = k}`; and self-consistency turns each of them into a statement about
candidate `i` alone,

    {i ∈ R, |R| ≤ j} ⊆ {p_i ≤ j·q/m},

whose probability is at most `j·q/m` by validity. Summing the telescoping coefficients gives
exactly the harmonic number:

    E[1{i ∈ R}/|R|] ≤ (q/m)·H_m      (`mean_wgt_le`),

because `Σ_{j≤m} (1/j − 1/(j+1))·j + m/(m+1) = H_m`. No independence, no positive-dependence
condition, and no property of the joint law is used anywhere — which is why the theorem holds for
arbitrary dependence.

Summing over the nulls:

* `selfConsistent_fdr_le_harmonic` — any self-consistent step-up rule: `E[FDP] ≤ q·H_m·|H₀|/m`.
* `benjamini_yekutieli`, `benjamini_yekutieli_level` — BH run at `α/H_m`: `E[FDP] ≤ α·|H₀|/m ≤ α`.
* `bh_uncorrected_bound` — uncorrected BH at level `q`: `E[FDP] ≤ q·H_m`.
* `by_dominates_bonferroni` — the corrected list still contains the Bonferroni list at the same
  overall level, so the correction never loses a discovery to the family-wise procedure.
* `harm_le_one_add_log`, `log_le_harm` — the price is `log(m+1) ≤ H_m ≤ 1 + log m`: logarithmic in
  the size of the screen, not linear as Bonferroni's `m` is.

## The factor is real

A guarantee inflated by `H_m` is worth paying for only if the inflation is not an artefact of the
proof. `DependentBHSharp.lean` settles this with an explicit construction, for every screen size
`m` and every level `q`.

Outcomes are `none` together with the pairs `(j, s)`, `j, s : Fin m`. In outcome `(j, s)` the
candidates of the cyclic window `win j s` of length `j+1` starting at `s` have p-value
`(j+1)q/m` and everyone else has p-value `1`; in outcome `none` all p-values are `1`. Outcome
`(j, s)` carries probability `q/((j+1)·m)`.

* The construction is balanced: each candidate lies in exactly `j+1` of the `m` windows of length
  `j+1` (`card_windowsThrough`), so it sees the value `(j+1)q/m` with probability exactly `q/m`,
  for every `j`. Hence every one of the `m` p-values is valid (`pval_superuniform`) — all `m`
  hypotheses are true nulls and the hypotheses of the theorem hold.
* In outcome `(j, s)` the observed p-values are `j+1` copies of `(j+1)q/m`, which is exactly the
  step-up fixed point: BH stops at `j+1` and rejects precisely the window (`bhR_pv`, `bhRej_pv`).
  Every discovery is false, so `FDP = 1` (`fdp_some`); in outcome `none` nothing is reported
  (`fdp_none`).
* The discovery outcomes of window length `j+1` carry total mass `q/(j+1)`, so

      E[FDP] = Σ_{j=1}^{m} q/j = q·H_m     (`bh_fdr_eq_harmonic`).

Consequences: the bound `bh_uncorrected_bound` is attained (`harmonic_factor_sharp`); from two
candidates on, uncorrected BH at level `q` has false discovery rate strictly above `q`
(`uncorrected_bh_exceeds_level`), so under arbitrary dependence the deflation is necessity rather
than conservatism; and the corrected procedure sits exactly at its own bound on the same instance
(`by_level_is_attained`), so `benjamini_yekutieli_level` cannot be improved either.
`hypotheses_satisfiable` exhibits an admissible level for every `m ≥ 2`, so none of this is
vacuous.

## Scope

This is multiplicity control, and it is orthogonal to calibration. `Superuniform` is a statement
about the *null law of the read-out*; if the instrument's baseline rate is wrong, the p-values are
not valid and every bound above is void. The calibration floor of Part XC is per candidate and is
not softened by anything here.

All statements are machine-checked with no `sorry` and no added axioms (`#print axioms` reports
only `propext`, `Classical.choice`, `Quot.sound`).
