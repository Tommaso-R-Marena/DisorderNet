# The distance law is part of the model (Part CXXXVIII)

`RequestProject/DistanceLaw.lean`.  Everything below is proved in Lean, with no `sorry` and only
the standard axioms.

Parts CXXXIV–CXXXVII read charge correlations off a salt titration and found that the answer
depends on the assumed separation kernel: with the chain kernel `d·e^{−κd}` the resolution horizon
sits at lag `≈ log(B/eps)/κ`, with the Debye–Hückel Gaussian-chain kernel `e^{−κb√d}/(b√d)` at
`≈ (log(B/eps)/(κb))²`.  Part CXXXVII therefore ended with an instruction: a model must *declare*
its distance law before quoting a resolution on a long-range correlation.  The obvious next
question is whether the distance law can itself be calibrated from the same experiment.  It has a
three-part answer.

## 1. The tool: exponentials with distinct rates are independent

`expSum_amplitudes_zero` / `expSum_eq_zero`.  If a finite sum `∑_j A_j e^{−κ a_j}` with pairwise
distinct rates `a_j` vanishes at every ionic strength on a half-line `κ ≥ κ₀`, then every `A_j`
is zero.  The proof is elementary and quantitative rather than asymptotic: the slowest-decaying
term dominates, and `e^{−x} ≤ 1/x` converts that domination into a bound `|A_{j₀}| ≤ M/(κ·gap)`
valid at every accessible condition, which forces `A_{j₀} = 0`; then induct on the remaining
rates.  This is the piece the earlier parts could not use, because their identification arguments
went through polynomials in `e^{−κ}` and so needed *integer* screening rates — the Debye–Hückel
rates `b√d` are not integers.

## 2. Identification, for a declared kernel

`genCurve N w a c κ = ∑_{d=1}^{N−1} w(d)·e^{−κ a(d)}·c(d)` is the titration reading of a
correlation profile `c` under an arbitrary separation kernel with rates `a` and amplitudes `w`.

* `genCurve_identifies` — if the rates are pairwise distinct across lags and no amplitude
  vanishes, a complete titration determines the whole profile: two profiles agreeing at every
  condition `κ ≥ κ₀` agree at every lag.
* `chain_profile_identifiable`, `debye_profile_identifiable` — the two kernels of interest.

Read against Part CXXXIII this is the exact boundary of the positive result: *finitely* many
conditions never identify the profile (`Titration.finite_titration_underdetermined`), and the
whole half-line always does, however ill-conditioned the inversion is (Part CXXXIV still governs
what a *finite-resolution* titration can see).

## 3. The distance law is falsifiable

`chain_debye_forces_short_range` — on a three-residue window, a chain-kernel model with profile
`c` and a unit-bond-length Debye model with profile `c'` give the same reading at every ionic
strength only if `c(2) = c'(2) = 0` and `c(1) = c'(1)`: the two kernels agree only where neither
is really being used.  Equivalently `misspecified_kernel_detected`: a region with a non-zero lag-2
correlation, read with the chain kernel, is not reproduced by *any* Debye profile — some condition
separates them.  The three rates involved are `1, 2` (chain) and `1, √2` (Debye), and the argument
is exactly the independence of §1.

So the distance law is not a matter of taste that must be assumed and declared: it is a testable
part of the model, and a sufficiently complete titration tests it.

## 4. But the bond length is not identifiable with the profile

`bond_length_lag_confound` — a region whose only charge correlation sits at lag `2`, modelled with
bond length `b`, has *literally the same* Debye–Hückel titration curve, at every ionic strength,
as a region carrying the same correlation at lag `1` modelled with bond length `b√2`.  The two
curves are the same function of `κ`, because `b·√2 = (b√2)·√1`.

* `bond_length_not_identifiable` — hence two models differing in both bond length and profile that
  no resolution, no number of conditions, and no fitting procedure can separate.
* `even_lag_bond_length_confound` — and this is not an artefact of the toy window: over a region of
  any length, *every* profile supported on even lags is reproduced exactly at bond length `b√2` by
  the profile of its even lags.  `confound_halves_apparent_range` states the special case: a
  correlation at lag `2D` at bond length `b` is a correlation of the same size at lag `D` at bond
  length `b√2`.

The mechanism is that the Debye rate depends on `b` and `d` only through `b√d`, so a `√2` change
of bond length is exactly a halving of sequence separation.  It is a genuine confound of the
model's *geometry* with the region's *sequence*, and it is invisible to the experiment.

## 4½. What breaks the confound

The degeneracy lives entirely in profiles whose short-lag correlation vanishes — in the example of
§4 the `b`-model has no lag-1 correlation at all.  That is exactly the obstruction:

* `lag1_zero_of_bond_length_lt` — if two Debye models with bond lengths `b < b'` agree at every
  condition, then the lag-1 correlation of the `b`-model is zero.  The reason is the minimal-rate
  lemma of §1: the rate of lag 1 in the `b`-model is `b`, every other rate in either model is
  strictly larger (`b√d > b` for `d ≥ 2`, and `b'√d ≥ b' > b`), so nothing can cancel it.
* `bond_length_identifiable_of_lag1` — hence if both candidate profiles have a non-zero lag-1
  correlation, agreement at every condition forces `b = b'`.
* `bond_length_identified_given_profile` — the design rule.  Measure the correlation profile with
  the separation-resolved panel of Part CXXXVI, whose inversion is `2`-Lipschitz and does not
  involve the bond length at all; if its lag-1 value is non-zero, the titration then determines
  the bond length.  Two probes suffice where one does not.

## 5. What this adds to the design of a model

Three requirements, on top of those of Parts CXXXIV–CXXXVII:

1. Report the assumed distance law with every fitted correlation; the resolution floor quoted for
   a long-range correlation is meaningless without it.
2. Test that law rather than assume it, where the titration is complete enough: the chain kernel
   and the Gaussian-chain Debye kernel make different predictions across conditions unless the
   region is short-range.
3. Do not fit the bond length from the titration *alone*.  Fitted jointly with the correlation
   profile it trades one-for-one against sequence separation, and a model that fits both will
   report a sequence range that can be off by a factor of two with a perfect fit to the data.  The
   repair is a second, bond-length-free probe of the profile: with the profile known and non-zero
   at lag 1, the titration does determine the bond length.
