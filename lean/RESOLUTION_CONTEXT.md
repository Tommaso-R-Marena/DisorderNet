# Parts CXXXIV–CXXXVI — Resolution: what a *noisy* salt series identifies, and which probe to use instead

Parts CXXIX–CXXXIII (`SOLUTION_CONTEXT.md`) settle the noiseless identifiability question for a
charged disordered region: the screened energy at inverse screening length `κ` is exactly the
Dirichlet series

    E(κ) = ∑_{d=1}^{N−1} d · e^{−κd} · C(d),      C(d) = ∑_i q_i q_{i+d}

in the charge autocorrelations, a titration over infinitely many conditions determines every
`C(d)`, and `N − 1` conditions are needed to do it with finitely many.  Those statements assume
an exact read-out.  These three parts add the noise, and the picture changes qualitatively.

| File | Content |
| --- | --- |
| `RequestProject/TitrationResolution.lean` | Part CXXXIV — the titration is exponentially ill-conditioned |
| `RequestProject/ResolutionDesign.lean` | Part CXXXV — the same law in numbers, and the horizon on sequence separation |
| `RequestProject/ContactPanel.lean` | Part CXXXVI — a separation-resolved probe panel, stably invertible |
| `RequestProject/DebyeResolution.lean` | Part CXXXVII — how severe the ill-conditioning is depends on the distance law |

## 1. Each lag enters the data through an exponentially small factor

The lag-`D` correlation reaches the observable only through `D e^{−κD}`.  Perturbing the profile
at that one lag by `δ` moves the whole curve by exactly `D e^{−κD} δ`
(`IDR.Resolution.single_lag_gap`), so if the accessible conditions are `κ ≥ κ₀` and the energy
resolution is `eps`, a perturbation as large as

    δ = eps · e^{κ₀ D} / D

moves no measurement by as much as the resolution — at *any* accessible condition, not merely at
the ones that were measured (`single_lag_invisible`, `resolution_floor`).  This is a floor on
what the experiment can report, not an artefact of a fitting procedure.

Two sharp complements make the statement exact rather than merely an estimate:

* `separating_condition_below_threshold` — a condition that *does* separate a lag-`D` discrepancy
  `δ` must satisfy `κ < log(D|δ|/eps)/D`.  Up to the logarithm, the Debye screening length has to
  be as long as the sequence separation one wants to probe.
* `detectable_at_low_salt` — and every condition at or below that threshold does separate it.
  The threshold is the truth, not a bound.

Over regions of unbounded length there is then no uniform recovery at all
(`no_uniform_profile_recovery`): for any accuracy target there are a length and two profiles that
are `eps`-indistinguishable at every accessible condition and differ by more than the target.

## 2. The concrete instance: the two ends of the region

Take a unit charge at each end of an `N`-residue region, and compare it with a unit charge at one
end only.  Their titration curves differ by exactly `(N−1) e^{−κ(N−1)}`
(`IDR.Resolution.endpoint_pair_gap`): a signal of `N − 1` at zero salt
(`endpoint_pair_gap_zero_salt`), and below any resolution `eps` as soon as
`(N−1) e^{−κ₀(N−1)} ≤ eps` (`endpoint_pair_invisible`).

Part CXXXV puts numbers on it.  For a twenty-residue region read to `10⁻³ kT` from conditions
whose Debye length is one residue spacing (`κ₀ = 1` in inverse residue units):

* the lag-10 correlation is undetermined over a range of at least two charge units
  (`demo_lag10_floor`);
* the lag-15 correlation is undetermined over its *entire* physical range `[−20, 20]`
  (`demo_lag15_unconstrained`);
* the end-to-end contact is worth exactly `19 kT` unscreened and stays below `10⁻³ kT` at every
  accessible condition (`demo_endpoint_zero_salt`, `demo_endpoint_invisible`).

The general form is `lag_unconstrained_of_bounded`: a lag whose a-priori range is already below
the floor is not constrained by the data at all; and `high_lags_unconstrained` gives the explicit
horizon `D₀ ≈ 4B/(eps κ₀²)` beyond which that is the case for every lag, whatever the region, the
buffer and the instrument.

Read against Part CXXXIII, the two axes of the design are now both fixed: the *number* of
conditions bounds how many correlations are identified, the *lowest* condition bounds which ones.

## 3. The positive complement: a separation-resolved panel inverts with an absolute constant

None of this says the long-range correlations of a disordered region are unknowable.  It says the
screened energy is the wrong observable for them, because screening is an exponential filter on
sequence separation.  Part CXXXVI exhibits an observable of the *same pairwise model class* with
no such filter: a crosslinking-style reagent of reach `D`, whose kernel is a step in separation
and whose reading is the cumulative correlation `S(D) = ∑_{d≤D} C(d)`
(`IDR.ContactPanel.panelReading_eq_partial_sum`).

* `panel_inverts` — the inversion is the first difference, `C(D) = S(D) − S(D−1)`.
* `panel_stability` — if two regions' readings agree to `eps`, their correlations agree to
  `2 eps` at every lag.  The constant is `2`: independent of the lag, of the length of the
  region, and of the solution condition.  The titration's corresponding constant is
  `e^{κ₀D}/D`.
* `panel_identifies` — exact readings determine the autocorrelation exactly, hence (Part LXXIII)
  the pairwise energetics under every separation kernel and every ionic strength.
* `panel_resolves_lag15` against `titration_leaves_lag15_free` — the same worked case, the two
  observables side by side: at resolution `10⁻³` the panel pins the lag-15 correlation of a
  twenty-residue region to `±2·10⁻³`, the titration leaves it free over `[−20, 20]`.

The panel has a limitation of its own, and it is a different one: its *reach*.  A panel whose
longest reagent spans `R` says nothing about any lag beyond `R` (`panel_reach_blind`), and
`blind_to_lag_without_reach_or_low_salt` puts the two observables' blind spots together — against
a discrepancy at separation `D₀`, a panel of reach `R < D₀` and a titration confined to
`κ ≥ log(D₀|δ|/eps)/D₀` are both blind.  To learn about a separation one needs either a reagent
that spans it, or an ionic strength whose Debye length does.

## 3½. How severe?  The answer depends on the assumed distance law

Parts CXXXIV–CXXXV use the Debye-damped *linear chain* kernel `d e^{−κd}` of Part CXXVI, whose
screening exponent is proportional to the sequence separation.  The standard Debye–Hückel
coupling on a Gaussian chain has exponent `κ b √d`, because residues `d` apart are a spatial
distance `b√d` apart.  Part CXXXVII redoes the analysis for that kernel
(`IDR.DebyeResolution`): the floor at lag `D` is `eps · b√D · e^{κ₀ b √D}`
(`debye_single_lag_invisible`), it is exact (`debye_detectable`), and a horizon still exists
(`debye_high_lags_unconstrained`) — but it sits at separation of order `(log(B/eps)/(κb))²`
instead of `log(B/eps)/κ`, the square.  Concretely (`demo_debye_lag16_resolved`), at `b = 1`,
`κ = 1` and resolution `10⁻³` a lag-16 discrepancy of `1/4` is *detected* at that single
condition, whereas the exponentially screened chain kernel leaves an entire physical range free
at lag 15.

Ill-conditioning is therefore generic — any kernel decaying in separation filters the long lags —
but its severity is a property of the kernel, not of the region.  A model must declare which
distance law it assumes before it can quote a resolution on a long-range correlation.

## 4. What this adds to the design of a model

A model of a charged disordered region should carry, with each fitted long-range charge
correlation, the resolution floor `eps · e^{κ₀ D} / D` implied by the lowest ionic strength of its
calibration set — and should treat everything below that floor as unconstrained, never as a
measured zero.  Where the long-range correlations matter, they must be calibrated against a
separation-resolved observable; `probe_choice_law` is the statement that this is not a preference
but a difference between a `2`-Lipschitz inversion and an exponentially ill-conditioned one.

Nothing here weakens the earlier limitative results: the autocorrelation is still not a complete
description of the sequence (Part CXXVIII's homometric pair shares it at every lag), so a stably
measured autocorrelation is a stably measured *projection*, and the many-body requirement stands.
