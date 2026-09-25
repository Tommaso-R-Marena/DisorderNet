# Parts CXXIX–CXXXIII — The solution context: what a salt series identifies, and what condensation hides

Parts CXX–CXXVIII solve the electrostatics of a charged disordered region exactly and show that
its properties are set by the *order* of its charges and by the ionic strength of the buffer.
Those parts fix the buffer and vary the sequence.  Parts CXXIX–CXXXII do the opposite: they fix
the sequence and ask what an experiment that varies the buffer can, and cannot, determine.  Every
statement is machine-checked in Lean, exactly, with no asymptotics.

| File | Content |
| --- | --- |
| `RequestProject/SaltTitration.lean` | Part CXXIX — identifiability of the charge autocorrelations from a salt series |
| `RequestProject/Manning.lean` | Part CXXX — counterion condensation: saturation, unidentifiability of the bare charge, and the ceiling |
| `RequestProject/IonicStrength.lean` | Part CXXXI — ionic strength versus salt concentration as the model's context input |
| `RequestProject/SolutionContext.lean` | Part CXXXII — the capstone, `solution_context_law` |
| `RequestProject/TitrationDesign.lean` | Part CXXXIII — the design numbers: how many conditions, and how far up in salt |

## 1. The titration curve is a Dirichlet series in the autocorrelations

With the Debye-damped chain kernel of Part CXXVI, the measured energy at inverse screening
length `κ` is exactly

    E(κ) = ∑_{d=1}^{N−1} d · e^{−κd} · C(d),      C(d) = ∑_i q_i q_{i+d}

(`IDR.Titration.energy_eq_sum_autocorr`).  The pairwise model therefore reaches the data only
through the `N − 1` charge autocorrelations: no experiment in this class can constrain anything
finer.

## 2. A salt series identifies exactly those `N − 1` numbers

`IDR.Titration.autocorr_eq_of_energy_eq_on_infinite` — if two sequences give the same energy at
**infinitely many** ionic strengths (an interval of concentrations, say), their autocorrelations
agree at every lag.  The proof substitutes `x = e^{−κ}`, turning the titration curve into a
polynomial with infinitely many roots.

The converse (`energy_eq_of_autocorr_eq`) gives the exact statement `titration_iff`, and
`titration_determines_every_kernel` draws the payoff: a salt series determines the energy under
*every* separation kernel, including the salt-free limit that no experiment reaches.
Extrapolation off the measured conditions is legitimate — inside the pairwise class.

## 3. But one condition identifies nothing, and a short series is underdetermined

* `single_condition_degenerate` — at any `κ ≥ 0`, the unit-charge sequences `(+1, 0, 0)` and
  `(+1, −e^{−κ}, +1)` have *exactly* equal energy and different autocorrelations;
  `single_condition_needs_a_second` shows a separating condition must exist elsewhere.
* `finite_titration_underdetermined`, `finite_titration_two_profiles` — with `k` conditions and
  `k + 1 < N` there is a whole correlation profile invisible to the experiment.  `N − 1`
  independent conditions are needed to resolve `N − 1` lags.
* `energy_high_salt_bound`, `high_salt_uninformative` — the curve decays like `N³e^{−κ}`, so an
  instrument of resolution `eps` sees nothing beyond `κ = log(2N³/eps)`.  The usable range of a
  titration grows only *logarithmically* with the precision of the calorimeter, which is why the
  conditions must be well separated and low-to-moderate.

## 4. Counterion condensation caps the signal

Above the Manning threshold — one charge per Bjerrum length — the effective charge density is
`b/lB`, independent of the bare density (`IDR.Manning.effDensity_above`).  Hence:

* `manning_saturation`, `bare_charge_unidentifiable` — two regions of *different* bare charge
  density, both above threshold, have identical energies at every ionic strength and under every
  kernel.  The bare density of a strongly charged region is not an observable of this class.
* `manning_quadratic_below` — below threshold the signal grows exactly as `s²`, so a mutagenesis
  series in charge density is informative only in the weakly charged regime.
* `manning_ceiling`, `patterning_contrast_ceiling` — `|E| ≤ (b/lB)² · 4N/κ²` uniformly over all
  bare densities and all patterns.  Charge patterning has a bounded thermodynamic budget, set by
  the solvent and the chain geometry rather than by the sequence.

## 5. The context input is the ionic strength

With `κ = A√I` and `I = ½ ∑ c_i z_i²`:

* `equal_ionicStrength_equal_prediction` — buffers of equal ionic strength are exactly
  indistinguishable, whatever their composition; `divalent_equivalent_concentration` — a `2:2`
  salt at `c` screens like a `1:1` salt at `4c`.
* `concentration_blind_error`, `no_concentration_blind_model` — at the same *concentration*, a
  monovalent and a divalent buffer differ by exactly `e^{−A√c} − e^{−2A√c} > 0`, so a predictor
  told only the concentration is wrong by at least half that gap on one of them.

## Capstone

`IDR.SolutionContext.solution_context_law` states 1–5 in a single theorem.

## 6. The design numbers (Part CXXXIII)

* `IDR.TitrationDesign.identification_requires_at_least` — if a set of `k` ionic strengths leaves
  no non-trivial correlation profile invisible, then `N − 1 ≤ k`: one condition per lag, with no
  discount for a clever choice of conditions.
* `demo_conditions_needed`, `demo_usable_range`, `demo_report` — the worked case of a
  twenty-residue region read out to `10⁻³ kT`: eighteen conditions always leave a blind
  direction, and every inverse screening length above `17` (in inverse residue units) is useless,
  because no two unit-charge sequences differ there by as much as the resolution.  The window
  grows only as `log(1/eps)`, so precision does not substitute for conditions.

## What is not claimed

The model class throughout is pairwise and separation-dependent, as in Parts LXXIII and CXX; the
identifiability statements are statements *within* that class, and say nothing about many-body or
conformation-resolved contributions.  The condensation treatment uses the standard
`s_eff = s / max(1, lB·s/b)` prescription, which is a mean-field renormalisation, not an exact
solution of the counterion problem.  The instrument model in §3 is a plain energy resolution
`eps`; the reporter-level analysis of Part XC applies unchanged on top of it.
