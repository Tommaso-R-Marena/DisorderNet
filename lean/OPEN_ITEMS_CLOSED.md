# The five remaining items: Parts CX–CXV

`LIMITATIONS_CLOSED.md` ended with a list of five things the development did not claim.  This
note records what has been proved about each.  Everything referenced below is a machine-checked
Lean theorem, free of `sorry` and of non-standard axioms (`propext`, `Classical.choice`,
`Quot.sound` only).

New files: `RequestProject/MBARRate.lean` (Part CX), `ManyElectron.lean` (Part CXI),
`WaterTheory.lean` (Part CXII), `NullCalibration.lean` (Part CXIII), `ConnectiveExact.lean`
(Part CXIV), `ConnectiveLower.lean` (Part CXV).

## 1. A convergence rate for the MBAR iteration (Part CX) — closed

*Was:* the MBAR/WHAM self-consistent iteration was shown to have a fixed point, with no rate.

`MBARRate.lean` proves the iteration is a contraction in the *spread* (the difference between the
largest and smallest component of the free-energy error).  `spread_contraction` shows the spread
is multiplied by at most `1 − 2·eps` at each sweep, where `eps` is the overlap constant
`mixWeight` — the smallest normalised weight any sample has in any window.  `spread_iterate`
iterates it, `freeEnergy_error_le` converts it into an explicit bound on the free-energy error
after `k` sweeps, and `freeEnergy_error_tendsto_zero` gives geometric convergence.  The rate is
governed entirely by window overlap, which is the physical statement one wants: windows that do
not overlap give `eps = 0` and no contraction.

## 2. A many-electron treatment of reactive chemistry (Part CXI) — closed

*Was:* bond making and breaking was modelled by a two-state (diabatic) surface, Part CVIII.

`ManyElectron.lean` builds the Hubbard dimer: two electrons in two orbitals, the full space of six
determinants, Hamiltonian `hub U t`.  `feshbach_exact` is the exact block (Löwdin) downfolding of
the many-electron problem onto the covalent subspace; `groundEnergy_char` and `hub_ground_eigen`
identify the ground state and `hub_qform_ge` proves the variational principle for it.
`groundEnergy_eq_adiaLow` then shows that the two-state model of Part CVIII is not an
approximation but *exactly* the many-electron answer with coupling `V = 2t` — the earlier part is
thereby derived rather than assumed.  `singlet_ground` identifies the spin state,
`superexchange` gives the strong-correlation law `|E − (−4t²/U)| ≤ 16t⁴/U³` for `4|t| ≤ U`, and
`rhf_ge_ground`, `rhf_error_ge`, `rhf_error_dissociation` prove that restricted Hartree–Fock — a
single determinant — is wrong by an amount that grows without bound at dissociation.

## 3. A theory of water (Part CXII) — closed

*Was:* only the exact finite-solvent potential of mean force (Part LX), with no theory of the
solvent itself.

`WaterTheory.lean` solves an orientational hydrogen-bond chain exactly by transfer matrix
(`sum_hbWeight`, `Zcond_eq`, `Zchain_eq`), extracts the free energy per molecule in the
thermodynamic limit (`log_Zchain_div_tendsto`) and the hydrogen-bonded fraction
(`bondFraction_*`).  A two-state (open/dense) model then yields the density anomaly:
`pOpen_strictAnti`, `density_anomaly`, `density_anomaly_nonvacuous` and `exists_density_maximum`
prove that the density has an interior maximum — water is densest at an intermediate temperature.
Hydrophobic hydration is treated by `solv_pos`, `solv_le_entropic`, `solv_strictMono`,
`solv_tendsto_zero_cold`, and `cold_denat`: the hydrophobic driving force weakens on cooling and
vanishes, which is cold denaturation.

## 4. External calibration of the null read rate (Part CXIII) — closed, in both directions

*Was:* the sequential and multiplicity results assumed the null read rates were known;
Parts XC–CIX said explicitly that neither sequential analysis nor multiplicity control repairs
calibration.

`NullCalibration.lean` proves that statement and then discharges the requirement with a protocol.
`uncalibrated_no_power`: a rejection rule that is valid at level `alpha` for *every* possible null
rate has power at most `alpha` against every alternative — with an unknown null, validity forces
powerlessness, whatever machinery is layered on top.  In the other direction,
`prob_mono_of_upward` proves that the rejection probability of any upward-closed region is
monotone in the read rate, so `plugin_valid` shows a conservative externally supplied null rate is
sound; `seqReject_upward` and `countReject_upward` confirm that both anytime sequential rules and
fixed-horizon count rules are upward closed, so the guarantee applies to them.
`calibrated_test_valid` and `calibrated_test_valid_samples` give the finite protocol: `m` control
reads plus a conservative plug-in rate bound the total type-I error by `alpha + 1/(4·m·eps²)`, and
by `alpha + delta` at the design sample size.  Calibration is therefore an experimental
requirement with an explicit price, not an assumption.

## 5. Connective constants and the Flory exponent (Parts CXIV, CXV) — partly closed; the exact values remain open

The exact value of the connective constant `μ` of `ℤ²` and the exact Flory exponent `ν` are open
problems of mathematics.  They are not claimed here.  What is proved:

* **An exactly solvable lattice.**  On the directed square lattice (`ddir`, east and north only)
  every walk is self-avoiding, so `cntOf_ddir` gives the count exactly, `2ⁿ`, and
  `connectiveConstant_directed` gives the connective constant exactly, `log 2`.  `directed_extent`
  shows such a walk ends at distance exactly `n`: the Flory exponent of the directed lattice is
  exactly `1`.
* **A rigorous two-sided bracket for the genuine square lattice.**  `connectiveConstant_bracket`:
  `log 251 / 7 ≤ μ ≤ log 780 / 6`, i.e. `0.789… ≤ μ ≤ 1.110…`.  The upper bound comes from the
  exact count `cnt 6 = 780` with the submultiplicativity of Part XII.0; the lower bound comes from
  the *bridge* construction of Part CXV.  `IsBridge` isolates the walks that leave the line `x=0`
  at once and never pass to the right of their own endpoint, `bridge_append` proves that two such
  walks concatenate to a self-avoiding walk (the geometric heart of the argument), and
  `bcnt_supermultiplicative` turns this into supermultiplicativity of the bridge count — the exact
  mirror of the submultiplicativity used for the upper bound.  `log_bcnt_div_le_connectiveConstant`
  makes every finite bridge count into a rigorous lower bound, dual to
  `connectiveConstant_le_div`; with `bcnt 7 = 251` this gives the bracket above.  Both bounds
  strictly improve the previously available pair `log 2 ≤ μ ≤ log 100 / 4`
  (`improves_on_log_two`, `improves_on_cnt_four`), and both constructions convert any further
  finite computation into a sharper bracket.  No finite computation determines `μ`.
* **A deterministic Flory window.**  `length_succ_le_extent_sq` is the exact box bound: a
  self-avoiding walk of `n` steps occupies `n + 1` distinct sites, which cannot fit in a box of
  side `2R + 1` unless `(2R+1)² ≥ n + 1`.  Hence `sqrt_le_extent` and `flory_window`: *every*
  `n`-step self-avoiding walk with `n ≥ 4` has extension between `√n / 4` and `n` — the swelling
  exponent of an individual conformation is confined to `[1/2, 1]`, with no ensemble average and
  no probability involved.  `saw_flory_window` states this for the conformations counted by `cnt`,
  and `length_succ_le_extent3_cube`, `cubic_flory_exponent_third` give the three-dimensional
  version, `ν ≥ 1/3`.  The conjectured typical values `ν = 3/4` in two dimensions and
  `ν ≈ 0.588` in three are outside this window's resolution and remain open.

## What is still open

Stated without softening: the exact value of the connective constant of `ℤ²` (and of `ℤ³`) and the
exact Flory exponent — genuinely open mathematics, of which Parts CXIV and CXV deliver exact
values in the solvable case and rigorous brackets otherwise; and the empirical question of what
any particular protein does, about which nothing in this development is claimed.
