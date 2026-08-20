# Seventeen limitations closed: Parts XCIII–CIX

All statements below are machine-checked Lean theorems, free of `sorry` and of any non-standard
axiom.  Each section names an item that `PAPER.md` §7 recorded as *not claimed*, and the part that
now closes it.  Nothing here weakens an earlier statement: the new parts add hypotheses-free
generality where the old ones assumed it away, and where a limitation could only be replaced by a
sharper negative result, that is what was proved instead.

New files: `RequestProject/DependentScreen.lean`, `Imbalance.lean`, `CorrelatedCoverage.lean`,
`CrooksDerivation.lean`, `NonlinearIdentifiability.lean`, `ToleranceFit.lean`, `Ageing.lean`,
`MarkovEstimation.lean`, `PhotonCorrections.lean`, `ReplicaRoundTrip.lean`, `FibrilLength.lean`,
`ImageFormation.lean`, `MBAR.lean`, `Clustering.lean`, `AnharmonicIntegrator.lean`,
`BondBreaking.lean`, `DependentReads.lean`.

## 1. Screening without independence (Part XCIII)

*Was:* the false discovery rate theorem assumed one independent coordinate per candidate region.

`DependentScreen.lean` proves FDR control under **arbitrary dependence** by replacing p-values with
e-values: `ebh_fdr_le_level` gives the e-BH guarantee with no assumption whatsoever on how the
candidates are coupled, `selfConsistent_fdr_le` covers any self-consistent selection rule,
`bonferroni_fwer` gives the familywise version, and `isEValue_likelihoodRatio` exhibits the
e-values a real screen can compute.  The per-candidate calibration floor of Part XC is untouched by
multiplicity control, and the file says so where it is provable.

## 2. Class imbalance and the figures of merit (Part XCIV)

*Was:* the benchmark parts counted residues with equal weight; balanced accuracy, Matthews
correlation and per-protein aggregation were not modelled.

`Imbalance.lean` defines all of them and proves the inversions: `imbalance_inversion` (a predictor
with higher accuracy and lower balanced accuracy), `macro_micro_inversion` (macro- and
micro-averaging ranking two methods in opposite orders), together with the range and degeneracy
facts a reported correlation must satisfy (`mcc_le_one`, `neg_one_le_mcc`, `mcc_eq_zero_iff`).

## 3. Coverage for a correlated trajectory (Part XCV)

*Was:* the coverage bounds assumed independent draws, and a molecular-dynamics trajectory is
correlated.

`CorrelatedCoverage.lean` proves the missing-mass bound for a Markov sample under Doeblin
minorisation (`unseen_le`), recovers the independent bound as the special case (`unseen_le_iid`),
bounds the probability of avoiding a set over a block of steps (`avoid_block_le`), and exhibits the
frozen chain that saturates it (`avoid_eq_one_of_frozen`): correlation costs exactly the
minorisation constant, and nothing weaker is true.

## 4. Crooks derived, not assumed (Part XCVI)

*Was:* Crooks' relation was a hypothesis on a finite set of trajectories.

`CrooksDerivation.lean` derives it (`crooks_derived`, `crooks_derived_ratio`) from local detailed
balance of the individual steps, with the first law (`first_law`) and the reversal of the path
weights (`path_weight_ratio`, `revLaw_revPath`) proved on the way.  Every hypothesis is
load-bearing.

## 5. Identifiability of nonlinear reports (Part XCVII)

*Was:* the identifiability characterisation was for linear functionals of the ensemble.

`NonlinearIdentifiability.lean` treats reports that are not linear: `reachable_interval` computes
the exact interval of values a nonlinear report can take over the ensembles consistent with the
data, `determinedF_comp` shows a monotone relabelling of a determined report is determined, and
`not_determinedF_ratio` exhibits a ratio of populations that the data leaves free.

## 6. Fitting within error bars (Part XCVIII)

*Was:* the fitting results were for exact agreement with the measured averages.

`ToleranceFit.lean` replaces exact fits by fits within a tolerance: `isFitTol_of_isFit`,
`fitTol_convex`, `tol_fit_perturb` (the perturbation that stays inside the error bars),
`tol_fit_not_unique`, and `tol_fit_without_exact_fit` — the admissible set is larger, not smaller,
so every underdetermination statement survives the relaxation.

## 7. A region that never reaches equilibrium (Part XCIX)

*Was:* genuinely non-stationary (ageing, transient) behaviour was outside the scope.

`Ageing.lean` proves the lag bound for a time-inhomogeneous chain whose instantaneous equilibrium
itself moves: `ageing_lag` gives `‖p_t − π_t‖₁ ≤ (1−ε)^t‖p₀−π₀‖₁ + δ/ε`, a geometric transient plus
a permanent lag equal to drive speed over mixing rate; `l1_contract` derives the contraction from
minorisation rather than assuming it; `no_stationary_matches` exhibits a driven region no
time-independent ensemble reproduces.

## 8. Markov state models with finite statistics (Part C)

*Was:* statistical error in the transition counts, and the continuous-time case, were outside.

`MarkovEstimation.lean` propagates count noise into the answer: `stationary_perturb` bounds the
stationary-population error by `η/ε`, count error over spectral gap, `observable_perturb` carries
it to any bounded observable, `rows_within_of_counts` supplies `η` from the counts, and
`generator_stationary_iff` bridges to continuous time.

## 9. What the FRET detector actually counts (Part CI)

*Was:* background, crosstalk, gamma correction and direct excitation were not modelled.

`PhotonCorrections.lean` shows the raw proximity ratio is a Möbius function of the true efficiency
(`prPhys_moebius`), that the correction is exact when the parameters are known, that two physically
distinct instruments produce identical raw ratios at every efficiency
(`gamma_leakage_unidentifiable`) — so the parameters must come from a calibration measurement — and
that averaging does not commute with correcting (`avg_correction_noncommute`).

## 10. Replica-exchange round trips (Part CII)

*Was:* scheduling, and the round-trip statistics it controls, were not modelled.

`ReplicaRoundTrip.lean` proves the ballistic bound (`round_trip_time`: a round trip on a `K`-rung
ladder needs `2K` sweeps whatever the schedule) and the diffusive one (`reach_prob_le`: the
probability of having travelled `R` rungs in `T` sweeps is at most `T/R²`, so
`sweeps_needed_for_traversal` gives `R²/2` sweeps for even odds).  Adding rungs raises acceptance
but costs sweeps quadratically; the results hold for time-inhomogeneous, adaptively retuned
ladders.

## 11. Fragmentation, secondary nucleation, fibril lengths (Part CIII)

*Was:* the aggregation parts tracked total fibril mass only.

`FibrilLength.lean` models the population as a multiset of lengths: a split conserves mass exactly,
raises the fibril number by one and strictly lowers the mean length (`Splits.mass_eq`,
`Splits.card_eq`, `mean_lt_of_splits`), and the mass observable is blind to the distribution
(`mass_blind_to_length`).  For the channels, `channels_unidentifiable` shows one monomer
concentration cannot separate fragmentation from secondary nucleation and `channels_identified`
shows two concentrations can; `sec_dominates` separates the shapes.

## 12. Image formation in cryo-EM (Part CIV)

*Was:* no image-formation model, contrast transfer function, noise or alignment error; bulk solvent
and resolution truncation outside.

`ImageFormation.lean` proves the transfer function vanishes at `√(k/λ·df)` and that the micrograph
carries no information there (`ctf_zero_at`, `image_blind_at_zero`); that a commensurate defocus
pair shares every zero while an incommensurate one shares none
(`commensurate_defocus_shares_zeros`, `incommensurate_defocus_fills_zeros`), so
`two_defocus_recovers` applies; that a map processed without a solvent model returns an occupancy
low by exactly `1 − ρ_s/ρ_p` (`occupancy_solvent_bias`); that alignment error of `±d` mimics
occupancy loss by `exp(−d²/2σ²)` (`alignment_peak_drop`); that truncated data is blind to
high-frequency content (`truncation_blind`); and that the particle count needed grows as the
inverse square of the transfer function (`particles_needed`).

## 13. The estimator behind the umbrella windows (Part CV)

*Was:* the WHAM/MBAR self-consistency iteration and its convergence were not treated.

`MBAR.lean` proves the equations exactly scale invariant, so only free energy differences exist
(`mbarMap_smul`, `freeEnergy_diff_eq`); that the true partition functions are a fixed point in the
infinite-sampling limit (`mbar_consistent`); that the positive fixed point is unique up to that
scale (`mbar_unique_up_to_scale`); and that the iteration is non-expansive in the ratio bracket
around a solution (`mbar_iter_bracket`).  A convergence *rate* is still not proved and is stated as
open.

## 14. Whatever the clustering, the timescales come out short (Part CVI)

*Was:* the Markov state model results were for a fixed clustering; the choice of clustering was
outside.

`Clustering.lean` proves the lumped chain is the exact restriction of the microscopic chain to
observables constant on clusters (`lump_inner_eq`, `lump_mean_eq`, `lump_form_eq`), so every
relaxation bound of the true chain is inherited by the model (`lump_gap_transfer`): the implied
timescale is a lower bound for *every* clustering.  Lumping twice is lumping by the composite map
(`lump_comp`, `lumpPi_comp`), so coarsening only makes it shorter (`lump_refine_transfer`).  An
implied-timescale plateau is a lower bound that has stopped improving, not a validation.

## 15. Integrators beyond the Gaussian case (Part CVII)

*Was:* the integrator results were exact for the harmonic mode and were not theorems about
anharmonic force fields.

`AnharmonicIntegrator.lean` proves velocity Verlet exactly reversible for every force
(`verlet_reversible`); exhibits the exactly conserved shadow energy of the harmonic mode
(`harmonic_shadow_invariant`) and the boundedness it gives below `dt·ω < 2` (`harmonic_bounded`);
and proves that for the quartic oscillator no timestep is unconditionally stable
(`quartic_unstable`): for every `dt > 0` some trajectory grows at least like `5ⁿ`.

## 16. Bonds that break (Part CVIII)

*Was:* quantum-mechanical bond making and breaking remained outside the molecular model.

`BondBreaking.lean` supplies the two-state (diabatic) model: `adiabatic_char` verifies the reactive
surface is an eigenvalue, `adiaLow_le_left`/`adiaLow_le_right` place it below both bonded surfaces
everywhere, `barrier_lowering` computes the barrier reduction at the crossing as exactly the
coupling `|V|`, `no_crossing` shows the surfaces never touch, and `harmonic_cannot_dissociate`
shows a harmonic bond is wrong by more than any tolerance at large separation.  A many-electron
treatment remains outside.

## 17. Stopping when you like, with reads that are not independent (Part CIX)

*Was:* the anytime-valid sequential test assumed the reads were independent.

`DependentReads.lean` replaces the single null rate by a **predictable conditional rate**, so the
probability that the next read is positive may depend on the entire history in an arbitrary way.
`sum_probD` shows that this defines a law, `wealthD_eq_ratio` that the wealth is still the
likelihood ratio, and `villeD` that Ville's inequality survives unchanged; `anytime_validD` and
`anytime_validD_all_horizons` give the test, and `ville_bursty_reads` instantiates it on reads that
are strongly correlated by construction.  The second hypothesis of that item — that the null rates
are known — is untouched, and the file says so: sequential analysis does not repair calibration.

---

## What is still open

The five items listed here in the previous revision — the exact connective constants and Flory
exponent for genuine self-avoiding walks, a theory of water, a many-electron treatment of reactive
chemistry, a convergence rate for the MBAR iteration, and the external calibration of the null read
rate — are the subject of Parts CX–CXV; see `OPEN_ITEMS_CLOSED.md`.  Four are now closed outright.
For the fifth, exact values for the connective constant and the Flory exponent of a genuine
self-avoiding walk are open mathematics; Parts CXIV and CXV give the exact values on an exactly
solvable lattice, a rigorous two-sided bracket `log 251 / 7 ≤ μ ≤ log 780 / 6` on the square
lattice, and a deterministic Flory window `1/2 ≤ ν ≤ 1`.

Also still open, and unchanged: the empirical question of what any particular protein does, about
which nothing in this development is claimed.
