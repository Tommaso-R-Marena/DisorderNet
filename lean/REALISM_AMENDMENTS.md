# Four realism amendments to the disorder model

All statements below are machine-checked Lean theorems, free of `sorry` and of any non-standard
axiom.  Entropies are in nats.  New files: `RequestProject/MarginalStability.lean`,
`RequestProject/ConditionalPropensity.lean`, `RequestProject/KmerEntropy.lean`,
`RequestProject/EpistaticSpike.lean`, and the capstone `RequestProject/PartNinetyThree.lean`.

## 1. The native state need only hold 20–30% of the ensemble

`RequestProject/MarginalStability.lean` replaces the model's majority criterion ("some
conformation carries ≥ ½ of the population") by an occupancy criterion
`IDR.Marginal.Functional theta`, and re-proves the model at that threshold.

* `occupancy_gap` — the thermodynamic gate at occupancy `theta`: a conformation holding a
  fraction `theta` against `|D|` competitors must lie `kT·log(theta·|D|/(1−theta))` below them.
  At `theta = 1/2` this is the classical `kT·log|D|`; marginal stability buys exactly
  `kT·log((1−theta)/theta)` of slack — about `1.1 kT` at 25%, `0.85 kT` at 30%.  It is not
  vacuous: `gate_quarter_thirteen` gives `log 4 ≈ 1.4 kT` for thirteen conformations at 25%.
* `functional_needs_spread`, `functional_needs_minority`,
  `disordered_of_low_composition_entropy'` — gate, heterogeneity requirement and the
  compositional **entropy floor**, all valid at occupancy `theta`.  Marginal stability does not
  repeal the floor; it only lowers the constant.
* `flat_four_functional_not_ordered` — the two criteria really differ: on a flat four-state
  landscape every conformation holds exactly ¼, which is functional at `theta = 1/4` and not
  ordered at `1/2`.
* The switch: occupancy `theta` corresponds to a free energy `kT·log((1−theta)/theta)`
  (`occupancy_of_free_energy`); a perturbation `ddG` multiplies the native odds by
  `exp(beta·ddG)` exactly (`occ_shift`); and `kT·log((1−theta)/theta)` of stabilisation makes the
  native state the majority species (`shift_to_majority`).  A 25% ensemble is one hydrogen bond
  away from a 50% one — which is why marginal stability is a design feature.
* `single_structure_error_ge` — the price for prediction: if no conformation exceeds occupancy
  `thetaMax`, a single-structure answer is wrong with probability at least `1 − thetaMax`.

## 2. Folding is driven by conditional probabilities

`RequestProject/ConditionalPropensity.lean` develops pair (Markov) sequence models.

* `H_kernel` — the chain rule `H(p·T) = H(p) + Σₐ p(a)·H(T(a,·))`.
* `H_le_marginals`, `mutualInfo_nonneg` — subadditivity: conditional structure only removes
  entropy.
* `condH_eq_sub_mutualInfo`, `condH_le_single` — conditioning on the previous residue reduces
  the per-residue entropy by **exactly** the adjacent-pair mutual information `I`.
* `blockH_eq`, `block_entropy_deficit` — a length-`n+1` region under the pair model carries
  `H(p) + n·(H(p) − I)` nats, i.e. exactly `n·I` fewer than the composition-matched independent
  model; the conditional ensemble is smaller by a factor `exp(n·I)`.
* `hp_mutualInfo`, `hp_quarter_mutualInfo_pos`, `hp_block_deficit` — a two-letter
  hydrophobic/polar chain with switch probability `e` has `I = log 2 − h₂(e)`; at `e = 1/4`
  (blocky HP patterning) `I = (3/4)·log 3 − log 2 ≈ 0.131` nats per residue, so a 100-residue
  region is over-counted by `e^{13}` if scored on composition alone.

**Design consequence.**  The sequence-entropy ceiling must be evaluated with the conditional
entropy rate; single-residue entropy overestimates the ensemble and over-predicts disorder.

## 3. High single-residue entropy, low `k`-mer entropy

`RequestProject/KmerEntropy.lean` defines the empirical `k`-mer distribution of one sequence and
its entropy.

* `kmerEnt_le_log_distinct` — `H_k ≤ log(#distinct k-mers)`.
* `kmerEnt_le_log_period`, `kmer_rate_le` — a sequence of period `P` shows at most `P` distinct
  `k`-mers, so `H_k ≤ log P` for every `k` and the rate `H_k/k` decays like `log P / k`.
* `alt_single_entropy`, `alt_kmerEnt_le_log_two`, `alt_rate_eventually_small` — `ATATAT…` has
  single-residue entropy exactly `log 2`, the maximum for two letters, while its `k`-mer entropy
  rate falls below any positive threshold once `k` is large enough.
* `composition_blind_one`, `composition_blind_two` — the sharp form: `ATAT` and `AABB` have
  *identical* single-residue distributions, hence identical composition statistics, while their
  2-mer entropies are `log 2` and `log 4`.  No composition-based screen can separate them.

## 4. A single mutation can spike `ΔΔG`

`RequestProject/EpistaticSpike.lean` models a salt bridge as a two-body contact of depth `J`.

* `bridge_spike` — one substitution changes the energy by exactly `|J|`: a non-linear spike.
* `bridge_not_siteLip` — hence the site-Lipschitz hypothesis of the smooth model is false for
  every constant below `J`.
* `additive_error_ge` — no site-independent model `c + Σᵢ aᵢ(sᵢ)` can fit the landscape: it is
  off by at least `|J|/4` on one of the four sequences of the two-site mutational cycle
  (`additive_cycle_zero` versus `bridge_cycle`).
* `SplitLip`, `splitLip_wdist`, `wdist_le` — the repair: energies `L`-Lipschitz off a sparse set
  `Bad` of spike sites and `Smax`-bounded on it obey a weighted Hamming bound.
* `ordered_needs_minority_with_spikes`, `disordered_of_low_entropy_with_spikes`,
  `spike_sites_needed` — the entropy floor survives, with the correction `2·Smax·|Bad|`; a
  low-complexity tract can be rescued by epistasis only if the number of spike sites is at least
  `(kT·log(M−1) − S₀ − 2·L·κ·N)/(2·Smax)`.  One dramatic contact is never enough.

## Capstone

`RequestProject/PartNinetyThree.lean`:

* `exists_modal_letter_low_minority` — the composition step, isolated once and reused.
* `functional_needs_minority_with_spikes` and
  `disordered_of_low_entropy_marginal_and_spiky` — the central obstruction with **both** hard
  amendments at once: native occupancy `theta` instead of a majority, and a sparse set of
  epistatic spike sites.
* `four_realism_amendments` — one representative statement from each amendment, in a single
  theorem.

## What is not claimed

The pair model of amendment 2 is first order; real propensities are longer-ranged, so `n·I` is a
lower bound on what a richer conditional model removes.  The spike model of amendment 4 charges
each spike site its worst case, so `2·Smax·|Bad|` is conservative.  Amendment 1 changes a
threshold, not the physics: a region whose population is spread over exponentially many
conformations is disordered at every occupancy threshold.
