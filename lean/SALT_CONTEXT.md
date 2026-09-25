# Salt is part of the model: Parts CXXVI–CXXVIII

Everything this development proved about charge patterning in Parts CXX–CXXV — the mean squared
dipole, the dielectric response, the polarization law, the electro-stretching contrast, the
suppression of conformational fluctuation — was proved at zero ionic strength.  The coupling
between two charges was the bare chain kernel.  A disordered region in a cell is in an
electrolyte.  Part LXXIX had already put the salt dependence of the pairwise model into a single
fugacity parameter and identified what survives at high salt; the question these three parts
answer is quantitative — how *large* the patterning effect is as a function of the screening
length, and where the crossover lies.  The answer, in one line: *charge patterning is a low-salt phenomenon, with an
explicit crossover, and a model that does not know the ionic strength cannot know the answer.*

All statements below are machine-checked Lean theorems that build without `sorry` and use only
the standard axioms `propext`, `Classical.choice`, `Quot.sound`.

## Part CXXVI — `RequestProject/SaltCrossover.lean`

The kernel is the ideal-chain linear coupling damped by the Debye factor,
`kern κ d = d · exp(−κ d)`, and `energy N κ q` is the resulting pairwise charge energy.

* `energy_zero` — at `κ = 0` the model is the unscreened one: for a neutral sequence the energy
  is exactly minus the sum of squared prefix charges, the functional of Part CXX.
* `abs_energy_le` — **the screening bound**: `|energy N κ q| ≤ 4 N / κ²` for every sequence of
  unit charges.  Screened electrostatics is at most *linear* in the length of the region,
  uniformly in the pattern.  `energy_tendsto_zero`: at fixed length it vanishes as the salt
  concentration grows.
* `unscreened_gap_ge`, `unscreened_gap_pos` — at zero salt the diblock `++…+−−…−` and the
  perfectly mixed `+−+−…` sequence, which have *identical composition*, differ in energy by at
  least `t³/3 − t`: the contrast is cubic in the length.
* `screened_gap_le` — at inverse screening length `κ` that same contrast is at most `16 t / κ²`.
* `patterning_needs_long_screening_length`, `screening_kills_contrast`, `salt_crossover` — the
  crossover.  If the screened contrast is even half the unscreened one then `κ t ≤ 12`;
  equivalently, once `κ t > 12` the two sequences have become thermodynamically alike.  Charge
  patterning is therefore visible only while the Debye screening length `1/κ` is comparable to
  the length of the region itself.
* `patterning_survives_at_low_salt`, `crossover_two_sided` — the crossover in both directions.
  Perturbing the unscreened energy (`energy_perturb`: `|energy N κ q − energy N 0 q| ≤ κ N⁴`)
  shows that for `κ ≤ 1/(288 t)` the contrast is *still* at least half its unscreened value.
  With `screening_kills_contrast` this brackets the transition: it happens at `κ t` of order one,
  neither sooner nor later.
* `no_salt_blind_prediction` — the modelling consequence.  Any predictor `f` that assigns an
  energy to a sequence *and to a sequence only* has error at least `(t³/3 − t)/8` on one of the
  four sequence/condition pairs (mixed and diblock, at zero salt and above the crossover).  A
  patterning parameter is not a function of sequence.

## Part CXXVII — `RequestProject/DebyeScreening.lean`

The same collapse for the kernel polymer electrostatics actually uses, the Debye–Hückel
interaction at the root-mean-square separation of a Gaussian chain,
`w d = exp(−κ b √d)/(b √d)`, already defined in Part LXXIII.

* `cube_div_le_exp`, `sum_inv_sq_le`, `debye_kern_le` — the screening factor beats every power:
  `w d ≤ 27/(κ³ b⁴ d²)`, and `∑_{d≥1} 1/d² ≤ 2`, both proved from scratch.
* `abs_debye_le` — **the Debye screening bound**: `|debye N b κ q| ≤ 54 N / (κ³ b⁴)`.
* `debye_tendsto_zero`, `debye_pattern_collapse` — hence *any two* unit-charge sequences of the
  same length, not merely the two extremal patterns, have energies within `108 N / (κ³ b⁴)` of
  each other.  At high salt the whole patterning landscape flattens; the conclusion is a property
  of screening, not of a conveniently chosen kernel.

## Part CXXVIII — `RequestProject/SaltAwareDesign.lean`

The design rule, in one theorem (`salt_aware_design`):

1. **Context is indispensable.**  A sequence-only predictor errs by at least `(t³/3 − t)/8`.
2. **What a pairwise model must store.**  `autoModel_exact`, `autoModel_debye_exact`: the charge
   autocorrelation of the region, together with the kernel supplied by the solution condition,
   reproduces every pairwise electrostatic energy *exactly* — nothing else about the sequence is
   needed for that class of models.
3. **And it is still not enough.**  The homometric pair of Part LXXIII has the same
   autocorrelation at every lag, hence the same Debye energy at every ionic strength and bond
   length, while a three-body correlator separates the two sequences.

So a model of a charged disordered region must be *conditional* — the solution condition is an
input, not a constant — and *many-body* in the sequence.  Neither requirement can be traded for
more data or for more parameters of the wrong kind.
