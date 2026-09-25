# Parts CXVI–CXIX: exact polymer results and a sharper connective constant

`OPEN_ITEMS_CLOSED.md` closed four of the five items it listed and left one genuinely open: the
exact value of the connective constant `μ` of the square lattice and the exact Flory exponent.
Those remain open problems of mathematics and are still not claimed.  What follows is what has
been added since: two exactly solved models, a verified enumerator that narrows the bracket on
`μ` from both sides, and an exact law linking sequence *pattern* to chain statistics.

Everything below is a machine-checked Lean theorem, free of `sorry`, depending only on
`propext`, `Classical.choice` and `Quot.sound`.

## Part CXVI — the partially directed chain, solved exactly (`RequestProject/PartiallyDirected.lean`)

A partially directed conformation uses east, north and south bonds — every direction except
backwards — and never follows a north bond immediately by a south bond, or the reverse.

* `adm_isSAW`: every such conformation is genuinely self-avoiding.  This is the geometric heart
  of the file: a chain that never steps west and never reverses a vertical bond never revisits a
  site.  (Proved through two invariants: the abscissa is non-negative, and on the initial column
  the chain stays weakly above, respectively below, its origin.)
* `pdCnt_bounds`: the conformation count satisfies `λ ^ n ≤ pdCnt n ≤ √2 · λ ^ n` with
  `λ = 1 + √2`; the counts are `1, 3, 7, 17, 41, 99, 239, 577, …`, the Pell-like sequence
  `P(n+2) = 2 P(n+1) + P(n)`.
* `pd_entropy_per_residue`: the conformational entropy per residue of the model is *exactly*
  `log (1 + √2) = 0.8813…` — a second exactly solvable case, strictly richer than the directed
  chain of Part CXIV, whose value is `log 2`.
* `log_one_add_sqrt_two_le_connectiveConstant`: since these conformations are self-avoiding,
  `log (1 + √2) ≤ μ`.  This strictly improves the bridge bound `log 251 / 7 = 0.7896…` of
  Part CXV (`improves_on_bridge_bound`).

## Part CXVII — the exact force–extension law (`RequestProject/StretchedChain.lean`)

The same chain under tension: a force `f` weights every axial bond by `u = exp (β f)`, and the
partition function `Zp u n p` is the honest sum over conformations.

* `Zp_succ_east/north/south`, `Zp_north_eq_south`: the exact transfer relations and the up/down
  symmetry.
* `lam_quadratic`, `Zp_bounds`, `free_energy_per_residue`: the growth constant `λ(u)` is the
  positive root of `λ² = (1 + u) λ + u`, the partition function is bracketed between two constant
  multiples of `λ ^ n`, and the free energy per residue is exactly `log λ(u)` at every force.
  `free_energy_zero_force` recovers `log (1 + √2)`; `lamF_strictMono` says the free energy
  increases strictly with the force.
* `hasDerivAt_lam`, `extension_one`, `extension_pos`, `extension_lt_one`,
  `one_sub_extension_le`, `extension_tendsto_one`: the extension per residue
  `u λ'(u) / λ(u)` is well defined and lies strictly between 0 and 1; **at zero force it is
  exactly one half of the contour length**; and it saturates at full stretching as the force
  grows, with slack at most `4 / u`.
* `mean_east_eq`: at finite length, the logarithmic derivative of the partition function in the
  fugacity is exactly the mean number of axial bonds — extension is conjugate to force.

## Part CXVIII — a verified enumerator, and a sharper bracket (`RequestProject/SawEnumerate.lean`)

Earlier upper bounds on `μ` came from exhaustive enumeration of all `4 ^ n` bond sequences, which
stops at `n = 7`.  This part supplies a depth-first enumerator with pruning and proves it
correct.

* `contCount_eq`, `cnt_eq_contCount`: `contCount visited cur k` counts exactly the `k`-bond
  self-avoiding continuations of a partial chain, so `cnt n = contCount [] 0 n`.  The enumerator
  visits one node per self-avoiding prefix rather than one per bond sequence.
* `cnt_eight : cnt 8 = 5916` and `cnt_ten : cnt 10 = 44100`, both by kernel computation.
* `connectiveConstant_le_log44100_div10`: `μ ≤ log 44100 / 10 = 1.0694…`, improving every earlier
  upper bound (`improves_on_cnt_seven`).
* `connectiveConstant_bracket_sharp`: the bracket

      log (1 + √2)  ≤  μ  ≤  log 44100 / 10,      i.e.   2.4142… ≤ e^μ ≤ 2.9137…,

  against the previously available `0.7896… ≤ μ ≤ 1.0976…` (`2.2027… ≤ e^μ ≤ 2.9967…`).  Both
  ends are strictly better, and each further finite computation now genuinely narrows the
  machine-checked bracket.

## Part CXIX — an exact patterning law (`RequestProject/SequencePatterningExact.lean`)

Earlier parts proved *composition blindness* abstractly, by exhibiting ensembles.  This part
derives it from polymer physics in the solvable model.  Each residue carries its own fugacity
`u i` for placing its bond along the chain axis; the composition of a sequence is the multiset of
its fugacities, the pattern is their order.

* `card_eastAt`: an axial bond cuts the chain into two independent chains — the number of
  conformations with bond `j` axial is exactly `pdCnt j · pdCnt (n − 1 − j)`.
* `Zseq_markAt`: for a sequence with one heavy residue `t` at position `j`, the partition
  function is exactly `t · pdCnt j · pdCnt (n−1−j) + (pdCnt n − pdCnt j · pdCnt (n−1−j))`.
* `patterning_gap`: moving that residue from the terminus to the neighbouring interior position
  changes the partition function by exactly `(t − 1) (pdCnt (n−2) − pdCnt (n−3))`.
* `patterning_matters`, `patterning_gap_ge`: this is strictly positive for every `t > 1` and
  every `n ≥ 3`, and is at least `(t − 1) λ^(n−3)` with `λ = 1 + √2` — the two sequences have
  *identical composition* and conformational free energies that differ by an amount exponentially
  large in the length of the region.  Patterning is not a small correction, and a
  composition-only score is wrong by an exponential factor.

## Part CXIX (continued) — the chain is ballistic (`RequestProject/SequencePatterningExact.lean`)

`pdCnt_submultiplicative` (cutting an admissible word in two leaves two admissible words) and
`card_eastAt` combine into `card_eastAt_ge`: at every position, at least a third of the
conformations put an axial bond there.  Summing over positions gives `mean_axial_ge_third` —
averaged over conformations, at least `n/3` of the bonds point along the chain axis, so the mean
extension of the partially directed chain grows linearly in its length.  Its swelling exponent is
exactly `1`, the extreme of the deterministic window `[1/2, 1]` of Part CXIV.

## Parts CXX–CXXIII — charge patterning, solved exactly

Almost every disordered region is a polyampholyte, and experiment finds that the *order* of its
charges, not their number, sets its dimensions.  These four parts prove that, exactly, for the
ideal chain of `RequestProject/Chain.lean`.

### Part CXX — the extremal law for the dipole (`RequestProject/ChargePatterningExact.lean`)

* `neutral_pair_identity`: for **any** neutral charge assignment and **any** conformation, the
  pairwise squared-distance charge coupling `∑_{i<j} q_i q_j (r_i − r_j)²` is exactly minus the
  squared dipole moment.  No averaging, no approximation.
* `neutral_lin_kernel`: the "cut" identity `∑_{i<j} q_i q_j (j − i) = −∑_k S_k²`, where
  `S_k` is the charge of the first `k` residues.
* `mean_dipole_sq`: averaging over all `2^N` conformations, `⟨M²⟩ = b² ∑_k S_k²` — the
  conformational sum is performed exactly and leaves a pure sequence functional.
* `sum_pre_sq_ge` / `sum_pre_sq_alt`: over neutral `±1` sequences of `m = 2t` residues that
  functional is at least `m/2`, attained exactly by the perfectly mixed sequence `+-+-…`
  (odd prefixes can never be neutral).
* `sum_pre_sq_le` / `sum_pre_sq_blk`: it is at most `∑_k min(k, m−k)²`, attained exactly by the
  diblock `++…+--…-`, and that is at least `t³/3`.
* `dipole_bracket` and `patterning_amplification`: hence
  `t² ⟨M²⟩(alternating) ≤ 3 ⟨M²⟩(diblock)` — at fixed composition, segregating the charges
  amplifies the mean squared dipole by a factor growing like the square of the chain length.

### Part CXXI — the field free energy (`RequestProject/DielectricResponse.lean`)

The dipole is a linear form in the independent bonds (`dipole_eq_linear`), so the field
partition function factorises exactly: `Zdip_prod`,
`Z(u) = ∏_i 2 cosh(u b S_{i+1})`, and `logZ_eq` is a sum of one `log cosh` per bond.  Two
elementary bounds sandwich it (`logZ_lower`, `logZ_upper`) between `|ub| ∑_k |S_k|` and that
plus `N log 2`, and evaluating the total absolute prefix charge on the two extremal patterns
gives `dielectric_amplification`: the diblock's field free energy exceeds the mixed sequence's
by at least `|ub|(t(t+1)/2 − t) − N log 2`.

### Part CXXII — polarization and fluctuation–dissipation (`RequestProject/PolarizationLaw.lean`)

Differentiating the exact solution gives the polarization law `polarization_eq`,
`⟨M⟩_u = ∑_i b S_{i+1} tanh(u b S_{i+1})`: each bond responds with a Langevin `tanh` in a field
set by its own prefix charge.  The polarization vanishes at zero field (`polarization_zero`) and
saturates at `|b|∑_k|S_k|` (`abs_polarization_le`).  `fluctuation_dissipation` proves — from the
model, not by assumption — that the zero-field susceptibility `d⟨M⟩/du` equals the mean squared
dipole `b²∑_k S_k²` of Part CXX.  The extremal patterning law is therefore a law about a
measurable susceptibility.

### Part CXXIII — electro-stretching (`RequestProject/ElectroStretching.lean`)

Adding a second conjugate field and differentiating gives the exact force–extension law
`extension_eq`, `⟨R⟩_u = −∑_i b tanh(u b S_{i+1})`, and with it the sharpest form of the
patterning statement:

* `extension_alt_le_half`: **half the bonds of the perfectly mixed sequence sit at zero prefix
  charge and are invisible to the field**, so however strong the field it cannot be extended
  beyond `t|b|`, half of its contour length `(2t−1)|b|`;
* `extension_blk_le`: every prefix charge of the diblock is at least one, so the whole chain
  responds and its extension reaches `N b tanh(u b)` — the full contour length as the field grows.

`electro_stretching_contrast` states the two together: two sequences of identical amino acid
composition, differing only in the order of their residues, have saturation extensions that
differ by a factor of two.

### Part CXXIV — what the exact dipole cannot see (`RequestProject/DipoleAutocorrelation.lean`)

`mean_dipole_sq_eq_pairEnergy` identifies the exactly computed mean squared dipole with `-b²`
times a *pairwise separation-dependent* charge energy, in the precise sense of Part LXXIII, with
the linear kernel `w d = d`.  By the invariance theorem of that part, `mean_dipole_sq_of_autocorr_eq`
follows: two neutral sequences whose charge autocorrelations agree at every lag have exactly the
same mean squared dipole.  The exact solution and the incompleteness theorem are two sides of one
statement — the susceptibility of a disordered region is computable in closed form and is extremal
for the diblock, and it still cannot be inverted to recover the sequence.

### Part CXXV — how much conformational freedom the field can take away (`RequestProject/FluctuationSuppression.lean`)

Differentiating the two-field solution a second time gives `extension_variance_eq`: the exact
fluctuation of the extension is `b² ∑_i (1 − tanh²(u b S_{i+1}))`, one `sech²` per bond in the
field that bond feels.  Hence `variance_alt_ge` — the perfectly mixed sequence has `t − 1` bonds
at zero prefix charge, which feel no field at all, so its fluctuation never falls below
`(t−1) b²` however strong the field — and `variance_blk_le` — every prefix charge of the diblock
is at least one, so its fluctuation is at most `N b² (1 − tanh²(u b))`, which the field drives to
zero.  `fluctuation_suppression_contrast` states the two together: at identical composition, one
sequence can be frozen by the field and the other cannot.

## What is still open

Unchanged and stated without softening: the exact value of the connective constant of `ℤ²`
(and of `ℤ³`) and the exact Flory exponent.  Parts CXVI and CXVIII narrow the rigorous bracket
and give the exact answer in the solvable cases; no finite computation determines `μ`.
