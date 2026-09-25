# Is there a maximum information-entropy limit on amino-acid sequences that forces a region to stay disordered?

**Short answer.** Yes, there is a maximum-entropy limit, and it can be written down exactly — but
it is not the limit the question usually imagines, and in the parameter range of real proteins it
is not what makes intrinsically disordered regions (IDRs) disordered.

Three separate statements have to be kept apart, and all three are now proved in Lean
(`RequestProject/SequenceEntropyCore.lean`, `RequestProject/SequenceEntropyLimit.lean`,
`RequestProject/SequenceEntropyExamples.lean`; all sorry-free, depending only on
`propext`, `Classical.choice`, `Quot.sound`).

1. For a **single sequence**, the operative information-theoretic limit is a **floor**, not a
   ceiling: below a computable compositional entropy the region *must* be disordered.
2. For an **ensemble of sequences** (a design ensemble, an evolutionary family) there is a sharp
   **ceiling**, and it is exactly `log |F_N|`, the log-count of foldable sequences of that length.
3. Evaluating `log |F_N|` from physics gives an explicit per-residue number; it lies below the
   alphabet maximum `log 20` only when the per-substitution energy contrast is weak
   (≲ 1 kT in the worked instance), which is the IDR regime, not the folded-domain regime.

---

## The model

A region of `N` residues over a `q`-letter alphabet (`q = 20`), with `M` accessible conformations,
a sequence-dependent energy `E s j`, and the Boltzmann distribution at inverse temperature
`β = 1/kT`. As throughout this project, the region is **ordered** when a single conformation
carries at least half of the equilibrium population (`IDR.SeqLimit.Ordered`), and **disordered**
otherwise: the target of prediction is then irreducibly an ensemble.

Two structural properties of the energy are used, and both are stated as hypotheses rather than
assumed silently:

* **site-Lipschitz** (`SiteLip L E`): one substitution moves every conformational energy by at most
  `L`. For a pairwise contact energy with bounded coordination number, `L = 2z‖e‖∞`.
* **homopolymer degeneracy**: constant sequences have an energy spread of at most `S₀` across the
  conformational library — exact (`S₀ = 0`) on an ensemble of equally compact conformations.

`RequestProject/SequenceEntropyExamples.lean` exhibits a concrete Hamiltonian with both properties
(the burial model: each buried residue that is not the reference letter pays `−L`), so nothing
below is vacuous.

## 1. The thermodynamic gate

`ordered_native_le`, `ordered_needs_spread`: if the region is ordered, the sequence must generate
an energy spread of at least

    kT · log (M − 1)

across its conformational library — the conformational entropy has to be paid for in energy. With
`M = e^{σN}` this is a gap **linear in the length of the region**: `σN·kT`. This is the classical
gap criterion; everything else is a bound on how much spread a sequence of a given entropy can
generate.

## 2. Single sequences: a floor, not a ceiling

Site-Lipschitz energies give `spread(s) ≤ S₀ + 2L·n`, where `n` is the number of residues that
differ from any chosen letter. Combining with the gate (`ordered_needs_minority`), order requires a
**minimum number of minority residues**

    n ≥ κN,      κ = (kT·log(M − 1) − S₀) / (2LN).

Since compositional Shannon entropy is bounded below by the minority weight
(`SeqEnt.H_ge_minority`, `SeqEnt.H_ge_neg_log_max`), this becomes an entropy statement
(`disordered_of_low_composition_entropy`):

> If the compositional Shannon entropy of the region satisfies `H₁(s) < κ·log 2`,
> the region **must be disordered** — at every temperature, for every force field with that
> per-substitution contrast.

Worked number (`worked_low_complexity_verdict`): a 100-residue region, `2^100` accessible
conformations (`σ = log 2 ≈ 0.69` nats/residue), contrast `L = 1 kT`. Then `κ = 0.3` is admissible
and every sequence with `H₁ < 0.3·log 2 ≈ 0.21` nats (≈ 0.30 bits) per residue is provably
disordered. In composition terms that is a region more than about 96% one residue type. The
extreme case is `homopolymer_disordered`: poly-X of length 100 cannot fold.

This is the direction that matches the data: IDRs are systematically **low**-complexity
(poly-Q, poly-G, S/G/P- and charge-rich tracts), not high-complexity. A sequence needs *enough*
information to specify and stabilise a fold; too little, and no force field can help it.

And there is no ceiling in this direction: `ordered_of_far_codewords` shows that in the coding
model *any* sequence — including one of maximal compositional entropy — is ordered as soon as the
competing conformations are far enough away in sequence space. Nothing about the composition of a
single sequence can certify disorder from above.

## 3. Ensembles: the ceiling, and it is exactly `log |F_N|`

Let `F = F_N` be the set of foldable sequences of length `N` (`foldable`). Then:

* `entropy_ceiling`: every ensemble supported on `F` has Shannon entropy `H ≤ log |F|`;
* `entropy_ceiling_attained`: the uniform ensemble on `F` attains it.

So the **maximum information entropy compatible with order is exactly `log |F_N|` nats**, i.e.
`(1/N) log |F_N|` per residue. This is a statement about *ensembles*: a single sequence has zero
entropy, so no single-sequence entropy can be compared with it.

Quantitatively (`disorder_fraction_ge`), an ensemble of entropy `H` must place a fraction

    ε ≥ (H − log |F| − log 2) / (N log q)

of its weight on disordered sequences. Exceed the ceiling by `Δ` nats per residue and at least
`Δ/log q` of the ensemble is disordered.

## 4. What `log |F_N|` is, in physical constants

In the coding model `E s j = L·d_H(s, w j)` — each conformation designed by the sequences near a
codeword, `L` the per-mutation contrast — the gate confines every foldable sequence to a Hamming
ball of radius `r = N(1 − σ/(βL))` around a codeword (`ordered_codeE_near`), so `F` is covered by
`M` balls (`card_foldable_codeE_le`). Sphere packing (`card_ball_le`, `SeqEnt.log_choose_le`) then
gives (`log_card_foldable_codeE_le`)

    (1/N) log |F_N|  ≤  σ + h₂(x) + x·log q  =:  ceilingRate,      x = r/N = 1 − σ/(βL),

with `σ = (log M)/N` the conformational entropy per residue and `h₂` the binary entropy.

**When does this bite?**

* `ceilingRate_binds`: with `q = 20`, `σ = log 2` and `x ≤ 1/4` (i.e. `βL ≤ (4/3)σ ≈ 0.9 kT`),
  the ceiling is strictly below `log 20`: `≈ 2.0` nats ≈ `2.9` bits per residue, against the
  alphabet maximum `log 20 ≈ 3.0` nats ≈ `4.32` bits. Here the ceiling is a real constraint, and a
  design ensemble above it must contain disordered sequences.
* `ceilingRate_vacuous`: as soon as `σ + x·log q ≥ log q` the bound exceeds `log q` and constrains
  nothing. With `σ = log 2` that happens for `x ≥ 1 − log 2/log 20 ≈ 0.77`, i.e. for
  `βL ≥ σ/0.23 ≈ 3 kT`.

Measured single-substitution effects in folded domains are of order 1–3 kcal/mol ≈ 2–5 kT, i.e.
`βL` of a few `kT`. So for real folded domains the entropy ceiling of part 3 is **not** the
operative constraint; it becomes operative exactly where the per-residue energetic contrast is
weak — polar/charged, hydrophobic-depleted sequence, screened electrostatics — which is the
physical signature of an IDR.

## The answer, stated once

* There **is** a maximum information-entropy limit, and it is `H_max = log |F_N|` nats for the
  region's sequence *ensemble*, attained by the uniform ensemble on the foldable set; exceeding it
  by `Δ` nats per residue forces at least `Δ/log q` of the ensemble to be disordered.
* Evaluated in physical constants it is `σ + h₂(1 − σ/βL) + (1 − σ/βL)·log q` per residue: below
  the alphabet maximum only in the weak-contrast (IDR-like) regime, above it — hence empty — in the
  strong-contrast (folded-domain) regime.
* For a **single** amino-acid sequence there is no such ceiling at all. The rigorous
  single-sequence limit runs the other way: below `κ·log 2` of compositional entropy, with
  `κ = (kT log(M−1) − S₀)/(2LN)`, the region **must** be disordered. That floor, together with the
  gap criterion, is what forces real low-complexity regions to remain disordered.

## Limits of the claim

* "Order" is the two-state criterion (one conformation ≥ half the population) over a *fixed finite*
  conformational library `M`; `σ = (log M)/N` is an input, not derived.
* `L` lumps every mutational effect into one Lipschitz constant, and the floor uses only the
  modal-letter decomposition of the composition: it is deliberately sequence-order-blind, so it
  says nothing about charge patterning, which other parts of this project treat.
* The ceiling's evaluation uses the coding caricature of design (each conformation designed by a
  Hamming ball). That is an upper bound on `|F|` only under that model; for a general energy
  function `log |F_N|` remains the exact ceiling but its value must be bounded some other way.
* The sphere-packing step uses `x·log q` where the optimal `q`-ary bound would give `x·log(q−1)`;
  the difference is `x·log(q/(q−1)) ≈ 0.05x` nats per residue at `q = 20`, and it makes the stated
  ceiling slightly conservative (i.e. slightly weaker), never optimistic.
