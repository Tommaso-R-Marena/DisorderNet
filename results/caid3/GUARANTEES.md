# What can be promised about a residue: a distribution-free operating guarantee

Every disorder predictor in CAID reports a ranking statistic. None answers the
question a biologist actually has — *can I act on this call, and how often will
it be wrong?* A probability of 0.62 is not an answer unless the probabilities
are calibrated, and `Calibration.risk_decomposition` is exact about what
calibration buys:

    risk = calError + resolution

A perfectly calibrated model has paid the first term and **nothing else**; the
rest is contextual variation its internal code conflated, which no
recalibration can touch (`risk_eq_resolution_of_calibrated`). Calibration is
necessary and provably insufficient.

Split conformal prediction supplies the missing guarantee, assuming nothing
about the model, the distribution, or the calibration of the scores — only
exchangeability. Applied here on the **temporal holdout**: structures released
after the training caches were built, homology-filtered, so the exchangeability
in question is between two random halves of one set of proteins nobody had seen.

## The guarantee that is valid, and the one that is not

**Per-residue split conformal is not valid here, and the shortfall is
measured.** Residues within a chain are not exchangeable — they share a protein,
a fold, a construct, an experiment — so the split is by chain, and a chain-level
split does not deliver a per-residue promise. On 186 chains at a 90% target the
realised per-residue coverage is **0.876** (`mt_windowed`) and **0.870**
(`mt_pbias`). Quoting those as guarantees would be false, so they are quoted as
measurements of how far the assumption is strained.

**Conformal risk control is valid**, because it makes the promise at the level
the split respects. For a bounded loss monotone in a threshold,

    lambda = inf { t : (n/(n+1))·mean_i L_i(t) + 1/(n+1) <= alpha }

gives `E[L_test(lambda)] <= alpha` over a fresh **chain**. The loss here is the
fraction of a chain's disordered residues the call misses.

## The result, on 61 chains every method can score

Matched: identical chains, identical calibration/test split, so differences are
the methods and not the draw.

| method | risk ≤ 0.10 realised | **residues flagged** | risk ≤ 0.05 realised | residues flagged |
|---|---:|---:|---:|---:|
| **`mt_windowed`** | 0.085 ✓ | **38.6%** | 0.036 ✓ | **47.6%** |
| `mt_pbias` | 0.079 ✓ | 38.5% | 0.035 ✓ | 52.7% |
| AlphaFold-pLDDT | 0.081 ✓ | 42.2% | 0.021 ✓ | 65.1% |
| AlphaFold-rsa | 0.017 ✓ | **96.1%** | 0.000 ✓ | **99.9%** |

**Every method achieves the guarantee. What separates them is the price.**

At a certified miss rate of at most 10% on a protein nobody had seen,
DisorderNet flags **38.6%** of its residues as disordered. AlphaFold-rsa
achieves the same guarantee by flagging **96.1%** — it satisfies the constraint
vacuously, by calling almost the whole chain disordered. At 5% it flags
**99.9%**: the entire protein.

A prediction that flags 96% of a protein is useless, and **AUC does not say so**.
AlphaFold-rsa scores 0.8229 on this set against our 0.8933 — a gap of 0.07 that
reads as modest. The operating guarantee says the two are not comparable
instruments.

AlphaFold-pLDDT is the honest competitor at 42.2%, which is consistent with the
within-protein decomposition, where pLDDT is the best training-free residue-level
discriminator in the CAID3 field.

## What is new here

To our knowledge no disorder predictor has been published with a distribution-free
operating guarantee. The field reports AUC, APS and MCC — all threshold-free or
threshold-arbitrary rankings — and a practitioner choosing a cutoff is on their
own. This says: *pick your tolerated miss rate; here is the threshold that
achieves it with a finite-sample guarantee, and here is how much of the protein
you will have to flag to get it.*

It also separates methods that AUC ranks closely, on a criterion that decides
whether a prediction is usable.

## Traps found while building this, kept so they are not rediscovered

**The singleton rate does not rank models.** A sharper model gets a tighter
conformal threshold, and the empty-set band is exactly `q < p < 1-q`, so a
smaller `q` widens it. Measured on synthetic data: 0.542, 0.966, 0.902 at
increasing skill. Pinned in a test.

**The set semantics are easy to state backwards.** With score `1-p_y` and
`q < 0.5`, the uninformative outcome is the **empty** set, not `{both}`;
`{both}` requires `q >= 0.5`. An earlier draft of this analysis had it the wrong
way round.

**Unmatched chain sets invert the conclusion.** A first run scored our models on
186 chains and the baselines on the 61 with an AlphaFold model, using different
splits. It reported AlphaFold-rsa's class-conditional coverage on disordered
residues collapsing to 0.527 and the risk guarantee failing. **Neither survives
matching**: on identical chains rsa covers 0.994 and achieves the guarantee. The
finding is not that rsa fails the constraint, it is the price it pays to meet
it, and the first version would have been a wrong claim in the strong direction.

## Caveats

- **61 chains** for every matched comparison — 30 calibration, 31 test. The
  intervals are wide and the thresholds are estimated from little data.
- **Not pre-registered.** Built after the CAID3 and temporal results existed.
- The chain-level guarantee is an expectation over chains, not a per-chain
  promise: a given protein can do worse than `alpha`. The distribution of
  per-chain miss rates is recorded in `temporal/results.json`.
- Conformal risk control's validity needs the calibration and test chains to be
  exchangeable. They are two random halves of one temporal set, which is the
  best-supported version of that assumption available here, but the temporal set
  is one week of the PDB and is not a random sample of proteins.

## Reproduction

```bash
sbatch rockfish/slurm/eval_temporal.sbatch
```

Job 30035126, 2m17s. Implementation `colab/conformal.py`, 30 tests in
`tests/test_conformal.py` including the guarantee under a deliberately
signal-free model, where validity is the only thing left.
