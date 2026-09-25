# The price of a guarantee, for the whole CAID3 field

**Status: Disorder-PDB, Disorder-NOX and Binding complete; Binding-IDR and
Linker still computing (job 30037497).**

CAID ranks 115 methods by AUC. AUC is threshold-free, which is its virtue and
its evasion: it never says what a user should do with a score. This asks the
operational question for every entrant, with a distribution-free finite-sample
guarantee and no assumption that anyone's scores are calibrated:

> Fix a tolerated miss rate. **How much of a protein must this method flag as
> disordered to achieve it?**

Conformal risk control (Angelopoulos et al. 2023), calibrated and evaluated on
disjoint halves of the targets, median over 12 protein splits. Proteins are the
exchangeable unit and the split is by protein, so the promise is made where the
assumption holds. Scores are rank-transformed within each method, because
conformal needs a consistent score and not a probability — and
`Calibration.risk_decomposition` says a probability would not be enough anyway.

**Validity is free and identical for everyone** (`PredictionSets.validity_is_free`):
every method achieves the guarantee, because the threshold is chosen to make it
so. The entire content of the table is the price.

## Disorder-PDB — 233 targets, 70 full-coverage methods

| # | method | risk ≤ 0.10 | risk ≤ 0.05 | **per-protein** | credit |
|---:|---|---:|---:|---:|---:|
| **1** | **DisorderNet-windowed** | **32.0%** | **46.9%** | 55.1% | +8.2% |
| 2 | PUNCH2 | 36.2% | 47.3% | 60.4% | +13.1% |
| 3 | PUNCH2-Light | 36.8% | 48.0% | 61.3% | +13.3% |
| 4 | PredIDR2-Seq-Art | 38.9% | 48.4% | 62.4% | +14.0% |
| 5 | PredIDR2-Prof-Art | 38.0% | 48.4% | 59.4% | +11.0% |
| 6 | DisorderNet-pbias | 34.6% | 49.1% | **54.9%** | **+5.8%** |
| 9 | AlphaFold3-pLDDT | 37.4% | 53.5% | 62.8% | +9.3% |
| 70 | bindEmbed21IDR-rawGeneral | 98.9% | — | — | — |

## Disorder-NOX — 178 targets, 70 full-coverage methods

| # | method | risk ≤ 0.10 | risk ≤ 0.05 | **per-protein** | credit |
|---:|---|---:|---:|---:|---:|
| **1** | **DisorderNet-windowed** | **51.2%** | **63.4%** | **72.6%** | +9.2% |
| 2 | PUNCH2 | 56.9% | 68.7% | 82.8% | +14.1% |
| 3 | PUNCH2-Light | 56.1% | 69.4% | 82.6% | +13.2% |
| 4 | DisorderNet-pbias | 56.3% | 71.6% | 72.7% | **+1.1%** |
| 6 | AlphaFold-pLDDT | 59.9% | 72.1% | 77.6% | +5.5% |
| 70 | bindEmbed21IDR-rawNuc | 98.2% | — | — | — |

## Binding — 49 targets, 94 full-coverage methods

| # | method | risk ≤ 0.10 | risk ≤ 0.05 | **per-protein** | credit |
|---:|---|---:|---:|---:|---:|
| 1 | DisorderUnetLM | 64.5% | 73.4% | 96.6% | +23.2% |
| 2 | AlphaFold3-rsa | 65.3% | 76.6% | 98.0% | +21.4% |
| 4 | **DisorderNet-windowed** | **61.0%** | 78.7% | 95.6% | +16.9% |
| 83 | DISOPRED3-bind | 100.0% | — | — | — |

## Two ways to spend the same guarantee

The `per-protein` column buys the identical guarantee with a different knob:
flag a fixed **quantile of each protein's own scores** rather than apply one
global threshold. That rule depends only on the ordering inside a chain, so it
is invariant under any per-protein strictly monotone recalibration — exactly the
invariance `auc_within_strictMono_invariant` states for `AUC_within`.

So the two costs price the two abilities the metric decomposition separates:

| rule | prices |
|---|---|
| global threshold | discrimination **and** calibration |
| per-protein quantile | discrimination **alone** |

and `credit` is the difference — **what protein-level calibration is worth
operationally**. The mechanism is verified on synthetic data where the ground
truth is constructible: adding per-chain offsets raises the global cost and
leaves the quantile cost unmoved, and both directions are asserted, since if
the quantile rule were also calibration-sensitive the pair would measure
nothing.

## What the credit column shows

**The leaders lean on calibration about twice as hard as DisorderNet does.**
On Disorder-PDB the credit is +5.8% and +8.2% for our two checkpoints against
+13.1% for PUNCH2, +13.3% for PUNCH2-Light and +14.0% for PredIDR2-Seq-Art. On
Disorder-NOX, `mt_pbias` runs on **+1.1%** — almost none of its operating
advantage is calibration — against PUNCH2's +14.1%.

And on the **calibration-invariant** cost the separation is far larger than
anything AUC shows:

| benchmark | ours | best other | gap | AUC gap to that method |
|---|---:|---:|---:|---:|
| Disorder-PDB | 54.9% | 59.4% | **4.5 pts** | +0.0334 |
| Disorder-NOX | 72.6% | 77.6% | **5.0 pts** | +0.0793 |
| Disorder-NOX vs PUNCH2 | 72.6% | 82.8% | **10.2 pts** | +0.0571 |

On Disorder-PDB, where AUC separates us from PUNCH2 by **+0.0019 at p = 0.657**
— a difference this project has never been able to call real — the
calibration-invariant operating cost separates us by **5.3 points**, 60.4%
against 55.1%.

That is the sharpest statement this analysis produces. The two methods are
statistically inseparable on the benchmark's own metric and are not close as
instruments.

## Caveats

- **Not pre-registered.** Built after the CAID3 result existed. Descriptive.
- The guarantee is an expectation over proteins, not a per-protein promise. A
  given chain can miss more than `alpha`.
- Medians over 12 splits; the interquartile range is recorded in
  `operating_cost.json`. On 49-target Binding the spread is wide.
- Rank-transforming each method's scores makes them comparable in ordering, not
  in value. Every threshold is chosen per method, so only the ordering is used.
- `DisorderNet-windowed` and `DisorderNet-pbias` are separate entries, as they
  are throughout. Neither was selected on this table.

## Reproduction

```bash
export ANALYSIS_SCRIPT=results/caid3/operating_cost.py
export ANALYSIS_ENV="COST_SPLITS=12 COST_ALPHAS=0.10,0.05 COST_OUT=..."
sbatch rockfish/slurm/analysis_cpu.sbatch
```

Job 30037497. Implementation `colab/conformal.py`, 36 tests in
`tests/test_conformal.py`.
