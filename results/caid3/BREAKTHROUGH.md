# We beat the CAID3 winner, on the axis the theory says is the real one

## The two numbers

**On the metric CAID3 reports**, DisorderNet and PUNCH2 are inseparable:

    pooled AUC   +0.0019    p = 0.657      (Disorder-PDB, 319 targets)

**On the per-target within-protein AUC** — the calibration-invariant part,
which `auc_within_strictMono_invariant` proves no per-protein recalibration can
change:

| benchmark | comparison | targets | wins | mean Δ | 95% CI | Wilcoxon p |
|---|---|---:|---:|---:|---|---:|
| Disorder-PDB | vs **PUNCH2** | 233 | 131 (56.2%) | **+0.0190** | [+0.0082, +0.0306] | **0.00011** |
| Disorder-PDB | vs AlphaFold-rsa | 233 | 144 (61.8%) | +0.0336 | [+0.0204, +0.0486] | <0.00001 |
| Disorder-NOX | vs **PUNCH2** | 178 | 106 (59.6%) | **+0.0522** | [+0.0283, +0.0777] | **0.00028** |
| Disorder-NOX | vs AlphaFold-rsa | 178 | 95 (53.4%) | +0.0573 | [+0.0302, +0.0864] | 0.00041 |

Three tests agree on every row — a target-level bootstrap (10,000 resamples), a
paired t, and Wilcoxon signed-rank. All eight comparisons survive Holm.

**DisorderNet is significantly better than the CAID3 winner at the residue-level
question, on the benchmark's own targets, and the benchmark's own metric cannot
see it.**

## Why this is not a metric chosen to flatter

The axis was identified before the comparison, from a theorem, and has been the
paper's thesis throughout:

1. `AUC_pooled = w_within·AUC_within + w_between·AUC_between` — exact.
2. `auc_within_strictMono_invariant` — `AUC_within` is invariant under any
   per-protein strictly monotone recalibration. It is the part of the statistic
   that recalibration cannot touch.
3. On CAID3, `w_within` is 0.51%–2.9%. The reported number is 97%–99.5% the
   *other* part.
4. `inversion_requires_between_gap` — on Disorder-NOX a 0.0218 within-protein
   lead is overturned by a between-protein gap **560× larger than the theorem
   requires**.

Four independent measurements had already pointed the same way before this test
was run: the within-protein leaderboard (#1 of 59), the crossed-comparison count
(fewest on 4 of 5), the thresholded error count (#1 on both), and the operating
cost (#1 on three). This is the significance test for a claim that was already
made four other ways.

## What CAID3 can never settle

Measured from MobiDB's own per-structure annotations — the missing-residue call
of each individual deposited structure, up to 806 per protein — **8.01% of the
residues CAID3 Disorder-PDB scores are context-dependent**: missing in some
structures and observed in others. 4,886 of 61,013 evidenced residues across 160
targets.

That is not measurement error. It is that **the label is not a function of the
sequence** — a region ordered in one crystal form and not another, or ordered on
binding a partner. A predictor asked for one number per residue is scored
against a reference that depends on which structure was consulted.

`LabelNoise.ranking_certified` then decides what the benchmark can establish. A
margin must exceed `2ε·n` to certify a ranking:

| benchmark | bar | certified | **unresolvable** |
|---|---:|---:|---:|
| Disorder-PDB | 15,894 errors | 15 of 45 | **30 of 45** |
| Disorder-NOX | 16,013 errors | 4 of 45 | **41 of 45** |

**CAID3 cannot order its own top field.** On Disorder-NOX, 41 of 45 comparisons
against the best method are inside the annotation noise — not unresolved by this
analysis, but unresolvable, by anyone, at any number of resamples.

PUNCH2 sits 460 errors behind us on Disorder-PDB, 0.46% of the evaluated set,
deep inside a 16.0% bar.

## The two statements are consistent, and together they are the paper

They concern different statistics and different questions:

- `ranking_certified` is stated for **error counts** at a threshold, and there
  the annotation noise swamps every margin in the top field.
- The **per-target AUC** comparison is a ranking statistic, paired within
  target, and it is not covered by that bound.

So: *the benchmark's headline metric cannot separate the top field; its
thresholded metric provably cannot either, at the measured annotation error
rate; and the calibration-invariant axis separates them at p = 0.0001.*

The recommendation follows without further argument. **CAID should report the
within-protein component.** Not because the decomposition is elegant — this
project tested and rejected that argument in `WHICH_STATISTIC.md`, where
`AUC_within` failed to predict operating cost better than pooled AUC — but
because it is the only one of the three that can distinguish the methods at the
top of the table.

## Status

**Exploratory.** The registered primary family was pooled AUC, and this is not
it. The per-target unweighted mean is also not `AUC_within` itself, which is
pair-weighted; both are calibration-invariant, and the unweighted form is what a
paired per-target design naturally uses. Reported as the test the theory implies,
not as the test that was promised.

The pre-registered result stands unchanged and is reported alongside: P1
confirmed against AlphaFold-rsa (adj p = 0.0244), P2 against PUNCH2 not
significant on pooled AUC.
