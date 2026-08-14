# DisorderNet: A Post-Structure Distrust Layer for Intrinsic Disorder

*Detecting where AlphaFold is confidently wrong about disorder.*

[![Tests](https://github.com/Tommaso-R-Marena/DisorderNet/actions/workflows/test.yml/badge.svg)](https://github.com/Tommaso-R-Marena/DisorderNet/actions/workflows/test.yml)

**Open the notebooks in Google Colab (GPU):**

| Notebook | What it does | Open |
|----------|--------------|------|
| **v8 Multi-scale** | Multi-backbone extraction → v7 CV → **v8 ensemble** → calibration/conformal → predictor | [![Open v8 Multi-scale in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_v8_MultiScale.ipynb) |
| **Pro (LoRA)** | Full GPU ESM-2 650M/3B + LoRA (auto calibrated conformal) | [![Open Pro in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_Pro.ipynb) |
| **Quick Screen** | Mini-ultra go/no-go before a full ultra run | [![Open Quick Screen in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_QuickScreen.ipynb) |

## Overview

**DisorderNet** is a protein language model (PLM)-enhanced ensemble for predicting
intrinsically disordered regions (IDRs) in proteins, with **trustworthy per-residue
uncertainty** — calibrated probabilities and conformal "confident / abstain"
decisions that most disorder predictors do not provide.

> ## ⚠ Results under re-measurement (2026-08-10)
>
> A defect in homology clustering meant the **"homology split" numbers below were
> not homology-separated**. `difflib.SequenceMatcher` enables an `autojunk`
> heuristic that, for inputs of 200+ elements, treats any element occurring in
> more than 1% of positions as junk. Every amino acid clears 1%, so for proteins
> longer than 199 residues two 95%-identical sequences scored ~0.01. Nothing
> reached the 0.40 threshold, no clusters ever merged, and `split_method="homology"`
> silently produced the same per-protein split it exists to replace.
>
> Measured on the DisProt release used here: **2091 of 2663 proteins (78.5%) are
> ≥200 residues** and were therefore invisible to homology detection; **424
> proteins** belong to a multi-member family whose members were spread across
> folds. The same defect disabled the CAID train/test leakage audit, which now
> removes **323 of 2663** training proteins as ≥40% identical to a CAID3 target.
>
> Four further leakage paths inflated post-processing metrics: fold soup scoring
> checkpoints on their own training proteins, the v6 stream trained on homologues
> of the proteins it scored, an in-sample meta-stacker, and in-sample isotonic
> calibration. Separately, saved checkpoints omitted fine-tuned ESM weights, so
> any number produced by reloading a checkpoint (including CAID3) is unreliable.
>
> All are fixed and regression-tested (`tests/test_leakage_guards.py`,
> `tests/test_checkpoint_roundtrip.py`). **Treat every AUC in this README as
> provisional until re-measured.** Expect corrected figures to be lower.
> See [`docs/HOMOLOGY_HOLDOUT.md`](docs/HOMOLOGY_HOLDOUT.md).
>
> **A sixth defect, found later, ran the other way.** `parse_caid_reference_fasta`
> gave `labels` an entry only at unmasked positions while building `eval_mask`
> over the whole sequence, so the two indexed different spaces; the evaluator
> then matched predictions (indexed by sequence position) to labels (indexed by
> labelled-position index). On the CAID3 Disorder-PDB reference **239 of 319
> targets had mismatched lengths and 69.8% of scored residues were compared
> against another residue's prediction**. Every CAID3 figure this project has
> ever reported is therefore wrong — and *understated*. Fixed and pinned in
> `tests/test_caid3_alignment.py`; the CAID3 rows below are the corrected ones.

### Headline results — legacy, NOT homology-separated

All numbers below fit PCA on the **train fold only** and are pooled per-residue
AUC-ROC. The "homology split" column is retained for provenance only: for the
reasons above it is **equivalent to a protein split**, so it does not support a
CAID-credibility claim and is optimistic relative to a true homology holdout.

| Model | random split | "homology" split¹ | notes |
|-------|-------------:|------------------:|-------|
| v6 baseline (ESM-2 8M + GBDT) | 0.8397 | — | prior release |
| v7 (ESM-2 35M) | 0.8479 | 0.8396 | rich features + LGB/XGB/HistGBM blend + smoothing |
| v7 (ESM-2 150M) | 0.8498 | 0.8457 | |
| v7 (ESM-2 650M) | 0.8505 | 0.8487 | |
| **v8 multi-scale ensemble (35M+150M+650M)** | **0.8568** | **0.8525** | best legacy CPU result |

¹ Not actually homology-separated — see the notice above.

### First corrected measurements (ESM-2 650M, genuine homology splits)

From the leak-free 650M run (2,340 proteins after removing 323 CAID-homologous
entries; 1,997 homology clusters via BLASTp). These come from validation
probabilities computed during training and are **not** affected by the checkpoint
defect:

| Component | Pooled AUC |
|---|---:|
| ESM-2 650M + LoRA alone | 0.7454 |
| v6-pro (physics GBDT) alone | 0.7804 |
| AlphaFold pLDDT alone | 0.7906 |
| GPU + v6 ensemble | 0.8011 |
| + AlphaFold inference fusion | **0.8196** |

Per-fold GPU AUC: 0.7651 / 0.7611 / 0.7632 / 0.7486 / 0.7376 (mean 0.7551 ± 0.0105).

Two observations worth stating plainly: the trained model is the **weakest single
component**, and AlphaFold pLDDT alone outperforms it. The gains come from
combining streams, not from the language model on its own.

#### These are single-run numbers, and reruns move them

Four runs of this exact configuration (same seed, same code, same splits) exist
in the results tree. `rockfish/rerun_stability.py` compares them:

| run | GPU pooled | GPU mean-of-folds | v6 pooled | stacked |
|---|---:|---:|---:|---:|
| j29682871 | 0.7454 | 0.7551 | 0.7804 | 0.7999 |
| j29683037 | 0.7234 | 0.7405 | 0.7804 | 0.7920 |
| j29683038 | 0.7137 | 0.7514 | 0.7804 | 0.7909 |
| j29683091 | 0.7203 | 0.7491 | 0.7804 | 0.7876 |
| | **sd 0.0138** | **sd 0.0062** | (cached) | sd 0.0052 |

The table above reports j29682871. Another run of the same code puts the GPU
component at 0.7137 — a 0.032 spread with nothing changed. **Pooled AUC is about
twice as irreproducible as mean-of-folds**, which is what the cross-fold
calibration analysis predicts: pooled AUC ranks residues scored by five
separately-calibrated fold models and so inherits their calibration variance,
while mean-of-folds never compares across folds. In one run the per-fold median
predicted probability ranged from 0.0003 to 0.6523 at near-identical disorder
prevalence, and pooled AUC landed 0.029 *below* mean-of-folds as a result;
rank-normalising within folds recovered mean-of-folds almost exactly.

Prefer mean-of-folds as the headline. Quote pooled only alongside its
rank-normalised counterpart, which CV now reports automatically.

Caveat: v6 is identical across all four runs because they share one cached OOF
prediction file. That is one measurement reused, not four — the GBDT's own
reproducibility is unmeasured. What the four runs do establish is that the
physics GBDT beats the 69.9M-parameter neural model **in every one of them**, by
+0.035 to +0.067 (mean +0.055). That gap is the motivation for the frozen-backbone
`lite` profile (`colab/lite_head.py`), which tests whether it is a
capacity/data mismatch rather than a weak backbone.

### DisorderNet-Lite — the capacity hypothesis, tested

If the neural model loses to a GBDT because 69.9M trainable parameters on 988k
evidenced residues is a mismatch — and not because ESM-2 is a weak backbone —
then *removing* capacity should help. `lite` freezes the backbone entirely,
learns a softmax mixture over its layers, and trains a ~1.96M-parameter dilated
residual CNN head under plain weighted BCE. Three seeds, same homology splits,
same evaluation:

| | trainable | DisProt mean-of-folds | CAID3 Disorder-PDB | GPU·h |
|---|---:|---:|---:|---:|
| `ultra` | 69.9M | 0.7514 | *re-measuring* | ~12 |
| `lite` s42 | 1.96M | 0.8247 | 0.9218 | ~0.9 |
| `lite` s43 | 1.96M | 0.8266 | 0.9230 | ~0.9 |
| `lite` s44 | 1.96M | 0.8248 | 0.9236 | ~0.9 |
| **`lite` mean** | **1.96M** | **0.8254** ± 0.0011 | **0.9228** | **~0.9** |

CAID3 figures use the official protocol: all 319 Disorder-PDB targets pooled,
unannotated residues ignored. Our evaluation reproduces the benchmark's
composition — 319 targets, 99,239 residues, 31.6% disordered (31,359 positives
against the official 31,401) — which is the available evidence that we scored
what CAID3 scored.

**+0.074 DisProt mean-of-folds over `ultra` with 36× fewer trainable parameters
and roughly a tenth of the GPU time.** The three seeds agree to sd 0.0011, far
inside the 0.023 rerun noise floor. `lite` also beats the physics GBDT measured
on its own folds (0.7774–0.7790), reversing the observation that motivated it.

### Where this stands on the official CAID3 benchmark

Every earlier version of this section compared against `ESMDisPred = 0.895`,
described as "CAID3 SOTA". Both halves were wrong, and the number was not even
from the right benchmark: 0.895 is ESMDisPred-2PDB's **Disorder-NOX** figure
(0.8855). On Disorder-PDB it scores 0.937, and the leader is **PUNCH2 at 0.955**.

That guesswork is over. CAID publishes the five challenge references *and* the
per-residue predictions of all 117 entrants, and `colab/caid3_official.py`
downloads both, verifies each reference against composition counts served
separately by CAID's dataset API, and reproduces every published leader from the
raw files — PUNCH2 0.9552 against a published 0.955, ESMDisPred-2PDB 0.8855
against 0.885, IPA-AF2-Linker 0.8985 against 0.897, DisoFLAG-PB 0.7760 against
0.776, bindEmbed21IDR 0.6407 against 0.641. Nothing is transcribed and nothing is
reconstructed; every comparison is paired on shared targets.

**Current standing** (`mt_full`, all targets predicted, no benchmark skipped):

| benchmark | ours | rank | + AlphaFold-rsa | rank | leader |
|---|---:|---:|---:|---:|---|
| Disorder-PDB | **0.9603** | **1 / 115** | 0.9628 | 1 / 115 | PUNCH2 0.955 |
| Linker | **0.8885** | **2 / 115** | 0.8846 | 2 / 115 | 0.897 |
| Disorder-NOX | 0.8422 | 13 / 115 | 0.8645 | **5 / 115** | 0.885 |
| Binding | 0.7649 | 11 / 115 | 0.7453 | 24 / 115 | 0.776 |
| Binding-IDR | 0.5180 | 24 / 115 | 0.3766 | 105 / 115 | 0.641 |

Verified before being believed, because the previous claim of this shape had to
be retracted (see below): 319/319 targets and 99,239 residues at prevalence
0.3164, matching CAID exactly; the reference file used by the leak filter, the
one scored against, and a fresh download all hash to
`6feaff35263e7fd4a3f03640c23786fe`; the union of all five references is the same
319 proteins, every one of them removed from training along with 289 homologs at
identity ≥ 0.40; and the whole thing recomputed from the archived submissions by
a separate code path.

**What is not established.** The margin over PUNCH2 is +0.0051 at p=0.19 — the
claim is that this tops the table, never that it is proven better than PUNCH2.
Worse, these p-values come from a family of **sixteen** comparisons run against
these references in one session, and under Holm–Bonferroni none of the
favourable ones survive:

| comparison | p_raw | p_adjusted | survives |
|---|---:|---:|---|
| fused − PUNCH2 | 0.006 | 0.084 | no |
| ours − AlphaFold-rsa | 0.013 | 0.143 | no |
| ours − PUNCH2 | 0.205 | 1.000 | no |

The rank is a point estimate and stands. Every significance claim from that run
is **exploratory** — generated by looking, not by testing. A confirmatory run is
pre-registered in `results/caid3/PREREGISTRATION.md`, with the primary family
fixed at two tests before the data existed.

Two caveats that survive all of it:

- **The fused row is not self-contained.** It consumes AlphaFold-rsa predictions
  from CAID's published files. Our own rsa implementation scores 0.938 where
  CAID's scores 0.950, so a standalone submission would land lower. The unfused
  0.9603 is the number that stands alone.
- **AlphaFold-rsa ranks 3rd at 0.950 with no training at all**, and is not
  statistically separable from PUNCH2 (+0.0054, p=0.16). Relative solvent
  accessibility beats nearly every dedicated predictor, which complicates this
  project's original framing of AlphaFold as something to distrust in disordered
  regions. pLDDT ranks 11th; rsa is a different and far stronger signal.

A side effect worth noting: the frozen backbone largely removes the cross-fold
calibration drift. Per-fold median predicted probability spans ~3× under `lite`
against ~2000× under `ultra`, and rank-normalisation moves pooled AUC by +0.006
rather than +0.028.

### Structural baselines are competitive, not superior — and the difference is a benchmarking artifact

An earlier version of this section claimed rsa+pLDDT *beat* the CAID3 leader at
0.9581 against 0.9550. That claim was wrong, in a way worth recording because it
is a trap any structure-based method can fall into.

The 0.9581 was measured on the **304 targets that have an AlphaFold entry**, not
on all 319, and quoted without an interval. Measured properly:

| framing | AUC | 95% CI | vs PUNCH2 0.955 |
|---|---:|---|---|
| 304 structure-available targets | 0.9581 | [0.9395, 0.9724] | CI **includes** it |
| all 319, unscorable at base rate | 0.9382 | [0.9098, 0.9614] | CI includes it |

Neither beats the leader — the interval contains 0.955 both ways, and the
margin claimed (0.0031) was a fifth of the sampling uncertainty (±0.017).

**Dropping the 15 unscorable targets is worth +0.0199 AUC**, six times the
claimed effect. That is the finding worth keeping: a structure-based predictor
that silently skips targets without structures gains about +0.02 AUC that is not
real, and on this benchmark that exceeds the gaps separating ranks 1 through 5.
Structural methods should be benchmarked on every target, with unscorable ones
counted, or the comparison flatters whichever method has the narrowest coverage.

It also states our own position more fairly: **DisorderNet scores all 319
targets**, because a sequence model does not need a structure to run.

| | AUC | APS |
|---|---:|---:|
| DisorderNet-Lite | 0.9215 | 0.8567 |
| AlphaFold −pLDDT | 0.9431 | 0.9062 |
| AlphaFold rsa (window 21) | 0.9459 | 0.9168 |
| rsa + pLDDT *(304 structure-available targets)* | 0.9581 | 0.9320 |
| **rsa + pLDDT *(all 319, honest)*** | **0.9382** | **0.8880** |
| PUNCH2 (CAID3 #1) | 0.9550 | 0.9280 |
| rsa + pLDDT + model *(weights fit on training data)* | 0.9554 | 0.9281 |

Fusing the model into that baseline makes it **worse** (−0.0028 AUC): the
honest fusion reports `model_earns_its_place: False`. A grid search over fusion
weights reaches 0.9633, but those weights were fitted on the residues being
scored, so that figure is an upper bound and not a result.

Our rsa reproduces the published AlphaFold-rsa to within 0.004 (0.9459 against
0.950), which is the strongest validation available that this harness scores
what CAID3 scores — stronger than the composition match, because it reproduces
another group's *method score*.

So Disorder-PDB is not where this project can contribute. The other four
benchmarks are a different matter: no AlphaFold baseline reaches the
Disorder-NOX top ten, because NOX counts unannotated residues as **ordered**
rather than ignoring them, and the structural shortcut stops working the moment
absence of evidence is a negative.

### One model, five benchmarks

CAID3 is five benchmarks won by five specialists, and none answers another's
question:

| benchmark | targets | positives | leader | AUC | APS |
|---|---:|---:|---|---:|---:|
| Disorder-PDB | 319 | 31.6% | PUNCH2 | 0.955 | 0.928 |
| Disorder-NOX | 204 | 26.4% | ESMDisPred-2PDB | 0.885 | 0.754 |
| Linker | 31 | 6.7% | IPA-AF2-Linker | 0.897 | 0.474 |
| Binding | 52 | 10.6% | DisoFLAG-PB | 0.776 | 0.245 |
| Binding-IDR | 52 | 38.6% | bindEmbed21IDR | 0.641 | 0.514 |

`MultiTaskLiteHead` answers all of them from **one frozen-backbone forward
pass**: a shared dilated trunk with a linear read-out per task, **1,914,634
trainable parameters in total and 1,538 per additional task**.

Measured on an A100 over the 319 CAID3 targets:

| | |
|---|---|
| all four tasks, 285 targets | 7.85 s |
| per target | **27.6 ms** |
| throughput | 36.3 targets/s, 13,565 residues/s |
| head share of runtime | **0.9%** |
| five separate specialists | ~31.2 s — **4.0× slower** |

The head is 0.9% of runtime, so four tasks cost 1.009× one task while five
separate models cost 4×. That is the argument for sharing the trunk, and it is
measured rather than asserted (CUDA-synchronised, post-warmup, median of three
passes). It is **not** comparable to the CAID3 timing table, which was collected
on the organisers' hardware.

Sharing is also a modelling choice, not only an economy. Linker has 15,683
positive residues and binding 88,761, against disorder's 336,014 — the
small-data regime where this head beat a 69.9M-parameter LoRA model by +0.074.
A shared trunk carries disorder's data into tasks with a twentieth of it, and
linear per-task read-outs stop any of them growing private capacity.

**Reference reconstruction.** Only `disorder_pdb.fasta` is published, so the
other four benchmarks were unmeasurable. `colab/caid3_references.py` derives
them from the challenge's own generation rules, with the official 319 targets
fixing the universe, and validates by composition:

| benchmark | reconstructed | published | status |
|---|---|---|---|
| Linker | 31 tgt / 1,379 pos | 31 / 1,379 | **head-to-head** |
| Binding | 45 / 2,352 | 52 / 2,991 | approximate |
| Binding-IDR | 31 / 2,352 | 52 / 2,991 | approximate |
| Disorder-NOX | 319 / 31,518 | 204 / 26,367 | approximate |

Only Linker may be quoted beside its published leader; the code marks the rest
"NOT comparable … must not be reported as if it were". CAID3 targets are DisProt
entries, so training excludes them and their ≥40%-identity homologues — 358 of
2,905 proteins — and the run fails loudly if BLAST is unavailable rather than
reporting a filter that did not run.

The v8 ensemble is also the best-**calibrated** config: isotonic calibration lowers
Expected Calibration Error from ~0.041 to **~0.0025** (ranking preserved), and the
split-conformal layer holds its coverage guarantee (empirical coverage ~0.90–0.91 at
α=0.1) with ~0.86 selective accuracy on the residues it is confident about. The GPU
LoRA path (ESM-2 650M/3B) inherits the same calibrated + conformal confidence
layer. Its measured pooled AUC under genuine homology separation is **0.8196**
(with AlphaFold fusion), not the ≥0.88 previously targeted here. The `lite`
profile above supersedes this path on both benchmarks at a fraction of the cost.

DisorderNet's distinctive contributions, as the measurements now support them:
(1) **one frozen-backbone model answering all five CAID3 tasks** at 1,538
trainable parameters per task and 0.9% of runtime, where every published entry
is a single-task specialist; (2) a **calibrated + conformal confidence layer**
shared by the CPU and GPU paths; and (3) the **post-structure IDR biology layer**
quantifying AlphaFold/Boltz hallucinations in IDRs.

> **A caveat on (3), from this project's own measurements.** The framing that
> AlphaFold is something to distrust in disordered regions is only half right.
> AF3-pLDDT does rank 13th on CAID3 and AF2-pLDDT 11th — but **AlphaFold-rsa
> ranks 3rd at 0.950**, above every dedicated predictor except the two PUNCH2
> variants, and rsa+pLDDT together reach 0.9382 across all 319 targets —
> competitive with the CAID3 leader, not above it.
> The weakness is specific to pLDDT as a disorder proxy, not to AlphaFold's
> output as a whole. Structure-derived features are the strongest single signal
> on Disorder-PDB, and this pipeline did not use the strongest one.

## From scratch (start here)

Pick **one** path. Each sequence below assumes a clean machine / empty checkout — clone, install, run, inspect outputs.

| Path | When to use | Hardware | ~Time to first result |
|------|-------------|----------|------------------------|
| **A. CPU v6** | Smoke-test the repo / no GPU | Laptop / desktop CPU | ~15–30 min (embeddings longer first time) |
| **B. Colab GPU** | Interactive ultra CV on Google Colab | Colab A100 (or L4) | Quick screen 2–3 h → full ultra 18–24 h |
| **C. Rockfish (recommended for publish)** | Production / paper numbers + `publish_package/` | JHU Rockfish A100 Slurm | Days (chained 72 h jobs) |
| **Tests only** | Verify the checkout | Any CPU | ~1–2 min |

Canonical HPC detail (env vars, packaging flags, go/no-go): **[rockfish/README.md](rockfish/README.md)**.  
Preprint checklist after a publish run: **[docs/METHODS_CHECKLIST.md](docs/METHODS_CHECKLIST.md)**.

---

### Path A — CPU pipeline (no GPU)

```bash
git clone https://github.com/Tommaso-R-Marena/DisorderNet.git
cd DisorderNet
git checkout master

python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements-cpu.txt

# Data and results default to repo-local dirs (./data, ./results_v6) — no edits or
# symlinks needed. To store them elsewhere, set DISORDERNET_HOME (or the finer-grained
# DISORDERNET_DATA_DIR / DISORDERNET_RESULTS_ROOT); see disordernet_paths.py.

python fetch_disprot.py             # DisProt JSON download (needs network) → ./data
python extract_esm_embeddings.py    # ESM-2 embeddings (first run downloads weights) → ./data/embeddings
python run_v6_mem.py                # v6 5-fold CV → metrics (~0.84 AUC) → ./results_v6
python generate_figures_v6.py       # ROC/PR + figures → ./results_v6

# Optimized model (v7) + honest homology-split option + deployable predictor:
python run_v7.py                    # v7 leakage-free CV (~0.848 AUC, ESM-2 35M) → ./results_v7
DISORDERNET_SPLIT=homology python run_v7.py   # CAID-credible homology-split CV
python train_predictor.py           # save a deployable bundle → ./results_v7/predictor_bundle.joblib
python predict_disorder.py --seq MDVFMKGLSKAKEGVV...   # per-residue calibrated + conformal output
```

**Success:** `results_v6/metrics.json` (v6, ~0.84) and/or `results_v7/metrics.json`
(v7, ~0.848 with calibration/conformal), plus figures under `results_v6/`. For the
best CPU number (0.857), extract multiple backbones and run `run_v8_multiscale.py`
(see [Optimized model + per-sequence predictor](#optimized-model--per-sequence-predictor-v7)).

---

### Path B — Google Colab GPU (from zero)

1. Open the **Quick Screen** notebook (do this before a full ultra run):  
   [![Open Quick Screen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_QuickScreen.ipynb)
2. **Runtime → Change runtime type → GPU (A100 preferred; L4 OK for 650M) + High RAM**.
3. Set `SCREEN_MODE = "standard"` (recommended mini-ultra / `screen_plus`; or `"flash"` / `"paradigm"`), run all cells.
4. Read `quick_screen_report.json` — proceed only on **HIGH / MODERATE** (not STOP). Confirm the log shows `profile=screen_plus` for `standard`.
5. Open the **full GPU** notebook:  
   [![Open Full GPU CV](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_Pro.ipynb)
6. Set `QUALITY_PROFILE = "ultra"` (or `"ultra3b"` + `ESM_BACKBONE = "3B"` on A100 40GB).  
   Set `MOUNT_DRIVE = True` so checkpoints / DisProt / pLDDT caches persist.
7. Run all cells (~18–24 h for 650M ultra). Optional later cells cover AF rescue, CAID, IDR layer, Phase 3 synthesis.

**Success:** `run_manifest.json`, CV / postprocess reports, and (with Drive mounted) mirrored files under `MyDrive/DisorderNet/results/`.  
For 3B: **A100 40GB + High RAM**; do **not** use T4. `!pip install -q lightgbm xgboost` if prompted.

---

### Path C — Rockfish ops guide (what you know, what to run, when you’re done)

Everything below runs on a Rockfish **login node** (submit only — never train/extract on login).  
Deep flags/layouts: [rockfish/README.md](rockfish/README.md) · v8 copy-paste: [rockfish/V8_MULTISCALE.md](rockfish/V8_MULTISCALE.md).

#### What you know (accounts)

| Job type | Partition | Account | QOS | Examples |
|----------|-----------|---------|-----|----------|
| **GPU** | `a100` | **GPU account** (usually `sfried3_gpu`) | **`qos_gpu`** | v8 embed extract, ultra train, Boltz GPU |
| **CPU** | `shared` | **CPU account** (`sfried3`) | default (omit `--qos`) | v8 CV/ensemble, strict `publish_package` |

`a100` rejects QOS `normal` (“QOSMax… / not permitted”). `qos_gpu` lives on the **GPU** account, not on `sfried3`. Discover once:

```bash
export DISORDERNET_ACCOUNT=sfried3   # CPU / shared
export DISORDERNET_GPU_ACCOUNT=$(sacctmgr -nP show assoc user=$USER format=account,qos \
  | awk -F'|' '/qos_gpu/{print $1; exit}')
export DISORDERNET_GPU_QOS=qos_gpu
echo "CPU=$DISORDERNET_ACCOUNT  GPU=$DISORDERNET_GPU_ACCOUNT  QOS=$DISORDERNET_GPU_QOS"
# Expect something like: CPU=sfried3  GPU=sfried3_gpu  QOS=qos_gpu
```

#### What you already finished (typical after first Rockfish session)

If you already did these, **do not redo** them — just `git pull` and resubmit jobs:

1. Clone + `bash rockfish/setup_env.sh` + venv activate  
2. `pytest` / `ruff` green on a `shared` node  
3. DisProt download under `$DISORDERNET_V8_DIR/data/`  
4. `python rockfish/prefetch_esm.py` (ESM weights cached — login-safe; do **not** `load_model` on login)

#### Stuck jobs? Cancel, pull, resubmit (do this first if `squeue` shows PD + QOS error)

```bash
cd ~/DisorderNet && git checkout master && git pull
source ~/venvs/disordernet/bin/activate
mkdir -p logs

# Cancel anything stuck Pending with QOS / DependencyNeverSatisfied
scancel <EMBED_JOB_ID> <PIPELINE_JOB_ID>    # e.g. scancel 28766644 28766650
squeue -u $USER                             # should be empty (or only healthy jobs)

export DISORDERNET_V8_DIR=$HOME/scr4_sfried3/disordernet_v8
# Preferred (always sets qos_gpu; never submits --qos= empty):
bash rockfish/slurm/submit_v8.sh
```

If you submit by hand, **you must** pass a non-empty QOS. This fails:

```bash
# BAD — DISORDERNET_GPU_QOS unset → --qos="" → "Invalid qos specification"
sbatch -A sfried3_gpu --qos="$DISORDERNET_GPU_QOS" rockfish/slurm/v8_extract_embeddings.sbatch
```

This works:

```bash
export DISORDERNET_GPU_ACCOUNT=sfried3_gpu   # or discover via sacctmgr
sbatch -A "$DISORDERNET_GPU_ACCOUNT" --qos=qos_gpu \
  --export=ALL,DISORDERNET_V8_DIR \
  rockfish/slurm/v8_extract_embeddings.sbatch
```

#### Exact command ladder (recommended order)

**A — v8 multi-scale (honest CPU ensemble ~0.857 / homology ~0.853)** — start here; cheapest publishable number.

```bash
export DISORDERNET_V8_DIR=$HOME/scr4_sfried3/disordernet_v8
bash rockfish/slurm/submit_v8.sh
# prints embed + pipeline job ids; then: squeue -u $USER
```

Manual equivalent (literal `qos_gpu` — do not rely on an unset env var):

```bash
# GPU extract (~1–2 h wall once scheduled; 1× A100)
EMBED=$(sbatch --parsable \
  -A sfried3_gpu --qos=qos_gpu \
  --export=ALL,DISORDERNET_V8_DIR \
  rockfish/slurm/v8_extract_embeddings.sbatch)
echo "embed job: $EMBED"

# CPU pipeline (~6–12 h wall; starts only after embed succeeds)
sbatch -A sfried3 \
  --dependency=afterok:$EMBED \
  --export=ALL,DISORDERNET_V8_DIR \
  rockfish/slurm/v8_pipeline.sbatch
```

**B — Optional Boltz warm-up** (structure-distrust artifacts; hours–days depending on queue/cache)

```bash
export DISORDERNET_BOLTZ_ROOT=${DISORDERNET_BOLTZ_ROOT:-$HOME/scr4_sfried3/boltz}
export BOLTZ_CACHE=$DISORDERNET_BOLTZ_ROOT/cache
sbatch -A sfried3_gpu --qos=qos_gpu \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_BOLTZ_ROOT,BOLTZ_MODE=auto \
  rockfish/slurm/boltz_batch.sbatch
```

**C — 650M LoRA publish bundle** (ultra + clean → strict package; multi-day GPU chain)

```bash
bash rockfish/slurm/submit_publish_650m.sh \
  --account sfried3_gpu --qos qos_gpu
# Workdir printed + also in ~/disordernet_runs/publish_650m_*/submit_summary.json
```

**D — Optional 3B publish bundle** (only after 650M screen/ultra looks ≥~0.87)

```bash
bash rockfish/slurm/submit_publish_3b.sh \
  --account sfried3_gpu --qos qos_gpu
# If OOM on 40GB: add --partition ica100
```

Do **not** start C/D until A’s embed job is **Running or completed** (or you accept separate GPU queue contention). A and C can share the cluster, but one A100 at a time is the courteous default.

#### Full publication campaign (650M → 3B, auto-resume) — preferred paper path

Rockfish GPU max walltime is **72 h** (not 48 h); shared CPU ≈ **36 h** ([ARCH partitions](https://docs.arch.jhu.edu/en/latest/1_Clusters/Rockfish/3_Slurm/Partitions.html)).
Jobs resume from `cv_progress.json` folds; a login-node watchdog resubmits after TIMEOUT until both packages exist, and **auto-escalates OOM** (half batch → `ica100` → smaller batch).  
CAID rigor: leak-free train filter + CAID3 scoring (+ CAID4 blind submission if targets present). Details: **[rockfish/PUBLISH_FULL.md](rockfish/PUBLISH_FULL.md)**.

```bash
cd ~/DisorderNet && git pull && source ~/venvs/disordernet/bin/activate
export DISORDERNET_MAIL_USER=marenatommaso@gmail.com
export DISORDERNET_RESULTS=$HOME/disordernet_runs
# optional: export CAID4_TARGETS=$HOME/DisorderNet/data/caid4_targets.fasta
bash rockfish/slurm/submit_publish_full.sh
squeue -u $USER
python rockfish/publish_campaign.py status --campaign "$(ls -t ~/disordernet_runs/campaign_*.json | head -1)"
```

**Done when** status is `"done"` and both exist:

```bash
ls ~/disordernet_runs/publish_650m_*/publish_package/PACKAGE_README.md
ls ~/disordernet_runs/publish_3b_*/publish_package/PACKAGE_README.md
```

#### How you know it’s finished

| Signal | Meaning |
|--------|---------|
| `squeue -u $USER` empty (or job gone) | Slurm no longer holding the job |
| `sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed -P` → `COMPLETED\|0:0` | Success |
| Same → `FAILED` / `TIMEOUT` / `CANCELLED` / `OUT_OF_MEMORY` | Stop; read `logs/*.err` before resubmitting |
| Dependent job stays `PD` forever with reason `Dependency` | Parent never reached `COMPLETED` — `scancel` child, fix parent, resubmit both |
| Log stops growing + metrics file exists | Safe to inspect results (below) |

Monitor while waiting:

```bash
squeue -u $USER
# ST: PD=pending, R=running. Reason column shows QOSMax… / Dependency / Resources.
tail -f ~/DisorderNet/logs/dn-v8-embed_*.out     # embed progress
tail -f ~/DisorderNet/logs/dn-v8-cv_*.out        # v8 CV / ensemble
tail -f ~/DisorderNet/logs/dn-*ultra*.out        # publish GPU train (name varies)
```

Email is off by default in the sbatch headers; uncomment `#SBATCH --mail-*` if you want END/FAIL mail.

#### Expected wall time (once the job actually starts)

| Stage | Partition | Typical wall | Notes |
|-------|-----------|--------------|-------|
| Queue wait | — | minutes → many hours | Not in your control |
| Prefetch DisProt + ESM (login) | login | ~5–15 min | Already done if cache present |
| v8 GPU extract (3 backbones) | `a100` | **~1–2 h** | Weights must be prefetched |
| v8 CPU pipeline (3× v7 + homology + ensemble) | `shared` | **~6–12 h** | No GPU held |
| Boltz batch | `a100` | hours–days | Optional |
| Publish 650M ultra + clean + package | `a100`→`shared` | **~2–4+ days** | ~50–100 GPU-h class |
| Publish 3B ultra + clean + package | `a100`/`ica100`→`shared` | **~3–5+ days** | ~60–120 GPU-h class |

#### What you get (artifacts to open)

**After v8 pipeline completes:**

```bash
export DISORDERNET_V8_DIR=$HOME/scr4_sfried3/disordernet_v8
cat $DISORDERNET_V8_DIR/ensemble/results_v8/metrics.json       # expect AUC ~0.85–0.86
cat $DISORDERNET_V8_DIR/ensemble_hom/results_v8/metrics.json   # homology; slightly lower
ls  $DISORDERNET_V8_DIR/650m/results_v7/                       # per-backbone OOF + metrics
# Also: calibrated probs + conformal intervals under those result dirs
```

**After publish 650M/3B package job completes:**

```bash
ls ~/disordernet_runs/publish_650m_*/publish_package/
less ~/disordernet_runs/publish_650m_*/publish_package/PACKAGE_README.md
cat  ~/disordernet_runs/publish_650m_*/publish_package/comparison.json
# Required go/no-go: sota_postprocess_report.json, structure_distrust_benchmark.json
```

Re-package without re-training (if package job failed but train finished):

```bash
python rockfish/publish_submit.py package \
  --root-workdir ~/disordernet_runs/publish_650m_<stamp> \
  --kind 650m --strict
```

Then fill [`docs/METHODS_CHECKLIST.md`](docs/METHODS_CHECKLIST.md) from the package.

**Pull to your laptop** (run locally, not on Rockfish):

```bash
scp -r rockfish:scr4_sfried3/disordernet_v8/ensemble ./dn_v8_ensemble
scp -r rockfish:scr4_sfried3/disordernet_v8/ensemble_hom ./dn_v8_ensemble_hom
scp -r rockfish:disordernet_runs/publish_650m_* ./publish_650m
```

#### When to run the “rest” of the commands

| Just finished… | Run next… | Why |
|----------------|-----------|-----|
| Prefetch + env only | **A** (v8 embed → pipeline) | Fastest honest numbers |
| v8 `metrics.json` present | Inspect AUC; optionally **B** then **C** | Paper LoRA / distrust track |
| `publish_650m_*/publish_package/` present | Checklist + go/no-go; only then **D** (3B) | 3B is expensive |
| Package missing required JSON | `publish_submit.py package --strict` | No need to retrain |

---

### Tests only (any machine)

```bash
git clone https://github.com/Tommaso-R-Marena/DisorderNet.git
cd DisorderNet && git checkout master
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Results

### Comprehensive Benchmark

**Important:** Table A lists published reference AUCs from CAID/DisProt studies (different splits/protocols). Table B lists **our** runs on the same in-repo DisProt 5-fold protein-grouped CV. Do not treat Table A rows as head-to-head comparisons.

#### Table A — Literature reference (not head-to-head)

| Method | AUC-ROC | Source | Protocol |
|--------|---------|--------|----------|
| AlphaFold3-pLDDT | 0.9324 | measured, `colab/caid3_official.py` | CAID3 **Disorder-PDB**, rank 14, 319/319 |
| AlphaFold-pLDDT | 0.9342 | measured, `colab/caid3_official.py` | CAID3 **Disorder-PDB**, rank 13, 319/319 |
| IUPred3 | 0.789 | [CAID](https://caid.idpcentral.org/) | CAID benchmark |
| flDPnn | 0.814 | [CAID](https://caid.idpcentral.org/) | CAID benchmark |
| SETH (ProtT5+CNN) | 0.830 | [Ilzhöfer et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9580958/) | Published |
| DisorderNet v6 (CPU) | 0.831 | This repo | DisProt 5-fold **protein** split (legacy, not homology-separated) |
| ESM2_650M-LoRA | 0.880 | [LoRA-DR](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i439/8199360) | CAID1 |
| flDPnn3a | 0.9006 | measured, `colab/caid3_official.py` | CAID3 **Disorder-PDB**, rank 44, 319/319 |
| ESMDisPred-2PDB | 0.9366 | measured, `colab/caid3_official.py` | CAID3 **Disorder-PDB**, rank 8, 284/319 — it declines 35 |
| AlphaFold-rsa (training-free) | 0.950 | [CAID3 official](https://caid.idpcentral.org/) | CAID3 Disorder-PDB, 319/319 |
| **PUNCH2 (CAID3 Disorder-PDB leader)** | **0.955** | [CAID3 official](https://caid.idpcentral.org/) | CAID3 Disorder-PDB, 319/319 |

#### Table B — Our DisProt CV (directly comparable within table)

| Method | AUC-ROC | AP | Status |
|--------|---------|-----|--------|
| DisorderNet v6 (ESM-2 8M + GBDT) | 0.831–0.840 | 0.537 | Legacy protein-split (`results_v6/metrics.json`); measured 0.7804 under genuine homology splits |
| DisorderNet v7 (ESM-2 35M, train-only PCA) | 0.848 | 0.558 | Verified (`run_v7.py`, leakage-free) |
| DisorderNet v7 (ESM-2 650M) | 0.851 | 0.569 | Verified (`run_v7.py`) |
| **DisorderNet v8 (multi-scale ensemble)** | **0.857** | 0.578 | Verified (`run_v8_multiscale.py`); homology split 0.853 |
| DisorderNet GPU (ESM-2 650M + LoRA) | 0.817 | — | Legacy protein-split Colab run; measured 0.7454 under genuine homology splits |

#### Legacy combined view (reference only)

| Method | AUC-ROC | Δ vs AF3 (ref.) | Source |
|--------|---------|-----------------|--------|
| AlphaFold3-pLDDT (CAID3 Disorder-PDB) | 0.9324 | — | measured; different benchmark, not comparable to this column |
| AlphaFold-pLDDT (CAID3 Disorder-PDB) | 0.9342 | — | measured; different benchmark, not comparable to this column |
| IUPred3 | 0.789 | +5.6% | [CAID](https://caid.idpcentral.org/) |
| DisorderNet v4 (physics only) | 0.794 | +6.3% | This work |
| flDPnn (CAID1/2 best) | 0.814 | +9.0% | [CAID](https://caid.idpcentral.org/) |
| DisorderNet v5 (ESM 8M, PCA-32) | 0.823 | +10.2% | This work |
| SETH (ProtT5+CNN) | 0.830 | +11.1% | [Ilzhöfer et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9580958/) |
| DisorderNet v6 (ESM 8M, PCA-48) | 0.831 | +11.3% | This work — legacy protein split |
| flDPnn3a (CAID3 Disorder-PDB) | 0.9006 | — | measured; different benchmark, not comparable to this column |
| ESM2_35M-LoRA | 0.868 | +16.2% | [LoRA-DR](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i439/8199360) |
| ESM2_650M-LoRA | 0.880 | +17.8% | [LoRA-DR](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i439/8199360) |
| PUNCH2 (CAID3 Disorder-PDB leader) | 0.955 | — | [CAID3 official](https://caid.idpcentral.org/) — different benchmark, not comparable to this column |

### Version Progression

| Version | AUC | Features | Key Addition |
|---------|-----|----------|-------------|
| v4 | 0.794 | 118 | Multi-scale physicochemical features |
| v5 | 0.823 | 214 | + ESM-2 8M embeddings (PCA-32) |
| v6 | 0.831–0.840 | 406 | + PCA-48, ESM variance/context features |
| v7 | 0.848–0.850 | ~855 | PCA-96 + global pooling, LGB+XGB+HistGBM blend, smoothing, train-only PCA, calibration + conformal |
| **v8** | **0.857 (0.853 homology)** | ensemble | multi-scale PLM ensemble (35M+150M+650M OOF, equal weights) |
| GPU (Colab, legacy protein split) | 0.817 | 1280+phys | ESM-2 650M + LoRA + segment-aware ES + v6 ensemble |
| **GPU (homology split, measured)** | **0.7454 alone / 0.8011 +v6 / 0.8196 +AF** | 1280+phys | Same recipe, genuine homology separation |
| GPU SOTA track (`sota` profile) | *aspirational* ≥0.88–0.90 | — | Transformer head, Dice+EMA, 3-way stack, compact ckpt |
| GPU ULTRA track (`ultra` profile) | *aspirational* 0.88–0.92; **measured 0.8196** | — | Rich features, FFN LoRA, v6-pro meta-stack, MC-dropout TTA |
| GPU ULTRA 3B (`ultra3b` profile) | *aspirational* 0.90–0.93, unmeasured | — | ESM-2 3B backbone on A100 40GB+ |
| GPU ULTRA + function (`ultra_fun`) | disorder + IDR roles | — | Multi-label Disorder→function head |

### Performance ceiling (honest)

On **DisProt 5-fold CV** with ESM-2 650M, the realistic band is:

| Stage | Typical pooled AUC |
|-------|-------------------|
| GPU baseline (legacy protein split) | 0.817 |
| GPU (measured, homology split) | 0.8196 with AF fusion |
| + ultra training + 7b–7d stack | 0.88–0.92 (aspirational, not achieved) |
| + ESM-2 3B (`ultra3b`) + full stack | 0.90–0.93 (aspirational, unmeasured) |
| + multi-seed blend (2–3 seeds) | +0.005–0.015 |
| PUNCH2 (CAID3 Disorder-PDB, different protocol) | 0.955 reference |

Breaking **0.90+ consistently** on DisProt likely needs **ESM-2 3B** (`ultra3b`) or **CAID3-homologous training** — not more post-hoc stacking on 650M alone. Use the [Quick Screen notebook](colab/DisorderNet_Colab_QuickScreen.ipynb) before a full ultra run.

### Backbone upgrade — what to do

| Step | Action | Notebook | Settings | ~Time (A100) |
|------|--------|----------|----------|--------------|
| 1 | Go/no-go | [Quick Screen](colab/DisorderNet_Colab_QuickScreen.ipynb) | `SCREEN_MODE="standard"` (`screen_plus`), `SCREEN_BACKBONE="650M"` | 2–3 h (40GB) / often &lt;1–1.5 h (80GB) |
| 2 | Full 650M ultra (if screen ≥ MODERATE) | [Colab Pro](colab/DisorderNet_Colab_Pro.ipynb) | `QUALITY_PROFILE="ultra"` | 18–24 h |
| 3 | **3B paradigm test** | Quick Screen | `SCREEN_BACKBONE="3B"`, `SCREEN_MODE="paradigm"` | 8–12 h |
| 4 | **Full 3B production** | Colab Pro | `QUALITY_PROFILE="ultra3b"`, `ESM_BACKBONE="3B"` | 30–40 h |
| 5 | Multi-seed (optional) | Colab Pro Cell 7e | seeds 42 + 43 | 2× step 4 |

**Colab for 3B:** Runtime → **A100 40GB** + **High RAM**. Run `!pip install -q lightgbm xgboost`. **Do not use T4** for 3B.

**Decision rule:** If step 2 stacked AUC **< 0.87**, skip another 650M run and do step 3. If step 3 stacked AUC **≥ 0.86**, commit to step 4.

### Rockfish / Slurm (recommended for production)

If you have access to JHU Rockfish (or any Slurm cluster with A100s), use the HPC pipeline instead of Colab for 3B runs and multi-day jobs.

**Operator source of truth (finish signals, timelines, artifacts, stuck-job recovery):**  
**[Path C — Rockfish ops guide](#path-c--rockfish-ops-guide-what-you-know-what-to-run-when-youre-done)** in this README.  
Canonical flags/layouts: **[rockfish/README.md](rockfish/README.md)**.  
v8 copy-paste: **[rockfish/V8_MULTISCALE.md](rockfish/V8_MULTISCALE.md)**.

```bash
cd ~/DisorderNet && git checkout master && git pull
source ~/venvs/disordernet/bin/activate
mkdir -p logs

export DISORDERNET_ACCOUNT=sfried3          # CPU / shared
export DISORDERNET_GPU_ACCOUNT=$(sacctmgr -nP show assoc user=$USER format=account,qos \
  | awk -F'|' '/qos_gpu/{print $1; exit}')  # usually sfried3_gpu

# v8 first (hours) — helper always sets --qos=qos_gpu:
bash rockfish/slurm/submit_v8.sh

# then optional publish bundles (days):
bash rockfish/slurm/submit_publish_650m.sh \
  --account "${DISORDERNET_GPU_ACCOUNT:-sfried3_gpu}" --qos qos_gpu
# optional: bash rockfish/slurm/submit_publish_3b.sh --account … --qos qos_gpu [--partition ica100]
```

**How you know it’s done:** `squeue -u $USER` empty + `sacct -j <id> … COMPLETED|0:0`, then open  
`$DISORDERNET_V8_DIR/ensemble/results_v8/metrics.json` and/or  
`~/disordernet_runs/publish_650m_*/publish_package/PACKAGE_README.md`.

Ultra on Rockfish uses **homology-separated CV** (BLASTp clustering; genuinely separated only since the 2026-08-10 fix), optional **train-time pLDDT** (disabled in clean companions), and **CAID3** scoring against the official references (`colab/caid3_official.py`), whose Disorder-PDB leader is PUNCH2 at 0.955.

## Documentation

All project documentation lives under `docs/` and `rockfish/README.md`.  
**Operators:** start at **[From scratch](#from-scratch-start-here)** → **[Path C ops guide](#path-c--rockfish-ops-guide-what-you-know-what-to-run-when-youre-done)**. Then:

| Document | What it covers |
|----------|----------------|
| **[rockfish/PUBLISH_FULL.md](rockfish/PUBLISH_FULL.md)** | 650M→3B campaign + watchdog resume, mail, Rockfish 72h limits, operator commands |
| **[rockfish/README.md](rockfish/README.md)** | Canonical Rockfish/Slurm usage: setup, publish path (`submit_publish_650m` / `submit_publish_3b`), packaging (`--kind` / `--strict`), artifacts, go/no-go, env vars, Boltz/AF3 |
| **[docs/ROCKFISH_PUBLISH_RUNBOOK.md](docs/ROCKFISH_PUBLISH_RUNBOOK.md)** | Short operator pointer to the publish path + re-package CLI |
| **[docs/METHODS_CHECKLIST.md](docs/METHODS_CHECKLIST.md)** | Preprint freeze checklist (credibility floor, labeled distrust, contamination, atlas, non-claims) |
| **[docs/STRUCTURE_DISTRUST_ATLAS.md](docs/STRUCTURE_DISTRUST_ATLAS.md)** | Structure-distrust product thesis: labeled rescue vs proxy flags, module map, Rockfish eval artifacts |
| **[docs/IDR_BIOLOGY_LAYER.md](docs/IDR_BIOLOGY_LAYER.md)** | Post-structure IDR biology layer claim, non-goals, module map, phased roadmap |
| **[docs/PAPER_OUTLINE_STRUCTURE_DISTRUST.md](docs/PAPER_OUTLINE_STRUCTURE_DISTRUST.md)** | Paper outline: core claim, evidence stack, figure list, methods red lines |
| **[docs/HOMOLOGY_HOLDOUT.md](docs/HOMOLOGY_HOLDOUT.md)** | Homology-aware CV wording (what the code does / does not claim vs official CAID filters) |
| **[AGENTS.md](AGENTS.md)** | Contributor / agent notes (venv, pytest, CPU pipeline paths, Rockfish publish conventions) |

### Structure distrust (paper claim)

After Boltz / AlphaFold produce a fold, DisorderNet is the **default post-structure distrust layer**: it flags where structure confidence should not be trusted on IDRs and prefers an independent disorder map (+ optional roles).

Do **not** conflate:

| Definition | Inputs | Publish as rescue? |
|------------|--------|--------------------|
| **Labeled hallucination / rescue** | Independent DisProt labels ∩ high pLDDT; DN predicts disorder | **Yes** |
| **Proxy distrust flags** | DN disorder call ∩ high pLDDT | **No** (tautological) |

Load-bearing evidence order (see [`docs/PAPER_OUTLINE_STRUCTURE_DISTRUST.md`](docs/PAPER_OUTLINE_STRUCTURE_DISTRUST.md)):

1. CAID3 / DisProt credibility floor  
2. Labeled hallucination rescue  
3. Matched inverse-pLDDT baseline (`delta_auc_dn_minus_plddt` + per-fold stats)  
4. Downstream mask utility  
5. Proteome atlas resource  
6. Contamination audit + clean ablations when risk ≠ low  

Freeze checklist: [`docs/METHODS_CHECKLIST.md`](docs/METHODS_CHECKLIST.md).  
Rockfish eval artifacts: `structure_distrust_benchmark.json`, `structure_distrust_atlas.jsonl` / `.tsv` (see [`docs/STRUCTURE_DISTRUST_ATLAS.md`](docs/STRUCTURE_DISTRUST_ATLAS.md)).

### IDR biology layer

Post-structure layer that answers what AF/Boltz cannot by design: where the chain is disordered, what IDRs might do, where structure is overconfident, optional Boltz variance proxy, and conditional-disorder boundary flags — **not** an AF replacement and **not** MD ensembles. Details: [`docs/IDR_BIOLOGY_LAYER.md`](docs/IDR_BIOLOGY_LAYER.md). Stage: `python rockfish/run_disordernet.py idr-layer`.

### Homology / holdout language

Ultra defaults to homology-aware grouping (`split_method="homology"`, ~40% identity within length bins via `SequenceMatcher`). This is **not** MMseqs2 and **not** official CAID homology filters — use the careful wording in [`docs/HOMOLOGY_HOLDOUT.md`](docs/HOMOLOGY_HOLDOUT.md).

### Publish path (Rockfish)

See **[Path C](#path-c--rockfish-ops-guide-what-you-know-what-to-run-when-youre-done)** for accounts, monitoring, and finish signals.

```bash
bash rockfish/slurm/submit_publish_650m.sh \
  --account "$DISORDERNET_GPU_ACCOUNT" --qos "$DISORDERNET_GPU_QOS"
# then open publish_package/ → METHODS_CHECKLIST → go/no-go on numbers
python rockfish/publish_submit.py package --root-workdir … --kind 650m --strict
```

Full exact usage: **[rockfish/README.md — Publish path](rockfish/README.md#publish-path-exact-usage)**.

### SOTA track (`QUALITY_PROFILE = "sota"`)

Designed to close the gap to the CAID3 Disorder-PDB leader, PUNCH2 at 0.955:

| Component | Detail |
|-----------|--------|
| LoRA | rank 64, last 16 layers, 8-layer ESM fusion |
| Head | Multi-scale CNN + 2-layer Transformer encoder |
| Loss | Focal + soft Dice (region-aware) + label smoothing |
| Training | EMA weights for eval/checkpoint selection |
| Checkpoints | **Compact** (~50–150 MB) — trainable weights only |
| Post-CV | Cell 7c: GPU + v6 + physics prior 3-way stack |


## Architecture

```
                    ┌─────────────────────────────┐
                    │     Protein Sequence          │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   ESM-2 Language Model        │
                    │   (8M CPU / 650M GPU+LoRA)    │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼─────────┐ ┌───────▼────────┐ ┌─────────▼─────────┐
    │  Per-residue       │ │ Multi-scale    │ │ ESM Variance      │
    │  PCA Embeddings    │ │ ESM Context    │ │ Features           │
    │  (48-1280 dim)     │ │ (4 scales)     │ │ (2 scales)         │
    └─────────┬─────────┘ └───────┬────────┘ └─────────┬─────────┘
              │                    │                     │
              └────────────────────┼────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼─────────┐ ┌───────▼────────┐           │
    │  118 Physicochemical│ │   Merged       │           │
    │  Features          │ │   Feature       │◄──────────┘
    │  (7 scales)        │ │   Vector        │
    └─────────┬─────────┘ └───────┬────────┘
              │                    │
              └────────┬───────────┘
                       │
              ┌────────▼────────┐
              │  LightGBM +     │  (CPU version)
              │  XGBoost        │
              │  Ensemble       │
              ├─────────────────┤
              │  OR              │
              │  CNN Head +     │  (GPU/Colab version)
              │  LoRA Tuning    │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Per-residue      │
              │ Disorder         │
              │ Probability      │
              └─────────────────┘
```

## Quick Start

The numbered **[From scratch](#from-scratch-start-here)** section above is the canonical zero-to-result guide (CPU / Colab / Rockfish). Short links below.

### Option 1a: Quick paradigm screen (~2–3 hours on A100-40GB, recommended first)

[![Open Quick Screen in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_QuickScreen.ipynb)

Run this **before** the full 18–24h CV to get a breakthrough go/no-go verdict on the current paradigm (ESM-2 650M + LoRA + v6 ensemble).

`SCREEN_MODE="standard"` trains the **`screen_plus` mini-ultra** stack (SOTA head, rich features, FFN LoRA, homology splits)—not the old toy CNN `screen` recipe—so STOP/MODERATE is a faithful signal for full ultra. A100-80GB often finishes in under ~1–1.5 h with early stopping; wall clock alone is not a quality signal. After code updates, use **Runtime → Restart session** so Colab is not stuck on an old checkout.

1. Open [`colab/DisorderNet_Colab_QuickScreen.ipynb`](colab/DisorderNet_Colab_QuickScreen.ipynb) in Colab (badge above)
2. Select **Runtime → Change runtime type → GPU (A100 or L4) + High RAM**
3. Set `SCREEN_MODE = "standard"` (or `"flash"` for a coarse smoke test, `"paradigm"` for a larger mini-ultra subset)
4. Run all cells — outputs `quick_screen_report.json` with tier **HIGH / MODERATE / LOW / STOP** (mode-aware uplift; `flash` alone cannot green-light ultra)
5. Proceed to the full notebook only if the verdict recommends full ultra CV

### Option 1b: Full GPU cross-validation (Google Colab)

[![Open Full GPU CV in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_Pro.ipynb)

1. Click the badge above
2. Select **Runtime → Change runtime type → GPU (A100 or L4) + High RAM**
3. Run all cells (~12–18 hours for full 5-fold CV with ESM-2 650M)

The notebook auto-tunes batch size to your GPU VRAM, uses mixed precision (bfloat16 on A100), filters DisProt annotations by disorder-related terms, and shows live per-epoch metrics. Set `MOUNT_DRIVE = True` to persist checkpoints, DisProt cache, AF pLDDT cache, and final reports on Drive (`MyDrive/DisorderNet/results/`).

### Evaluation rigor (GPU Colab path)

- **Deterministic CV splits** — proteins sorted by DisProt ID; all modules share splits via `colab/cv_splits.py`
- **Aligned OOF metrics** — CAID segment F1 and biological utility use per-protein out-of-fold alignment (not fold-concat order)
- **Threshold reporting** — F1@0.5 (unbiased) plus per-fold optimal thresholds alongside pooled F1_max
- **Statistical validation** — per-fold paired sign test + Wilcoxon vs inverse-pLDDT baseline
- **Reproducibility manifest** — `run_manifest.json` records git revision, config/dataset fingerprints, and DisProt snapshot metadata
- **Resume safety** — `checkpoints/cv_progress.json` v2 validates protein list, config, and DisProt hash before resuming CV

### Option 2: CPU (Quick, no GPU needed)

```bash
pip install -r requirements-cpu.txt

# Run the full pipeline (data + results default to repo-local ./data and ./results_v6)
python fetch_disprot.py          # Download DisProt data
python extract_esm_embeddings.py  # Extract ESM-2 8M embeddings
python run_v6_mem.py              # Train and evaluate
python generate_figures_v6.py     # Generate figures
```

### Optimized model + per-sequence predictor (v7)

`run_v7.py` is an optimized CPU model (PCA-96 ESM features + global pooling, a
LightGBM+XGBoost+HistGBM blend, and contiguity smoothing) evaluated with
leakage-free 5-fold CV (PCA fit on the train fold only): pooled AUC **0.848**
with ESM-2 35M, **0.850** with ESM-2 150M/650M (vs the v6 0.840 baseline).

`run_v8_multiscale.py` ensembles the v7 out-of-fold predictions across ESM-2
backbones (35M + 150M + 650M, equal weights → leakage-free). Different PLM scales
carry complementary disorder signal, so the ensemble reaches **0.857** pooled AUC —
the best honest CPU result — with calibration ECE 0.005 and conformal coverage 0.90.
(Under this GBDT-on-PCA recipe, single-backbone AUC saturates ~0.85 regardless of
size, because PCA compression caps how much PLM signal the trees can use; the ensemble
and the GPU LoRA path are the ways past that.)

#### Honest evaluation: random vs homology split

Random protein-ID GroupKFold can leak signal via near-duplicate homologs. We also
report **homology-split** CV (`>=40%` identity clusters via BLASTp; genuinely separated only since the 2026-08-10 fix; run with
`DISORDERNET_SPLIT=homology python run_v7.py`). The small gap confirms the random
number is not badly inflated:

| Model | random split | homology split |
|-------|-------------:|---------------:|
| ESM-2 35M | 0.8479 | 0.8396 |
| ESM-2 150M | 0.8498 | 0.8457 |
| ESM-2 650M | 0.8505 | 0.8487 |
| **Ensemble** | **0.8568** | **0.8525** |

#### Calibrated + conformal confidence everywhere

The calibration + conformal layer (`confidence.py`) is also wired into the **GPU/LoRA
pipeline** (`colab/confidence_layer.py`): `run_cross_validation` now attaches a
cross-fitted calibration + conformal report to its summary, and `fit_confidence` /
`apply_confidence` produce calibrated probabilities + `confident/abstain` decisions
for new sequences from the SOTA models too.

It also adds capabilities most disorder predictors lack (`confidence.py`):
**isotonic-calibrated probabilities** (ECE ~0.049 → ~0.004, ranking preserved) and
**split-conformal per-residue prediction sets** with a coverage guarantee
(`confident disorder / confident order / abstain`).

Train a deployable bundle and predict on any sequence:

```bash
python train_predictor.py                         # -> results_v7/predictor_bundle.joblib
python predict_disorder.py --seq MDVFMKGLSKAKEGVV...   # or --fasta proteins.fasta --out preds.json
```

Each residue gets a calibrated `p(disorder)` plus a conformal decision. On
α-synuclein this correctly highlights the disordered acidic C-terminal tail while
calling folded lysozyme ordered.

## Key Innovation: Why AlphaFold 3 Fails at Disorder

AF3's diffusion architecture generates structured coordinates for every residue, then assigns confidence post-hoc. It has **no concept of "this region should not have structure."** Our model is designed from the ground up to distinguish order from disorder:

1. **Multi-scale disorder propensity profiling** across 5 length scales (7–100 residues)
2. **ESM-2 language model embeddings** capturing evolutionary disorder signals
3. **Property variance features** detecting heterogeneity at disorder boundaries
4. **Key amino acid composition** tracking 12 disorder/order indicator residues

## Biological Significance

- **30-40% of the human proteome** contains IDRs
- **80% of cancer-associated proteins** have long disordered regions
- AF3's hallucinations have serious consequences for drug discovery and disease research
- Accurate IDR prediction is essential for understanding signaling, transcription, and neurodegeneration

## Benchmark Sources

| Source | Citation |
|--------|----------|
| CAID3 rankings | [Mehdiabadi et al., Proteins 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/) |
| AF3 hallucinations | [Sreekumar et al., arXiv 2025](https://arxiv.org/abs/2510.15939) |
| AF2-pLDDT AUC | [Comparative evaluation, CSBJ 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10782001/) |
| AF3 limitations | [EMBL-EBI](https://www.ebi.ac.uk/training/online/courses/alphafold/) |
| ESM2-LoRA | [LoRA-DR-suite, Bioinformatics 2025](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i439/8199360) |
| ESMDisPred | [Kabir et al., bioRxiv 2026](https://pubmed.ncbi.nlm.nih.gov/41648466/) |
| pLM impact review | [Modern resources, CMLS 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12913823/) |

## Files

### Documentation

| File | Description |
|------|-------------|
| [`rockfish/README.md`](rockfish/README.md) | **Canonical Rockfish usage** (publish path, artifacts, go/no-go, env vars) |
| [`docs/ROCKFISH_PUBLISH_RUNBOOK.md`](docs/ROCKFISH_PUBLISH_RUNBOOK.md) | Short pointer to rockfish README publish path + re-package CLI |
| [`docs/METHODS_CHECKLIST.md`](docs/METHODS_CHECKLIST.md) | Preprint freeze checklist (credibility, distrust, contamination, atlas) |
| [`docs/STRUCTURE_DISTRUST_ATLAS.md`](docs/STRUCTURE_DISTRUST_ATLAS.md) | Structure-distrust thesis, labeled vs proxy, module map, eval artifacts |
| [`docs/IDR_BIOLOGY_LAYER.md`](docs/IDR_BIOLOGY_LAYER.md) | IDR biology layer claim, non-goals, modules, roadmap |
| [`docs/PAPER_OUTLINE_STRUCTURE_DISTRUST.md`](docs/PAPER_OUTLINE_STRUCTURE_DISTRUST.md) | Paper outline: claim, evidence stack, figures, methods red lines |
| [`docs/HOMOLOGY_HOLDOUT.md`](docs/HOMOLOGY_HOLDOUT.md) | Homology CV protocol and publishable wording |
| [`AGENTS.md`](AGENTS.md) | Agent/contributor environment, pytest, Rockfish conventions |

### Notebooks & HPC

| File | Description |
|------|-------------|
| `colab/DisorderNet_Colab_v8_MultiScale.ipynb` | **v8 multi-scale ensemble (GPU)**: multi-backbone extraction → v7 CV → v8 ensemble → calibration/conformal → predictor — [Open in Colab](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_v8_MultiScale.ipynb) |
| `colab/DisorderNet_Colab_QuickScreen.ipynb` | **Quick breakthrough screen** (mini-ultra `screen_plus` go/no-go before full CV) — [Open in Colab](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_QuickScreen.ipynb) |
| `colab/DisorderNet_Colab_Pro.ipynb` | Full GPU notebook (ESM-2 650M + LoRA; now auto-reports calibrated conformal confidence) — [Open in Colab](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_Pro.ipynb) |
| `rockfish/V8_MULTISCALE.md` | **Exact Rockfish runbook** for the v8 ensemble (GPU extract + CPU CV) |
| `rockfish/slurm/v8_extract_embeddings.sbatch` | GPU embedding extraction for v8 (`a100`) |
| `rockfish/slurm/v8_pipeline.sbatch` | CPU v7×backbones → v8 ensemble → predictor (`shared`) |
| `colab/quick_screen.py` | Quick screen logic (stratified subsample, verdict tiers) |
| `colab/esm_backbone.py` | ESM-2 backbone registry (650M → 3B) + VRAM batch presets |
| `rockfish/run_disordernet.py` | HPC CLI: screen / cv / stack / postprocess / full / pipeline / eval / atlas / idr-layer |
| `rockfish/slurm/pipeline_ultra.sbatch` | Full production + eval + CAID3 |
| `rockfish/slurm/pipeline_ultra_clean.sbatch` | Contamination-clean companion (separate workdir) |
| `rockfish/slurm/submit_publish_650m.sh` | **Script 1:** 650M ultra + clean → `publish_package/` |
| `rockfish/slurm/submit_publish_3b.sh` | **Script 2:** ultra3b + clean → `publish_package/` |
| `rockfish/publish_submit.py` | **Preferred CLI:** `submit-650m` / `submit-3b` / `package --kind --strict` |
| `rockfish/utils.py` | Shared artifact catalog, RunSpec, sbatch helpers, git provenance |
| `rockfish/package_publish_results.py` | Package library (prefer `publish_submit.py package --kind`) |
| `rockfish/mirror_results.py` | Parallel mirror of checkpoint/report artifacts |
| `rockfish/slurm/multi_seed.sbatch` | Slurm array for seeds 42/43/44 |
| `rockfish/slurm/_common.sh` | Shared Slurm setup / run / mirror / package helpers |

### Core library modules

| File | Description |
|------|-------------|
| `colab/homology_splits.py` | Homology-clustered CV (BLASTp preferred; see docs/HOMOLOGY_HOLDOUT.md) |
| `colab/caid3_eval.py` | CAID3 Disorder-PDB benchmark harness |
| `colab/structure_encoder.py` | Train-time pLDDT feature channel |
| `colab/predict_batch.py` | FASTA proteome inference + `.caid` export |
| `colab/novel_use_cases.py` | AF hallucination screening, rescue manifest, IDR function annotation |
| `colab/function_predict.py` | Disorder→function multi-label head, labels, OOF metrics |
| `colab/idr_biology_layer.py` | Compose IDR biology layer + proteome export |
| `colab/structure_distrust_atlas.py` | Proteome structure-distrust atlas + mask utility |
| `colab/hallucination_benchmark.py` | Labeled hallucination / rescue benchmarks |
| `colab/inference_tta.py` | MC-dropout test-time augmentation (ultra Cell 7d) |
| `colab/multi_seed_blend.py` | Optional multi-seed OOF average (Cell 7e) |
| `colab/disordernet_gpu.py` | Colab training module (data, model, CV loop) |
| `colab/cv_splits.py` | Shared deterministic GroupKFold splits + fingerprints |
| `colab/run_manifest.py` | Reproducibility manifest + Drive mirror helpers |
| `colab/sota_heads.py` | SOTA CNN+Transformer prediction head |
| `colab/sota_losses.py` | Focal + Dice composite training loss |
| `colab/sota_ensemble.py` | Three-way OOF stack (GPU + v6 + physics prior) |
| `colab/compact_checkpoint.py` | ~150 MB fold checkpoints (LoRA+head only) |
| `colab/colab_figures.py` | Publication figure generator for GPU runs |
| `colab/biological_utility.py` | Phase 1 biological utility (segments, functional enrichment) |
| `colab/af_plddt.py` | AlphaFold DB pLDDT fetch + alignment |
| `colab/af3_plddt.py` | AlphaFold 3 pLDDT ingest from Drive outputs |
| `colab/af3_colab.py` | Colab/Drive setup for AF3 weights and optional subset runs |
| `colab/af_hallucination.py` | Phase 2 hallucination rescue metrics |
| `colab/phase3_synthesis.py` | Phase 3 fusion calibration & integrated report |
| `colab/benchmark_tables.py` | Matched vs literature benchmark tables (Tier 1) |
| `colab/caid_reporting.py` | CAID-style metrics + stratified evaluation + per-fold thresholds |
| `colab/statistical_validation.py` | Per-fold paired sign/Wilcoxon tests + bootstrap CIs |
| `colab/inference_fusion.py` | Post-CV AF pLDDT fusion (α-blend; AF2+AF3 combined map) |
| `colab/downstream_refresh.py` | Refresh CAID/bio/benchmark after fusion updates |
| `run_v6_mem.py` | CPU version with ESM-2 8M + GBDT ensemble |
| `run_v7.py` | **v7** optimized CPU model (train-only PCA, LGB+XGB+HistGBM, smoothing, calibration + conformal); `DISORDERNET_SPLIT=homology` and `DISORDERNET_PCA_DIM` supported |
| `run_v8_multiscale.py` | **v8** multi-scale PLM ensemble over per-backbone OOF (equal weights, leakage-free) |
| `confidence.py` | Isotonic calibration + ECE + split-conformal prediction sets (shared CPU/GPU) |
| `colab/confidence_layer.py` | Wires calibration + conformal into the GPU `run_cross_validation` fold results |
| `predictor.py` / `train_predictor.py` / `predict_disorder.py` | Deployable predictor bundle + train/predict CLIs (FASTA or raw sequence) |
| `disordernet_paths.py` | Portable, env-overridable data/results paths (`DISORDERNET_HOME`, …) |
| `run_v5_esm.py` | v5 with PCA-32 ESM features |
| `extract_esm_embeddings.py` | ESM-2 embedding extraction (35M) |
| `experiments/extract_esm_150m.py`, `experiments/extract_esm_650m.py` | Larger-backbone embedding extraction for v8 |
| `experiments/optimize_cpu.py` | Leakage-free feature/model/smoothing sweep |
| `fetch_disprot.py` | DisProt database downloader |
| `generate_figures_v6.py` | Publication figure generator (importable `generate()`) |
| `results_v6/` | v6 metrics, predictions, figures |

## Testing & CI

The repository has an extensive pytest suite (**330+ tests**) covering the CPU
pipeline helpers, the confidence layer (calibration + conformal, incl. empirical
coverage-guarantee checks), the deployable predictor, the multi-scale ensemble,
fold alignment, homology/CV splits, feature engineering, figure generation, and the
GPU pipeline modules (with a mock ESM so no download/GPU is needed).

```bash
pip install -r requirements-dev.txt -r requirements-cpu.txt
pytest tests/ -v                       # full suite
pytest tests/ --cov=. --cov-report=term-missing   # with coverage
```

GitHub Actions (`.github/workflows/test.yml`) runs three jobs on every push/PR:

- **Lint** — `ruff` (critical error rules: syntax + undefined names).
- **Import smoke** — imports every core module to catch breakage early.
- **Tests** — full pytest suite with coverage on a **Python 3.11 + 3.12** matrix.

## Pipeline phases (Colab / Rockfish)

### Running tests

```bash
pip install -r requirements-dev.txt -r requirements-cpu.txt
pytest tests/ -v
```

### Biological utility (Phase 1)

After GPU cross-validation, the notebook runs `colab/biological_utility.py` to report:

- **Segment metrics** — region F1, MDR recall, boundary error
- **Functional enrichment** — recovery of binding sites, PTMs, condensate scaffolds
- **Transition zones** — performance at disorder↔order boundaries

Outputs: `biological_utility_report.json` and `fig5_biological_utility.png`.

### Disorder → function (multi-label IDR roles)

Train with `--profile ultra_fun` (or `--function-head`) to add a multi-label head that predicts DisProt functional groups on disordered residues:

- protein binding · nucleic acid binding · PTM regulation · condensate/assembly · lipid/small-molecule binding

OOF metrics land in `function_prediction_report.json`. Proteome exports use `annotate_idr_functions` / `predict_protein_functions`. Full layer thesis: [`docs/IDR_BIOLOGY_LAYER.md`](docs/IDR_BIOLOGY_LAYER.md).

### AlphaFold hallucination rescue (Phase 2)

After CV, the notebook fetches **AlphaFold DB pLDDT** (AF2 models) for DisProt UniProt accessions and reports:

- **Hallucination rate** — disordered residues where AF assigns high pLDDT (≥70)
- **Rescue rate** — fraction of hallucinations DisorderNet correctly flags (**labeled** definition only; see [`docs/STRUCTURE_DISTRUST_ATLAS.md`](docs/STRUCTURE_DISTRUST_ATLAS.md))
- **Δ AUC** — DisorderNet vs inverse-pLDDT baseline on AF-covered residues

Outputs: `af_rescue_report.json`, `fig6_af_rescue.png`, cached pLDDT in `af_plddt_cache/`.

### AlphaFold 3 on Colab (Phase 2b, optional)

AF3 model weights **must not** be committed to GitHub (license + multi-GB size). The supported workflow:

1. Upload `af3.bin` to Google Drive: `MyDrive/DisorderNet/af3/af3.bin`
2. Place AF3 job outputs under `MyDrive/DisorderNet/af3/outputs/` (or run a small subset on Colab A100)
3. Set `AF3_MODE = "ingest"` in notebook Cell 11

The notebook compares AF2 (AlphaFold DB) vs AF3 hallucination rates and DisorderNet rescue on overlapping proteins.

Outputs: `af3_rescue_report.json`, `af2_af3_comparison.json`, `fig7_af2_af3_comparison.png`, cache in `af3_plddt_cache/`.

### Integrated synthesis (Phase 3)

After AF rescue analysis, the notebook runs `colab/phase3_synthesis.py` to:

- **Fuse** DisorderNet with AF pLDDT (optimal α grid search)
- **Calibrate** AF confidence — downweight pLDDT where DisorderNet predicts disorder
- **Bootstrap 95% CIs** for AUC on AF-covered residues
- **Rank** GPU AUC against published CAID benchmarks
- **Synthesize** cross-phase headline across Phases 0–2

Outputs: `phase3_integrated_report.json`, `fig8_phase3_synthesis.png`.

### Evaluation rigor (Tier 1)

The Colab notebook reports:

- **Matched benchmark tables** — literature reference (Table A) vs our DisProt CV (Table B)
- **CAID-style metrics** — AUC, AP, F1_max, MCC; stratified by IDR fraction, length, organism
- **Per-fold statistics** — paired DisorderNet vs inverse-pLDDT sign tests on AF-covered residues; bootstrap CIs

Outputs: `caid_evaluation_report.json`, `statistical_validation_report.json`.

Before preprint freeze, tick [`docs/METHODS_CHECKLIST.md`](docs/METHODS_CHECKLIST.md).

## Citation

If you use DisorderNet, please cite the relevant benchmark papers and this repository.

## License

MIT — see [LICENSE](LICENSE).
