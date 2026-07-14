# DisorderNet: Beating AlphaFold 3 at Intrinsic Disorder Prediction

[![Tests](https://github.com/Tommaso-R-Marena/DisorderNet/actions/workflows/test.yml/badge.svg)](https://github.com/Tommaso-R-Marena/DisorderNet/actions/workflows/test.yml)
[![Open In Colab — Full GPU CV](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_Pro.ipynb)
[![Open In Colab — Quick Screen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_QuickScreen.ipynb)

## Overview

**DisorderNet** is a protein language model-enhanced ensemble for predicting intrinsically disordered regions (IDRs) in proteins. On our DisProt 5-fold CV, the CPU model (v6) reaches **0.831 AUC-ROC**; the GPU Colab path (ESM-2 650M + LoRA) targets **≥0.88** pending a full benchmark run.

Compared to **literature reference points** (different protocols — not head-to-head), dedicated disorder predictors substantially outperform using AlphaFold pLDDT as a disorder proxy: AF3-pLDDT scores **0.747** on CAID3 (rank 13), while current disorder SOTA (ESMDisPred) reaches **0.895**. DisorderNet's distinctive contribution is the **post-structure IDR biology layer**: quantifying AlphaFold/Boltz hallucinations in IDRs and (with `ultra_fun`) assigning functional roles those structure models cannot represent.

AlphaFold 3's diffusion architecture hallucinates structure in genuinely disordered regions — [22% of residues are hallucinations](https://arxiv.org/abs/2510.15939). AF3-pLDDT [ranks 13th on CAID3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/), *worse* than AF2 (rank 11th). DisorderNet exploits this fundamental weakness.

## Results

### Comprehensive Benchmark

**Important:** Table A lists published reference AUCs from CAID/DisProt studies (different splits/protocols). Table B lists **our** runs on the same in-repo DisProt 5-fold protein-grouped CV. Do not treat Table A rows as head-to-head comparisons.

#### Table A — Literature reference (not head-to-head)

| Method | AUC-ROC | Source | Protocol |
|--------|---------|--------|----------|
| AF3-pLDDT (CAID3, rank 13) | 0.747 | [CAID3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/) | CAID3 eval set |
| AF2-pLDDT (CAID3, rank 11) | 0.770 | [CAID3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/) | CAID3 eval set |
| IUPred3 | 0.789 | [CAID](https://caid.idpcentral.org/) | CAID benchmark |
| flDPnn | 0.814 | [CAID](https://caid.idpcentral.org/) | CAID benchmark |
| SETH (ProtT5+CNN) | 0.830 | [Ilzhöfer et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9580958/) | Published |
| **DisorderNet v6 (CPU)** | **0.831** | **This repo** | DisProt 5-fold CV |
| ESM2_650M-LoRA | 0.880 | [LoRA-DR](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i439/8199360) | CAID1 |
| flDPnn3a (CAID3) | 0.871 | [CAID3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/) | CAID3 eval set |
| ESMDisPred (CAID3 SOTA) | 0.895 | [Kabir et al.](https://pubmed.ncbi.nlm.nih.gov/41648466/) | CAID3 eval set |

#### Table B — Our DisProt CV (directly comparable within table)

| Method | AUC-ROC | AP | Status |
|--------|---------|-----|--------|
| DisorderNet v6 (ESM-2 8M + GBDT) | 0.831 | 0.537 | Verified (`results_v6/metrics.json`) |
| DisorderNet GPU (ESM-2 650M + LoRA) | 0.817 | — | Verified Colab A100 (`disordernet_gpu_results_*.json`) |

#### Legacy combined view (reference only)

| Method | AUC-ROC | Δ vs AF3 (ref.) | Source |
|--------|---------|-----------------|--------|
| AF3-pLDDT (CAID3, rank 13) | 0.747 | baseline | [CAID3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/) |
| AF2-pLDDT (CAID3, rank 11) | 0.770 | +3.1% | [CAID3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/) |
| IUPred3 | 0.789 | +5.6% | [CAID](https://caid.idpcentral.org/) |
| DisorderNet v4 (physics only) | 0.794 | +6.3% | This work |
| flDPnn (CAID1/2 best) | 0.814 | +9.0% | [CAID](https://caid.idpcentral.org/) |
| DisorderNet v5 (ESM 8M, PCA-32) | 0.823 | +10.2% | This work |
| SETH (ProtT5+CNN) | 0.830 | +11.1% | [Ilzhöfer et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9580958/) |
| **DisorderNet v6 (ESM 8M, PCA-48)** | **0.831** | **+11.3%** | **This work** |
| flDPnn3a (CAID3) | 0.871 | +16.6% | [CAID3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/) |
| ESM2_35M-LoRA | 0.868 | +16.2% | [LoRA-DR](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i439/8199360) |
| ESM2_650M-LoRA | 0.880 | +17.8% | [LoRA-DR](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i439/8199360) |
| ESMDisPred (CAID3 SOTA) | 0.895 | +19.8% | [Kabir et al.](https://pubmed.ncbi.nlm.nih.gov/41648466/) |

### Version Progression

| Version | AUC | Features | Key Addition |
|---------|-----|----------|-------------|
| v4 | 0.794 | 118 | Multi-scale physicochemical features |
| v5 | 0.823 | 214 | + ESM-2 8M embeddings (PCA-32) |
| v6 | 0.831 | 406 | + PCA-48, ESM variance/context features |
| GPU (Colab) | 0.817 (0.831 AF-fusion subset) | 1280+phys | ESM-2 650M + LoRA + segment-aware ES + v6 ensemble |
| GPU SOTA track (`sota` profile) | target ≥0.88–0.90 | — | Transformer head, Dice+EMA, 3-way stack, compact ckpt |
| GPU ULTRA track (`ultra` profile) | target 0.88–0.92 | — | Rich features, FFN LoRA, v6-pro meta-stack, MC-dropout TTA |
| GPU ULTRA 3B (`ultra3b` profile) | target 0.90–0.93 | — | ESM-2 3B backbone on A100 40GB+ |
| GPU ULTRA + function (`ultra_fun`) | disorder + IDR roles | — | Multi-label Disorder→function head |

### Performance ceiling (honest)

On **DisProt 5-fold CV** with ESM-2 650M, the realistic band is:

| Stage | Typical pooled AUC |
|-------|-------------------|
| Verified GPU baseline | 0.817 |
| + ultra training + 7b–7d stack | 0.88–0.92 (target) |
| + ESM-2 3B (`ultra3b`) + full stack | 0.90–0.93 (target) |
| + multi-seed blend (2–3 seeds) | +0.005–0.015 |
| ESMDisPred (CAID3, different protocol) | 0.895 reference |

Breaking **0.90+ consistently** on DisProt likely needs **ESM-2 3B** (`ultra3b`) or **CAID3-homologous training** — not more post-hoc stacking on 650M alone. Use the [Quick Screen notebook](colab/DisorderNet_Colab_QuickScreen.ipynb) before a full ultra run.

### Backbone upgrade — what to do

| Step | Action | Notebook | Settings | ~Time (A100) |
|------|--------|----------|----------|--------------|
| 1 | Go/no-go | [Quick Screen](colab/DisorderNet_Colab_QuickScreen.ipynb) | `SCREEN_MODE="standard"`, `SCREEN_BACKBONE="650M"` | 2–3 h |
| 2 | Full 650M ultra (if screen ≥ MODERATE) | [Colab Pro](colab/DisorderNet_Colab_Pro.ipynb) | `QUALITY_PROFILE="ultra"` | 18–24 h |
| 3 | **3B paradigm test** | Quick Screen | `SCREEN_BACKBONE="3B"`, `SCREEN_MODE="paradigm"` | 8–12 h |
| 4 | **Full 3B production** | Colab Pro | `QUALITY_PROFILE="ultra3b"`, `ESM_BACKBONE="3B"` | 30–40 h |
| 5 | Multi-seed (optional) | Colab Pro Cell 7e | seeds 42 + 43 | 2× step 4 |

**Colab for 3B:** Runtime → **A100 40GB** + **High RAM**. Run `!pip install -q lightgbm xgboost`. **Do not use T4** for 3B.

**Decision rule:** If step 2 stacked AUC **< 0.87**, skip another 650M run and do step 3. If step 3 stacked AUC **≥ 0.86**, commit to step 4.

### Rockfish / Slurm (recommended for production)

If you have access to JHU Rockfish (or any Slurm cluster with A100s), use the HPC pipeline instead of Colab for 3B runs and multi-day jobs. **Full usage instructions** (setup, publish path, artifacts, go/no-go, env vars, Boltz/AF3) live in **[rockfish/README.md](rockfish/README.md)**.

Until `feature/idr-biology-layer-c41e` is merged, checkout that branch on Rockfish:

```bash
git clone https://github.com/Tommaso-R-Marena/DisorderNet.git ~/DisorderNet
cd ~/DisorderNet
git fetch origin feature/idr-biology-layer-c41e && git checkout feature/idr-biology-layer-c41e
bash rockfish/setup_env.sh && source ~/venvs/disordernet/bin/activate
mkdir -p logs
export DISORDERNET_ACCOUNT=your_pi_gpu
export DISORDERNET_BOLTZ_ROOT=$HOME/boltz
export BOLTZ_CACHE=$DISORDERNET_BOLTZ_ROOT/cache

# Main publish run (ultra + CAID3 + distrust benchmark finalize)
export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_main
export RUN_CAID3=1
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR,RUN_CAID3 \
  rockfish/slurm/pipeline_ultra.sbatch

# Contamination-clean companion (separate workdir — required)
export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_clean
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR \
  rockfish/slurm/pipeline_ultra_clean.sbatch
```

Operator path: checkout → setup → `pipeline_ultra` → `pipeline_ultra_clean` → verify mirrored artifacts → [`docs/METHODS_CHECKLIST.md`](docs/METHODS_CHECKLIST.md) → publish go/no-go (criteria in [rockfish/README.md](rockfish/README.md#5-publish-go--no-go)).

Ultra on Rockfish uses **homology-safe CV**, optional **train-time pLDDT** (disabled in `ultra_clean`), and **CAID3** scoring for fair comparison vs ESMDisPred (0.895).

### SOTA track (`QUALITY_PROFILE = "sota"`)

Designed to close the gap to ESMDisPred (0.895 CAID3 reference):

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

### Option 1a: Quick paradigm screen (~2–3 hours, recommended first)

[![Open Quick Screen in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_QuickScreen.ipynb)

Run this **before** the full 18–24h CV to get a breakthrough go/no-go verdict on the current paradigm (ESM-2 650M + LoRA + v6 ensemble).

1. Open [`colab/DisorderNet_Colab_QuickScreen.ipynb`](colab/DisorderNet_Colab_QuickScreen.ipynb) in Colab (badge above)
2. Select **Runtime → Change runtime type → GPU (A100 or L4) + High RAM**
3. Set `SCREEN_MODE = "standard"` (or `"flash"` for ~1h, `"paradigm"` for mini-ultra)
4. Run all cells — outputs `quick_screen_report.json` with tier **HIGH / MODERATE / LOW / STOP**
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
pip install numpy scikit-learn lightgbm xgboost fair-esm torch requests

# Run the full pipeline
python fetch_disprot.py          # Download DisProt data
python extract_esm_embeddings.py  # Extract ESM-2 8M embeddings
python run_v6_mem.py              # Train and evaluate
python generate_figures_v6.py     # Generate figures
```

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

| File | Description |
|------|-------------|
| `colab/DisorderNet_Colab_QuickScreen.ipynb` | **Quick breakthrough screen** (~2–3h go/no-go before full CV) — [Open in Colab](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_QuickScreen.ipynb) |
| `colab/DisorderNet_Colab_Pro.ipynb` | Full GPU notebook (ESM-2 650M + LoRA) — [Open in Colab](https://colab.research.google.com/github/Tommaso-R-Marena/DisorderNet/blob/master/colab/DisorderNet_Colab_Pro.ipynb) |
| `colab/quick_screen.py` | Quick screen logic (stratified subsample, verdict tiers) |
| `colab/esm_backbone.py` | ESM-2 backbone registry (650M → 3B) + VRAM batch presets |
| `rockfish/run_disordernet.py` | HPC CLI: screen / cv / stack / postprocess / full / pipeline / eval / atlas |
| `rockfish/slurm/pipeline_ultra.sbatch` | Full production + eval + CAID3 |
| `rockfish/slurm/pipeline_ultra_clean.sbatch` | Contamination-clean companion (separate workdir) |
| `rockfish/slurm/multi_seed.sbatch` | Slurm array for seeds 42/43/44 |
| `rockfish/README.md` | **Canonical Rockfish usage** (publish path, artifacts, go/no-go) |
| `docs/ROCKFISH_PUBLISH_RUNBOOK.md` | Short pointer to rockfish README publish path |
| `docs/METHODS_CHECKLIST.md` | Preprint freeze checklist |
| `colab/homology_splits.py` | CAID-credible homology-clustered CV |
| `colab/caid3_eval.py` | CAID3 Disorder-PDB benchmark harness |
| `colab/structure_encoder.py` | Train-time pLDDT feature channel |
| `colab/predict_batch.py` | FASTA proteome inference + `.caid` export |
| `colab/novel_use_cases.py` | AF hallucination screening, rescue manifest, IDR function annotation |
| `colab/function_predict.py` | Disorder→function multi-label head, labels, OOF metrics |
| `colab/inference_tta.py` | MC-dropout test-time augmentation (ultra Cell 7d) |
| `colab/multi_seed_blend.py` | Optional multi-seed OOF average (Cell 7e) |
| `colab/disordernet_gpu.py` | Colab training module (data, model, CV loop) |
| `colab/cv_splits.py` | Shared deterministic GroupKFold splits + fingerprints |
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

### Running tests

```bash
pip install -r requirements-dev.txt
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

OOF metrics land in `function_prediction_report.json`. Proteome exports use `annotate_idr_functions` / `predict_protein_functions`.

### AlphaFold hallucination rescue (Phase 2)

After CV, the notebook fetches **AlphaFold DB pLDDT** (AF2 models) for DisProt UniProt accessions and reports:

- **Hallucination rate** — disordered residues where AF assigns high pLDDT (≥70)
- **Rescue rate** — fraction of hallucinations DisorderNet correctly flags
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

| `run_v6_mem.py` | CPU version with ESM-2 8M + GBDT ensemble |
| `run_v5_esm.py` | v5 with PCA-32 ESM features |
| `extract_esm_embeddings.py` | ESM-2 embedding extraction |
| `fetch_disprot.py` | DisProt database downloader |
| `generate_figures_v6.py` | Publication figure generator |
| `results_v6/` | v6 metrics, predictions, figures |

## Citation

If you use DisorderNet, please cite the relevant benchmark papers and this repository.

## License

MIT
