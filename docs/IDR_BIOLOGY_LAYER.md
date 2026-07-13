# IDR Biology Layer — DisorderNet product thesis

## Claim

After Boltz / AlphaFold produce a structure, **DisorderNet is the layer that answers what those models cannot by design**:

1. **Where is the chain disordered?** (CAID-credible residue map)
2. **What might those IDRs do?** (roles + sequence / partner / ligand / biophysics cues)
3. **Where is the structure model overconfident?** (hallucination rescue)
4. **Optional cheap ensemble proxy:** Boltz multi-sample pLDDT variance on IDRs
5. **Conditional-disorder cues:** order↔disorder boundaries / folding-upon-binding *flags* (not MD)

This is the path toward becoming a **default post-structure biology layer** — not an AlphaFold replacement, and **not** full MD conformational ensembles.

## Explicit non-goals (frozen)

| Non-goal | Why |
|----------|-----|
| Sequence → full conformational ensembles at AF scale | Needs decade-scale physics+ML + different model class |
| Replace Boltz/AF for folds | Wrong problem; we *complement* them |
| Equal priority on dynamics + IDR function moonshots | Scope death |

## Module map

| Piece | Location |
|-------|----------|
| Layer compose + proteome export | `colab/idr_biology_layer.py` |
| Pred / partner / ligand / function I/O | `colab/idr_layer_io.py` |
| Biophysics patterning cues (SCD/κ/entropy) | `colab/idr_layer_biophysics.py` |
| Cache, landscape, markdown/HTML/GFF/cards, compare, schema | `colab/idr_layer_ops.py` |
| Disorder → function head / labels / calibration | `colab/function_predict.py` + `ultra_fun` |
| Boltz pLDDT + multi-sample variance | `colab/boltz_plddt.py` |
| Rockfish stage | `python rockfish/run_disordernet.py idr-layer` |

## Phased roadmap

### Phase A–C (shipped through v1.6)
- Unified export bundle: JSON, Markdown, HTML, GFF3, JSONL[.gz], triage, BED, bedGraph, roles TSV, CAID, per-role bedGraphs, triage cards, run manifest
- Partner / ligand / sequence / biophysics cues with transparent `conditioned_prob` + role confidence
- Expanded SLiM library (NLS/NES/KEN/D-box/SH3/WW/TRAF/PDZ)
- Role validation vs DisProt; structure-distrust aggregate; proteome landscape
- OOF-tuned function threshold + per-group temperature calibration
- Quality / quarantine flags; multi-role conflict review flags
- Per-protein cache; proteome resume; schema validation
- Function-pred reload (`--idr-function-preds-dir`); triage filters
- Richer JSONL compare (role gains/losses + quality deltas)

### Phase D — optional biophysics collaborations
- Small-system MD / SAXS / NMR benchmarks validating variance proxy

## Rockfish usage

```bash
python rockfish/run_disordernet.py idr-layer \
  --structure-backend boltz --boltz-mode ingest \
  --idr-partners partners.json --idr-ligands ligands.json \
  --idr-auto-threshold --idr-calibrate-function \
  --idr-cache --idr-workers 8 --idr-cards-top-n 30

# Overlay standalone disorder + function prediction dirs:
python rockfish/run_disordernet.py idr-layer \
  --idr-preds-dir predictions/ --idr-function-preds-dir function_preds/
```

Outputs:
- `idr_biology_layer_report.json` / `.md` / `.html`
- `idr_biology_layer.jsonl[.gz]`, `_triage.tsv`, `.bed`, `.gff3`, `_disorder.bedgraph`, `_roles.tsv`
- `idr_biology_layer_role_bedgraphs/`, `idr_biology_layer_cards/`, `idr_biology_layer_caid/`
- `idr_biology_layer_manifest.json` — CLI recipe for reproducibility
