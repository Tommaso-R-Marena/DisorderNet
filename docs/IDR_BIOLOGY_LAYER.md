# IDR Biology Layer — DisorderNet product thesis

## Claim

After Boltz / AlphaFold produce a structure, **DisorderNet is the layer that answers what those models cannot by design**:

1. **Where is the chain disordered?** (CAID-credible residue map)
2. **What might those IDRs do?** (roles + sequence / partner / ligand cues)
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
| Pred / partner / ligand I/O | `colab/idr_layer_io.py` |
| Cache, landscape, markdown/HTML, compare, schema | `colab/idr_layer_ops.py` |
| Disorder → function head / labels / calibration | `colab/function_predict.py` + `ultra_fun` |
| Boltz pLDDT + multi-sample variance | `colab/boltz_plddt.py` |
| Rockfish stage | `python rockfish/run_disordernet.py idr-layer` |

## Phased roadmap

### Phase A–C (shipped through v1.5)
- Unified export bundle: JSON, Markdown, HTML, JSONL[.gz], triage, BED, bedGraph, roles TSV, CAID dir, per-role bedGraphs
- Partner / ligand / sequence cues with transparent `conditioned_prob`
- Role validation vs DisProt; structure-distrust aggregate; proteome landscape
- OOF-tuned function threshold (`--idr-auto-threshold`)
- Per-group OOF temperature calibration (`--idr-calibrate-function`)
- Quality / quarantine flags on every protein record
- Per-protein record cache (`--idr-cache` / `--idr-cache-dir`)
- Resume large proteomes (`--idr-resume`)
- Schema validation + JSONL compare (`--idr-compare`)
- Threaded proteome builds + threshold / worker CLI knobs

### Phase D — optional biophysics collaborations
- Small-system MD / SAXS / NMR benchmarks validating variance proxy

## Rockfish usage

```bash
python rockfish/run_disordernet.py idr-layer \
  --structure-backend boltz --boltz-mode ingest \
  --idr-partners partners.json --idr-ligands ligands.json \
  --idr-auto-threshold --idr-calibrate-function \
  --idr-cache --idr-workers 8

# Resume a partial proteome export:
python rockfish/run_disordernet.py idr-layer \
  --idr-resume checkpoints/idr_biology_layer.jsonl.gz

# Diff against a previous export:
python rockfish/run_disordernet.py idr-layer \
  --idr-compare checkpoints_prev/idr_biology_layer.jsonl.gz
```

Outputs:
- `idr_biology_layer_report.json` / `.md` / `.html`
- `idr_biology_layer.jsonl[.gz]`, `_triage.tsv`, `.bed`, `_disorder.bedgraph`, `_roles.tsv`
- `idr_biology_layer_role_bedgraphs/` — per-role probability tracks
- `idr_biology_layer_caid/` — CAID-format disorder predictions
- `idr_biology_layer_compare.json` — when `--idr-compare` is set
