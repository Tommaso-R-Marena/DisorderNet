# IDR Biology Layer — DisorderNet product thesis

## Claim

After Boltz / AlphaFold produce a structure, **DisorderNet is the layer that answers what those models cannot by design**:

1. **Where is the chain disordered?** (CAID-credible residue map)
2. **What might those IDRs do?** (binding / PTM / condensate / lipid roles)
3. **Where is the structure model overconfident?** (hallucination rescue)
4. **Optional cheap ensemble proxy:** Boltz multi-sample pLDDT variance on IDRs

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
| Disorder → function head / labels | `colab/function_predict.py` + `ultra_fun` |
| Boltz pLDDT + multi-sample variance | `colab/boltz_plddt.py` |
| Hallucination screening | `colab/novel_use_cases.py` / `af_hallucination.py` |
| Rockfish stage | `python rockfish/run_disordernet.py idr-layer` |

## Phased roadmap

### Phase A (this PR) — shipping layer
- Unified per-protein / proteome JSON + JSONL export
- Boltz variance as optional ensemble *proxy*
- Compose disorder + optional function probs + structure confidence

### Phase B — make the layer trusted
- Train `ultra_fun` on Rockfish; publish OOF function metrics
- CAID3 + homology-safe disorder numbers
- Boltz-default rescue rates on DisProt

### Phase C — conditional IDR state (still not MD)
- Partner / motif / ligand-conditioned role scores
- Folding-upon-binding style flags where DisProt transition terms allow

### Phase D — optional biophysics collaborations
- Small-system MD / SAXS / NMR benchmarks validating variance proxy
- Not required for the core product

## Rockfish usage

```bash
# After CV (or predict) has disorder probs in fold results / predictions:
python rockfish/run_disordernet.py idr-layer \
  --structure-backend boltz --boltz-mode ingest

# Training with function head (feeds role calls into the layer):
python rockfish/run_disordernet.py cv --profile ultra_fun
```

Outputs:
- `checkpoints/idr_biology_layer_report.json`
- `checkpoints/idr_biology_layer.jsonl`
