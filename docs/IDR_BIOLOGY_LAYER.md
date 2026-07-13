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
| Pred reload + partner / ligand I/O | `colab/idr_layer_io.py` |
| Disorder → function head / labels | `colab/function_predict.py` + `ultra_fun` |
| Boltz pLDDT + multi-sample variance | `colab/boltz_plddt.py` |
| Hallucination screening | `colab/novel_use_cases.py` / `af_hallucination.py` |
| Rockfish stage | `python rockfish/run_disordernet.py idr-layer` |

## Phased roadmap

### Phase A–B (shipped)
- Unified JSON / JSONL / BED / bedGraph / triage / role-track exports
- Boltz variance proxy + cached / overlapped loads
- Full-sequence function OOF + DisProt role validation
- Structure-distrust aggregate in the layer report
- Function OOF metrics embedded when available

### Phase C (in progress) — conditional IDR state (still not MD)
- Sequence cues; partner cues; **ligand cues** (`--idr-ligands`)
- Role ∩ hallucination intersections; triage ranking
- Boundary / folding-upon-binding cues
- Threshold + worker CLI knobs; optional gzip JSONL
- Multi-condition score stacking (partner + ligand) with transparent `conditioned_prob`

### Phase D — optional biophysics collaborations
- Small-system MD / SAXS / NMR benchmarks validating variance proxy

## Rockfish usage

```bash
python rockfish/run_disordernet.py idr-layer \
  --structure-backend boltz --boltz-mode ingest \
  --idr-partners partners.json --idr-ligands ligands.json \
  --idr-workers 8

python rockfish/run_disordernet.py predict --fasta query.fa --export-idr-layer \
  --idr-gzip
```

Outputs:
- `idr_biology_layer_report.json` — summary, triage, role validation, structure distrust
- `idr_biology_layer.jsonl[.gz]` — full per-protein records
- `idr_biology_layer_triage.tsv` / `.bed` / `_disorder.bedgraph` / `_roles.tsv`
