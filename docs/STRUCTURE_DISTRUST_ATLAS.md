# Structure Distrust Atlas — paper architecture (1)

## Claim

After Boltz / AlphaFold produce a fold, **DisorderNet is the default post-structure distrust layer**: it flags where structure confidence should not be trusted on intrinsically disordered regions, and prefers an independent disorder map (+ optional roles) instead.

This is the primary purely-computational paradigm-shaped bet for DisorderNet — **not** MD ensembles, **not** an AF replacement.

## Two definitions (do not conflate)

| Name | Inputs | May claim as rescue? | Use |
|------|--------|----------------------|-----|
| **Labeled hallucination / rescue** | Independent DisProt (etc.) labels ∩ high pLDDT; DN predicts disorder | **Yes** | Scientific benchmark / paper figures |
| **Proxy distrust flags** | DN disorder call ∩ high pLDDT | **No** (tautological rescue) | Proteome triage / deployment |

Proxy flags are still useful operationally (“re-interpret these AF regions”). They must never be published as hallucination rescue rates.

## Module map

| Piece | Location |
|-------|----------|
| Labeled benchmark + DN vs inverse-pLDDT | `colab/hallucination_benchmark.py` |
| Proteome atlas + mask utility | `colab/structure_distrust_atlas.py` |
| Screening API (labeled or proxy) | `colab/novel_use_cases.screen_af_hallucinations` |
| Eval / Rockfish export | `rockfish/run_disordernet.py eval` (default on) |
| Layer rollup | `summarize_structure_distrust` in `idr_biology_layer.py` |

## Rockfish outputs (eval)

```bash
python rockfish/run_disordernet.py eval
# optional skip:
python rockfish/run_disordernet.py eval --no-structure-distrust-atlas
```

Artifacts under checkpoint dir:
- `structure_distrust_benchmark.json` — labeled rescue + matched baselines + optional downstream mask utility
- `structure_distrust_atlas_report.json` — proteome summary
- `structure_distrust_atlas.jsonl` / `.tsv` — per-protein rows
- `af_rescue_manifest.json` — pipeline triage (uses labels when available)

## Evidence stack for the paper

1. **Fair disorder competitiveness** (CAID / homology-aware) — credibility floor  
2. **Labeled hallucination rescue** vs independent DisProt (this module)  
3. **Matched baseline**: beat inverse-pLDDT on the same residues (`delta_auc_dn_minus_plddt`)  
4. **Downstream computational utility**: among high-pLDDT residues, DN distrust mask precision vs size-matched pLDDT mask (`estimate_downstream_mask_utility`)  
5. **Proteome atlas** as the public resource  
6. Roles / cues / Boltz variance remain supporting layers, not the load-bearing claim  

## Explicit non-claims

- Proxy DN∩high-pLDDT intersection is not independent rescue  
- Not a conformational ensemble predictor  
- Not an AlphaFold replacement  

## Operator path

To run Rockfish end-to-end and decide publish go/no-go, follow
[`ROCKFISH_PUBLISH_RUNBOOK.md`](ROCKFISH_PUBLISH_RUNBOOK.md).
Fill [`METHODS_CHECKLIST.md`](METHODS_CHECKLIST.md) from mirrored artifacts.  
