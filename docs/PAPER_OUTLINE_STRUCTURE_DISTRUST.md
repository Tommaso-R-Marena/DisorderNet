# Paper outline — Structure distrust as the default post-AF layer

## Working title
**DisorderNet: a post-structure distrust layer for intrinsically disordered regions**

## Core claim (one sentence)
After AlphaFold/Boltz, many residue-level structure confidences misrepresent IDRs; DisorderNet provides a proteome-scale, independently evaluable map of where that confidence should be discarded.

## What this paper is / is not

| Is | Is not |
|----|--------|
| Practice-shifting post-AF analysis layer | AlphaFold replacement |
| Computational distrust atlas + benchmarks | Full conformational ensembles / MD |
| Labeled hallucination rescue + matched pLDDT baselines | Proxy DN∩pLDDT “rescue” as science |

## Evidence stack (load-bearing order)

1. **Disorder competitiveness** — CAID3 / DisProt fair protocol (credibility floor)
2. **Labeled hallucination rescue** — DisProt ∩ high pLDDT; DN recovers disordered residues
3. **Matched baseline** — beat inverse-pLDDT on identical residues (`delta_auc_dn_minus_plddt` + per-fold sign/Wilcoxon)
4. **Downstream computational utility** — DN distrust mask precision among high-pLDDT residues vs size-matched pLDDT mask
5. **Proteome atlas resource** — public JSONL/TSV + triage actions
6. **Honesty layer** — training contamination audit + required ablations when risk ≠ low

Supporting (not load-bearing): IDR roles, SLiMs, biophysics cues, Boltz variance proxy, boundary/conditional-disorder flags.

## Suggested figure list

| Fig | Content | Generator / artifact |
|-----|---------|----------------------|
| 1 | Claim schematic: AF fold → distrust mask → IDR-aware reinterpretation | conceptual |
| 2 | Labeled rescue + DN vs inv-pLDDT AUC | `generate_distrust_benchmark_figure` |
| 3 | Downstream mask utility | `generate_downstream_mask_utility_figure` |
| 4 | Proteome atlas burden + top proteins | `generate_distrust_atlas_figure` |
| 5 | Case gallery (structure + DN overlay) | manual / notebook |
| 6 | Contamination risk + ablation Δ | audit JSON + companion CV |

## Methods red lines

- Never publish proxy DN∩high-pLDDT intersection as rescue rate
- Always report `training_contamination.risk_tier`
- Homology CV stats must use the same `split_method` as training
- SequenceMatcher clustering ≠ MMseqs — say so

## Venue target (if numbers land)

Strong methods / genome biology tier if utility + atlas + baselines are crisp; top venue only with undeniable proteome resource + clean low-contamination ablation + fair CAID competitiveness.

See also: `METHODS_CHECKLIST.md`, `STRUCTURE_DISTRUST_ATLAS.md`,
[`rockfish/README.md`](../rockfish/README.md#publish-path-exact-usage)
(`submit_publish_650m.sh` / `submit_publish_3b.sh` → strict `publish_package/` → go/no-go).
