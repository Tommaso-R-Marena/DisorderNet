# Methods checklist — structure distrust paper

Tick before freezing a preprint. Each item points at a concrete artifact.

## Credibility floor
- [ ] `caid3_eval_report.json` (or stratified CAID) attached into distrust benchmark via `attach_caid3_credibility_floor`
- [ ] Fair protocol / homology note documented (`split_method`, see `HOMOLOGY_HOLDOUT.md`)

## Labeled distrust (scientific)
- [ ] `structure_distrust_benchmark.json` with `definition: labeled_independent`
- [ ] Rescue rates come only from DisProt (or other independent) labels
- [ ] Matched inv-pLDDT baseline: `matched_baselines.delta_auc_dn_minus_plddt`
- [ ] Per-fold paired stats: `statistical_validation_report.json` with correct homology splits
- [ ] Downstream mask utility: `downstream_mask_utility.enabled == true`

## Honesty / contamination
- [ ] `training_contamination` block present on benchmark
- [ ] If `risk_tier` is medium/high: companion ablation with `use_hallucination_weighting=False` and `use_plddt_features=False`
- [ ] Proxy screening never labeled as rescue in figures/text

## Atlas resource
- [ ] `structure_distrust_atlas_report.json`
- [ ] `structure_distrust_atlas.jsonl` + `.tsv`
- [ ] Non-claims list includes `proxy_flags_are_not_independent_rescue`
- [ ] Optional figures from `generate_distrust_*_figure`

## Reproducibility
- [ ] Rockfish `eval` (or `structure-distrust-atlas`) command recorded in run manifest
- [ ] Structure backend named (`boltz2` / `af2` / `af3`)
- [ ] Thresholds frozen (`disorder_threshold`, `high_plddt_threshold`)

## Explicit non-claims (must appear in paper)
- [ ] Not an AF replacement
- [ ] Not MD / ensemble prediction
- [ ] Proxy flags ≠ independent rescue

## Operator runbook
- [ ] Follow [`ROCKFISH_PUBLISH_RUNBOOK.md`](ROCKFISH_PUBLISH_RUNBOOK.md) for main + clean companion
- [ ] Publish go/no-go decided from runbook §6 thresholds framing (not calendar)