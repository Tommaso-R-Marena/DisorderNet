# Rockfish publish runbook — structure distrust paper

Frozen checklist → commands path. After this works, remaining work is
**submit jobs → judge numbers → go/no-go**, not more feature scaffolding.

Branch: `feature/idr-biology-layer-c41e` (use until merged to `master`).
Related: [`METHODS_CHECKLIST.md`](METHODS_CHECKLIST.md),
[`STRUCTURE_DISTRUST_ATLAS.md`](STRUCTURE_DISTRUST_ATLAS.md),
[`PAPER_OUTLINE_STRUCTURE_DISTRUST.md`](PAPER_OUTLINE_STRUCTURE_DISTRUST.md),
[`rockfish/README.md`](../rockfish/README.md).

---

## 1. Checkout + env

```bash
git clone https://github.com/Tommaso-R-Marena/DisorderNet.git ~/DisorderNet
cd ~/DisorderNet
git fetch origin feature/idr-biology-layer-c41e
git checkout feature/idr-biology-layer-c41e

bash rockfish/setup_env.sh
source ~/venvs/disordernet/bin/activate
mkdir -p logs

export DISORDERNET_ACCOUNT=your_pi_gpu   # required
export DISORDERNET_BOLTZ_ROOT=$HOME/boltz
export BOLTZ_CACHE=$DISORDERNET_BOLTZ_ROOT/cache
```

Optional: pre-warm Boltz weights with `rockfish/slurm/boltz_batch.sbatch`
(`BOLTZ_MODE=auto`) before the main pipeline.

---

## 2. Main run (contaminated / production ultra)

Boltz ingest + full pipeline + CAID3. Evaluation writes the labeled distrust
benchmark; CAID3 then **patches** `structure_distrust_benchmark.json` in place
via `finalize_distrust_benchmark_with_caid3`.

```bash
export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_main
export RUN_CAID3=1
export BOLTZ_MODE=ingest   # or auto if structures not yet available
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR,RUN_CAID3,BOLTZ_MODE \
  rockfish/slurm/pipeline_ultra.sbatch
```

Artifacts land under `$DISORDERNET_WORKDIR/checkpoints/` and are mirrored to
`$DISORDERNET_RESULTS/<run_tag>/`.

---

## 3. Clean companion (required for publish honesty)

Same capacity as ultra, but `use_hallucination_weighting=False` and
`use_plddt_features=False`. **Must use a separate workdir** so it cannot
clobber the main run.

```bash
export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_clean
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR \
  rockfish/slurm/pipeline_ultra_clean.sbatch
```

`pipeline_ultra_clean.sbatch` sets `PROFILE=ultra_clean`,
`CHECKPOINT_SUBDIR=checkpoints_ultra_clean`, and
`RUN_NO_HALLUC_WEIGHT=1` / `RUN_NO_PLDDT_FEATURES=1`.

---

## 4. Verify mirrored artifact list

After both jobs finish, confirm these exist (main workdir or mirror dest):

| Artifact | Purpose |
|----------|---------|
| `structure_distrust_benchmark.json` | Labeled rescue + matched baselines + CAID floor + contamination |
| `structure_distrust_atlas_report.json` | Proteome summary |
| `structure_distrust_atlas.jsonl` / `.tsv` | Per-protein rows |
| `distrust_figures/**` | Paper figures (if matplotlib available) |
| `caid3_eval_report.json` | Credibility floor source |
| `af_rescue_manifest.json` | Pipeline triage |
| `statistical_validation_report.json` | Paired / fold stats |
| `idr_biology_layer_*` | Supporting layer exports |
| `function_prediction_report.json` | If function head enabled |
| `run_manifest.json` / `cv_summary.json` | Reproducibility |

Check CAID attachment:

```bash
python -c "
import json
b=json.load(open('structure_distrust_benchmark.json'))
print(b.get('caid3_credibility_floor'))
print(b.get('training_contamination',{}).get('risk_tier'))
"
```

Expect `caid3_credibility_floor.available == true` on the main run after CAID3.

---

## 5. Fill methods checklist

Open [`METHODS_CHECKLIST.md`](METHODS_CHECKLIST.md) and tick every box from the
files above. Do not invent metrics that are not in the JSON.

Build a **main vs clean** comparison table (ΔAUC vs inv-pLDDT, labeled rescue
rate, mask utility, contamination risk_tier). Keep both runs' artifact paths in
the lab notebook / supplement.

---

## 6. Publish go / no-go criteria

All of the following must hold on the **main** run, with the **clean** run
confirming the claim is not entirely an artifact of structure training:

1. **Matched distrust** — clean and/or main: DisorderNet beats inv-pLDDT on
   matched residues (`matched_baselines.delta_auc_dn_minus_plddt` > 0) with
   supportive paired/fold stats in `statistical_validation_report.json`.
2. **Labeled rescue** — meaningful `rescue_rate` under
   `definition: labeled_independent` (not proxy).
3. **Mask utility** — DN distrust mask ≥ size-matched pLDDT mask
   (`downstream_mask_utility`).
4. **CAID floor** — attached (`available: true`) and competitive enough that
   disorder credibility is not undermined (do not publish distrust-only if
   CAID collapses).
5. **Contamination honesty** — `training_contamination` present; main vs clean
   table exists; if clean collapses the ΔAUC claim, do **not** frame rescue as
   a breakthrough.

### If criteria fail

- Do **not** publish “rescue-as-breakthrough.”
- Fall back to a disorder + atlas triage / methods paper, **or** iterate
  training/hparams.
- Do **not** add more feature scaffolding to paper over a failed claim.

---

## 7. One-line operator path

```text
checkout PR branch → setup_env → sbatch pipeline_ultra → sbatch pipeline_ultra_clean
  → verify mirrored artifacts → METHODS_CHECKLIST → go/no-go on numbers
```
