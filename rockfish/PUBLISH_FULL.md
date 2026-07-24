# Full publication campaign (650M → 3B) with auto-resume

## Rockfish walltime (verified against ARCH docs)

| Partition | Max walltime | Notes |
|-----------|--------------|-------|
| **a100 / ica100** | **72 h (3 days)** | Not 48 h. Needs `-A <pi>_gpu --qos=qos_gpu` |
| **shared** (CPU package) | **36 h** | Package job is short (~2 h request) |

Sources: [Partitions](https://docs.arch.jhu.edu/en/latest/1_Clusters/Rockfish/3_Slurm/Partitions.html), [GPU Jobs](https://docs.arch.jhu.edu/en/latest/1_Clusters/Rockfish/3_Slurm/GPU_Jobs.html).

If a GPU job hits `TIMEOUT`, fold progress in `checkpoints/cv_progress.json` is kept. Re-submitting the same `DISORDERNET_WORKDIR` resumes at the next unfinished fold. If CV already finished, `STAGE=pipeline` skips re-training and continues stack/postprocess/eval.

## What the campaign does

1. Submit **650M** ultra (+ clean) → strict `publish_package/`
2. When that package exists, submit **3B** ultra (+ clean) → package
3. Login-node **watchdog** polls (~10 min), and if the queue is empty but packages are incomplete, resubmits from checkpoints (up to 128 times)
4. Emails **marenatommaso@gmail.com** on END / FAIL / TIME_LIMIT (override with `DISORDERNET_MAIL_USER`)
5. tqdm progress in the watchdog log (fold fraction + ETA heuristic)

## CAID3 / CAID4 rigor (leak-free)

Publish jobs default to:

- `RUN_CAID3=1` / `RUN_CAID_CHALLENGE=1` — full challenge suite after pipeline
- `CAID_LEAK_FREE_TRAIN=1` — drop DisProt train proteins that ID/homology-hit CAID3 refs (≥40% SequenceMatcher) **before** CV
- Strict package requires `caid3_eval_report.json` + `caid_leakage_audit.json`

**CAID4** is a blind challenge until ~Dec 2026. If you place organizer targets at
`~/DisorderNet/data/caid4_targets.fasta` (or set `CAID4_TARGETS`), the suite writes
`caid4_submission/disorder/*.caid` + `timings.csv` and efficiency stats. Scoring runs
automatically if/when labels appear in that FASTA.

## Adaptive OOM retries

On `OUT_OF_MEMORY` (or CUDA OOM in logs), the watchdog escalates automatically:

1. half batch  
2. `ica100` + half batch  
3. `ica100` + quarter batch  
4. `ica100` + min batch (`batch=1`, `accum=16`)

No command change required — same `submit_publish_full.sh`.

---

## Fresh shell — exact commands

```bash
ssh rockfish
cd ~/DisorderNet
git checkout master && git pull
source ~/venvs/disordernet/bin/activate
mkdir -p logs ~/disordernet_runs data

# Optional CAID4 blind targets (when you have the organizer FASTA):
#   cp /path/to/caid4_targets.fasta ~/DisorderNet/data/caid4_targets.fasta
#   export CAID4_TARGETS=$HOME/DisorderNet/data/caid4_targets.fasta

export DISORDERNET_MAIL_USER=marenatommaso@gmail.com
export DISORDERNET_RESULTS=$HOME/disordernet_runs
export DISORDERNET_GPU_ACCOUNT=$(sacctmgr -nP show assoc user=$USER format=account,qos \
  | awk -F'|' '/qos_gpu/{print $1; exit}')

bash rockfish/slurm/submit_publish_full.sh
```

**Commands are unchanged** from the previous campaign launch (still one script).
New defaults (CAID leak-free + OOM escalate) activate automatically after `git pull`.

Expected immediately:

```text
Campaign file: /home/…/disordernet_runs/campaign_<stamp>.json
GPU account:   sfried3_gpu  QOS=qos_gpu  mail=marenatommaso@gmail.com
Wrote …/campaign_<stamp>.json
Watchdog PID <n>  log=…/logs/publish_watchdog_<stamp>.out
```

```bash
squeue -u $USER
# expect dn-pub-650m-… on a100 (R or PD) and maybe dn-pkg-650m PD (Dependency)
```

## How to check progress

```bash
CAMPAIGN=$(ls -t ~/disordernet_runs/campaign_*.json | head -1)
python rockfish/publish_campaign.py status --campaign "$CAMPAIGN"
tail -f ~/DisorderNet/logs/publish_watchdog_*.out
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed -P
# fold resume signal:
python -c "import json;print(len(json.load(open('$HOME/disordernet_runs/publish_650m_*/ultra_650M/checkpoints/cv_progress.json'.replace('*','')))['fold_results']))" 2>/dev/null || \
  ls ~/disordernet_runs/publish_650m_*/ultra_650M/checkpoints/cv_progress.json
```

Safer fold count:

```bash
python - <<'PY'
import json,glob
paths=sorted(glob.glob("$HOME/disordernet_runs/publish_650m_*/ultra_650M/checkpoints/cv_progress.json".replace("$HOME",__import__("os").path.expanduser("~"))))
print(paths[-1] if paths else "none")
if paths:
    print("folds", len(json.load(open(paths[-1]))["fold_results"]))
PY
```

## When everything is finished

```bash
CAMPAIGN=$(ls -t ~/disordernet_runs/campaign_*.json | head -1)
python rockfish/publish_campaign.py status --campaign "$CAMPAIGN"
# campaign_status == "done"

ls ~/disordernet_runs/publish_650m_*/publish_package/PACKAGE_README.md
ls ~/disordernet_runs/publish_3b_*/publish_package/PACKAGE_README.md
less ~/disordernet_runs/publish_650m_*/publish_package/PACKAGE_README.md
less ~/disordernet_runs/publish_3b_*/publish_package/comparison.json
```

Then fill `docs/METHODS_CHECKLIST.md` from the packages.

## Walltime / GPU-hour expectations (order of magnitude)

| Phase | Typical GPU wall once scheduled | Notes |
|-------|----------------------------------|-------|
| 650M ultra | ~24–48 h (may span 1 job within 72 h cap) | Fold resume if TIMEOUT |
| 650M clean | ~24–48 h | afterok after main |
| 650M package | minutes–2 h CPU | shared |
| 3B ultra | ~36–72 h+ | may need multiple TIMEOUT resumes; try `ica100` |
| 3B clean + package | similar | |

Total calendar time is dominated by **queue wait** + possible multi-job resumes for 3B.

## Confidence modes (honesty preserved)

| Mode | Meaning |
|------|---------|
| **Selective (conformal)** | decision ∈ {disorder, order, **abstain**} with coverage ≥ 1−α |
| **Forced** | always `y_hat_forced` + `confidence_pct` = 100×max(p,1−p) on calibrated p |

Abstaining less only by “turning down α” would **weaken** coverage — we do **not** do that. Lower abstain rates come from a **better** model (650M/3B LoRA ultra). Forced mode always reports a best guess + % even when conformal abstains.

CLI: `predict_disorder.py` JSON now includes `confidence_pct` and `y_hat_forced`.

## Tests

```bash
pytest tests/test_publish_campaign.py tests/test_confidence.py tests/test_confidence_layer.py -q
```
