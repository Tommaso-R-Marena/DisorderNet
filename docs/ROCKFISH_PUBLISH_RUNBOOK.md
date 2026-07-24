# Rockfish publish runbook

**From scratch (clone → env → submit → package → checklist):**  
see **[root README — Path C ops guide](../README.md#path-c--rockfish-ops-guide-what-you-know-what-to-run-when-youre-done)** and  
**[rockfish/README.md — From scratch on Rockfish](../rockfish/README.md#from-scratch-on-rockfish-start-here)**.

Canonical usage (two scripts + CLI) is in
**[rockfish/README.md — Publish path (exact usage)](../rockfish/README.md#publish-path-exact-usage)**.

```bash
# Discover GPU account (a100 needs qos_gpu on the GPU account, not sfried3)
export DISORDERNET_ACCOUNT=sfried3
export DISORDERNET_GPU_ACCOUNT=$(sacctmgr -nP show assoc user=$USER format=account,qos \
  | awk -F'|' '/qos_gpu/{print $1; exit}')
export DISORDERNET_GPU_QOS=qos_gpu
export DISORDERNET_MAIL_USER=marenatommaso@gmail.com

# Preferred: full 650M→3B campaign with TIMEOUT auto-resume (see rockfish/PUBLISH_FULL.md)
bash rockfish/slurm/submit_publish_full.sh
# optional: --partition-3b ica100

# Or one-shot bundles:
bash rockfish/slurm/submit_publish_650m.sh \
  --account "$DISORDERNET_GPU_ACCOUNT" --qos qos_gpu
bash rockfish/slurm/submit_publish_3b.sh \
  --account "$DISORDERNET_GPU_ACCOUNT" --qos qos_gpu
```

**v8 first (cheaper):** `bash rockfish/slurm/submit_v8.sh` (always sets `--qos=qos_gpu`).

**Rockfish limits:** GPU a100 = **72 h** max (not 48 h); shared ≈ 36 h.

**Done when:** campaign status `done`, or `squeue` empty + both `publish_*/publish_package/PACKAGE_README.md` exist.

Related: [`METHODS_CHECKLIST.md`](METHODS_CHECKLIST.md),
[`STRUCTURE_DISTRUST_ATLAS.md`](STRUCTURE_DISTRUST_ATLAS.md),
[`../rockfish/V8_MULTISCALE.md`](../rockfish/V8_MULTISCALE.md) (cheaper v8 path first).

```bash
# Re-package (strict by default)
python rockfish/publish_submit.py package \
  --root-workdir ~/disordernet_runs/publish_650m_<stamp> \
  --kind 650m --strict
```

```text
clone → setup_env → prefetch_esm → discover GPU account
  → (optional) v8 embed+pipeline
  → submit_publish_650m.sh --account $GPU --qos qos_gpu
  → squeue / sacct → open publish_package/ → METHODS_CHECKLIST → go/no-go
```
