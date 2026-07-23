#!/usr/bin/env bash
# Submit the v8 Rockfish pair: GPU embedding extract → CPU pipeline.
#
# Fixes the common failure mode where --qos="$DISORDERNET_GPU_QOS" expands to
# empty (Invalid qos specification) because DISORDERNET_GPU_QOS was never set.
#
# Usage (login node, from repo root):
#   bash rockfish/slurm/submit_v8.sh
#   bash rockfish/slurm/submit_v8.sh --dry-run
#
# Env overrides:
#   DISORDERNET_V8_DIR, DISORDERNET_GPU_ACCOUNT, DISORDERNET_GPU_QOS,
#   DISORDERNET_ACCOUNT (CPU / shared), DISORDERNET_BACKBONES, RUN_HOMOLOGY

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

CPU_ACCOUNT="${DISORDERNET_ACCOUNT:-sfried3}"
GPU_QOS="${DISORDERNET_GPU_QOS:-qos_gpu}"
# Guard against accidental empty exports (export VAR= with no value).
if [[ -z "${GPU_QOS}" ]]; then
  GPU_QOS="qos_gpu"
fi

if [[ -n "${DISORDERNET_GPU_ACCOUNT:-}" ]]; then
  GPU_ACCOUNT="$DISORDERNET_GPU_ACCOUNT"
else
  GPU_ACCOUNT="$(sacctmgr -nP show assoc user="$USER" format=account,qos \
    | awk -F'|' '/qos_gpu/{print $1; exit}')"
fi

if [[ -z "${GPU_ACCOUNT}" ]]; then
  echo "ERROR: no GPU account with qos_gpu found for user=$USER" >&2
  echo "Ask your PI / ARCH for a *_gpu account, or export DISORDERNET_GPU_ACCOUNT=…" >&2
  exit 1
fi

export DISORDERNET_V8_DIR="${DISORDERNET_V8_DIR:-$HOME/scr4_sfried3/disordernet_v8}"

echo "CPU account : $CPU_ACCOUNT"
echo "GPU account : $GPU_ACCOUNT"
echo "GPU QOS     : $GPU_QOS"
echo "V8 workdir  : $DISORDERNET_V8_DIR"

EMBED_CMD=(sbatch --parsable
  -A "$GPU_ACCOUNT" --qos="$GPU_QOS"
  --export=ALL,DISORDERNET_V8_DIR
  rockfish/slurm/v8_extract_embeddings.sbatch)

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '[dry-run]';
  printf ' %q' "${EMBED_CMD[@]}"
  printf '\n'
  printf '[dry-run] sbatch -A %q --dependency=afterok:<EMBED> --export=ALL,DISORDERNET_V8_DIR rockfish/slurm/v8_pipeline.sbatch\n' \
    "$CPU_ACCOUNT"
  exit 0
fi

EMBED_ID="$("${EMBED_CMD[@]}")"
echo "embed job   : $EMBED_ID"

PIPE_ID="$(sbatch --parsable \
  -A "$CPU_ACCOUNT" \
  --dependency="afterok:${EMBED_ID}" \
  --export=ALL,DISORDERNET_V8_DIR \
  rockfish/slurm/v8_pipeline.sbatch)"
echo "pipeline job: $PIPE_ID"
echo
echo "Monitor:  squeue -u \$USER"
echo "Success:  sacct -j $EMBED_ID,$PIPE_ID --format=JobID,State,ExitCode,Elapsed -P"
echo "Results:  cat \$DISORDERNET_V8_DIR/ensemble/results_v8/metrics.json"
