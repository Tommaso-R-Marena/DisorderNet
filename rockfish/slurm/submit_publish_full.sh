#!/usr/bin/env bash
# Kick off the full publication campaign (650M → 3B) with login-node watchdog.
#
# Rockfish walltime (ARCH docs): GPU a100/ica100 = 72 h max; shared CPU ≈ 36 h.
# Training resumes from cv_progress.json fold checkpoints; the watchdog
# resubmits after TIMEOUT / empty queue until both publish_package/ dirs exist.
#
# Fresh shell (login node):
#   cd ~/DisorderNet && git pull && source ~/venvs/disordernet/bin/activate
#   bash rockfish/slurm/submit_publish_full.sh
#
# Optional:
#   bash rockfish/slurm/submit_publish_full.sh --dry-run
#   bash rockfish/slurm/submit_publish_full.sh --partition-3b ica100
#   bash rockfish/slurm/submit_publish_full.sh --no-watch   # submit step only via python later

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

MAIL_USER="${DISORDERNET_MAIL_USER:-marenatommaso@gmail.com}"
GPU_ACCOUNT="${DISORDERNET_GPU_ACCOUNT:-}"
if [[ -z "$GPU_ACCOUNT" ]]; then
  GPU_ACCOUNT="$(sacctmgr -nP show assoc user="$USER" format=account,qos \
    | awk -F'|' '/qos_gpu/{print $1; exit}')"
fi
GPU_ACCOUNT="${GPU_ACCOUNT:-sfried3_gpu}"
QOS="${DISORDERNET_GPU_QOS:-qos_gpu}"
RESULTS="${DISORDERNET_RESULTS:-$HOME/disordernet_runs}"
STAMP="${DISORDERNET_PUBLISH_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
CAMPAIGN="${DISORDERNET_CAMPAIGN:-$RESULTS/campaign_${STAMP}.json}"
POLL="${DISORDERNET_POLL_SECONDS:-600}"
PARTITION_3B=""
DRY=()
NO_WATCH=0
NO_CLEAN=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=(--dry-run); shift ;;
    --no-watch) NO_WATCH=1; shift ;;
    --no-clean) NO_CLEAN=(--no-clean); shift ;;
    --partition-3b) PARTITION_3B="$2"; shift 2 ;;
    --poll-seconds) POLL="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --stamp) STAMP="$2"; shift 2 ;;
    --account) GPU_ACCOUNT="$2"; shift 2 ;;
    --mail-user) MAIL_USER="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

export DISORDERNET_MAIL_USER="$MAIL_USER"
export DISORDERNET_RESULTS="$RESULTS"
mkdir -p "$RESULTS" logs

INIT_ARGS=(
  init --campaign "$CAMPAIGN"
  --results-root "$RESULTS"
  --stamp "$STAMP"
  --account "$GPU_ACCOUNT"
  --qos "$QOS"
  --mail-user "$MAIL_USER"
)
if [[ ${#NO_CLEAN[@]} -gt 0 ]]; then
  INIT_ARGS+=("${NO_CLEAN[@]}")
fi
if [[ -n "$PARTITION_3B" ]]; then
  INIT_ARGS+=(--partition-3b "$PARTITION_3B")
fi

echo "Campaign file: $CAMPAIGN"
echo "GPU account:   $GPU_ACCOUNT  QOS=$QOS  mail=$MAIL_USER"
python rockfish/publish_campaign.py "${INIT_ARGS[@]}"

if [[ "$NO_WATCH" -eq 1 ]]; then
  echo "Initialized only. Start watchdog with:"
  echo "  nohup python rockfish/publish_campaign.py watch --campaign $CAMPAIGN --poll-seconds $POLL > logs/publish_watchdog.out 2>&1 &"
  exit 0
fi

# Prefer detached watchdog so SSH disconnect does not kill the poller.
WATCH_LOG="$REPO_ROOT/logs/publish_watchdog_${STAMP}.out"
nohup python rockfish/publish_campaign.py watch \
  --campaign "$CAMPAIGN" \
  --poll-seconds "$POLL" \
  "${DRY[@]}" \
  >"$WATCH_LOG" 2>&1 &
echo "Watchdog PID $!  log=$WATCH_LOG"
echo "Status:  python rockfish/publish_campaign.py status --campaign $CAMPAIGN"
echo "squeue:  squeue -u \$USER"
