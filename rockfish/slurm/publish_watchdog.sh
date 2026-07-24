#!/usr/bin/env bash
# Legacy-style login-node poller wrapper (matches lab "sqme then resubmit" pattern).
# Prefer submit_publish_full.sh; this script only re-enters an existing campaign.
#
#   nohup bash rockfish/slurm/publish_watchdog.sh /path/to/campaign.json &

set -euo pipefail
CAMPAIGN="${1:?usage: publish_watchdog.sh <campaign.json> [poll_seconds]}"
POLL="${2:-600}"
RECURSIONS="${3:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs
export HOME="${HOME:-/home/$USER}"
exec 2>>"${HOME}/disordernet_publish_poller.log"
printf "[%(%Y-%m-%d %H:%M:%S)T] Starting DisorderNet publish poller pid=%s campaign=%s\n" -1 "$$" "$CAMPAIGN" >&2

while squeue -h -u "$USER" -o '%i' | grep -q .; do
  printf "[%(%Y-%m-%d %H:%M:%S)T] jobs still queued/running — sleep %ss\n" -1 "$POLL" >&2
  sleep "$POLL"
done

source "${HOME}/venvs/disordernet/bin/activate"
if python rockfish/publish_campaign.py step --campaign "$CAMPAIGN"; then
  status="$(python -c "import json;print(json.load(open('$CAMPAIGN'))['status'])")"
  if [[ "$status" == "done" ]]; then
    printf "[%(%Y-%m-%d %H:%M:%S)T] campaign done\n" -1 >&2
    exit 0
  fi
  if (( RECURSIONS < 128 )); then
    printf "[%(%Y-%m-%d %H:%M:%S)T] re-entering poller recursions=%s\n" -1 "$RECURSIONS" >&2
    exec bash "$0" "$CAMPAIGN" "$POLL" "$((RECURSIONS + 1))"
  fi
fi
printf "[%(%Y-%m-%d %H:%M:%S)T] poller stopping (failed or max recursions)\n" -1 >&2
exit 1
