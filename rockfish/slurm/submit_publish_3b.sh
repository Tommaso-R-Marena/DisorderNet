#!/usr/bin/env bash
# Submit the ESM-2 3B publish bundle (ultra3b + optional clean → package).
#
# Exact usage:
#   cd ~/DisorderNet && git checkout master
#   source ~/venvs/disordernet/bin/activate
#   mkdir -p logs
#   export DISORDERNET_ACCOUNT=your_pi_gpu
#   bash rockfish/slurm/submit_publish_3b.sh
#
# If OOM on 40GB A100:
#   bash rockfish/slurm/submit_publish_3b.sh --partition ica100
#
# Equivalent CLI:
#   python rockfish/publish_submit.py submit-3b --account "$DISORDERNET_ACCOUNT"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

exec python rockfish/publish_submit.py submit-3b "$@"
