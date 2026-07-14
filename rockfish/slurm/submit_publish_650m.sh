#!/usr/bin/env bash
# Submit the ESM-2 650M publish bundle (ultra + optional clean → package).
#
# Exact usage:
#   cd ~/DisorderNet && git checkout master
#   source ~/venvs/disordernet/bin/activate
#   mkdir -p logs
#   export DISORDERNET_ACCOUNT=sfried3
#   bash rockfish/slurm/submit_publish_650m.sh
#
# Equivalent CLI:
#   python rockfish/publish_submit.py submit-650m --account "$DISORDERNET_ACCOUNT"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

exec python rockfish/publish_submit.py submit-650m "$@"
