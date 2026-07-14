#!/usr/bin/env bash
# DEPRECATED: use the two separate submitters instead.
#
#   bash rockfish/slurm/submit_publish_650m.sh   # ESM-2 650M + clean + package
#   bash rockfish/slurm/submit_publish_3b.sh     # ESM-2 3B + clean + package
#
# Or CLI:
#   python rockfish/publish_submit.py submit-650m --account "$DISORDERNET_ACCOUNT"
#   python rockfish/publish_submit.py submit-3b  --account "$DISORDERNET_ACCOUNT"

set -euo pipefail
echo "ERROR: submit_publish_all.sh is retired." >&2
echo "Use:" >&2
echo "  bash rockfish/slurm/submit_publish_650m.sh" >&2
echo "  bash rockfish/slurm/submit_publish_3b.sh" >&2
echo "See rockfish/README.md § Publish path." >&2
exit 2
