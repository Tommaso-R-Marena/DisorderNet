# Rockfish publish runbook

Canonical usage (two scripts + CLI) is in
**[rockfish/README.md — Publish path (exact usage)](../rockfish/README.md#publish-path-exact-usage)**.

```bash
# Script 1 — 650M
bash rockfish/slurm/submit_publish_650m.sh

# Script 2 — 3B
bash rockfish/slurm/submit_publish_3b.sh

# CLI equivalents
python rockfish/publish_submit.py submit-650m --account "$DISORDERNET_ACCOUNT"
python rockfish/publish_submit.py submit-3b  --account "$DISORDERNET_ACCOUNT"
```

Related: [`METHODS_CHECKLIST.md`](METHODS_CHECKLIST.md),
[`STRUCTURE_DISTRUST_ATLAS.md`](STRUCTURE_DISTRUST_ATLAS.md).

```text
setup → submit_publish_650m.sh and/or submit_publish_3b.sh
  → open publish_package/ → METHODS_CHECKLIST → go/no-go on numbers
```
