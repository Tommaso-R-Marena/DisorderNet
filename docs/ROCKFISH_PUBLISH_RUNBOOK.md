# Rockfish publish runbook

Canonical usage (checkout, main + clean sbatch, artifact list, go/no-go) is in
**[rockfish/README.md — Publish path](../rockfish/README.md#publish-path-main--clean-companion)**.

Related checklists / claim docs:

- [`METHODS_CHECKLIST.md`](METHODS_CHECKLIST.md)
- [`STRUCTURE_DISTRUST_ATLAS.md`](STRUCTURE_DISTRUST_ATLAS.md)
- [`PAPER_OUTLINE_STRUCTURE_DISTRUST.md`](PAPER_OUTLINE_STRUCTURE_DISTRUST.md)

```text
checkout master → setup_env → bash rockfish/slurm/submit_publish_all.sh
  → open publish_package/ → METHODS_CHECKLIST → go/no-go on numbers
```
