# Rockfish publish runbook

Canonical usage (checkout, main + clean sbatch, artifact list, go/no-go) is in
**[rockfish/README.md — Publish path](../rockfish/README.md#publish-path-main--clean-companion)**.

Related checklists / claim docs:

- [`METHODS_CHECKLIST.md`](METHODS_CHECKLIST.md)
- [`STRUCTURE_DISTRUST_ATLAS.md`](STRUCTURE_DISTRUST_ATLAS.md)
- [`PAPER_OUTLINE_STRUCTURE_DISTRUST.md`](PAPER_OUTLINE_STRUCTURE_DISTRUST.md)

```text
checkout PR branch → setup_env → sbatch pipeline_ultra → sbatch pipeline_ultra_clean
  → verify mirrored artifacts → METHODS_CHECKLIST → go/no-go on numbers
```
