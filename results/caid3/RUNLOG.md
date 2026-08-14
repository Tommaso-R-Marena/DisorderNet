# Run log — CAID3 evaluations

Exact commands and provenance, so every number in `results/caid3/` can be traced
to the run that produced it. Hypotheses live in `PREREGISTRATION.md` and are not
edited after a run starts; this file only records what was executed.

## mt_full — exploratory (commit `7bd80d6`)

Launched from an ad-hoc heredoc before `train_multitask.sbatch` existed, which is
why Slurm records no Command for job 29800793. Configuration recovered from the
checkpoint payload and results file:

    tasks                 disorder_nox, linker, binding, binding_idr, disorder_pdb
    n_folds               5
    epochs                8
    backbone              ESM-2 650M, frozen, layers 21-32
    structure_dim         24
    wide_receptive_field  True
    min_identity          0.40
    n_train_proteins      20,053
    long proteins         dropped (2,440 on the pdb_missing source alone)

Evaluation: job 29830157, `rockfish/eval_caid3_official.py`, windowed inference,
2,000 bootstrap resamples. Results in `mt_full_caid3_official.json`, CV in
`mt_full_cv.json`.

Status: **exploratory**. Sixteen comparisons were made against these references
across the session; under Holm none of the favourable ones survive. The rank
(0.9603, 1/115, 319/319 targets) is a point estimate and stands.

## mt_windowed — confirmatory (job 29833016)

    sbatch --job-name=dn-mt-win \
      --export=ALL,DISORDERNET_REPO=$HOME/dn_rigor,\
    MT_WORKDIR=/scratch4/sfried3/jbeale3_disordernet/multitask_windowed,\
    MT_FOLDS=5,MT_EPOCHS=8,MT_STRUCTURE_DIM=24 \
      rockfish/slurm/train_multitask.sbatch --wide-receptive-field --final-model

Identical to `mt_full` in every respect above except one: proteins longer than
`--max-len` (1022) are kept as overlapping windows instead of dropped. Inputs
are the shared cache — `disprot_raw.json`, `mobidb_pdbcov.ndjson` — and the
official CAID3 Disorder-PDB reference, md5 `6feaff35263e7fd4a3f03640c23786fe`,
which matches a fresh download from CAID.

Gated on smoke test 29830158 (2 folds, 1 epoch, `--pdb-missing-limit 600`),
COMPLETED in 01:07:18: 770 long proteins became 2,547 windows, folds grouped by
parent, no task duplicated, no degenerate task.

Analysis is fixed in advance by `PREREGISTRATION.md` and executed by the
evaluator rather than chosen afterwards: primary family of two tests on
Disorder-PDB unfused, Holm across those alone; non-inferiority floor 0.9553;
319/319 coverage required or the run is void.

Expect the structure stage to be slow. Shrake-Rupley SASA is computed per
protein and the long proteins have never been cached, having been dropped
before they reached it. The smoke test spent roughly 45 of its 67 minutes there
for 770 proteins.
