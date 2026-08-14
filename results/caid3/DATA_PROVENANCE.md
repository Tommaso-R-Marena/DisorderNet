# CAID3 data provenance

Every number in `results/caid3/` is computed from these files. They are not in
the repository — the prediction archive alone is 76 MB — so their identity is
pinned here instead. `colab/caid3_official.py` re-downloads and re-verifies from
scratch, and `verify_composition` refuses any reference whose composition does
not match what CAID's dataset API reports independently.

## Reference FASTAs

Source: `https://caid.idpcentral.org/assets/sections/challenge/static/references/3/`

| file | md5 | targets | positives | negatives | unevaluated |
|---|---|---:|---:|---:|---:|
| `disorder_pdb.fasta` | `6feaff35263e7fd4a3f03640c23786fe` | 319 | 31,401 | 67,838 | 61,183 |
| `disorder_nox.fasta` | `e2056cccfe186ab9da8d1e15f1508bef` | 204 | 26,367 | 73,610 | 541 |
| `binding.fasta` | `3094881568be4105b0e9afc576a79558` | 52 | 2,991 | 25,272 | 0 |
| `binding_idr.fasta` | `4331f14502505e5dea58fcef0a29b2b7` | 52 | 2,991 | 4,750 | 20,522 |
| `linker.fasta` | `603e3055d5bed0d5af421a81123b252f` | 31 | 1,379 | 19,119 | 0 |

Composition cross-checked against
`https://caid.idpcentral.org/dataset-repository/api/dataset/CAID3%20v3/reference-set/<name>/`,
which serves those counts separately from the files themselves.

**Use the dataset named "CAID3 v3".** The API also has an entry named plainly
"CAID3" — 185 proteins — which is a different, smaller set and does not
reproduce the published leaderboard. Scoring against it silently answers a
different question.

## Entrant predictions

Source: `https://caid.idpcentral.org/assets/sections/challenge/static/predictions/3/predictions.zip`
(76,480,806 bytes), extracted to `predictions/merged/`.

117 `.caid` files. Aggregate md5 of `md5sum *.caid`:
`cdef7ae6dc3dd6ec8055aacdb14a54a7`.

Three of them — `FoldUnfold`, `NeProc-binding`, `NeProc-disorder` — leave the
score field blank on residues they decline. Those parse to NaN, are counted, and
are never silently dropped.

## Locations

    /scratch4/sfried3/jbeale3_disordernet/caid3_official/      five references
    /scratch4/sfried3/jbeale3_disordernet/caid3_predictions/   117 .caid files

Not `$HOME` — the 50 GB quota there is invisible to `df` and `quota`, and a job
that crosses it fails on its next write, including the write of its own log.
Not `/tmp` either: it is tmpfs on the workstation and was cleared mid-session,
taking a local copy with it.

## The check that matters

`verify_against_published` recomputes each challenge's published leader from
these files. All five reproduce, which is what licenses every other number here:

| challenge | leader | computed | published |
|---|---|---:|---:|
| Disorder-PDB | PUNCH2 | 0.9552 | 0.955 |
| Disorder-NOX | ESMDisPred-2PDB | 0.8855 | 0.885 |
| Binding | DisoFLAG-PB | 0.7760 | 0.776 |
| Binding-IDR | bindEmbed21IDR-rawGeneral | 0.6407 | 0.641 |
| Linker | IPA-AF2-Linker | 0.8985 | 0.897 |

If a leader stops reproducing, the pipeline is wrong or the upstream files have
changed. Either way no result computed after that point should be believed until
it is explained.
