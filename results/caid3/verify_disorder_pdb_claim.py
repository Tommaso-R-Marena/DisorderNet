"""Independent recomputation of the Disorder-PDB claim.

The last time a number like this appeared it was wrong: a structural baseline
scored 0.9581 on the subset of targets it could reach, and 0.9382 on all 319.
So recompute from the archived submission with a separate code path, confirm the
target set is complete, and rerun the paired tests from scratch.
"""
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "/home/tommaso_marena/DisorderNet")
from colab.caid3_official import (EXPECTED, composition, evaluated_mask,
                                  paired_bootstrap, read_caid_predictions,
                                  read_reference)

S = os.path.dirname(os.path.abspath(__file__))
REFS, PREDS, MINE = f"{S}/caid3_official", f"{S}/caid3_preds/predictions/merged", f"{S}/mtfull"

ref = read_reference(f"{REFS}/disorder_pdb.fasta")
assert composition(ref) == EXPECTED["disorder_pdb"], composition(ref)
print(f"reference composition verified: {composition(ref)}")

for name, path in (("DisorderNet", f"{MINE}/DisorderNet-disorder_pdb.caid"),
                   ("DisorderNet-fused", f"{MINE}/DisorderNet-fused-disorder_pdb.caid"),
                   ("PUNCH2", f"{PREDS}/PUNCH2.caid"),
                   ("AlphaFold-rsa", f"{PREDS}/AlphaFold-rsa.caid")):
    p = read_caid_predictions(path)
    ys, ss, miss = [], [], 0
    for tid, (seq, lab) in ref.items():
        v = p.get(tid)
        if v is None or len(v) != len(lab):
            miss += 1
            continue
        m = evaluated_mask(lab)
        y = (np.frombuffer(lab.encode(), dtype=np.uint8)[m] - ord("0")).astype(int)
        s = v[m]
        ok = np.isfinite(s)
        ys.append(y[ok]); ss.append(s[ok])
    y, s = np.concatenate(ys), np.concatenate(ss)
    print(f"{name:<20} AUC {roc_auc_score(y, s):.4f}  APS "
          f"{average_precision_score(y, s):.4f}  targets {len(ys)}/319  "
          f"missing {miss}  residues {len(y):,}")

print("\npaired, protein-clustered, recomputed from scratch:")
stage = f"{S}/verify_stage"
os.makedirs(stage, exist_ok=True)
import shutil
for fn in os.listdir(PREDS):
    d = os.path.join(stage, fn)
    if not os.path.exists(d):
        os.symlink(os.path.join(PREDS, fn), d)
shutil.copy(f"{MINE}/DisorderNet-disorder_pdb.caid", f"{stage}/DisorderNet.caid")
shutil.copy(f"{MINE}/DisorderNet-fused-disorder_pdb.caid", f"{stage}/DisorderNet-fused.caid")
shutil.copy(f"{REFS}/disorder_pdb.fasta", f"{stage}/disorder_pdb.fasta")
for a in ("DisorderNet", "DisorderNet-fused"):
    for b in ("PUNCH2", "AlphaFold-rsa", "PUNCH2-Light"):
        r = paired_bootstrap("disorder_pdb", stage, stage, a, b, n_boot=1500, seed=1)
        print(f"  {a:<18} - {b:<14} {r['delta_auc']:+.4f} "
              f"[{r['delta_ci'][0]:+.4f},{r['delta_ci'][1]:+.4f}] p={r['p_two_sided']:.4f} "
              f"n={r['n_common_targets']}")
