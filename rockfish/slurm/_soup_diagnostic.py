"""Why does the fold soup score worse than the CV it reloads?

The 650M run's held_out soup produced pooled AUC 0.6867 against 0.7999 going in,
and against 0.7454 for the CV's own validation probabilities — i.e. reloading a
checkpoint did not reproduce the predictions it was saved from.

Two candidate explanations, and they call for different fixes:

  A. reload fidelity — the checkpoint does not fully restore (LoRA adapters not
     re-injected, EMA weights lost, dtype drift). Predictions differ from
     val_probs even with TTA off.
  B. MC-dropout TTA — the reload is faithful but averaging 6 stochastic passes
     degrades the ranking.

This compares, on ONE fold's own validation proteins:
    stored val_probs  vs  reload + TTA off  vs  reload + TTA on
so the two are separable. Runs on a single fold: minutes, not hours.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from colab.disordernet_gpu import TrainConfig, fetch_disprot, process_disprot  # noqa: E402
from colab.esm_backbone import apply_backbone_to_config, load_esm_backbone  # noqa: E402
from colab.fold_model_soup import _load_fold_model, _predict_proteins  # noqa: E402

CKPT = os.environ["DISORDERNET_SOUP_CKPT"]
FOLD = int(os.environ.get("DISORDERNET_SOUP_FOLD", "1"))
DISPROT = os.environ.get(
    "DISORDERNET_DISPROT_CACHE",
    str(Path.home() / ".cache" / "disordernet" / "disprot_raw.json"),
)

cfg = TrainConfig.from_profile("ultra")
cfg.checkpoint_dir = CKPT
apply_backbone_to_config(cfg, "650M")
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.num_workers = 2

print(f"checkpoint dir : {CKPT}")
print(f"fold           : {FOLD}")
print(f"TTA configured : {getattr(cfg, 'use_mc_dropout_tta', False)} "
      f"x{getattr(cfg, 'mc_dropout_tta_passes', 1)}")

prog = json.loads(Path(CKPT, "cv_progress.json").read_text())
fr = prog["fold_results"][FOLD - 1]
val_ids = fr["val_ids"]
stored = np.asarray(fr["val_probs"], dtype=np.float32)
stored_labels = np.asarray(fr["val_labels"], dtype=np.float32)
print(f"stored val_probs: {len(stored):,} residues over {len(val_ids)} proteins")

raw = fetch_disprot(cache_path=DISPROT)
proteins, _ = process_disprot(raw, cfg)
by_id = {p["id"]: p for p in proteins}
val_proteins = [by_id[i] for i in val_ids if i in by_id]
print(f"matched proteins: {len(val_proteins)}/{len(val_ids)}")

model_esm, _alphabet, converter, _spec = load_esm_backbone(cfg.device, "650M")
fold_model = _load_fold_model(
    os.path.join(CKPT, f"fold{FOLD}_best.pt"), model_esm, cfg, cfg.device
)
print("checkpoint loaded without error (strict check passed)")

results = {}
for label, use_tta in (("tta_off", False), ("tta_on", True)):
    token_cache: dict = {}
    preds = _predict_proteins(
        fold_model, val_proteins, converter, token_cache, cfg,
        plddt_by_id=None, use_tta=use_tta,
        tta_passes=int(getattr(cfg, "mc_dropout_tta_passes", 6)),
    )
    flat = np.concatenate([preds[p["id"]] for p in val_proteins if p["id"] in preds])
    n = min(len(flat), len(stored))
    rho = float(spearmanr(flat[:n], stored[:n]).statistic)
    auc = float(roc_auc_score(stored_labels[:n], flat[:n]))
    results[label] = {"auc": auc, "spearman_vs_stored": rho, "n": int(n)}
    print(f"  {label:8s}  AUC={auc:.4f}  spearman_vs_stored={rho:.4f}  n={n:,}")

stored_auc = float(roc_auc_score(stored_labels, stored))
print(f"\n  stored    AUC={stored_auc:.4f}  (what the CV recorded)")

off, on = results["tta_off"], results["tta_on"]
print("\n=== verdict ===")
if off["spearman_vs_stored"] < 0.95:
    print("  A. RELOAD FIDELITY. Even with TTA off the reloaded model does not")
    print(f"     reproduce stored predictions (spearman={off['spearman_vs_stored']:.3f}).")
    print("     The checkpoint is not fully restoring the trained state.")
elif on["auc"] < off["auc"] - 0.01:
    print("  B. MC-DROPOUT TTA. The reload is faithful "
          f"(spearman={off['spearman_vs_stored']:.3f}) but TTA costs "
          f"{off['auc'] - on['auc']:.4f} AUC.")
    print("     Disable use_mc_dropout_tta for the soup, or average probabilities")
    print("     rather than logits.")
else:
    print("  Neither reproduces the reported regression on this fold — the loss")
    print("  likely arises in write-back/alignment rather than in prediction.")

out = Path(os.environ.get("DISORDERNET_RESULTS", str(Path.home()))) / "soup_diagnostic.json"
out.write_text(json.dumps(
    {"fold": FOLD, "stored_auc": stored_auc, **results}, indent=2) + "\n")
print(f"\nWrote {out}")
