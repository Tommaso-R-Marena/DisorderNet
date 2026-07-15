"""Train and save a deployable DisorderNet predictor bundle.

Uses the DisProt cache + ESM embeddings (fetch_disprot.py / extract_esm_embeddings.py).
Saves a joblib bundle for predict_disorder.py. Override data location with DISORDERNET_HOME.

Run:  python train_predictor.py
"""
import json, os, time
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score

from disordernet_paths import DISPROT_JSON, EMB_DIR, results_dir
from confidence import expected_calibration_error
from predictor import fit_bundle, phys_features

SEED = 42
MAX_PROT = 1500
MAX_LEN = 800


def main():
    t0 = time.time()
    with open(DISPROT_JSON) as f:
        data = json.load(f)
    proteins = [
        p for p in data
        if os.path.exists(os.path.join(str(EMB_DIR), f"{p['disprot_id']}.npy"))
        and 30 <= p["length"] <= MAX_LEN and sum(p["disorder_labels"]) >= 3
        and p["length"] - sum(p["disorder_labels"]) >= 3
    ]
    rng = np.random.RandomState(SEED)
    if len(proteins) > MAX_PROT:
        idx = rng.choice(len(proteins), MAX_PROT, replace=False)
        proteins = [proteins[i] for i in sorted(idx)]
    print(f"Training predictor on {len(proteins)} proteins...", flush=True)

    phys_list, esm_list, lab_list = [], [], []
    for p in proteins:
        L = p["length"]
        phys_list.append(phys_features(p["sequence"][:L]))
        esm_list.append(np.load(os.path.join(str(EMB_DIR), f"{p['disprot_id']}.npy")).astype(np.float16)[:L])
        lab_list.append(np.array(p["disorder_labels"][:L], dtype=np.float32))

    bundle = fit_bundle(phys_list, esm_list, lab_list, alpha=0.10)

    out_dir = results_dir("results_v7", create=True)
    path = os.path.join(str(out_dir), "predictor_bundle.joblib")
    joblib.dump(bundle, path)
    print(f"Saved bundle -> {path}  [{(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
