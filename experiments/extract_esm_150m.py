"""Extract ESM-2 150M (esm2_t30_150M_UR50D, 640-dim) per-residue embeddings.

Experiment: does a larger protein language model backbone push CPU AUC above the
35M-based v7 result (0.848)? Writes float16 .npy per protein into a separate home
so run_v7.py can be pointed at it via DISORDERNET_HOME.

Run:  python experiments/extract_esm_150m.py
"""
import os, time, gc, json
import numpy as np
import torch
import esm

SRC = "/home/user/workspace/disorder_model/data/disprot_processed.json"
EMB_DIR = "/home/user/workspace/dn150m/data/embeddings"
os.makedirs(EMB_DIR, exist_ok=True)
MAX_SEQ_LEN = 1022


def main():
    print("Loading esm2_t30_150M_UR50D ...", flush=True)
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model.eval()
    repr_layer = 30
    bc = alphabet.get_batch_converter()

    with open(SRC) as f:
        proteins = json.load(f)
    print(f"Total proteins: {len(proteins)}", flush=True)

    t0 = time.time(); done = 0; skip = 0
    for idx, p in enumerate(proteins):
        pid = p["disprot_id"]
        seq = p["sequence"][:MAX_SEQ_LEN]
        out = os.path.join(EMB_DIR, f"{pid}.npy")
        if os.path.exists(out):
            done += 1
            continue
        if len(seq) < 10:
            skip += 1
            continue
        try:
            _, _, toks = bc([(pid, seq)])
            with torch.no_grad():
                res = model(toks, repr_layers=[repr_layer])
            emb = res["representations"][repr_layer][0, 1:len(seq) + 1].numpy()
            np.save(out, emb.astype(np.float16))
            done += 1
        except (RuntimeError, MemoryError) as e:
            print(f"  err {pid} len={len(seq)}: {e}", flush=True)
            skip += 1
        if (idx + 1) % 200 == 0:
            el = time.time() - t0
            rate = (idx + 1) / el
            eta = (len(proteins) - idx - 1) / rate
            print(f"  {idx+1}/{len(proteins)}  {el:.0f}s ETA {eta:.0f}s (done={done})", flush=True)
            gc.collect()
    print(f"Done. processed={done} skipped={skip} time={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
