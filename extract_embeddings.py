"""Unified, GPU-aware ESM-2 embedding extractor for the multi-scale pipeline.

Extracts per-residue ESM-2 embeddings for any backbone into an output directory,
using a GPU when available (with length-sorted batching) and falling back to CPU.
Replaces the three hard-coded extraction scripts; the Colab notebook and the
Rockfish sbatch both call this.

Examples:
  python extract_embeddings.py --model esm2_t12_35M_UR50D  --out data/emb_35m
  python extract_embeddings.py --model esm2_t33_650M_UR50D --out data/emb_650m --batch-tokens 8000
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np

# name -> per-residue embedding dim (for reference/validation)
BACKBONE_DIM = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t36_3B_UR50D": 2560,
}


def repr_layer_for(model_name: str) -> int:
    """The final layer index equals the layer count encoded as ``_t<N>_``."""
    try:
        return int(model_name.split("_t", 1)[1].split("_", 1)[0])
    except (IndexError, ValueError) as exc:  # pragma: no cover - guarded by CLI
        raise ValueError(f"cannot parse repr layer from model name '{model_name}'") from exc


def _iter_length_batches(items, batch_tokens):
    """Yield batches of (idx, id, seq) grouped by ascending length under a token budget."""
    order = sorted(range(len(items)), key=lambda i: len(items[i][1]))
    batch, budget = [], 0
    for i in order:
        L = len(items[i][1]) + 2  # BOS/EOS
        if batch and (budget + L > batch_tokens or len(batch) >= 64):
            yield batch
            batch, budget = [], 0
        batch.append((i, items[i][0], items[i][1]))
        budget += L
    if batch:
        yield batch


def extract(model_name, proteins, out_dir, max_len=1022, batch_tokens=6000,
            device=None, verbose=True):
    import torch
    import esm

    os.makedirs(out_dir, exist_ok=True)
    fn = getattr(esm.pretrained, model_name)
    model, alphabet = fn()
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    bc = alphabet.get_batch_converter()
    layer = repr_layer_for(model_name)

    todo = []
    for p in proteins:
        pid = p["disprot_id"]
        if os.path.exists(os.path.join(out_dir, f"{pid}.npy")):
            continue
        seq = p["sequence"][:max_len]
        if len(seq) >= 10:
            todo.append((pid, seq))
    if verbose:
        print(f"[{model_name}] device={device} layer={layer} to-extract={len(todo)}", flush=True)

    t0 = time.time()
    done = 0
    for batch in _iter_length_batches(todo, batch_tokens):
        data = [(pid, seq) for (_, pid, seq) in batch]
        _, _, toks = bc(data)
        toks = toks.to(device)
        with torch.no_grad():
            rep = model(toks, repr_layers=[layer])["representations"][layer]
        for k, (_, pid, seq) in enumerate(batch):
            emb = rep[k, 1:len(seq) + 1].float().cpu().numpy().astype(np.float16)
            np.save(os.path.join(out_dir, f"{pid}.npy"), emb)
            done += 1
        del toks, rep
        if device == "cuda":
            torch.cuda.empty_cache()
        if verbose and done % 500 < len(batch):
            el = time.time() - t0
            print(f"  {done}/{len(todo)}  {el:.0f}s ({done/max(el,1e-9):.1f}/s)", flush=True)
        gc.collect()
    if verbose:
        print(f"[{model_name}] done {done} in {time.time()-t0:.0f}s -> {out_dir}", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(BACKBONE_DIM))
    ap.add_argument("--data", default=None, help="DisProt processed json (default: disordernet_paths)")
    ap.add_argument("--out", required=True, help="output embeddings dir")
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--batch-tokens", type=int, default=6000)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    args = ap.parse_args()

    data_path = args.data
    if data_path is None:
        from disordernet_paths import DISPROT_JSON
        data_path = str(DISPROT_JSON)
    with open(data_path) as f:
        proteins = json.load(f)
    extract(args.model, proteins, args.out, max_len=args.max_len,
            batch_tokens=args.batch_tokens, device=args.device)


if __name__ == "__main__":
    main()
