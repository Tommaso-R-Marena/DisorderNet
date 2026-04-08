"""
Extract ESM-2 protein language model embeddings for all DisProt proteins.
Uses esm2_t12_35M_UR50D (480-dim per residue) for best quality within memory limits.
Falls back to esm2_t6_8M_UR50D (320-dim) if memory issues.
"""
import torch
import esm
import numpy as np
import json
import os
import time
import gc

DATA_PATH = "/home/user/workspace/disorder_model/data/disprot_processed.json"
EMB_DIR = "/home/user/workspace/disorder_model/data/embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

MAX_SEQ_LEN = 1022  # ESM-2 max (with BOS/EOS tokens = 1024)


def extract_embeddings(model_name="esm2_t12_35M_UR50D", repr_layer=12):
    """Extract per-residue embeddings for all DisProt proteins."""
    
    print(f"Loading {model_name}...")
    if model_name == "esm2_t12_35M_UR50D":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    else:
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        repr_layer = 6
    
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    # Load data
    with open(DATA_PATH) as f:
        proteins = json.load(f)
    
    print(f"Total proteins: {len(proteins)}")
    
    # Process one by one to manage memory
    t0 = time.time()
    processed = 0
    skipped = 0
    
    for idx, protein in enumerate(proteins):
        disprot_id = protein["disprot_id"]
        seq = protein["sequence"]
        
        # Check if already processed
        outfile = os.path.join(EMB_DIR, f"{disprot_id}.npy")
        if os.path.exists(outfile):
            processed += 1
            continue
        
        # Truncate if needed
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[:MAX_SEQ_LEN]
        
        if len(seq) < 10:
            skipped += 1
            continue
        
        try:
            data = [(disprot_id, seq)]
            _, _, tokens = batch_converter(data)
            
            with torch.no_grad():
                results = model(tokens, repr_layers=[repr_layer])
            
            emb = results["representations"][repr_layer][0, 1:len(seq)+1].numpy()
            
            # Save compressed
            np.save(outfile, emb.astype(np.float16))  # float16 to save space
            processed += 1
            
        except Exception as e:
            print(f"  Error on {disprot_id} (len={len(seq)}): {e}")
            skipped += 1
        
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(proteins) - idx - 1) / rate
            mem_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            print(f"  {idx+1}/{len(proteins)} ({elapsed:.0f}s, ETA: {eta:.0f}s, processed: {processed})")
            gc.collect()
    
    elapsed = time.time() - t0
    print(f"\nDone! Processed: {processed}, Skipped: {skipped}, Time: {elapsed:.0f}s")
    print(f"Embeddings saved to {EMB_DIR}/")
    
    return processed


if __name__ == "__main__":
    # Try 35M first, fall back to 8M if needed
    try:
        n = extract_embeddings("esm2_t12_35M_UR50D", repr_layer=12)
    except (RuntimeError, MemoryError) as e:
        print(f"35M model failed ({e}), falling back to 8M model...")
        n = extract_embeddings("esm2_t6_8M_UR50D", repr_layer=6)
    
    print(f"\nTotal embeddings: {n}")
