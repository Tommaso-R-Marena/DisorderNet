"""Download ESM-2 weight files into the torch hub cache — login-node friendly.

Instantiating ESM models (``esm.pretrained.<model>()``) loads them into RAM, which
gets OOM-killed on HPC login nodes. This instead streams the ``.pt`` files straight
to the shared hub cache with ``torch.hub.download_url_to_file`` (chunked I/O, near-zero
memory), so the compute-node jobs later load them from cache without internet.

Usage (on the login node):
  python rockfish/prefetch_esm.py                       # 35M + 150M + 650M
  python rockfish/prefetch_esm.py esm2_t33_650M_UR50D   # a specific backbone
"""
from __future__ import annotations

import os
import sys

_BASE = "https://dl.fbaipublicfiles.com/fair-esm"
DEFAULT_MODELS = [
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
]


def esm_ckpt_targets(model: str, ckpt_dir: str) -> list[tuple[str, str]]:
    """(url, dest) pairs for a model's weights + contact-regression file."""
    return [
        (f"{_BASE}/models/{model}.pt", os.path.join(ckpt_dir, f"{model}.pt")),
        (f"{_BASE}/regression/{model}-contact-regression.pt",
         os.path.join(ckpt_dir, f"{model}-contact-regression.pt")),
    ]


def main(models=None):
    import torch

    models = models or (sys.argv[1:] or DEFAULT_MODELS)
    ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for model in models:
        for url, dest in esm_ckpt_targets(model, ckpt_dir):
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                print(f"cached: {dest}", flush=True)
                continue
            print(f"downloading {url}", flush=True)
            torch.hub.download_url_to_file(url, dest, progress=True)
    print(f"ESM weights ready in {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
