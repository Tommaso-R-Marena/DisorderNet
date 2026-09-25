"""GPU-node preflight checks for the DisorderNet publish campaign.

Run by ``gpu_smoke.sbatch``. Exits non-zero on any failure so the smoke job's
Slurm state reflects the result instead of burying it in the log.
"""

from __future__ import annotations

import sys

failures: list[str] = []


def check(label: str, fn) -> None:
    try:
        detail = fn()
        print(f"  PASS  {label}{f' — {detail}' if detail else ''}")
    except Exception as exc:  # noqa: BLE001 - preflight reports, never masks
        failures.append(f"{label}: {exc}")
        print(f"  FAIL  {label} — {exc}")


def _torch_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA not visible (torch {torch.__version__}, built for "
            f"cuda {torch.version.cuda}). Check the driver vs the wheel's CUDA."
        )
    dev = torch.cuda.get_device_name(0)
    x = torch.randn(2048, 2048, device="cuda")
    y = (x @ x).sum().item()
    if y != y:  # NaN
        raise RuntimeError("matmul produced NaN")
    return f"{dev}, torch {torch.__version__}/cu{torch.version.cuda}, matmul OK"


def _repo_imports() -> str:
    import colab.calibration  # noqa: F401
    import colab.disordernet_gpu  # noqa: F401
    import colab.homology_splits  # noqa: F401
    import colab.meta_ensemble  # noqa: F401

    return "core modules import"


def _autojunk_guard() -> str:
    """The bug that made homology splits a no-op: difflib junks every amino acid
    for inputs >= 200 residues, collapsing a near-duplicate's score to ~0.01."""
    import random

    from colab.homology_splits import sequence_identity

    rng = random.Random(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    base = "".join(rng.choice(aa) for _ in range(400))
    mutant = list(base)
    for i in rng.sample(range(400), 20):
        mutant[i] = rng.choice(aa)
    ident = sequence_identity(base, "".join(mutant))
    if ident < 0.80:
        raise RuntimeError(
            f"identity of a 95%-identical 400aa pair is {ident:.3f}; "
            "autojunk is back on and homology splits are a no-op"
        )
    return f"400aa near-duplicate identity = {ident:.3f}"


def _clustering_works() -> str:
    import random

    from colab.homology_splits import cluster_proteins_by_homology

    rng = random.Random(1)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    proteins = []
    for f in range(5):
        base = "".join(rng.choice(aa) for _ in range(450))
        for k in range(4):
            s = list(base)
            for i in rng.sample(range(450), 22):
                s[i] = rng.choice(aa)
            proteins.append({"id": f"F{f}_{k}", "sequence": "".join(s), "length": 450})
    _, meta = cluster_proteins_by_homology(proteins, min_identity=0.40)
    if meta["n_clusters"] != 5 or meta["degenerate"]:
        raise RuntimeError(f"expected 5 clusters, got {meta}")
    return f"{meta['n_clusters']} clusters from 20 proteins in 5 families"


check("torch + CUDA", _torch_cuda)
check("repo imports", _repo_imports)
check("homology autojunk guard", _autojunk_guard)
check("homology clustering", _clustering_works)

if failures:
    print(f"\n{len(failures)} preflight check(s) FAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    sys.exit(1)
print("\nAll preflight checks passed.")
