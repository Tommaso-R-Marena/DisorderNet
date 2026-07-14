"""
Boltz-2 on Rockfish / Slurm HPC.

Default structure backend for DisorderNet — MIT-licensed, auto-downloads
pinned ``boltz==2.2.1`` weights (no DeepMind AF3 license file).
"""

from __future__ import annotations

import json
import os
from typing import Optional

from colab.boltz_runner import (
    PINNED_BOLTZ_VERSION,
    default_boltz_root,
    ensure_boltz_installed,
    print_boltz_setup_instructions,
    resolve_boltz_paths,
    run_boltz_batch,
    save_boltz_manifest,
)


def setup_boltz_for_rockfish(
    mode: str = "auto",
    boltz_root: Optional[str] = None,
    ensure_install: bool = True,
) -> dict:
    """
    Configure Boltz-2 for HPC.

    Layout::
      $DISORDERNET_BOLTZ_ROOT/
        inputs/
        outputs/          # boltz predict --out_dir
        cache/            # weights (or $BOLTZ_CACHE)
        boltz_run_manifest.json
    """
    root = boltz_root or default_boltz_root()
    paths = resolve_boltz_paths(root)
    os.makedirs(paths["boltz_root"], exist_ok=True)
    os.makedirs(paths["input_dir"], exist_ok=True)
    os.makedirs(paths["output_dir"], exist_ok=True)
    os.makedirs(paths["cache_dir"], exist_ok=True)

    cfg: dict = {
        "cluster": "rockfish",
        "mode": mode,
        "boltz_root": root,
        "paths": paths,
        "pinned_version": PINNED_BOLTZ_VERSION,
        "skipped": mode == "off",
    }
    if mode == "off":
        cfg["ready"] = False
        return cfg

    from colab.boltz_plddt import select_proteins_for_boltz

    # outputs_ok if directory exists (may be empty — still valid for run/auto)
    cfg["outputs_ok"] = os.path.isdir(paths["output_dir"])
    cfg["install"] = None
    if ensure_install and mode in ("run", "auto"):
        cfg["install"] = ensure_boltz_installed(version=PINNED_BOLTZ_VERSION)
        cfg["ready"] = bool(cfg["install"].get("already_installed"))
    elif mode == "ingest":
        cfg["ready"] = cfg["outputs_ok"]
    else:
        cfg["ready"] = True
    return cfg


def write_boltz_pending_list(
    proteins: list,
    output_dir: str,
    list_path: str,
) -> list[dict]:
    """Write JSONL of proteins still missing Boltz outputs (Slurm arrays)."""
    from colab.boltz_plddt import select_proteins_for_boltz

    _, pending = select_proteins_for_boltz(proteins, output_dir)
    os.makedirs(os.path.dirname(list_path) or ".", exist_ok=True)
    with open(list_path, "w") as f:
        for p in pending:
            f.write(json.dumps({
                "id": p["id"],
                "uniprot_acc": p.get("uniprot_acc", ""),
                "sequence": p["sequence"],
                "length": p.get("length", len(p["sequence"])),
            }) + "\n")
    print(f"Boltz pending list: {len(pending)} proteins → {list_path}")
    return pending


def load_boltz_pending_shard(
    list_path: str,
    shard_index: int,
    shard_count: int,
) -> list[dict]:
    proteins: list[dict] = []
    with open(list_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i % shard_count != shard_index:
                continue
            proteins.append(json.loads(line))
    return proteins


def run_boltz_on_rockfish(
    proteins: list,
    mode: str = "auto",
    boltz_root: Optional[str] = None,
    max_proteins: Optional[int] = None,
    msa_free: bool = True,
    use_msa_server: bool = False,
    timeout_s: int = 7200,
    shard_index: Optional[int] = None,
    shard_count: Optional[int] = None,
    sampling_steps: int = 50,
    diffusion_samples: int = 5,
) -> dict:
    """
    Ingest existing Boltz outputs or run missing proteins (weights auto-download).

    mode:
      off     — skip
      ingest  — load existing outputs only
      run     — install pinned boltz + predict pending
      auto    — ingest then run pending

    ``diffusion_samples>=2`` enables the multi-sample pLDDT variance proxy used
    by the IDR biology layer (does not change mean-confidence model_0 path).
    """
    cfg = setup_boltz_for_rockfish(mode=mode, boltz_root=boltz_root)
    print_boltz_setup_instructions(cfg["paths"])

    if mode == "off":
        return {"skipped": True, "mode": "off", "paths": cfg["paths"]}

    from colab.boltz_plddt import select_proteins_for_boltz

    paths = cfg["paths"]
    done, pending = select_proteins_for_boltz(proteins, paths["output_dir"])

    if mode == "ingest":
        return {
            "success": True,
            "mode": "ingest",
            "n_done": len(done),
            "n_pending": len(pending),
            "paths": paths,
            "pinned_version": PINNED_BOLTZ_VERSION,
            "diffusion_samples": diffusion_samples,
        }

    # run / auto
    if shard_index is not None and shard_count is not None and shard_count > 1:
        run_proteins = [
            p for i, p in enumerate(pending) if i % shard_count == shard_index
        ]
        print(
            f"Boltz array shard {shard_index}/{shard_count}: "
            f"{len(run_proteins)}/{len(pending)} pending",
            flush=True,
        )
        if max_proteins is not None:
            run_proteins = run_proteins[:max_proteins]
        batch_proteins = run_proteins
        batch_max = None
    elif mode == "auto":
        batch_proteins = pending[:max_proteins] if max_proteins is not None else pending
        batch_max = None
    else:
        batch_proteins = proteins
        batch_max = max_proteins

    batch = run_boltz_batch(
        proteins=batch_proteins,
        boltz_root=paths["boltz_root"],
        max_proteins=batch_max,
        msa_free=msa_free,
        use_msa_server=use_msa_server,
        timeout_s=timeout_s,
        pin_version=PINNED_BOLTZ_VERSION,
        ensure_install=True,
        sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
        verbose=True,
    )
    batch["mode"] = mode
    batch["shard_index"] = shard_index
    batch["shard_count"] = shard_count
    save_boltz_manifest(batch, paths["manifest_path"])
    return batch
