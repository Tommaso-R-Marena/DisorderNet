"""
AlphaFold 3 on Rockfish / Slurm HPC.

Reuses Colab AF3 helpers with filesystem paths instead of Google Drive.
Weights stay off GitHub — place licensed af3.bin under $DISORDERNET_AF3_ROOT.
"""

from __future__ import annotations

import os
import textwrap
from typing import Optional

from colab.af3_colab import (
    DEFAULT_AF3_REPO_URL,
    clone_alphafold3_repo,
    databases_available,
    docker_available,
    print_af3_setup_instructions as _print_colab_af3,
    resolve_af3_paths,
    run_af3_batch,
    select_proteins_for_af3,
    setup_af3_for_colab,
    verify_af3_outputs,
    verify_af3_weights,
)


def default_af3_root() -> str:
    return (
        os.environ.get("DISORDERNET_AF3_ROOT")
        or os.environ.get("AF3_ROOT")
        or os.path.join(os.path.expanduser("~"), "af3")
    )


def default_af3_repo() -> str:
    return (
        os.environ.get("DISORDERNET_AF3_REPO")
        or os.environ.get("AF3_REPO")
        or os.path.join(os.path.expanduser("~"), "software", "alphafold3")
    )


def setup_af3_for_rockfish(
    mode: str = "ingest",
    af3_root: Optional[str] = None,
    alphafold3_repo: Optional[str] = None,
    clone_repo: bool = True,
) -> dict:
    """
    Configure AF3 for HPC.

    Layout (same as Colab Drive layout):
      $DISORDERNET_AF3_ROOT/
        af3.bin
        outputs/
        inputs/
        public_databases/   # optional; skip with msa_free=True
    """
    root = af3_root or default_af3_root()
    repo = alphafold3_repo or default_af3_repo()
    os.makedirs(root, exist_ok=True)

    # Reuse Colab setup with drive_root = HPC root (no Drive mount)
    cfg = setup_af3_for_colab(
        mode=mode if mode != "off" else "ingest",
        drive_root=root,
        alphafold3_repo=repo,
        mount_drive=False,
        clone_repo=clone_repo and mode in ("run", "auto"),
    )
    cfg["cluster"] = "rockfish"
    cfg["af3_root"] = root
    cfg["mode"] = mode

    if mode == "off":
        cfg["ready"] = False
        cfg["skipped"] = True
        return cfg

    w_ok, w_msg = verify_af3_weights(cfg["paths"])
    cfg["weights_ok"] = w_ok
    cfg["weights_message"] = w_msg
    o_ok, o_msg = verify_af3_outputs(cfg["paths"])
    cfg["outputs_ok"] = o_ok
    cfg["outputs_message"] = o_msg
    cfg["docker_ok"] = docker_available()
    cfg["databases_ok"] = databases_available(cfg["paths"])

    if mode == "ingest":
        cfg["ready"] = o_ok
    elif mode in ("run", "auto"):
        cfg["ready"] = w_ok
    return cfg


def print_af3_rockfish_instructions(paths: Optional[dict] = None) -> None:
    paths = paths or resolve_af3_paths(drive_root=default_af3_root())
    print(textwrap.dedent(f"""
    ═══════════════════════════════════════════════════════════════
     AlphaFold 3 on Rockfish — licensed weights (NOT in GitHub)
    ═══════════════════════════════════════════════════════════════
    1. Request AF3 parameters from Google DeepMind (terms apply).
    2. Place af3.bin at: {paths['weights_path']}
    3. Outputs (ingest or run): {paths['output_dir']}
    4. Clone code (once):
         git clone --depth 1 {DEFAULT_AF3_REPO_URL} {default_af3_repo()}
    5. Export:
         export DISORDERNET_AF3_ROOT={paths['drive_root']}
         export DISORDERNET_AF3_REPO={default_af3_repo()}
    6. Submit (MSA-free by default — no 630 GB DBs):
         sbatch rockfish/slurm/af3_batch.sbatch
    ═══════════════════════════════════════════════════════════════
    """))


def run_af3_on_rockfish(
    proteins: list,
    mode: str = "ingest",
    af3_root: Optional[str] = None,
    max_proteins: Optional[int] = None,
    msa_free: bool = True,
    timeout_s: int = 7200,
    prefer_docker: bool = False,
) -> dict:
    """
    Ingest existing AF3 outputs or run missing proteins on the GPU node.

    prefer_docker=False on Rockfish (singularity/native more common).
    """
    cfg = setup_af3_for_rockfish(mode=mode, af3_root=af3_root)
    print_af3_rockfish_instructions(cfg["paths"])

    if mode == "off":
        return {"skipped": True, "mode": "off"}

    if mode == "ingest":
        if not cfg.get("outputs_ok"):
            return {
                "success": False,
                "error": cfg.get("outputs_message"),
                "mode": "ingest",
            }
        done, pending = select_proteins_for_af3(proteins, cfg["paths"]["output_dir"])
        return {
            "success": True,
            "mode": "ingest",
            "n_done": len(done),
            "n_pending": len(pending),
            "paths": cfg["paths"],
        }

    # run / auto
    if not cfg.get("weights_ok"):
        return {"success": False, "error": cfg.get("weights_message"), "mode": mode}

    batch = run_af3_batch(
        proteins=proteins,
        paths=cfg["paths"],
        alphafold3_repo=cfg.get("alphafold3_repo") or default_af3_repo(),
        max_proteins=max_proteins,
        timeout_s=timeout_s,
        msa_free=msa_free,
        prefer_docker=prefer_docker,
        verbose=True,
    )
    batch["mode"] = mode
    batch["paths"] = cfg["paths"]
    return batch
