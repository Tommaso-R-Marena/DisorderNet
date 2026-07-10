"""
AlphaFold 3 Google Colab integration (Phase 2b).

Weights and outputs live on Google Drive — NOT in this GitHub repo.
AF3 model parameters are licensed separately; af3.bin must be obtained
from Google DeepMind and uploaded to Drive by the user.

Modes:
  - off:    skip AF3 (AF2-only Phase 2)
  - ingest: read precomputed AF3 outputs from Drive (recommended)
  - run:    run AF3 inference on a small subset on Colab A100 (optional)
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from typing import Optional

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/DisorderNet/af3"
DEFAULT_WEIGHTS_NAME = "af3.bin"
DEFAULT_MODEL_DIR_NAME = "af3_params"
DEFAULT_OUTPUT_DIR_NAME = "outputs"
DEFAULT_DATABASE_DIR_NAME = "public_databases"


def mount_google_drive() -> str:
    """Mount Google Drive in Colab; no-op if not in Colab."""
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        return "/content/drive/MyDrive"
    except ImportError:
        return ""


def resolve_af3_paths(
    drive_root: str = DEFAULT_DRIVE_ROOT,
    weights_name: str = DEFAULT_WEIGHTS_NAME,
) -> dict[str, str]:
    """
    Standard Drive layout:

    {drive_root}/
      af3.bin                 # model weights (user upload)
      af3_params/             # extracted parameter tree (if applicable)
      outputs/                # AF3 job outputs (ingest or run)
      public_databases/       # optional; ~630 GB for full pipeline
    """
    return {
        "drive_root": drive_root,
        "weights_path": os.path.join(drive_root, weights_name),
        "model_dir": os.path.join(drive_root, DEFAULT_MODEL_DIR_NAME),
        "output_dir": os.path.join(drive_root, DEFAULT_OUTPUT_DIR_NAME),
        "database_dir": os.path.join(drive_root, DEFAULT_DATABASE_DIR_NAME),
    }


def verify_af3_weights(paths: dict[str, str]) -> tuple[bool, str]:
    """Check that AF3 weights exist on Drive."""
    wp = paths["weights_path"]
    if os.path.isfile(wp):
        size_gb = os.path.getsize(wp) / (1024 ** 3)
        return True, f"Found weights: {wp} ({size_gb:.2f} GB)"
    return False, (
        f"AF3 weights not found at {wp}. "
        "Upload af3.bin from Google DeepMind to this Drive path. "
        "Do NOT commit weights to GitHub."
    )


def verify_af3_outputs(paths: dict[str, str]) -> tuple[bool, str]:
    """Check that an AF3 output directory exists for ingest mode."""
    out = paths["output_dir"]
    if os.path.isdir(out):
        n = len([e for e in os.listdir(out) if not e.startswith(".")])
        return True, f"AF3 output dir: {out} ({n} entries)"
    return False, (
        f"No AF3 outputs at {out}. "
        "Run AF3 locally or on GCP, then copy job folders here, "
        "or use AF3_MODE='run' to generate a small subset on Colab."
    )


def print_af3_setup_instructions(paths: Optional[dict[str, str]] = None) -> None:
    """Print Drive setup steps for Colab users."""
    paths = paths or resolve_af3_paths()
    msg = textwrap.dedent(f"""
    ═══════════════════════════════════════════════════════════════
     AlphaFold 3 on Colab — Google Drive setup (NOT GitHub)
    ═══════════════════════════════════════════════════════════════
    1. Obtain af3.bin from Google DeepMind (license required).
    2. Upload to: {paths['weights_path']}
    3. Put AF3 prediction outputs under: {paths['output_dir']}
       (one folder per protein, or *{{uniprot}}*_confidences.json + *_model.cif)
    4. Optional full DB for --run_data_pipeline: {paths['database_dir']}
       (~630 GB — impractical for full DisProt on Colab)

    GitHub cannot hide licensed multi-GB weights. Drive is the supported path.
    ═══════════════════════════════════════════════════════════════
    """)
    print(msg)


def build_protein_json(
    protein: dict,
    job_name: Optional[str] = None,
) -> dict:
    """Minimal AF3 input JSON for a single protein chain."""
    name = job_name or protein.get("uniprot_acc") or protein["id"]
    return {
        "name": name,
        "modelSeeds": [1],
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": protein["sequence"],
                },
            },
        ],
    }


def write_af3_input_jsons(
    proteins: list,
    input_dir: str,
    max_proteins: Optional[int] = None,
) -> list[str]:
    """Write per-protein AF3 input JSON files; return paths."""
    os.makedirs(input_dir, exist_ok=True)
    subset = proteins[:max_proteins] if max_proteins else proteins
    paths: list[str] = []
    for p in subset:
        job = p.get("uniprot_acc") or p["id"]
        path = os.path.join(input_dir, f"{job}.json")
        with open(path, "w") as f:
            json.dump(build_protein_json(p, job_name=job), f, indent=2)
        paths.append(path)
    return paths


def run_af3_subset_on_colab(
    proteins: list,
    paths: dict[str, str],
    alphafold3_repo: str = "/content/alphafold3",
    max_proteins: int = 25,
    run_data_pipeline: bool = False,
    timeout_s: int = 3600,
) -> dict:
    """
    Run AF3 on a small protein subset via subprocess.

    Requires alphafold3 cloned and weights on Drive. Full data pipeline
  needs ~630 GB genetic databases — default is inference-only with
    pre-built input JSONs that include MSA (user must supply *_data.json
    or enable run_data_pipeline with database_dir populated).
    """
    ok, msg = verify_af3_weights(paths)
    if not ok:
        return {"success": False, "error": msg, "n_run": 0}

    run_script = os.path.join(alphafold3_repo, "run_alphafold.py")
    if not os.path.isfile(run_script):
        return {
            "success": False,
            "error": (
                f"Clone alphafold3 to {alphafold3_repo} and install per "
                "https://github.com/google-deepmind/alphafold3 — Docker or bare-metal Linux."
            ),
            "n_run": 0,
        }

    input_dir = os.path.join(paths["drive_root"], "inputs")
    json_paths = write_af3_input_jsons(proteins, input_dir, max_proteins=max_proteins)
    os.makedirs(paths["output_dir"], exist_ok=True)

    n_ok = 0
    errors: list[str] = []
    for jp in json_paths:
        cmd = [
            "python", run_script,
            "--json_path", jp,
            "--model_dir", paths["model_dir"],
            "--output_dir", paths["output_dir"],
        ]
        if paths.get("database_dir") and os.path.isdir(paths["database_dir"]):
            cmd.extend(["--db_dir", paths["database_dir"]])
        if not run_data_pipeline:
            cmd.append("--run_data_pipeline=false")
        cmd.append("--run_inference=true")

        try:
            subprocess.run(cmd, check=True, timeout=timeout_s, capture_output=True, text=True)
            n_ok += 1
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{os.path.basename(jp)}: {exc}")

    return {
        "success": n_ok > 0,
        "n_run": n_ok,
        "n_attempted": len(json_paths),
        "output_dir": paths["output_dir"],
        "errors": errors[:5],
    }


def setup_af3_for_colab(
    mode: str = "ingest",
    drive_root: str = DEFAULT_DRIVE_ROOT,
    mount_drive: bool = True,
) -> dict:
    """
    Colab entry point: mount Drive, verify paths, return config dict.

    mode: 'off' | 'ingest' | 'run'
    """
    if mount_drive:
        mount_google_drive()

    paths = resolve_af3_paths(drive_root=drive_root)
    config = {
        "mode": mode,
        "paths": paths,
        "weights_ok": False,
        "outputs_ok": False,
        "ready": False,
    }

    if mode == "off":
        config["ready"] = True
        return config

    w_ok, w_msg = verify_af3_weights(paths)
    config["weights_ok"] = w_ok
    config["weights_message"] = w_msg

    if mode == "ingest":
        o_ok, o_msg = verify_af3_outputs(paths)
        config["outputs_ok"] = o_ok
        config["outputs_message"] = o_msg
        config["ready"] = o_ok
    elif mode == "run":
        config["ready"] = w_ok
        config["outputs_message"] = f"Outputs will be written to {paths['output_dir']}"
    else:
        config["error"] = f"Unknown AF3_MODE: {mode}"

    return config
