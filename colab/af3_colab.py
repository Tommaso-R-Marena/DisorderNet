"""
AlphaFold 3 Google Colab integration (Phase 2b).

Weights and outputs live on Google Drive — NOT in this GitHub repo.
AF3 model parameters are licensed separately; af3.bin must be obtained
from Google DeepMind and uploaded to Drive by the user.

Drive layout (matches screenshot):
  MyDrive/DisorderNet/af3/
    af3.bin              # model weights (~1 GB)
    outputs/             # AF3 job outputs (ingest or run)
    inputs/              # auto-generated input JSONs
    public_databases/    # optional (~630 GB for full MSA pipeline)
    af3_run_manifest.json

Modes:
  - off:    skip AF3 (AF2-only Phase 2)
  - ingest: read precomputed AF3 outputs from Drive (recommended if batch run elsewhere)
  - run:    run AF3 on proteins missing from outputs/ (subset or all)
  - auto:   mount Drive → clone repo → verify weights → run missing → ingest all
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
import time
from typing import Optional

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/DisorderNet/af3"
DEFAULT_WEIGHTS_NAME = "af3.bin"
DEFAULT_OUTPUT_DIR_NAME = "outputs"
DEFAULT_INPUT_DIR_NAME = "inputs"
DEFAULT_DATABASE_DIR_NAME = "public_databases"
DEFAULT_MANIFEST_NAME = "af3_run_manifest.json"
DEFAULT_AF3_REPO = "/content/alphafold3"
DEFAULT_AF3_REPO_URL = "https://github.com/google-deepmind/alphafold3.git"
DEFAULT_DOCKER_IMAGE = "alphafold3"


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
      outputs/                # AF3 job outputs (ingest or run)
      inputs/                 # per-protein input JSONs
      public_databases/       # optional; ~630 GB for full pipeline
    """
    return {
        "drive_root": drive_root,
        "weights_path": os.path.join(drive_root, weights_name),
        "model_dir": drive_root,  # af3.bin lives here per DeepMind layout
        "output_dir": os.path.join(drive_root, DEFAULT_OUTPUT_DIR_NAME),
        "input_dir": os.path.join(drive_root, DEFAULT_INPUT_DIR_NAME),
        "database_dir": os.path.join(drive_root, DEFAULT_DATABASE_DIR_NAME),
        "manifest_path": os.path.join(drive_root, DEFAULT_MANIFEST_NAME),
    }


def verify_af3_weights(paths: dict[str, str]) -> tuple[bool, str]:
    """Check that AF3 weights exist on Drive."""
    wp = paths["weights_path"]
    if os.path.isfile(wp):
        size_gb = os.path.getsize(wp) / (1024 ** 3)
        return True, f"Found weights: {wp} ({size_gb:.2f} GB)"
    return False, (
        f"AF3 weights not found at {wp}. "
        "Upload af3.bin from Google DeepMind to MyDrive/DisorderNet/af3/. "
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
        "Use AF3_MODE='auto' or 'run' to generate predictions, "
        "or copy job folders from GCP/HPC into outputs/."
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
    3. AF3 predictions are written to: {paths['output_dir']}
    4. Input JSONs auto-generated at: {paths['input_dir']}
    5. Optional full DB for MSA pipeline: {paths['database_dir']}
       (~630 GB — skip with MSA-free mode in auto/run)

    Set AF3_MODE = "auto" to clone GitHub, use Drive weights, run all
    missing DisProt proteins, and ingest pLDDT for Phase 2b/3.
    ═══════════════════════════════════════════════════════════════
    """)
    print(msg)


def clone_alphafold3_repo(
    repo_dir: str = DEFAULT_AF3_REPO,
    repo_url: str = DEFAULT_AF3_REPO_URL,
    depth: int = 1,
) -> tuple[bool, str]:
    """Clone google-deepmind/alphafold3 if not present."""
    run_script = os.path.join(repo_dir, "run_alphafold.py")
    if os.path.isfile(run_script):
        return True, f"AF3 repo already at {repo_dir}"

    os.makedirs(os.path.dirname(repo_dir) or ".", exist_ok=True)
    cmd = ["git", "clone", f"--depth={depth}", repo_url, repo_dir]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        return True, f"Cloned {repo_url} → {repo_dir}"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return False, f"git clone failed: {exc}"


def docker_available() -> bool:
    """Check if Docker CLI is usable (Colab Pro may have it)."""
    try:
        result = subprocess.run(
            ["docker", "version"], capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def databases_available(paths: dict[str, str]) -> bool:
    """True if public_databases dir exists with key AF3 DB files."""
    db_dir = paths.get("database_dir", "")
    if not db_dir or not os.path.isdir(db_dir):
        return False
    markers = [
        "uniref90_2022_05.fasta",
        "mgy_clusters_2022_05.fa",
        "bfd-first_non_consensus_sequences.fasta",
    ]
    return any(os.path.exists(os.path.join(db_dir, m)) for m in markers)


def build_protein_json(
    protein: dict,
    job_name: Optional[str] = None,
    msa_free: bool = True,
) -> dict:
    """
    AF3 input JSON for a single protein chain.

    msa_free=True: empty MSAs + no templates → inference without 630 GB DBs.
    """
    name = job_name or protein.get("uniprot_acc") or protein["id"]
    protein_block: dict = {
        "id": ["A"],
        "sequence": protein["sequence"],
    }
    if msa_free:
        protein_block["unpairedMsa"] = ""
        protein_block["pairedMsa"] = ""
        protein_block["templates"] = []

    return {
        "name": name,
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
        "sequences": [{"protein": protein_block}],
    }


def write_af3_input_jsons(
    proteins: list,
    input_dir: str,
    max_proteins: Optional[int] = None,
    msa_free: bool = True,
    skip_existing: bool = True,
) -> list[str]:
    """Write per-protein AF3 input JSON files; return paths."""
    os.makedirs(input_dir, exist_ok=True)
    subset = proteins[:max_proteins] if max_proteins else proteins
    paths: list[str] = []

    for p in subset:
        job = p.get("uniprot_acc") or p["id"]
        path = os.path.join(input_dir, f"{job}.json")
        if skip_existing and os.path.isfile(path):
            paths.append(path)
            continue
        with open(path, "w") as f:
            json.dump(build_protein_json(p, job_name=job, msa_free=msa_free), f, indent=2)
        paths.append(path)
    return paths


def _output_exists_for_protein(output_dir: str, protein: dict) -> bool:
    """Check if AF3 outputs already exist for this protein."""
    from colab.af3_plddt import find_af3_output_pair

    pair = find_af3_output_pair(
        output_dir,
        protein_id=protein["id"],
        uniprot_acc=protein.get("uniprot_acc", ""),
    )
    return pair is not None


def select_proteins_for_af3(
    proteins: list,
    output_dir: str,
    require_uniprot: bool = True,
) -> tuple[list, list]:
    """
    Split proteins into (already_done, needs_run).

    Prefer proteins with UniProt accessions for stable job names.
    """
    done: list = []
    pending: list = []
    for p in proteins:
        if require_uniprot and not p.get("uniprot_acc"):
            continue
        if _output_exists_for_protein(output_dir, p):
            done.append(p)
        else:
            pending.append(p)
    return done, pending


def load_run_manifest(manifest_path: str) -> dict:
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_updated": None}


def save_run_manifest(manifest_path: str, manifest: dict) -> None:
    manifest["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _run_af3_docker(
    json_path: str,
    paths: dict[str, str],
    run_data_pipeline: bool,
    timeout_s: int,
) -> tuple[bool, str]:
    """Run one AF3 job via official Docker image."""
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{os.path.dirname(json_path)}:/root/af_input",
        "-v", f"{paths['output_dir']}:/root/af_output",
        "-v", f"{paths['model_dir']}:/root/models",
    ]
    if run_data_pipeline and databases_available(paths):
        cmd.extend(["-v", f"{paths['database_dir']}:/root/public_databases"])
    else:
        run_data_pipeline = False

    json_basename = os.path.basename(json_path)
    cmd.extend([
        DEFAULT_DOCKER_IMAGE,
        "python", "run_alphafold.py",
        f"--json_path=/root/af_input/{json_basename}",
        "--model_dir=/root/models",
        "--output_dir=/root/af_output",
        f"--run_data_pipeline={'true' if run_data_pipeline else 'false'}",
        "--run_inference=true",
    ])
    if run_data_pipeline:
        cmd.append("--db_dir=/root/public_databases")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)
        return True, ""
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc))[:500]
        return False, err
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s}s"


def _run_af3_native(
    json_path: str,
    paths: dict[str, str],
    alphafold3_repo: str,
    run_data_pipeline: bool,
    timeout_s: int,
) -> tuple[bool, str]:
    """Run one AF3 job via cloned repo (requires built AF3 environment)."""
    run_script = os.path.join(alphafold3_repo, "run_alphafold.py")
    if not os.path.isfile(run_script):
        return False, f"Missing {run_script}"

    cmd = [
        "python", run_script,
        "--json_path", json_path,
        "--model_dir", paths["model_dir"],
        "--output_dir", paths["output_dir"],
        f"--run_data_pipeline={'true' if run_data_pipeline else 'false'}",
        "--run_inference=true",
    ]
    if run_data_pipeline and databases_available(paths):
        cmd.extend(["--db_dir", paths["database_dir"]])

    env = os.environ.copy()
    env["PYTHONPATH"] = alphafold3_repo + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True,
            timeout=timeout_s, cwd=alphafold3_repo, env=env,
        )
        return True, ""
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc))[:500]
        return False, err
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s}s"


def run_af3_batch(
    proteins: list,
    paths: dict[str, str],
    alphafold3_repo: str = DEFAULT_AF3_REPO,
    max_proteins: Optional[int] = None,
    timeout_s: int = 3600,
    msa_free: bool = True,
    prefer_docker: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run AF3 on all proteins missing from outputs/.

    Uses MSA-free JSONs by default (no 630 GB DB required).
    Resumes via manifest on Drive.
    """
    ok, msg = verify_af3_weights(paths)
    if not ok:
        return {"success": False, "error": msg, "n_run": 0, "n_ok": 0}

    os.makedirs(paths["output_dir"], exist_ok=True)
    done, pending = select_proteins_for_af3(proteins, paths["output_dir"])
    if max_proteins is not None:
        pending = pending[:max_proteins]

    if not pending:
        return {
            "success": True,
            "n_run": 0,
            "n_ok": 0,
            "n_already_done": len(done),
            "message": "All eligible proteins already have AF3 outputs.",
        }

    clone_ok, clone_msg = clone_alphafold3_repo(alphafold3_repo)
    use_docker = prefer_docker and docker_available()
    if use_docker:
        img_ok, img_msg = ensure_alphafold3_docker_image()
        if not img_ok:
            use_docker = False
            clone_msg = img_msg
    use_native = not use_docker and clone_ok

    if not use_docker and not use_native:
        return {
            "success": False,
            "error": (
                "Cannot run AF3: Docker unavailable and native repo not cloned. "
                f"Clone status: {clone_msg}. "
                "Install Docker on Colab or run AF3 on GCP/HPC and use ingest mode."
            ),
            "n_run": 0,
            "n_ok": 0,
            "n_pending": len(pending),
        }

    json_paths = write_af3_input_jsons(
        pending, paths["input_dir"], msa_free=msa_free,
    )
    manifest = load_run_manifest(paths["manifest_path"])
    completed_set = set(manifest.get("completed", []))

    run_data_pipeline = databases_available(paths) and not msa_free
    n_ok = 0
    errors: list[str] = []

    if verbose:
        mode = "Docker" if use_docker else "native"
        print(f"AF3 batch: {len(pending)} proteins pending ({mode}, msa_free={msa_free})")

    for jp in json_paths:
        job = os.path.splitext(os.path.basename(jp))[0]
        if job in completed_set and _output_exists_for_protein(
            paths["output_dir"],
            {"id": job, "uniprot_acc": job, "sequence": ""},
        ):
            n_ok += 1
            continue

        if use_docker:
            success, err = _run_af3_docker(jp, paths, run_data_pipeline, timeout_s)
        else:
            success, err = _run_af3_native(
                jp, paths, alphafold3_repo, run_data_pipeline, timeout_s,
            )

        if success:
            n_ok += 1
            if job not in manifest["completed"]:
                manifest["completed"].append(job)
            save_run_manifest(paths["manifest_path"], manifest)
            if verbose:
                print(f"  ✓ {job}")
        else:
            errors.append(f"{job}: {err}")
            if job not in manifest["failed"]:
                manifest["failed"].append({"job": job, "error": err[:200]})
            save_run_manifest(paths["manifest_path"], manifest)
            if verbose:
                print(f"  ✗ {job}: {err[:120]}")

    return {
        "success": n_ok > 0,
        "n_run": len(json_paths),
        "n_ok": n_ok,
        "n_failed": len(json_paths) - n_ok,
        "n_already_done": len(done),
        "n_pending": len(pending),
        "output_dir": paths["output_dir"],
        "runner": "docker" if use_docker else "native",
        "msa_free": msa_free,
        "run_data_pipeline": run_data_pipeline,
        "clone_message": clone_msg,
        "errors": errors[:10],
    }


def run_af3_subset_on_colab(
    proteins: list,
    paths: dict[str, str],
    alphafold3_repo: str = DEFAULT_AF3_REPO,
    max_proteins: int = 25,
    run_data_pipeline: bool = False,
    timeout_s: int = 3600,
    msa_free: bool = True,
) -> dict:
    """Backward-compatible wrapper: run AF3 on a small protein subset."""
    if run_data_pipeline and not databases_available(paths):
        run_data_pipeline = False
    return run_af3_batch(
        proteins=proteins,
        paths=paths,
        alphafold3_repo=alphafold3_repo,
        max_proteins=max_proteins,
        timeout_s=timeout_s,
        msa_free=msa_free or not run_data_pipeline,
    )


def setup_af3_for_colab(
    mode: str = "auto",
    drive_root: str = DEFAULT_DRIVE_ROOT,
    mount_drive: bool = True,
    clone_repo: bool = True,
    alphafold3_repo: str = DEFAULT_AF3_REPO,
) -> dict:
    """
    Colab entry point: mount Drive, verify paths, clone repo, return config.

    mode: 'off' | 'ingest' | 'run' | 'auto'
    """
    if mount_drive:
        mount_google_drive()

    paths = resolve_af3_paths(drive_root=drive_root)

    if mode != "off":
        try:
            os.makedirs(paths["output_dir"], exist_ok=True)
            os.makedirs(paths["input_dir"], exist_ok=True)
        except OSError:
            pass  # non-Colab CI: /content/drive may not exist

    config: dict = {
        "mode": mode,
        "paths": paths,
        "weights_ok": False,
        "outputs_ok": False,
        "repo_ok": False,
        "docker_ok": docker_available(),
        "databases_ok": databases_available(paths),
        "ready": False,
    }

    if mode == "off":
        config["ready"] = True
        return config

    w_ok, w_msg = verify_af3_weights(paths)
    config["weights_ok"] = w_ok
    config["weights_message"] = w_msg

    if clone_repo:
        r_ok, r_msg = clone_alphafold3_repo(alphafold3_repo)
        config["repo_ok"] = r_ok
        config["repo_message"] = r_msg
        config["repo_dir"] = alphafold3_repo

    o_ok, o_msg = verify_af3_outputs(paths)
    config["outputs_ok"] = o_ok
    config["outputs_message"] = o_msg

    if mode == "ingest":
        config["ready"] = o_ok
    elif mode == "run":
        config["ready"] = w_ok
    elif mode == "auto":
        config["ready"] = w_ok
        config["auto_actions"] = [
            "clone_alphafold3_repo",
            "run_missing_proteins",
            "ingest_all_outputs",
        ]
    else:
        config["error"] = f"Unknown AF3_MODE: {mode}"

    return config


def run_af3_auto_pipeline(
    proteins: list,
    paths: dict[str, str],
    alphafold3_repo: str = DEFAULT_AF3_REPO,
    max_proteins: Optional[int] = None,
    timeout_s: int = 3600,
    msa_free: bool = True,
) -> dict:
    """
    Full auto pipeline: run missing AF3 jobs then report coverage.

    Returns batch result + protein coverage stats.
    """
    done, pending = select_proteins_for_af3(proteins, paths["output_dir"])
    batch = run_af3_batch(
        proteins=proteins,
        paths=paths,
        alphafold3_repo=alphafold3_repo,
        max_proteins=max_proteins,
        timeout_s=timeout_s,
        msa_free=msa_free,
    )

    done_after, pending_after = select_proteins_for_af3(proteins, paths["output_dir"])
    eligible = len([p for p in proteins if p.get("uniprot_acc")])

    return {
        **batch,
        "coverage_before": {
            "n_done": len(done),
            "n_pending": len(pending),
            "n_eligible": eligible,
        },
        "coverage_after": {
            "n_done": len(done_after),
            "n_pending": len(pending_after),
            "n_eligible": eligible,
            "fraction": len(done_after) / max(eligible, 1),
        },
    }


def setup_af3_for_colab_legacy(*args, **kwargs):
    """Alias kept for backward compatibility."""
    return setup_af3_for_colab(*args, **kwargs)
