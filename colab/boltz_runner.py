"""
Boltz-2 runner — pinned install, YAML inputs, ``boltz predict``.

Weights auto-download into ``~/.boltz`` (or ``$BOLTZ_CACHE``) on first run.
No AF3-style license file is required.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from typing import Optional

# Pin for reproducible Rockfish / CI runs (weights follow the package)
PINNED_BOLTZ_VERSION = "2.2.1"
PINNED_BOLTZ_SPEC = f"boltz[cuda]=={PINNED_BOLTZ_VERSION}"
PINNED_BOLTZ_SPEC_CPU = f"boltz=={PINNED_BOLTZ_VERSION}"


def job_name_for_protein(protein: dict) -> str:
    from colab.boltz_plddt import _sanitize_name

    acc = (protein.get("uniprot_acc") or "").strip()
    raw = acc or protein.get("id", "protein")
    return _sanitize_name(raw) or "protein"


def write_boltz_yaml(
    protein: dict,
    path: str,
    msa_mode: str = "empty",
) -> str:
    """
    Write a single-chain protein YAML for Boltz-2.

    msa_mode:
      empty     — single-sequence (fast; MSA-free Rockfish default)
      server    — omit msa key; pair with ``boltz predict --use_msa_server``
    """
    name = job_name_for_protein(protein)
    seq = protein["sequence"]
    lines = [
        "version: 1",
        "sequences:",
        "  - protein:",
        "      id: A",
        f"      sequence: {seq}",
    ]
    if msa_mode == "empty":
        lines.append("      msa: empty")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return name


def ensure_boltz_installed(
    version: str = PINNED_BOLTZ_VERSION,
    prefer_cuda: bool = True,
    upgrade: bool = False,
) -> dict:
    """
    Ensure pinned ``boltz`` is importable; ``pip install`` if missing/wrong version.

    First ``boltz predict`` then auto-downloads model weights into BOLTZ_CACHE.
    """
    report: dict = {
        "pinned_version": version,
        "already_installed": False,
        "installed_now": False,
        "spec": None,
        "error": None,
    }
    try:
        import boltz  # noqa: F401
        ver = getattr(boltz, "__version__", None)
        if ver is None:
            try:
                from importlib.metadata import version as pkg_version
                ver = pkg_version("boltz")
            except Exception:
                ver = "unknown"
        report["installed_version"] = ver
        if str(ver) == str(version) and not upgrade:
            report["already_installed"] = True
            return report
    except ImportError:
        report["installed_version"] = None

    spec = PINNED_BOLTZ_SPEC if prefer_cuda else PINNED_BOLTZ_SPEC_CPU
    if version != PINNED_BOLTZ_VERSION:
        spec = f"boltz[cuda]=={version}" if prefer_cuda else f"boltz=={version}"
    report["spec"] = spec
    cmd = [sys.executable, "-m", "pip", "install", spec]
    if upgrade:
        cmd.append("-U")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        report["installed_now"] = True
        report["already_installed"] = True
    except subprocess.CalledProcessError as exc:
        # Fallback to CPU wheels if CUDA extra fails
        if prefer_cuda:
            cpu_spec = f"boltz=={version}"
            report["spec"] = cpu_spec
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", cpu_spec],
                    check=True, capture_output=True, text=True,
                )
                report["installed_now"] = True
                report["already_installed"] = True
                report["cuda_fallback"] = True
            except subprocess.CalledProcessError as exc2:
                report["error"] = (exc2.stderr or str(exc2))[-500:]
        else:
            report["error"] = (exc.stderr or str(exc))[-500:]
    return report


def boltz_cli_available() -> bool:
    return shutil.which("boltz") is not None or _module_boltz_ok()


def _module_boltz_ok() -> bool:
    try:
        import boltz  # noqa: F401
        return True
    except ImportError:
        return False


def run_boltz_predict(
    input_path: str,
    out_dir: str,
    *,
    use_msa_server: bool = False,
    model: str = "boltz2",
    cache_dir: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 50,
    diffusion_samples: int = 1,
    timeout_s: int = 7200,
    override: bool = False,
) -> dict:
    """Invoke ``boltz predict`` (weights download automatically into cache)."""
    os.makedirs(out_dir, exist_ok=True)
    cache = cache_dir or os.environ.get("BOLTZ_CACHE") or os.path.join(
        os.path.expanduser("~"), ".boltz",
    )
    os.makedirs(cache, exist_ok=True)

    cmd = [
        "boltz", "predict", input_path,
        "--out_dir", out_dir,
        "--cache", cache,
        "--model", model,
        "--devices", str(devices),
        "--accelerator", accelerator,
        "--recycling_steps", str(recycling_steps),
        "--sampling_steps", str(sampling_steps),
        "--diffusion_samples", str(diffusion_samples),
        "--output_format", "mmcif",
    ]
    if use_msa_server:
        cmd.append("--use_msa_server")
    if override:
        cmd.append("--override")

    env = os.environ.copy()
    env["BOLTZ_CACHE"] = cache

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, env=env,
        )
    except FileNotFoundError:
        # ``python -m boltz.main`` fallback
        cmd[0:1] = [sys.executable, "-m", "boltz.main"]
        # boltz.main may use click differently — try boltz.main predict
        cmd = [sys.executable, "-m", "boltz.main", "predict", *cmd[2:]]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s, env=env,
            )
        except FileNotFoundError as exc:
            return {"success": False, "error": str(exc), "cmd": cmd}

    return {
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "cmd": cmd,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "out_dir": out_dir,
        "cache": cache,
    }


def run_boltz_batch(
    proteins: list,
    boltz_root: str,
    *,
    max_proteins: Optional[int] = None,
    msa_free: bool = True,
    use_msa_server: bool = False,
    timeout_s: int = 7200,
    pin_version: str = PINNED_BOLTZ_VERSION,
    ensure_install: bool = True,
    sampling_steps: int = 50,
    verbose: bool = True,
) -> dict:
    """
    Write YAMLs for pending proteins and run Boltz-2 predict.

    Outputs land under ``{boltz_root}/outputs/`` (Boltz adds ``predictions/``).
    """
    from colab.boltz_plddt import select_proteins_for_boltz

    paths = resolve_boltz_paths(boltz_root)
    os.makedirs(paths["input_dir"], exist_ok=True)
    os.makedirs(paths["output_dir"], exist_ok=True)

    install_report = None
    if ensure_install:
        install_report = ensure_boltz_installed(version=pin_version, prefer_cuda=True)
        if install_report.get("error") and not install_report.get("already_installed"):
            return {
                "success": False,
                "error": f"Boltz install failed: {install_report['error']}",
                "install": install_report,
                "paths": paths,
            }

    done, pending = select_proteins_for_boltz(proteins, paths["output_dir"])
    if max_proteins is not None:
        pending = pending[:max_proteins]

    if verbose:
        print(
            f"Boltz-2 batch: {len(done)} done, {len(pending)} pending "
            f"(msa_free={msa_free}, pin={pin_version})",
            flush=True,
        )

    results: list[dict] = []
    n_ok = 0
    for p in pending:
        name = job_name_for_protein(p)
        yaml_path = os.path.join(paths["input_dir"], f"{name}.yaml")
        write_boltz_yaml(
            p, yaml_path,
            msa_mode="empty" if msa_free and not use_msa_server else "server",
        )
        # Per-protein out_dir so prediction folders nest under outputs/predictions/{name}
        pred = run_boltz_predict(
            yaml_path,
            paths["output_dir"],
            use_msa_server=use_msa_server,
            cache_dir=paths["cache_dir"],
            timeout_s=timeout_s,
            sampling_steps=sampling_steps,
        )
        pred["protein_id"] = p["id"]
        pred["job_name"] = name
        results.append(pred)
        if pred.get("success"):
            n_ok += 1
        elif verbose:
            print(f"  Boltz failed {p['id']}: {pred.get('stderr_tail', '')[-300:]}", flush=True)

    return {
        "success": True,
        "n_done_before": len(done),
        "n_pending": len(pending),
        "n_ran_ok": n_ok,
        "n_ran_fail": len(pending) - n_ok,
        "results": results,
        "install": install_report,
        "paths": paths,
        "pinned_version": pin_version,
    }


def resolve_boltz_paths(boltz_root: Optional[str] = None) -> dict:
    root = boltz_root or default_boltz_root()
    cache = os.environ.get("BOLTZ_CACHE") or os.path.join(root, "cache")
    return {
        "boltz_root": root,
        "input_dir": os.path.join(root, "inputs"),
        "output_dir": os.path.join(root, "outputs"),
        "cache_dir": cache,
        "manifest_path": os.path.join(root, "boltz_run_manifest.json"),
    }


def default_boltz_root() -> str:
    return (
        os.environ.get("DISORDERNET_BOLTZ_ROOT")
        or os.environ.get("BOLTZ_ROOT")
        or os.path.join(os.path.expanduser("~"), "boltz")
    )


def print_boltz_setup_instructions(paths: Optional[dict] = None) -> None:
    paths = paths or resolve_boltz_paths()
    print(textwrap.dedent(f"""
    ═══════════════════════════════════════════════════════════════
     Boltz-2 on Rockfish — open weights (auto-download)
    ═══════════════════════════════════════════════════════════════
    Pin: {PINNED_BOLTZ_SPEC}
    Root:  {paths['boltz_root']}
    Inputs: {paths['input_dir']}
    Outputs: {paths['output_dir']}
    Cache:  {paths['cache_dir']}   # model weights land here on first run

    export DISORDERNET_BOLTZ_ROOT={paths['boltz_root']}
    export BOLTZ_CACHE={paths['cache_dir']}

    sbatch rockfish/slurm/boltz_batch.sbatch
    # or: python rockfish/run_disordernet.py boltz --boltz-mode auto
    ═══════════════════════════════════════════════════════════════
    """))


def save_boltz_manifest(batch: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    slim = {k: v for k, v in batch.items() if k != "results"}
    slim["n_result_records"] = len(batch.get("results") or [])
    with open(path, "w") as f:
        json.dump(slim, f, indent=2)
    return path
