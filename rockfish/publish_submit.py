#!/usr/bin/env python3
"""
CLI: submit Rockfish publish bundles (650M or 3B) and package results.

Examples:
  python rockfish/publish_submit.py submit-650m --account sfried3
  python rockfish/publish_submit.py submit-3b --account sfried3
  python rockfish/publish_submit.py package --root-workdir ~/…/publish_650m_… --kind 650m --strict
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Ensure the repo root is importable when run as a script (`python rockfish/publish_submit.py`),
# where sys.path[0] is the rockfish/ dir, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rockfish.package_publish_results import (  # noqa: E402
    PackageIncompleteError,
    assemble_publish_package,
    build_runs_for_kind,
)
from rockfish.utils import (  # noqa: E402
    PACKAGE_SBATCH,
    PHASE_SBATCH,
    REPO_ROOT,
    RunSpec,
    default_publish_root,
    ensure_bundle_dirs,
    ensure_repo_on_path,
    env_defaults,
    git_revision,
    require_account,
    run_specs_3b,
    run_specs_650m,
    sbatch_export_keys,
    submit_sbatch,
    utc_stamp,
)

ensure_repo_on_path()


def _submit_gpu_chain(
    *,
    kind: str,
    root: Path,
    specs: list[RunSpec],
    account: str,
    partition: str,
    package_dir: Path,
    package_id: str,
    dry_run: bool,
    strict_package: bool,
    qos: Optional[str] = None,
) -> dict:
    """Submit main (+ optional clean) GPU jobs, then CPU package with dependencies.

    GPU jobs (a100) are submitted with ``--qos`` (Rockfish a100 requires a GPU QOS
    such as ``qos_gpu``). The CPU package job runs on the default partition and uses
    the non-GPU account (``account`` with a trailing ``_gpu`` stripped), since GPU
    QOS/accounts are not valid there.
    """
    defaults = env_defaults()
    cpu_account = os.environ.get("DISORDERNET_CPU_ACCOUNT") or account.replace("_gpu", "")
    base_env = {
        **defaults,
        "DISORDERNET_ACCOUNT": account,
        "DISORDERNET_CPU_ACCOUNT": cpu_account,
        "DISORDERNET_PUBLISH_ROOT": str(root),
        "DISORDERNET_PACKAGE_DIR": str(package_dir),
        "PACKAGE_ID": package_id,
        "BUNDLE_KIND": kind,
        "INCLUDE_CLEAN": "1" if any(s["clean"] for s in specs) else "0",
        "PACKAGE_STRICT": "1" if strict_package else "0",
    }
    # Keep os.environ updated for nested sbatch --export=ALL consumers
    os.environ.update(base_env)
    ensure_bundle_dirs(root, specs)

    job_ids: dict[str, str] = {}
    main_jid: Optional[str] = None

    for spec in specs:
        label = spec["label"]
        is_clean = bool(spec["clean"])
        dep = f"afterok:{main_jid}" if (is_clean and main_jid) else None

        job_env = {
            **base_env,
            "PROFILE": spec["profile"],
            "BACKBONE": spec["backbone"],
            "DISORDERNET_WORKDIR": str(root / label),
            "CHECKPOINT_SUBDIR": spec["checkpoint_subdir"],
            "STAGE": "pipeline",
            "RUN_NO_HALLUC_WEIGHT": "1" if is_clean else "0",
            "RUN_NO_PLDDT_FEATURES": "1" if is_clean else "0",
        }
        os.environ.update(job_env)

        jid = submit_sbatch(
            PHASE_SBATCH,
            account=account,
            job_name=f"dn-pub-{kind}-{label}"[:50],
            export=sbatch_export_keys(),
            partition=partition,
            qos=qos,
            dependency=dep,
            dry_run=dry_run,
            env=job_env,
        )
        job_ids[label] = jid
        print(
            f"Submitted {label} → job {jid}  "
            f"(profile={spec['profile']} backbone={spec['backbone']})"
        )
        if not is_clean and main_jid is None:
            main_jid = jid

    after = "afterok:" + ":".join(job_ids.values())
    pkg_env = {
        **base_env,
        "DISORDERNET_PUBLISH_ROOT": str(root),
        "DISORDERNET_PACKAGE_DIR": str(package_dir),
        "PACKAGE_ID": package_id,
        "BUNDLE_KIND": kind,
        "INCLUDE_CLEAN": "1" if any(s["clean"] for s in specs) else "0",
        "PACKAGE_STRICT": "1" if strict_package else "0",
    }
    os.environ.update(pkg_env)
    cpu_account = pkg_env["DISORDERNET_CPU_ACCOUNT"]
    pkg_jid = submit_sbatch(
        PACKAGE_SBATCH,
        account=cpu_account,
        job_name=f"dn-pkg-{kind}",
        export=sbatch_export_keys(),
        partition=None,
        dependency=after,
        dry_run=dry_run,
        env=pkg_env,
    )
    job_ids["package"] = pkg_jid
    print(f"Submitted package → job {pkg_jid}  (after {after})")

    summary = {
        "kind": kind,
        "publish_root": str(root),
        "package_dir": str(package_dir),
        "package_id": package_id,
        "job_ids": job_ids,
        "dry_run": dry_run,
        "strict_package": strict_package,
        "account": account,
        "partition": partition,
        "git_revision": git_revision(),
        "run_caid3": base_env.get("RUN_CAID3"),
        "boltz_mode": base_env.get("BOLTZ_MODE"),
        "include_clean": any(s["clean"] for s in specs),
        "specs": [
            {
                "label": s["label"],
                "profile": s["profile"],
                "backbone": s["backbone"],
                "checkpoint_subdir": s["checkpoint_subdir"],
                "clean": s["clean"],
            }
            for s in specs
        ],
    }
    summary_path = root / "submit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}")
    print("Monitor: squeue -u $USER")
    print(f"Package (when done): {package_dir}/PACKAGE_README.md")
    return summary


def cmd_submit_650m(args: argparse.Namespace) -> int:
    account = require_account(args.account)
    stamp = args.stamp or utc_stamp()
    root = Path(args.root_workdir) if args.root_workdir else default_publish_root("650m", stamp)
    package_dir = Path(args.package_dir) if args.package_dir else root / "publish_package"
    package_id = args.package_id or f"publish_650m_{stamp}"
    include_clean = not args.no_clean
    partition = args.partition or env_defaults()["DISORDERNET_PARTITION"]
    specs = run_specs_650m(root, include_clean=include_clean)

    print("DisorderNet publish bundle — ESM-2 650M")
    print(f"  Root:    {root}")
    print(f"  Package: {package_dir}")
    print(f"  Clean:   {include_clean}")

    _submit_gpu_chain(
        kind="650m",
        root=root,
        specs=specs,
        account=account,
        partition=partition,
        package_dir=package_dir,
        package_id=package_id,
        dry_run=args.dry_run,
        strict_package=not args.no_strict_package,
        qos=args.qos,
    )
    return 0


def cmd_submit_3b(args: argparse.Namespace) -> int:
    account = require_account(args.account)
    stamp = args.stamp or utc_stamp()
    root = Path(args.root_workdir) if args.root_workdir else default_publish_root("3b", stamp)
    package_dir = Path(args.package_dir) if args.package_dir else root / "publish_package"
    package_id = args.package_id or f"publish_3b_{stamp}"
    include_clean = not args.no_clean
    partition = args.partition or env_defaults()["DISORDERNET_PARTITION"]
    specs = run_specs_3b(root, include_clean=include_clean)

    print("DisorderNet publish bundle — ESM-2 3B")
    print(f"  Root:    {root}")
    print(f"  Package: {package_dir}")
    print(f"  Clean:   {include_clean}")
    print("  Tip: if OOM on a100 40GB, use --partition ica100")

    _submit_gpu_chain(
        kind="3b",
        root=root,
        specs=specs,
        account=account,
        partition=partition,
        package_dir=package_dir,
        package_id=package_id,
        dry_run=args.dry_run,
        strict_package=not args.no_strict_package,
        qos=args.qos,
    )
    return 0


def cmd_package(args: argparse.Namespace) -> int:
    root = Path(args.root_workdir)
    kind = args.kind
    include_clean = not args.no_clean
    runs = build_runs_for_kind(root, kind, include_clean=include_clean)
    package_dir = Path(args.package_dir) if args.package_dir else root / "publish_package"
    package_id = args.package_id or f"publish_{kind}_{utc_stamp()}"
    strict = bool(args.strict)

    try:
        manifest = assemble_publish_package(
            package_dir,
            runs,
            package_id=package_id,
            strict=strict,
            kind=kind,
        )
    except PackageIncompleteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Publish package → {package_dir}")
    print(
        json.dumps(
            {
                "package_id": manifest["package_id"],
                "n_runs": len(manifest["runs"]),
                "git_revision": manifest.get("git_revision"),
                "strict": strict,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rockfish/publish_submit.py",
        description="Submit or package DisorderNet Rockfish publish bundles (650M or 3B)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_shared(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--account",
            default=os.environ.get("DISORDERNET_ACCOUNT", "sfried3"),
            help="Rockfish account (default: sfried3)",
        )
        sp.add_argument(
            "--partition",
            default=None,
            help="GPU partition (default: a100; try ica100 for 3B OOM)",
        )
        sp.add_argument(
            "--qos",
            default=os.environ.get("DISORDERNET_GPU_QOS", "qos_gpu"),
            help="GPU QOS for a100 jobs (default: qos_gpu; a100 rejects the 'normal' "
                 "QOS). The CPU package job uses the default QOS.",
        )
        sp.add_argument("--root-workdir", default=None, help="Bundle parent directory")
        sp.add_argument("--package-dir", default=None, help="Output publish_package dir")
        sp.add_argument("--package-id", default=None)
        sp.add_argument("--stamp", default=None, help="UTC stamp override for default paths")
        sp.add_argument(
            "--no-clean",
            action="store_true",
            help="Skip contamination-clean companion job",
        )
        sp.add_argument(
            "--no-strict-package",
            action="store_true",
            help="Allow package job to succeed even if required artifacts are missing",
        )
        sp.add_argument(
            "--dry-run",
            action="store_true",
            help="Print sbatch commands without submitting",
        )

    sp650 = sub.add_parser("submit-650m", help="Submit ultra 650M (+ clean) then package")
    add_shared(sp650)
    sp650.set_defaults(func=cmd_submit_650m)

    sp3b = sub.add_parser("submit-3b", help="Submit ultra3b (+ clean) then package")
    add_shared(sp3b)
    sp3b.set_defaults(func=cmd_submit_3b)

    spkg = sub.add_parser("package", help="Assemble package from an existing bundle root")
    spkg.add_argument("--root-workdir", required=True)
    spkg.add_argument("--kind", required=True, choices=["650m", "3b"])
    spkg.add_argument("--package-dir", default=None)
    spkg.add_argument("--package-id", default=None)
    spkg.add_argument("--no-clean", action="store_true")
    spkg.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail if required go/no-go artifacts are missing (default: on)",
    )
    spkg.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Allow incomplete packages",
    )
    spkg.set_defaults(func=cmd_package)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
