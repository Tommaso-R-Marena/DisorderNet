#!/usr/bin/env python3
"""Full publish campaign: 650M → 3B with walltime-aware checkpoint resume.

Rockfish limits (ARCH docs, 2026):
  - a100 / ica100 GPU partitions: max **72 h** (3 days), not 48 h
  - shared CPU partition: max **36 h** (sinfo often shows 1-12:00:00)

GPU training already resumes at the next unfinished fold via ``cv_progress.json``.
This module adds the missing Slurm-level loop: when jobs leave the queue (completed,
TIMEOUT, or cancelled deps), inspect workdirs and either resubmit incomplete phases
from the same checkpoint directories or advance 650M → 3B → done.

Designed to run under ``nohup`` / ``tmux`` on a login node (like a lab poller).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rockfish.publish_submit import (  # noqa: E402
    cmd_submit_3b,
    cmd_submit_650m,
)
from rockfish.utils import (  # noqa: E402
    ensure_repo_on_path,
    mail_sbatch_args,
    run_specs_3b,
    run_specs_650m,
    utc_stamp,
)

ensure_repo_on_path()

DEFAULT_MAIL = "marenatommaso@gmail.com"
CAMPAIGN_VERSION = 1
TERMINAL_OK = {"COMPLETED"}
TERMINAL_BAD = {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL"}
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}


@dataclass
class JobState:
    job_id: str
    state: str
    exit_code: str = "0:0"

    @property
    def active(self) -> bool:
        return self.state in ACTIVE

    @property
    def ok(self) -> bool:
        return self.state in TERMINAL_OK

    @property
    def bad(self) -> bool:
        return self.state in TERMINAL_BAD or (
            self.state in TERMINAL_OK and self.exit_code not in ("0:0", "0")
        )


def checkpoint_dir_for_spec(root: Path, spec: dict) -> Path:
    return Path(root) / spec["label"] / spec["checkpoint_subdir"]


def count_completed_folds(ckpt_dir: Path) -> int:
    progress = ckpt_dir / "cv_progress.json"
    if not progress.is_file():
        return 0
    try:
        payload = json.loads(progress.read_text())
    except Exception:
        return 0
    return len(payload.get("fold_results", []))


def pipeline_artifacts_ready(ckpt_dir: Path) -> bool:
    """Heuristic: CV finished and key postprocess artifacts exist."""
    needed = (
        ckpt_dir / "cv_summary.json",
        ckpt_dir / "cv_progress.json",
    )
    if not all(p.is_file() for p in needed):
        return False
    # Prefer strict go/no-go files when present; accept either naming under ckpt.
    go_no_go = (
        ckpt_dir / "sota_postprocess_report.json",
        ckpt_dir / "structure_distrust_benchmark.json",
    )
    return all(p.is_file() for p in go_no_go)


def package_ready(root: Path) -> bool:
    pkg = Path(root) / "publish_package" / "PACKAGE_README.md"
    return pkg.is_file()


def phase_progress(root: Path, kind: str, include_clean: bool = True) -> dict:
    specs = run_specs_650m(root, include_clean=include_clean) if kind == "650m" else run_specs_3b(
        root, include_clean=include_clean
    )
    per = []
    folds_done = 0
    folds_total = 0
    for spec in specs:
        ckpt = checkpoint_dir_for_spec(root, spec)
        n = count_completed_folds(ckpt)
        # ultra defaults to 5 folds
        total = 5
        folds_done += n
        folds_total += total
        per.append(
            {
                "label": spec["label"],
                "folds_done": n,
                "folds_total": total,
                "pipeline_ready": pipeline_artifacts_ready(ckpt),
                "checkpoint_dir": str(ckpt),
            }
        )
    return {
        "kind": kind,
        "root": str(root),
        "package_ready": package_ready(root),
        "folds_done": folds_done,
        "folds_total": folds_total,
        "fraction": (folds_done / folds_total) if folds_total else 0.0,
        "runs": per,
    }


def parse_sacct(text: str) -> dict[str, JobState]:
    """Parse ``sacct -nP -o JobID,State,ExitCode`` output (batch lines only)."""
    out: dict[str, JobState] = {}
    for line in (text or "").strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        jid, state = parts[0], parts[1]
        exit_code = parts[2] if len(parts) > 2 else "0:0"
        if "." in jid:
            continue  # skip .batch / .extern
        out[jid] = JobState(job_id=jid, state=state, exit_code=exit_code)
    return out


def query_jobs(
    job_ids: list[str],
    *,
    runner: Optional[Callable[[list[str]], str]] = None,
) -> dict[str, JobState]:
    ids = [j for j in job_ids if j and not str(j).startswith("DRYRUN")]
    if not ids:
        return {}
    if runner is not None:
        return parse_sacct(runner(ids))
    cmd = [
        "sacct", "-nP",
        "-j", ",".join(ids),
        "-o", "JobID,State,ExitCode",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return {}
    return parse_sacct(proc.stdout or "")


def user_has_active_jobs(
    user: Optional[str] = None,
    *,
    runner: Optional[Callable[[], str]] = None,
) -> bool:
    if runner is not None:
        text = runner()
    else:
        cmd = ["squeue", "-h", "-u", user or os.environ.get("USER", ""), "-o", "%i"]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            text = proc.stdout or ""
        except FileNotFoundError:
            return False
    return bool(text.strip())


def new_campaign(
    *,
    results_root: Path,
    stamp: Optional[str] = None,
    gpu_account: str = "sfried3_gpu",
    qos: str = "qos_gpu",
    mail_user: str = DEFAULT_MAIL,
    include_clean: bool = True,
    partition_3b: Optional[str] = None,
    max_resubmits: int = 128,
) -> dict:
    stamp = stamp or utc_stamp()
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    os.environ["DISORDERNET_RESULTS"] = str(results_root)
    root_650 = results_root / f"publish_650m_{stamp}"
    root_3b = results_root / f"publish_3b_{stamp}"
    return {
        "version": CAMPAIGN_VERSION,
        "stamp": stamp,
        "mail_user": mail_user,
        "gpu_account": gpu_account,
        "qos": qos,
        "include_clean": include_clean,
        "partition_3b": partition_3b,
        "max_resubmits": max_resubmits,
        "resubmit_count": 0,
        "status": "pending",
        "phases": [
            {
                "kind": "650m",
                "root": str(root_650),
                "status": "pending",
                "job_ids": {},
            },
            {
                "kind": "3b",
                "root": str(root_3b),
                "status": "pending",
                "job_ids": {},
            },
        ],
    }


def save_campaign(path: Path, campaign: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(campaign, indent=2) + "\n")


def load_campaign(path: Path) -> dict:
    return json.loads(path.read_text())


def _submit_kind(campaign: dict, phase: dict, *, dry_run: bool = False) -> dict:
    kind = phase["kind"]
    ns = argparse.Namespace(
        account=campaign["gpu_account"],
        qos=campaign.get("qos") or "qos_gpu",
        root_workdir=phase["root"],
        package_dir=str(Path(phase["root"]) / "publish_package"),
        package_id=f"publish_{kind}_{campaign['stamp']}",
        stamp=campaign["stamp"],
        no_clean=not campaign.get("include_clean", True),
        no_strict_package=False,
        dry_run=dry_run,
        partition=campaign.get("partition_3b") if kind == "3b" else None,
    )
    os.environ["DISORDERNET_MAIL_USER"] = campaign.get("mail_user") or DEFAULT_MAIL
    # publish_submit reads mail via mail_sbatch_args in submit path after our patch
    if kind == "650m":
        cmd_submit_650m(ns)
    else:
        cmd_submit_3b(ns)
    summary_path = Path(phase["root"]) / "submit_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        phase["job_ids"] = summary.get("job_ids", {})
    phase["status"] = "running"
    return phase


def needs_resubmit(phase: dict, states: dict[str, JobState]) -> bool:
    if package_ready(Path(phase["root"])):
        return False
    if not phase.get("job_ids"):
        return True
    if not states:
        # jobs vanished from accounting window — treat as need inspect/resubmit
        return True
    any_bad = False
    any_active = False
    for jid in phase["job_ids"].values():
        st = states.get(str(jid))
        if st is None:
            continue
        if st.active:
            any_active = True
        if st.bad or st.state == "TIMEOUT":
            any_bad = True
    if any_active:
        return False
    # idle queue for this phase and package missing
    return True if any_bad or not package_ready(Path(phase["root"])) else False


def advance_campaign(
    campaign: dict,
    *,
    dry_run: bool = False,
    job_query: Optional[Callable[[list[str]], dict[str, JobState]]] = None,
) -> dict:
    """One decision step: start / resubmit / mark done. Mutates campaign."""
    for phase in campaign["phases"]:
        root = Path(phase["root"])
        if package_ready(root):
            phase["status"] = "done"
            continue
        if phase["status"] == "done":
            continue

        jids = [str(j) for j in phase.get("job_ids", {}).values()]
        states = (job_query(jids) if job_query else query_jobs(jids)) if jids else {}

        if any(st.active for st in states.values()):
            phase["status"] = "running"
            campaign["status"] = "running"
            return campaign

        if phase["status"] == "pending" or needs_resubmit(phase, states):
            if campaign["resubmit_count"] >= campaign.get("max_resubmits", 128):
                phase["status"] = "failed"
                campaign["status"] = "failed"
                return campaign
            if phase.get("job_ids"):
                campaign["resubmit_count"] += 1
            _submit_kind(campaign, phase, dry_run=dry_run)
            campaign["status"] = "running"
            return campaign

    if all(p.get("status") == "done" for p in campaign["phases"]):
        campaign["status"] = "done"
    return campaign


def estimate_eta_seconds(progress: dict, elapsed_s: float) -> Optional[float]:
    frac = float(progress.get("fraction") or 0.0)
    if frac <= 0.02 or elapsed_s < 60:
        return None
    # crude: remaining ≈ elapsed * (1-frac)/frac, plus package buffer
    return elapsed_s * (1.0 - frac) / frac + 3600.0


def run_watchdog(
    campaign_path: Path,
    *,
    poll_seconds: int = 600,
    dry_run: bool = False,
    max_cycles: Optional[int] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    job_query: Optional[Callable[[list[str]], dict[str, JobState]]] = None,
    active_jobs_fn: Optional[Callable[[], bool]] = None,
) -> int:
    """Poll until campaign status is done/failed. Returns process exit code."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    campaign = load_campaign(campaign_path)
    t0 = time.time()
    cycles = 0
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=100, desc="publish campaign", unit="%")

    log_path = campaign_path.with_suffix(".log")

    def log(msg: str) -> None:
        line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log(f"Watchdog start campaign={campaign_path} poll={poll_seconds}s")

    while True:
        cycles += 1
        campaign = load_campaign(campaign_path)
        campaign = advance_campaign(campaign, dry_run=dry_run, job_query=job_query)
        save_campaign(campaign_path, campaign)

        # progress from current non-done phase
        cur = next((p for p in campaign["phases"] if p["status"] != "done"), None)
        if cur is None:
            log("Campaign complete — both 650M and 3B packages ready.")
            if pbar is not None:
                pbar.n = 100
                pbar.refresh()
                pbar.close()
            return 0

        prog = phase_progress(
            Path(cur["root"]), cur["kind"], include_clean=campaign.get("include_clean", True)
        )
        # overall: 650m = 0-50%, 3b = 50-100%
        base = 0.0 if cur["kind"] == "650m" else 50.0
        overall = base + 50.0 * float(prog["fraction"])
        if package_ready(Path(cur["root"])):
            overall = 50.0 if cur["kind"] == "650m" else 100.0
        if pbar is not None:
            pbar.n = min(99.0, overall)
            eta = estimate_eta_seconds(prog, time.time() - t0)
            pbar.set_postfix(
                kind=cur["kind"],
                folds=f"{prog['folds_done']}/{prog['folds_total']}",
                resubmits=campaign["resubmit_count"],
                eta_h=(f"{eta/3600:.1f}" if eta else "?"),
            )
            pbar.refresh()

        log(
            f"status={campaign['status']} phase={cur['kind']} "
            f"folds={prog['folds_done']}/{prog['folds_total']} "
            f"package={prog['package_ready']} resubmits={campaign['resubmit_count']}"
        )

        if campaign["status"] == "failed":
            log("Campaign failed (max resubmits or hard error).")
            if pbar is not None:
                pbar.close()
            return 1

        if max_cycles is not None and cycles >= max_cycles:
            log(f"Stopping after max_cycles={max_cycles}")
            if pbar is not None:
                pbar.close()
            return 0

        # Sleep while cluster still has our jobs, or briefly if we just resubmitted
        active = active_jobs_fn() if active_jobs_fn else user_has_active_jobs()
        if active or campaign["status"] == "running":
            sleep_fn(poll_seconds)
        else:
            # No jobs and not done — advance_campaign will resubmit next loop
            sleep_fn(min(60, poll_seconds))


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DisorderNet full publish campaign (650M→3B)")
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create campaign state JSON")
    init.add_argument("--campaign", type=Path, required=True)
    init.add_argument("--results-root", type=Path, default=None,
                      help="Defaults to $DISORDERNET_RESULTS or ~/disordernet_runs")
    init.add_argument("--stamp", default=None)
    init.add_argument("--account", default="sfried3_gpu")
    init.add_argument("--qos", default="qos_gpu")
    init.add_argument("--mail-user", default=DEFAULT_MAIL)
    init.add_argument("--no-clean", action="store_true")
    init.add_argument("--partition-3b", default=None,
                      help="Optional GPU partition for 3B (e.g. ica100)")
    init.add_argument("--max-resubmits", type=int, default=128)

    watch = sub.add_parser("watch", help="Run the resume poller until done")
    watch.add_argument("--campaign", type=Path, required=True)
    watch.add_argument("--poll-seconds", type=int, default=600)
    watch.add_argument("--dry-run", action="store_true")
    watch.add_argument("--max-cycles", type=int, default=None)

    step = sub.add_parser("step", help="Single advance decision (for tests/debug)")
    step.add_argument("--campaign", type=Path, required=True)
    step.add_argument("--dry-run", action="store_true")

    st = sub.add_parser("status", help="Print progress JSON")
    st.add_argument("--campaign", type=Path, required=True)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_cli().parse_args(argv)
    if args.command == "init":
        results = args.results_root
        if results is None:
            results = Path(os.environ.get(
                "DISORDERNET_RESULTS", str(Path.home() / "disordernet_runs")
            ))
        camp = new_campaign(
            results_root=results,
            stamp=args.stamp,
            gpu_account=args.account,
            qos=args.qos,
            mail_user=args.mail_user,
            include_clean=not args.no_clean,
            partition_3b=args.partition_3b,
            max_resubmits=args.max_resubmits,
        )
        save_campaign(args.campaign, camp)
        print(f"Wrote {args.campaign}")
        print(json.dumps(camp, indent=2))
        return 0
    if args.command == "step":
        camp = load_campaign(args.campaign)
        camp = advance_campaign(camp, dry_run=args.dry_run)
        save_campaign(args.campaign, camp)
        print(json.dumps(camp, indent=2))
        return 0
    if args.command == "status":
        camp = load_campaign(args.campaign)
        out = {"campaign_status": camp.get("status"), "resubmit_count": camp.get("resubmit_count"), "phases": []}
        for phase in camp["phases"]:
            out["phases"].append(
                phase_progress(
                    Path(phase["root"]),
                    phase["kind"],
                    include_clean=camp.get("include_clean", True),
                )
            )
        print(json.dumps(out, indent=2))
        return 0
    if args.command == "watch":
        return run_watchdog(
            args.campaign,
            poll_seconds=args.poll_seconds,
            dry_run=args.dry_run,
            max_cycles=args.max_cycles,
        )
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
