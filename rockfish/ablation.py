#!/usr/bin/env python3
"""Ablation harness: what each lever buys, and what it costs.

Every arm is a full CV run under one changed factor, launched as its own Slurm
job so cost is measured by the scheduler rather than estimated. The collector
reads accuracy from the run's own reports and cost from ``sacct``, then emits a
table with both.

Design rules that keep the comparison honest:

* **One factor per arm.** Arms differ from the baseline in exactly one field, so
  a delta is attributable. Compound arms are allowed but flagged as such.
* **Shared seed and folds.** Every arm uses the same seed and fold count. Where
  the protein set is identical, the homology clustering is identical too, so
  fold membership matches and paired statistics are meaningful.
* **Accuracy comes from the corrected metrics.** ``sota_postprocess_report.json``
  now runs through out-of-fold stacking, leave-one-fold-out isotonic and
  evidence-masked metrics, so arms are compared on numbers that are not
  self-scored.
* **Cost is billed, not guessed.** GPU-hours come from AllocTRES x Elapsed, which
  is what the allocation is actually charged.

Label-source arms change the *task*, not just the model. A pdb_missing arm is
not comparable to a disprot arm on DisProt CV AUC -- they predict different
things. The only cross-source comparison that means anything is on a common
external benchmark, which is why every arm also reports its CAID3 number.

Usage
-----
    python rockfish/ablation.py plan                     # show the matrix
    python rockfish/ablation.py submit --arms baseline,pdb_missing
    python rockfish/ablation.py collect --root ~/disordernet_runs_rigor/ablation_<stamp>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rockfish.utils import (  # noqa: E402
    PHASE_SBATCH,
    default_results_root,
    ensure_repo_on_path,
    env_defaults,
    git_revision,
    mail_sbatch_args,
    sbatch_export_keys,
    submit_sbatch,
    utc_stamp,
)

ensure_repo_on_path()


@dataclass
class Arm:
    """One ablation arm: a named single-factor change from the baseline."""

    name: str
    description: str
    hypothesis: str
    env: dict = field(default_factory=dict)
    partition: str = "ica100"
    gpu_mem: str = "64000M"
    # Arms that change what the model predicts cannot be compared to the
    # baseline on DisProt CV AUC; only on a shared external benchmark.
    changes_task: bool = False
    compound: bool = False


BASELINE_ENV = {
    "PROFILE": "ultra",
    "BACKBONE": "650M",
    "STAGE": "pipeline",
    "SEED": "42",
    "DISORDERNET_LABEL_SOURCE": "disprot",
}

# Measured evidenced-residue counts per label source (rockfish/slurm/_label_probe.py,
# global PDB-coverage selector). Used to hold optimizer steps constant across arms.
# These are measurements, and the epoch budget is derived from them, so they are
# load-bearing: DISORDERNET_EXPECTED_RESIDUES ships the value to the runner,
# which refuses to train if the label set it builds disagrees by >25%. Update
# here only from an observed count, never by estimate.
#
# The pdb_missing and union figures are for the cross-organism selector
# (--mobidb-global). Under the human reference proteome the same source yields
# 1,218,504 residues from 3,388 proteins — 5.4x smaller — which is what an arm
# silently got when the global flag was declared but never read.
EVIDENCED_RESIDUES = {
    "disprot": 988_872,          # 2,340 proteins after CAID leak-free filtering
    "mobidb_curated": 781_386,   # 1,721 proteins
    "pdb_missing": 6_611_600,    # 19,819 proteins, global PDB-coverage selector
    "union": 6_856_167,          # 19,908 proteins, global PDB-coverage selector
}
BASELINE_EPOCHS = 35

# Measured: two identical runs of fold 1 (same seed, same split, same code)
# gave max-epoch AUC 0.7651 and 0.7421. cudnn.benchmark and non-deterministic
# GPU kernels let trajectories diverge over ~30 epochs, and the gap is
# comparable to the between-fold spread. Used as the noise floor when an
# ablation has no replicates of its own.
MEASURED_RERUN_SD = 0.023

# Every key any arm can set, with its neutral (baseline) value. Each is written
# explicitly for each arm so an omitted key cannot inherit a neighbouring arm's
# setting — see the note in cmd_submit.
ABLATION_KEY_DEFAULTS = {
    # Deterministic kernels for every arm. Autotuned cuDNN plus TF32 let
    # identical reruns diverge by ~0.023 AUC over ~30 epochs, which is larger
    # than most effects in this matrix — an ablation that cannot reproduce
    # itself cannot attribute a delta. Costs some throughput; worth it here.
    "DISORDERNET_DETERMINISTIC": "1",
    "PROFILE": "ultra",
    "BACKBONE": "650M",
    "DISORDERNET_LABEL_SOURCE": "disprot",
    "DISORDERNET_MOBIDB_GLOBAL": "0",
    "RUN_NO_PLDDT_FEATURES": "0",
    "RUN_NO_HALLUC_WEIGHT": "0",
    "DISORDERNET_NUM_EPOCHS": str(BASELINE_EPOCHS),
}

# Keys computed per arm rather than declared above. They need the same
# clear-before-set treatment, since a value left over from a previous arm is
# indistinguishable from one chosen for this one.
DERIVED_ABLATION_KEYS = ("DISORDERNET_EXPECTED_RESIDUES",)


def compute_matched_epochs(label_source: str, baseline_epochs: int = BASELINE_EPOCHS) -> int:
    """Epoch budget that holds optimizer steps roughly constant across arms.

    A data-scale arm at a fixed epoch count is really two changes at once: more
    data *and* proportionally more gradient steps, so an accuracy gain cannot be
    attributed. Matching steps isolates the data effect; running longer is a
    separate, explicitly-labelled arm.

    pdb_missing carries 6.7x the evidenced residues of DisProt, so ~5 epochs
    there costs about what 35 epochs cost on DisProt — the difference between a
    ~20 GPU-hour arm and a ~150 GPU-hour one.
    """
    base = EVIDENCED_RESIDUES.get("disprot", 1)
    this = EVIDENCED_RESIDUES.get(label_source, base)
    return max(3, round(baseline_epochs * base / max(this, 1)))


ARMS: dict[str, Arm] = {
    "baseline": Arm(
        name="baseline",
        description="ultra / 650M / DisProt curated labels",
        hypothesis="Reference point. All deltas are measured against this.",
        env={},
    ),
    # ---- Lever 1: data scale -------------------------------------------------
    "mobidb_curated": Arm(
        name="mobidb_curated",
        description="MobiDB curated consensus (1,721 proteins — SMALLER than DisProt)",
        hypothesis=(
            "Not a data-scale arm: measurement showed curated consensus yields "
            "1,721 proteins / 781k evidenced residues, i.e. 0.7x DisProt, not the "
            "30x a raw proteome count suggests. Retained as a label-quality "
            "control — same definition kind, different curation pipeline — so a "
            "gain here would indicate label noise in DisProt rather than scale."
        ),
        env={"DISORDERNET_LABEL_SOURCE": "mobidb_curated"},
        # Same *kind* of label as DisProt, but a different protein set and
        # curation pipeline — so its CV AUC is measured over different proteins
        # and is not comparable to the baseline's. CAID3 is the common ground.
        changes_task=True,
    ),
    # ---- Lever 2: task-matched labels ---------------------------------------
    "pdb_missing": Arm(
        name="pdb_missing",
        description="PDB missing-residue labels (CAID3 Disorder-PDB definition)",
        hypothesis=(
            "Two effects at once, both favourable. (1) Task match: training on "
            "curated functional disorder while scoring on crystallographic "
            "disorder is a domain shift, and this removes it. (2) Scale: 19,819 "
            "proteins / 6.6M evidenced residues, 8.5x the proteins and 6.7x the "
            "evidenced residues of DisProt. Step-matched epochs keep the cost "
            "comparable, so a CAID3 gain here is the single most informative "
            "result in the matrix."
        ),
        env={
            "DISORDERNET_LABEL_SOURCE": "pdb_missing",
            # Cross-organism, not one proteome: human alone yields 3.4k
            # proteins, the global PDB-coverage set yields 19.8k.
            "DISORDERNET_MOBIDB_GLOBAL": "1",
        },
        changes_task=True,
    ),
    "union_labels": Arm(
        name="union_labels",
        description="Curated OR PDB-missing positives",
        hypothesis="Broader positive definition; tests whether the two label kinds are complementary or conflicting.",
        env={"DISORDERNET_LABEL_SOURCE": "union", "DISORDERNET_MOBIDB_GLOBAL": "1"},
        changes_task=True,
    ),
    # ---- Lever 3: pLDDT as a first-class input ------------------------------
    "no_plddt": Arm(
        name="no_plddt",
        description="pLDDT structure channel disabled",
        hypothesis=(
            "Isolates what the structure channel contributes. AF pLDDT is a "
            "strong standalone disorder signal and coverage here is 95.5%, so "
            "this should cost accuracy if the channel is doing real work."
        ),
        env={"RUN_NO_PLDDT_FEATURES": "1"},
    ),
    "no_halluc_weight": Arm(
        name="no_halluc_weight",
        description="Hallucination weighting disabled",
        hypothesis="Isolates the structure-distrust training signal from the pLDDT input channel.",
        env={"RUN_NO_HALLUC_WEIGHT": "1"},
    ),
    # ---- Lever 4: backbone scale --------------------------------------------
    "backbone_3b": Arm(
        name="backbone_3b",
        description="ESM-2 3B backbone",
        hypothesis=(
            "Capacity rather than data. Expected smaller than the data levers "
            "and ~4x the cost -- the point is to measure that trade, not assume it."
        ),
        env={"PROFILE": "ultra3b", "BACKBONE": "3B"},
        gpu_mem="64000M",
    ),
    # ---- Lever 5: capacity, in the other direction --------------------------
    "lite_frozen": Arm(
        name="lite_frozen",
        description="Frozen ESM-2 650M + ~1.9M-parameter dilated head (profile 'lite')",
        hypothesis=(
            "The one lever pointing down instead of up. Every other arm adds "
            "data, capacity or signal; this removes capacity. ultra trains 69.9M "
            "parameters on 988k evidenced residues from 2,340 proteins under ten "
            "simultaneous regularisers, reaches train loss 0.069 against "
            "validation AUC 0.66, and is beaten by a physics GBDT (0.7454 vs "
            "0.7804) — the signature of overfitting, not of a weak backbone. "
            "Freezing the backbone and training ~1.9M head parameters tests that "
            "directly. Precedent: SETH (frozen ProtT5 + CNN) reaches 0.830 on "
            "CAID above this project's 0.8155, with no fine-tuning at all. "
            "A win reframes the whole project; a loss rules out the cheapest "
            "explanation for the ~0.08 AUC gap to ESMDisPred."
        ),
        env={"PROFILE": "lite"},
        # No backward pass through the backbone, so activation memory is a
        # fraction of ultra's and the arm is by far the cheapest in the matrix.
        # Still 40G rather than 32G: the profile raises the batch to 16 to spend
        # the freed memory on throughput, and an OOM at hour three costs far more
        # than the wait for a slightly larger allocation.
        gpu_mem="40000M",
    ),
    # ---- Compound: the combination worth trying if the levers hold ----------
    "scaled_task_matched": Arm(
        name="scaled_task_matched",
        description="PDB-missing labels at proteome scale + pLDDT channel",
        hypothesis=(
            "If data scale and task matching both pay off independently, this "
            "is the configuration that should be competitive on CAID3."
        ),
        env={"DISORDERNET_LABEL_SOURCE": "pdb_missing", "DISORDERNET_MOBIDB_GLOBAL": "1"},
        changes_task=True,
        compound=True,
    ),
}


def cmd_plan(args: argparse.Namespace) -> int:
    print(f"Ablation matrix ({len(ARMS)} arms)\n")
    for arm in ARMS.values():
        tags = []
        if arm.changes_task:
            tags.append("CHANGES TASK — compare on CAID3 only")
        if arm.compound:
            tags.append("compound")
        print(f"  {arm.name}")
        print(f"    what : {arm.description}")
        print(f"    why  : {arm.hypothesis}")
        if arm.env:
            print(f"    env  : {arm.env}")
        if tags:
            print(f"    note : {'; '.join(tags)}")
        print()
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    names = [n.strip() for n in args.arms.split(",") if n.strip()]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        print(f"ERROR: unknown arms {unknown}. Known: {sorted(ARMS)}", file=sys.stderr)
        return 2

    stamp = args.stamp or utc_stamp()
    root = Path(args.root_workdir) if args.root_workdir else default_results_root() / f"ablation_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    # --root-workdir is a literal path, not a parent to stamp under, so pointing
    # two submissions at the same root overwrites the manifest. `collect` reads
    # the manifest to find the arms, so the earlier batch would still be running
    # on the cluster with nothing left that knows how to collect it. Refuse.
    manifest_path = root / "ablation_manifest.json"
    if manifest_path.exists() and not args.overwrite_manifest:
        try:
            prior = json.loads(manifest_path.read_text())
            prior_arms = ", ".join(a["arm"] for a in prior.get("arms", []))
        except (OSError, ValueError, KeyError):
            prior_arms = "unreadable"
        print(
            f"ERROR: {manifest_path} already exists (arms: {prior_arms}).\n"
            "  Overwriting it would orphan those jobs from `collect`.\n"
            "  Use a fresh --root-workdir, drop the flag to get a stamped dir,\n"
            "  or pass --overwrite-manifest if the prior batch is finished with.",
            file=sys.stderr,
        )
        return 2
    defaults = env_defaults()
    mail = mail_sbatch_args(os.environ.get("DISORDERNET_MAIL_USER"))

    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]

    submitted = []
    for name in names:
      for seed in seeds:
        arm = ARMS[name]
        # One workdir per (arm, seed) so replicates never share checkpoints.
        label = name if len(seeds) == 1 else f"{name}_s{seed}"
        workdir = root / label
        workdir.mkdir(parents=True, exist_ok=True)
        # EVERY ablation key is set explicitly for EVERY arm, to its arm value or
        # its neutral default. os.environ persists across loop iterations and
        # sbatch_export_keys() forwards any key that is merely *set*, so an arm
        # that simply omits a key would silently inherit the previous arm's value
        # — e.g. no_plddt running at pdb_missing's 5 epochs, or inheriting its
        # RUN_NO_PLDDT_FEATURES. That would corrupt the comparison invisibly.
        env = {
            **defaults,
            **BASELINE_ENV,
            **{k: v for k, v in ABLATION_KEY_DEFAULTS.items() if k not in arm.env},
            **arm.env,
            "DISORDERNET_WORKDIR": str(workdir),
            "CHECKPOINT_SUBDIR": "checkpoints",
            "DISORDERNET_ACCOUNT": args.account,
            "SEED": str(seed),
        }

        # Hold optimizer steps constant unless the caller overrides, so a
        # data-scale arm measures the data and not a larger step budget.
        src = env.get("DISORDERNET_LABEL_SOURCE", BASELINE_ENV["DISORDERNET_LABEL_SOURCE"])
        if args.num_epochs:
            epochs = int(args.num_epochs)
        elif args.epochs_fixed:
            epochs = BASELINE_EPOCHS
        else:
            epochs = compute_matched_epochs(src)
        env["DISORDERNET_NUM_EPOCHS"] = str(epochs)
        # Ship the count the budget was computed from, so the runner can refuse
        # to train if the label set it actually builds is a different size. The
        # epoch budget and the data are coupled through this constant, and a
        # silent selector fallback once made an arm run 5 epochs on
        # DisProt-sized data while reporting itself as step-matched.
        env["DISORDERNET_EXPECTED_RESIDUES"] = str(EVIDENCED_RESIDUES.get(src, 0))
        if epochs != BASELINE_EPOCHS:
            print(
                f"    {name}: {src} carries {EVIDENCED_RESIDUES.get(src, 0):,} "
                f"evidenced residues → {epochs} epochs "
                f"(step-matched to {BASELINE_EPOCHS} on DisProt)"
            )

        # Drop stale ablation keys from the parent environment before applying
        # this arm's, so nothing survives from the previous iteration. Derived
        # keys count too: DISORDERNET_EXPECTED_RESIDUES is computed per arm and
        # is not in ABLATION_KEY_DEFAULTS, so it outlived the loop and leaked
        # into the ambient environment — where the runner's budget guard would
        # then read one arm's figure while checking another arm's label set.
        for key in (*ABLATION_KEY_DEFAULTS, *DERIVED_ABLATION_KEYS):
            os.environ.pop(key, None)
        os.environ.update(env)

        jid = submit_sbatch(
            PHASE_SBATCH,
            account=args.account,
            job_name=f"dn-abl-{label}"[:50],
            export=sbatch_export_keys(
                ("DISORDERNET_LABEL_SOURCE", "DISORDERNET_MOBIDB_PROTEOME",
                 "DISORDERNET_MOBIDB_LIMIT", "DISORDERNET_MIN_EVIDENCE",
                 "DISORDERNET_NUM_EPOCHS", "DISORDERNET_MOBIDB_GLOBAL",
                 "DISORDERNET_EXPECTED_RESIDUES", "DISORDERNET_DETERMINISTIC")
            ),
            partition=args.partition or arm.partition,
            qos=args.qos,
            dry_run=args.dry_run,
            env=env,
            extra_args=mail,
            mem=arm.gpu_mem,
        )
        submitted.append({
            "arm": name, "seed": seed, "label": label,
            "job_id": jid, "workdir": str(workdir), "env": arm.env,
        })
        print(f"  {label:24s} → job {jid}  ({arm.description})")

    manifest = {
        "stamp": stamp,
        "root": str(root),
        "git_revision": git_revision(),
        "baseline_env": BASELINE_ENV,
        "seeds": seeds,
        "arms": submitted,
        "noise_floor_note": (
            "Same-fold, same-seed reruns of this pipeline differ by ~0.023 AUC "
            "(cudnn.benchmark plus non-deterministic GPU kernels diverge over ~30 "
            "epochs), which is comparable to the between-fold spread. Any arm "
            "delta smaller than the measured seed spread is not a result."
        ),
    }
    (root / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {root / 'ablation_manifest.json'}")
    print(f"Collect with: python rockfish/ablation.py collect --root {root}")
    return 0


def _sacct_cost(job_id: str) -> dict:
    """Billed cost for a job: elapsed, CPU-hours, GPU-hours, peak RSS."""
    if not job_id or job_id.startswith("DRYRUN"):
        return {}
    try:
        out = subprocess.run(
            ["sacct", "-nP", "-j", str(job_id),
             "-o", "JobID,State,ElapsedRaw,AllocTRES,MaxRSS"],
            check=False, capture_output=True, text=True,
        ).stdout
    except FileNotFoundError:
        return {}

    state, elapsed_s, tres, max_rss = "", 0, "", ""
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid = parts[0]
        if "." not in jid:
            state = parts[1]
            elapsed_s = int(parts[2] or 0)
            tres = parts[3]
        elif jid.endswith(".batch"):
            max_rss = parts[4]

    n_gpu, n_cpu = 0, 0
    for field_ in (tres or "").split(","):
        if field_.startswith("gres/gpu="):
            n_gpu = int(field_.split("=", 1)[1] or 0)
        elif field_.startswith("cpu="):
            n_cpu = int(field_.split("=", 1)[1] or 0)

    hours = elapsed_s / 3600.0
    return {
        "state": state,
        "elapsed_hours": round(hours, 3),
        "gpu_hours": round(hours * n_gpu, 3),
        "cpu_hours": round(hours * n_cpu, 3),
        "n_gpu": n_gpu,
        "n_cpu": n_cpu,
        "max_rss": max_rss,
    }


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _arm_accuracy(workdir: Path) -> dict:
    """Pull accuracy from the run's own reports.

    Both numbers matter and they answer different questions: DisProt CV is the
    internal comparison, CAID3 is the only figure comparable across label
    sources and to published methods.
    """
    ckpt = workdir / "checkpoints"
    acc: dict = {}

    post = _read_json(ckpt / "sota_postprocess_report.json")
    if post:
        fp = post.get("final_pooled") or {}
        acc["cv_auc"] = fp.get("auc")
        acc["cv_ap"] = fp.get("ap")
        acc["cv_n_residues"] = fp.get("n_residues")

    cvs = _read_json(ckpt / "cv_summary.json")
    if cvs:
        acc["cv_fold_aucs"] = cvs.get("fold_aucs") or cvs.get("per_fold_auc")

    caid = _read_json(ckpt / "caid3_eval_report.json")
    if caid:
        pooled = caid.get("pooled") or {}
        acc["caid3_auc"] = pooled.get("auc")
        acc["caid3_ap"] = pooled.get("ap")
        acc["caid3_delta_vs_esmdispred"] = caid.get("delta_vs_esmdispred")

    leak = _read_json(ckpt / "caid_leakage_audit.json")
    if leak:
        acc["caid_leak_free"] = leak.get("leak_free")
        acc["n_train_removed_for_caid"] = (leak.get("filter") or {}).get("n_removed")

    manifest = _read_json(ckpt / "run_manifest.json")
    if manifest:
        acc["n_proteins"] = manifest.get("n_proteins")
    return acc


def _aggregate_replicates(replicates: list) -> list:
    """Collapse per-seed runs into one row per arm, carrying the spread.

    Reporting each seed separately invites quoting the best one. The spread is
    also the only thing that makes a delta interpretable: identical reruns of
    this pipeline differ by ~0.023 AUC, so an arm that gains less than its own
    seed spread has not demonstrated anything.
    """
    import statistics

    by_arm: dict = {}
    for r in replicates:
        by_arm.setdefault(r["arm"], []).append(r)

    out = []
    for arm, reps in by_arm.items():
        head = dict(reps[0])
        head["n_seeds"] = len(reps)
        head["seeds"] = [r.get("seed") for r in reps]
        for metric in ("cv_auc", "caid3_auc"):
            vals = [r[metric] for r in reps if r.get(metric) is not None]
            if not vals:
                continue
            head[metric] = round(statistics.fmean(vals), 4)
            head[f"{metric}_sd"] = (
                round(statistics.stdev(vals), 4) if len(vals) > 1 else None
            )
            head[f"{metric}_values"] = [round(v, 4) for v in vals]
        for cost in ("gpu_hours", "cpu_hours", "elapsed_hours"):
            vals = [r[cost] for r in reps if r.get(cost) is not None]
            if vals:
                head[cost] = round(sum(vals), 3)   # total spend across replicates
        states = {r.get("state") for r in reps}
        head["state"] = "COMPLETED" if states == {"COMPLETED"} else ",".join(sorted(s for s in states if s))
        out.append(head)
    return out


def cmd_collect(args: argparse.Namespace) -> int:
    root = Path(args.root)
    manifest = _read_json(root / "ablation_manifest.json")
    if not manifest:
        print(f"ERROR: no ablation_manifest.json under {root}", file=sys.stderr)
        return 2

    replicates = []
    for entry in manifest["arms"]:
        arm = ARMS.get(entry["arm"])
        replicates.append({
            "arm": entry["arm"],
            "seed": entry.get("seed"),
            "description": arm.description if arm else "",
            "changes_task": bool(arm and arm.changes_task),
            "job_id": entry["job_id"],
            **_sacct_cost(entry["job_id"]),
            **_arm_accuracy(Path(entry["workdir"])),
        })

    # Aggregate replicates so an arm is one row with a spread, not N rows that
    # invite cherry-picking the best seed.
    rows = _aggregate_replicates(replicates)

    # The publish run is already the baseline configuration, so an ablation can
    # borrow it rather than spend another ~12 GPU-hours reproducing it. Only
    # valid when that run used the same code revision — the manifest records
    # git_revision for exactly this check.
    base = next((r for r in rows if r["arm"] == "baseline"), None)
    if base is None and args.baseline_workdir:
        base = {
            "arm": "baseline (external)",
            "description": f"borrowed from {args.baseline_workdir}",
            "changes_task": False,
            "job_id": "",
            **_arm_accuracy(Path(args.baseline_workdir)),
        }
        rows.insert(0, base)
    # Noise floor: the largest observed seed spread, falling back to the value
    # measured directly on this pipeline when replicates are unavailable.
    observed_sd = [
        r[f"{m}_sd"] for r in rows for m in ("cv_auc", "caid3_auc")
        if r.get(f"{m}_sd") is not None
    ]
    noise_floor = max(observed_sd) if observed_sd else MEASURED_RERUN_SD
    single_seed = all((r.get("n_seeds") or 1) < 2 for r in rows)

    for r in rows:
        if base and r is not base:
            for metric in ("cv_auc", "caid3_auc"):
                if r.get(metric) is not None and base.get(metric) is not None:
                    d = round(r[metric] - base[metric], 4)
                    r[f"delta_{metric}"] = d
                    # A delta inside the noise floor is not a result, and the
                    # table must say so rather than leave it to the reader.
                    r[f"delta_{metric}_exceeds_noise"] = bool(abs(d) > 2 * noise_floor)
            if r.get("gpu_hours") and base.get("gpu_hours"):
                r["gpu_hours_vs_baseline"] = round(r["gpu_hours"] / base["gpu_hours"], 2)

    out = {
        "root": str(root),
        "git_revision": manifest.get("git_revision"),
        "baseline_env": manifest.get("baseline_env"),
        "noise_floor_auc": round(noise_floor, 4),
        "noise_floor_source": "observed seed spread" if observed_sd else "measured rerun sd",
        "single_seed_warning": (
            "Every arm ran one seed. Identical reruns of this pipeline differ by "
            f"~{MEASURED_RERUN_SD} AUC, so no delta below ~{2 * MEASURED_RERUN_SD:.3f} "
            "is interpretable. Re-run with --seeds 42,43,44 before claiming an "
            "improvement."
        ) if single_seed else None,
        "rows": rows,
        "interpretation": {
            "cv_auc": "DisProt homology-CV. Comparable only across arms with the same label source.",
            "caid3_auc": "CAID3 Disorder-PDB. The only figure comparable across label sources and to published methods.",
            "gpu_hours": "Billed GPU-hours from sacct AllocTRES x Elapsed.",
        },
    }
    (root / "ablation_results.json").write_text(json.dumps(out, indent=2) + "\n")
    _print_table(rows, noise_floor)
    print(f"\n  noise floor (AUC): ±{noise_floor:.4f}   "
          f"({'observed seed spread' if observed_sd else 'measured rerun sd'})")
    if single_seed:
        print(
            "  WARNING: single seed per arm. Identical reruns of this pipeline "
            f"differ by ~{MEASURED_RERUN_SD} AUC, so deltas below "
            f"~{2 * MEASURED_RERUN_SD:.3f} mean nothing. Use --seeds 42,43,44."
        )
    print(f"\nWrote {root / 'ablation_results.json'}")
    return 0


def _print_table(rows: list, noise_floor: Optional[float] = None) -> None:
    def fmt(v, spec=".4f"):
        if v is None:
            return "—"
        try:
            return format(v, spec)
        except (TypeError, ValueError):
            return str(v)

    hdr = (
        f"{'arm':<22} {'state':<10} {'CV AUC':>8} {'ΔCV':>8} "
        f"{'CAID3':>8} {'ΔCAID3':>8} {'GPU·h':>7} {'×cost':>6}"
    )
    print("\n" + hdr)
    print("─" * len(hdr))
    for r in rows:
        flag = " *" if r.get("changes_task") else ""
        print(
            f"{r['arm']:<22} {str(r.get('state', '—')):<10} "
            f"{fmt(r.get('cv_auc')):>8} {fmt(r.get('delta_cv_auc'), '+.4f'):>8} "
            f"{fmt(r.get('caid3_auc')):>8} {fmt(r.get('delta_caid3_auc'), '+.4f'):>8} "
            f"{fmt(r.get('gpu_hours'), '.1f'):>7} {fmt(r.get('gpu_hours_vs_baseline'), '.2f'):>6}"
            f"{flag}"
        )
    if any(r.get("changes_task") for r in rows):
        print(
            "\n  * changes the prediction task — its CV AUC is NOT comparable to "
            "the baseline's.\n    Compare those arms on CAID3 only."
        )
    flagged = [
        r["arm"] for r in rows
        if r.get("delta_caid3_auc") is not None
        and r.get("delta_caid3_auc_exceeds_noise") is False
    ]
    if flagged:
        print(
            "\n  Within noise (delta below 2x the noise floor) — not a result: "
            + ", ".join(flagged)
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rockfish/ablation.py",
        description="Submit and collect DisorderNet ablation arms with cost accounting",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="Print the ablation matrix and hypotheses").set_defaults(
        func=cmd_plan
    )

    sp = sub.add_parser("submit", help="Submit selected arms")
    sp.add_argument("--arms", required=True, help="Comma-separated arm names (see: plan)")
    sp.add_argument("--account", default=os.environ.get("DISORDERNET_ACCOUNT", "sfried3_gpu"))
    sp.add_argument("--qos", default=os.environ.get("DISORDERNET_GPU_QOS", "qos_gpu"))
    sp.add_argument("--partition", default=None)
    sp.add_argument("--root-workdir", default=None)
    sp.add_argument("--stamp", default=None)
    sp.add_argument(
        "--overwrite-manifest", action="store_true",
        help="Replace an existing manifest in --root-workdir. Only safe once the "
             "batch it describes has been collected — `collect` reads the "
             "manifest to find the arms, so replacing it orphans those jobs.",
    )
    sp.add_argument("--num-epochs", type=int, default=None,
                    help="Force an epoch budget for every arm")
    sp.add_argument("--epochs-fixed", action="store_true",
                    help="Use the profile epoch budget as-is instead of "
                         "step-matching data-scale arms (costs ~8x on pdb_missing)")
    sp.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seeds per arm. Replicates measure the noise "
             "floor: this pipeline varies by ~0.023 AUC between identical "
             "runs, so a single-seed delta is uninterpretable. Use at least "
             "3 for any claimed improvement.",
    )
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_submit)

    sc = sub.add_parser("collect", help="Assemble the results table")
    sc.add_argument("--root", required=True)
    sc.add_argument(
        "--baseline-workdir",
        default=None,
        help="Use an existing run as the baseline row instead of submitting a "
             "duplicate arm (e.g. the publish run, which is already the "
             "baseline configuration). Only valid if it used the same code revision.",
    )
    sc.set_defaults(func=cmd_collect)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
