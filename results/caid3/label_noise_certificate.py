#!/usr/bin/env python3
"""Which CAID3 margins survive the annotation error, and which cannot.

`LabelNoise.ranking_certified` (Lean 4, sorry-free): if a predictor `P` beats
`Q` on the *truth* by more than twice the annotation noise, then it beats `Q`
on the *benchmark*. Contrapositive, which is the usable direction here: a
measured margin of at most `2·noise` certifies nothing about which method is
actually better. `perfect_predictor_penalised` is the companion — a model that
reproduces the truth residue for residue is measured to make exactly `noise`
mistakes, not zero.

The theorem is about **error counts**, not AUC, so this uses the binary column
every `.caid` file carries and every entrant supplies. Applying it to AUC would
be citing a theorem about a different quantity.

## Estimating the noise without an oracle — and the attempt that failed

The first attempt compared CAID3's own references. Disorder-PDB and
Disorder-NOX judge overlapping targets under different conventions, so a
residue they disagree about looked like the benchmark contradicting itself.

**It returned exactly zero disagreeing residues out of 58,015**, and zero again
for Binding against Binding-IDR. That is not evidence of clean labels. It is
evidence that these are not independent annotations: NOX is derived from the
same underlying assignment by a rule about *unannotated* residues, so wherever
both evaluate a residue they agree by construction. The comparison measures
nothing.

It is kept here, reported, and **explicitly refused as a noise estimate**,
because the failure mode it would otherwise produce is the worst one available:
a noise estimate of zero sets the bar at zero, and every comparison in the
field comes back "certified" — the strongest-looking possible output from a
measurement that failed. Certification is therefore gated on an estimate that
is both positive and independent, and this one is neither.

A defensible estimate needs annotations that disagree for real: MobiDB's
missing-residue call at two coverage thresholds, or per-structure coverage for
proteins with several deposited structures. Neither is in the cached data, and
this analysis reports that rather than substituting something that is not it.

## What to do without one: report the frontier, not a verdict

`RobustCertificate.certificate_under_mean_error` is the pattern. A certificate
whose input is known only to within `delta` does not become worthless; it
degrades continuously and by a computable amount, and the honest output is the
degraded bound rather than silence.

The same applies here. The annotation error rate `eps` is unknown, but
`ranking_certified` is monotone in it: every comparison certified at `eps` is
certified at every smaller rate, and the bar is exactly `2*eps*n_evaluated`. So
instead of one verdict this reports the **frontier** — how many of the field's
comparisons against the leader survive, as a function of `eps`. A reader who
believes the annotation is 1% wrong reads off one row; a reader who believes 5%
reads off another. Nothing is assumed on their behalf.

The largest `eps` at which a given comparison still certifies is its
**breakdown rate**: the annotation error rate that would have to be exceeded
before that ranking could be an artefact. Quoting it is the strongest honest
form of "this margin is real", and it needs no estimate of the noise at all.

    export ANALYSIS_SCRIPT=results/caid3/label_noise_certificate.py
    sbatch rockfish/slurm/analysis_cpu.sbatch
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import (  # noqa: E402
    evaluated_mask,
    read_reference,
    verify_composition_for,
)

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("NOISE_REFS", f"{ROOT}/caid3_official")
PREDS = os.environ.get("NOISE_PREDS", f"{ROOT}/caid3_predictions")
EXTRA = os.environ.get("NOISE_EXTRA", "")
OUT = os.environ.get("NOISE_OUT", "")


def read_binary_predictions(path: str) -> dict[str, np.ndarray]:
    """The binary column of a .caid file — column 4, the thresholded call.

    CAID's own format carries it and every entrant fills it. Read separately
    from the score column because the theorem is about error counts, and
    rethresholding someone else's scores at 0.5 would be scoring a method they
    did not submit.
    """
    out: dict[str, list] = {}
    cur = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                cur = line[1:].strip()
                out[cur] = []
                continue
            if cur is None:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                out[cur].append(np.nan)
                continue
            try:
                out[cur].append(float(parts[3]))
            except ValueError:
                out[cur].append(np.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def label_array(lab: str, mask: np.ndarray) -> np.ndarray:
    return np.frombuffer(lab.encode(), dtype=np.uint8)[mask].astype(np.int8) \
        - ord("0")


def observed_noise(ref_a, ref_b) -> dict:
    """Residues on which two official references contradict each other.

    Counted only where both evaluate the residue and the two agree on the
    protein's sequence — a length mismatch is a different chain, not a
    disagreement.
    """
    shared = set(ref_a) & set(ref_b)
    disagree = agree = 0
    per_target = {}
    for tid in sorted(shared):
        sa, la = ref_a[tid]
        sb, lb = ref_b[tid]
        if sa != sb:
            continue
        both = evaluated_mask(la) & evaluated_mask(lb)
        if not both.any():
            continue
        ya = label_array(la, both)
        yb = label_array(lb, both)
        d = int((ya != yb).sum())
        disagree += d
        agree += int(both.sum()) - d
        per_target[tid] = {"evaluated_by_both": int(both.sum()), "disagree": d}
    total = disagree + agree
    return {
        "n_shared_targets": len(shared),
        "n_comparable_targets": len(per_target),
        "residues_evaluated_by_both": total,
        "residues_disagreeing": disagree,
        "disagreement_rate": (disagree / total) if total else None,
        "per_target": per_target,
    }


def error_counts(ref, preds_dir, extra_spec, task) -> dict:
    """Symmetric-difference error count per method, over evaluated residues."""
    files = [(fn[:-5], os.path.join(preds_dir, fn))
             for fn in sorted(os.listdir(preds_dir)) if fn.endswith(".caid")]
    for item in filter(None, (s.strip() for s in extra_spec.split(","))):
        parts = item.split(":")
        prefix = parts[1] if len(parts) > 1 and parts[1] else "DisorderNet"
        label = parts[2] if len(parts) > 2 and parts[2] else prefix
        p = os.path.join(parts[0], f"{prefix}-{task}.caid")
        if os.path.isfile(p):
            files.append((label, p))

    out = {}
    for name, path in files:
        pred = read_binary_predictions(path)
        errs = n_eval = 0
        complete = True
        for tid, (_seq, lab) in ref.items():
            p = pred.get(tid)
            if p is None or len(p) != len(lab):
                complete = False
                break
            m = evaluated_mask(lab)
            y = label_array(lab, m)
            q = p[m]
            if not np.isfinite(q).all():
                complete = False
                break
            errs += int((q.astype(np.int8) != y).sum())
            n_eval += int(m.sum())
        if complete and n_eval:
            out[name] = {"errors": errs, "n_evaluated": n_eval,
                         "error_rate": errs / n_eval}
    return out


def main() -> int:
    report = {}
    refs = {}
    for task in ("disorder_pdb", "disorder_nox", "binding", "binding_idr",
                 "linker"):
        path = os.path.join(REFS, f"{task}.fasta")
        if os.path.isfile(path):
            verify_composition_for("caid3", task, path)
            refs[task] = read_reference(path)

    print("=" * 92)
    print(" Annotation disagreement between official references")
    print(" (a lower bound on annotation noise — where both agree and both are")
    print("  wrong, this sees nothing)")
    print("=" * 92)
    noise = {}
    for a, b in (("disorder_pdb", "disorder_nox"),
                 ("binding", "binding_idr")):
        if a not in refs or b not in refs:
            continue
        n = observed_noise(refs[a], refs[b])
        noise[f"{a}|{b}"] = n
        if n["disagreement_rate"] is None:
            continue
        print(f" {a} vs {b}: {n['n_comparable_targets']} comparable targets, "
              f"{n['residues_evaluated_by_both']:,} residues judged by both")
        print(f"   contradicting residues: {n['residues_disagreeing']:,} "
              f"({n['disagreement_rate']:.2%})")
    report["noise"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_target"}
                       for k, v in noise.items()}

    key = "disorder_pdb|disorder_nox"
    rate = (noise.get(key) or {}).get("disagreement_rate")
    # A zero or missing estimate must refuse, not certify. With bar = 0 every
    # non-tied comparison passes, so a failed measurement would print the
    # strongest result in the file.
    if not rate:
        print("\n" + "=" * 92)
        print(" REFUSING TO CERTIFY")
        print("=" * 92)
        print(" The internal disagreement rate is "
              f"{'0.00%' if rate == 0 else 'unavailable'}, which means these"
              " references are not")
        print(" independent annotations rather than that the labels are"
              " clean — NOX is derived")
        print(" from the same assignment, so where both judge a residue they"
              " agree by construction.")
        print(" A bar of zero would certify every comparison in the field."
              " Error counts are")
        print(" reported below for reference; none of them is certified"
              " against anything.")
        rate = None

    for task in ("disorder_pdb", "disorder_nox"):
        if task not in refs:
            continue
        ref = refs[task]
        counts = error_counts(ref, PREDS, EXTRA, task)
        if len(counts) < 5:
            continue
        order = sorted(counts, key=lambda n: counts[n]["errors"])
        n_eval = counts[order[0]]["n_evaluated"]
        best = order[0]

        print(f"\n{'=' * 92}")
        print(f" {task}: {len(counts)} methods with a complete binary "
              f"submission, {n_eval:,} evaluated residues")
        if rate is None:
            print(" error counts only — no certification, see above")
        else:
            noise_residues = rate * n_eval
            bar = 2.0 * noise_residues
            print(f" estimated noise {noise_residues:,.0f} residues "
                  f"({rate:.2%}); ranking_certified needs a margin above "
                  f"2*noise = {bar:,.0f}")
        print("=" * 92)
        head = f"{'margin vs #1':>14}" + ("   certified?" if rate else "")
        print(f" {'#':>3} {'method':<28}{'errors':>10}{'rate':>8}{head}")
        for i, n in enumerate(order[:15], 1):
            c = counts[n]
            margin = c["errors"] - counts[best]["errors"]
            mark = ""
            if rate:
                mark = "   —" if n == best else (
                    "   YES" if margin > 2.0 * rate * n_eval else "   no")
            print(f" {i:>3} {n:<28}{c['errors']:>10,}{c['error_rate']:>8.3f}"
                  f"{margin:>14,}{mark}")

        # The frontier. ranking_certified needs margin > 2*eps*n_eval, so the
        # breakdown rate of a comparison is margin/(2*n_eval): the annotation
        # error rate that would have to be exceeded before that ranking could
        # be an artefact. Monotone, so this is a complete answer for every eps
        # at once rather than a verdict at one guessed value.
        margins = [counts[n]["errors"] - counts[best]["errors"]
                   for n in order[1:]]
        breakdown = sorted(m / (2.0 * n_eval) for m in margins)
        print(f"\n certification frontier — how many of the {len(margins)} "
              f"comparisons against {best}\n survive at an assumed annotation "
              f"error rate eps:")
        print(f"   {'eps':>8}{'bar (residues)':>16}{'certified':>12}"
              f"{'of':>5}")
        for eps in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10):
            k = sum(1 for b in breakdown if b > eps)
            print(f"   {eps:>7.2%}{2.0 * eps * n_eval:>16,.0f}{k:>12}"
                  f"{len(margins):>5}")
        if breakdown:
            print(f"\n breakdown rate of the closest comparison: "
                  f"{breakdown[0]:.3%} — below that annotation error rate, "
                  f"even\n the narrowest margin in the field is certified; "
                  f"above it, that one is not.")
            print(f" median breakdown rate: {breakdown[len(breakdown)//2]:.2%}")

        row = {"n_evaluated": n_eval, "best": best, "counts": counts,
               "certified": bool(rate),
               "breakdown_rates": breakdown,
               "frontier": {str(e): sum(1 for b in breakdown if b > e)
                            for e in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03,
                                      0.05, 0.10)}}
        if rate:
            bar = 2.0 * rate * n_eval
            certified = [n for n in order[1:]
                         if counts[n]["errors"] - counts[best]["errors"] > bar]
            print(f"\n {best} is certified better than {len(certified)} of "
                  f"{len(order) - 1} other methods; the rest are inside the "
                  f"annotation noise\n and cannot be ordered by this "
                  f"benchmark, however many resamples are taken.")
            row.update({"noise_residues": rate * n_eval,
                        "certification_bar": bar,
                        "n_certified": len(certified),
                        "n_uncertifiable": len(order) - 1 - len(certified)})
        report[task] = row

    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
