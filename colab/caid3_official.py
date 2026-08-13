"""The official CAID3 references and every entrant's raw predictions.

CAID publishes more than the leaderboard. The five challenge references and the
per-residue predictions of all 117 submitted methods are served as static files
behind the results site, and the dataset repository serves each reference's
composition independently. Three consequences, all of which this module exists
to make available:

**Every benchmark becomes head-to-head.** Four of the five references were
previously reconstructed from DisProt using rules read out of CAID's own
``references.ipynb``. Only Linker reconstructed exactly. Disorder-NOX came out
as 319 targets against a true 204, Binding-IDR as 31 against 52 — different
benchmarks wearing the same name, whose scores could not honestly sit beside a
published figure. The reconstructions are obsolete.

**Comparisons become paired.** With a competitor's per-residue scores on the
same targets, "is our model better than PUNCH2" is a paired test on matched
residues, not the much weaker "our confidence interval happens to contain their
point estimate".

**Coverage becomes measurable.** CAID pools over whatever targets a method
returned, so a method that declines the hard ones is scored on an easier
benchmark. SPOT-Disorder2 ranks 4th on Disorder-PDB having skipped 21 of 319
targets; ESMDisPred-2PDB leads Disorder-NOX on 181 of 204; IPA-AF2-Linker leads
Linker on 27 of 31. Whether that changes the ordering is an empirical question,
and :func:`common_subset_leaderboard` answers it.

The composition of every downloaded reference is checked against the repository
API's independently-served counts, and :func:`verify_against_published` checks
that the published leaders reproduce from the raw files. Both currently pass:
PUNCH2 0.9552 against a published 0.955, ESMDisPred-2PDB 0.8855 against 0.885,
IPA-AF2-Linker 0.8985 against 0.897, DisoFLAG-PB 0.7760 against 0.776,
bindEmbed21IDR-rawGeneral 0.6407 against 0.641.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

CHALLENGE = 3
_SITE = "https://caid.idpcentral.org"
_STATIC = f"{_SITE}/assets/sections/challenge/static"
REFERENCE_URL = f"{_STATIC}/references/{CHALLENGE}"
PREDICTIONS_URL = f"{_STATIC}/predictions/{CHALLENGE}/predictions.zip"
DATASET_API = f"{_SITE}/dataset-repository/api/dataset"

#: The repository dataset whose composition matches the published leaderboard.
#: "CAID3" in the same API is a different, smaller set (185 proteins) and does
#: not reproduce the published figures; "CAID3 v3" is the challenge as scored.
DATASET = "CAID3 v3"

TASKS = ("disorder_pdb", "disorder_nox", "binding", "binding_idr", "linker")

#: Reference name in the dataset API, per task.
API_NAME = {
    "disorder_pdb": "Disorder PDB", "disorder_nox": "Disorder NOX",
    "binding": "Binding", "binding_idr": "Binding IDR", "linker": "Linker",
}

#: Composition as served by the dataset API, recorded so a silently changed or
#: truncated download is caught rather than scored. (targets, pos, neg, undef)
EXPECTED = {
    "disorder_pdb": (319, 31401, 67838, 61183),
    "disorder_nox": (204, 26367, 73610, 541),
    "binding": (52, 2991, 25272, 0),
    "binding_idr": (52, 2991, 4750, 20522),
    "linker": (31, 1379, 19119, 0),
}

#: Published leader per challenge, used to verify the pipeline reproduces the
#: leaderboard. Name is the prediction filename stem.
LEADERS = {
    "disorder_pdb": ("PUNCH2", 0.955, 0.928),
    "disorder_nox": ("ESMDisPred-2PDB", 0.885, 0.754),
    "binding": ("DisoFLAG-PB", 0.776, 0.245),
    "binding_idr": ("bindEmbed21IDR-rawGeneral", 0.641, 0.514),
    "linker": ("IPA-AF2-Linker", 0.897, 0.474),
}

UNEVALUATED = ord("-")


class ReferenceCompositionError(RuntimeError):
    """A reference does not match the composition CAID reports for it."""


def _fetch(url: str, dest: str, timeout: int = 180) -> str:
    tmp = dest + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dest)  # atomic: a truncated download never lands as valid
    return dest


def download_references(cache_dir: str, force: bool = False) -> dict[str, str]:
    """Fetch the five official reference FASTAs, verifying each composition."""
    out = {}
    for task in TASKS:
        path = os.path.join(cache_dir, f"{task}.fasta")
        if force or not os.path.isfile(path):
            _fetch(f"{REFERENCE_URL}/{task}.fasta", path)
        verify_composition(task, path)
        out[task] = path
    return out


def read_reference(path: str) -> dict[str, tuple[str, str]]:
    """Parse a CAID reference: header, sequence, label line of ``0``/``1``/``-``."""
    lines = [ln.strip() for ln in open(path) if ln.strip()]
    out: dict[str, tuple[str, str]] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            raise ValueError(f"{path}: expected a header at line {i + 1}")
        if i + 2 >= len(lines):
            raise ValueError(f"{path}: truncated record {lines[i]}")
        tid, seq, lab = lines[i][1:].split()[0], lines[i + 1], lines[i + 2]
        if len(seq) != len(lab):
            raise ValueError(
                f"{path}: {tid} has {len(seq)} residues but {len(lab)} labels")
        out[tid] = (seq, lab)
        i += 3
    return out


def composition(ref: dict[str, tuple[str, str]]) -> tuple[int, int, int, int]:
    lab = "".join(v[1] for v in ref.values())
    pos, neg = lab.count("1"), lab.count("0")
    return len(ref), pos, neg, len(lab) - pos - neg


def verify_composition(task: str, path: str) -> None:
    """A reference that does not match CAID's own counts is not the benchmark."""
    got = composition(read_reference(path))
    want = EXPECTED[task]
    if got != want:
        raise ReferenceCompositionError(
            f"{task}: downloaded reference is (targets, pos, neg, undef)={got}, "
            f"CAID reports {want}. Scoring against it would produce a number "
            f"that cannot be compared to the published leaderboard."
        )


def read_caid_predictions(path: str) -> dict[str, np.ndarray]:
    """Parse a ``.caid`` submission: ``>ID`` then ``pos<TAB>aa<TAB>score...``.

    A blank score field means the method declined that residue — FoldUnfold and
    NeProc do this. Those become NaN and are counted, never quietly dropped.
    """
    out: dict[str, np.ndarray] = {}
    tid, scores = None, []
    with open(path) as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            if ln.startswith(">"):
                if tid is not None:
                    out[tid] = np.asarray(scores, dtype=np.float64)
                tid, scores = ln[1:].split()[0], []
            else:
                parts = ln.split("\t")
                raw = parts[2].strip() if len(parts) > 2 else ""
                scores.append(float(raw) if raw else np.nan)
    if tid is not None:
        out[tid] = np.asarray(scores, dtype=np.float64)
    return out


def evaluated_mask(label_line: str) -> np.ndarray:
    return np.frombuffer(label_line.encode(), dtype=np.uint8) != UNEVALUATED


def per_target_arrays(
    ref: dict[str, tuple[str, str]], pred: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], dict]:
    """Labels and scores per target, kept unpooled so bootstrap can cluster.

    Targets the method did not submit, and targets whose prediction length
    disagrees with the reference, are excluded and counted. Both are coverage
    losses that make a score less comparable, not more.
    """
    ys, ss, kept = [], [], []
    missing, lenbad = [], []
    for tid, (_seq, lab) in ref.items():
        p = pred.get(tid)
        if p is None:
            missing.append(tid)
            continue
        if len(p) != len(lab):
            lenbad.append(tid)
            continue
        m = evaluated_mask(lab)
        if not m.any():
            continue
        y = np.frombuffer(lab.encode(), dtype=np.uint8)[m].astype(np.int8) - ord("0")
        s = p[m]
        ok = np.isfinite(s)
        if not ok.any() or len(np.unique(y[ok])) == 0:
            continue
        ys.append(y[ok])
        ss.append(s[ok])
        kept.append(tid)
    cov = {
        "n_reference_targets": len(ref), "n_scored_targets": len(kept),
        "missing_targets": sorted(missing), "length_mismatch_targets": sorted(lenbad),
        "coverage": len(kept) / max(len(ref), 1),
    }
    return ys, ss, kept, cov


def score_method(
    ref: dict[str, tuple[str, str]], pred: dict[str, np.ndarray],
) -> dict | None:
    """Pool the way CAID pools, and report exactly what was pooled over."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    ys, ss, _kept, cov = per_target_arrays(ref, pred)
    if not ys:
        return None
    y, s = np.concatenate(ys), np.concatenate(ss)
    if len(np.unique(y)) < 2:
        return None
    return {
        "auc": float(roc_auc_score(y, s)),
        "aps": float(average_precision_score(y, s)),
        "n_residues": int(len(y)),
        "prevalence": float(y.mean()),
        **cov,
    }


def official_leaderboard(
    task: str, refs_dir: str, preds_dir: str,
) -> list[dict]:
    """Recompute the whole challenge from raw files, ranked by AUC."""
    ref = read_reference(os.path.join(refs_dir, f"{task}.fasta"))
    rows = []
    for fn in sorted(os.listdir(preds_dir)):
        if not fn.endswith(".caid"):
            continue
        r = score_method(ref, read_caid_predictions(os.path.join(preds_dir, fn)))
        if r:
            r["method"] = fn[:-5]
            rows.append(r)
    rows.sort(key=lambda r: -r["auc"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def verify_against_published(
    refs_dir: str, preds_dir: str, tol: float = 0.0015,
) -> dict:
    """Do the published leaders reproduce from CAID's own files?

    If they do, every figure this project cites can be recomputed rather than
    transcribed, and a transcription error becomes impossible to carry forward.
    """
    out = {}
    for task, (leader, auc, aps) in LEADERS.items():
        board = official_leaderboard(task, refs_dir, preds_dir)
        hit = next((r for r in board if r["method"] == leader), None)
        if hit is None:
            out[task] = {"ok": False, "reason": f"{leader} not among predictions"}
            continue
        d_auc = hit["auc"] - auc
        out[task] = {
            "ok": abs(d_auc) <= tol,
            "leader": leader,
            "computed_auc": round(hit["auc"], 4), "published_auc": auc,
            "computed_aps": round(hit["aps"], 4), "published_aps": aps,
            "delta_auc": round(d_auc, 4),
            "computed_rank": hit["rank"],
            "coverage": round(hit["coverage"], 4),
        }
    return out


def common_subset_leaderboard(
    task: str, refs_dir: str, preds_dir: str, methods: list[str] | None = None,
) -> dict:
    """Re-rank on the targets every named method actually predicted.

    CAID scores each method on whatever it returned. A method that declines the
    targets it finds hard is therefore scored on an easier benchmark than one
    that answers everywhere, and the two numbers sit in the same column of the
    leaderboard. This restricts all methods to their common targets, so the
    ordering reflects skill rather than which targets were attempted.
    """
    ref = read_reference(os.path.join(refs_dir, f"{task}.fasta"))
    preds = {}
    for fn in sorted(os.listdir(preds_dir)):
        if not fn.endswith(".caid"):
            continue
        name = fn[:-5]
        if methods is not None and name not in methods:
            continue
        preds[name] = read_caid_predictions(os.path.join(preds_dir, fn))

    common = set(ref)
    for name, p in preds.items():
        ok = {t for t in ref if t in p and len(p[t]) == len(ref[t][1])}
        common &= ok
    sub = {t: ref[t] for t in ref if t in common}

    full_rows, sub_rows = [], []
    for name, p in preds.items():
        rf, rs = score_method(ref, p), score_method(sub, p)
        if rf:
            rf["method"] = name
            full_rows.append(rf)
        if rs:
            rs["method"] = name
            sub_rows.append(rs)
    full_rows.sort(key=lambda r: -r["auc"])
    sub_rows.sort(key=lambda r: -r["auc"])
    full_rank = {r["method"]: i for i, r in enumerate(full_rows, 1)}
    sub_rank = {r["method"]: i for i, r in enumerate(sub_rows, 1)}
    for r in sub_rows:
        r["rank_as_published"] = full_rank.get(r["method"])
        r["rank_on_common"] = sub_rank[r["method"]]
        r["rank_change"] = (r["rank_as_published"] or 0) - r["rank_on_common"]
        full = next((f for f in full_rows if f["method"] == r["method"]), None)
        r["auc_as_published"] = full["auc"] if full else None
        r["auc_change"] = (r["auc"] - full["auc"]) if full else None
    return {
        "task": task,
        "n_reference_targets": len(ref),
        "n_common_targets": len(sub),
        "n_targets_dropped": len(ref) - len(sub),
        "rows": sub_rows,
    }


def paired_bootstrap(
    task: str,
    refs_dir: str,
    preds_dir: str,
    method_a: str,
    method_b: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Compare two methods on the targets *both* predicted, resampling proteins.

    The published leaderboard scores each method on whatever it returned, so a
    difference between two published numbers can come from the targets attempted
    rather than from skill. Restricting to the pair's common targets removes
    that; resampling proteins rather than residues respects the fact that
    residues within a protein are heavily correlated, which is what makes a
    residue-level interval far too narrow.

    ``delta`` is A minus B. ``p_two_sided`` is the bootstrap proportion of
    resamples on the wrong side of zero, doubled — not a DeLong test, and not
    valid against a *published* number, only against predictions in hand.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    ref = read_reference(os.path.join(refs_dir, f"{task}.fasta"))
    pa = read_caid_predictions(os.path.join(preds_dir, f"{method_a}.caid"))
    pb = read_caid_predictions(os.path.join(preds_dir, f"{method_b}.caid"))

    ys, sa, sb = [], [], []
    for tid, (_seq, lab) in ref.items():
        a, b = pa.get(tid), pb.get(tid)
        if a is None or b is None:
            continue
        if len(a) != len(lab) or len(b) != len(lab):
            continue
        m = evaluated_mask(lab)
        if not m.any():
            continue
        y = np.frombuffer(lab.encode(), dtype=np.uint8)[m].astype(np.int8) - ord("0")
        va, vb = a[m], b[m]
        ok = np.isfinite(va) & np.isfinite(vb)
        if not ok.any():
            continue
        ys.append(y[ok])
        sa.append(va[ok])
        sb.append(vb[ok])

    if not ys:
        return {"error": "no common targets", "n_common_targets": 0}

    y_all = np.concatenate(ys)
    if len(np.unique(y_all)) < 2:
        return {"error": "common subset has a single class",
                "n_common_targets": len(ys)}

    auc_a = roc_auc_score(y_all, np.concatenate(sa))
    auc_b = roc_auc_score(y_all, np.concatenate(sb))
    aps_a = average_precision_score(y_all, np.concatenate(sa))
    aps_b = average_precision_score(y_all, np.concatenate(sb))

    rng = np.random.default_rng(seed)
    idx = np.arange(len(ys))
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        yy = np.concatenate([ys[i] for i in pick])
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(roc_auc_score(yy, np.concatenate([sa[i] for i in pick]))
                      - roc_auc_score(yy, np.concatenate([sb[i] for i in pick])))
    deltas = np.asarray(deltas)
    if deltas.size == 0:
        return {"error": "no usable bootstrap resamples",
                "n_common_targets": len(ys)}
    d = auc_a - auc_b
    # Both tails must include the boundary, and both need the +1 correction
    # (Davison & Hinkley). Using `1 - P(delta <= 0)` for the upper tail drops
    # the ties: two identical methods have every resampled delta exactly zero,
    # giving P(<=0)=1, upper=0, and p=0 — a perfect null reported as maximally
    # significant. The +1 also stops p from ever reaching 0, which is honest:
    # B resamples cannot resolve a p-value below 1/(B+1).
    n = deltas.size
    lo = (1.0 + float(np.count_nonzero(deltas <= 0.0))) / (n + 1.0)
    hi = (1.0 + float(np.count_nonzero(deltas >= 0.0))) / (n + 1.0)
    p = 2.0 * min(lo, hi)
    return {
        "task": task, "method_a": method_a, "method_b": method_b,
        "auc_a": float(auc_a), "auc_b": float(auc_b), "delta_auc": float(d),
        "aps_a": float(aps_a), "aps_b": float(aps_b),
        "delta_aps": float(aps_a - aps_b),
        "delta_ci": [float(np.percentile(deltas, 2.5)),
                     float(np.percentile(deltas, 97.5))],
        "p_two_sided": float(min(p, 1.0)),
        "n_common_targets": len(ys), "n_residues": int(len(y_all)),
        "n_boot_used": int(deltas.size),
    }
