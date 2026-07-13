"""
Operational helpers for the IDR biology layer (v1.5+).

Caching, landscape stats, markdown/HTML reports, JSONL compare / resume,
schema validation — keep ``idr_biology_layer.py`` focused on composition.
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import os
from collections import Counter
from typing import Optional

import numpy as np


def sequence_fingerprint(sequence: str) -> str:
    return hashlib.sha1((sequence or "").encode("utf-8")).hexdigest()[:16]


def layer_record_cache_key(
    *,
    protein_id: str,
    sequence: str,
    layer_version: str,
    disorder_threshold: float,
    function_threshold: float,
    high_plddt_threshold: float,
    has_function: bool,
    has_plddt: bool,
    has_variance: bool,
    has_partners: bool,
    has_ligands: bool,
    cache_tag: str = "",
) -> str:
    payload = "|".join([
        protein_id,
        sequence_fingerprint(sequence),
        layer_version,
        f"{disorder_threshold:.4f}",
        f"{function_threshold:.4f}",
        f"{high_plddt_threshold:.1f}",
        str(int(has_function)),
        str(int(has_plddt)),
        str(int(has_variance)),
        str(int(has_partners)),
        str(int(has_ligands)),
        cache_tag or "",
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_cached_layer_record(cache_dir: str, key: str) -> Optional[dict]:
    if not cache_dir:
        return None
    path = os.path.join(cache_dir, f"{key}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_cached_layer_record(cache_dir: str, key: str, record: dict) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    with open(path, "w") as f:
        json.dump(record, f)
    return path


def proteome_landscape_summary(records: list[dict]) -> dict:
    """Compact proteome composition / role frequency summary."""
    if not records:
        return {"n_proteins": 0, "insufficient_data": True}

    fracs = [float(r.get("disorder_fraction", 0.0)) for r in records]
    bins = {
        "mostly_ordered_<0.1": sum(1 for x in fracs if x < 0.1),
        "mixed_0.1_0.4": sum(1 for x in fracs if 0.1 <= x < 0.4),
        "idr_rich_0.4_0.7": sum(1 for x in fracs if 0.4 <= x < 0.7),
        "mostly_disordered_>=0.7": sum(1 for x in fracs if x >= 0.7),
    }
    role_counts: Counter = Counter()
    cue_counts: Counter = Counter()
    for r in records:
        for seg in r.get("idr_segments", []):
            for role in seg.get("predicted_roles", []):
                role_counts[role["group"]] += 1
                for ev in role.get("evidence") or []:
                    cue_counts[ev] += 1
            for tag in (seg.get("sequence_cues") or {}).get("cue_tags", []):
                cue_counts[f"seq:{tag}"] += 1

    return {
        "n_proteins": len(records),
        "mean_disorder_fraction": round(float(np.mean(fracs)), 4),
        "median_disorder_fraction": round(float(np.median(fracs)), 4),
        "disorder_fraction_bins": bins,
        "role_call_frequency": dict(role_counts.most_common()),
        "evidence_frequency": dict(cue_counts.most_common(20)),
        "n_proteins_with_roles": sum(1 for r in records if r.get("n_role_assignments", 0) > 0),
        "n_proteins_with_hallucination": sum(
            1 for r in records
            if (r.get("hallucination") or {}).get("n_hallucinated", 0) > 0
        ),
    }


def quality_summary(records: list[dict]) -> dict:
    """Aggregate quality / quarantine counts across the proteome."""
    n_ok = n_review = n_quarantine = 0
    flag_counts: Counter = Counter()
    for r in records:
        q = r.get("quality") or {}
        sev = q.get("severity") or "ok"
        if sev == "quarantine":
            n_quarantine += 1
        elif sev == "review":
            n_review += 1
        else:
            n_ok += 1
        for fl in q.get("flags") or []:
            flag_counts[fl] += 1
    return {
        "n_ok": n_ok,
        "n_review": n_review,
        "n_quarantine": n_quarantine,
        "flag_frequency": dict(flag_counts.most_common()),
        "quarantine_fraction": (
            round(n_quarantine / len(records), 4) if records else 0.0
        ),
    }


_REQUIRED_RECORD_KEYS = (
    "layer_version",
    "protein_id",
    "length",
    "disorder_fraction",
    "n_idr_segments",
    "idr_segments",
    "n_role_assignments",
    "triage",
)


def validate_idr_layer_record(record: dict) -> list[str]:
    """Return a list of schema issues (empty = valid)."""
    issues: list[str] = []
    if not isinstance(record, dict):
        return ["record_not_dict"]
    for key in _REQUIRED_RECORD_KEYS:
        if key not in record:
            issues.append(f"missing:{key}")
    pid = record.get("protein_id")
    if not pid or not isinstance(pid, str):
        issues.append("invalid:protein_id")
    length = record.get("length")
    if not isinstance(length, int) or length < 0:
        issues.append("invalid:length")
    fr = record.get("disorder_fraction")
    if not isinstance(fr, (int, float)) or fr < -1e-6 or fr > 1.0 + 1e-6:
        issues.append("invalid:disorder_fraction")
    segs = record.get("idr_segments")
    if not isinstance(segs, list):
        issues.append("invalid:idr_segments")
    else:
        for i, seg in enumerate(segs):
            if not isinstance(seg, dict):
                issues.append(f"invalid:idr_segments[{i}]")
                continue
            for sk in ("start", "end", "length", "mean_disorder_prob", "predicted_roles"):
                if sk not in seg:
                    issues.append(f"missing:idr_segments[{i}].{sk}")
            if isinstance(seg.get("start"), int) and isinstance(seg.get("end"), int):
                if seg["start"] < 1 or seg["end"] < seg["start"]:
                    issues.append(f"invalid:idr_segments[{i}].range")
    triage = record.get("triage")
    if not isinstance(triage, dict) or "score" not in triage:
        issues.append("invalid:triage")
    quality = record.get("quality")
    if quality is not None and not isinstance(quality, dict):
        issues.append("invalid:quality")
    return issues


def validate_idr_layer_records(records: list[dict], *, max_issues: int = 50) -> dict:
    """Batch schema check for proteome exports."""
    invalid: list[dict] = []
    for rec in records:
        issues = validate_idr_layer_record(rec)
        if issues:
            invalid.append({
                "protein_id": rec.get("protein_id") if isinstance(rec, dict) else None,
                "issues": issues,
            })
            if len(invalid) >= max_issues:
                break
    return {
        "n_records": len(records),
        "n_valid": len(records) - len(invalid),
        "n_invalid": len(invalid),
        "invalid_examples": invalid[:20],
        "ok": len(invalid) == 0,
    }


def write_idr_layer_markdown(report: dict, path: str) -> str:
    """Human-readable Markdown summary for Rockfish / lab notebooks."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        f"# DisorderNet IDR Biology Layer",
        "",
        f"- **version**: `{report.get('layer_version')}`",
        f"- **proteins**: {report.get('n_proteins', 0)}",
        f"- **mean disorder fraction**: {report.get('mean_disorder_fraction', 0):.3f}",
        f"- **role assignments**: {report.get('total_role_assignments', 0)}",
        f"- **hallucinated residues**: {report.get('total_hallucinated_residues', 0)}",
        "",
        "## Thesis",
        "",
        str(report.get("thesis", "")),
        "",
        "## Non-goals",
        "",
    ]
    for ng in report.get("non_goals") or []:
        lines.append(f"- `{ng}`")

    land = report.get("landscape") or {}
    if land and not land.get("insufficient_data"):
        lines += ["", "## Proteome landscape", ""]
        lines.append(
            f"- mean / median disorder: "
            f"{land.get('mean_disorder_fraction')} / {land.get('median_disorder_fraction')}"
        )
        bins = land.get("disorder_fraction_bins") or {}
        for k, v in bins.items():
            lines.append(f"- {k}: {v}")
        roles = land.get("role_call_frequency") or {}
        if roles:
            lines += ["", "### Role call frequency", ""]
            for name, n in list(roles.items())[:10]:
                lines.append(f"- {name}: {n}")

    qual = report.get("quality") or {}
    if qual:
        lines += [
            "", "## Quality / quarantine", "",
            f"- ok={qual.get('n_ok')}  review={qual.get('n_review')}  "
            f"quarantine={qual.get('n_quarantine')}",
        ]
        for name, n in list((qual.get("flag_frequency") or {}).items())[:10]:
            lines.append(f"- `{name}`: {n}")

    rv = report.get("role_validation") or {}
    if rv.get("enabled"):
        micro = rv.get("micro") or {}
        lines += [
            "", "## Role validation (vs DisProt)", "",
            f"- precision={micro.get('precision')}  recall={micro.get('recall')}  "
            f"F1={micro.get('f1')}",
            f"- proteins with annotations: {rv.get('n_proteins_with_annotations')}",
        ]

    thr = report.get("thresholds") or {}
    tune = report.get("function_threshold_tuning") or {}
    cal = report.get("function_calibration") or {}
    if thr or tune or cal:
        lines += ["", "## Thresholds & calibration", ""]
        lines.append(
            f"- disorder={thr.get('disorder_threshold')}  "
            f"function={thr.get('function_threshold')}"
        )
        if cal.get("enabled"):
            lines.append(
                f"- OOF function temperature mean_T={cal.get('mean_temperature')}"
            )
        if tune.get("enabled"):
            lines.append(
                f"- OOF-tuned function threshold={tune.get('threshold')}  "
                f"({tune.get('metric')}={tune.get('best_score')})"
            )

    top = report.get("top_priority_proteins") or []
    if top:
        lines += [
            "", "## Top priority proteins", "",
            "| protein | score | roles | halluc | quality | reasons |",
            "|---|---:|---:|---:|---|---|",
        ]
        for row in top[:15]:
            reasons = ", ".join(row.get("reasons") or [])
            lines.append(
                f"| {row.get('protein_id')} | {row.get('triage_score')} | "
                f"{row.get('n_role_assignments')} | {row.get('n_hallucinated')} | "
                f"{row.get('quality_severity', '')} | {reasons} |"
            )

    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def write_idr_layer_html(report: dict, path: str) -> str:
    """Self-contained HTML summary for sharing layer QA outside notebooks."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    land = report.get("landscape") or {}
    qual = report.get("quality") or {}
    top = report.get("top_priority_proteins") or []
    thr = report.get("thresholds") or {}
    cal = report.get("function_calibration") or {}
    tune = report.get("function_threshold_tuning") or {}

    def esc(x) -> str:
        return html.escape(str(x if x is not None else ""))

    rows = []
    for row in top[:20]:
        reasons = ", ".join(row.get("reasons") or [])
        rows.append(
            "<tr>"
            f"<td>{esc(row.get('protein_id'))}</td>"
            f"<td>{esc(row.get('triage_score'))}</td>"
            f"<td>{esc(row.get('n_role_assignments'))}</td>"
            f"<td>{esc(row.get('n_hallucinated'))}</td>"
            f"<td>{esc(row.get('quality_severity'))}</td>"
            f"<td>{esc(reasons)}</td>"
            "</tr>"
        )
    bins_html = "".join(
        f"<li>{esc(k)}: {esc(v)}</li>"
        for k, v in (land.get("disorder_fraction_bins") or {}).items()
    )
    flags_html = "".join(
        f"<li><code>{esc(k)}</code>: {esc(v)}</li>"
        for k, v in list((qual.get("flag_frequency") or {}).items())[:12]
    )
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>DisorderNet IDR Biology Layer</title>
<style>
  :root {{
    --ink: #1a2421; --muted: #5c6b66; --bg: #f3f6f4; --card: #ffffff;
    --accent: #0f6b5c; --line: #d5e0db;
  }}
  body {{
    margin: 0; font-family: "Iowan Old Style", "Palatino Linotype", Palatino, serif;
    background:
      radial-gradient(1200px 600px at 10% -10%, #d9ece7 0%, transparent 55%),
      radial-gradient(900px 500px at 110% 0%, #e8ebd8 0%, transparent 50%),
      var(--bg);
    color: var(--ink);
  }}
  main {{ max-width: 920px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }}
  h1 {{ font-size: 2rem; letter-spacing: -0.02em; margin: 0 0 0.35rem; }}
  .brand {{ color: var(--accent); font-weight: 700; }}
  .sub {{ color: var(--muted); margin-bottom: 1.5rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
           gap: 0.75rem; margin: 1.25rem 0 1.75rem; }}
  .stat {{ background: var(--card); border: 1px solid var(--line); padding: 0.85rem 1rem; }}
  .stat b {{ display: block; font-size: 1.35rem; }}
  .stat span {{ color: var(--muted); font-size: 0.85rem; }}
  section {{ margin: 1.75rem 0; }}
  h2 {{ font-size: 1.2rem; border-bottom: 1px solid var(--line); padding-bottom: 0.35rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
  th, td {{ text-align: left; padding: 0.45rem 0.4rem; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--muted); font-weight: 600; }}
  code {{ background: #e7efec; padding: 0.05rem 0.3rem; }}
  ul {{ line-height: 1.45; }}
</style>
</head>
<body>
<main>
  <h1><span class="brand">DisorderNet</span> IDR Biology Layer</h1>
  <p class="sub">version {esc(report.get('layer_version'))} · post-structure default</p>
  <div class="grid">
    <div class="stat"><b>{esc(report.get('n_proteins', 0))}</b><span>proteins</span></div>
    <div class="stat"><b>{esc(f"{float(report.get('mean_disorder_fraction', 0)):.3f}")}</b>
      <span>mean disorder</span></div>
    <div class="stat"><b>{esc(report.get('total_role_assignments', 0))}</b><span>role calls</span></div>
    <div class="stat"><b>{esc(report.get('total_hallucinated_residues', 0))}</b>
      <span>hallucinated residues</span></div>
    <div class="stat"><b>{esc(qual.get('n_quarantine', 0))}</b><span>quarantine</span></div>
  </div>
  <section>
    <h2>Thesis</h2>
    <p>{esc(report.get('thesis', ''))}</p>
  </section>
  <section>
    <h2>Landscape</h2>
    <ul>{bins_html or '<li>insufficient data</li>'}</ul>
  </section>
  <section>
    <h2>Quality flags</h2>
    <p>ok={esc(qual.get('n_ok'))} · review={esc(qual.get('n_review'))} ·
       quarantine={esc(qual.get('n_quarantine'))}</p>
    <ul>{flags_html or '<li>none</li>'}</ul>
  </section>
  <section>
    <h2>Thresholds &amp; calibration</h2>
    <ul>
      <li>disorder={esc(thr.get('disorder_threshold'))},
          function={esc(thr.get('function_threshold'))}</li>
      <li>calibration enabled={esc(cal.get('enabled', False))},
          mean_T={esc(cal.get('mean_temperature'))}</li>
      <li>OOF threshold tune={esc(tune.get('threshold') if tune.get('enabled') else 'off')}</li>
    </ul>
  </section>
  <section>
    <h2>Top priority proteins</h2>
    <table>
      <thead><tr>
        <th>protein</th><th>score</th><th>roles</th><th>halluc</th>
        <th>quality</th><th>reasons</th>
      </tr></thead>
      <tbody>
        {''.join(rows) or '<tr><td colspan="6">none</td></tr>'}
      </tbody>
    </table>
  </section>
</main>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


def _iter_jsonl(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with opener(path, mode, encoding="utf-8") as f:  # type: ignore[arg-type]
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_idr_layer_jsonl(path: str) -> list[dict]:
    """Load a prior IDR layer JSONL[.gz] export (for resume / compare)."""
    return list(_iter_jsonl(path))


def resume_protein_ids_from_jsonl(path: str) -> set:
    """Protein IDs already present in a prior export (skip on resume)."""
    return {rec["protein_id"] for rec in _iter_jsonl(path) if "protein_id" in rec}


def compare_idr_layer_jsonl(path_a: str, path_b: str) -> dict:
    """
    Diff two IDR layer JSONL exports (e.g. before/after threshold change).

    Accuracy-neutral QA tool — ranks proteins by triage / role / hallucination deltas.
    """
    a = {rec["protein_id"]: rec for rec in _iter_jsonl(path_a)}
    b = {rec["protein_id"]: rec for rec in _iter_jsonl(path_b)}
    shared = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))

    deltas: list[dict] = []
    for pid in shared:
        ra, rb = a[pid], b[pid]
        sa = float((ra.get("triage") or {}).get("score", 0.0))
        sb = float((rb.get("triage") or {}).get("score", 0.0))
        deltas.append({
            "protein_id": pid,
            "triage_delta": round(sb - sa, 4),
            "roles_a": ra.get("n_role_assignments", 0),
            "roles_b": rb.get("n_role_assignments", 0),
            "roles_delta": int(rb.get("n_role_assignments", 0)) - int(ra.get("n_role_assignments", 0)),
            "halluc_a": (ra.get("hallucination") or {}).get("n_hallucinated", 0),
            "halluc_b": (rb.get("hallucination") or {}).get("n_hallucinated", 0),
        })
    deltas.sort(key=lambda d: -abs(d["triage_delta"]))

    return {
        "path_a": path_a,
        "path_b": path_b,
        "n_a": len(a),
        "n_b": len(b),
        "n_shared": len(shared),
        "only_in_a": only_a[:50],
        "only_in_b": only_b[:50],
        "largest_triage_deltas": deltas[:30],
        "mean_abs_triage_delta": (
            round(float(np.mean([abs(d["triage_delta"]) for d in deltas])), 4)
            if deltas else 0.0
        ),
    }


def export_disorder_caid_bundle(
    proteins: list[dict],
    disorder_probs_by_id: dict[str, np.ndarray],
    out_dir: str,
    *,
    threshold: float = 0.5,
) -> list[str]:
    """Write CAID-format disorder predictions next to the layer bundle."""
    from colab.caid3_eval import write_caid_prediction_file

    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    for p in proteins:
        pid = p["id"]
        if pid not in disorder_probs_by_id:
            continue
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in pid)
        path = os.path.join(out_dir, f"{safe}.caid")
        write_caid_prediction_file(
            path, pid, p["sequence"], disorder_probs_by_id[pid], threshold=threshold,
        )
        paths.append(path)
    return paths
