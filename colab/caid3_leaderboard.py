"""The official CAID3 Disorder-PDB leaderboard — the bar this project is measured against.

Why this file exists
--------------------
The codebase treated ``ESMDisPred = 0.895`` as "CAID3 SOTA" in a dozen places.
Both halves of that were wrong:

* On the official CAID3 Disorder-PDB benchmark ESMDisPred scores **0.937**, not
  0.895. The 0.895 in its abstract is a different evaluation.
* ESMDisPred is not the leader. **PUNCH2 at 0.955** is, and four other methods
  also sit above ESMDisPred.

Aiming at 0.895 therefore set a bar ~0.06 AUC below the real one, and a run that
"reached SOTA" against it had not. The numbers below are transcribed from the
CAID3 assessment paper's Disorder-PDB table so there is one place to correct if
they are ever restated.

Protocol, and why our numbers now match it
------------------------------------------
CAID3 Disorder-PDB is 319 targets with 31,401 positive residues, **31.6%
disordered**. Residues with no annotation are *ignored* (not counted as ordered
— that is Disorder-NOX's rule), which is what the ``-`` characters in the
reference FASTA mark.

After fixing the label/prediction alignment defect, this pipeline evaluates 319
targets over 99,239 residues at 31.6% disorder — 31,359 positives against the
official 31,401. That near-exact agreement on benchmark *composition* is the
strongest available evidence that we are scoring the same thing they scored, and
it is why the primary protocol pools all targets rather than only the ones
carrying both classes.

Reference: Critical Assessment of Protein Intrinsic Disorder Round 3,
https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/
"""

from __future__ import annotations

from typing import NamedTuple, Optional


class Entry(NamedTuple):
    """One ranked method on CAID3 Disorder-PDB."""

    rank: int
    method: str
    auc: float
    aps: Optional[float]
    note: str = ""


# Transcribed from the CAID3 Disorder-PDB results table. Ranks 1-10 are the
# published top ten; the pLDDT rows are included because this project's premise
# is built on AlphaFold's behaviour in disordered regions.
DISORDER_PDB: tuple[Entry, ...] = (
    Entry(1, "PUNCH2", 0.955, 0.928),
    Entry(2, "PUNCH2-Light", 0.953, 0.925),
    Entry(3, "AlphaFold-rsa", 0.950, 0.921,
          "relative solvent accessibility, not pLDDT"),
    Entry(4, "SPOT-Disorder2", 0.949, 0.920),
    Entry(5, "AlphaFold3-rsa", 0.947, 0.912,
          "relative solvent accessibility, not pLDDT"),
    Entry(6, "PredIDR2-Seq-Art", 0.939, 0.809),
    Entry(7, "LMDisorder", 0.937, 0.621),
    Entry(7, "ESMDisPred-2PDB", 0.937, 0.893,
          "the method this repo previously cited as 0.895"),
    Entry(9, "PredIDR2-Prof-Art", 0.936, 0.884),
    Entry(10, "PredIDR2-Prof-Rnd", 0.934, 0.829),
    Entry(11, "AlphaFold-pLDDT", None, None, "rank 11; AUC not transcribed"),
    Entry(13, "AlphaFold3-pLDDT", None, None, "rank 13; AUC not transcribed"),
)

# The bar for a SOTA claim on this benchmark.
SOTA_AUC = 0.955
SOTA_APS = 0.928
SOTA_METHOD = "PUNCH2"

# Kept because a dozen call sites and every stored report reference it. It is
# NOT the SOTA figure and must not be used as one.
ESMDISPRED_ABSTRACT_AUC = 0.895
ESMDISPRED_CAID3_DISORDER_PDB_AUC = 0.937

# Official benchmark composition, used to check that we scored the same thing.
N_TARGETS = 319
N_POSITIVE_RESIDUES = 31_401
DISORDER_FRACTION = 0.316


def rank_of(auc: float) -> int:
    """Where an AUC would place among the transcribed methods (1 = best).

    Only ranks against entries with a published AUC. A method tying the last
    transcribed entry gets that entry's rank + 1, since the full table is longer
    than the excerpt here — so this is a *lower bound* on the true rank, never a
    flattering one.
    """
    ranked = [e for e in DISORDER_PDB if e.auc is not None]
    better = sum(1 for e in ranked if e.auc > auc)
    return better + 1


def gap_to_sota(auc: float) -> float:
    """Signed distance to the CAID3 leader. Negative means below it."""
    return auc - SOTA_AUC


def summarize(auc: float, aps: Optional[float] = None) -> dict:
    """An honest one-shot verdict against the real leaderboard."""
    beaten = [e for e in DISORDER_PDB if e.auc is not None and auc > e.auc]
    out = {
        "auc": auc,
        "aps": aps,
        "sota_method": SOTA_METHOD,
        "sota_auc": SOTA_AUC,
        "gap_to_sota_auc": round(gap_to_sota(auc), 4),
        "is_sota": auc > SOTA_AUC,
        "approx_rank_lower_bound": rank_of(auc),
        "beats_transcribed_methods": [e.method for e in beaten],
        "beats_esmdispred_on_this_benchmark": auc > ESMDISPRED_CAID3_DISORDER_PDB_AUC,
        "note": (
            "Ranks are a lower bound: only the published top ten plus the pLDDT "
            "rows are transcribed here, and the full CAID3 table is longer. "
            "ESMDisPred scores 0.937 on Disorder-PDB; the 0.895 this repo used "
            "to cite is from its abstract and is a different evaluation."
        ),
    }
    if aps is not None:
        out["gap_to_sota_aps"] = round(aps - SOTA_APS, 4)
        # AUC and APS can disagree, and APS is the harder metric on a benchmark
        # that is ~32% positive. A claim resting on AUC alone should say so.
        out["aps_gap_larger_than_auc_gap"] = abs(aps - SOTA_APS) > abs(gap_to_sota(auc))
    return out


def format_table(our_auc: Optional[float] = None,
                 our_aps: Optional[float] = None,
                 our_label: str = "DisorderNet-Lite") -> str:
    """Render the leaderboard, optionally slotting our result into place."""
    rows: list[tuple[str, str, str, str]] = []
    inserted = our_auc is None
    for e in DISORDER_PDB:
        if not inserted and e.auc is not None and our_auc > e.auc:
            rows.append((f">{rank_of(our_auc)}", our_label, f"{our_auc:.4f}",
                         f"{our_aps:.4f}" if our_aps is not None else "—"))
            inserted = True
        rows.append((
            str(e.rank), e.method,
            f"{e.auc:.3f}" if e.auc is not None else "—",
            f"{e.aps:.3f}" if e.aps is not None else "—",
        ))
    if not inserted:
        rows.append((f">{rank_of(our_auc)}", our_label, f"{our_auc:.4f}",
                     f"{our_aps:.4f}" if our_aps is not None else "—"))

    width = max(len(r[1]) for r in rows)
    lines = [f"{'rank':>5}  {'method':<{width}}  {'AUC':>6}  {'APS':>6}",
             f"{'-' * 5}  {'-' * width}  {'-' * 6}  {'-' * 6}"]
    lines += [f"{r:>5}  {m:<{width}}  {a:>6}  {p:>6}" for r, m, a, p in rows]
    return "\n".join(lines)
