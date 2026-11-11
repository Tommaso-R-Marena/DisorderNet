#!/usr/bin/env python3
"""Is there a way around the capacity bound? Measure the noise that AUC sees.

`card_le_benchCapacity` bounds capacity by `⌈1/(2ε)⌉` where `ε` is the rate at
which the annotation gets a **label** wrong. At the measured ε = 0.0801 that is
7 methods, and CAID3 has 117.

But AUC is not a function of labels one at a time. It is a function of **ordered
pairs**: the probability a positive outranks a negative. So the noise that
matters for a ranking statistic is not the label-flip rate — it is the rate at
which the annotation **reverses a pair's order**.

Those are different, and the difference is the escape route. Label noise in
protein structure is not independent across residues: a whole region orders in
one crystal form and not another. When a region flips *together*, every pair
**inside** it keeps its relation (both flip, so the pair stays tied or stays
ordered the same way), and only pairs **crossing** the region's boundary
reverse. If the flipping is spatially correlated, the pairwise discordance rate
is far below the label-flip rate — and the capacity of a pairwise statistic is
correspondingly larger.

This measures both rates on the same data: MobiDB's per-structure
missing-residue calls, pairs of structures of the same protein, restricted to
residues both structures cover.

    eps_label     = P(the two structures disagree about one residue)
    eps_pairwise  = P(the two structures order one pair oppositely)

If `eps_pairwise << eps_label`, the within-protein protocol recovers resolution
the residue-level protocol throws away, and the capacity bound for the pairwise
statistic is the one to quote. If they are comparable, the escape route is
closed and the bound stands as computed.

Either answer is worth having. The first is a prescription; the second closes a
line of attack.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.environ.get("REPO", os.path.expanduser("~/dn_rigor")))

from colab.caid3_official import read_reference  # noqa: E402

ROOT = "/scratch4/sfried3/jbeale3_disordernet"
REFS = os.environ.get("RN_REFS", f"{ROOT}/caid3_official")
DISPROT = os.environ.get(
    "RN_DISPROT", os.path.expanduser("~/.cache/disordernet/disprot_raw.json"))
OUT = os.environ.get("RN_OUT", "")
WORKERS = int(os.environ.get("RN_WORKERS", "8"))
MAX_STRUCT_PAIRS = int(os.environ.get("RN_MAX_PAIRS", "40"))


def regions_to_mask(regions, length: int) -> np.ndarray:
    """MobiDB regions are inclusive 1-based [start, end] pairs."""
    m = np.zeros(length, dtype=bool)
    for r in regions or []:
        try:
            a, b = int(r[0]), int(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        m[max(a, 1) - 1:min(b, length)] = True
    return m


def fetch(acc: str) -> dict | None:
    url = f"https://mobidb.org/api/download?acc={acc}&format=json"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as fh:
                rec = json.load(fh)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == 3:
                return None
        except Exception:
            if attempt == 3:
                return None
        time.sleep(1.5 * (attempt + 1))
    else:
        return None
    if isinstance(rec, list):
        rec = rec[0] if rec else {}
    if not isinstance(rec, dict):
        return None
    try:
        length = int(rec.get("length") or 0)
    except (TypeError, ValueError):
        return None
    if length <= 0:
        return None

    # Per-structure missing calls, and the aggregated observed regions which
    # bound what any structure could have resolved.
    per = {}
    for k, v in rec.items():
        if not k.startswith("derived-missing_residues-mobi-"):
            continue
        if not isinstance(v, dict):
            continue
        per[k.rsplit("-", 1)[-1]] = regions_to_mask(v.get("regions"), length)
    obs = rec.get("derived-observed-th_90")
    observed = (regions_to_mask(obs.get("regions"), length)
                if isinstance(obs, dict) else np.zeros(length, bool))
    if len(per) < 2:
        return None

    # A structure's coverage: the residues it resolved plus those it declared
    # missing. Approximated by the span between its first and last annotated
    # residue intersected with the protein's observed-or-missing footprint,
    # because MobiDB gives per-structure *missing* regions and not per-structure
    # coverage. Residues outside a structure's span are "not determined by that
    # structure", not "observed by it" — conflating those would manufacture
    # disagreement out of absence.
    footprint = observed.copy()
    for m in per.values():
        footprint |= m

    cover = {}
    for sid, miss in per.items():
        idx = np.nonzero(miss)[0]
        span = np.zeros(length, bool)
        if idx.size:
            span[idx.min():idx.max() + 1] = True
        # Extend the span to the contiguous footprint around it.
        cover[sid] = span | (footprint & span)
        cover[sid] = span
    return {"acc": acc, "length": length, "per": per, "cover": cover,
            "footprint": footprint}


def rates(rec) -> dict | None:
    """Label-flip and pairwise-discordance rates over structure pairs."""
    sids = sorted(rec["per"])
    lab_dis = lab_tot = 0
    pair_dis = pair_tot = pair_tot_old = 0
    agree_dis = agree_ord = flip_d = flip_u = 0
    sp_checked = sp_bound_ok = sp_balanced = sp_eps_small = 0
    sp_hyp_ok = sp_bound_ok_under_hyp = 0
    sp_imb_checked = sp_imb_ok = 0
    used = 0
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            if used >= MAX_STRUCT_PAIRS:
                break
            a, b = rec["per"][sids[i]], rec["per"][sids[j]]
            both = rec["cover"][sids[i]] & rec["cover"][sids[j]]
            n = int(both.sum())
            if n < 20:
                continue
            ya, yb = a[both].astype(np.int8), b[both].astype(np.int8)
            lab_dis += int((ya != yb).sum())
            lab_tot += n
            # Ordered pairs: a pair (p, q) is strictly ordered by a labelling
            # when exactly one of them is called missing. Two labellings are
            # discordant on that pair when both order it and they disagree.
            pa, na = int(ya.sum()), n - int(ya.sum())
            pb, nb = int(yb.sum()), n - int(yb.sum())
            # concordant-with-a-and-b pairs: (i disordered in both, j ordered
            # in both) etc. Count via the 2x2 contingency of (ya, yb).
            n11 = int(((ya == 1) & (yb == 1)).sum())
            n10 = int(((ya == 1) & (yb == 0)).sum())
            n01 = int(((ya == 0) & (yb == 1)).sum())
            n00 = int(((ya == 0) & (yb == 0)).sum())
            # a orders (p,q) as p>q iff ya[p]=1, ya[q]=0. b orders it the other
            # way iff yb[p]=0, yb[q]=1. So discordant pairs are those with
            # p in {ya=1, yb=0} and q in {ya=0, yb=1}, in either direction:
            #
            #   discordant = 2 * d * u          `discordant_eq_flip_product`
            #
            # and the pairs *both* labellings order — the only ones a pairwise
            # protocol scores — are the ones separated by both:
            #
            #   comparable = 2 * (a * e + d * u)   `card_comparablePairs`
            #
            # with a = |T n L| (both call disordered), e = |(T u L)^c| (both
            # call ordered), d = |T \ L|, u = |L \ T|.
            #
            # An earlier version of this denominator counted
            #   2du + (a+d)(e+u) + (a+u)(e+d) = 2ae + 4du + (a+e)(d+u),
            # which is the pairs ordered by *either* labelling with the
            # discordant ones counted twice — larger than the comparable set,
            # and mixing ordered with unordered counts. It understated the rate
            # by a factor of 1.17 at this composition. Both are recorded so the
            # correction is visible rather than silent.
            pair_dis += 2 * n10 * n01
            pair_tot += 2 * (n11 * n00 + n10 * n01)
            pair_tot_old += 2 * (n10 * n01) + \
                (n11 + n10) * (n00 + n01) + (n11 + n01) * (n00 + n10)
            agree_dis += n11
            agree_ord += n00
            flip_d += n10
            flip_u += n01
            # The theorem's unit is one (truth, annotation) pair, not one
            # protein, so the bound is checked here rather than on a protein's
            # pooled ratios -- a sum of ratios satisfies no bound its terms do.
            cmp_ij = 2 * (n11 * n00 + n10 * n01)
            if cmp_ij:
                eps_ij = (n10 + n01) / n
                nu_ij = (2 * n10 * n01) / cmp_ij
                sp_checked += 1
                sp_bound_ok += int(nu_ij <= 2 * eps_ij ** 2)
                sp_balanced += int(min(n11, n00) >= 0.9 * max(n11, n00, 1))
                sp_eps_small += int(4 * (n10 + n01) <= n)
                sp_hyp_ok += int(min(n11, n00) >= 0.9 * max(n11, n00, 1)
                                 and 4 * (n10 + n01) <= n)
                # Candidate generalisation, checked before it is asked for:
                #   nu_pair <= kappa * eps^2 / (1-eps)^2,
                #   kappa = (a+e)^2 / (4*a*e)   the imbalance factor, >= 1,
                # from nu_pair <= du/ae, du <= nu^2/4 and a+e = n(1-eps). It
                # needs no balance hypothesis, and reduces to the balanced form
                # when kappa = 1. If it holds here it restores a closed form on
                # references the published bound cannot reach.
                if n11 * n00 > 0 and n10 + n01 > 0:
                    kap = (n11 + n00) ** 2 / (4.0 * n11 * n00)
                    sp_imb_checked += 1
                    sp_imb_ok += int(nu_ij <= kap * eps_ij ** 2
                                     / (1 - eps_ij) ** 2 + 1e-12)
                if (min(n11, n00) >= 0.9 * max(n11, n00, 1)
                        and 4 * (n10 + n01) <= n):
                    sp_bound_ok_under_hyp += int(nu_ij <= 2 * eps_ij ** 2)
            used += 1
        if used >= MAX_STRUCT_PAIRS:
            break
    if lab_tot == 0 or pair_tot == 0:
        return None
    return {"acc": rec["acc"], "n_structures": len(sids),
            "n_structure_pairs": used,
            "eps_label": lab_dis / lab_tot,
            "eps_pairwise": pair_dis / pair_tot,
            "eps_pairwise_superseded": pair_dis / pair_tot_old,
            "label_residues": lab_tot, "label_disagree": lab_dis,
            "pair_total": pair_tot, "pair_total_superseded": pair_tot_old,
            "pair_discordant": pair_dis,
            # The balance hypothesis of `nuPair_le_two_eps_sq` is
            # |T n L| = |(T u L)^c|. It is recorded, not assumed: on a reference
            # that is 31.6% disordered it does not hold, and the 2*eps^2 bound
            # is then an observation rather than a guarantee.
            "agree_disordered": agree_dis, "agree_ordered": agree_ord,
            "flip_down": flip_d, "flip_up": flip_u,
            "sp_checked": sp_checked, "sp_bound_ok": sp_bound_ok,
            "sp_balanced": sp_balanced, "sp_eps_small": sp_eps_small,
            "sp_hypotheses_ok": sp_hyp_ok,
            "sp_bound_ok_under_hypotheses": sp_bound_ok_under_hyp,
            "sp_imbalanced_checked": sp_imb_checked,
            "sp_imbalanced_ok": sp_imb_ok}


def main() -> int:
    acc_of = {}
    if os.path.isfile(DISPROT):
        d = json.load(open(DISPROT))
        for e in (d if isinstance(d, list) else d.get("data", [])):
            if isinstance(e, dict) and e.get("disprot_id") and e.get("acc"):
                acc_of[str(e["disprot_id"])] = e["acc"]
    ref = read_reference(os.path.join(REFS, "disorder_pdb.fasta"))
    accs = sorted({acc_of.get(t, t) for t in ref})
    print(f"{len(accs)} accessions; keeping those with >=2 structures")

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, rec in enumerate(pool.map(fetch, accs), 1):
            if rec:
                r = rates(rec)
                if r:
                    rows.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(accs)}  usable={len(rows)}  "
                      f"[{(time.time()-t0)/60:.1f}m]", flush=True)
    print(f"\n{len(rows)} proteins with >=2 structures and comparable coverage")
    if len(rows) < 20:
        print("too few", file=sys.stderr)
        return 1

    lab = sum(r["label_disagree"] for r in rows) / sum(
        r["label_residues"] for r in rows)
    prs = sum(r["pair_discordant"] for r in rows) / sum(
        r["pair_total"] for r in rows)
    print(f"\n{'':<34}{'pooled':>12}{'median protein':>17}")
    print(f"{'label-flip rate  eps_label':<34}{lab:>12.4f}"
          f"{np.median([r['eps_label'] for r in rows]):>17.4f}")
    print(f"{'pairwise discordance  eps_pair':<34}{prs:>12.4f}"
          f"{np.median([r['eps_pairwise'] for r in rows]):>17.4f}")
    if prs > 0:
        print(f"\nratio eps_label / eps_pairwise = {lab / prs:.2f}")
    import math
    print(f"\ncapacity ceil(1/(2*eps)):")
    print(f"  from label noise    {max(1, math.ceil(1/(2*lab))) if lab else '-':>6}")
    print(f"  from pairwise noise "
          f"{max(1, math.ceil(1/(2*prs))) if prs else '-':>6}")

    prs_old = sum(r["pair_discordant"] for r in rows) / sum(
        r["pair_total_superseded"] for r in rows)
    a_tot = sum(r["agree_disordered"] for r in rows)
    e_tot = sum(r["agree_ordered"] for r in rows)
    print(f"\nsuperseded denominator gave eps_pair = {prs_old:.4f} "
          f"(capacity {max(1, math.ceil(1 / (2 * prs_old)))}); "
          f"the comparable-pairs denominator gives {prs:.4f}")
    print(f"balance hypothesis |T n L| = |(T u L)^c|: "
          f"{a_tot:,} vs {e_tot:,}  ratio {a_tot / e_tot:.3f}"
          f"  -> {'holds' if abs(a_tot - e_tot) <= 0.02 * (a_tot + e_tot) else 'FAILS'}")
    print(f"bound 2*eps_label^2 = {2 * lab ** 2:.5f}   measured {prs:.5f}   "
          f"{'satisfied' if prs <= 2 * lab ** 2 else 'VIOLATED'}")
    spc = sum(r["sp_checked"] for r in rows)
    spb = sum(r["sp_bound_ok"] for r in rows)
    spbal = sum(r["sp_balanced"] for r in rows)
    speps = sum(r["sp_eps_small"] for r in rows)
    sphyp = sum(r["sp_hypotheses_ok"] for r in rows)
    spok = sum(r["sp_bound_ok_under_hypotheses"] for r in rows)
    print(f"\nper structure pair, the theorem's own unit ({spc:,} pairs):")
    print(f"  balanced agreement classes within 10%   {spbal:6,d}"
          f"  ({spbal / spc:6.1%})")
    print(f"  noise rate eps <= 1/4                   {speps:6,d}"
          f"  ({speps / spc:6.1%})")
    print(f"  both hypotheses of nuPair_le_two_eps_sq {sphyp:6,d}"
          f"  ({sphyp / spc:6.1%})")
    print(f"  bound nu_pair <= 2 eps^2 holds          {spb:6,d}"
          f"  ({spb / spc:6.1%})")
    print(f"  ... among those satisfying both         {spok:6,d}"
          f"  ({spok / sphyp:6.1%})" if sphyp else "  ... none satisfy both")
    spic = sum(r["sp_imbalanced_checked"] for r in rows)
    spio = sum(r["sp_imbalanced_ok"] for r in rows)
    if spic:
        print(f"  candidate: nu <= kappa*eps^2/(1-eps)^2  {spio:6,d}"
              f"  ({spio / spic:6.1%} of {spic:,}, no balance needed)")

    report = {"n_proteins": len(rows), "eps_label_pooled": lab,
              "eps_pairwise_pooled": prs,
              "eps_pairwise_pooled_superseded": prs_old,
              "bound_two_eps_sq": 2 * lab ** 2,
              "bound_satisfied": bool(prs <= 2 * lab ** 2),
              "agree_disordered_total": a_tot, "agree_ordered_total": e_tot,
              "balance_ratio": a_tot / e_tot if e_tot else None,
              "capacity_label": max(1, math.ceil(1 / (2 * lab))) if lab else None,
              "capacity_pairwise": max(1, math.ceil(1 / (2 * prs))) if prs else None,
              "capacity_pairwise_superseded": (
                  max(1, math.ceil(1 / (2 * prs_old))) if prs_old else None),
              "capacity_from_bound": (
                  max(1, math.ceil(1 / (4 * lab ** 2))) if lab else None),
              "structure_pairs": {
                  "checked": spc, "bound_holds": spb,
                  "balanced_within_10pct": spbal, "eps_le_quarter": speps,
                  "both_hypotheses": sphyp,
                  "bound_holds_under_hypotheses": spok,
                  "imbalanced_checked": spic, "imbalanced_holds": spio},
              "proteins": rows}
    if OUT:
        with open(OUT + ".part", "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        os.replace(OUT + ".part", OUT)
        print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
