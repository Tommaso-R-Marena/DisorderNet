#!/usr/bin/env python3
"""Score checkpoints on structures released after the training data was built.

Every other leak control in this project is a filter that has to be *correct*.
This one is the calendar. The training caches are dated 2026-08-08 and
2026-08-10; a structure first released on 2026-08-11 or later was not in them,
whatever any filter does or fails to do.

Two things the calendar does *not* handle, both done here:

**Older paralogues.** A protein released last week may be 95% identical to one
solved in 2019 and sitting in the training union. The same BLAST pass that
keeps CAID targets out of training removes those, and the run reports how many.

**Structures of proteins already in training.** A new crystal form of a protein
whose sequence is already in the union is caught by the same pass, since it is
its own 100% homologue.

Baselines are scored alongside, from the same AlphaFold models the CAID
evaluation uses: `AlphaFold-rsa` ranks 3rd on CAID3 Disorder-PDB and is
training-free, so beating the field while failing to beat it would mean
nothing. Reported on the subset with a structure, and that subset is named.

    python rockfish/eval_temporal.py --reference .../temporal/disorder_pdb.fasta \
        --checkpoint .../multitask_pbias --checkpoint .../multitask_windowed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colab.auc_decomposition import decompose_auc  # noqa: E402
from colab.bias_ceiling import ceiling_attainable  # noqa: E402
from colab.caid3_official import evaluated_mask, read_reference  # noqa: E402
from rockfish.eval_caid3_official import _taper, _windows  # noqa: E402


def training_sequences(disprot: str, pdb_missing: str, limit: int = 0):
    """Every sequence the model could have trained on, from the same caches."""
    seqs = []
    if disprot and os.path.isfile(disprot):
        with open(disprot) as fh:
            data = json.load(fh)
        entries = data if isinstance(data, list) else data.get("data", [])
        for e in entries:
            s = (e.get("sequence") or "").strip()
            if s:
                seqs.append({"id": str(e.get("disprot_id") or len(seqs)),
                             "sequence": s})
    if pdb_missing and os.path.isfile(pdb_missing):
        with open(pdb_missing) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = (rec.get("sequence") or "").strip()
                if s:
                    seqs.append({"id": str(rec.get("acc") or len(seqs)),
                                 "sequence": s})
                if limit and len(seqs) >= limit:
                    break
    return seqs


def drop_training_homologues(ref: dict, train_seqs, min_identity: float,
                             allow_missing: bool):
    """Remove temporal chains homologous to anything in the training union."""
    exact = {r["sequence"] for r in train_seqs}
    kept = {k: v for k, v in ref.items() if v[0] not in exact}
    n_exact = len(ref) - len(kept)

    ids = list(kept)
    try:
        from colab.homology_splits import blast_cross_identity_hits

        hits = blast_cross_identity_hits(
            [{"id": t, "sequence": kept[t][0]} for t in ids],
            train_seqs, min_identity=min_identity)
    except ImportError:
        hits = None

    if hits is None:
        if not allow_missing:
            raise SystemExit(
                "BLAST unavailable. The calendar excludes these structures but "
                "not their older paralogues, and a 95%-identical protein solved "
                "in 2019 leaks nearly as much as the target. Pass "
                "--allow-missing-homology-filter to proceed knowingly.")
        homologous = set()
    else:
        # (query_index, subject_index, identity) triples, not ids.
        homologous = {ids[q] for q, _s, _i in hits}

    out = {k: v for k, v in kept.items() if k not in homologous}
    return out, {"n_before": len(ref), "n_exact_sequence": n_exact,
                 "n_homologous": len(homologous), "n_after": len(out),
                 "min_identity": min_identity,
                 "homology_filter": "blast" if hits is not None else "skipped"}


def load_head(checkpoint: str, backbone: str, device):
    from colab.esm_backbone import load_esm_backbone
    from colab.lite_head import (WIDE_DILATIONS, MultiTaskLiteHead, ScalarMix,
                                 freeze_backbone)

    payload = torch.load(os.path.join(checkpoint, "multitask_head.pt"),
                         map_location="cpu", weights_only=False)
    esm, _a, batch_converter, spec = load_esm_backbone(
        device, backbone=backbone, use_gradient_checkpointing=False)
    freeze_backbone(esm)
    layer_ids = list(payload["layer_ids"])
    mix = ScalarMix(len(layer_ids)).to(device)
    mix.load_state_dict(payload["mix"])
    head = MultiTaskLiteHead(
        in_dim=spec.embed_dim, tasks=tuple(payload["tasks"]),
        structure_dim=int(payload.get("structure_dim", 0)),
        condition_binding=bool(payload.get("condition_binding", False)),
        protein_bias=bool(payload.get("protein_bias", False)),
        private_trunk=bool(payload.get("private_trunk", False)),
        private_narrow=bool(payload.get("private_narrow", False)),
        private_detach=bool(payload.get("private_detach", True)),
        chiral=bool(payload.get("chiral", False)),
        dilations=(WIDE_DILATIONS if payload.get("wide_receptive_field")
                   else None)).to(device)
    head.load_state_dict(payload["head"])
    head.eval()
    mix.eval()
    return head, mix, esm, batch_converter, layer_ids, payload


def predict(head, mix, esm, batch_converter, layer_ids, device, ref, task,
            structures, structure_dim, batch_size=8):
    """Windowed inference, identical to the CAID path."""
    jobs = []
    for tid, (seq, _lab) in ref.items():
        for (a, b) in _windows(len(seq)):
            jobs.append((tid, a, b, seq[a:b]))
    acc = {t: np.zeros(len(s), np.float64) for t, (s, _l) in ref.items()}
    wsum = {t: np.zeros(len(s), np.float64) for t, (s, _l) in ref.items()}

    with torch.no_grad():
        for s0 in range(0, len(jobs), batch_size):
            batch = jobs[s0:s0 + batch_size]
            _, _, tokens = batch_converter([(f"{t}:{a}", sub)
                                            for t, a, _b, sub in batch])
            tokens = tokens.to(device)
            rep = esm(tokens, repr_layers=layer_ids, return_contacts=False)
            feats = mix([rep["representations"][i][:, 1:-1, :]
                         for i in layer_ids])
            kw = {}
            if structure_dim:
                L = feats.shape[1]
                z = lambda: torch.zeros(len(batch), L, device=device)  # noqa: E731
                sr, sp, sa, sc = z(), z(), z(), z()
                sh = torch.full((len(batch), L), float("nan"), device=device)
                for bi, (tid, a, b, _sub) in enumerate(batch):
                    f = structures.get(tid) or {}
                    if not f:
                        continue
                    r = np.asarray(f["rsa"], np.float32)[a:b]
                    q = np.asarray(f["plddt"], np.float32)[a:b]
                    k = min(len(r), len(q), L)
                    if k <= 0:
                        continue
                    sr[bi, :k] = torch.from_numpy(r[:k]).to(device)
                    sp[bi, :k] = torch.from_numpy(q[:k]).to(device)
                    sa[bi, :k] = 1.0
                    ct = np.asarray(f.get("contacts", np.zeros(b - a)),
                                    np.float32)[a:b]
                    sc[bi, :min(len(ct), k)] = torch.from_numpy(
                        ct[:min(len(ct), k)]).to(device)
                    hd = np.asarray(f.get("handedness", np.full(b - a, np.nan)),
                                    np.float32)[a:b]
                    kh = min(len(hd), k)
                    if kh > 0:
                        sh[bi, :kh] = torch.from_numpy(
                            np.ascontiguousarray(hd[:kh])).to(device)
                kw = {"rsa": sr, "plddt": sp, "structure_available": sa,
                      "contacts": sc, "handedness": sh}
            probs = torch.sigmoid(head(feats, **kw)[task]).float().cpu().numpy()
            for bi, (tid, a, _b, sub) in enumerate(batch):
                n = min(len(sub), probs.shape[1])
                w = _taper(n)
                acc[tid][a:a + n] += probs[bi, :n].astype(np.float64) * w
                wsum[tid][a:a + n] += w
    return {t: acc[t] / np.maximum(wsum[t], 1e-12) for t in acc}


def arrays(ref, preds):
    ys, ss, kept = [], [], []
    for tid, (_seq, lab) in ref.items():
        p = preds.get(tid)
        if p is None or len(p) != len(lab):
            continue
        m = evaluated_mask(lab)
        y = np.frombuffer(lab.encode(), np.uint8)[m].astype(np.int8) - ord("0")
        s = np.asarray(p)[m]
        if not np.isfinite(s).all() or len(np.unique(y)) < 2:
            continue
        ys.append(y)
        ss.append(s.astype(np.float64))
        kept.append(tid)
    return ys, ss, kept


def report(name, ys, ss):
    d = decompose_auc(ys, ss)
    if d.get("pooled") is None:
        return None
    att = ceiling_attainable(ys, ss)
    row = {"pooled": d["pooled"], "within": d["auc_within"],
           "within_unweighted": d["auc_within_unweighted"],
           "between": d["auc_between"], "w_within": d["w_within"],
           "n_targets": d["n_targets_with_both_classes"],
           "ceiling": d["w_within"] * d["auc_within"] + d["w_between"],
           "ceiling_attainable": bool(att["attainable"])}
    print(f" {name:<28}{row['pooled']:>9.4f}{row['within']:>9.4f}"
          f"{row['between']:>10.4f}{row['n_targets']:>8}")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--checkpoint", action="append", default=[])
    ap.add_argument("--task", default="disorder_pdb")
    ap.add_argument("--backbone", default="650M")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--structure-cache",
                    default="/scratch4/sfried3/jbeale3_disordernet/af_structures")
    ap.add_argument("--raw-records", default=None,
                    help="the builder's raw JSON, for UniProt accessions")
    ap.add_argument("--disprot",
                    default=os.path.expanduser("~/.cache/disordernet/disprot_raw.json"))
    ap.add_argument("--pdb-missing",
                    default=os.path.expanduser("~/.cache/disordernet/mobidb_pdbcov.ndjson"))
    ap.add_argument("--min-identity", type=float, default=0.40)
    ap.add_argument("--allow-missing-homology-filter", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ref = read_reference(args.reference)
    print(f"temporal reference: {len(ref)} chains")

    print("loading the training union for the homology pass…")
    train = training_sequences(args.disprot, args.pdb_missing)
    print(f"  {len(train):,} training sequences")
    ref, filt = drop_training_homologues(
        ref, train, args.min_identity, args.allow_missing_homology_filter)
    print(f"  removed {filt['n_exact_sequence']} exact and "
          f"{filt['n_homologous']} homologous at >={args.min_identity}; "
          f"{filt['n_after']} chains remain")
    if len(ref) < 20:
        print("ERROR: too few chains survive to report anything",
              file=sys.stderr)
        return 1

    from colab.disordernet_gpu import TrainConfig, setup_environment
    cfg = setup_environment(TrainConfig.from_profile("lite",
                                                     esm_backbone=args.backbone))
    device = cfg.device

    acc_by_id = {}
    if args.raw_records and os.path.isfile(args.raw_records):
        for r in json.load(open(args.raw_records)):
            if r.get("uniprot"):
                acc_by_id[r["id"]] = r["uniprot"]

    # Which chains a training-free structural baseline can score at all. Ours
    # are reported on this subset too: comparing a 186-chain score against a
    # 61-chain score is comparing two different benchmarks, and the direction
    # of that error is not knowable in advance.
    from colab.structure_rsa import structure_features

    baseline_preds = {}
    for label, key, sign in (("AlphaFold-rsa", "rsa", 1.0),
                             ("AlphaFold-pLDDT", "plddt", -1.0)):
        p_map = {}
        for tid, (seq, _lab) in ref.items():
            a = acc_by_id.get(tid)
            f = (structure_features(a, seq, args.structure_cache,
                                    allow_fetch=False) if a else None)
            if f and len(f[key]) == len(seq):
                p_map[tid] = sign * np.asarray(f[key], np.float64)
        baseline_preds[label] = p_map
    structured = sorted(set(baseline_preds["AlphaFold-rsa"])
                        & set(baseline_preds["AlphaFold-pLDDT"]))
    sub_ref = {t: ref[t] for t in structured}
    print(f"\n{len(structured)} of {len(ref)} chains have an AlphaFold model; "
          f"the structural baselines are scored on those, and so are we.")

    print(f"\n{'method':<29}{'pooled':>9}{'within':>9}{'between':>10}"
          f"{'targets':>8}")
    results, preds_by_name = {}, {}

    for ckpt in args.checkpoint:
        name = os.path.basename(ckpt.rstrip("/"))
        head, mix, esm, bc, layer_ids, payload = load_head(
            ckpt, args.backbone, device)
        structure_dim = int(payload.get("structure_dim", 0))
        structures = {}
        if structure_dim:
            from colab.structure_rsa import structure_features
            for tid, (seq, _l) in ref.items():
                a = acc_by_id.get(tid)
                structures[tid] = (structure_features(
                    a, seq, args.structure_cache, allow_fetch=False)
                    if a else None) or {}
        t0 = time.perf_counter()
        preds = predict(head, mix, esm, bc, layer_ids, device, ref, args.task,
                        structures, structure_dim, args.batch_size)
        wall = time.perf_counter() - t0
        ys, ss, kept = arrays(ref, preds)
        row = report(name, ys, ss)
        if row:
            row["inference_seconds"] = round(wall, 1)
            row["n_with_structure"] = sum(1 for v in structures.values() if v)
            results[name] = row
            preds_by_name[name] = preds
            if sub_ref:
                sy, sx, _ = arrays(sub_ref, preds)
                sub = report(f"  {name} (structured subset)", sy, sx)
                if sub:
                    row["structured_subset"] = sub
        del head, mix, esm
        torch.cuda.empty_cache()

    for label, p_map in baseline_preds.items():
        if len(sub_ref) < 20:
            print(f" {label:<28}  only {len(sub_ref)} chains have a model")
            continue
        ys, ss, _k = arrays(sub_ref, p_map)
        row = report(f"{label}", ys, ss)
        if row:
            results[label] = row
            preds_by_name[label] = p_map

    # Paired, on shared chains, resampled by protein. A difference is only a
    # difference if the interval says so, and the interval must cluster on
    # proteins because residues within a chain are anything but independent.
    if preds_by_name and len(sub_ref) >= 20:
        rng = np.random.default_rng(20260818)
        print(f"\n paired against the best structural baseline, "
              f"protein-clustered bootstrap ({args.n_boot:,} resamples), "
              f"on the {len(sub_ref)} structured chains")
        print(f" {'comparison':<44}{'delta':>9}{'95% CI':>21}{'p':>8}")
        base_label = max(
            (l for l in baseline_preds if l in results),
            key=lambda l: results[l]["pooled"], default=None)
        for name in [n for n in preds_by_name if n not in baseline_preds]:
            if base_label is None:
                break
            ay, ax, ka = arrays(sub_ref, preds_by_name[name])
            by, bx, kb = arrays(sub_ref, preds_by_name[base_label])
            if ka != kb or not ka:
                print(f" {name:<44}  chain sets differ — skipped")
                continue
            deltas = np.empty(args.n_boot)
            n = len(ka)
            for i in range(args.n_boot):
                take = rng.integers(0, n, size=n)
                da = decompose_auc([ay[j] for j in take], [ax[j] for j in take])
                db = decompose_auc([by[j] for j in take], [bx[j] for j in take])
                deltas[i] = ((da.get("pooled") or np.nan)
                             - (db.get("pooled") or np.nan))
            deltas = deltas[np.isfinite(deltas)]
            point = (decompose_auc(ay, ax)["pooled"]
                     - decompose_auc(by, bx)["pooled"])
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            # Both tails carry the ties, with the Davison-Hinkley correction.
            m = deltas.size
            p_lo = (1.0 + np.count_nonzero(deltas <= 0.0)) / (m + 1.0)
            p_hi = (1.0 + np.count_nonzero(deltas >= 0.0)) / (m + 1.0)
            pval = 2.0 * min(p_lo, p_hi)
            print(f" {name + ' - ' + base_label:<44}{point:>+9.4f}"
                  f"{f'[{lo:+.4f},{hi:+.4f}]':>21}{min(pval, 1.0):>8.4f}")
            results.setdefault(name, {})["vs_baseline"] = {
                "opponent": base_label, "delta": point,
                "ci": [float(lo), float(hi)], "p": float(min(pval, 1.0)),
                "n_chains": n}

    # ── Distribution-free coverage ───────────────────────────────────────
    # Split conformal, calibrated on half the chains and measured on the other
    # half. Both halves are post-cutoff and homology-filtered, so the exchange-
    # ability the guarantee needs is between two random halves of one set of
    # structures nobody had seen — the cleanest version of that assumption
    # available anywhere in this project.
    from colab.conformal import (calibrate, control_chain_risk,
                                 evaluate_chain_risk, evaluate_sets,
                                 prediction_sets, split_by_chain)

    conformal = {}
    if preds_by_name:
        print(f"\n{'=' * 100}")
        print(" DISTRIBUTION-FREE COVERAGE — chains split 50/50, "
              "class-conditional")
        print(" Validity is guaranteed whatever the model does; what a model "
              "earns is how much it")
        print(" has to flag. Every method below is scored on the SAME chains, "
              "so the comparison is")
        print(" matched — the structural baselines exist only where an "
              "AlphaFold model does.")
        print("=" * 100)

        def conformal_block(scope_name, scope_ref, names):
            rng2 = np.random.default_rng(20260818)
            # One split, shared by every method in the block, so differences
            # are the methods and not the draw.
            probe = arrays(scope_ref, preds_by_name[names[0]])[2]
            if len(probe) < 20:
                print(f" {scope_name}: only {len(probe)} chains — skipped")
                return
            split_rng = np.random.default_rng(20260818)
            cal_chains, _ = split_by_chain(np.arange(len(probe)), split_rng,
                                           0.5)
            print(f"\n {scope_name}: {len(probe)} chains, "
                  f"{int(cal_chains.sum())} calibrate / "
                  f"{int((~cal_chains).sum())} test")
            for name in names:
                ys, ss, kept = arrays(scope_ref, preds_by_name[name])
                if kept != probe:
                    print(f"  {name}: chain set differs — skipped")
                    continue
                if name in baseline_preds:
                    flat = np.concatenate(ss)
                    rank = (np.argsort(np.argsort(flat)) + 0.5) / len(flat)
                    P, off = [], 0
                    for y in ys:
                        P.append(rank[off:off + len(y)])
                        off += len(y)
                else:
                    P = [np.clip(np.asarray(x), 1e-6, 1 - 1e-6) for x in ss]
                cal_idx = [i for i in range(len(ys)) if cal_chains[i]]
                test_idx = [i for i in range(len(ys)) if not cal_chains[i]]

                y_cal = np.concatenate([ys[i] for i in cal_idx])
                p_cal = np.concatenate([P[i] for i in cal_idx])
                y_te = np.concatenate([ys[i] for i in test_idx])
                p_te = np.concatenate([P[i] for i in test_idx])

                row = {"scope": scope_name, "n_chains": len(probe)}
                for alpha in (0.10, 0.05):
                    c = calibrate(p_cal, y_cal, alpha, class_conditional=True)
                    e = evaluate_sets(prediction_sets(p_te, c), y_te)
                    row[f"split_alpha_{alpha}"] = e
                    print(f"  {name:<24}{1-alpha:>5.0%} per-residue target |"
                          f" realised {e['coverage']:.3f}"
                          f" (ord {e.get('coverage_class0', float('nan')):.3f},"
                          f" dis {e.get('coverage_class1', float('nan')):.3f})")
                for alpha in (0.10, 0.05):
                    got = control_chain_risk([P[i] for i in cal_idx],
                                             [ys[i] for i in cal_idx], alpha)
                    ev = evaluate_chain_risk([P[i] for i in test_idx],
                                             [ys[i] for i in test_idx],
                                             got["threshold"])
                    row[f"risk_alpha_{alpha}"] = {"calibrated": got,
                                                  "realised": ev}
                    ok = "ok" if ev["mean_chain_miss_rate"] <= alpha else "MISSED"
                    print(f"  {name:<24}chain risk <= {alpha:.2f}      |"
                          f" realised {ev['mean_chain_miss_rate']:.3f} {ok:<7}|"
                          f" flags {ev['mean_fraction_called_disordered']:.1%}")
                conformal.setdefault(name, {})[scope_name] = row

        ours = [n for n in preds_by_name if n not in baseline_preds]
        base = [n for n in preds_by_name if n in baseline_preds]
        if ours:
            conformal_block("all chains", ref, ours)
        if base and sub_ref:
            conformal_block("chains with an AlphaFold model", sub_ref,
                            ours + base)

    print(f"\nfilter: {json.dumps(filt)}")
    if args.out:
        with open(args.out + ".part", "w") as fh:
            json.dump({"filter": filt, "results": results,
                       "conformal": conformal,
                       "reference": args.reference, "task": args.task}, fh,
                      indent=2, default=float)
        os.replace(args.out + ".part", args.out)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
