"""Predict per-residue intrinsic disorder for a protein sequence.

Loads a trained bundle (train_predictor.py) and outputs, for each residue, a
calibrated disorder probability and a conformal decision (confident disorder /
confident order / abstain) with a coverage guarantee at the bundle's alpha.

Examples:
  python predict_disorder.py --seq MDDQRDLISNNEQLP...
  python predict_disorder.py --fasta my_proteins.fasta --out preds.json
"""
import argparse, json, os, sys
import numpy as np

from disordernet_paths import results_dir
from predictor import predict_from_embeddings, phys_features, DECISION_LABEL


def read_fasta(path):
    seqs, name, buf = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs.append((name, "".join(buf)))
                name, buf = line[1:].split()[0] if line[1:].strip() else "seq", []
            elif line:
                buf.append(line)
    if name is not None:
        seqs.append((name, "".join(buf)))
    return seqs


def load_esm(model_name, repr_layer):
    import torch, esm
    fn = getattr(esm.pretrained, model_name)
    model, alphabet = fn()
    model.eval()
    bc = alphabet.get_batch_converter()

    def embed(seq):
        _, _, toks = bc([("q", seq[:1022])])
        with torch.no_grad():
            res = model(toks, repr_layers=[repr_layer])
        return res["representations"][repr_layer][0, 1:len(seq[:1022]) + 1].numpy()

    return embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", help="raw amino-acid sequence")
    ap.add_argument("--fasta", help="FASTA file with one or more sequences")
    ap.add_argument("--bundle", help="path to predictor_bundle.joblib")
    ap.add_argument("--out", help="write full per-residue JSON here")
    args = ap.parse_args()

    import joblib
    bundle_path = args.bundle or os.path.join(str(results_dir("results_v7")), "predictor_bundle.joblib")
    if not os.path.exists(bundle_path):
        sys.exit(f"No bundle at {bundle_path}. Run train_predictor.py first.")
    bundle = joblib.load(bundle_path)

    if args.seq:
        records = [("query", args.seq.strip().upper())]
    elif args.fasta:
        records = [(n, s.upper()) for n, s in read_fasta(args.fasta)]
    else:
        sys.exit("Provide --seq or --fasta")

    embed = load_esm(bundle["esm_model"], bundle["esm_repr_layer"])
    results = {}
    for name, seq in records:
        emb = embed(seq)
        L = emb.shape[0]
        out = predict_from_embeddings(bundle, phys_features(seq[:L]), emb)
        dec = out["decision"]
        pct_dis = float((out["p_calibrated"] >= 0.5).mean())
        confident = float(np.isin(dec, (0, 1)).mean())
        print(f"\n>{name}  len={L}")
        print(f"  predicted disordered residues: {100*pct_dis:.1f}%")
        print(f"  confident (non-abstain) residues @ alpha={bundle['alpha']}: {100*confident:.1f}%")
        print(f"  mean calibrated p(disorder): {out['p_calibrated'].mean():.3f}")
        results[name] = {
            "length": L,
            "p_calibrated": out["p_calibrated"].round(4).tolist(),
            "decision": dec.tolist(),
            "decision_labels": [DECISION_LABEL[int(d)] for d in dec],
        }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f)
        print(f"\nWrote per-residue predictions -> {args.out}")


if __name__ == "__main__":
    main()
