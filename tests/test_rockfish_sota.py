"""Tests for homology splits, CAID3 eval, structure encoder, novel use cases."""

from __future__ import annotations

import numpy as np
import torch

from colab.caid3_eval import evaluate_caid_predictions, parse_caid_reference_fasta
from colab.homology_splits import cluster_proteins_by_homology, sequence_identity
from colab.structure_encoder import PlddtFeatureEncoder, build_plddt_feature_tensor
from colab.novel_use_cases import proteome_disorder_summary, screen_af_hallucinations


class TestHomology:
    def test_identical_sequences_cluster(self):
        proteins = [
            {"id": "A", "sequence": "ACDEFGHIK", "length": 9},
            {"id": "B", "sequence": "ACDEFGHIK", "length": 9},
            {"id": "C", "sequence": "ZZZZZZZZZ", "length": 9},
        ]
        clusters, meta = cluster_proteins_by_homology(proteins, min_identity=0.9)
        assert clusters[0] == clusters[1]
        assert clusters[0] != clusters[2]
        assert meta["n_clusters"] == 2

    def test_sequence_identity(self):
        assert sequence_identity("ACDE", "ACDE") == 1.0
        assert sequence_identity("ACDE", "ZZZZ") < 0.5


class TestCAID3:
    def test_parse_mini_fasta(self):
        path = "tests/fixtures/caid3_mini.fasta"
        proteins = parse_caid_reference_fasta(path)
        assert len(proteins) == 2
        assert proteins[0]["id"] == "TEST01"
        assert len(proteins[0]["labels"]) > 0

    def test_evaluate_perfect_preds(self):
        path = "tests/fixtures/caid3_mini.fasta"
        proteins = parse_caid_reference_fasta(path)
        preds = {
            p["id"]: np.array(p["labels"], dtype=np.float32) for p in proteins
        }
        report = evaluate_caid_predictions(proteins, preds)
        assert not report["insufficient_data"]
        assert report["pooled"]["auc"] >= 0.99


class TestStructureEncoder:
    def test_plddt_features_shape(self):
        plddt = np.array([50.0, 85.0, 90.0], dtype=np.float32)
        feats = build_plddt_feature_tensor(plddt, 3)
        assert feats.shape == (3, 2)

    def test_encoder_forward(self):
        enc = PlddtFeatureEncoder(out_dim=8)
        x = torch.randn(2, 10, 2)
        out = enc(x)
        assert out.shape == (2, 10, 8)


class TestNovelUseCases:
    def test_hallucination_screen(self):
        seq = "A" * 20
        probs = np.array([0.8] * 20, dtype=np.float32)
        plddt = np.array([85.0] * 20, dtype=np.float32)
        report = screen_af_hallucinations(seq, probs, plddt, protein_id="X")
        assert report["metrics"]["n_hallucinated"] > 0

    def test_proteome_summary(self):
        proteins = [{"id": "P1", "length": 10}, {"id": "P2", "length": 10}]
        preds = {
            "P1": np.array([0.9] * 10, dtype=np.float32),
            "P2": np.array([0.1] * 10, dtype=np.float32),
        }
        summary = proteome_disorder_summary(proteins, preds)
        assert summary["n_proteins"] == 2
        assert summary["high_disorder_proteins"][0]["id"] == "P1"
