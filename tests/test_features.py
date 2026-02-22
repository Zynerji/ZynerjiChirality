"""Tests for ML feature pipeline."""

import numpy as np
import pytest

from zynerji_chirality.ml.features import build_feature_vector, descriptor_names


class TestBuildFeatureVector:
    def test_basic_output(self):
        """Basic call returns expected keys."""
        features = build_feature_vector("N[C@@H](C)C(=O)O")
        assert "spectral_fingerprint" in features
        assert "rdkit_descriptors" in features
        assert "global_chirality" in features
        assert "combined" in features

    def test_spectral_fingerprint_shape(self):
        """Spectral fingerprint should be 128d by default."""
        features = build_feature_vector("N[C@@H](C)C(=O)O")
        assert features["spectral_fingerprint"].shape == (128,)

    def test_rdkit_descriptors_shape(self):
        """RDKit descriptors should be 10d."""
        features = build_feature_vector("N[C@@H](C)C(=O)O")
        assert features["rdkit_descriptors"].shape == (10,)

    def test_global_chirality_shape(self):
        """Global chirality should be 2d (score, sign)."""
        features = build_feature_vector("N[C@@H](C)C(=O)O")
        assert features["global_chirality"].shape == (2,)

    def test_combined_is_concatenation(self):
        """Combined vector should be concatenation of all features."""
        features = build_feature_vector("CCO")
        expected_len = sum(
            v.shape[0] for k, v in features.items() if k != "combined"
        )
        assert features["combined"].shape[0] == expected_len

    def test_per_center_scores(self):
        """With per-center enabled, should include per_center_scores."""
        features = build_feature_vector("N[C@@H](C)C(=O)O", include_per_center=True)
        assert "per_center_scores" in features
        assert features["per_center_scores"].shape == (16,)  # 8 centers * 2

    def test_ensemble_stats(self):
        """With ensemble enabled, should include ensemble_stats."""
        features = build_feature_vector(
            "N[C@@H](C)C(=O)O",
            include_ensemble=True,
            n_conformers=3,
        )
        assert "ensemble_stats" in features
        assert features["ensemble_stats"].shape == (4,)

    def test_achiral_molecule(self):
        """Achiral molecule should have zero chirality score."""
        features = build_feature_vector("CCO")
        assert features["global_chirality"][0] == 0.0  # score
        assert features["global_chirality"][1] == 0.0  # sign

    def test_chiral_molecule(self):
        """Chiral molecule should have nonzero chirality score."""
        features = build_feature_vector("N[C@@H](C)C(=O)O")
        assert features["global_chirality"][0] > 0.0


class TestDescriptorNames:
    def test_length(self):
        """Should return 10 descriptor names."""
        assert len(descriptor_names()) == 10

    def test_names(self):
        """Should include common descriptors."""
        names = descriptor_names()
        assert "MW" in names
        assert "LogP" in names
        assert "TPSA" in names
