"""Tests for fingerprint.py — spectral chirality fingerprints."""

import numpy as np
import pytest

from zynerji_chirality.chirality.fingerprint import (
    chirality_fingerprint,
    chirality_differential_fingerprint,
    batch_differential_fingerprint,
    fingerprint_similarity,
)


class TestChiralityFingerprint:
    def test_output_shape(self):
        fp = chirality_fingerprint("N[C@@H](C)C(=O)O")
        assert fp.shape == (128,)

    def test_custom_nbits(self):
        fp = chirality_fingerprint("N[C@@H](C)C(=O)O", nbits=64)
        assert fp.shape == (64,)

    def test_deterministic(self):
        fp1 = chirality_fingerprint("N[C@@H](C)C(=O)O")
        fp2 = chirality_fingerprint("N[C@@H](C)C(=O)O")
        np.testing.assert_array_almost_equal(fp1, fp2)

    def test_enantiomers_different(self):
        fp_l = chirality_fingerprint("N[C@@H](C)C(=O)O")
        fp_d = chirality_fingerprint("N[C@H](C)C(=O)O")
        # Enantiomers should produce different fingerprints
        assert not np.allclose(fp_l, fp_d)

    def test_achiral_vs_chiral_different(self):
        fp_gly = chirality_fingerprint("NCC(=O)O")
        fp_ala = chirality_fingerprint("N[C@@H](C)C(=O)O")
        assert not np.allclose(fp_gly, fp_ala)


class TestFingerprintSimilarity:
    def test_identical_is_one(self):
        fp = chirality_fingerprint("N[C@@H](C)C(=O)O")
        sim = fingerprint_similarity(fp, fp)
        assert abs(sim - 1.0) < 1e-6

    def test_range_zero_to_one(self):
        fp1 = chirality_fingerprint("N[C@@H](C)C(=O)O")
        fp2 = chirality_fingerprint("NCC(=O)O")
        sim = fingerprint_similarity(fp1, fp2)
        assert 0.0 <= sim <= 1.0

    def test_zero_vector(self):
        fp = chirality_fingerprint("N[C@@H](C)C(=O)O")
        zero = np.zeros_like(fp)
        sim = fingerprint_similarity(fp, zero)
        assert sim == 0.0

    def test_enantiomers_similarity(self):
        fp_l = chirality_fingerprint("N[C@@H](C)C(=O)O")
        fp_d = chirality_fingerprint("N[C@H](C)C(=O)O")
        sim = fingerprint_similarity(fp_l, fp_d)
        # Enantiomers should be somewhat similar (same structure)
        # but not identical (different handedness)
        assert 0.0 < sim < 1.0


class TestDifferentialFingerprint:
    """Tests for chirality_differential_fingerprint() — chirality-only FP."""

    def test_output_shape(self):
        fp = chirality_differential_fingerprint("N[C@@H](C)C(=O)O")
        assert fp.shape == (128,)

    def test_custom_nbits(self):
        fp = chirality_differential_fingerprint("N[C@@H](C)C(=O)O", nbits=64)
        assert fp.shape == (64,)

    def test_deterministic(self):
        fp1 = chirality_differential_fingerprint("N[C@@H](C)C(=O)O")
        fp2 = chirality_differential_fingerprint("N[C@@H](C)C(=O)O")
        np.testing.assert_array_almost_equal(fp1, fp2)

    def test_enantiomers_anti_correlated(self):
        """Core property: enantiomer differential FPs should have cosine < 0.5."""
        fp_l = chirality_differential_fingerprint("N[C@@H](C)C(=O)O")  # L-Ala
        fp_d = chirality_differential_fingerprint("N[C@H](C)C(=O)O")   # D-Ala
        cos_sim = float(np.dot(fp_l, fp_d) / (
            np.linalg.norm(fp_l) * np.linalg.norm(fp_d)
        ))
        # Differential FP should be anti-correlated for enantiomers
        assert cos_sim < 0.5, f"Expected cosine < 0.5, got {cos_sim:.4f}"

    def test_ibuprofen_enantiomers_anti_correlated(self):
        """R/S ibuprofen differential FPs should be anti-correlated."""
        fp_r = chirality_differential_fingerprint("CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1")
        fp_s = chirality_differential_fingerprint("CC(C)Cc1ccc([C@H](C)C(=O)O)cc1")
        cos_sim = float(np.dot(fp_r, fp_s) / (
            np.linalg.norm(fp_r) * np.linalg.norm(fp_s)
        ))
        assert cos_sim < 0.5, f"Ibuprofen enantiomers cosine {cos_sim:.4f}"

    def test_achiral_near_zero_norm(self):
        """Achiral molecules should have near-zero differential FP norm."""
        fp = chirality_differential_fingerprint("NCC(=O)O")  # glycine
        norm = np.linalg.norm(fp)
        # Achiral: no chiral modulation → diff_cos = diff_sin = 0 → fp ≈ 0
        assert norm < 1e-6, f"Achiral norm should be ~0, got {norm:.6f}"

    def test_mode_chi_diff(self):
        """chi_diff mode should produce valid fingerprint with different content."""
        fp_pure = chirality_differential_fingerprint(
            "N[C@@H](C)C(=O)O", mode="pure_diff"
        )
        fp_chi = chirality_differential_fingerprint(
            "N[C@@H](C)C(=O)O", mode="chi_diff"
        )
        assert fp_chi.shape == (128,)
        # Different modes should generally give different vectors
        # (chi_diff includes asym_chi which adds non-differential signal)
        assert not np.allclose(fp_pure, fp_chi)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            chirality_differential_fingerprint("CC", mode="invalid")

    def test_batch_differential(self):
        smiles = [
            "N[C@@H](C)C(=O)O",
            "N[C@H](C)C(=O)O",
            "NCC(=O)O",
        ]
        fps = batch_differential_fingerprint(smiles)
        assert len(fps) == 3
        assert all(fp is not None for fp in fps)
        assert all(fp.shape == (128,) for fp in fps)

    def test_camphor_enantiomers(self):
        """Camphor (bridged bicyclic) enantiomers differential FP."""
        # (+)-camphor and (-)-camphor
        fp_r = chirality_differential_fingerprint(
            "[C@@H]1(C(=O)C2(C)C)C[C@H]2CC1C"
        )
        fp_s = chirality_differential_fingerprint(
            "[C@H]1(C(=O)C2(C)C)C[C@@H]2CC1C"
        )
        n1, n2 = np.linalg.norm(fp_r), np.linalg.norm(fp_s)
        if n1 > 1e-8 and n2 > 1e-8:
            cos_sim = float(np.dot(fp_r, fp_s) / (n1 * n2))
            assert cos_sim < 0.5, f"Camphor enantiomers cosine {cos_sim:.4f}"
