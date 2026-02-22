"""Tests for fingerprint.py — spectral chirality fingerprints."""

import numpy as np
import pytest

from zynerji_chirality.chirality.fingerprint import (
    chirality_fingerprint,
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
