"""Tests for detector.py — chirality detection."""

import numpy as np
import pytest

from zynerji_chirality.chirality.detector import (
    HelixChiralityDetector,
    ChiralityResult,
    EnantiomerComparison,
)


@pytest.fixture
def detector():
    return HelixChiralityDetector()


class TestChiralityDetection:
    def test_alanine_l_is_chiral(self, detector):
        result = detector.detect("N[C@@H](C)C(=O)O")
        assert isinstance(result, ChiralityResult)
        assert result.is_chiral
        assert result.chirality_score > 0

    def test_alanine_d_is_chiral(self, detector):
        result = detector.detect("N[C@H](C)C(=O)O")
        assert result.is_chiral
        assert result.chirality_score > 0

    def test_glycine_is_achiral(self, detector):
        result = detector.detect("NCC(=O)O")
        assert not result.is_chiral
        assert result.chirality_score < detector.threshold

    def test_enantiomers_opposite_sign(self, detector):
        """L and D alanine should have opposite chirality signs."""
        l_res = detector.detect("N[C@@H](C)C(=O)O")
        d_res = detector.detect("N[C@H](C)C(=O)O")

        assert l_res.is_chiral and d_res.is_chiral
        assert l_res.chirality_sign * d_res.chirality_sign < 0, \
            f"Expected opposite signs, got L={l_res.chirality_sign}, D={d_res.chirality_sign}"

    def test_eigenvalues_present(self, detector):
        result = detector.detect("N[C@@H](C)C(=O)O")
        assert len(result.cos_eigenvalues) > 0
        assert len(result.sin_eigenvalues) > 0

    def test_spectral_coords_shape(self, detector):
        result = detector.detect("N[C@@H](C)C(=O)O")
        coords = result.spectral_coords.coords
        assert coords.ndim == 2
        assert coords.shape[0] > 0  # n_atoms

    def test_confidence_positive_for_chiral(self, detector):
        result = detector.detect("N[C@@H](C)C(=O)O")
        assert result.confidence > 0

    def test_confidence_zero_for_achiral(self, detector):
        result = detector.detect("NCC(=O)O")
        assert result.confidence == 0.0

    def test_repr(self, detector):
        result = detector.detect("N[C@@H](C)C(=O)O")
        s = repr(result)
        assert "CHIRAL" in s or "ACHIRAL" in s


class TestEnantiomerComparison:
    def test_compare_alanine(self, detector):
        comp = detector.compare_enantiomers(
            "N[C@@H](C)C(=O)O",
            "N[C@H](C)C(=O)O",
        )
        assert isinstance(comp, EnantiomerComparison)
        assert comp.result_a.is_chiral
        assert comp.result_b.is_chiral
        assert comp.signs_opposite
        assert comp.are_enantiomers

    def test_same_molecule_not_enantiomers(self, detector):
        comp = detector.compare_enantiomers(
            "N[C@@H](C)C(=O)O",
            "N[C@@H](C)C(=O)O",
        )
        # Same molecule should NOT be classified as enantiomers
        assert not comp.signs_opposite

    def test_achiral_pair_not_enantiomers(self, detector):
        comp = detector.compare_enantiomers("CCO", "CCO")
        assert not comp.are_enantiomers

    def test_repr(self, detector):
        comp = detector.compare_enantiomers(
            "N[C@@H](C)C(=O)O",
            "N[C@H](C)C(=O)O",
        )
        s = repr(comp)
        assert "ENANTIOMERS" in s


class TestClassifyRS:
    def test_classify_achiral(self, detector):
        assert detector.classify_rs("NCC(=O)O") == "achiral"

    def test_classify_single_center(self, detector):
        cls = detector.classify_rs("N[C@@H](C)C(=O)O")
        assert cls in ("R", "S")

    def test_classify_multiple_centers(self, detector):
        # Isoleucine has 2 chiral centers
        cls = detector.classify_rs("N[C@@H]([C@@H](C)CC)C(=O)O")
        assert cls == "multiple_centers"


class TestAsymmetryComputation:
    def test_zero_for_identical(self):
        evals = np.array([1.0, 2.0, 3.0])
        score, sign = HelixChiralityDetector._compute_asymmetry(evals, evals)
        assert score == 0.0

    def test_nonzero_for_different(self):
        ec = np.array([1.0, 2.0, 3.0])
        es = np.array([1.1, 2.2, 2.8])
        score, sign = HelixChiralityDetector._compute_asymmetry(ec, es)
        assert score > 0

    def test_empty_arrays(self):
        score, sign = HelixChiralityDetector._compute_asymmetry(
            np.array([]), np.array([])
        )
        assert score == 0.0
        assert sign == 0.0
