"""Tests for detector.py — chirality detection."""

import numpy as np
import pytest

from zynerji_chirality.chirality.detector import (
    HelixChiralityDetector,
    ChiralityResult,
    EnantiomerComparison,
    EnsembleResult,
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


class TestEnsembleDetection:
    def test_ensemble_chiral(self, detector):
        """Ensemble on chiral molecule should show consensus chiral."""
        result = detector.detect_ensemble("N[C@@H](C)C(=O)O", n_conformers=5)
        assert isinstance(result, EnsembleResult)
        assert result.consensus_is_chiral
        assert result.n_conformers == 5
        assert result.mean_score > 0
        assert result.std_score >= 0
        assert len(result.per_conformer_results) == 5

    def test_ensemble_achiral(self, detector):
        """Ensemble on achiral molecule should show consensus achiral."""
        result = detector.detect_ensemble("NCC(=O)O", n_conformers=5)
        assert not result.consensus_is_chiral
        assert result.mean_score == 0.0

    def test_ensemble_min_max(self, detector):
        """Min/max scores should be within per-conformer range."""
        result = detector.detect_ensemble("N[C@@H](C)C(=O)O", n_conformers=5)
        assert result.min_score <= result.mean_score <= result.max_score

    def test_ensemble_repr(self, detector):
        result = detector.detect_ensemble("N[C@@H](C)C(=O)O", n_conformers=3)
        s = repr(result)
        assert "CHIRAL" in s or "ACHIRAL" in s
        assert "n=3" in s


class TestPerCenterDetection:
    def test_single_center_matches_global(self, detector):
        """Single-center molecule: per-center result should match global."""
        global_res = detector.detect("N[C@@H](C)C(=O)O")
        per_center = detector.detect_per_center("N[C@@H](C)C(=O)O")
        assert len(per_center) == 1
        center_idx = list(per_center.keys())[0]
        assert per_center[center_idx].is_chiral == global_res.is_chiral

    def test_multi_center_has_multiple_entries(self, detector):
        """Multi-center molecule should have an entry per center."""
        # Isoleucine: 2 chiral centers
        per_center = detector.detect_per_center("N[C@@H]([C@@H](C)CC)C(=O)O")
        assert len(per_center) >= 2

    def test_achiral_returns_empty(self, detector):
        """Achiral molecule should return empty dict."""
        per_center = detector.detect_per_center("NCC(=O)O")
        assert per_center == {}


class TestMesoDetection:
    def test_meso_tartaric_acid_achiral(self, detector):
        """Meso-tartaric acid should be detected as achiral."""
        result = detector.detect("O[C@H](C(=O)O)[C@@H](O)C(=O)O")
        assert not result.is_chiral
        assert result.chirality_score == 0.0

    def test_meso_butanediol_achiral(self, detector):
        """Meso-2,3-butanediol should be detected as achiral."""
        result = detector.detect("C[C@@H](O)[C@@H](O)C")
        assert not result.is_chiral
        assert result.chirality_score == 0.0

    def test_non_meso_multi_center_still_chiral(self, detector):
        """Multi-center molecules that aren't meso should still be chiral."""
        # L-Isoleucine has 2 chiral centers (both S), not meso
        result = detector.detect("N[C@@H]([C@@H](C)CC)C(=O)O")
        assert result.is_chiral


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
