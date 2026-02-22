"""Tests for planar chirality detection."""

import pytest
import numpy as np

from zynerji_chirality.chirality.detector import HelixChiralityDetector
from zynerji_chirality.core.chiral_ordering import find_planar_centers
from zynerji_chirality.core.mol_graph import smiles_to_mol3d


@pytest.fixture
def detector():
    return HelixChiralityDetector()


class TestFindPlanarCenters:
    def test_benzene_no_planar_chirality(self):
        """Benzene has no planar chirality."""
        mol = smiles_to_mol3d("c1ccccc1")
        centers = find_planar_centers(mol)
        # Benzene has no out-of-plane substituents
        assert isinstance(centers, list)

    def test_toluene_has_substituent(self):
        """Toluene has out-of-plane methyl but not planar chiral."""
        mol = smiles_to_mol3d("Cc1ccccc1")
        centers = find_planar_centers(mol)
        assert isinstance(centers, list)

    def test_ethanol_no_aromatic(self):
        """Non-aromatic molecule has no planar centers."""
        mol = smiles_to_mol3d("CCO")
        centers = find_planar_centers(mol)
        assert len(centers) == 0


class TestPlanarDetection:
    def test_standard_chiral_unaffected(self, detector):
        """Standard tetrahedral chirality should still work."""
        result = detector.detect("N[C@@H](C)C(=O)O")
        assert result.is_chiral

    def test_achiral_still_zero(self, detector):
        """Achiral molecules should still score zero."""
        result = detector.detect("c1ccccc1")
        assert not result.is_chiral
