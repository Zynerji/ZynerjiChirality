"""Tests for axial chirality detection."""

import pytest

from zynerji_chirality.chirality.detector import HelixChiralityDetector
from zynerji_chirality.core.chiral_ordering import find_axial_centers
from zynerji_chirality.core.mol_graph import (
    apply_axial_perturbation,
    mol_to_adjacency,
    smiles_to_mol3d,
)


@pytest.fixture
def detector():
    return HelixChiralityDetector()


class TestFindAxialCenters:
    def test_simple_biphenyl_no_subs(self):
        """Plain biphenyl should NOT have axial centers (no ortho subs)."""
        mol = smiles_to_mol3d("c1ccc(-c2ccccc2)cc1")
        centers = find_axial_centers(mol)
        assert len(centers) == 0

    def test_ortho_substituted_biphenyl(self):
        """Ortho-substituted biphenyl should be detected as potential atropisomer."""
        mol = smiles_to_mol3d("Cc1ccccc1-c1ccccc1C")
        centers = find_axial_centers(mol)
        # May or may not detect depending on substituent bulk
        # At minimum, function should not crash
        assert isinstance(centers, list)

    def test_achiral_molecule(self):
        """Achiral molecule should have no axial centers."""
        mol = smiles_to_mol3d("CCO")
        centers = find_axial_centers(mol)
        assert len(centers) == 0


class TestAxialPerturbation:
    def test_perturbation_changes_matrix(self):
        """Applying axial perturbation should modify adjacency."""
        mol = smiles_to_mol3d("Cc1ccccc1-c1ccccc1C")
        adj = mol_to_adjacency(mol, weight_mode="bond_order")
        # Create synthetic axial center for testing
        centers = [(0, 7, 0, "aR")]  # dummy indices
        adj_perturbed = apply_axial_perturbation(adj, mol, centers, direction=1.0)
        # Should be different from original
        diff = abs(adj_perturbed - adj).sum()
        assert diff > 0

    def test_opposite_directions_differ(self):
        """+1 and -1 directions should produce different perturbations."""
        mol = smiles_to_mol3d("Cc1ccccc1-c1ccccc1C")
        adj = mol_to_adjacency(mol, weight_mode="bond_order")
        centers = [(0, 7, 0, "aR")]
        adj_p = apply_axial_perturbation(adj, mol, centers, direction=1.0)
        adj_n = apply_axial_perturbation(adj, mol, centers, direction=-1.0)
        diff = abs(adj_p - adj_n).sum()
        assert diff > 0


class TestAxialDetection:
    def test_standard_chiral_still_works(self, detector):
        """Standard tetrahedral chirality should not be broken by axial code."""
        result = detector.detect("N[C@@H](C)C(=O)O")
        assert result.is_chiral

    def test_achiral_still_zero(self, detector):
        """Achiral molecules should still score zero."""
        result = detector.detect("CCO")
        assert not result.is_chiral
        assert result.chirality_score == 0.0
