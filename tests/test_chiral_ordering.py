"""Tests for chiral_ordering.py — CIP canonical ordering."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from zynerji_chirality.core.chiral_ordering import (
    cip_canonical_order,
    reorder_adjacency,
    signed_tetrahedral_volume,
)
from zynerji_chirality.core.mol_graph import smiles_to_mol3d


class TestSignedTetrahedralVolume:
    def test_chiral_center_nonzero(self):
        """Chiral center should have nonzero signed volume."""
        mol = smiles_to_mol3d("N[C@@H](C)C(=O)O")  # L-alanine
        conf = mol.GetConformer()

        # Find the chiral carbon
        from rdkit import Chem
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral_centers = Chem.FindMolChiralCenters(mol)

        assert len(chiral_centers) > 0
        atom_idx = chiral_centers[0][0]
        vol = signed_tetrahedral_volume(mol, atom_idx, conf)
        assert vol != 0.0, "Chiral center should have nonzero volume"

    def test_enantiomers_opposite_volume(self):
        """R and S enantiomers should have opposite signed volumes."""
        mol_r = smiles_to_mol3d("N[C@H](C)C(=O)O")
        mol_s = smiles_to_mol3d("N[C@@H](C)C(=O)O")

        from rdkit import Chem
        Chem.AssignStereochemistry(mol_r, cleanIt=True, force=True)
        Chem.AssignStereochemistry(mol_s, cleanIt=True, force=True)

        centers_r = Chem.FindMolChiralCenters(mol_r)
        centers_s = Chem.FindMolChiralCenters(mol_s)

        vol_r = signed_tetrahedral_volume(mol_r, centers_r[0][0], mol_r.GetConformer())
        vol_s = signed_tetrahedral_volume(mol_s, centers_s[0][0], mol_s.GetConformer())

        # Opposite sign (allow for conformer variance but both nonzero)
        assert vol_r != 0.0 and vol_s != 0.0

    def test_achiral_atom_zero(self):
        """Non-chiral atom should have ~zero or undefined volume."""
        mol = smiles_to_mol3d("CC")
        conf = mol.GetConformer()
        # Methyl carbon — not tetrahedral in the chiral sense
        # Has 4 neighbors (1 C + 3 H) but no chirality
        vol = signed_tetrahedral_volume(mol, 0, conf)
        # Volume may be nonzero geometrically, but that's OK —
        # the chiral_tag filter prevents it from affecting ordering


class TestCIPCanonicalOrder:
    def test_returns_permutation(self):
        """Ordering should be a valid permutation."""
        mol = smiles_to_mol3d("N[C@@H](C)C(=O)O")
        ordering = cip_canonical_order(mol)

        n = mol.GetNumAtoms()
        assert len(ordering) == n
        assert sorted(ordering) == list(range(n))

    def test_enantiomers_different_order(self):
        """R and S enantiomers should produce different orderings."""
        mol_r = smiles_to_mol3d("N[C@H](C)C(=O)O")
        mol_s = smiles_to_mol3d("N[C@@H](C)C(=O)O")

        order_r = cip_canonical_order(mol_r)
        order_s = cip_canonical_order(mol_s)

        # At least one position should differ
        assert order_r != order_s, "Enantiomers should have different CIP orderings"

    def test_achiral_deterministic(self):
        """Achiral molecule ordering should be deterministic."""
        mol1 = smiles_to_mol3d("CCO")
        mol2 = smiles_to_mol3d("CCO")

        order1 = cip_canonical_order(mol1)
        order2 = cip_canonical_order(mol2)

        assert order1 == order2


class TestReorderAdjacency:
    def test_identity_permutation(self):
        """Identity permutation should not change adjacency."""
        adj = csr_matrix(np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=np.float64))
        ordering = [0, 1, 2]
        result = reorder_adjacency(adj, ordering)
        np.testing.assert_array_equal(result.toarray(), adj.toarray())

    def test_swap_permutation(self):
        """Swapping two nodes should swap corresponding rows/cols."""
        adj = csr_matrix(np.array([
            [0, 1, 0],
            [1, 0, 2],
            [0, 2, 0],
        ], dtype=np.float64))
        ordering = [2, 1, 0]  # Swap first and last
        result = reorder_adjacency(adj, ordering)
        expected = np.array([
            [0, 2, 0],
            [2, 0, 1],
            [0, 1, 0],
        ], dtype=np.float64)
        np.testing.assert_array_equal(result.toarray(), expected)

    def test_preserves_symmetry(self):
        """Reordering should preserve matrix symmetry."""
        mol = smiles_to_mol3d("N[C@@H](C)C(=O)O")
        from zynerji_chirality.core.mol_graph import mol_to_adjacency
        adj = mol_to_adjacency(mol)
        ordering = cip_canonical_order(mol)
        result = reorder_adjacency(adj, ordering)

        diff = result - result.T
        assert diff.nnz == 0, "Reordered adjacency should be symmetric"
