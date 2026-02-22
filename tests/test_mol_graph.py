"""Tests for mol_graph.py — RDKit to sparse adjacency conversion."""

import numpy as np
import pytest
from scipy.sparse import issparse

from zynerji_chirality.core.mol_graph import (
    mol_to_adjacency,
    mol_to_chiral_adjacency,
    smiles_to_adjacency,
    smiles_to_mol3d,
)


class TestMolToAdjacency:
    def test_ethane_binary(self):
        mol = smiles_to_mol3d("CC")
        adj = mol_to_adjacency(mol, weight_mode="binary")
        assert issparse(adj)
        # Ethane with H: 2 C + 6 H = 8 atoms
        assert adj.shape[0] == 8
        # Should be symmetric
        diff = adj - adj.T
        assert diff.nnz == 0

    def test_ethane_bond_order(self):
        mol = smiles_to_mol3d("CC")
        adj = mol_to_adjacency(mol, weight_mode="bond_order")
        assert issparse(adj)
        # All bonds in ethane are single (1.0)
        assert np.allclose(adj.data, 1.0)

    def test_benzene_aromatic(self):
        mol = smiles_to_mol3d("c1ccccc1")
        adj = mol_to_adjacency(mol, weight_mode="bond_order")
        # Benzene should have aromatic bonds (1.5)
        assert 1.5 in adj.data

    def test_ethylene_double(self):
        mol = smiles_to_mol3d("C=C")
        adj = mol_to_adjacency(mol, weight_mode="bond_order")
        assert 2.0 in adj.data

    def test_distance_mode(self):
        mol = smiles_to_mol3d("CC")
        adj = mol_to_adjacency(mol, weight_mode="distance")
        assert issparse(adj)
        # All weights should be positive
        assert np.all(adj.data > 0)

    def test_symmetry(self):
        mol = smiles_to_mol3d("c1ccccc1")
        for mode in ["binary", "bond_order", "distance"]:
            adj = mol_to_adjacency(mol, weight_mode=mode)
            diff = adj - adj.T
            assert diff.nnz == 0, f"Asymmetric adjacency for mode={mode}"


class TestChiralAdjacency:
    def test_alanine_r_vs_s(self):
        """R and S alanine should produce different chiral adjacencies."""
        mol_r = smiles_to_mol3d("N[C@H](C)C(=O)O")
        mol_s = smiles_to_mol3d("N[C@@H](C)C(=O)O")
        adj_r = mol_to_chiral_adjacency(mol_r)
        adj_s = mol_to_chiral_adjacency(mol_s)

        # Both should be sparse and symmetric
        assert issparse(adj_r) and issparse(adj_s)
        assert adj_r.shape == adj_s.shape

        # Chiral adjacencies should differ (different CIP assignments)
        diff = np.abs(adj_r.toarray() - adj_s.toarray()).sum()
        assert diff > 0, "R and S chiral adjacencies should differ"

    def test_achiral_no_modulation(self):
        """Glycine (achiral) chiral adjacency = standard bond order adjacency."""
        mol = smiles_to_mol3d("NCC(=O)O")
        adj_chiral = mol_to_chiral_adjacency(mol)
        adj_std = mol_to_adjacency(mol, weight_mode="bond_order")

        # Should be identical (no chiral centers to modulate)
        diff = np.abs(adj_chiral.toarray() - adj_std.toarray()).sum()
        assert diff < 1e-10, "Achiral molecule should have no chiral modulation"


class TestSmilesToAdjacency:
    def test_returns_tuple(self):
        adj, mol = smiles_to_adjacency("CCO")
        assert issparse(adj)
        assert mol is not None
        assert mol.GetNumAtoms() > 0

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError):
            smiles_to_adjacency("INVALID_SMILES_XYZ")

    def test_3d_conformer_exists(self):
        mol = smiles_to_mol3d("CCO")
        assert mol.GetNumConformers() > 0
