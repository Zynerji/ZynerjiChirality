"""RDKit molecule to sparse adjacency matrix conversion.

Converts RDKit Mol objects to weighted scipy sparse matrices suitable
for dual-helix spectral decomposition.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops


# Bond type to weight mapping
_BOND_ORDER = {
    Chem.BondType.SINGLE: 1.0,
    Chem.BondType.AROMATIC: 1.5,
    Chem.BondType.DOUBLE: 2.0,
    Chem.BondType.TRIPLE: 3.0,
}


def mol_to_adjacency(mol: Chem.Mol, weight_mode: str = "bond_order") -> csr_matrix:
    """Convert RDKit Mol to weighted adjacency matrix.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object.
    weight_mode : str
        "binary"     -- 0/1 connectivity
        "bond_order" -- 1.0/1.5/2.0/3.0 for single/aromatic/double/triple
        "distance"   -- 1/d(i,j) using 3D conformer Euclidean distances

    Returns
    -------
    csr_matrix
        Symmetric weighted adjacency matrix (n_atoms x n_atoms).
    """
    n = mol.GetNumAtoms()
    if n == 0:
        return csr_matrix((0, 0))

    if weight_mode == "binary":
        adj_np = Chem.GetAdjacencyMatrix(mol).astype(np.float64)
        return csr_matrix(adj_np)

    if weight_mode == "distance":
        conf = mol.GetConformer()
        rows, cols, vals = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            pi = np.array(conf.GetAtomPosition(i))
            pj = np.array(conf.GetAtomPosition(j))
            d = np.linalg.norm(pi - pj)
            w = 1.0 / max(d, 1e-6)
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([w, w])
        return csr_matrix((vals, (rows, cols)), shape=(n, n))

    # Default: bond_order
    rows, cols, vals = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = _BOND_ORDER.get(bond.GetBondType(), 1.0)
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([w, w])

    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def mol_to_chiral_adjacency(
    mol: Chem.Mol,
    chiral_weight: float = 0.5,
) -> csr_matrix:
    """Bond-order adjacency with chirality-modulated weights.

    Bonds incident to chiral centers get +/- chiral_weight based on
    CIP assignment (R=+, S=-). This encodes handedness directly into
    the graph weights.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule with CIP assignments.
    chiral_weight : float
        Magnitude of chirality modulation on incident bond weights.

    Returns
    -------
    csr_matrix
        Chirality-modulated weighted adjacency matrix.
    """
    n = mol.GetNumAtoms()
    if n == 0:
        return csr_matrix((0, 0))

    # Ensure CIP labels are assigned
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Build chiral center map: atom_idx -> sign (+1 for R, -1 for S)
    chiral_signs = {}
    chiral_info = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    for atom_idx, label in chiral_info:
        if label == "R":
            chiral_signs[atom_idx] = +1.0
        elif label == "S":
            chiral_signs[atom_idx] = -1.0
        # '?' stays absent — no modulation for unassigned

    rows, cols, vals = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = _BOND_ORDER.get(bond.GetBondType(), 1.0)

        # Apply chirality modulation if either endpoint is a chiral center
        mod = 0.0
        if i in chiral_signs:
            mod += chiral_signs[i] * chiral_weight
        if j in chiral_signs:
            mod += chiral_signs[j] * chiral_weight

        w_mod = w + mod

        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([w_mod, w_mod])

    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def smiles_to_mol3d(smiles: str) -> Chem.Mol:
    """Parse SMILES and generate 3D conformer with ETKDG.

    Parameters
    ----------
    smiles : str
        SMILES string (may include chirality annotations like @/@@ or /\\).

    Returns
    -------
    Chem.Mol
        RDKit Mol with 3D coordinates embedded.

    Raises
    ------
    ValueError
        If SMILES cannot be parsed or 3D embedding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    # ETKDG for 3D conformer generation
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status == -1:
        # Fallback: try without ETKDG constraints
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status == -1:
            raise ValueError(f"Cannot embed 3D conformer for: {smiles}")

    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

    # Assign stereochemistry from 3D
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    return mol


def smiles_to_adjacency(
    smiles: str,
    weight_mode: str = "bond_order",
) -> tuple[csr_matrix, Chem.Mol]:
    """Convenience: SMILES string -> adjacency matrix + Mol object.

    Generates 3D conformer via ETKDG for distance mode and
    CIP assignment.
    """
    mol = smiles_to_mol3d(smiles)
    adj = mol_to_adjacency(mol, weight_mode=weight_mode)
    return adj, mol
