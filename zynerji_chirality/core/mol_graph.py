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


def apply_axial_perturbation(
    adj: csr_matrix,
    mol: Chem.Mol,
    axial_centers: list[tuple[int, int, int, str]],
    direction: float = 1.0,
    perturbation_weight: float = 0.3,
) -> csr_matrix:
    """Modulate edge weights around axial chirality centers.

    For each atropisomeric bond, perturbs the weights of bonds adjacent
    to the axis based on the dihedral geometry, creating asymmetric
    spectral signatures for Ra vs Sa atropisomers.

    Parameters
    ----------
    adj : csr_matrix
        Base adjacency matrix.
    mol : Chem.Mol
        Molecule with 3D coordinates.
    axial_centers : list
        Output from find_axial_centers().
    direction : float
        +1 or -1 to apply perturbation in opposite directions.
    perturbation_weight : float
        Magnitude of the weight perturbation.

    Returns
    -------
    csr_matrix
        Modified adjacency with axial perturbation applied.
    """
    adj_dense = adj.toarray().copy()

    for a_idx, b_idx, bond_idx, label in axial_centers:
        a_atom = mol.GetAtomWithIdx(a_idx)
        b_atom = mol.GetAtomWithIdx(b_idx)

        # Get neighbors of each axis atom (excluding the other axis atom)
        a_neighbors = [nb.GetIdx() for nb in a_atom.GetNeighbors() if nb.GetIdx() != b_idx]
        b_neighbors = [nb.GetIdx() for nb in b_atom.GetNeighbors() if nb.GetIdx() != a_idx]

        # Apply asymmetric perturbation to bonds around the axis
        sign = direction
        for i, nb_idx in enumerate(a_neighbors):
            mod = sign * perturbation_weight * ((-1) ** i)
            adj_dense[a_idx, nb_idx] += mod
            adj_dense[nb_idx, a_idx] += mod

        for i, nb_idx in enumerate(b_neighbors):
            mod = -sign * perturbation_weight * ((-1) ** i)
            adj_dense[b_idx, nb_idx] += mod
            adj_dense[nb_idx, b_idx] += mod

    return csr_matrix(adj_dense)


def smiles_to_mol3d_ensemble(
    smiles: str,
    n_conformers: int = 10,
    random_seed: int = 42,
) -> list[Chem.Mol]:
    """Parse SMILES and generate multiple 3D conformers.

    Each conformer is MMFF-optimized independently. Returns a list of
    Mol objects each with a single conformer embedded.

    Parameters
    ----------
    smiles : str
        SMILES string.
    n_conformers : int
        Number of conformers to generate.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[Chem.Mol]
        List of Mol objects, each with one 3D conformer.

    Raises
    ------
    ValueError
        If SMILES cannot be parsed or no conformers generated.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 1

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if len(conf_ids) == 0:
        raise ValueError(f"Cannot generate conformers for: {smiles}")

    # Optimize each conformer
    for cid in conf_ids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=200)

    # Split into individual Mol objects (one conformer each)
    result = []
    for cid in conf_ids:
        conf_mol = Chem.RWMol(mol)
        # Remove all conformers except this one
        conf_ids_to_remove = [c.GetId() for c in conf_mol.GetConformers() if c.GetId() != cid]
        for rid in conf_ids_to_remove:
            conf_mol.RemoveConformer(rid)
        Chem.AssignStereochemistry(conf_mol, cleanIt=True, force=True)
        result.append(conf_mol.GetMol())

    return result


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
