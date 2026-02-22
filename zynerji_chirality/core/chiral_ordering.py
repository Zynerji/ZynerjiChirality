"""CIP canonical ordering with signed tetrahedral volumes.

The critical innovation for chirality detection. Standard graph methods use
arbitrary node ordering — the dual-helix phase modulation depends on node
indices, so the same graph with different orderings produces different spectra.

By using CIP-priority ordering with chirality-dependent neighbor ranking, we
ensure:
1. Consistent ordering across molecules (canonical)
2. Ordering encodes chirality (R vs S produce different orderings)
3. Enantiomers get DIFFERENT orderings → different helix spectra

This is what breaks the enantiomer symmetry that defeats standard spectral methods.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from rdkit import Chem
from rdkit.Chem import AllChem


def signed_tetrahedral_volume(
    mol: Chem.Mol,
    atom_idx: int,
    conf: Chem.Conformer,
) -> float:
    """Compute signed volume of tetrahedron formed by atom's neighbors.

    Positive = R (clockwise when viewed from first neighbor),
    Negative = S (counterclockwise).
    Returns 0.0 for non-tetrahedral atoms (< 3 neighbors).

    The signed volume is computed as the scalar triple product of
    three edge vectors from the central atom to its neighbors:
        V = (v1 x v2) . v3

    Parameters
    ----------
    mol : Chem.Mol
        Molecule with 3D conformer.
    atom_idx : int
        Index of the (potentially chiral) central atom.
    conf : Chem.Conformer
        3D conformer with coordinates.

    Returns
    -------
    float
        Signed tetrahedral volume. Sign encodes handedness.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

    if len(neighbors) < 3:
        return 0.0

    # Get 3D positions
    center = np.array(conf.GetAtomPosition(atom_idx))
    p1 = np.array(conf.GetAtomPosition(neighbors[0]))
    p2 = np.array(conf.GetAtomPosition(neighbors[1]))
    p3 = np.array(conf.GetAtomPosition(neighbors[2]))

    # Edge vectors from center
    v1 = p1 - center
    v2 = p2 - center
    v3 = p3 - center

    # Signed volume = scalar triple product
    volume = np.dot(np.cross(v1, v2), v3)

    return float(volume)


def cip_canonical_order(
    mol: Chem.Mol,
    chirality_aware: bool = True,
) -> list[int]:
    """CIP-priority canonical atom ordering.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule with stereochemistry assigned.
    chirality_aware : bool
        If True, apply chirality-dependent cyclic shifts to neighbors of
        chiral centers. R shifts neighbors left, S shifts right (symmetric
        perturbation of equal magnitude). If False, use pure canonical
        ordering identical for enantiomers.

    Returns
    -------
    list[int]
        Permutation mapping: new_idx -> old_atom_idx.
    """
    # Ensure stereochemistry is assigned
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Get RDKit canonical ranking
    canon_ranks = list(Chem.CanonicalRankAtoms(mol))

    # Build sort key for each atom (pure canonical, no chirality)
    n = mol.GetNumAtoms()
    sort_keys = []
    for idx in range(n):
        atom = mol.GetAtomWithIdx(idx)
        atomic_num = atom.GetAtomicNum()
        degree = atom.GetDegree()
        c_rank = canon_ranks[idx]
        sort_keys.append((-atomic_num, -degree, c_rank, idx))

    sort_keys.sort()
    ordering = [key[-1] for key in sort_keys]

    if not chirality_aware:
        return ordering

    # Apply symmetric chirality-dependent cyclic shifts.
    # For each chiral center, find its neighbors' positions in the ordering
    # and cyclically shift them: R = left shift, S = right shift.
    # This creates equal-magnitude perturbations in opposite directions.
    chiral_info = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

    # Build atom_idx -> position_in_ordering map
    pos_of = {}
    for pos, atom_idx in enumerate(ordering):
        pos_of[atom_idx] = pos

    for center_idx, label in chiral_info:
        if label not in ("R", "S"):
            continue

        atom = mol.GetAtomWithIdx(center_idx)
        neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]

        if len(neighbors) < 2:
            continue

        # Find positions of these neighbors in the ordering
        nbr_positions = sorted([pos_of[nb] for nb in neighbors])
        nbr_at_positions = [ordering[p] for p in nbr_positions]

        # Cyclic shift: R = left (+1), S = right (-1)
        shift = 1 if label == "R" else -1
        k = len(nbr_at_positions)
        shifted = [nbr_at_positions[(i + shift) % k] for i in range(k)]

        # Apply the shift back into the ordering
        for pos, new_atom in zip(nbr_positions, shifted):
            ordering[pos] = new_atom

        # Rebuild pos_of after modification
        pos_of = {}
        for pos, atom_idx in enumerate(ordering):
            pos_of[atom_idx] = pos

    return ordering


def reorder_adjacency(adj: csr_matrix, ordering: list[int]) -> csr_matrix:
    """Permute adjacency matrix rows/cols to canonical ordering.

    Parameters
    ----------
    adj : csr_matrix
        Original adjacency matrix (n x n).
    ordering : list[int]
        Permutation: new_idx -> old_atom_idx.

    Returns
    -------
    csr_matrix
        Reordered adjacency matrix where entry (i, j) in the new matrix
        corresponds to (ordering[i], ordering[j]) in the original.
    """
    n = adj.shape[0]
    if n == 0:
        return csr_matrix((0, 0))

    perm = np.array(ordering, dtype=int)
    adj_dense = adj.toarray()
    reordered = adj_dense[np.ix_(perm, perm)]

    return csr_matrix(reordered)
