"""Distance bounds computation from molecular graph.

Extracts bond connectivity from RDKit Mol, computes lower/upper distance
bounds for all atom pairs using bond lengths, angles, and VDW radii.

The bounds matrix is the input to the GPU distance geometry solver.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem

import logging

logger = logging.getLogger(__name__)


# Standard bond lengths (Angstroms) keyed by (atomic_num_lo, atomic_num_hi, bond_order)
# bond_order: 1=single, 1.5=aromatic, 2=double, 3=triple
_BOND_LENGTHS: dict[tuple[int, int, float], float] = {
    # C-C
    (6, 6, 1.0): 1.54, (6, 6, 1.5): 1.40, (6, 6, 2.0): 1.34, (6, 6, 3.0): 1.20,
    # C-H
    (1, 6, 1.0): 1.09,
    # C-N
    (6, 7, 1.0): 1.47, (6, 7, 1.5): 1.34, (6, 7, 2.0): 1.29, (6, 7, 3.0): 1.16,
    # C-O
    (6, 8, 1.0): 1.43, (6, 8, 1.5): 1.36, (6, 8, 2.0): 1.23,
    # C-S
    (6, 16, 1.0): 1.82, (6, 16, 2.0): 1.60,
    # C-F, C-Cl, C-Br, C-I
    (6, 9, 1.0): 1.35, (6, 17, 1.0): 1.77, (6, 35, 1.0): 1.94, (6, 53, 1.0): 2.14,
    # C-P
    (6, 15, 1.0): 1.84,
    # N-H
    (1, 7, 1.0): 1.01,
    # N-N
    (7, 7, 1.0): 1.45, (7, 7, 2.0): 1.25, (7, 7, 3.0): 1.10,
    # N-O
    (7, 8, 1.0): 1.40, (7, 8, 2.0): 1.21,
    # O-H
    (1, 8, 1.0): 0.96,
    # O-O
    (8, 8, 1.0): 1.48,
    # S-S
    (16, 16, 1.0): 2.05,
    # S-H
    (1, 16, 1.0): 1.34,
    # S-O
    (8, 16, 2.0): 1.43,
    # P-O
    (8, 15, 1.0): 1.63, (8, 15, 2.0): 1.48,
    # P-N
    (7, 15, 1.0): 1.68,
    # Si-C, Si-O
    (6, 14, 1.0): 1.87, (8, 14, 1.0): 1.63,
}

# VDW radii (Angstroms)
_VDW_RADII: dict[int, float] = {
    1: 1.20, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
}

# Typical bond angles (radians)
_SP3_ANGLE = np.radians(109.5)
_SP2_ANGLE = np.radians(120.0)
_SP_ANGLE = np.radians(180.0)


def _get_bond_length(anum1: int, anum2: int, bond_order: float) -> float:
    """Look up bond length, with fallback for unknown pairs."""
    lo, hi = min(anum1, anum2), max(anum1, anum2)
    length = _BOND_LENGTHS.get((lo, hi, bond_order))
    if length is not None:
        return length
    # Fallback: try closest bond order
    for bo in [1.0, 1.5, 2.0, 3.0]:
        length = _BOND_LENGTHS.get((lo, hi, bo))
        if length is not None:
            return length
    # Generic fallback from sum of covalent radii
    covalent = {1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
                14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39}
    r1 = covalent.get(anum1, 1.0)
    r2 = covalent.get(anum2, 1.0)
    return r1 + r2


def _get_vdw_radius(anum: int) -> float:
    """Get VDW radius with fallback."""
    return _VDW_RADII.get(anum, 1.70)


def _get_angle(hybridization) -> float:
    """Get expected bond angle from hybridization."""
    from rdkit.Chem import HybridizationType
    if hybridization == HybridizationType.SP3:
        return _SP3_ANGLE
    elif hybridization == HybridizationType.SP2:
        return _SP2_ANGLE
    elif hybridization == HybridizationType.SP:
        return _SP_ANGLE
    return _SP3_ANGLE  # default


@dataclass
class MolecularGraph:
    """Extracted molecular graph info for distance geometry."""
    n_atoms: int
    atomic_nums: np.ndarray          # (n_atoms,) int
    bonds: list[tuple[int, int, float]]  # (atom_i, atom_j, bond_order)
    bond_lengths: np.ndarray         # (n_bonds,) float64
    hybridizations: list             # RDKit hybridization per atom
    chiral_centers: list[tuple[int, list[int], str]]  # (center_idx, neighbor_idxs, R/S)
    adjacency_list: dict[int, list[int]] = field(default_factory=dict)


def extract_graph(mol: Chem.Mol) -> MolecularGraph:
    """Extract molecular graph from RDKit Mol (no 3D coords needed)."""
    n = mol.GetNumAtoms()
    atomic_nums = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)])
    hybridizations = [mol.GetAtomWithIdx(i).GetHybridization() for i in range(n)]

    bonds = []
    bond_lengths_list = []
    adj_list = defaultdict(list)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bo = bond.GetBondTypeAsDouble()
        bl = _get_bond_length(int(atomic_nums[i]), int(atomic_nums[j]), bo)
        bonds.append((i, j, bo))
        bond_lengths_list.append(bl)
        adj_list[i].append(j)
        adj_list[j].append(i)

    # Chiral centers
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    chiral_info = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    chiral_centers = []
    for atom_idx, label in chiral_info:
        if label in ("R", "S"):
            neighbors = [nb.GetIdx() for nb in mol.GetAtomWithIdx(atom_idx).GetNeighbors()]
            chiral_centers.append((atom_idx, neighbors, label))

    return MolecularGraph(
        n_atoms=n,
        atomic_nums=atomic_nums,
        bonds=bonds,
        bond_lengths=np.array(bond_lengths_list, dtype=np.float64),
        hybridizations=hybridizations,
        chiral_centers=chiral_centers,
        adjacency_list=dict(adj_list),
    )


def compute_bounds(graph: MolecularGraph) -> tuple[np.ndarray, np.ndarray]:
    """Compute distance bounds matrix from molecular graph.

    Returns (lower, upper) where:
    - lower[i][j]: minimum possible distance between atoms i and j
    - upper[i][j]: maximum possible distance between atoms i and j

    Uses bond lengths for 1,2 pairs, angle constraints for 1,3 pairs,
    and VDW radii for lower bounds of non-bonded pairs.
    """
    n = graph.n_atoms

    # Initialize bounds
    lower = np.zeros((n, n), dtype=np.float64)
    upper = np.full((n, n), 1000.0, dtype=np.float64)  # large upper bound
    np.fill_diagonal(upper, 0.0)

    # 1,2 pairs (directly bonded)
    for idx, (i, j, bo) in enumerate(graph.bonds):
        bl = graph.bond_lengths[idx]
        tol = 0.05  # small tolerance
        lower[i][j] = lower[j][i] = bl - tol
        upper[i][j] = upper[j][i] = bl + tol

    # 1,3 pairs (two bonds apart, angle constraint)
    for center in range(n):
        neighbors = graph.adjacency_list.get(center, [])
        angle = _get_angle(graph.hybridizations[center])

        for ni in range(len(neighbors)):
            for nj in range(ni + 1, len(neighbors)):
                a, b = neighbors[ni], neighbors[nj]
                # Get bond lengths a-center and center-b
                d_ac = _find_bond_length(graph, a, center)
                d_cb = _find_bond_length(graph, center, b)

                if d_ac is None or d_cb is None:
                    continue

                # Law of cosines: d_ab^2 = d_ac^2 + d_cb^2 - 2*d_ac*d_cb*cos(angle)
                d_ab = np.sqrt(d_ac**2 + d_cb**2 - 2 * d_ac * d_cb * np.cos(angle))

                # Set tighter bounds for 1,3 pairs
                tol_13 = 0.2
                lb = max(lower[a][b], d_ab - tol_13)
                ub = min(upper[a][b], d_ab + tol_13)

                if lb <= ub:
                    lower[a][b] = lower[b][a] = lb
                    upper[a][b] = upper[b][a] = ub

    # Non-bonded lower bounds from VDW radii
    for i in range(n):
        for j in range(i + 1, n):
            vdw_lb = _get_vdw_radius(int(graph.atomic_nums[i])) + \
                     _get_vdw_radius(int(graph.atomic_nums[j])) - 0.5
            # Don't override tighter bounds from bonds/angles
            if lower[i][j] < vdw_lb and upper[i][j] > vdw_lb:
                lower[i][j] = lower[j][i] = max(lower[i][j], vdw_lb * 0.7)

    # Shortest path upper bounds (graph-theoretic)
    _apply_shortest_path_upper(graph, upper)

    # Ensure consistency
    np.maximum(lower, 0.0, out=lower)
    np.fill_diagonal(lower, 0.0)
    np.fill_diagonal(upper, 0.0)

    return lower, upper


def _find_bond_length(graph: MolecularGraph, a: int, b: int) -> float | None:
    """Find bond length between atoms a and b."""
    for idx, (i, j, bo) in enumerate(graph.bonds):
        if (i == a and j == b) or (i == b and j == a):
            return graph.bond_lengths[idx]
    return None


def _apply_shortest_path_upper(graph: MolecularGraph, upper: np.ndarray):
    """Set upper bounds from shortest bond-path distances."""
    n = graph.n_atoms
    # BFS from each atom to compute shortest path lengths
    for start in range(n):
        visited = {start: 0.0}
        queue = [(start, 0.0)]
        qi = 0
        while qi < len(queue):
            node, dist = queue[qi]
            qi += 1
            for nb in graph.adjacency_list.get(node, []):
                bl = _find_bond_length(graph, node, nb) or 1.5
                new_dist = dist + bl
                if nb not in visited or new_dist < visited[nb]:
                    visited[nb] = new_dist
                    queue.append((nb, new_dist))

        for end, path_len in visited.items():
            if end != start:
                upper[start][end] = min(upper[start][end], path_len * 1.1)
                upper[end][start] = upper[start][end]


def try_rdkit_bounds(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to get bounds matrix from RDKit (fast, chemically accurate).

    Returns (lower, upper) or None if RDKit fails.
    RDKit's GetMoleculeBoundsMatrix returns a combined matrix where
    lower triangle = lower bounds, upper triangle = upper bounds.
    """
    try:
        from rdkit.Chem import rdDistGeom
        bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
        n = mol.GetNumAtoms()
        lower = np.zeros((n, n), dtype=np.float64)
        upper = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                upper[i][j] = upper[j][i] = bm[i][j]
                lower[i][j] = lower[j][i] = bm[j][i]
        return lower, upper
    except Exception as e:
        logger.debug("RDKit bounds failed: %s", e)
        return None
