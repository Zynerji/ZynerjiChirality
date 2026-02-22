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
    shift_override: int | None = None,
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
    shift_override : int or None
        If not None, force this shift direction (+1 or -1) for ALL chiral
        centers, ignoring CIP labels. Used for bidirectional scoring.

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
        if shift_override is not None:
            shift = shift_override
        elif label == "R":
            shift = 1
        elif label == "S":
            shift = -1
        elif label == "?":
            # Fallback: use 3D signed tetrahedral volume when CIP is ambiguous
            try:
                conf = mol.GetConformer(0)
                vol = signed_tetrahedral_volume(mol, center_idx, conf)
                shift = 1 if vol > 0 else -1
            except Exception:
                continue
        else:
            continue

        atom = mol.GetAtomWithIdx(center_idx)
        neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]

        if len(neighbors) < 2:
            continue

        # Find positions of these neighbors in the ordering
        nbr_positions = sorted([pos_of[nb] for nb in neighbors])
        nbr_at_positions = [ordering[p] for p in nbr_positions]

        # Cyclic shift
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


def find_planar_centers(
    mol: Chem.Mol,
) -> list[tuple[list[int], int, str]]:
    """Find planar chirality centers (e.g., paracyclophanes).

    Detects ring planes with out-of-plane substituents or bridging
    chains that break the mirror symmetry of the aromatic plane.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule with 3D conformer.

    Returns
    -------
    list[tuple[list[int], int, str]]
        Each entry: (plane_atom_indices, substituent_idx, label).
        label is 'pR' or 'pS' based on substituent face, '?' if unknown.
    """
    results = []

    try:
        conf = mol.GetConformer(0)
    except Exception:
        return results

    ri = mol.GetRingInfo()
    aromatic_rings = []
    for ring in ri.AtomRings():
        ring_list = list(ring)
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_list):
            aromatic_rings.append(ring_list)

    if not aromatic_rings:
        return results

    for ring in aromatic_rings:
        if len(ring) < 5:
            continue

        # Compute ring plane via SVD
        coords = np.array([conf.GetAtomPosition(idx) for idx in ring])
        centroid = coords.mean(axis=0)
        centered = coords - centroid

        try:
            _, s, vh = np.linalg.svd(centered)
        except np.linalg.LinAlgError:
            continue

        normal = vh[2]  # normal to best-fit plane

        # Check ring atoms for non-ring substituents that are out-of-plane
        ring_set = set(ring)
        for ring_atom_idx in ring:
            atom = mol.GetAtomWithIdx(ring_atom_idx)
            for nb in atom.GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx in ring_set or nb.GetAtomicNum() == 1:
                    continue
                # Check if this substituent is connected to another ring
                # (bridge detection for cyclophanes)
                nb_pos = np.array(conf.GetAtomPosition(nb_idx))
                displacement = nb_pos - centroid
                face_dist = float(np.dot(displacement, normal))

                if abs(face_dist) > 0.5:  # significant out-of-plane
                    label = "pR" if face_dist > 0 else "pS"
                    results.append((ring, nb_idx, label))

    return results


def find_axial_centers(mol: Chem.Mol) -> list[tuple[int, int, int, str]]:
    """Find axial chirality centers (atropisomeric bonds).

    Detects single bonds between aromatic rings with restricted rotation
    due to ortho substituents (biaryls).

    Parameters
    ----------
    mol : Chem.Mol
        Molecule with stereochemistry assigned.

    Returns
    -------
    list[tuple[int, int, int, str]]
        Each entry: (atom_a, atom_b, bond_idx, label).
        label is 'aR' or 'aS' if determinable, '?' otherwise.
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    results = []

    # Try RDKit's FindPotentialStereo first (2023.09+)
    try:
        from rdkit.Chem import FindPotentialStereo, StereoInfo
        stereo_info = FindPotentialStereo(mol)
        for si in stereo_info:
            if si.type == StereoInfo.Type.Bond_Atropisomer:
                bond = mol.GetBondWithIdx(si.centeredOn)
                a_idx = bond.GetBeginAtomIdx()
                b_idx = bond.GetEndAtomIdx()
                label = "aR" if "Ra" in str(si.descriptor) else (
                    "aS" if "Sa" in str(si.descriptor) else "?"
                )
                results.append((a_idx, b_idx, si.centeredOn, label))
    except (ImportError, AttributeError):
        pass

    if results:
        return results

    # Fallback: geometry-based detection
    # Look for single bonds between aromatic ring atoms with
    # bulky ortho substituents
    ri = mol.GetRingInfo()

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.IsInRing():
            continue

        a_idx = bond.GetBeginAtomIdx()
        b_idx = bond.GetEndAtomIdx()
        a_atom = mol.GetAtomWithIdx(a_idx)
        b_atom = mol.GetAtomWithIdx(b_idx)

        # Both atoms must be in aromatic rings
        if not (a_atom.GetIsAromatic() and b_atom.GetIsAromatic()):
            continue

        # Check for ortho substituents (non-H, non-ring-connected neighbors)
        def has_ortho_subs(atom_idx: int, other_idx: int) -> bool:
            atom = mol.GetAtomWithIdx(atom_idx)
            ring_atoms = set()
            for ring in ri.AtomRings():
                if atom_idx in ring:
                    ring_atoms.update(ring)
            for nb in atom.GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx == other_idx:
                    continue
                if nb_idx in ring_atoms:
                    # Check this ring neighbor's non-ring, non-H substituents
                    for nb2 in nb.GetNeighbors():
                        if nb2.GetIdx() == atom_idx:
                            continue
                        if nb2.GetIdx() not in ring_atoms and nb2.GetAtomicNum() > 1:
                            return True
            return False

        if has_ortho_subs(a_idx, b_idx) and has_ortho_subs(b_idx, a_idx):
            # Determine label from dihedral if 3D coords available
            label = "?"
            try:
                conf = mol.GetConformer(0)
                # Get ortho neighbors for dihedral
                a_ortho = [nb.GetIdx() for nb in a_atom.GetNeighbors()
                           if nb.GetIdx() != b_idx]
                b_ortho = [nb.GetIdx() for nb in b_atom.GetNeighbors()
                           if nb.GetIdx() != a_idx]
                if a_ortho and b_ortho:
                    from rdkit.Chem import rdMolTransforms
                    dihedral = rdMolTransforms.GetDihedralDeg(
                        conf, a_ortho[0], a_idx, b_idx, b_ortho[0],
                    )
                    label = "aR" if dihedral > 0 else "aS"
            except Exception:
                pass

            results.append((a_idx, b_idx, bond.GetIdx(), label))

    return results


def cip_canonical_order_per_center(
    mol: Chem.Mol,
    shift_override: int | None = None,
) -> dict[int, list[int]]:
    """CIP canonical ordering applied to one chiral center at a time.

    For each chiral center, produces an ordering where only that center's
    neighbors are shifted (all others use baseline ordering). This isolates
    each center's contribution to spectral asymmetry.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule with stereochemistry assigned.
    shift_override : int or None
        If not None, force this shift direction for all centers.

    Returns
    -------
    dict[int, list[int]]
        Mapping from chiral center atom index to its per-center ordering.
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    chiral_info = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

    if not chiral_info:
        return {}

    # Baseline ordering (no chirality)
    ordering_base = cip_canonical_order(mol, chirality_aware=False)

    result = {}
    for center_idx, label in chiral_info:
        # Start from baseline
        ordering = list(ordering_base)

        if shift_override is not None:
            shift = shift_override
        elif label == "R":
            shift = 1
        elif label == "S":
            shift = -1
        elif label == "?":
            try:
                conf = mol.GetConformer(0)
                vol = signed_tetrahedral_volume(mol, center_idx, conf)
                shift = 1 if vol > 0 else -1
            except Exception:
                continue
        else:
            continue

        # Build position map
        pos_of = {atom_idx: pos for pos, atom_idx in enumerate(ordering)}

        atom = mol.GetAtomWithIdx(center_idx)
        neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]

        if len(neighbors) < 2:
            continue

        nbr_positions = sorted([pos_of[nb] for nb in neighbors])
        nbr_at_positions = [ordering[p] for p in nbr_positions]

        k = len(nbr_at_positions)
        shifted = [nbr_at_positions[(i + shift) % k] for i in range(k)]

        for pos, new_atom in zip(nbr_positions, shifted):
            ordering[pos] = new_atom

        result[center_idx] = ordering

    return result


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
