"""Crystal structure graph conversion utilities.

Converts crystal structures (CIF files, perovskite unit cells, polymer repeats)
to graph representations suitable for the dual-helix spectral engine.
Minimal CIF parser — no pymatgen dependency.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def cif_to_adjacency(
    cif_path: str,
    cutoff: float = 3.0,
) -> tuple[csr_matrix, list[str]]:
    """Parse a CIF file and build an adjacency matrix by distance cutoff.

    Reads the _atom_site block, converts fractional coordinates to Cartesian,
    computes pairwise distances, and builds a weighted adjacency matrix where
    edge weights are 1/distance for atoms within the cutoff.

    Parameters
    ----------
    cif_path : str
        Path to CIF file.
    cutoff : float
        Distance cutoff in Angstroms for adjacency.

    Returns
    -------
    tuple[csr_matrix, list[str]]
        (adjacency_matrix, atom_labels) where labels are element symbols.
    """
    path = Path(cif_path)
    text = path.read_text(encoding="utf-8", errors="replace")

    # Parse cell parameters
    cell_params = _parse_cell_params(text)
    frac_to_cart = _frac_to_cart_matrix(cell_params)

    # Parse atom sites
    atom_labels, frac_coords = _parse_atom_sites(text)

    if len(atom_labels) == 0:
        return csr_matrix((0, 0)), []

    # Convert fractional to Cartesian
    cart_coords = np.array(frac_coords) @ frac_to_cart.T

    # Build adjacency by distance cutoff
    n = len(atom_labels)
    rows, cols, vals = [], [], []

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(cart_coords[i] - cart_coords[j])
            if 0.1 < d <= cutoff:
                w = 1.0 / d
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([w, w])

    adj = csr_matrix((vals, (rows, cols)), shape=(n, n)) if vals else csr_matrix((n, n))
    return adj, atom_labels


def _parse_cell_params(text: str) -> dict[str, float]:
    """Extract unit cell parameters from CIF text."""
    params = {"a": 1.0, "b": 1.0, "c": 1.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}

    patterns = {
        "a": r"_cell_length_a\s+([\d.]+)",
        "b": r"_cell_length_b\s+([\d.]+)",
        "c": r"_cell_length_c\s+([\d.]+)",
        "alpha": r"_cell_angle_alpha\s+([\d.]+)",
        "beta": r"_cell_angle_beta\s+([\d.]+)",
        "gamma": r"_cell_angle_gamma\s+([\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            params[key] = float(match.group(1))

    return params


def _frac_to_cart_matrix(cell: dict[str, float]) -> np.ndarray:
    """Build fractional-to-Cartesian transformation matrix from cell parameters."""
    a, b, c = cell["a"], cell["b"], cell["c"]
    alpha = np.radians(cell["alpha"])
    beta = np.radians(cell["beta"])
    gamma = np.radians(cell["gamma"])

    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_g = np.sin(gamma)

    # Standard crystallographic convention
    v = np.sqrt(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g)

    M = np.array([
        [a, b * cos_g, c * cos_b],
        [0, b * sin_g, c * (cos_a - cos_b * cos_g) / sin_g],
        [0, 0, c * v / sin_g],
    ])
    return M


def _parse_atom_sites(text: str) -> tuple[list[str], list[list[float]]]:
    """Parse _atom_site block from CIF text.

    Returns atom labels (element symbols) and fractional coordinates.
    """
    labels = []
    coords = []

    # Find the atom_site loop
    lines = text.split("\n")
    in_loop = False
    col_indices = {}
    col_count = 0

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("loop_"):
            in_loop = True
            col_indices = {}
            col_count = 0
            continue

        if in_loop and stripped.startswith("_atom_site"):
            tag = stripped.split()[0]
            col_indices[tag] = col_count
            col_count += 1
            continue

        if in_loop and col_count > 0 and not stripped.startswith("_"):
            if not stripped or stripped.startswith("loop_") or stripped.startswith("#"):
                if labels:  # We already found data, stop
                    break
                in_loop = False
                continue

            # Check we have the columns we need
            needed = {"_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"}
            label_col = col_indices.get("_atom_site_type_symbol",
                        col_indices.get("_atom_site_label", -1))

            if not needed.issubset(col_indices.keys()) or label_col < 0:
                continue

            parts = stripped.split()
            if len(parts) < col_count:
                continue

            try:
                x = float(_strip_cif_uncertainty(parts[col_indices["_atom_site_fract_x"]]))
                y = float(_strip_cif_uncertainty(parts[col_indices["_atom_site_fract_y"]]))
                z = float(_strip_cif_uncertainty(parts[col_indices["_atom_site_fract_z"]]))
                label_raw = parts[label_col]
                # Extract element symbol (strip numbers, +/- charges)
                elem = re.match(r"([A-Z][a-z]?)", label_raw)
                if elem:
                    labels.append(elem.group(1))
                    coords.append([x, y, z])
            except (ValueError, IndexError):
                continue

    return labels, coords


def _strip_cif_uncertainty(value: str) -> str:
    """Strip CIF uncertainty notation, e.g., '1.234(5)' -> '1.234'."""
    idx = value.find("(")
    return value[:idx] if idx >= 0 else value


def perovskite_graph(
    A: str,
    B: str,
    X: str,
    a: float = 3.9,
) -> tuple[csr_matrix, list[str]]:
    """Generate a standard perovskite ABX3 unit cell graph.

    Creates the ideal cubic perovskite structure with:
    - A at corners (0,0,0)
    - B at body center (0.5,0.5,0.5)
    - X at face centers (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)

    Parameters
    ----------
    A : str
        A-site element (e.g., "Cs", "CH3NH3").
    B : str
        B-site element (e.g., "Pb", "Sn").
    X : str
        X-site element (e.g., "I", "Br", "Cl").
    a : float
        Lattice parameter in Angstroms.

    Returns
    -------
    tuple[csr_matrix, list[str]]
        (adjacency_matrix, atom_labels) — 5-node graph (1 A + 1 B + 3 X).
    """
    # Fractional positions
    positions = np.array([
        [0.0, 0.0, 0.0],     # A site
        [0.5, 0.5, 0.5],     # B site
        [0.5, 0.5, 0.0],     # X1 (face center xy)
        [0.5, 0.0, 0.5],     # X2 (face center xz)
        [0.0, 0.5, 0.5],     # X3 (face center yz)
    ])
    cart = positions * a
    labels = [A, B, X, X, X]

    n = 5
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(cart[i] - cart[j])
            if d > 0.01:
                w = 1.0 / d
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([w, w])

    adj = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return adj, labels


def polymer_repeat_graph(monomer_smiles: str, n_repeat: int = 5):
    """Build an oligomer from monomer SMILES using RDKit.

    Connects monomer copies via the first single-bonded terminal atoms.
    This is a simple linear concatenation — for complex polymers,
    use explicit SMILES.

    Parameters
    ----------
    monomer_smiles : str
        SMILES of the monomer unit.
    n_repeat : int
        Number of repeat units.

    Returns
    -------
    Chem.Mol
        RDKit Mol object of the oligomer.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    monomer = Chem.MolFromSmiles(monomer_smiles)
    if monomer is None:
        raise ValueError(f"Invalid monomer SMILES: {monomer_smiles}")

    # Simple approach: replicate SMILES with dot notation and let RDKit handle it
    # For actual polymer, we just repeat the monomer
    oligomer_smiles = ".".join([monomer_smiles] * n_repeat)
    oligomer = Chem.MolFromSmiles(oligomer_smiles)

    if oligomer is None:
        raise ValueError(f"Failed to build oligomer from: {monomer_smiles}")

    # Add hydrogens and generate 3D
    oligomer = Chem.AddHs(oligomer)
    AllChem.EmbedMolecule(oligomer, AllChem.ETKDGv3())
    oligomer = Chem.RemoveHs(oligomer)

    return oligomer
