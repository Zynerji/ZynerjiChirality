"""Spectral chirality fingerprint for ML and similarity search.

Fixed-length chirality-aware spectral fingerprint that encodes both
structural topology and handedness. Useful for:
- ML classification of R vs S
- Similarity search across chiral molecule databases
- Clustering molecules by chirality profile
"""

from __future__ import annotations

import numpy as np

from zynerji_chirality.core.dual_helix import HelixParams, compute_spectral_coords
from zynerji_chirality.core.mol_graph import (
    mol_to_adjacency,
    mol_to_chiral_adjacency,
    smiles_to_mol3d,
)
from zynerji_chirality.core.chiral_ordering import cip_canonical_order, reorder_adjacency

from rdkit import Chem


def chirality_fingerprint(
    smiles_or_mol: str | Chem.Mol,
    params: HelixParams | None = None,
    nbits: int = 128,
    chiral_weight: float = 0.5,
) -> np.ndarray:
    """Compute fixed-length chirality-aware spectral fingerprint.

    Uses eigenvalues from both standard and chirality-modulated adjacency
    matrices to construct a fingerprint that encodes structural topology
    and chiral handedness.

    Parameters
    ----------
    smiles_or_mol : str or Chem.Mol
        SMILES string or RDKit Mol object.
    params : HelixParams, optional
        Helix parameters. Defaults to standard params.
    nbits : int
        Length of output fingerprint vector.
    chiral_weight : float
        Chirality modulation weight for adjacency.

    Returns
    -------
    np.ndarray
        Fixed-length fingerprint vector of shape (nbits,).
    """
    if params is None:
        params = HelixParams()

    # Parse and embed
    if isinstance(smiles_or_mol, str):
        mol = smiles_to_mol3d(smiles_or_mol)
    else:
        mol = smiles_or_mol

    # CIP ordering
    ordering = cip_canonical_order(mol)

    # Build both standard and chiral adjacencies
    adj_std = mol_to_adjacency(mol, weight_mode="bond_order")
    adj_chiral = mol_to_chiral_adjacency(mol, chiral_weight=chiral_weight)

    adj_std_ordered = reorder_adjacency(adj_std, ordering)
    adj_chiral_ordered = reorder_adjacency(adj_chiral, ordering)

    # Spectral decomposition for both
    spectral_std = compute_spectral_coords(adj_std_ordered, params)
    spectral_chiral = compute_spectral_coords(adj_chiral_ordered, params)

    # Build raw feature vector from eigenvalues only (deterministic)
    k = params.k

    def pad_evals(evals, length):
        padded = np.zeros(length)
        n = min(len(evals), length)
        padded[:n] = evals[:n]
        return padded

    # Standard eigenvalues (structural topology)
    std_cos = pad_evals(spectral_std.eigenvalues_cos, k)
    std_sin = pad_evals(spectral_std.eigenvalues_sin, k)

    # Chiral eigenvalues (topology + handedness)
    chi_cos = pad_evals(spectral_chiral.eigenvalues_cos, k)
    chi_sin = pad_evals(spectral_chiral.eigenvalues_sin, k)

    # Differential features (chirality signal)
    diff_cos = chi_cos - std_cos
    diff_sin = chi_sin - std_sin
    asym_std = std_cos - std_sin
    asym_chi = chi_cos - chi_sin

    # Raw features: 8k total
    raw = np.concatenate([
        std_cos, std_sin,           # structural topology (2k)
        chi_cos, chi_sin,           # chiral topology (2k)
        diff_cos, diff_sin,         # differential (2k)
        asym_std, asym_chi,         # asymmetry (2k)
    ])

    # Hash to fixed length using random projection (deterministic seed)
    rng = np.random.RandomState(42)
    proj = rng.randn(len(raw), nbits).astype(np.float64)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    fp = raw @ proj

    return fp


def fingerprint_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Tanimoto-like similarity between chirality fingerprints.

    Uses cosine similarity mapped to [0, 1] range.

    Parameters
    ----------
    fp1 : np.ndarray
        First fingerprint.
    fp2 : np.ndarray
        Second fingerprint.

    Returns
    -------
    float
        Similarity score in [0, 1]. 1 = identical, 0 = orthogonal.
    """
    norm1 = np.linalg.norm(fp1)
    norm2 = np.linalg.norm(fp2)

    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0

    cos_sim = float(np.dot(fp1, fp2) / (norm1 * norm2))
    # Map from [-1, 1] to [0, 1]
    return (cos_sim + 1.0) / 2.0
