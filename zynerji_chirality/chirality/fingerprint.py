"""Spectral chirality fingerprint for ML and similarity search.

Fixed-length chirality-aware spectral fingerprint that encodes both
structural topology and handedness. Useful for:
- ML classification of R vs S
- Similarity search across chiral molecule databases
- Clustering molecules by chirality profile
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from zynerji_chirality.core.dual_helix import HelixParams, compute_spectral_coords
from zynerji_chirality.core.mol_graph import (
    mol_to_adjacency,
    mol_to_chiral_adjacency,
    smiles_to_mol3d,
)
from zynerji_chirality.core.chiral_ordering import cip_canonical_order, reorder_adjacency

from rdkit import Chem


# Projection matrix cache: keyed by (raw_dim, nbits)
_PROJ_CACHE: dict[tuple[int, int], np.ndarray] = {}

# Fingerprint result cache: keyed by (smiles, nbits) for default-params calls.
# Eliminates redundant fingerprint computation when the same SMILES is fingerprinted
# across multiple target models or in batch training (same molecule, many targets).
# Each entry: 128 floats × 8 bytes = 1KB. 100K entries = ~100MB.
_FP_CACHE: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
_FP_CACHE_MAX = 120_000

# Differential fingerprint cache: keyed by (smiles, nbits, mode)
_DIFF_FP_CACHE: OrderedDict[tuple[str, int, str], np.ndarray] = OrderedDict()
_DIFF_FP_CACHE_MAX = 120_000


def clear_fingerprint_cache() -> None:
    """Clear the chirality fingerprint result cache."""
    _FP_CACHE.clear()
    _DIFF_FP_CACHE.clear()


def _get_projection_matrix(raw_dim: int, nbits: int) -> np.ndarray:
    """Get or create a cached random projection matrix."""
    key = (raw_dim, nbits)
    if key not in _PROJ_CACHE:
        rng = np.random.RandomState(42)
        proj = rng.randn(raw_dim, nbits).astype(np.float64)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        _PROJ_CACHE[key] = proj
    return _PROJ_CACHE[key]


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

    # Cache lookup: only for string SMILES with default params/weight
    # (training always uses these defaults, so this covers ~100% of training calls)
    _cache_key = None
    if isinstance(smiles_or_mol, str) and chiral_weight == 0.5:
        _cache_key = (smiles_or_mol, nbits)
        if _cache_key in _FP_CACHE:
            _FP_CACHE.move_to_end(_cache_key)
            return _FP_CACHE[_cache_key].copy()

    # Parse molecule
    if isinstance(smiles_or_mol, str):
        # Fast path: if stereo is already annotated in SMILES (@ or @@),
        # CIP labels can be assigned from 2D topology — no 3D embedding needed.
        # This is true for all ChEMBL/training data with explicit stereo marks.
        # Only fall back to smiles_to_mol3d() for molecules with unannotated
        # stereocenters that require 3D geometry to determine R/S.
        mol_2d = Chem.MolFromSmiles(smiles_or_mol)
        if mol_2d is not None:
            Chem.AssignStereochemistry(mol_2d, cleanIt=True, force=True)
            assigned = Chem.FindMolChiralCenters(mol_2d, includeUnassigned=False)
            unassigned = Chem.FindMolChiralCenters(mol_2d, includeUnassigned=True)
            # Use 2D if: achiral, or all stereocenters are already assigned
            if len(unassigned) == 0 or len(assigned) == len(unassigned):
                mol = mol_2d
            else:
                mol = smiles_to_mol3d(smiles_or_mol)
        else:
            raise ValueError(f"Cannot parse SMILES: {smiles_or_mol}")
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

    # Hash to fixed length using cached random projection (deterministic seed)
    proj = _get_projection_matrix(len(raw), nbits)
    fp = raw @ proj

    # Store in cache
    if _cache_key is not None:
        if len(_FP_CACHE) >= _FP_CACHE_MAX:
            _FP_CACHE.popitem(last=False)
        _FP_CACHE[_cache_key] = fp.copy()

    return fp


def _compute_one_fp(args: tuple) -> np.ndarray | None:
    """Module-level worker for batch_fingerprint (must be picklable)."""
    smiles, nbits = args
    try:
        return chirality_fingerprint(smiles, nbits=nbits)
    except Exception:
        return None


def batch_fingerprint(
    smiles_list: list[str],
    params: HelixParams | None = None,
    nbits: int = 128,
    n_workers: int = 1,
) -> list[np.ndarray]:
    """Compute fingerprints for multiple SMILES in parallel.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.
    params : HelixParams, optional
        Helix parameters.
    nbits : int
        Fingerprint length.
    n_workers : int
        Number of parallel workers. Use 1 for serial.

    Returns
    -------
    list[np.ndarray]
        List of fingerprint vectors (None entries for failed molecules).
    """
    if n_workers <= 1:
        return [_compute_one_fp((s, nbits)) for s in smiles_list]

    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_compute_one_fp, [(s, nbits) for s in smiles_list])
    return results


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


def chirality_differential_fingerprint(
    smiles_or_mol: str | Chem.Mol,
    params: HelixParams | None = None,
    nbits: int = 128,
    chiral_weight: float = 0.5,
    mode: str = "pure_diff",
) -> np.ndarray:
    """Compute chirality-only differential fingerprint.

    Uses ONLY the differential sub-vectors (chi - std eigenvalues) which
    flip sign between enantiomers, discarding shared topology features.
    For enantiomers: diff(R) = -diff(S) → projected FP is negated → cosine ≈ -1.0.

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
    mode : str
        "pure_diff" — only diff_cos + diff_sin (16 raw dims → nbits)
        "chi_diff" — diff_cos + diff_sin + asym_chi (24 raw dims → nbits)

    Returns
    -------
    np.ndarray
        Fixed-length fingerprint vector of shape (nbits,).
    """
    if mode not in ("pure_diff", "chi_diff"):
        raise ValueError(f"mode must be 'pure_diff' or 'chi_diff', got {mode!r}")

    if params is None:
        params = HelixParams()

    # Cache lookup
    _cache_key = None
    if isinstance(smiles_or_mol, str) and chiral_weight == 0.5:
        _cache_key = (smiles_or_mol, nbits, mode)
        if _cache_key in _DIFF_FP_CACHE:
            _DIFF_FP_CACHE.move_to_end(_cache_key)
            return _DIFF_FP_CACHE[_cache_key].copy()

    # Parse molecule (same logic as chirality_fingerprint)
    if isinstance(smiles_or_mol, str):
        mol_2d = Chem.MolFromSmiles(smiles_or_mol)
        if mol_2d is not None:
            Chem.AssignStereochemistry(mol_2d, cleanIt=True, force=True)
            assigned = Chem.FindMolChiralCenters(mol_2d, includeUnassigned=False)
            unassigned = Chem.FindMolChiralCenters(mol_2d, includeUnassigned=True)
            if len(unassigned) == 0 or len(assigned) == len(unassigned):
                mol = mol_2d
            else:
                mol = smiles_to_mol3d(smiles_or_mol)
        else:
            raise ValueError(f"Cannot parse SMILES: {smiles_or_mol}")
    else:
        mol = smiles_or_mol

    # CIP ordering
    ordering = cip_canonical_order(mol)

    # Build both standard and chiral adjacencies
    adj_std = mol_to_adjacency(mol, weight_mode="bond_order")
    adj_chiral = mol_to_chiral_adjacency(mol, chiral_weight=chiral_weight)

    adj_std_ordered = reorder_adjacency(adj_std, ordering)
    adj_chiral_ordered = reorder_adjacency(adj_chiral, ordering)

    # Spectral decomposition
    spectral_std = compute_spectral_coords(adj_std_ordered, params)
    spectral_chiral = compute_spectral_coords(adj_chiral_ordered, params)

    k = params.k

    def pad_evals(evals, length):
        padded = np.zeros(length)
        n = min(len(evals), length)
        padded[:n] = evals[:n]
        return padded

    std_cos = pad_evals(spectral_std.eigenvalues_cos, k)
    std_sin = pad_evals(spectral_std.eigenvalues_sin, k)
    chi_cos = pad_evals(spectral_chiral.eigenvalues_cos, k)
    chi_sin = pad_evals(spectral_chiral.eigenvalues_sin, k)

    # Differential features only — these flip sign for enantiomers
    diff_cos = chi_cos - std_cos
    diff_sin = chi_sin - std_sin

    if mode == "pure_diff":
        raw = np.concatenate([diff_cos, diff_sin])  # 2k dims
    else:  # chi_diff
        asym_chi = chi_cos - chi_sin
        raw = np.concatenate([diff_cos, diff_sin, asym_chi])  # 3k dims

    # Project to fixed length
    proj = _get_projection_matrix(len(raw), nbits)
    fp = raw @ proj

    # Store in cache
    if _cache_key is not None:
        if len(_DIFF_FP_CACHE) >= _DIFF_FP_CACHE_MAX:
            _DIFF_FP_CACHE.popitem(last=False)
        _DIFF_FP_CACHE[_cache_key] = fp.copy()

    return fp


def _compute_one_diff_fp(args: tuple) -> np.ndarray | None:
    """Module-level worker for batch_differential_fingerprint."""
    smiles, nbits, mode = args
    try:
        return chirality_differential_fingerprint(smiles, nbits=nbits, mode=mode)
    except Exception:
        return None


def batch_differential_fingerprint(
    smiles_list: list[str],
    params: HelixParams | None = None,
    nbits: int = 128,
    mode: str = "pure_diff",
    n_workers: int = 1,
) -> list[np.ndarray | None]:
    """Compute differential fingerprints for multiple SMILES.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.
    params : HelixParams, optional
        Helix parameters.
    nbits : int
        Fingerprint length.
    mode : str
        "pure_diff" or "chi_diff".
    n_workers : int
        Number of parallel workers. Use 1 for serial.

    Returns
    -------
    list[np.ndarray | None]
        List of fingerprint vectors (None entries for failed molecules).
    """
    if n_workers <= 1:
        return [_compute_one_diff_fp((s, nbits, mode)) for s in smiles_list]

    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(
            _compute_one_diff_fp, [(s, nbits, mode) for s in smiles_list]
        )
    return results
