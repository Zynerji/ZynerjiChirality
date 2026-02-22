"""Spectral coordinate alignment and cost matrix construction.

Pure numpy/scipy math extracted from ZynerjiQuantumCompiler.
No Qiskit dependencies — only spectral coordinate operations.
"""

from __future__ import annotations

import numpy as np

from zynerji_chirality.core.dual_helix import SpectralCoords


def _align_coords(
    coords_a: SpectralCoords,
    coords_b: SpectralCoords,
) -> tuple[np.ndarray, np.ndarray]:
    """Align spectral coordinate dimensions between two sets."""
    n_a = coords_a.coords.shape[0]
    n_b = coords_b.coords.shape[0]

    a_dim = coords_a.coords.shape[1] if n_a > 0 else 0
    b_dim = coords_b.coords.shape[1] if n_b > 0 else 0
    max_dim = max(a_dim, b_dim, 1)

    a_coords = np.zeros((n_a, max_dim))
    if n_a > 0 and a_dim > 0:
        a_coords[:, :a_dim] = coords_a.coords

    b_coords = np.zeros((n_b, max_dim))
    if n_b > 0 and b_dim > 0:
        b_coords[:, :b_dim] = coords_b.coords

    return a_coords, b_coords


def build_cost_matrix(
    coords_a: SpectralCoords,
    coords_b: SpectralCoords,
) -> np.ndarray:
    """Build a cost matrix from spectral coordinate distances (L2).

    Returns
    -------
    np.ndarray
        Cost matrix (n_a, n_b) where C[i,j] is the squared
        Euclidean distance between node i and node j in spectral space.
    """
    a_coords, b_coords = _align_coords(coords_a, coords_b)
    n_a, n_b = a_coords.shape[0], b_coords.shape[0]

    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b))

    a_sq = np.sum(a_coords ** 2, axis=1, keepdims=True)
    b_sq = np.sum(b_coords ** 2, axis=1, keepdims=True)
    cross = a_coords @ b_coords.T
    cost = a_sq + b_sq.T - 2.0 * cross
    np.maximum(cost, 0.0, out=cost)

    return cost


def build_angular_cost_matrix(
    coords_a: SpectralCoords,
    coords_b: SpectralCoords,
) -> np.ndarray:
    """Build a cost matrix from angular (cosine) similarity.

    C[i,j] = 1 - cos(angle between coords_a[i] and coords_b[j])
    Range: [0, 2] where 0 = identical direction, 2 = opposite.
    """
    a_coords, b_coords = _align_coords(coords_a, coords_b)
    n_a, n_b = a_coords.shape[0], b_coords.shape[0]

    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b))

    a_norms = np.linalg.norm(a_coords, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b_coords, axis=1, keepdims=True)
    a_norms = np.maximum(a_norms, 1e-12)
    b_norms = np.maximum(b_norms, 1e-12)

    a_unit = a_coords / a_norms
    b_unit = b_coords / b_norms

    cos_sim = a_unit @ b_unit.T
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)

    cost = 1.0 - cos_sim
    return cost


def build_hybrid_cost_matrix(
    coords_a: SpectralCoords,
    coords_b: SpectralCoords,
    angular_weight: float = 0.5,
) -> np.ndarray:
    """Build a hybrid cost matrix combining L2 distance and angular similarity."""
    l2_cost = build_cost_matrix(coords_a, coords_b)
    angular_cost = build_angular_cost_matrix(coords_a, coords_b)

    if l2_cost.size == 0:
        return l2_cost

    l2_max = l2_cost.max()
    ang_max = angular_cost.max()
    if l2_max > 1e-12:
        l2_norm = l2_cost / l2_max
    else:
        l2_norm = l2_cost
    if ang_max > 1e-12:
        ang_norm = angular_cost / ang_max
    else:
        ang_norm = angular_cost

    return (1.0 - angular_weight) * l2_norm + angular_weight * ang_norm


def build_qfactor_cost_matrix(
    coords_a: SpectralCoords,
    coords_b: SpectralCoords,
    q_amplification: float = 1.5,
    alignment_threshold: float = 0.9,
) -> np.ndarray:
    """Build a cost matrix with Q-factor amplification at structural antinodes."""
    a_coords, b_coords = _align_coords(coords_a, coords_b)
    n_a, n_b = a_coords.shape[0], b_coords.shape[0]

    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b))

    a_sq = np.sum(a_coords ** 2, axis=1, keepdims=True)
    b_sq = np.sum(b_coords ** 2, axis=1, keepdims=True)
    cross = a_coords @ b_coords.T
    cost = a_sq + b_sq.T - 2.0 * cross
    np.maximum(cost, 0.0, out=cost)

    a_norms = np.linalg.norm(a_coords, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b_coords, axis=1, keepdims=True)
    a_norms = np.maximum(a_norms, 1e-12)
    b_norms = np.maximum(b_norms, 1e-12)

    cos_sim = (a_coords / a_norms) @ (b_coords / b_norms).T
    np.clip(cos_sim, -1.0, 1.0, out=cos_sim)

    antinode_mask = np.abs(cos_sim) > alignment_threshold
    cost[antinode_mask] /= q_amplification

    return cost
