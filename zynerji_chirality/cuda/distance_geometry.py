"""GPU-accelerated distance geometry solver.

Core algorithm:
1. Triangle inequality smoothing (Floyd-Warshall on GPU)
2. Random distance sampling within bounds
3. Classical MDS embedding (cuSOLVER → Tensor Cores)
4. Coordinate refinement with distance + chirality constraints

All operations use CuPy with unified memory (zero-copy between host/device).
Eigendecomposition via cuSOLVER automatically leverages Tensor Cores on Blackwell.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    from cupy import RawKernel
    from zynerji_chirality.cuda.kernels import (
        TRIANGLE_SMOOTH_KERNEL,
        REFINEMENT_GRADIENT_KERNEL,
        CHIRALITY_VOLUME_KERNEL,
    )

    # Compile CUDA kernels (JIT, cached after first call)
    _smooth_kernel = RawKernel(TRIANGLE_SMOOTH_KERNEL, "triangle_smooth_step")
    _gradient_kernel = RawKernel(REFINEMENT_GRADIENT_KERNEL, "compute_refinement_gradient")
    _chirality_kernel = RawKernel(CHIRALITY_VOLUME_KERNEL, "chirality_volume_gradient")
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False


class CUDADistanceGeometry:
    """GPU-accelerated distance geometry solver.

    Solves the distance geometry problem: given lower/upper bounds on all
    pairwise distances, find 3D coordinates that satisfy the bounds.

    Uses multiple random restarts and returns the best solution (lowest
    bound violation energy).
    """

    def __init__(self, device: int = 0):
        if not _HAS_CUDA:
            raise RuntimeError("CuPy not available. Install cupy-cuda12x.")
        self.device = device
        cp.cuda.Device(device).use()
        # Enable unified memory for zero-copy host/device access
        self._mem_pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        cp.cuda.set_allocator(self._mem_pool.malloc)
        logger.info("CUDA DG solver initialized on device %d (unified memory)", device)

    def smooth_bounds(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Triangle inequality smoothing on GPU (Floyd-Warshall).

        O(N^3) algorithm parallelized across N^2 (i,j) pairs per step.
        Each step k processes all pairs simultaneously on GPU.
        """
        n = lower.shape[0]
        t0 = time.time()

        # Transfer to GPU (unified memory — zero-copy)
        d_lower = cp.asarray(lower, dtype=cp.float64)
        d_upper = cp.asarray(upper, dtype=cp.float64)

        threads = 256
        blocks = (n * n + threads - 1) // threads

        for k in range(n):
            _smooth_kernel(
                (blocks,), (threads,),
                (d_lower, d_upper, np.int32(n), np.int32(k)),
            )

        # Ensure lower <= upper (fix any inconsistencies)
        cp.minimum(d_lower, d_upper, out=d_lower)
        cp.maximum(d_lower, 0.0, out=d_lower)

        elapsed = time.time() - t0
        logger.debug("Triangle smoothing: %d atoms, %.3fs", n, elapsed)

        return cp.asnumpy(d_lower), cp.asnumpy(d_upper)

    def sample_distances(
        self, lower: np.ndarray, upper: np.ndarray, seed: int = 42
    ) -> np.ndarray:
        """Sample random distances within bounds on GPU."""
        cp.random.seed(seed)
        n = lower.shape[0]

        d_lower = cp.asarray(lower, dtype=cp.float64)
        d_upper = cp.asarray(upper, dtype=cp.float64)

        # Uniform sampling within bounds
        rand = cp.random.random((n, n), dtype=cp.float64)
        D = d_lower + rand * (d_upper - d_lower)

        # Make symmetric
        D = (D + D.T) / 2.0
        cp.fill_diagonal(D, 0.0)

        return cp.asnumpy(D)

    def embed_mds(self, D: np.ndarray, dims: int = 3) -> np.ndarray:
        """Classical MDS embedding on GPU.

        Uses cuSOLVER eigendecomposition which automatically leverages
        Tensor Cores on Blackwell/Ada architectures for FP64 matrix ops.
        """
        n = D.shape[0]
        t0 = time.time()

        d_D = cp.asarray(D, dtype=cp.float64)

        # Squared distance matrix
        D2 = d_D ** 2

        # Double centering: B = -1/2 * H * D^2 * H where H = I - 1/n * J
        H = cp.eye(n, dtype=cp.float64) - cp.ones((n, n), dtype=cp.float64) / n
        B = -0.5 * H @ D2 @ H  # cuBLAS matmul → Tensor Cores

        # Symmetrize (numerical precision)
        B = (B + B.T) / 2.0

        # Eigendecomposition (cuSOLVER → Tensor Cores on Blackwell)
        eigenvalues, eigenvectors = cp.linalg.eigh(B)

        # Take top 'dims' eigenvalues (largest)
        idx = cp.argsort(eigenvalues)[::-1][:dims]
        vals = eigenvalues[idx]
        vecs = eigenvectors[:, idx]

        # Clamp negative eigenvalues
        vals = cp.maximum(vals, 1e-10)

        # Coordinates = V * sqrt(Lambda)
        coords = vecs * cp.sqrt(vals)[None, :]

        elapsed = time.time() - t0
        logger.debug("MDS embedding: %d atoms → %dD, %.3fs", n, dims, elapsed)

        return cp.asnumpy(coords)

    def refine(
        self,
        coords: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        chiral_info: list[tuple[int, list[int], str]] | None = None,
        max_iter: int = 1000,
        lr: float = 0.005,
        tol: float = 1e-6,
        chirality_weight: float = 10.0,
    ) -> np.ndarray:
        """Refine coordinates to satisfy distance bounds + chirality constraints.

        Uses gradient descent on GPU with two types of constraints:
        1. Distance violations (above/below bounds)
        2. Chirality volume (signed tetrahedral volume must match R/S)
        """
        n = coords.shape[0]
        t0 = time.time()

        d_coords = cp.asarray(coords, dtype=cp.float64)
        d_lower = cp.asarray(lower, dtype=cp.float64)
        d_upper = cp.asarray(upper, dtype=cp.float64)
        d_grad = cp.zeros_like(d_coords)
        d_violation = cp.zeros(1, dtype=cp.float64)

        threads = min(256, n)
        blocks = (n + threads - 1) // threads

        # Prepare chirality constraints
        has_chirality = chiral_info and len(chiral_info) > 0
        if has_chirality:
            centers_array = []
            signs_array = []
            for center_idx, neighbors, label in chiral_info:
                if len(neighbors) >= 4:
                    centers_array.append([center_idx] + neighbors[:4])
                    signs_array.append(1.0 if label == "R" else -1.0)

            if centers_array:
                d_centers = cp.asarray(np.array(centers_array, dtype=np.int32))
                d_signs = cp.asarray(np.array(signs_array, dtype=np.float64))
                n_chiral = len(centers_array)
                chiral_threads = min(256, n_chiral)
                chiral_blocks = (n_chiral + chiral_threads - 1) // chiral_threads
            else:
                has_chirality = False

        prev_violation = float("inf")

        for iteration in range(max_iter):
            d_grad.fill(0.0)
            d_violation.fill(0.0)

            # Distance violation gradient
            _gradient_kernel(
                (blocks,), (threads,),
                (d_coords, d_lower, d_upper, d_grad, d_violation, np.int32(n)),
            )

            # Chirality volume gradient
            if has_chirality:
                _chirality_kernel(
                    (chiral_blocks,), (chiral_threads,),
                    (d_coords, d_centers, d_signs, d_grad,
                     np.int32(n_chiral), np.float64(chirality_weight)),
                )

            # Gradient descent step
            d_coords -= lr * d_grad

            # Check convergence
            violation = float(d_violation.item())
            if iteration % 100 == 0:
                logger.debug(
                    "Refine iter %d: violation=%.6f, grad_norm=%.6f",
                    iteration, violation, float(cp.linalg.norm(d_grad)),
                )

            if violation < tol:
                break

            # Adaptive learning rate
            if violation > prev_violation:
                lr *= 0.5
            elif iteration > 0 and iteration % 200 == 0:
                lr *= 0.95

            prev_violation = violation

        elapsed = time.time() - t0
        logger.debug(
            "Refinement: %d atoms, %d iters, final violation=%.6f, %.3fs",
            n, iteration + 1, violation, elapsed,
        )

        return cp.asnumpy(d_coords)

    def solve(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        chiral_info: list[tuple[int, list[int], str]] | None = None,
        n_restarts: int = 4,
        base_seed: int = 42,
        max_refine_iter: int = 1000,
    ) -> np.ndarray:
        """Full distance geometry solve with multiple restarts.

        Returns the 3D coordinates with lowest bound violation.
        """
        n = lower.shape[0]
        t0 = time.time()

        # Smooth bounds (done once, shared across restarts)
        lower_smooth, upper_smooth = self.smooth_bounds(lower, upper)

        best_coords = None
        best_violation = float("inf")

        for restart in range(n_restarts):
            seed = base_seed + restart * 1337

            # Sample random distances
            D = self.sample_distances(lower_smooth, upper_smooth, seed=seed)

            # MDS embedding (Tensor Cores)
            coords = self.embed_mds(D, dims=3)

            # Refine with constraints
            coords = self.refine(
                coords, lower_smooth, upper_smooth,
                chiral_info=chiral_info,
                max_iter=max_refine_iter,
            )

            # Compute final violation
            violation = self._compute_violation(coords, lower_smooth, upper_smooth)

            logger.debug(
                "Restart %d/%d: violation=%.6f",
                restart + 1, n_restarts, violation,
            )

            if violation < best_violation:
                best_violation = violation
                best_coords = coords.copy()

            # Early exit if near-zero violation
            if violation < 1e-4:
                break

        elapsed = time.time() - t0
        logger.info(
            "DG solve: %d atoms, %d restarts, best_violation=%.6f, %.3fs",
            n, n_restarts, best_violation, elapsed,
        )

        return best_coords

    @staticmethod
    def _compute_violation(
        coords: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> float:
        """Compute total distance bound violation energy."""
        n = coords.shape[0]
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                if d < lower[i][j]:
                    total += (lower[i][j] - d) ** 2
                elif d > upper[i][j]:
                    total += (d - upper[i][j]) ** 2
        return total
