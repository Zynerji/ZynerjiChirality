"""CUDA conformer pipeline: SMILES → 3D coordinates via GPU distance geometry.

Drop-in replacement for RDKit's smiles_to_mol3d() for molecules that hang ETKDG.
Uses CUDADistanceGeometry for the 3D embedding step, with RDKit for parsing
and stereochemistry assignment.

Zero-copy architecture: unified memory eliminates explicit host↔device transfers.
Tensor Cores: eigendecomposition via cuSOLVER uses TF32/FP64 Tensor Cores on Blackwell.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from zynerji_chirality.cuda.bounds import (
    MolecularGraph,
    extract_graph,
    compute_bounds,
    try_rdkit_bounds,
)

logger = logging.getLogger(__name__)

# Try to import CUDA distance geometry
try:
    from zynerji_chirality.cuda.distance_geometry import CUDADistanceGeometry
    _HAS_CUDA_DG = True
except ImportError:
    _HAS_CUDA_DG = False


class CUDAConformerPipeline:
    """GPU-accelerated 3D conformer generation pipeline.

    Pipeline stages:
    1. Parse SMILES → RDKit Mol (CPU)
    2. Extract molecular graph (CPU)
    3. Compute distance bounds (CPU: RDKit or our tables)
    4. GPU: Triangle smooth → sample → MDS embed → refine
    5. Set 3D coordinates on RDKit Mol (CPU)
    6. Assign stereochemistry (CPU)
    """

    def __init__(
        self,
        device: int = 0,
        n_restarts: int = 4,
        max_refine_iter: int = 1000,
        try_rdkit_first: bool = True,
        rdkit_timeout: int = 30,
    ):
        """Initialize CUDA conformer pipeline.

        Parameters
        ----------
        device : int
            CUDA device index.
        n_restarts : int
            Number of random restarts for distance geometry.
        max_refine_iter : int
            Maximum refinement iterations per restart.
        try_rdkit_first : bool
            If True, try RDKit's bounds matrix first (faster, more accurate).
        rdkit_timeout : int
            Timeout in seconds for RDKit bounds computation.
        """
        self.n_restarts = n_restarts
        self.max_refine_iter = max_refine_iter
        self.try_rdkit_first = try_rdkit_first
        self.rdkit_timeout = rdkit_timeout

        if _HAS_CUDA_DG:
            self.dg_solver = CUDADistanceGeometry(device=device)
            logger.info("CUDA conformer pipeline initialized (GPU DG)")
        else:
            self.dg_solver = None
            logger.warning("CUDA DG not available, will use CPU fallback")

    def embed_molecule(self, smiles: str) -> Chem.Mol:
        """Generate 3D conformer for a SMILES string using GPU distance geometry.

        Parameters
        ----------
        smiles : str
            SMILES string (may include stereo annotations).

        Returns
        -------
        Chem.Mol
            RDKit Mol with 3D conformer and stereochemistry assigned.
        """
        t0 = time.time()

        # Stage 1: Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        n_atoms = mol.GetNumAtoms()

        # Stage 2: Get distance bounds
        lower, upper, chiral_info = self._get_bounds(mol)

        # Stage 3: GPU distance geometry
        if self.dg_solver is not None:
            coords = self.dg_solver.solve(
                lower, upper,
                chiral_info=chiral_info,
                n_restarts=self.n_restarts,
                max_refine_iter=self.max_refine_iter,
            )
        else:
            # CPU fallback: simple random embedding + refinement
            coords = self._cpu_fallback(lower, upper, chiral_info, n_atoms)

        # Stage 4: Set coordinates on RDKit Mol
        conf = Chem.Conformer(n_atoms)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, (float(coords[i][0]),
                                     float(coords[i][1]),
                                     float(coords[i][2])))
        mol.AddConformer(conf, assignId=True)

        # Stage 5: Assign stereochemistry from 3D
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        elapsed = time.time() - t0
        logger.info(
            "CUDA embed: %d atoms, %.3fs, smiles_len=%d",
            n_atoms, elapsed, len(smiles),
        )

        return mol

    def embed_batch(self, smiles_list: list[str]) -> list[Chem.Mol | None]:
        """Embed multiple molecules (sequential, GPU handles parallelism internally)."""
        results = []
        for smi in smiles_list:
            try:
                mol = self.embed_molecule(smi)
                results.append(mol)
            except Exception as e:
                logger.warning("Failed to embed: %s (%s)", smi[:50], e)
                results.append(None)
        return results

    def _get_bounds(
        self, mol: Chem.Mol
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """Get distance bounds from RDKit (preferred) or our own computation."""
        chiral_info = []

        # Extract chiral center info for refinement constraints
        graph = extract_graph(mol)
        chiral_info = graph.chiral_centers

        # Try RDKit bounds first (fast, chemically accurate)
        if self.try_rdkit_first:
            result = try_rdkit_bounds(mol)
            if result is not None:
                lower, upper = result
                return lower, upper, chiral_info

        # Fall back to our own bounds computation
        lower, upper = compute_bounds(graph)
        return lower, upper, chiral_info

    @staticmethod
    def _cpu_fallback(
        lower: np.ndarray,
        upper: np.ndarray,
        chiral_info: list,
        n_atoms: int,
    ) -> np.ndarray:
        """CPU fallback when CUDA is not available.

        Simple random embedding + iterative refinement.
        """
        logger.warning("Using CPU fallback (no CUDA)")
        rng = np.random.RandomState(42)

        # Random initial coordinates
        coords = rng.randn(n_atoms, 3) * 2.0

        # Simple refinement: move atoms to satisfy bounds
        lr = 0.01
        for iteration in range(500):
            grad = np.zeros_like(coords)
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    diff = coords[i] - coords[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-10:
                        dist = 1e-10
                    if dist < lower[i][j]:
                        force = (lower[i][j] - dist) / dist
                        grad[i] += force * diff
                        grad[j] -= force * diff
                    elif dist > upper[i][j]:
                        force = (dist - upper[i][j]) / dist
                        grad[i] -= force * diff
                        grad[j] += force * diff

            coords -= lr * grad
            if np.linalg.norm(grad) < 1e-6:
                break

        return coords
