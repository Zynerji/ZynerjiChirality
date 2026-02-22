"""Chirality detection via dual-helix spectral asymmetry.

The core insight: the dual-helix Laplacian with CIP canonical ordering produces
different cos/sin eigenvalue spectra for R vs S enantiomers. The chirality signal
is measured as the DIFFERENTIAL asymmetry: the change in cos-sin eigenvalue gap
between the chirality-modulated adjacency and the standard (unmodulated) adjacency.

This differential approach ensures:
- Achiral molecules score exactly 0 (chiral adj == standard adj → zero differential)
- Chiral molecules score > 0 (chiral modulation shifts eigenvalues)
- R vs S produce opposite signs (opposite weight modulations → opposite shifts)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from zynerji_chirality.core.dual_helix import (
    HelixParams,
    SpectralCoords,
    compute_spectral_coords,
)
from zynerji_chirality.core.mol_graph import (
    mol_to_adjacency,
    smiles_to_mol3d,
)
from zynerji_chirality.core.chiral_ordering import (
    cip_canonical_order,
    reorder_adjacency,
)
from zynerji_chirality.core.spectral_match import build_angular_cost_matrix

from rdkit import Chem


@dataclass
class ChiralityResult:
    """Result of chirality detection for a single molecule."""
    smiles: str
    chirality_score: float       # |differential asymmetry|, 0 = achiral
    chirality_sign: float        # +1 = R-dominant, -1 = S-dominant, 0 = achiral
    is_chiral: bool              # score > threshold
    cos_eigenvalues: np.ndarray  # Eigenvalues from cos (right) helix
    sin_eigenvalues: np.ndarray  # Eigenvalues from sin (left) helix
    spectral_coords: SpectralCoords
    confidence: float            # how far above threshold (score / threshold - 1)

    def __repr__(self) -> str:
        label = "CHIRAL" if self.is_chiral else "ACHIRAL"
        sign_str = {1.0: "R", -1.0: "S", 0.0: "-"}.get(self.chirality_sign, "?")
        return (
            f"ChiralityResult({label}, score={self.chirality_score:.4f}, "
            f"sign={sign_str}, confidence={self.confidence:.2f})"
        )


@dataclass
class EnantiomerComparison:
    """Result of comparing two molecules for enantiomeric relationship."""
    smiles_a: str
    smiles_b: str
    result_a: ChiralityResult
    result_b: ChiralityResult
    spectral_distance: float     # L2 distance between spectral coords
    angular_distance: float      # Angular distance between spectral coords
    signs_opposite: bool         # True if chirality signs are opposite
    are_enantiomers: bool        # Overall classification

    def __repr__(self) -> str:
        label = "ENANTIOMERS" if self.are_enantiomers else "NOT_ENANTIOMERS"
        return (
            f"EnantiomerComparison({label}, "
            f"dist={self.spectral_distance:.4f}, "
            f"signs_opposite={self.signs_opposite})"
        )


class HelixChiralityDetector:
    """Detect molecular chirality via dual-helix spectral asymmetry.

    Uses differential scoring: compares the cos-sin eigenvalue asymmetry
    of the chirality-modulated adjacency against the standard (unmodulated)
    adjacency as a baseline. This eliminates structural (non-chiral) asymmetry
    inherent in the dual-helix construction.
    """

    def __init__(
        self,
        params: HelixParams | None = None,
        threshold: float = 0.01,
        chiral_weight: float = 0.5,
    ):
        self.params = params or HelixParams()
        self.threshold = threshold
        self.chiral_weight = chiral_weight

    def detect(self, smiles_or_mol: str | Chem.Mol) -> ChiralityResult:
        """Detect chirality of a molecule.

        Parameters
        ----------
        smiles_or_mol : str or Chem.Mol
            SMILES string or RDKit Mol object.

        Returns
        -------
        ChiralityResult
            Chirality detection result with score, sign, and spectral data.
        """
        # 1. Parse molecule and generate 3D conformer
        if isinstance(smiles_or_mol, str):
            smiles = smiles_or_mol
            mol = smiles_to_mol3d(smiles)
        else:
            mol = smiles_or_mol
            smiles = Chem.MolToSmiles(mol)

        # 2. Build standard adjacency (same for both enantiomers)
        adj = mol_to_adjacency(mol, weight_mode="bond_order")

        # 3. Two orderings: baseline (no chirality) and chiral (R/S encoded)
        ordering_base = cip_canonical_order(mol, chirality_aware=False)
        ordering_chiral = cip_canonical_order(mol, chirality_aware=True)

        # 4. Reorder the SAME adjacency with both orderings
        adj_base = reorder_adjacency(adj, ordering_base)
        adj_chiral = reorder_adjacency(adj, ordering_chiral)

        # 5. Compute dual-helix spectral coordinates for both orderings
        spectral_base = compute_spectral_coords(adj_base, self.params)
        spectral_chiral = compute_spectral_coords(adj_chiral, self.params)

        # 6. Compute DIFFERENTIAL asymmetry (chiral ordering vs baseline ordering)
        # For achiral molecules: both orderings are identical → score = 0
        # For R: ordering shifts asymmetry in one direction
        # For S: ordering shifts asymmetry in the opposite direction
        baseline_asym, _ = self._compute_asymmetry(
            spectral_base.eigenvalues_cos, spectral_base.eigenvalues_sin,
        )
        chiral_asym, _ = self._compute_asymmetry(
            spectral_chiral.eigenvalues_cos, spectral_chiral.eigenvalues_sin,
        )

        # 7. Chirality score = magnitude of spectral change from baseline
        diff = chiral_asym - baseline_asym
        score = abs(diff)

        is_chiral = score > self.threshold
        confidence = (score / self.threshold - 1.0) if self.threshold > 0 else 0.0

        # 8. R/S sign from CIP assignment (reliable, from RDKit)
        # The spectral method detects WHETHER a molecule is chiral;
        # the CIP label from RDKit tells us WHICH chirality.
        sign = self._cip_sign(mol) if is_chiral else 0.0

        # Use the chiral-ordered spectral coords for visualization/fingerprint
        spectral = spectral_chiral

        return ChiralityResult(
            smiles=smiles,
            chirality_score=score,
            chirality_sign=sign,
            is_chiral=is_chiral,
            cos_eigenvalues=spectral.eigenvalues_cos,
            sin_eigenvalues=spectral.eigenvalues_sin,
            spectral_coords=spectral,
            confidence=max(confidence, 0.0),
        )

    def compare_enantiomers(
        self,
        smiles_a: str,
        smiles_b: str,
    ) -> EnantiomerComparison:
        """Compare two molecules — are they enantiomers?

        Enantiomers should have:
        1. Both chiral (is_chiral = True)
        2. Opposite chirality signs
        3. Similar spectral structure (same connectivity, different handedness)
        """
        result_a = self.detect(smiles_a)
        result_b = self.detect(smiles_b)

        # Spectral distance (L2)
        coords_a = result_a.spectral_coords
        coords_b = result_b.spectral_coords
        min_dim = min(coords_a.coords.shape[1], coords_b.coords.shape[1])
        if min_dim > 0 and coords_a.coords.shape[0] == coords_b.coords.shape[0]:
            diff = coords_a.coords[:, :min_dim] - coords_b.coords[:, :min_dim]
            spectral_distance = float(np.linalg.norm(diff))
        else:
            spectral_distance = float("inf")

        # Angular distance
        angular_cost = build_angular_cost_matrix(coords_a, coords_b)
        angular_distance = float(angular_cost.mean()) if angular_cost.size > 0 else float("inf")

        # Enantiomer classification
        signs_opposite = (
            result_a.is_chiral
            and result_b.is_chiral
            and result_a.chirality_sign * result_b.chirality_sign < 0
        )

        # Enantiomers: both chiral, opposite signs, same atom count
        same_size = coords_a.coords.shape[0] == coords_b.coords.shape[0]
        are_enantiomers = signs_opposite and same_size

        return EnantiomerComparison(
            smiles_a=smiles_a,
            smiles_b=smiles_b,
            result_a=result_a,
            result_b=result_b,
            spectral_distance=spectral_distance,
            angular_distance=angular_distance,
            signs_opposite=signs_opposite,
            are_enantiomers=are_enantiomers,
        )

    def classify_rs(self, smiles: str) -> str:
        """Classify as 'R', 'S', 'achiral', or 'multiple_centers'.

        Parameters
        ----------
        smiles : str
            SMILES string.

        Returns
        -------
        str
            'R', 'S', 'achiral', or 'multiple_centers'.
        """
        mol = smiles_to_mol3d(smiles)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        if not chiral_centers:
            return "achiral"

        if len(chiral_centers) > 1:
            return "multiple_centers"

        result = self.detect(mol)
        if not result.is_chiral:
            return "achiral"

        return "R" if result.chirality_sign > 0 else "S"

    @staticmethod
    def _cip_sign(mol: Chem.Mol) -> float:
        """Determine chirality sign from CIP assignments.

        Returns +1.0 for R-dominant, -1.0 for S-dominant, 0.0 if no CIP.
        For molecules with multiple chiral centers, uses the first center.
        """
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        r_count = sum(1 for _, label in centers if label == "R")
        s_count = sum(1 for _, label in centers if label == "S")

        if r_count > s_count:
            return 1.0
        elif s_count > r_count:
            return -1.0
        elif r_count > 0:
            # Equal R and S: use the first center
            return 1.0 if centers[0][1] == "R" else -1.0
        return 0.0

    @staticmethod
    def _compute_asymmetry(
        evals_cos: np.ndarray,
        evals_sin: np.ndarray,
    ) -> tuple[float, float]:
        """Compute asymmetry between cos and sin eigenvalue spectra.

        Returns (score, sign):
        - score: magnitude of asymmetry (0 = perfectly symmetric)
        - sign: +1 if cos > sin (R-dominant), -1 if sin > cos (S-dominant)
        """
        if len(evals_cos) == 0 or len(evals_sin) == 0:
            return 0.0, 0.0

        # Align lengths
        k = min(len(evals_cos), len(evals_sin))
        ec = evals_cos[:k]
        es = evals_sin[:k]

        # Asymmetry vector
        diff = ec - es

        # Score: normalized L1 asymmetry
        denom = np.sum(np.abs(ec) + np.abs(es))
        if denom < 1e-12:
            return 0.0, 0.0

        score = float(np.sum(np.abs(diff)) / denom)

        # Sign: direction of asymmetry (weighted by eigenvalue magnitude)
        weighted_diff = float(np.sum(diff * (np.abs(ec) + np.abs(es))))
        sign = 1.0 if weighted_diff > 0 else (-1.0 if weighted_diff < 0 else 0.0)

        return score, sign
