"""Chiral material detection — wraps core engine for material-level analysis.

Extends HelixChiralityDetector with material-specific properties:
circular dichroism, spin selectivity (CISS), optical rotation, and
band gap chirality shifts. Supports both SMILES and raw adjacency inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from zynerji_chirality.chirality.detector import HelixChiralityDetector, ChiralityResult
from zynerji_chirality.core.dual_helix import (
    HelixParams,
    SpectralCoords,
    compute_spectral_coords,
)
from zynerji_chirality.core.mol_graph import mol_to_adjacency
from zynerji_chirality.core.chiral_ordering import cip_canonical_order, reorder_adjacency
from zynerji_chirality.materials.properties import (
    estimate_cd_activity,
    estimate_ciss_score,
    classify_optical_rotation,
    band_gap_chirality_shift,
)


@dataclass
class MaterialChiralityResult:
    """Result of chirality detection for a material."""

    # Core chirality fields
    chirality_score: float
    chirality_sign: float
    is_chiral: bool
    spectral_coords: SpectralCoords

    # Material-specific properties
    cd_activity: float
    cd_sign: str           # "positive", "negative", "inactive"
    ciss_score: float      # chirality-induced spin selectivity [0, 1]
    optical_rotation: str  # "dextrorotatory (+)", "levorotatory (-)", "inactive"
    material_class: str    # "molecular", "polymeric", "crystalline", "metamaterial"

    # Optional source info
    smiles: str | None = None
    band_gap_shift: float = 0.0

    def __repr__(self) -> str:
        label = "CHIRAL" if self.is_chiral else "ACHIRAL"
        return (
            f"MaterialChiralityResult({label}, score={self.chirality_score:.4f}, "
            f"CD={self.cd_sign}({self.cd_activity:.4f}), "
            f"CISS={self.ciss_score:.3f}, "
            f"rotation={self.optical_rotation}, "
            f"class={self.material_class})"
        )


class ChiralMaterialDetector:
    """Detect chirality in materials using the dual-helix spectral engine.

    Wraps HelixChiralityDetector for material-level analysis with
    additional property predictions (CD, CISS, optical rotation).

    Supports both molecular SMILES input and raw adjacency matrices
    (for crystal structures, metamaterials, etc.).
    """

    def __init__(
        self,
        params: HelixParams | None = None,
        threshold: float = 0.003,
    ):
        self.params = params or HelixParams()
        self.threshold = threshold
        self._detector = HelixChiralityDetector(
            params=self.params,
            threshold=threshold,
        )

    def detect_material(
        self,
        smiles_or_adj,
        atom_labels: list[str] | None = None,
    ) -> MaterialChiralityResult:
        """Detect chirality of a material.

        Parameters
        ----------
        smiles_or_adj : str or csr_matrix or np.ndarray
            SMILES string for molecular materials, or adjacency matrix
            for crystal/metamaterial structures.
        atom_labels : list[str], optional
            Atom labels for adjacency matrix input.

        Returns
        -------
        MaterialChiralityResult
            Material-level chirality analysis including CD, CISS, rotation.
        """
        if isinstance(smiles_or_adj, str):
            return self._detect_from_smiles(smiles_or_adj)
        else:
            return self.detect_from_adjacency(smiles_or_adj, atom_labels)

    def _detect_from_smiles(self, smiles: str) -> MaterialChiralityResult:
        """Detect chirality from molecular SMILES."""
        result = self._detector.detect(smiles)
        spectral = result.spectral_coords

        cd_intensity, cd_sign = estimate_cd_activity(spectral)
        ciss = estimate_ciss_score(spectral)
        rotation = classify_optical_rotation(result.chirality_sign, result.chirality_score)
        bg_shift = band_gap_chirality_shift(spectral)

        return MaterialChiralityResult(
            chirality_score=result.chirality_score,
            chirality_sign=result.chirality_sign,
            is_chiral=result.is_chiral,
            spectral_coords=spectral,
            cd_activity=cd_intensity,
            cd_sign=cd_sign,
            ciss_score=ciss,
            optical_rotation=rotation,
            material_class="molecular",
            smiles=smiles,
            band_gap_shift=bg_shift,
        )

    def detect_from_adjacency(
        self,
        adj: np.ndarray | csr_matrix,
        atom_labels: list[str] | None = None,
    ) -> MaterialChiralityResult:
        """Detect chirality from a raw adjacency matrix.

        For crystal structures, metamaterials, or any non-SMILES input.

        Parameters
        ----------
        adj : np.ndarray or csr_matrix
            Adjacency matrix (n x n), symmetric.
        atom_labels : list[str], optional
            Element labels for each node.

        Returns
        -------
        MaterialChiralityResult
            Material-level chirality analysis.
        """
        if isinstance(adj, np.ndarray):
            adj = csr_matrix(adj)

        spectral = compute_spectral_coords(adj, self.params)

        # Compute asymmetry score
        score, sign = HelixChiralityDetector._compute_asymmetry(
            spectral.eigenvalues_cos,
            spectral.eigenvalues_sin,
        )
        is_chiral = score > self.threshold

        cd_intensity, cd_sign = estimate_cd_activity(spectral)
        ciss = estimate_ciss_score(spectral)
        rotation = classify_optical_rotation(sign if is_chiral else 0.0, score)
        bg_shift = band_gap_chirality_shift(spectral)

        # Classify material type based on graph properties
        n = adj.shape[0]
        if n >= 20:
            material_class = "crystalline"
        elif atom_labels and len(set(atom_labels)) <= 3:
            material_class = "metamaterial"
        else:
            material_class = "molecular"

        return MaterialChiralityResult(
            chirality_score=score,
            chirality_sign=sign if is_chiral else 0.0,
            is_chiral=is_chiral,
            spectral_coords=spectral,
            cd_activity=cd_intensity,
            cd_sign=cd_sign,
            ciss_score=ciss,
            optical_rotation=rotation,
            material_class=material_class,
            band_gap_shift=bg_shift,
        )

    def predict_cd_spectrum(self, smiles: str) -> dict:
        """Approximate CD spectral features from eigenvalue distribution.

        Parameters
        ----------
        smiles : str
            SMILES string.

        Returns
        -------
        dict
            CD spectral features: peaks, sign, intensity, bandwidth_estimate.
        """
        result = self._detector.detect(smiles)
        spectral = result.spectral_coords

        ecos = spectral.eigenvalues_cos
        esin = spectral.eigenvalues_sin
        k = min(len(ecos), len(esin))

        if k == 0:
            return {"peaks": [], "sign": "inactive", "intensity": 0.0}

        diff = ecos[:k] - esin[:k]
        cd_intensity, cd_sign = estimate_cd_activity(spectral)

        # Approximate CD peaks from eigenvalue differences
        peaks = []
        for i in range(k):
            if abs(diff[i]) > 1e-6:
                peaks.append({
                    "mode_index": i,
                    "eigenvalue_cos": float(ecos[i]),
                    "eigenvalue_sin": float(esin[i]),
                    "cd_contribution": float(diff[i]),
                })

        return {
            "peaks": peaks,
            "sign": cd_sign,
            "intensity": cd_intensity,
            "n_modes": k,
        }

    def predict_spin_selectivity(self, smiles: str) -> float:
        """Predict CISS score from spectral gap magnitude.

        Parameters
        ----------
        smiles : str
            SMILES string.

        Returns
        -------
        float
            CISS score in [0, 1].
        """
        result = self._detector.detect(smiles)
        return estimate_ciss_score(result.spectral_coords)
