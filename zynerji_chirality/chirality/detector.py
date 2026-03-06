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
    apply_axial_perturbation,
    mol_to_adjacency,
    smiles_to_mol3d,
    smiles_to_mol3d_ensemble,
)
from zynerji_chirality.core.chiral_ordering import (
    cip_canonical_order,
    cip_canonical_order_per_center,
    find_axial_centers,
    reorder_adjacency,
)
from zynerji_chirality.core.spectral_match import build_angular_cost_matrix
from zynerji_chirality.chirality.fingerprint import (
    chirality_fingerprint,
    chirality_differential_fingerprint,
)

from rdkit import Chem
from rdkit.Chem import rdmolops


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
    fingerprint: np.ndarray | None = None               # 128-dim spectral FP
    differential_fingerprint: np.ndarray | None = None   # 128-dim chirality-only FP

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


@dataclass
class EnsembleResult:
    """Result of conformer ensemble chirality detection."""
    smiles: str
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    per_conformer_results: list[ChiralityResult]
    consensus_is_chiral: bool    # majority vote across conformers
    n_conformers: int

    def __repr__(self) -> str:
        label = "CHIRAL" if self.consensus_is_chiral else "ACHIRAL"
        return (
            f"EnsembleResult({label}, mean={self.mean_score:.4f}, "
            f"std={self.std_score:.4f}, n={self.n_conformers})"
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
        threshold: float = 0.003,
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

        # 1b. Meso compound check — achiral despite having chiral centers
        if self._is_meso(mol):
            adj = mol_to_adjacency(mol, weight_mode="bond_order")
            ordering = cip_canonical_order(mol, chirality_aware=False)
            spectral = compute_spectral_coords(
                reorder_adjacency(adj, ordering), self.params,
            )
            # Compute fingerprints even for meso (achiral) molecules
            fp = self._safe_fingerprint(smiles, mol)
            diff_fp = self._safe_differential_fingerprint(smiles, mol)
            return ChiralityResult(
                smiles=smiles,
                chirality_score=0.0,
                chirality_sign=0.0,
                is_chiral=False,
                cos_eigenvalues=spectral.eigenvalues_cos,
                sin_eigenvalues=spectral.eigenvalues_sin,
                spectral_coords=spectral,
                confidence=0.0,
                fingerprint=fp,
                differential_fingerprint=diff_fp,
            )

        # 2. Build standard adjacency (same for both enantiomers)
        adj = mol_to_adjacency(mol, weight_mode="bond_order")

        # 3. Baseline ordering (no chirality)
        ordering_base = cip_canonical_order(mol, chirality_aware=False)
        adj_base = reorder_adjacency(adj, ordering_base)
        spectral_base = compute_spectral_coords(adj_base, self.params)

        baseline_asym, _ = self._compute_asymmetry(
            spectral_base.eigenvalues_cos, spectral_base.eigenvalues_sin,
        )

        # 4. Bidirectional scoring: try BOTH shift directions (+1 and -1)
        # and take the max differential. This ensures detection works even
        # when one shift direction hits a low-sensitivity region of the
        # phase function (e.g., Propranolol R-shift gives 0.0003 while
        # S-shift gives 0.095 — same molecule, different sensitivity).
        # For achiral molecules: both shifts produce the same ordering
        # as baseline → max differential = 0.
        best_score = 0.0
        spectral_best = spectral_base

        for shift_dir in (+1, -1):
            ordering_shifted = cip_canonical_order(
                mol, chirality_aware=True, shift_override=shift_dir,
            )
            if ordering_shifted == ordering_base:
                continue  # No chiral centers affected this shift

            adj_shifted = reorder_adjacency(adj, ordering_shifted)
            spectral_shifted = compute_spectral_coords(adj_shifted, self.params)

            shifted_asym, _ = self._compute_asymmetry(
                spectral_shifted.eigenvalues_cos,
                spectral_shifted.eigenvalues_sin,
            )

            diff_score = abs(shifted_asym - baseline_asym)
            if diff_score > best_score:
                best_score = diff_score
                spectral_best = spectral_shifted

        score = best_score

        # 4b. Axial chirality fallback: if tetrahedral scoring found nothing,
        # check for atropisomeric (axial) chirality
        if score <= self.threshold:
            axial = find_axial_centers(mol)
            if axial:
                for direction in (+1.0, -1.0):
                    adj_axial = apply_axial_perturbation(
                        adj, mol, axial, direction=direction,
                    )
                    adj_axial_ordered = reorder_adjacency(adj_axial, ordering_base)
                    spectral_axial = compute_spectral_coords(
                        adj_axial_ordered, self.params,
                    )
                    axial_asym, _ = self._compute_asymmetry(
                        spectral_axial.eigenvalues_cos,
                        spectral_axial.eigenvalues_sin,
                    )
                    axial_diff = abs(axial_asym - baseline_asym)
                    if axial_diff > score:
                        score = axial_diff
                        spectral_best = spectral_axial

        is_chiral = score > self.threshold
        confidence = (score / self.threshold - 1.0) if self.threshold > 0 else 0.0

        # 5. R/S sign from CIP assignment (reliable, from RDKit)
        sign = self._cip_sign(mol) if is_chiral else 0.0

        spectral = spectral_best

        # Compute fingerprints
        fp = self._safe_fingerprint(smiles, mol)
        diff_fp = self._safe_differential_fingerprint(smiles, mol)

        return ChiralityResult(
            smiles=smiles,
            chirality_score=score,
            chirality_sign=sign,
            is_chiral=is_chiral,
            cos_eigenvalues=spectral.eigenvalues_cos,
            sin_eigenvalues=spectral.eigenvalues_sin,
            spectral_coords=spectral,
            confidence=max(confidence, 0.0),
            fingerprint=fp,
            differential_fingerprint=diff_fp,
        )

    def detect_per_center(
        self,
        smiles_or_mol: str | Chem.Mol,
    ) -> dict[int, ChiralityResult]:
        """Detect chirality independently for each stereocenter.

        Runs the full spectral pipeline with only one center's ordering
        shift active at a time, isolating each center's contribution.

        Parameters
        ----------
        smiles_or_mol : str or Chem.Mol
            SMILES string or RDKit Mol object.

        Returns
        -------
        dict[int, ChiralityResult]
            Mapping from chiral center atom index to its individual result.
            Empty dict if no chiral centers.
        """
        if isinstance(smiles_or_mol, str):
            smiles = smiles_or_mol
            mol = smiles_to_mol3d(smiles)
        else:
            mol = smiles_or_mol
            smiles = Chem.MolToSmiles(mol)

        adj = mol_to_adjacency(mol, weight_mode="bond_order")
        ordering_base = cip_canonical_order(mol, chirality_aware=False)
        adj_base = reorder_adjacency(adj, ordering_base)
        spectral_base = compute_spectral_coords(adj_base, self.params)
        baseline_asym, _ = self._compute_asymmetry(
            spectral_base.eigenvalues_cos, spectral_base.eigenvalues_sin,
        )

        results = {}
        for shift_dir in (+1, -1):
            per_center = cip_canonical_order_per_center(
                mol, shift_override=shift_dir,
            )
            for center_idx, ordering in per_center.items():
                if ordering == ordering_base:
                    continue

                adj_shifted = reorder_adjacency(adj, ordering)
                spectral = compute_spectral_coords(adj_shifted, self.params)
                shifted_asym, _ = self._compute_asymmetry(
                    spectral.eigenvalues_cos, spectral.eigenvalues_sin,
                )
                diff_score = abs(shifted_asym - baseline_asym)

                # Keep best score across shift directions
                prev = results.get(center_idx)
                if prev is None or diff_score > prev.chirality_score:
                    is_chiral = diff_score > self.threshold
                    confidence = (diff_score / self.threshold - 1.0) if self.threshold > 0 else 0.0

                    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
                    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
                    center_label = dict(centers).get(center_idx, "?")
                    sign = 1.0 if center_label == "R" else (-1.0 if center_label == "S" else 0.0)
                    if not is_chiral:
                        sign = 0.0

                    results[center_idx] = ChiralityResult(
                        smiles=smiles,
                        chirality_score=diff_score,
                        chirality_sign=sign,
                        is_chiral=is_chiral,
                        cos_eigenvalues=spectral.eigenvalues_cos,
                        sin_eigenvalues=spectral.eigenvalues_sin,
                        spectral_coords=spectral,
                        confidence=max(confidence, 0.0),
                    )

        return results

    def detect_ensemble(
        self,
        smiles: str,
        n_conformers: int = 10,
    ) -> EnsembleResult:
        """Detect chirality using multiple conformers for robust scoring.

        Generates an ensemble of 3D conformers and runs detection on each,
        then aggregates. Useful for flexible molecules where a single
        conformer may not capture the full conformational landscape.

        Parameters
        ----------
        smiles : str
            SMILES string.
        n_conformers : int
            Number of conformers to generate and test.

        Returns
        -------
        EnsembleResult
            Aggregated result with per-conformer data.
        """
        conformers = smiles_to_mol3d_ensemble(smiles, n_conformers=n_conformers)

        per_conf = [self.detect(mol) for mol in conformers]
        scores = np.array([r.chirality_score for r in per_conf])
        n_chiral = sum(1 for r in per_conf if r.is_chiral)

        return EnsembleResult(
            smiles=smiles,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            per_conformer_results=per_conf,
            consensus_is_chiral=n_chiral > len(per_conf) / 2,
            n_conformers=len(per_conf),
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
    def _safe_fingerprint(smiles: str, mol: Chem.Mol) -> np.ndarray | None:
        """Compute spectral fingerprint, returning None on failure."""
        try:
            return chirality_fingerprint(mol)
        except Exception:
            try:
                return chirality_fingerprint(smiles)
            except Exception:
                return None

    @staticmethod
    def _safe_differential_fingerprint(smiles: str, mol: Chem.Mol) -> np.ndarray | None:
        """Compute differential fingerprint, returning None on failure."""
        try:
            return chirality_differential_fingerprint(mol)
        except Exception:
            try:
                return chirality_differential_fingerprint(smiles)
            except Exception:
                return None

    @staticmethod
    def _is_meso(mol: Chem.Mol) -> bool:
        """Check if molecule is a meso compound.

        Meso compounds have chiral centers (R and S) that cancel due to
        internal symmetry. Detection: equal R/S count + stereo-stripped
        canonical ranks show R centers are symmetry-equivalent to S centers.
        """
        # Get CIP assignments on heavy-atom graph
        mol_noH = Chem.RemoveHs(mol)
        Chem.AssignStereochemistry(mol_noH, cleanIt=True, force=True)
        centers = Chem.FindMolChiralCenters(mol_noH, includeUnassigned=True)

        r_centers = [idx for idx, label in centers if label == "R"]
        s_centers = [idx for idx, label in centers if label == "S"]

        if len(r_centers) == 0 or len(r_centers) != len(s_centers):
            return False

        # Strip stereochemistry before ranking — stereo labels cause
        # otherwise-equivalent atoms to get different ranks
        mol_clean = Chem.RWMol(mol_noH)
        Chem.RemoveStereochemistry(mol_clean)
        ranks = list(Chem.CanonicalRankAtoms(mol_clean.GetMol(), breakTies=False))

        r_ranks = sorted([ranks[idx] for idx in r_centers])
        s_ranks = sorted([ranks[idx] for idx in s_centers])

        return r_ranks == s_ranks

    @staticmethod
    def _cip_sign(mol: Chem.Mol) -> float:
        """Determine chirality sign from CIP assignments.

        Returns +1.0 for R-dominant, -1.0 for S-dominant, 0.0 if equal or none.
        """
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        r_count = sum(1 for _, label in centers if label == "R")
        s_count = sum(1 for _, label in centers if label == "S")

        if r_count > s_count:
            return 1.0
        elif s_count > r_count:
            return -1.0
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
