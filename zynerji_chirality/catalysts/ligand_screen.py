"""Ligand screening for asymmetric catalysis.

Ranks candidate ligands by predicted enantioselectivity for a given substrate,
using either ML prediction (if a trained EEPredictor is available) or
heuristic chirality profile ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zynerji_chirality.chirality.detector import HelixChiralityDetector
from zynerji_chirality.chirality.fingerprint import chirality_fingerprint


@dataclass
class LigandScore:
    """Screening score for a candidate ligand."""
    ligand_smiles: str
    predicted_ee: float           # Predicted ee% (0-100, or chirality score if heuristic)
    fingerprint_distance: float   # Spectral distance to reference
    rank: int                     # 1 = best


class LigandScreener:
    """Screen ligands for enantioselectivity in asymmetric catalysis.

    Two modes:
    1. ML mode (with fitted EEPredictor): predicts ee% for each ligand-substrate pair
    2. Heuristic mode (no predictor): ranks by chirality score magnitude
    """

    def __init__(self, detector: HelixChiralityDetector | None = None):
        self._detector = detector or HelixChiralityDetector()

    def screen(
        self,
        substrate_smiles: str,
        ligand_smiles_list: list[str],
        ee_predictor=None,
    ) -> list[LigandScore]:
        """Screen candidate ligands for a substrate.

        Parameters
        ----------
        substrate_smiles : str
            Substrate SMILES.
        ligand_smiles_list : list[str]
            Candidate ligand SMILES to screen.
        ee_predictor : EEPredictor, optional
            Fitted predictor for ML-based screening.

        Returns
        -------
        list[LigandScore]
            Ranked list of ligand scores (best first).
        """
        if not ligand_smiles_list:
            return []

        scores = []

        if ee_predictor is not None and ee_predictor._fitted:
            # ML mode: predict ee for each ligand-substrate pair
            sub_fp = chirality_fingerprint(substrate_smiles, nbits=128)

            for lig_smi in ligand_smiles_list:
                try:
                    pred = ee_predictor.predict_ee(lig_smi, substrate_smiles)
                    lig_fp = chirality_fingerprint(lig_smi, nbits=128)
                    dist = float(np.linalg.norm(lig_fp - sub_fp))
                    scores.append(LigandScore(
                        ligand_smiles=lig_smi,
                        predicted_ee=pred.ee_percent,
                        fingerprint_distance=dist,
                        rank=0,
                    ))
                except Exception:
                    scores.append(LigandScore(
                        ligand_smiles=lig_smi,
                        predicted_ee=0.0,
                        fingerprint_distance=float("inf"),
                        rank=0,
                    ))
        else:
            # Heuristic mode: rank by chirality score (more chiral = potentially more selective)
            sub_fp = chirality_fingerprint(substrate_smiles, nbits=128)

            for lig_smi in ligand_smiles_list:
                try:
                    result = self._detector.detect(lig_smi)
                    lig_fp = chirality_fingerprint(lig_smi, nbits=128)
                    dist = float(np.linalg.norm(lig_fp - sub_fp))
                    scores.append(LigandScore(
                        ligand_smiles=lig_smi,
                        predicted_ee=result.chirality_score * 100,
                        fingerprint_distance=dist,
                        rank=0,
                    ))
                except Exception:
                    scores.append(LigandScore(
                        ligand_smiles=lig_smi,
                        predicted_ee=0.0,
                        fingerprint_distance=float("inf"),
                        rank=0,
                    ))

        # Sort by predicted ee (descending)
        scores.sort(key=lambda s: s.predicted_ee, reverse=True)

        # Assign ranks
        for i, s in enumerate(scores):
            s.rank = i + 1

        return scores


def rank_by_chirality_profile(ligands: list[str]) -> list[tuple[str, float]]:
    """Rank ligands by chirality score magnitude.

    Higher chirality score = more asymmetric = potentially more enantioselective.

    Parameters
    ----------
    ligands : list[str]
        List of ligand SMILES.

    Returns
    -------
    list[tuple[str, float]]
        Sorted (smiles, chirality_score) tuples, highest first.
    """
    detector = HelixChiralityDetector()
    results = []

    for smi in ligands:
        try:
            result = detector.detect(smi)
            results.append((smi, result.chirality_score))
        except Exception:
            results.append((smi, 0.0))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
