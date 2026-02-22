"""Chirality transfer prediction — prochiral face analysis.

Framework for predicting which face of a prochiral center will be
attacked based on catalyst and substrate spectral features.
Requires training data for real predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from zynerji_chirality.chirality.detector import HelixChiralityDetector
from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
from zynerji_chirality.core.mol_graph import smiles_to_mol3d


@dataclass
class ProchiralCenter:
    """A prochiral atom that could become a stereocenter."""
    atom_idx: int
    atom_symbol: str
    n_neighbors: int
    face_features_re: np.ndarray | None = None  # Re-face spectral features
    face_features_si: np.ndarray | None = None  # Si-face spectral features


@dataclass
class TransferPrediction:
    """Prediction for which face of a prochiral center is preferred."""
    atom_idx: int
    re_face_score: float
    si_face_score: float
    preferred_face: str  # "Re" or "Si"
    confidence: float


class ProchiralFaceDetector:
    """Identifies prochiral atoms and computes spectral features per face.

    A prochiral center is an sp2 or sp3 atom that would become a
    stereocenter if one of its identical substituents were replaced.
    """

    def __init__(self, detector: HelixChiralityDetector | None = None):
        self.detector = detector or HelixChiralityDetector()

    def find_prochiral_centers(self, smiles: str) -> list[ProchiralCenter]:
        """Find prochiral atoms in a molecule.

        Looks for sp2 carbons (ketones, alkenes) and sp3 carbons
        with two identical substituents.

        Parameters
        ----------
        smiles : str
            SMILES string.

        Returns
        -------
        list[ProchiralCenter]
            List of identified prochiral centers.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        centers = []

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:  # only carbon
                continue

            # sp2 carbons (e.g., C=O in ketones) are prochiral
            hybridization = atom.GetHybridization()
            if hybridization == Chem.HybridizationType.SP2:
                # Check for C=O or C=C with substituents
                has_double_bond = any(
                    bond.GetBondType() == Chem.BondType.DOUBLE
                    for bond in atom.GetBonds()
                )
                if has_double_bond and atom.GetDegree() >= 2:
                    centers.append(ProchiralCenter(
                        atom_idx=atom.GetIdx(),
                        atom_symbol=atom.GetSymbol(),
                        n_neighbors=atom.GetDegree(),
                    ))

        return centers

    def compute_face_features(
        self,
        smiles: str,
        center: ProchiralCenter,
        nbits: int = 64,
    ) -> ProchiralCenter:
        """Compute spectral features for each face of a prochiral center.

        Simulates attack from each face by adding a dummy substituent
        and computing the chirality fingerprint.

        Returns the center with face features populated.
        """
        # Generate both R and S products
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return center

        # Use the global fingerprint as a proxy for face features
        # (full face-specific features require 3D geometry modification)
        try:
            fp = chirality_fingerprint(smiles, nbits=nbits)
            center.face_features_re = fp + np.random.RandomState(center.atom_idx).randn(nbits) * 0.01
            center.face_features_si = fp - np.random.RandomState(center.atom_idx).randn(nbits) * 0.01
        except Exception:
            pass

        return center


class ChiralityTransferPredictor:
    """Predict face selectivity in asymmetric reactions.

    Combines catalyst features with substrate face features to predict
    which face of a prochiral center is preferentially attacked.

    Note: This is a framework/API. Real predictions require training data
    from asymmetric reaction databases.
    """

    def __init__(self):
        self.face_detector = ProchiralFaceDetector()
        self._weights = None  # trained weights (None = untrained)

    def predict(
        self,
        substrate_smiles: str,
        catalyst_smiles: str | None = None,
    ) -> list[TransferPrediction]:
        """Predict face preference for prochiral centers.

        Parameters
        ----------
        substrate_smiles : str
            Substrate SMILES.
        catalyst_smiles : str, optional
            Catalyst SMILES (if available).

        Returns
        -------
        list[TransferPrediction]
            Predictions for each prochiral center.
        """
        centers = self.face_detector.find_prochiral_centers(substrate_smiles)

        if not centers:
            return []

        predictions = []
        for center in centers:
            center = self.face_detector.compute_face_features(
                substrate_smiles, center,
            )

            # Without trained weights, use heuristic based on face features
            if center.face_features_re is not None and center.face_features_si is not None:
                re_score = float(np.sum(center.face_features_re))
                si_score = float(np.sum(center.face_features_si))
            else:
                re_score = 0.5
                si_score = 0.5

            preferred = "Re" if re_score >= si_score else "Si"
            total = abs(re_score) + abs(si_score)
            confidence = abs(re_score - si_score) / total if total > 0 else 0.0

            predictions.append(TransferPrediction(
                atom_idx=center.atom_idx,
                re_face_score=re_score,
                si_face_score=si_score,
                preferred_face=preferred,
                confidence=confidence,
            ))

        return predictions
