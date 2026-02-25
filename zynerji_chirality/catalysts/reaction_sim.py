"""Catalytic reaction simulation — extends ReactionStereoAnalyzer with catalyst analysis.

Combines catalyst chirality detection with substrate prochiral face analysis
to predict stereochemical outcomes of catalytic reactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from zynerji_chirality.chirality.detector import HelixChiralityDetector, ChiralityResult
from zynerji_chirality.reactions.reaction_graph import (
    ReactionStereoAnalyzer,
    ReactionStereoResult,
)
from zynerji_chirality.reactions.transfer import (
    ProchiralFaceDetector,
    TransferPrediction,
)


@dataclass
class CatalyticReactionResult:
    """Result of catalytic reaction stereochemical analysis."""

    # Inherited from ReactionStereoResult
    reaction_smarts: str
    reactant_results: list[ChiralityResult]
    product_results: list[ChiralityResult]
    new_centers: list[int]
    inverted_centers: list[int]
    retained_centers: list[int]
    lost_centers: list[int]

    # Catalyst-specific fields
    catalyst_chirality: ChiralityResult
    predicted_selectivity: str     # "Re" or "Si" face preference
    face_selectivity_score: float  # 0-1, how strong the face preference is
    face_predictions: list[TransferPrediction] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"CatalyticReactionResult("
            f"catalyst_chiral={self.catalyst_chirality.is_chiral}, "
            f"selectivity={self.predicted_selectivity}({self.face_selectivity_score:.3f}), "
            f"new={len(self.new_centers)}, "
            f"inverted={len(self.inverted_centers)})"
        )


class CatalyticReactionAnalyzer:
    """Analyze stereochemical outcomes of catalytic reactions.

    Extends ReactionStereoAnalyzer with catalyst chirality analysis
    and face selectivity prediction.
    """

    def __init__(self, detector: HelixChiralityDetector | None = None):
        self.detector = detector or HelixChiralityDetector()
        self._reaction_analyzer = ReactionStereoAnalyzer(self.detector)
        self._face_detector = ProchiralFaceDetector(self.detector)

    def analyze_catalytic(
        self,
        rxn_smarts: str,
        catalyst_smiles: str,
    ) -> CatalyticReactionResult:
        """Analyze stereochemistry of a catalytic reaction.

        Parameters
        ----------
        rxn_smarts : str
            Reaction SMARTS/SMILES with atom mapping.
        catalyst_smiles : str
            Catalyst SMILES.

        Returns
        -------
        CatalyticReactionResult
            Complete stereochemical analysis including catalyst contribution.
        """
        # Analyze reaction stereochemistry
        rxn_result = self._reaction_analyzer.analyze(rxn_smarts)

        # Analyze catalyst chirality
        catalyst_result = self.detector.detect(catalyst_smiles)

        # Predict face selectivity from catalyst chirality
        # Extract substrate from reaction SMARTS
        parts = rxn_smarts.split(">>")
        substrate_smiles_list = parts[0].split(".") if len(parts) == 2 else []

        face_predictions = []
        for sub_smi in substrate_smiles_list:
            try:
                preds = self._face_detector.find_prochiral_centers(sub_smi)
                for center in preds:
                    center = self._face_detector.compute_face_features(sub_smi, center)
                    if center.face_features_re is not None and center.face_features_si is not None:
                        re_score = float(np.sum(center.face_features_re))
                        si_score = float(np.sum(center.face_features_si))

                        # Modulate by catalyst chirality
                        if catalyst_result.is_chiral:
                            bias = catalyst_result.chirality_sign * catalyst_result.chirality_score
                            re_score += bias
                            si_score -= bias

                        preferred = "Re" if re_score >= si_score else "Si"
                        total = abs(re_score) + abs(si_score)
                        confidence = abs(re_score - si_score) / total if total > 0 else 0.0

                        face_predictions.append(TransferPrediction(
                            atom_idx=center.atom_idx,
                            re_face_score=re_score,
                            si_face_score=si_score,
                            preferred_face=preferred,
                            confidence=confidence,
                        ))
            except Exception:
                continue

        # Overall selectivity from face predictions
        if face_predictions:
            avg_re = np.mean([p.re_face_score for p in face_predictions])
            avg_si = np.mean([p.si_face_score for p in face_predictions])
            predicted_selectivity = "Re" if avg_re >= avg_si else "Si"
            total = abs(avg_re) + abs(avg_si)
            face_score = abs(avg_re - avg_si) / total if total > 0 else 0.0
        elif catalyst_result.is_chiral:
            predicted_selectivity = "Re" if catalyst_result.chirality_sign > 0 else "Si"
            face_score = min(catalyst_result.chirality_score * 10, 1.0)
        else:
            predicted_selectivity = "Re"
            face_score = 0.0

        return CatalyticReactionResult(
            reaction_smarts=rxn_smarts,
            reactant_results=rxn_result.reactant_results,
            product_results=rxn_result.product_results,
            new_centers=rxn_result.new_centers,
            inverted_centers=rxn_result.inverted_centers,
            retained_centers=rxn_result.retained_centers,
            lost_centers=rxn_result.lost_centers,
            catalyst_chirality=catalyst_result,
            predicted_selectivity=predicted_selectivity,
            face_selectivity_score=float(face_score),
            face_predictions=face_predictions,
        )
