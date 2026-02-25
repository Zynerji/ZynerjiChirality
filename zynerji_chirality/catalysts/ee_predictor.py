"""Enantiomeric excess (ee) prediction from catalyst-substrate fingerprints.

Uses chirality fingerprints of both catalyst and substrate combined with
interaction features to predict ee% for asymmetric reactions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
from zynerji_chirality.ml.base_model import BaseChiralPredictor, PredictionResult


@dataclass
class EEPrediction:
    """Result of enantiomeric excess prediction."""
    ee_percent: float          # Predicted ee (0-100%)
    major_enantiomer: str      # "R" or "S"
    confidence: float          # Prediction confidence
    catalyst_fp: np.ndarray    # Catalyst fingerprint used
    substrate_fp: np.ndarray   # Substrate fingerprint used


class EEPredictor(BaseChiralPredictor):
    """Predict enantiomeric excess from catalyst + substrate fingerprints.

    Subclasses BaseChiralPredictor with a combined fingerprint that
    concatenates catalyst and substrate spectral fingerprints plus
    interaction features (element-wise product and difference).

    Example
    -------
    >>> predictor = EEPredictor()
    >>> # Train on known reactions
    >>> reactions = [
    ...     {"catalyst": "SMILES1", "substrate": "SMILES2", "ee": 95.0, "major": "R"},
    ...     ...
    ... ]
    >>> predictor.fit(reactions)
    >>> # Predict new reaction
    >>> pred = predictor.predict_ee("catalyst_smiles", "substrate_smiles")
    >>> print(f"Predicted ee: {pred.ee_percent:.1f}%")
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        nbits: int = 128,
        **model_kwargs,
    ):
        super().__init__(
            model_type=model_type,
            include_per_center=False,
            include_ensemble=False,
            nbits=nbits,
            **model_kwargs,
        )

    def _combined_fingerprint(
        self,
        catalyst: str,
        substrate: str,
    ) -> np.ndarray:
        """Build combined catalyst-substrate fingerprint.

        Concatenates:
        1. Catalyst chirality fingerprint (nbits)
        2. Substrate chirality fingerprint (nbits)
        3. Element-wise product (interaction, nbits)
        4. Element-wise absolute difference (nbits)

        Total length: 4 * nbits.
        """
        cat_fp = chirality_fingerprint(catalyst, nbits=self.nbits)
        sub_fp = chirality_fingerprint(substrate, nbits=self.nbits)

        product = cat_fp * sub_fp
        diff = np.abs(cat_fp - sub_fp)

        return np.concatenate([cat_fp, sub_fp, product, diff])

    def _featurize(self, smiles: str) -> np.ndarray:
        """Override base featurize — not used for ee prediction.

        Use _combined_fingerprint() instead for catalyst-substrate pairs.
        """
        return chirality_fingerprint(smiles, nbits=self.nbits)

    def fit(self, reactions: list[dict]) -> None:
        """Train the ee predictor on known reaction data.

        Parameters
        ----------
        reactions : list[dict]
            Training data, each with keys:
            - catalyst: str (SMILES)
            - substrate: str (SMILES)
            - ee: float (enantiomeric excess, 0-100)
            - major: str ("R" or "S") — optional
        """
        X_list = []
        y_list = []

        for rxn in reactions:
            try:
                fp = self._combined_fingerprint(rxn["catalyst"], rxn["substrate"])
                X_list.append(fp)
                y_list.append(rxn["ee"])
            except Exception:
                continue

        if not X_list:
            raise ValueError("No valid reaction fingerprints could be computed")

        X = np.array(X_list)
        y = np.array(y_list)

        self._model = self._create_model()
        self._model.fit(X, y)
        self._fitted = True

    def predict_ee(
        self,
        catalyst_smiles: str,
        substrate_smiles: str,
    ) -> EEPrediction:
        """Predict enantiomeric excess for a catalyst-substrate pair.

        Parameters
        ----------
        catalyst_smiles : str
            Catalyst SMILES.
        substrate_smiles : str
            Substrate SMILES.

        Returns
        -------
        EEPrediction
            Predicted ee with confidence and fingerprints.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        cat_fp = chirality_fingerprint(catalyst_smiles, nbits=self.nbits)
        sub_fp = chirality_fingerprint(substrate_smiles, nbits=self.nbits)

        combined = self._combined_fingerprint(catalyst_smiles, substrate_smiles)
        pred = float(self._model.predict(combined.reshape(1, -1))[0])

        # Clamp to valid range
        ee = max(0.0, min(100.0, abs(pred)))

        # Determine major enantiomer from sign
        # Positive prediction → R, negative → S
        major = "R" if pred >= 0 else "S"

        # Confidence from model (if available) or based on magnitude
        confidence = min(ee / 100.0, 1.0)

        return EEPrediction(
            ee_percent=ee,
            major_enantiomer=major,
            confidence=confidence,
            catalyst_fp=cat_fp,
            substrate_fp=sub_fp,
        )

    def cross_validate(
        self,
        reactions: list[dict],
        cv: int = 5,
    ) -> dict:
        """Cross-validate the ee predictor.

        Parameters
        ----------
        reactions : list[dict]
            Reaction data (same format as fit()).
        cv : int
            Number of cross-validation folds.

        Returns
        -------
        dict
            Cross-validation results with mean_r2, std_r2, scores.
        """
        from sklearn.model_selection import cross_val_score

        X_list = []
        y_list = []

        for rxn in reactions:
            try:
                fp = self._combined_fingerprint(rxn["catalyst"], rxn["substrate"])
                X_list.append(fp)
                y_list.append(rxn["ee"])
            except Exception:
                continue

        X = np.array(X_list)
        y = np.array(y_list)

        model = self._create_model()
        scores = cross_val_score(model, X, y, cv=min(cv, len(X)), scoring="r2")

        return {
            "mean_r2": float(np.mean(scores)),
            "std_r2": float(np.std(scores)),
            "scores": scores.tolist(),
        }
