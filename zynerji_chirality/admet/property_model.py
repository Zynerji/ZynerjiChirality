"""ADMET property prediction with chirality awareness.

Per-property models that predict ADMET values for individual molecules,
with property-specific risk interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zynerji_chirality.ml.base_model import BaseChiralPredictor
from zynerji_chirality.admet.data_loader import ADMETPropertyDataset, ADMETRecord


# Risk thresholds per property
_RISK_THRESHOLDS: dict[str, dict] = {
    "metabolic_stability": {
        # CLint: higher = faster metabolism = less stable
        "high_risk": 100.0,  # CLint > 100 uL/min/mg
        "moderate_risk": 30.0,
        "direction": "higher_is_worse",
    },
    "cyp_inhibition": {
        # IC50: lower = stronger inhibition = worse
        "high_risk": 1.0,  # IC50 < 1 uM
        "moderate_risk": 10.0,
        "direction": "lower_is_worse",
    },
    "herg": {
        # IC50: lower = stronger inhibition = worse (cardiac risk)
        "high_risk": 10.0,  # IC50 < 10 uM
        "moderate_risk": 30.0,
        "direction": "lower_is_worse",
    },
    "plasma_protein_binding": {
        # fu: lower = more bound = less free drug
        "high_risk": 1.0,  # fu < 1%
        "moderate_risk": 5.0,
        "direction": "lower_is_worse",
    },
    "solubility": {
        # Higher is better
        "high_risk": 1.0,  # < 1 ug/mL
        "moderate_risk": 10.0,
        "direction": "lower_is_worse",
    },
}


@dataclass
class ADMETPropertyPrediction:
    """Result of ADMET property prediction."""
    property_name: str
    predicted_value: float
    units: str
    confidence: float
    interpretation: str  # "low_risk", "moderate_risk", "high_risk"


class ADMETPropertyPredictor(BaseChiralPredictor):
    """Predict a single ADMET property with chirality awareness.

    Uses standard single-molecule featurization (not pair fingerprints),
    since ADMET is a per-molecule property.

    Parameters
    ----------
    property_name : str
        ADMET property this model predicts.
    units : str
        Units for the predicted value.
    """

    def __init__(
        self,
        property_name: str = "",
        units: str = "",
        nbits: int = 128,
        **model_kwargs,
    ):
        defaults = {
            "model_type": "gradient_boosting",
            "include_per_center": True,
            "include_ensemble": False,
            "n_estimators": 150,
            "max_depth": 4,
        }
        defaults.update(model_kwargs)
        super().__init__(nbits=nbits, **defaults)
        self.property_name = property_name
        self.units = units

    def fit_from_dataset(self, dataset: ADMETPropertyDataset) -> None:
        """Train from an ADMET property dataset.

        Parameters
        ----------
        dataset : ADMETPropertyDataset
            Training data with SMILES and measured values.
        """
        smiles_list = []
        values = []

        for record in dataset.records:
            smiles_list.append(record.smiles)
            values.append(record.standard_value)

        if not smiles_list:
            raise ValueError(f"No records for property {dataset.property_name}")

        self.property_name = dataset.property_name
        self.fit(smiles_list, np.array(values))

    def predict_property(self, smiles: str) -> ADMETPropertyPrediction:
        """Predict ADMET property for a molecule.

        Parameters
        ----------
        smiles : str
            Molecule SMILES.

        Returns
        -------
        ADMETPropertyPrediction
            Predicted value with risk interpretation.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit_from_dataset() first.")

        result = self.predict(smiles)
        value = result.prediction
        interpretation = self._interpret_risk(value)

        return ADMETPropertyPrediction(
            property_name=self.property_name,
            predicted_value=value,
            units=self.units,
            confidence=result.confidence or 0.5,
            interpretation=interpretation,
        )

    def _interpret_risk(self, value: float) -> str:
        """Interpret predicted value as risk level."""
        thresholds = _RISK_THRESHOLDS.get(self.property_name)
        if thresholds is None:
            return "low_risk"

        direction = thresholds["direction"]
        high = thresholds["high_risk"]
        moderate = thresholds["moderate_risk"]

        if direction == "lower_is_worse":
            if value < high:
                return "high_risk"
            if value < moderate:
                return "moderate_risk"
            return "low_risk"
        else:  # higher_is_worse
            if value > high:
                return "high_risk"
            if value > moderate:
                return "moderate_risk"
            return "low_risk"

    def save(self, path):
        """Save model with property metadata."""
        import pickle
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        data = {
            "model": self._model,
            "model_type": self.model_type,
            "include_per_center": self.include_per_center,
            "include_ensemble": self.include_ensemble,
            "nbits": self.nbits,
            "property_name": self.property_name,
            "units": self.units,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        """Load model with property metadata."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self.model_type = data["model_type"]
        self.include_per_center = data["include_per_center"]
        self.include_ensemble = data["include_ensemble"]
        self.nbits = data["nbits"]
        self.property_name = data.get("property_name", "")
        self.units = data.get("units", "")
        self._fitted = True
