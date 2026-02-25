"""Per-target chirality sensitivity model.

Predicts whether a pair of enantiomers will show differential activity
at a specific drug target, using combined pair fingerprints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
from zynerji_chirality.ml.base_model import BaseChiralPredictor
from zynerji_chirality.targets.data_loader import TargetDataset, TargetTrainingRecord


@dataclass
class TargetPrediction:
    """Result of target-specific chirality prediction."""
    target_chembl_id: str
    target_name: str
    predicted_fold_change: float
    is_chirality_sensitive: bool
    confidence: float


class TargetChiralityModel(BaseChiralPredictor):
    """Predict chirality sensitivity for a specific drug target.

    Uses combined pair fingerprints: [fp_a, fp_b, fp_a*fp_b, |fp_a-fp_b|]
    following the same pattern as EEPredictor._combined_fingerprint.

    Parameters
    ----------
    target_chembl_id : str
        ChEMBL target ID this model is trained for.
    target_name : str
        Human-readable target name.
    sensitivity_threshold : float
        Fold change threshold to classify as chirality-sensitive.
    """

    def __init__(
        self,
        target_chembl_id: str = "",
        target_name: str = "",
        sensitivity_threshold: float = 3.0,
        nbits: int = 128,
        **model_kwargs,
    ):
        defaults = {
            "model_type": "gradient_boosting",
            "n_estimators": 150,
            "max_depth": 4,
        }
        defaults.update(model_kwargs)
        super().__init__(
            include_per_center=False,
            include_ensemble=False,
            nbits=nbits,
            **defaults,
        )
        self.target_chembl_id = target_chembl_id
        self.target_name = target_name
        self.sensitivity_threshold = sensitivity_threshold

    def _combined_fingerprint(self, smiles_a: str, smiles_b: str) -> np.ndarray:
        """Build combined pair fingerprint.

        Concatenates:
        1. Chirality fingerprint A (nbits)
        2. Chirality fingerprint B (nbits)
        3. Element-wise product (nbits)
        4. Element-wise absolute difference (nbits)

        Total length: 4 * nbits.
        """
        fp_a = chirality_fingerprint(smiles_a, nbits=self.nbits)
        fp_b = chirality_fingerprint(smiles_b, nbits=self.nbits)
        product = fp_a * fp_b
        diff = np.abs(fp_a - fp_b)
        return np.concatenate([fp_a, fp_b, product, diff])

    def fit_from_dataset(self, dataset: TargetDataset) -> None:
        """Train from a TargetDataset.

        Parameters
        ----------
        dataset : TargetDataset
            Per-target training data with enantiomer pairs and fold changes.
        """
        X_list = []
        y_list = []

        for record in dataset.records:
            try:
                fp = self._combined_fingerprint(record.smiles_a, record.smiles_b)
                X_list.append(fp)
                y_list.append(record.fold_change)
            except Exception:
                continue

        if not X_list:
            raise ValueError(f"No valid fingerprints for target {dataset.target_chembl_id}")

        X = np.array(X_list)
        y = np.array(y_list)

        self._model = self._create_model()
        self._model.fit(X, y)
        self._fitted = True
        self.target_chembl_id = dataset.target_chembl_id
        self.target_name = dataset.target_name

    def predict_sensitivity(
        self,
        smiles_a: str,
        smiles_b: str,
    ) -> TargetPrediction:
        """Predict chirality sensitivity for a pair at this target.

        Parameters
        ----------
        smiles_a : str
            First enantiomer SMILES.
        smiles_b : str
            Second enantiomer SMILES.

        Returns
        -------
        TargetPrediction
            Predicted fold change and sensitivity classification.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit_from_dataset() first.")

        fp = self._combined_fingerprint(smiles_a, smiles_b)
        pred = float(self._model.predict(fp.reshape(1, -1))[0])
        pred = max(1.0, pred)  # Fold change >= 1.0

        is_sensitive = pred >= self.sensitivity_threshold
        confidence = min(pred / (self.sensitivity_threshold * 2), 1.0)

        return TargetPrediction(
            target_chembl_id=self.target_chembl_id,
            target_name=self.target_name,
            predicted_fold_change=pred,
            is_chirality_sensitive=is_sensitive,
            confidence=confidence,
        )

    def save(self, path):
        """Save model with target metadata."""
        import pickle
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        data = {
            "model": self._model,
            "model_type": self.model_type,
            "include_per_center": self.include_per_center,
            "include_ensemble": self.include_ensemble,
            "nbits": self.nbits,
            "target_chembl_id": self.target_chembl_id,
            "target_name": self.target_name,
            "sensitivity_threshold": self.sensitivity_threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        """Load model with target metadata."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self.model_type = data["model_type"]
        self.include_per_center = data["include_per_center"]
        self.include_ensemble = data["include_ensemble"]
        self.nbits = data["nbits"]
        self.target_chembl_id = data.get("target_chembl_id", "")
        self.target_name = data.get("target_name", "")
        self.sensitivity_threshold = data.get("sensitivity_threshold", 3.0)
        self._fitted = True
