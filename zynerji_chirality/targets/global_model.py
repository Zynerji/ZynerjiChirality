"""Global chirality sensitivity model across all targets.

A single model that predicts chirality sensitivity for any target,
using pair fingerprints plus target identity encoding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
from zynerji_chirality.ml.base_model import BaseChiralPredictor
from zynerji_chirality.targets.data_loader import TargetTrainingRecord

logger = logging.getLogger(__name__)


@dataclass
class GlobalPrediction:
    """Result of global chirality prediction."""
    predicted_fold_change: float
    is_chirality_sensitive: bool
    confidence: float
    target_chembl_id: str


class GlobalChiralityPredictor(BaseChiralPredictor):
    """Predict chirality sensitivity across all targets.

    Features: [pair_fp (4*nbits), target_one_hot (n_targets)]
    Unknown targets get a zero vector, generalizing from molecular features.

    Parameters
    ----------
    sensitivity_threshold : float
        Fold change threshold for chirality-sensitive classification.
    """

    def __init__(
        self,
        sensitivity_threshold: float = 3.0,
        nbits: int = 128,
        **model_kwargs,
    ):
        defaults = {
            "model_type": "hist_gradient_boosting",
            "max_iter": 150,
            "max_depth": 4,
        }
        defaults.update(model_kwargs)
        super().__init__(
            include_per_center=False,
            include_ensemble=False,
            nbits=nbits,
            **defaults,
        )
        self.sensitivity_threshold = sensitivity_threshold
        self._target_vocab: dict[str, int] = {}
        self._n_targets: int = 0

    def _combined_fingerprint(self, smiles_a: str, smiles_b: str) -> np.ndarray:
        """Build combined pair fingerprint (4*nbits)."""
        fp_a = chirality_fingerprint(smiles_a, nbits=self.nbits)
        fp_b = chirality_fingerprint(smiles_b, nbits=self.nbits)
        product = fp_a * fp_b
        diff = np.abs(fp_a - fp_b)
        return np.concatenate([fp_a, fp_b, product, diff])

    def _target_encoding(self, target_id: str) -> np.ndarray:
        """One-hot encode target ID. Unknown targets get zero vector."""
        vec = np.zeros(self._n_targets, dtype=np.float64)
        if target_id in self._target_vocab:
            vec[self._target_vocab[target_id]] = 1.0
        return vec

    def fit(self, records: list[TargetTrainingRecord]) -> None:
        """Train the global model on records from all targets.

        Parameters
        ----------
        records : list[TargetTrainingRecord]
            Flattened training data from all targets.
        """
        # Build target vocabulary
        target_ids = sorted({r.target_chembl_id for r in records})
        self._target_vocab = {tid: i for i, tid in enumerate(target_ids)}
        self._n_targets = len(target_ids)

        logger.info(
            "Training global model: %d records, %d targets",
            len(records), self._n_targets,
        )

        X_list = []
        y_list = []

        for record in records:
            try:
                pair_fp = self._combined_fingerprint(record.smiles_a, record.smiles_b)
                target_enc = self._target_encoding(record.target_chembl_id)
                features = np.concatenate([pair_fp, target_enc])
                X_list.append(features)
                y_list.append(record.fold_change)
            except Exception:
                continue

        if not X_list:
            raise ValueError("No valid features could be computed")

        X = np.array(X_list)
        y = np.array(y_list)

        self._model = self._create_model()
        self._model.fit(X, y)
        self._fitted = True

    def predict_for_target(
        self,
        smiles_a: str,
        smiles_b: str,
        target_id: str,
    ) -> GlobalPrediction:
        """Predict chirality sensitivity for a specific target.

        Parameters
        ----------
        smiles_a : str
            First enantiomer SMILES.
        smiles_b : str
            Second enantiomer SMILES.
        target_id : str
            ChEMBL target ID.

        Returns
        -------
        GlobalPrediction
            Predicted fold change and sensitivity classification.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        pair_fp = self._combined_fingerprint(smiles_a, smiles_b)
        target_enc = self._target_encoding(target_id)
        features = np.concatenate([pair_fp, target_enc]).reshape(1, -1)

        pred = float(self._model.predict(features)[0])
        pred = max(1.0, pred)

        is_sensitive = pred >= self.sensitivity_threshold
        confidence = min(pred / (self.sensitivity_threshold * 2), 1.0)

        return GlobalPrediction(
            predicted_fold_change=pred,
            is_chirality_sensitive=is_sensitive,
            confidence=confidence,
            target_chembl_id=target_id,
        )

    def predict_all_targets(
        self,
        smiles_a: str,
        smiles_b: str,
    ) -> list[GlobalPrediction]:
        """Predict chirality sensitivity for all known targets.

        Returns predictions sorted by predicted fold change (descending).
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        predictions = []
        for target_id in self._target_vocab:
            pred = self.predict_for_target(smiles_a, smiles_b, target_id)
            predictions.append(pred)

        predictions.sort(key=lambda p: p.predicted_fold_change, reverse=True)
        return predictions

    def save(self, path):
        """Save model with target vocabulary."""
        import pickle
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        data = {
            "model": self._model,
            "model_type": self.model_type,
            "include_per_center": self.include_per_center,
            "include_ensemble": self.include_ensemble,
            "nbits": self.nbits,
            "target_vocab": self._target_vocab,
            "n_targets": self._n_targets,
            "sensitivity_threshold": self.sensitivity_threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        """Load model with target vocabulary."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self.model_type = data["model_type"]
        self.include_per_center = data["include_per_center"]
        self.include_ensemble = data["include_ensemble"]
        self.nbits = data["nbits"]
        self._target_vocab = data.get("target_vocab", {})
        self._n_targets = data.get("n_targets", 0)
        self.sensitivity_threshold = data.get("sensitivity_threshold", 3.0)
        self._fitted = True
