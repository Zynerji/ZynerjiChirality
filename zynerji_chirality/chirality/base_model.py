"""Base class for chirality-aware ML predictors.

Uses scikit-learn models with chirality-enhanced feature vectors.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from zynerji_chirality.ml.features import build_feature_vector


@dataclass
class PredictionResult:
    """Result from a model prediction."""
    prediction: float
    confidence: float | None = None
    feature_importances: dict[str, float] | None = None


class BaseChiralPredictor:
    """Base class for chirality-aware property prediction.

    Requires scikit-learn (optional dependency).
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        include_per_center: bool = True,
        include_ensemble: bool = False,
        nbits: int = 128,
        **model_kwargs,
    ):
        self.model_type = model_type
        self.include_per_center = include_per_center
        self.include_ensemble = include_ensemble
        self.nbits = nbits
        self.model_kwargs = model_kwargs
        self._model = None
        self._fitted = False

    def _create_model(self):
        """Create the sklearn model instance."""
        try:
            from sklearn.ensemble import (
                RandomForestRegressor,
                GradientBoostingRegressor,
                HistGradientBoostingRegressor,
            )
        except ImportError:
            raise ImportError(
                "scikit-learn is required for ML models. "
                "Install with: pip install zynerji-chirality[ml]"
            )

        if self.model_type == "random_forest":
            defaults = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
            defaults.update(self.model_kwargs)
            return RandomForestRegressor(**defaults)
        elif self.model_type == "gradient_boosting":
            defaults = {"n_estimators": 100, "random_state": 42, "max_depth": 5}
            defaults.update(self.model_kwargs)
            return GradientBoostingRegressor(**defaults)
        elif self.model_type == "hist_gradient_boosting":
            defaults = {"max_iter": 150, "random_state": 42, "max_depth": 4}
            defaults.update(self.model_kwargs)
            return HistGradientBoostingRegressor(**defaults)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _featurize(self, smiles: str) -> np.ndarray:
        """Convert SMILES to feature vector."""
        features = build_feature_vector(
            smiles,
            nbits=self.nbits,
            include_per_center=self.include_per_center,
            include_ensemble=self.include_ensemble,
        )
        return features["combined"]

    def _featurize_batch(self, smiles_list: list[str]) -> np.ndarray:
        """Convert list of SMILES to feature matrix."""
        return np.array([self._featurize(s) for s in smiles_list])

    def fit(self, smiles_list: list[str], targets: np.ndarray) -> None:
        """Train the model.

        Parameters
        ----------
        smiles_list : list[str]
            Training SMILES.
        targets : np.ndarray
            Target values (regression) or labels (classification).
        """
        X = self._featurize_batch(smiles_list)
        self._model = self._create_model()
        self._model.fit(X, targets)
        self._fitted = True

    def predict(self, smiles: str) -> PredictionResult:
        """Predict property for a single molecule."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._featurize(smiles).reshape(1, -1)
        pred = float(self._model.predict(X)[0])

        return PredictionResult(prediction=pred)

    def predict_batch(self, smiles_list: list[str]) -> list[PredictionResult]:
        """Predict for multiple molecules."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._featurize_batch(smiles_list)
        preds = self._model.predict(X)

        return [PredictionResult(prediction=float(p)) for p in preds]

    def cross_validate(
        self,
        smiles_list: list[str],
        targets: np.ndarray,
        cv: int = 5,
    ) -> dict:
        """Cross-validate the model.

        Returns dict with mean and std of scores.
        """
        from sklearn.model_selection import cross_val_score

        X = self._featurize_batch(smiles_list)
        model = self._create_model()
        scores = cross_val_score(model, X, targets, cv=cv, scoring="r2")

        return {
            "mean_r2": float(np.mean(scores)),
            "std_r2": float(np.std(scores)),
            "scores": scores.tolist(),
        }

    def save(self, path: str | Path) -> None:
        """Save fitted model to disk."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        data = {
            "model": self._model,
            "model_type": self.model_type,
            "include_per_center": self.include_per_center,
            "include_ensemble": self.include_ensemble,
            "nbits": self.nbits,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path) -> None:
        """Load a fitted model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._model = data["model"]
        self.model_type = data["model_type"]
        self.include_per_center = data["include_per_center"]
        self.include_ensemble = data["include_ensemble"]
        self.nbits = data["nbits"]
        self._fitted = True
