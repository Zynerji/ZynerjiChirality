"""Enantioselective activity prediction model.

Predicts how activity differs between enantiomers based on
chirality-enhanced molecular features.
"""

from __future__ import annotations

from zynerji_chirality.ml.base_model import BaseChiralPredictor


class ChiralActivityPredictor(BaseChiralPredictor):
    """Predict enantioselective pharmacological activity.

    Trained on pairs of enantiomers with known activity differences
    to predict whether chirality impacts biological activity.

    Example
    -------
    >>> predictor = ChiralActivityPredictor()
    >>> predictor.fit(smiles_list, activity_values)
    >>> result = predictor.predict("N[C@@H](C)C(=O)O")
    >>> print(result.prediction)
    """

    def __init__(self, **kwargs):
        defaults = {
            "model_type": "random_forest",
            "include_per_center": True,
            "include_ensemble": False,
            "n_estimators": 200,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
