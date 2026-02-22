"""ADMET property prediction with chirality awareness.

Predicts absorption, distribution, metabolism, excretion, and toxicity
properties while accounting for stereochemistry effects.
"""

from __future__ import annotations

from zynerji_chirality.ml.base_model import BaseChiralPredictor


class ChiralADMETPredictor(BaseChiralPredictor):
    """Predict ADMET properties with chirality awareness.

    Different enantiomers can have dramatically different ADMET profiles.
    This model captures stereochemistry-dependent differences through
    chirality-enhanced feature vectors.

    Example
    -------
    >>> predictor = ChiralADMETPredictor()
    >>> predictor.fit(smiles_list, logp_values)
    >>> result = predictor.predict("N[C@@H](C)C(=O)O")
    """

    def __init__(self, **kwargs):
        defaults = {
            "model_type": "gradient_boosting",
            "include_per_center": True,
            "include_ensemble": False,
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.1,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
