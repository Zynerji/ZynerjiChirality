"""ADMET profiling across all property models.

Generates comprehensive ADMET profiles for molecules by running
all available property-specific models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from zynerji_chirality.admet.property_model import ADMETPropertyPredictor, ADMETPropertyPrediction

logger = logging.getLogger(__name__)


@dataclass
class ADMETProfileEntry:
    """A single ADMET property prediction within a profile."""
    property_name: str
    predicted_value: float
    units: str
    risk_level: str
    confidence: float


@dataclass
class ADMETProfile:
    """Complete ADMET profile for a molecule."""
    smiles: str
    entries: list[ADMETProfileEntry] = field(default_factory=list)
    overall_risk: str = "low_risk"
    n_high_risk: int = 0
    n_moderate_risk: int = 0


class ADMETProfiler:
    """Generate ADMET profiles using all available property models.

    Parameters
    ----------
    models : dict[str, ADMETPropertyPredictor]
        Property-specific models keyed by property name.
    """

    def __init__(self, models: dict[str, ADMETPropertyPredictor] | None = None):
        self.models = models or {}

    def profile(self, smiles: str) -> ADMETProfile:
        """Generate complete ADMET profile for a molecule.

        Parameters
        ----------
        smiles : str
            Molecule SMILES.

        Returns
        -------
        ADMETProfile
            Profile with all property predictions and risk assessment.
        """
        entries = []

        for prop_name, model in self.models.items():
            try:
                pred = model.predict_property(smiles)
                entries.append(ADMETProfileEntry(
                    property_name=pred.property_name,
                    predicted_value=pred.predicted_value,
                    units=pred.units,
                    risk_level=pred.interpretation,
                    confidence=pred.confidence,
                ))
            except Exception as e:
                logger.debug("ADMET model %s failed for %s: %s", prop_name, smiles, e)

        n_high = sum(1 for e in entries if e.risk_level == "high_risk")
        n_moderate = sum(1 for e in entries if e.risk_level == "moderate_risk")

        if n_high > 0:
            overall_risk = "high_risk"
        elif n_moderate > 0:
            overall_risk = "moderate_risk"
        else:
            overall_risk = "low_risk"

        return ADMETProfile(
            smiles=smiles,
            entries=entries,
            overall_risk=overall_risk,
            n_high_risk=n_high,
            n_moderate_risk=n_moderate,
        )

    def profile_batch(self, smiles_list: list[str]) -> list[ADMETProfile]:
        """Generate ADMET profiles for multiple molecules."""
        return [self.profile(s) for s in smiles_list]

    @classmethod
    def load_from_directory(cls, models_dir: str | Path) -> ADMETProfiler:
        """Load all ADMET models from a directory.

        Scans for admet_*.pkl files.

        Parameters
        ----------
        models_dir : str or Path
            Directory containing saved model files.

        Returns
        -------
        ADMETProfiler
            Profiler with loaded models.
        """
        models_dir = Path(models_dir)
        models = {}

        for pkl_path in sorted(models_dir.glob("admet_*.pkl")):
            model = ADMETPropertyPredictor()
            model.load(pkl_path)
            models[model.property_name] = model
            logger.debug("Loaded ADMET model: %s", model.property_name)

        logger.info("Loaded %d ADMET models from %s", len(models), models_dir)
        return cls(models=models)
