"""Target sensitivity profiling for enantiomer pairs.

Combines per-target and global models to produce a comprehensive
target sensitivity profile for any pair of enantiomers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from zynerji_chirality.targets.target_model import TargetChiralityModel, TargetPrediction
from zynerji_chirality.targets.global_model import GlobalChiralityPredictor, GlobalPrediction

logger = logging.getLogger(__name__)


@dataclass
class TargetSensitivityEntry:
    """A single target sensitivity prediction."""
    target_chembl_id: str
    target_name: str
    predicted_fold_change: float
    confidence: float
    model_source: str  # "per_target" or "global"


@dataclass
class TargetProfile:
    """Complete target sensitivity profile for an enantiomer pair."""
    smiles_a: str
    smiles_b: str
    entries: list[TargetSensitivityEntry] = field(default_factory=list)
    n_sensitive: int = 0
    top_target: str = ""
    top_fold_change: float = 0.0


class TargetProfiler:
    """Profile enantiomer pairs for target-specific chirality sensitivity.

    Uses per-target models where available, falls back to global model.

    Parameters
    ----------
    per_target_models : dict[str, TargetChiralityModel]
        Target-specific models keyed by target ChEMBL ID.
    global_model : GlobalChiralityPredictor or None
        Global fallback model.
    sensitivity_threshold : float
        Fold change threshold for flagging as sensitive.
    """

    def __init__(
        self,
        per_target_models: dict[str, TargetChiralityModel] | None = None,
        global_model: GlobalChiralityPredictor | None = None,
        sensitivity_threshold: float = 3.0,
    ):
        self.per_target_models = per_target_models or {}
        self.global_model = global_model
        self.sensitivity_threshold = sensitivity_threshold

    def profile(self, smiles_a: str, smiles_b: str) -> TargetProfile:
        """Generate target sensitivity profile for an enantiomer pair.

        Uses per-target models where available, global model as fallback.
        """
        entries = []

        # Per-target predictions
        for target_id, model in self.per_target_models.items():
            try:
                pred = model.predict_sensitivity(smiles_a, smiles_b)
                entries.append(TargetSensitivityEntry(
                    target_chembl_id=pred.target_chembl_id,
                    target_name=pred.target_name,
                    predicted_fold_change=pred.predicted_fold_change,
                    confidence=pred.confidence,
                    model_source="per_target",
                ))
            except Exception as e:
                logger.debug("Per-target model %s failed: %s", target_id, e)

        # Global model predictions for targets without per-target models
        per_target_ids = set(self.per_target_models.keys())
        if self.global_model is not None:
            try:
                global_preds = self.global_model.predict_all_targets(smiles_a, smiles_b)
                for pred in global_preds:
                    if pred.target_chembl_id not in per_target_ids:
                        entries.append(TargetSensitivityEntry(
                            target_chembl_id=pred.target_chembl_id,
                            target_name="",
                            predicted_fold_change=pred.predicted_fold_change,
                            confidence=pred.confidence,
                            model_source="global",
                        ))
            except Exception as e:
                logger.debug("Global model failed: %s", e)

        # Sort by fold change descending
        entries.sort(key=lambda e: e.predicted_fold_change, reverse=True)

        n_sensitive = sum(
            1 for e in entries
            if e.predicted_fold_change >= self.sensitivity_threshold
        )

        top_target = entries[0].target_chembl_id if entries else ""
        top_fold = entries[0].predicted_fold_change if entries else 0.0

        return TargetProfile(
            smiles_a=smiles_a,
            smiles_b=smiles_b,
            entries=entries,
            n_sensitive=n_sensitive,
            top_target=top_target,
            top_fold_change=top_fold,
        )

    def flag_screening_hits(
        self,
        pair_activities: list,
        top_n: int = 50,
    ) -> list:
        """Flag screening hits based on predicted differential sensitivity.

        Parameters
        ----------
        pair_activities : list[PairActivity]
            Enriched pairs from ChEMBL pipeline.
        top_n : int
            Maximum hits to return.

        Returns
        -------
        list[ScreeningHit]
            Hits with hit_type="predicted_differential".
        """
        from zynerji_chirality.chembl.screen import ScreeningHit

        hits = []
        for pa in pair_activities:
            smiles_a = pa.pair.mol_a.canonical_smiles
            smiles_b = pa.pair.mol_b.canonical_smiles

            try:
                profile = self.profile(smiles_a, smiles_b)
            except Exception:
                continue

            if profile.n_sensitive > 0:
                desc = (
                    f"Predicted chirality-sensitive at {profile.n_sensitive} target(s). "
                    f"Top: {profile.top_target} ({profile.top_fold_change:.1f}x)"
                )
                hits.append(ScreeningHit(
                    pair=pa.pair,
                    pair_activity=pa,
                    hit_type="predicted_differential",
                    description=desc,
                    priority_score=profile.top_fold_change * profile.n_sensitive,
                ))

            if len(hits) >= top_n:
                break

        hits.sort(key=lambda h: h.priority_score, reverse=True)
        return hits[:top_n]

    @classmethod
    def load_from_directory(cls, models_dir: str | Path) -> TargetProfiler:
        """Load all models from a directory.

        Scans for target_*.pkl (per-target) and global_model.pkl.

        Parameters
        ----------
        models_dir : str or Path
            Directory containing saved model files.

        Returns
        -------
        TargetProfiler
            Profiler with loaded models.
        """
        models_dir = Path(models_dir)
        per_target = {}
        global_model = None

        # Load per-target models
        for pkl_path in sorted(models_dir.glob("target_*.pkl")):
            model = TargetChiralityModel()
            model.load(pkl_path)
            per_target[model.target_chembl_id] = model
            logger.debug("Loaded per-target model: %s", model.target_chembl_id)

        # Load global model
        global_path = models_dir / "global_model.pkl"
        if global_path.exists():
            global_model = GlobalChiralityPredictor()
            global_model.load(global_path)
            logger.debug("Loaded global model")

        logger.info(
            "Loaded %d per-target models + %s global model from %s",
            len(per_target),
            "1" if global_model else "0",
            models_dir,
        )
        return cls(per_target_models=per_target, global_model=global_model)
