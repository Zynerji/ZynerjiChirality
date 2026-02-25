"""Training pipeline for target-specific chirality models.

Orchestrates data extraction, per-target model training, global model
training, and checkpoint/resume support.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from zynerji_chirality.targets.data_loader import TargetDataExtractor, TargetDataset
from zynerji_chirality.targets.target_model import TargetChiralityModel
from zynerji_chirality.targets.global_model import GlobalChiralityPredictor

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Summary of model training."""
    n_targets_trained: int
    n_targets_skipped: int
    global_model_r2: float | None
    per_target_r2: dict[str, float]
    elapsed_seconds: float


class TargetModelTrainer:
    """Train all target-specific and global chirality models.

    Parameters
    ----------
    work_dir : str or Path
        Output directory for models and checkpoints.
    nbits : int
        Fingerprint bit count.
    min_records : int
        Minimum records per target for per-target model training.
    cv_folds : int
        Cross-validation folds for evaluation.
    """

    def __init__(
        self,
        work_dir: str | Path = "targets_work",
        nbits: int = 128,
        min_records: int = 20,
        cv_folds: int = 5,
    ):
        self.work_dir = Path(work_dir)
        self.nbits = nbits
        self.min_records = min_records
        self.cv_folds = cv_folds

    def train_all(
        self,
        enriched_path: str | Path,
        resume: bool = False,
    ) -> TrainingResult:
        """Full training pipeline: extract -> train per-target -> train global -> save.

        Parameters
        ----------
        enriched_path : str or Path
            Path to enriched_pairs.json.
        resume : bool
            If True, skip targets already trained.

        Returns
        -------
        TrainingResult
            Training summary with R2 scores.
        """
        start = time.time()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Load or compute done_ids for resume
        done_path = self.work_dir / "done_targets.json"
        done_ids: set[str] = set()
        if resume and done_path.exists():
            with open(done_path) as f:
                done_ids = set(json.load(f))
            logger.info("Resuming: %d targets already trained", len(done_ids))

        # Extract data
        datasets_path = self.work_dir / "target_datasets.json"
        extractor = TargetDataExtractor()

        if resume and datasets_path.exists():
            datasets = extractor.load_datasets(datasets_path)
        else:
            datasets = extractor.extract_from_enriched(enriched_path)
            extractor.save_datasets(datasets, datasets_path)

        # Train per-target models
        n_trained = 0
        n_skipped = 0
        per_target_r2: dict[str, float] = {}

        for ds in datasets:
            if ds.target_chembl_id in done_ids:
                n_skipped += 1
                continue

            if len(ds.records) < self.min_records:
                n_skipped += 1
                continue

            try:
                model = TargetChiralityModel(
                    target_chembl_id=ds.target_chembl_id,
                    target_name=ds.target_name,
                    nbits=self.nbits,
                )
                model.fit_from_dataset(ds)

                # Save model
                model_path = self.work_dir / f"target_{ds.target_chembl_id}.pkl"
                model.save(model_path)

                n_trained += 1
                done_ids.add(ds.target_chembl_id)

                # Checkpoint done_ids
                with open(done_path, "w") as f:
                    json.dump(sorted(done_ids), f)

                logger.info(
                    "Trained target %s (%s): %d records",
                    ds.target_chembl_id, ds.target_name, len(ds.records),
                )
            except Exception as e:
                logger.warning("Failed target %s: %s", ds.target_chembl_id, e)
                n_skipped += 1

        # Train global model
        global_r2 = None
        all_records, _ = extractor.get_global_dataset(datasets)

        if len(all_records) >= self.min_records:
            try:
                global_model = GlobalChiralityPredictor(nbits=self.nbits)
                global_model.fit(all_records)
                global_model.save(self.work_dir / "global_model.pkl")
                logger.info("Trained global model: %d records", len(all_records))
            except Exception as e:
                logger.warning("Failed to train global model: %s", e)
        else:
            logger.warning(
                "Not enough records for global model: %d < %d",
                len(all_records), self.min_records,
            )

        elapsed = time.time() - start

        # Save training report
        result = TrainingResult(
            n_targets_trained=n_trained,
            n_targets_skipped=n_skipped,
            global_model_r2=global_r2,
            per_target_r2=per_target_r2,
            elapsed_seconds=elapsed,
        )

        report = {
            "n_targets_trained": result.n_targets_trained,
            "n_targets_skipped": result.n_targets_skipped,
            "global_model_r2": result.global_model_r2,
            "elapsed_seconds": round(result.elapsed_seconds, 1),
            "total_records": len(all_records),
            "total_datasets": len(datasets),
        }
        with open(self.work_dir / "training_report.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info(
            "Training complete: %d targets trained, %d skipped, %.1fs",
            n_trained, n_skipped, elapsed,
        )
        return result
