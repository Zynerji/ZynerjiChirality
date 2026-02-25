"""Data extraction for target-specific chirality models.

Loads enriched enantiomer pair data and organizes it per-target
for training target-specific chirality sensitivity models.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TargetTrainingRecord:
    """A single training record for target chirality models."""
    smiles_a: str
    smiles_b: str
    target_chembl_id: str
    target_name: str
    fold_change: float
    is_differential: bool
    mean_activity_a: float
    mean_activity_b: float


@dataclass
class TargetDataset:
    """Training data for a single target."""
    target_chembl_id: str
    target_name: str
    records: list[TargetTrainingRecord] = field(default_factory=list)
    n_differential: int = 0
    n_non_differential: int = 0


class TargetDataExtractor:
    """Extract per-target training data from enriched enantiomer pairs."""

    def extract_from_enriched(
        self,
        enriched_path: str | Path,
        min_fold_change: float = 3.0,
    ) -> list[TargetDataset]:
        """Load enriched_pairs.json and extract per-target training records.

        Parameters
        ----------
        enriched_path : str or Path
            Path to enriched_pairs.json from the ChEMBL pipeline.
        min_fold_change : float
            Fold change threshold to classify as differential.

        Returns
        -------
        list[TargetDataset]
            One dataset per target with sufficient data.
        """
        with open(enriched_path) as f:
            enriched = json.load(f)

        logger.info("Loaded %d enriched pairs", len(enriched))

        # Collect records per target
        target_records: dict[str, list[TargetTrainingRecord]] = defaultdict(list)
        target_names: dict[str, str] = {}

        for entry in enriched:
            pair = entry.get("pair", {})
            smiles_a = pair.get("mol_a", {}).get("canonical_smiles", "")
            smiles_b = pair.get("mol_b", {}).get("canonical_smiles", "")

            if not smiles_a or not smiles_b:
                continue

            # Differential targets (positive examples)
            for dt in entry.get("differential_targets", []):
                target_id = dt.get("target_id", "")
                if not target_id:
                    continue

                fold_change = dt.get("fold_change", 1.0)
                target_name = dt.get("target_name", "")
                target_names[target_id] = target_name

                record = TargetTrainingRecord(
                    smiles_a=smiles_a,
                    smiles_b=smiles_b,
                    target_chembl_id=target_id,
                    target_name=target_name,
                    fold_change=fold_change,
                    is_differential=fold_change >= min_fold_change,
                    mean_activity_a=dt.get("mean_a", 0.0),
                    mean_activity_b=dt.get("mean_b", 0.0),
                )
                target_records[target_id].append(record)

            # Shared targets with low fold change (negative examples)
            for target_id in entry.get("shared_targets", []):
                if target_id in {dt.get("target_id") for dt in entry.get("differential_targets", [])}:
                    continue  # Already added as differential

                # This target had both enantiomers tested but no significant difference
                target_records[target_id].append(TargetTrainingRecord(
                    smiles_a=smiles_a,
                    smiles_b=smiles_b,
                    target_chembl_id=target_id,
                    target_name=target_names.get(target_id, ""),
                    fold_change=1.0,  # Near-equal activity
                    is_differential=False,
                    mean_activity_a=0.0,
                    mean_activity_b=0.0,
                ))

        # Build datasets
        datasets = []
        for target_id, records in target_records.items():
            n_diff = sum(1 for r in records if r.is_differential)
            n_non = len(records) - n_diff
            datasets.append(TargetDataset(
                target_chembl_id=target_id,
                target_name=target_names.get(target_id, ""),
                records=records,
                n_differential=n_diff,
                n_non_differential=n_non,
            ))

        datasets.sort(key=lambda d: len(d.records), reverse=True)
        logger.info(
            "Extracted %d target datasets (%d total records)",
            len(datasets), sum(len(d.records) for d in datasets),
        )
        return datasets

    def get_global_dataset(
        self,
        datasets: list[TargetDataset],
    ) -> tuple[list[TargetTrainingRecord], list[str]]:
        """Flatten all datasets into a single global training set.

        Returns
        -------
        tuple[list[TargetTrainingRecord], list[str]]
            (all_records, unique_target_ids)
        """
        all_records = []
        target_ids = set()
        for ds in datasets:
            all_records.extend(ds.records)
            target_ids.add(ds.target_chembl_id)
        return all_records, sorted(target_ids)

    def save_datasets(self, datasets: list[TargetDataset], path: str | Path) -> None:
        """Save extracted datasets to JSON."""
        data = []
        for ds in datasets:
            data.append({
                "target_chembl_id": ds.target_chembl_id,
                "target_name": ds.target_name,
                "n_differential": ds.n_differential,
                "n_non_differential": ds.n_non_differential,
                "records": [asdict(r) for r in ds.records],
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d datasets to %s", len(datasets), path)

    def load_datasets(self, path: str | Path) -> list[TargetDataset]:
        """Load datasets from JSON."""
        with open(path) as f:
            data = json.load(f)
        datasets = []
        for d in data:
            records = [TargetTrainingRecord(**r) for r in d["records"]]
            datasets.append(TargetDataset(
                target_chembl_id=d["target_chembl_id"],
                target_name=d["target_name"],
                records=records,
                n_differential=d["n_differential"],
                n_non_differential=d["n_non_differential"],
            ))
        logger.info("Loaded %d datasets from %s", len(datasets), path)
        return datasets
