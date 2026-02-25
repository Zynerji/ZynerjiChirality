"""ADMET data extraction from enriched ChEMBL data.

Classifies bioactivity records into ADMET property categories and
builds per-property datasets for model training.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Mapping from ADMET property names to ChEMBL standard_types and target keywords
ADMET_PROPERTIES: dict[str, dict] = {
    "metabolic_stability": {
        "standard_types": ["CLint", "T1/2"],
        "target_keywords": ["microsomal", "metabolic stability", "clearance", "liver microsome"],
        "description": "Hepatic metabolic stability (CLint, half-life)",
        "units": "uL/min/mg",
    },
    "cyp_inhibition": {
        "standard_types": ["IC50"],
        "target_keywords": ["CYP", "cytochrome", "P450"],
        "description": "Cytochrome P450 inhibition",
        "units": "uM",
    },
    "herg": {
        "standard_types": ["IC50", "EC50"],
        "target_keywords": ["hERG", "KCNH2", "potassium channel"],
        "description": "hERG channel inhibition (cardiac safety)",
        "units": "uM",
    },
    "plasma_protein_binding": {
        "standard_types": ["fu", "Plasma protein binding"],
        "target_keywords": ["plasma protein", "serum albumin", "protein binding"],
        "description": "Plasma protein binding (fraction unbound)",
        "units": "%",
    },
    "solubility": {
        "standard_types": ["Solubility"],
        "target_keywords": ["solubility", "aqueous"],
        "description": "Aqueous solubility",
        "units": "ug/mL",
    },
}


@dataclass
class ADMETRecord:
    """A single ADMET measurement."""
    smiles: str
    chembl_id: str
    property_name: str
    standard_type: str
    standard_value: float
    standard_units: str
    target_name: str


@dataclass
class ADMETPropertyDataset:
    """Training data for a single ADMET property."""
    property_name: str
    description: str
    records: list[ADMETRecord] = field(default_factory=list)
    pair_records: list[tuple[ADMETRecord, ADMETRecord]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "property_name": self.property_name,
            "description": self.description,
            "n_records": len(self.records),
            "n_pair_records": len(self.pair_records),
            "records": [asdict(r) for r in self.records],
        }


class ADMETDataLoader:
    """Extract ADMET property data from enriched enantiomer pairs."""

    def load_from_enriched(
        self,
        enriched_path: str | Path,
    ) -> dict[str, ADMETPropertyDataset]:
        """Load enriched_pairs.json and classify records into ADMET categories.

        Parameters
        ----------
        enriched_path : str or Path
            Path to enriched_pairs.json.

        Returns
        -------
        dict[str, ADMETPropertyDataset]
            One dataset per ADMET property.
        """
        with open(enriched_path) as f:
            enriched = json.load(f)

        logger.info("Loaded %d enriched pairs for ADMET extraction", len(enriched))

        datasets: dict[str, ADMETPropertyDataset] = {}
        for prop_name, prop_info in ADMET_PROPERTIES.items():
            datasets[prop_name] = ADMETPropertyDataset(
                property_name=prop_name,
                description=prop_info["description"],
            )

        # Collect records per molecule and classify
        for entry in enriched:
            pair = entry.get("pair", {})
            smiles_a = pair.get("mol_a", {}).get("canonical_smiles", "")
            smiles_b = pair.get("mol_b", {}).get("canonical_smiles", "")
            chembl_a = pair.get("mol_a", {}).get("chembl_id", "")
            chembl_b = pair.get("mol_b", {}).get("chembl_id", "")

            for side, smiles, chembl_id in [
                ("a", smiles_a, chembl_a),
                ("b", smiles_b, chembl_b),
            ]:
                for act in entry.get(f"activities_{side}", []):
                    prop_name = self._classify_record(act)
                    if prop_name is None:
                        continue

                    std_val = act.get("standard_value")
                    if std_val is None:
                        continue
                    try:
                        std_val = float(std_val)
                    except (ValueError, TypeError):
                        continue

                    record = ADMETRecord(
                        smiles=smiles,
                        chembl_id=chembl_id,
                        property_name=prop_name,
                        standard_type=act.get("standard_type", ""),
                        standard_value=std_val,
                        standard_units=act.get("standard_units", ""),
                        target_name=act.get("target_name", ""),
                    )
                    datasets[prop_name].records.append(record)

            # Build pair records where both enantiomers measured for same property
            for prop_name, ds in datasets.items():
                records_a = [r for r in ds.records if r.smiles == smiles_a]
                records_b = [r for r in ds.records if r.smiles == smiles_b]
                if records_a and records_b:
                    ds.pair_records.append((records_a[-1], records_b[-1]))

        for prop_name, ds in datasets.items():
            logger.info(
                "ADMET %s: %d records, %d pair records",
                prop_name, len(ds.records), len(ds.pair_records),
            )

        return datasets

    def _classify_record(self, activity: dict) -> str | None:
        """Classify an activity record into an ADMET property category.

        Matches by assay_type == "A" (ADMET) OR target keyword matching.
        """
        assay_type = activity.get("assay_type", "")
        standard_type = activity.get("standard_type", "")
        target_name = (activity.get("target_name", "") or "").lower()

        for prop_name, prop_info in ADMET_PROPERTIES.items():
            # Match by standard_type
            if standard_type in prop_info["standard_types"]:
                # For CYP inhibition, also check target keyword
                if prop_name == "cyp_inhibition":
                    if any(kw.lower() in target_name for kw in prop_info["target_keywords"]):
                        return prop_name
                    continue
                # For hERG, also check target keyword
                if prop_name == "herg":
                    if any(kw.lower() in target_name for kw in prop_info["target_keywords"]):
                        return prop_name
                    continue
                return prop_name

            # Match by target keyword (for ADMET assays)
            if assay_type == "A":
                if any(kw.lower() in target_name for kw in prop_info["target_keywords"]):
                    return prop_name

        return None

    def save_datasets(self, datasets: dict[str, ADMETPropertyDataset], path: str | Path) -> None:
        """Save extracted datasets to JSON."""
        data = {k: v.to_dict() for k, v in datasets.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved ADMET datasets to %s", path)

    def load_datasets(self, path: str | Path) -> dict[str, ADMETPropertyDataset]:
        """Load datasets from JSON."""
        with open(path) as f:
            data = json.load(f)
        datasets = {}
        for prop_name, d in data.items():
            records = [ADMETRecord(**r) for r in d.get("records", [])]
            datasets[prop_name] = ADMETPropertyDataset(
                property_name=d["property_name"],
                description=d["description"],
                records=records,
            )
        logger.info("Loaded ADMET datasets from %s", path)
        return datasets
