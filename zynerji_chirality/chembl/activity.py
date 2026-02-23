"""Bioactivity data enrichment from ChEMBL.

Fetches IC50, Ki, EC50, and other activity data for enantiomer pairs
to identify chirality-sensitive drug targets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Iterator
import json
import math

from zynerji_chirality.chembl.pairs import EnantiomerPair

logger = logging.getLogger(__name__)


@dataclass
class ActivityRecord:
    """A single bioactivity measurement from ChEMBL."""
    chembl_id: str
    target_chembl_id: str
    target_name: str
    assay_type: str              # "B" (binding), "F" (functional), "A" (ADMET)
    standard_type: str           # IC50, Ki, EC50, etc.
    standard_value: float        # numeric value
    standard_units: str          # nM, uM, etc.
    pchembl_value: float | None  # standardized -log10(value)


@dataclass
class PairActivity:
    """Bioactivity data for an enantiomer pair."""
    pair: EnantiomerPair
    activities_a: list[ActivityRecord] = field(default_factory=list)
    activities_b: list[ActivityRecord] = field(default_factory=list)
    shared_targets: list[str] = field(default_factory=list)
    differential_targets: list[dict] = field(default_factory=list)
    gap_targets_a: list[str] = field(default_factory=list)
    gap_targets_b: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair.to_dict(),
            "activities_a": [asdict(a) for a in self.activities_a],
            "activities_b": [asdict(a) for a in self.activities_b],
            "shared_targets": self.shared_targets,
            "differential_targets": self.differential_targets,
            "gap_targets_a": self.gap_targets_a,
            "gap_targets_b": self.gap_targets_b,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PairActivity:
        from zynerji_chirality.chembl.pairs import EnantiomerPair
        return cls(
            pair=EnantiomerPair.from_dict(d["pair"]),
            activities_a=[ActivityRecord(**a) for a in d.get("activities_a", [])],
            activities_b=[ActivityRecord(**a) for a in d.get("activities_b", [])],
            shared_targets=d.get("shared_targets", []),
            differential_targets=d.get("differential_targets", []),
            gap_targets_a=d.get("gap_targets_a", []),
            gap_targets_b=d.get("gap_targets_b", []),
        )


def _fetch_activities_for_ids(chembl_ids: list[str], batch_size: int = 50) -> dict[str, list[ActivityRecord]]:
    """Fetch activities from ChEMBL web API for a list of molecule IDs.

    Returns dict mapping chembl_id → list[ActivityRecord].
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError(
            "chembl_webresource_client required. Install with: "
            "pip install zynerji-chirality[chembl]"
        )

    activity_api = new_client.activity
    result: dict[str, list[ActivityRecord]] = {cid: [] for cid in chembl_ids}

    for i in range(0, len(chembl_ids), batch_size):
        batch = chembl_ids[i:i + batch_size]
        logger.debug("Fetching activities for batch %d-%d", i, i + len(batch))

        try:
            activities = activity_api.filter(
                molecule_chembl_id__in=batch,
                standard_type__in=["IC50", "Ki", "EC50", "Kd", "GI50"],
            ).only(
                "molecule_chembl_id",
                "target_chembl_id",
                "target_pref_name",
                "assay_type",
                "standard_type",
                "standard_value",
                "standard_units",
                "pchembl_value",
            )

            for act in activities:
                cid = act.get("molecule_chembl_id", "")
                if cid not in result:
                    continue

                std_val = act.get("standard_value")
                if std_val is None:
                    continue

                try:
                    std_val = float(std_val)
                except (ValueError, TypeError):
                    continue

                pchembl = act.get("pchembl_value")
                if pchembl is not None:
                    try:
                        pchembl = float(pchembl)
                    except (ValueError, TypeError):
                        pchembl = None

                result[cid].append(ActivityRecord(
                    chembl_id=cid,
                    target_chembl_id=act.get("target_chembl_id", ""),
                    target_name=act.get("target_pref_name", ""),
                    assay_type=act.get("assay_type", ""),
                    standard_type=act.get("standard_type", ""),
                    standard_value=std_val,
                    standard_units=act.get("standard_units", ""),
                    pchembl_value=pchembl,
                ))

        except Exception as e:
            logger.warning("Failed to fetch batch %d: %s", i, e)

    return result


class ActivityEnricher:
    """Fetch and analyze bioactivity data for enantiomer pairs."""

    def enrich_pairs(
        self,
        pairs: list[EnantiomerPair],
        batch_size: int = 50,
    ) -> list[PairActivity]:
        """Fetch bioactivity data for all molecules in pairs.

        Parameters
        ----------
        pairs : list[EnantiomerPair]
            Enantiomer pairs to enrich.
        batch_size : int
            Number of ChEMBL IDs to query per API call.

        Returns
        -------
        list[PairActivity]
            Pairs enriched with activity data.
        """
        # Collect all unique ChEMBL IDs
        all_ids = set()
        for pair in pairs:
            if pair.mol_a.chembl_id:
                all_ids.add(pair.mol_a.chembl_id)
            if pair.mol_b.chembl_id:
                all_ids.add(pair.mol_b.chembl_id)

        logger.info("Fetching activities for %d unique molecules", len(all_ids))
        all_activities = _fetch_activities_for_ids(list(all_ids), batch_size=batch_size)

        # Build PairActivity for each pair
        result = []
        for pair in pairs:
            acts_a = all_activities.get(pair.mol_a.chembl_id, [])
            acts_b = all_activities.get(pair.mol_b.chembl_id, [])

            pa = PairActivity(pair=pair, activities_a=acts_a, activities_b=acts_b)
            self._analyze_targets(pa)
            result.append(pa)

        n_with_data = sum(1 for pa in result if pa.activities_a or pa.activities_b)
        logger.info(
            "Enriched %d pairs, %d have activity data",
            len(result), n_with_data,
        )
        return result

    def enrich_pairs_from_cache(
        self,
        pairs: list[EnantiomerPair],
        activities: dict[str, list[ActivityRecord]],
    ) -> list[PairActivity]:
        """Enrich pairs using pre-fetched activity data (no API calls)."""
        result = []
        for pair in pairs:
            acts_a = activities.get(pair.mol_a.chembl_id, [])
            acts_b = activities.get(pair.mol_b.chembl_id, [])

            pa = PairActivity(pair=pair, activities_a=acts_a, activities_b=acts_b)
            self._analyze_targets(pa)
            result.append(pa)
        return result

    def _analyze_targets(self, pa: PairActivity) -> None:
        """Identify shared targets, differential activity, and gaps."""
        targets_a: dict[str, list[ActivityRecord]] = {}
        for act in pa.activities_a:
            targets_a.setdefault(act.target_chembl_id, []).append(act)

        targets_b: dict[str, list[ActivityRecord]] = {}
        for act in pa.activities_b:
            targets_b.setdefault(act.target_chembl_id, []).append(act)

        set_a = set(targets_a.keys())
        set_b = set(targets_b.keys())

        pa.shared_targets = sorted(set_a & set_b)
        pa.gap_targets_a = sorted(set_a - set_b)  # tested for A only
        pa.gap_targets_b = sorted(set_b - set_a)  # tested for B only

        # Differential activity on shared targets
        pa.differential_targets = []
        for target_id in pa.shared_targets:
            acts_a_target = targets_a[target_id]
            acts_b_target = targets_b[target_id]

            # Use pchembl values if available, else standard values
            vals_a = [a.pchembl_value for a in acts_a_target if a.pchembl_value is not None]
            vals_b = [a.pchembl_value for a in acts_b_target if a.pchembl_value is not None]

            if not vals_a or not vals_b:
                # Fall back to standard values (same type only)
                for std_type in ["IC50", "Ki", "EC50"]:
                    sv_a = [a.standard_value for a in acts_a_target
                            if a.standard_type == std_type and a.standard_value > 0]
                    sv_b = [a.standard_value for a in acts_b_target
                            if a.standard_type == std_type and a.standard_value > 0]
                    if sv_a and sv_b:
                        # Convert to fold change
                        mean_a = sum(sv_a) / len(sv_a)
                        mean_b = sum(sv_b) / len(sv_b)
                        fold_change = max(mean_a, mean_b) / max(min(mean_a, mean_b), 1e-10)
                        target_name = acts_a_target[0].target_name
                        pa.differential_targets.append({
                            "target_id": target_id,
                            "target_name": target_name,
                            "type": std_type,
                            "mean_a": mean_a,
                            "mean_b": mean_b,
                            "fold_change": fold_change,
                            "units": acts_a_target[0].standard_units,
                        })
                        break
                continue

            # pChEMBL-based comparison
            mean_a = sum(vals_a) / len(vals_a)
            mean_b = sum(vals_b) / len(vals_b)
            diff = abs(mean_a - mean_b)
            fold_change = 10 ** diff  # pChEMBL is -log10

            target_name = acts_a_target[0].target_name
            pa.differential_targets.append({
                "target_id": target_id,
                "target_name": target_name,
                "type": "pChEMBL",
                "mean_a": mean_a,
                "mean_b": mean_b,
                "diff": diff,
                "fold_change": fold_change,
            })

    def find_differential_activity(
        self,
        pair_activities: list[PairActivity],
        min_fold_change: float = 3.0,
    ) -> list[PairActivity]:
        """Find pairs where enantiomers differ by >N-fold on same target.

        Parameters
        ----------
        pair_activities : list[PairActivity]
            Enriched pairs.
        min_fold_change : float
            Minimum fold change to consider differential.

        Returns
        -------
        list[PairActivity]
            Pairs with significant differential activity.
        """
        result = []
        for pa in pair_activities:
            significant = [
                dt for dt in pa.differential_targets
                if dt.get("fold_change", 0) >= min_fold_change
            ]
            if significant:
                result.append(pa)
        return result

    def save_enriched(self, pair_activities: list[PairActivity], path: str) -> None:
        """Save enriched pairs to JSON."""
        data = [pa.to_dict() for pa in pair_activities]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d enriched pairs to %s", len(pair_activities), path)

    def load_enriched(self, path: str) -> list[PairActivity]:
        """Load enriched pairs from JSON."""
        with open(path) as f:
            data = json.load(f)
        pairs = [PairActivity.from_dict(d) for d in data]
        logger.info("Loaded %d enriched pairs from %s", len(pairs), path)
        return pairs
