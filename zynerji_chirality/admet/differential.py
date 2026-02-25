"""Differential ADMET analysis between enantiomers.

Compares ADMET profiles of two enantiomers to identify properties
where chirality significantly affects the ADMET outcome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from zynerji_chirality.admet.profiler import ADMETProfiler, ADMETProfile

logger = logging.getLogger(__name__)


@dataclass
class DifferentialADMETEntry:
    """Comparison of a single ADMET property between two enantiomers."""
    property_name: str
    value_a: float
    value_b: float
    fold_difference: float
    risk_a: str
    risk_b: str
    risk_divergent: bool


@dataclass
class DifferentialADMETResult:
    """Complete differential ADMET analysis for an enantiomer pair."""
    smiles_a: str
    smiles_b: str
    entries: list[DifferentialADMETEntry] = field(default_factory=list)
    n_divergent: int = 0
    worst_divergence: float = 0.0
    recommendation: str = "chirality_neutral"  # or "chirality_matters"


class DifferentialADMETAnalyzer:
    """Analyze differential ADMET between enantiomers.

    Parameters
    ----------
    profiler : ADMETProfiler
        ADMET profiler with loaded models.
    divergence_threshold : float
        Fold difference threshold to flag as divergent.
    """

    def __init__(
        self,
        profiler: ADMETProfiler,
        divergence_threshold: float = 2.0,
    ):
        self.profiler = profiler
        self.divergence_threshold = divergence_threshold

    def analyze(self, smiles_a: str, smiles_b: str) -> DifferentialADMETResult:
        """Compare ADMET profiles of two enantiomers.

        Parameters
        ----------
        smiles_a : str
            First enantiomer SMILES.
        smiles_b : str
            Second enantiomer SMILES.

        Returns
        -------
        DifferentialADMETResult
            Comparison with divergence flags and recommendation.
        """
        profile_a = self.profiler.profile(smiles_a)
        profile_b = self.profiler.profile(smiles_b)

        # Match entries by property name
        entries_a = {e.property_name: e for e in profile_a.entries}
        entries_b = {e.property_name: e for e in profile_b.entries}

        diff_entries = []
        for prop_name in set(entries_a.keys()) & set(entries_b.keys()):
            ea = entries_a[prop_name]
            eb = entries_b[prop_name]

            # Compute fold difference
            min_val = min(abs(ea.predicted_value), abs(eb.predicted_value))
            max_val = max(abs(ea.predicted_value), abs(eb.predicted_value))
            fold_diff = max_val / max(min_val, 1e-10)

            risk_divergent = ea.risk_level != eb.risk_level

            diff_entries.append(DifferentialADMETEntry(
                property_name=prop_name,
                value_a=ea.predicted_value,
                value_b=eb.predicted_value,
                fold_difference=fold_diff,
                risk_a=ea.risk_level,
                risk_b=eb.risk_level,
                risk_divergent=risk_divergent,
            ))

        n_divergent = sum(
            1 for e in diff_entries
            if e.fold_difference >= self.divergence_threshold or e.risk_divergent
        )
        worst = max((e.fold_difference for e in diff_entries), default=0.0)

        recommendation = (
            "chirality_matters" if n_divergent > 0 else "chirality_neutral"
        )

        return DifferentialADMETResult(
            smiles_a=smiles_a,
            smiles_b=smiles_b,
            entries=diff_entries,
            n_divergent=n_divergent,
            worst_divergence=worst,
            recommendation=recommendation,
        )

    def flag_safety_risks(
        self,
        pair_activities: list,
        top_n: int = 50,
    ) -> list:
        """Flag enantiomer pairs with divergent ADMET safety profiles.

        Parameters
        ----------
        pair_activities : list[PairActivity]
            Enriched pairs from ChEMBL pipeline.
        top_n : int
            Maximum hits to return.

        Returns
        -------
        list[ScreeningHit]
            Hits with hit_type="admet_divergence".
        """
        from zynerji_chirality.chembl.screen import ScreeningHit

        hits = []
        for pa in pair_activities:
            smiles_a = pa.pair.mol_a.canonical_smiles
            smiles_b = pa.pair.mol_b.canonical_smiles

            try:
                result = self.analyze(smiles_a, smiles_b)
            except Exception:
                continue

            if result.n_divergent > 0:
                desc = (
                    f"ADMET divergence: {result.n_divergent} property(ies) differ. "
                    f"Worst: {result.worst_divergence:.1f}x. "
                    f"Recommendation: {result.recommendation}"
                )
                hits.append(ScreeningHit(
                    pair=pa.pair,
                    pair_activity=pa,
                    hit_type="admet_divergence",
                    description=desc,
                    priority_score=result.worst_divergence * result.n_divergent,
                ))

            if len(hits) >= top_n:
                break

        hits.sort(key=lambda h: h.priority_score, reverse=True)
        return hits[:top_n]
