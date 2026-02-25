"""Chirality-aware matched molecular pair (MMP) analysis.

Identifies beneficial single-center stereochemistry flips
by scoring all single-flip variants of a molecule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from zynerji_chirality.generative.enumerator import StereoisomerEnumerator
from zynerji_chirality.generative.scorer import ChiralityScorer, ChiralityScore

logger = logging.getLogger(__name__)


@dataclass
class ChiralMMP:
    """A chirality matched molecular pair (single-center flip)."""
    smiles_a: str
    smiles_b: str
    transformation: str
    flipped_centers: list[int]
    score_delta: float
    score_a: ChiralityScore
    score_b: ChiralityScore


class ChiralMMPAnalyzer:
    """Analyze beneficial single-center chirality flips.

    Parameters
    ----------
    scorer : ChiralityScorer
        Scoring function for molecules.
    enumerator : StereoisomerEnumerator or None
        Stereoisomer enumerator (created with defaults if None).
    """

    def __init__(
        self,
        scorer: ChiralityScorer,
        enumerator: StereoisomerEnumerator | None = None,
    ):
        self.scorer = scorer
        self.enumerator = enumerator or StereoisomerEnumerator(
            compute_fingerprints=False,
        )

    def find_beneficial_flips(self, smiles: str) -> list[ChiralMMP]:
        """Find single-center flips that improve the combined score.

        Parameters
        ----------
        smiles : str
            Input molecule SMILES.

        Returns
        -------
        list[ChiralMMP]
            Beneficial flips sorted by score improvement (descending).
        """
        # Score the original
        score_a = self.scorer.score(smiles)

        # Get all single-center flips
        flips = self.enumerator.enumerate_single_flips(smiles)

        beneficial = []
        for flip in flips:
            score_b = self.scorer.score(flip.canonical_smiles)
            delta = score_b.combined_score - score_a.combined_score

            if delta > 0:
                # Identify which centers were flipped
                all_variants = self.enumerator.enumerate(smiles)
                original = next(
                    (v for v in all_variants if v.is_original),
                    all_variants[0] if all_variants else None,
                )
                flipped = []
                if original:
                    for idx in set(original.configuration.keys()) | set(flip.configuration.keys()):
                        if original.configuration.get(idx) != flip.configuration.get(idx):
                            flipped.append(idx)

                orig_config = original.configuration if original else {}
                flip_config = flip.configuration
                desc_parts = []
                for idx in flipped:
                    desc_parts.append(
                        f"C{idx}: {orig_config.get(idx, '?')}->{flip_config.get(idx, '?')}"
                    )
                transformation = ", ".join(desc_parts) if desc_parts else "stereo flip"

                beneficial.append(ChiralMMP(
                    smiles_a=smiles,
                    smiles_b=flip.canonical_smiles,
                    transformation=transformation,
                    flipped_centers=flipped,
                    score_delta=delta,
                    score_a=score_a,
                    score_b=score_b,
                ))

        beneficial.sort(key=lambda m: m.score_delta, reverse=True)
        return beneficial

    def analyze_pair_library(
        self,
        pair_activities: list,
        limit: int = 100,
    ) -> list[ChiralMMP]:
        """Analyze known enantiomer pairs from ChEMBL data.

        Parameters
        ----------
        pair_activities : list[PairActivity]
            Enriched pairs from the ChEMBL pipeline.
        limit : int
            Maximum pairs to analyze.

        Returns
        -------
        list[ChiralMMP]
            Scored pairs sorted by score delta.
        """
        results = []
        for pa in pair_activities[:limit]:
            smiles_a = pa.pair.mol_a.canonical_smiles
            smiles_b = pa.pair.mol_b.canonical_smiles

            try:
                score_a = self.scorer.score(smiles_a, reference_smiles=smiles_b)
                score_b = self.scorer.score(smiles_b, reference_smiles=smiles_a)
            except Exception:
                continue

            delta = score_b.combined_score - score_a.combined_score

            results.append(ChiralMMP(
                smiles_a=smiles_a,
                smiles_b=smiles_b,
                transformation="known pair",
                flipped_centers=[],
                score_delta=delta,
                score_a=score_a,
                score_b=score_b,
            ))

        results.sort(key=lambda m: abs(m.score_delta), reverse=True)
        return results
