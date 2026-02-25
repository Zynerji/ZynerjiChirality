"""Generative chemistry filter for chirality-aware molecule ranking.

Filters and scores lists of molecules (e.g., from generative models)
based on chirality optimization criteria.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem

from zynerji_chirality.generative.enumerator import StereoisomerEnumerator
from zynerji_chirality.generative.scorer import ChiralityScorer, ChiralityScore

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of filtering a molecule list."""
    input_count: int
    output_count: int
    filtered_count: int
    ranked_molecules: list[ChiralityScore] = field(default_factory=list)


class GenerativeFilter:
    """Filter and score molecules from generative chemistry pipelines.

    Parameters
    ----------
    scorer : ChiralityScorer
        Scoring function for molecules.
    enumerator : StereoisomerEnumerator or None
        For enumerating missing stereoisomers.
    min_score : float
        Minimum combined score to pass filter.
    enumerate_stereo : bool
        If True, enumerate missing stereo for molecules without defined centers.
    """

    def __init__(
        self,
        scorer: ChiralityScorer,
        enumerator: StereoisomerEnumerator | None = None,
        min_score: float = 0.0,
        enumerate_stereo: bool = False,
    ):
        self.scorer = scorer
        self.enumerator = enumerator or StereoisomerEnumerator(
            compute_fingerprints=False,
        )
        self.min_score = min_score
        self.enumerate_stereo = enumerate_stereo

    def filter(
        self,
        smiles_list: list[str],
        reference_smiles: str | None = None,
    ) -> FilterResult:
        """Filter, canonicalize, deduplicate, optionally enumerate, score, and sort.

        Parameters
        ----------
        smiles_list : list[str]
            Input SMILES from generative model or library.
        reference_smiles : str or None
            Reference SMILES for target scoring.

        Returns
        -------
        FilterResult
            Filtered and ranked molecules.
        """
        input_count = len(smiles_list)

        # Canonicalize and deduplicate
        canonical_set: set[str] = set()
        canonical_list: list[str] = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            if canonical not in canonical_set:
                canonical_set.add(canonical)
                canonical_list.append(canonical)

        # Optionally enumerate missing stereoisomers
        if self.enumerate_stereo:
            expanded: list[str] = []
            for smiles in canonical_list:
                try:
                    variants = self.enumerator.enumerate(smiles)
                    for v in variants:
                        if v.canonical_smiles not in canonical_set:
                            canonical_set.add(v.canonical_smiles)
                            expanded.append(v.canonical_smiles)
                except Exception:
                    pass
            canonical_list.extend(expanded)

        # Score all
        scores = []
        for smiles in canonical_list:
            try:
                score = self.scorer.score(smiles, reference_smiles=reference_smiles)
                scores.append(score)
            except Exception as e:
                logger.debug("Scoring failed for %s: %s", smiles, e)

        # Filter by min_score
        filtered_scores = [s for s in scores if s.combined_score >= self.min_score]
        filtered_scores.sort(key=lambda s: s.combined_score, reverse=True)

        return FilterResult(
            input_count=input_count,
            output_count=len(filtered_scores),
            filtered_count=input_count - len(filtered_scores),
            ranked_molecules=filtered_scores,
        )

    def filter_from_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> FilterResult:
        """Filter molecules from a file (one SMILES per line).

        Parameters
        ----------
        input_path : str or Path
            Input file with one SMILES per line.
        output_path : str or Path or None
            Output file for filtered SMILES (optional).

        Returns
        -------
        FilterResult
            Filtered and ranked molecules.
        """
        with open(input_path) as f:
            smiles_list = [line.strip() for line in f if line.strip()]

        result = self.filter(smiles_list)

        if output_path:
            with open(output_path, "w") as f:
                for score in result.ranked_molecules:
                    f.write(f"{score.smiles}\t{score.combined_score:.4f}\n")
            logger.info(
                "Wrote %d molecules to %s", len(result.ranked_molecules), output_path,
            )

        return result
