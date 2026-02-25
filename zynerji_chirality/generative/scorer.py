"""Chirality scoring framework for stereoisomer optimization.

Combines target sensitivity, ADMET risk, and fingerprint novelty
into a single optimization score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from zynerji_chirality.chirality.fingerprint import chirality_fingerprint
from zynerji_chirality.generative.enumerator import StereoisomerVariant

logger = logging.getLogger(__name__)


@dataclass
class ChiralityScore:
    """Combined chirality optimization score."""
    smiles: str
    target_score: float       # 0-1, higher = more target sensitivity
    admet_score: float        # 0-1, higher = better ADMET profile
    fingerprint_novelty: float  # 0-1, higher = more novel
    combined_score: float     # Weighted combination
    target_profile: object | None = None
    admet_profile: object | None = None


class ChiralityScorer:
    """Score molecules based on chirality-aware criteria.

    Graceful degradation: missing models default to neutral scores
    (target=0.0, admet=1.0, novelty=0.5).

    Parameters
    ----------
    target_profiler : TargetProfiler or None
        Target sensitivity profiler.
    admet_profiler : ADMETProfiler or None
        ADMET profiler.
    store : FingerprintStore or None
        Fingerprint database for novelty scoring.
    reference_smiles : str or None
        Reference SMILES for the "other enantiomer" in target profiling.
    weights : dict
        Scoring weights: target, admet, novelty.
    nbits : int
        Fingerprint bits for novelty search.
    """

    def __init__(
        self,
        target_profiler=None,
        admet_profiler=None,
        store=None,
        reference_smiles: str | None = None,
        weights: dict | None = None,
        nbits: int = 128,
    ):
        self.target_profiler = target_profiler
        self.admet_profiler = admet_profiler
        self.store = store
        self.reference_smiles = reference_smiles
        self.weights = weights or {"target": 0.4, "admet": 0.3, "novelty": 0.3}
        self.nbits = nbits

    def score(
        self,
        smiles: str,
        reference_smiles: str | None = None,
    ) -> ChiralityScore:
        """Score a molecule on target sensitivity, ADMET, and novelty.

        Parameters
        ----------
        smiles : str
            Molecule SMILES to score.
        reference_smiles : str or None
            Partner enantiomer for target profiling. Falls back to self.reference_smiles.

        Returns
        -------
        ChiralityScore
            Combined score with component breakdown.
        """
        ref = reference_smiles or self.reference_smiles

        # Target sensitivity score (0-1)
        target_score = 0.0
        target_profile = None
        if self.target_profiler is not None and ref is not None:
            try:
                target_profile = self.target_profiler.profile(smiles, ref)
                # Normalize: n_sensitive / total targets, capped at 1.0
                n_total = max(len(target_profile.entries), 1)
                target_score = min(target_profile.n_sensitive / n_total, 1.0)
            except Exception as e:
                logger.debug("Target scoring failed: %s", e)

        # ADMET score (0-1, 1.0 = best)
        admet_score = 1.0  # Default: no risk
        admet_profile = None
        if self.admet_profiler is not None:
            try:
                admet_profile = self.admet_profiler.profile(smiles)
                n_total = max(len(admet_profile.entries), 1)
                n_ok = sum(
                    1 for e in admet_profile.entries
                    if e.risk_level == "low_risk"
                )
                admet_score = n_ok / n_total
            except Exception as e:
                logger.debug("ADMET scoring failed: %s", e)

        # Fingerprint novelty (0-1, 1.0 = most novel)
        novelty = 0.5  # Default: neutral
        if self.store is not None:
            try:
                fp = chirality_fingerprint(smiles, nbits=self.nbits)
                results = self.store.search_similar(fp, k=1, nbits=self.nbits)
                if results:
                    # Novelty = 1 - similarity to nearest neighbor
                    novelty = 1.0 - results[0].similarity
                else:
                    novelty = 1.0  # No neighbors = fully novel
            except Exception as e:
                logger.debug("Novelty scoring failed: %s", e)

        # Combined score
        w = self.weights
        combined = (
            w.get("target", 0.4) * target_score
            + w.get("admet", 0.3) * admet_score
            + w.get("novelty", 0.3) * novelty
        )

        return ChiralityScore(
            smiles=smiles,
            target_score=target_score,
            admet_score=admet_score,
            fingerprint_novelty=novelty,
            combined_score=combined,
            target_profile=target_profile,
            admet_profile=admet_profile,
        )

    def score_batch(self, smiles_list: list[str]) -> list[ChiralityScore]:
        """Score multiple molecules, sorted by combined score descending."""
        scores = [self.score(s) for s in smiles_list]
        scores.sort(key=lambda s: s.combined_score, reverse=True)
        return scores

    def rank_stereoisomers(
        self,
        variants: list[StereoisomerVariant],
    ) -> list[tuple[StereoisomerVariant, ChiralityScore]]:
        """Rank stereoisomer variants by combined score.

        Parameters
        ----------
        variants : list[StereoisomerVariant]
            Enumerated stereoisomers.

        Returns
        -------
        list[tuple[StereoisomerVariant, ChiralityScore]]
            Sorted (variant, score) pairs, best first.
        """
        ranked = []
        for variant in variants:
            score = self.score(variant.canonical_smiles)
            ranked.append((variant, score))
        ranked.sort(key=lambda x: x[1].combined_score, reverse=True)
        return ranked
