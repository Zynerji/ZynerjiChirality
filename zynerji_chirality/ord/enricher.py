"""Chirality enrichment for ORD enantioselective reactions.

Computes chirality fingerprints for catalysts and substrates,
builds combined fingerprints for ee prediction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zynerji_chirality.ord.extractor import EnantioselectiveReaction

logger = logging.getLogger(__name__)


@dataclass
class ReactionEnrichment:
    """Fingerprint-enriched enantioselective reaction."""

    reaction: EnantioselectiveReaction
    catalyst_fp: np.ndarray | None
    substrate_fp: np.ndarray | None
    combined_fp: np.ndarray | None
    predicted_ee: float | None = None

    def to_dict(self) -> dict:
        return {
            "reaction": self.reaction.to_dict(),
            "catalyst_fp": self.catalyst_fp.tolist() if self.catalyst_fp is not None else None,
            "substrate_fp": self.substrate_fp.tolist() if self.substrate_fp is not None else None,
            "combined_fp": self.combined_fp.tolist() if self.combined_fp is not None else None,
            "predicted_ee": self.predicted_ee,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ReactionEnrichment:
        return cls(
            reaction=EnantioselectiveReaction.from_dict(d["reaction"]),
            catalyst_fp=np.array(d["catalyst_fp"]) if d.get("catalyst_fp") is not None else None,
            substrate_fp=np.array(d["substrate_fp"]) if d.get("substrate_fp") is not None else None,
            combined_fp=np.array(d["combined_fp"]) if d.get("combined_fp") is not None else None,
            predicted_ee=d.get("predicted_ee"),
        )


class ORDEnricher:
    """Enrich enantioselective reactions with chirality fingerprints.

    Parameters
    ----------
    nbits : int
        Fingerprint dimensionality.
    """

    def __init__(self, nbits: int = 128):
        self.nbits = nbits

    def enrich_reaction(
        self, reaction: EnantioselectiveReaction
    ) -> ReactionEnrichment:
        """Compute fingerprints for a single reaction's catalyst and substrate."""
        from zynerji_chirality.chirality.fingerprint import chirality_fingerprint

        cat_fp = None
        sub_fp = None
        combined_fp = None

        if reaction.catalyst_smiles:
            try:
                cat_fp = chirality_fingerprint(
                    reaction.catalyst_smiles, nbits=self.nbits
                )
            except Exception:
                pass

        if reaction.substrate_smiles:
            try:
                sub_fp = chirality_fingerprint(
                    reaction.substrate_smiles, nbits=self.nbits
                )
            except Exception:
                pass

        # Build combined fingerprint (same pattern as EEPredictor)
        if cat_fp is not None and sub_fp is not None:
            combined_fp = np.concatenate([
                cat_fp,
                sub_fp,
                cat_fp * sub_fp,          # interaction
                np.abs(cat_fp - sub_fp),  # difference
            ])

        return ReactionEnrichment(
            reaction=reaction,
            catalyst_fp=cat_fp,
            substrate_fp=sub_fp,
            combined_fp=combined_fp,
        )

    def enrich_batch(
        self,
        reactions: list[EnantioselectiveReaction],
        chunk_size: int = 200,
        checkpoint_path: str | None = None,
        progress_path: str | None = None,
    ) -> list[ReactionEnrichment]:
        """Enrich a batch of reactions with incremental checkpointing.

        Parameters
        ----------
        reactions : list
            Reactions to enrich.
        chunk_size : int
            Save checkpoint after this many reactions.
        checkpoint_path : str, optional
            Path for enrichment checkpoint JSON.
        progress_path : str, optional
            Path for lightweight progress JSON.
        """
        enrichments: list[ReactionEnrichment] = []
        start_idx = 0

        if checkpoint_path and Path(checkpoint_path).exists():
            enrichments = self.load_enriched(checkpoint_path)
            if len(enrichments) >= len(reactions):
                logger.info("Enrichment complete: %d reactions", len(enrichments))
                return enrichments
            start_idx = len(enrichments)
            logger.info("Resuming enrichment from %d/%d", start_idx, len(reactions))

        total = len(reactions)

        for chunk_start in range(start_idx, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            logger.info(
                "Enriching reactions %d-%d of %d (%.1f%%)",
                chunk_start, chunk_end, total,
                chunk_end / total * 100,
            )

            for rxn in reactions[chunk_start:chunk_end]:
                enrichment = self.enrich_reaction(rxn)
                enrichments.append(enrichment)

            if checkpoint_path:
                self.save_enriched(enrichments, checkpoint_path)

            if progress_path:
                n_with_fp = sum(
                    1 for e in enrichments if e.combined_fp is not None
                )
                progress = {
                    "stage": "enrich",
                    "reactions_enriched": len(enrichments),
                    "reactions_total": total,
                    "reactions_with_fingerprint": n_with_fp,
                    "percent_complete": round(len(enrichments) / total * 100, 1),
                }
                with open(progress_path, "w") as f:
                    json.dump(progress, f)

        return enrichments

    @staticmethod
    def save_enriched(enrichments: list[ReactionEnrichment], path: str) -> None:
        """Save enrichments to JSON."""
        data = [e.to_dict() for e in enrichments]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load_enriched(path: str) -> list[ReactionEnrichment]:
        """Load enrichments from JSON."""
        with open(path) as f:
            data = json.load(f)
        return [ReactionEnrichment.from_dict(d) for d in data]
